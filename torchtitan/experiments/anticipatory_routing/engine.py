# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from typing import Any, cast

import torch

from torchtitan.components.data.types import TrainingMicrobatch
from torchtitan.distributed import utils as dist_utils
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.common.moe import QuantileBalancer, TokenChoiceTopKRouter
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols import BaseModel
from torchtitan.tools import filesystem
from torchtitan.training_engine import TrainingEngine

from .cache import RoutingIndexCache, RoutingMode, RoutingSlot


logger = logging.getLogger(__name__)


class AnticipatoryTrainingEngine(TrainingEngine):
    """Training engine with routing-index capture, replay, and mid-run rollback.

    Everything here is either a thin wrapper that calls ``super()`` or a new
    method; no base behavior is reimplemented.
    """

    routing_cache: RoutingIndexCache | None
    last_global_loss: float | None

    def __init__(self, config, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.routing_cache = None
        self.last_global_loss = None
        self.suppress_checkpoint_saves = False
        self.loss_reduce_every = 1
        self._step_routing_slots: list[RoutingSlot] | None = None
        self._step_loss_sum: torch.Tensor | None = None

    # -- routing cache plumbing ------------------------------------------

    def set_routing_cache(self, cache: RoutingIndexCache) -> None:
        self.routing_cache = cache

    def set_step_routing_slots(self, slots: list[RoutingSlot] | None) -> None:
        """Supply this optimizer step's index slots, one per accumulation unit.

        ``None`` means the step runs with the cache off.
        """
        self._step_routing_slots = slots

    def forward_backward_microbatch(
        self,
        *,
        microbatch_group: list[TrainingMicrobatch],
        global_valid_tokens: torch.Tensor,
        accumulation_index: int = 0,
    ) -> torch.Tensor:
        """Select this microbatch's index slot, then run the base step.

        Also accumulates the step's loss unconditionally. The base trainer only
        keeps the loss on logging steps, but the spike detector needs it every
        step, and doing it here avoids touching the base ``train_step``.
        """
        cache = self.routing_cache
        if cache is not None and cache.mode is not RoutingMode.OFF:
            # Without pipeline parallelism a microbatch group holds exactly one
            # microbatch, so the accumulation index identifies it. A pipeline
            # schedule interleaves the forwards of a group's microbatches, which
            # is why pipeline parallelism is rejected in the trainer config.
            assert len(microbatch_group) == 1
            assert self._step_routing_slots is not None
            cache.slot = self._step_routing_slots[accumulation_index]

        if accumulation_index == 0:
            self._step_loss_sum = None

        detached_loss = super().forward_backward_microbatch(
            microbatch_group=microbatch_group,
            global_valid_tokens=global_valid_tokens,
            accumulation_index=accumulation_index,
        )

        if self._step_loss_sum is None:
            self._step_loss_sum = detached_loss.clone()
        else:
            self._step_loss_sum.add_(detached_loss)
        return detached_loss

    def optimizer_step(self) -> torch.Tensor:
        """Run the base optimizer step, then publish this step's global loss."""
        grad_norm = super().optimizer_step()
        self._reduce_step_loss()
        return grad_norm

    def _reduce_step_loss(self) -> None:
        """All-reduce the step's loss so every rank sees the same value.

        The loss function already normalizes by the step's global valid-token
        count, so summing the per-rank sums over the loss mesh gives the global
        average loss -- the same quantity the base trainer logs, just computed
        every step instead of every log step. ``dist_sum`` with a ``None`` mesh
        degenerates to a local ``item()``, which covers the single-rank case.

        Supporting pipeline parallelism would need a second hop over the pp
        mesh, since only the last stage holds a real loss.
        """
        if self._step_loss_sum is None:
            return
        if self.num_completed_steps % self.loss_reduce_every != 0:
            # The reduction ends in an item(), so it costs a host
            # synchronization; skip it on steps the detector will not look at.
            return
        loss_mesh = self.parallel_dims.get_optional_mesh("loss")
        self.last_global_loss = dist_utils.dist_sum(self._step_loss_sum, loss_mesh)

    # -- checkpointing -----------------------------------------------------

    def save_checkpoint(self, *, last_step: bool = False) -> bool:
        """Skip saves while the data stream runs ahead of the training step.

        During warmup and the active phase the prefetch has already consumed
        ``delay_steps`` steps' worth of data, so the dataloader state a
        checkpoint would capture sits that far ahead of the parameters beside
        it. Restarting from such a checkpoint would silently skip those
        batches, so no checkpoint is written until the queue has drained and
        the two are back in step.
        """
        if self.suppress_checkpoint_saves:
            return False
        return super().save_checkpoint(last_step=last_step)

    # -- forward-only pass -----------------------------------------------

    @torch.no_grad()
    def forward_only_microbatch(self, microbatch: TrainingMicrobatch) -> None:
        """Run one microbatch forward with no gradients and no training state.

        Eval mode is what keeps this pass from polluting training statistics: it
        is the guard on the router's ``tokens_per_expert_E`` accumulation, on the
        MoE auxiliary loss, and on the quantile balancer's histogram. The loss is
        not computed and ``ntokens_seen`` is not advanced, because this
        microbatch is trained on later and would otherwise be counted twice.

        The decoders use no dropout, so this pass consumes no RNG and cannot
        perturb the training forward that follows.
        """
        model = self.model_parts[0]
        was_training = model.training
        model.eval()
        try:
            input_dict = microbatch.to_input_dict(self.device, non_blocking=True)
            inputs, _labels, extra_kwargs = cast(BaseModel, model).preprocess_inputs(
                input_dict,
                parallel_dims=self.parallel_dims,
                parallelism=self.config.parallelism,
                max_num_documents=self.max_num_documents,
                max_context_length=self.config.training.max_context_length,
                **self.preprocess_inputs_kwargs,
            )
            with self.train_context():
                model(inputs, **extra_kwargs)
        finally:
            if was_training:
                model.train()

    # -- rollback ---------------------------------------------------------

    def find_rollback_target(self, onset_step: int) -> int | None:
        """Newest resumable checkpoint strictly before ``onset_step``.

        Strictly before, because a checkpoint saved at or after the step where
        the spike began already contains the damaged state. Returns ``None``
        when retention has purged everything old enough.

        Every rank scans the same directory and applies the same rule, so the
        answer is identical everywhere without a collective.
        """
        checkpointer = self.checkpointer
        folder = checkpointer.folder
        if not checkpointer._storage.isdir(folder):
            return None

        best: int | None = None
        for dirname in checkpointer._storage.listdir(folder):
            step = checkpointer._parse_step(dirname)
            # Step 0 is a seed checkpoint and holds model state only, so it
            # cannot restore the optimizer or the data stream.
            if step is None or step <= 0 or step >= onset_step:
                continue
            if not checkpointer._is_resumable_checkpoint(
                filesystem.join(folder, dirname)
            ):
                continue
            if best is None or step > best:
                best = step
        return best

    @sl.log_trace_span("anticipatory_rollback")
    def rollback_to(self, step: int) -> None:
        """Restore model, optimizer, scheduler and data stream to ``step``."""
        checkpointer = self.checkpointer
        # An in-flight asynchronous save still references the parameter storage
        # the load is about to overwrite.
        checkpointer.maybe_wait_for_staging()
        checkpointer.maybe_wait_for_saving()
        checkpointer.load(step=step)
        self._reset_transient_training_state()
        # The caller warms the index cache next, which runs the data stream
        # ahead; saving before it drains would persist a mismatched pair.
        self.suppress_checkpoint_saves = True
        logger.info(
            "Rolled back to step %d; %d completed steps, %d tokens seen.",
            step,
            self.num_completed_steps,
            self.ntokens_seen,
        )

    def _reset_transient_training_state(self) -> None:
        """Discard state the checkpoint does not carry.

        The router token counters, the quantile histogram and the auxiliary-loss
        accumulators are non-persistent buffers holding a partial accumulation
        for the abandoned trajectory. Left in place they would fold
        pre-rollback statistics into the first post-rollback optimizer step.
        """
        for part in self.model_parts:
            for module in part.modules():
                if isinstance(module, TokenChoiceTopKRouter):
                    module.tokens_per_expert_E.zero_()
                elif isinstance(module, QuantileBalancer):
                    module.required_bias_histogram_EB.zero_()
                elif isinstance(module, AuxLoss):
                    module.instance_acc.zero_()
        AuxLoss.group_acc.clear()
        self._step_loss_sum = None
        self.last_global_loss = None
        self._num_optimizer_steps_since_cuda_graph_init = 0
