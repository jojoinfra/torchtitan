# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import enum
import logging
import time
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from torchtitan.components.data.loader import DataloaderExhaustedError
from torchtitan.components.data.types import TrainingMicrobatch
from torchtitan.observability import structured_logger as sl
from torchtitan.trainer import Trainer

from .cache import RoutingMode, RoutingSlot, slot_nbytes
from .config import AnticipatoryRoutingConfig
from .engine import AnticipatoryTrainingEngine
from .router import build_routing_cache


logger = logging.getLogger(__name__)

_UNLIMITED = -1


class _Phase(enum.Enum):
    """Which schedule the next optimizer step follows.

    Warmup is deliberately absent: it runs inline inside the spike handler
    rather than as outer-loop iterations, so one iteration of ``Trainer.train``
    stays exactly one optimizer step in every phase and checkpointing,
    validation and profiling keep their upstream cadence.
    """

    NORMAL = "normal"
    ACTIVE = "active"
    DRAIN = "drain"


@dataclass
class _QueuedStep:
    """One optimizer step's data, with the routing indices captured for it."""

    microbatch_groups: list[list[TrainingMicrobatch]]
    slots: list[RoutingSlot]


class _SuppliedMicrobatches:
    """Hand a pre-fetched batch to the base ``Trainer.train_step``.

    ``Trainer.train_step`` takes its data iterator as an argument and pulls
    exactly ``gradient_accumulation_steps * num_pp_microbatches`` microbatches
    from it, so a queued batch can be supplied without touching the base
    implementation. This iterator holds exactly that many and fails loudly in
    either direction, so a future change to the base fetch count cannot quietly
    train on a short batch.
    """

    def __init__(self, microbatches: list[TrainingMicrobatch]) -> None:
        self._microbatches = microbatches
        self._index = 0

    def __iter__(self) -> _SuppliedMicrobatches:
        return self

    def __next__(self) -> TrainingMicrobatch:
        if self._index >= len(self._microbatches):
            raise RuntimeError(
                "Trainer.train_step requested more microbatches than the "
                f"{len(self._microbatches)} anticipatory routing pre-fetched for "
                "this optimizer step."
            )
        microbatch = self._microbatches[self._index]
        self._index += 1
        return microbatch

    def verify_consumed(self) -> None:
        if self._index != len(self._microbatches):
            raise RuntimeError(
                "Trainer.train_step consumed "
                f"{self._index} of {len(self._microbatches)} supplied "
                "microbatches. Anticipatory routing pre-fetches exactly one "
                "optimizer step's worth; the rest would be trained on with no "
                "cached routing indices."
            )


class AnticipatoryTrainer(Trainer):
    """Trainer that trains on stale MoE routing indices after a loss spike.

    On a spike the run rolls back to an earlier checkpoint, pre-computes routing
    indices for the next ``delay_steps`` batches, then trains each batch with the
    indices computed ``delay_steps`` optimizer steps earlier. After
    ``active_steps`` it drains the queue and reverts to standard training.

    Every override here is a wrapper that calls ``super()``; no base method is
    reimplemented.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        anticipatory: AnticipatoryRoutingConfig = field(
            default_factory=AnticipatoryRoutingConfig
        )

        def __post_init__(self) -> None:
            Trainer.Config.__post_init__(self)

            if not self.training.disable_cuda_graphs:
                raise ValueError(
                    "Anticipatory routing requires "
                    "training.disable_cuda_graphs=True. A CUDA graph is "
                    "captured once and replayed, so whichever routing mode was "
                    "in effect at capture time would be replayed for every "
                    "later step."
                )
            if self.parallelism.pipeline_parallel_degree > 1:
                raise ValueError(
                    "Anticipatory routing does not support pipeline "
                    "parallelism yet: a pipeline schedule interleaves the "
                    "forwards of the microbatches in one step, so the "
                    "per-microbatch index slot cannot be selected per "
                    "forward_backward_microbatch call."
                )
            if self.checkpointer is None:
                raise ValueError(
                    "Anticipatory routing rolls back to a checkpoint on a loss "
                    "spike, so a checkpointer must be configured."
                )
            if self.checkpointer.load_only:
                raise ValueError(
                    "Anticipatory routing needs checkpoints written during the "
                    "run, but checkpointer.load_only disables saving."
                )
            if self.debug.moe_force_load_balance:
                raise ValueError(
                    "debug.moe_force_load_balance bypasses the router's "
                    "_select_experts, which is where anticipatory routing "
                    "captures and replays indices."
                )
            if self.sdc_replayer is not None:
                raise ValueError(
                    "Anticipatory routing does not support SDC replay yet."
                )

    engine_cls = AnticipatoryTrainingEngine
    engine: AnticipatoryTrainingEngine
    config: Config

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        anticipatory = config.anticipatory

        self.routing_cache = build_routing_cache(
            self.engine.model_parts,
            index_store_dtype=anticipatory.index_store_dtype,
            offload_to_cpu=anticipatory.offload_indices_to_cpu,
        )
        self.engine.set_routing_cache(self.routing_cache)
        self.detector = anticipatory.detector.build()
        self.engine.loss_reduce_every = anticipatory.detector.check_every

        self._phase = _Phase.NORMAL
        self._queue: deque[_QueuedStep] = deque()
        self._active_steps_left = 0
        self._num_rollbacks = 0
        self._data_exhausted = False
        self._needs_initial_warmup = anticipatory.always_on
        self._logged_cache_size = False
        self._microbatches_per_step = (
            self.gradient_accumulation_steps * self.num_pp_microbatches
        )

        checkpointer = config.checkpointer
        assert checkpointer is not None  # enforced by Config.__post_init__
        if checkpointer.keep_latest_k > 0:
            logger.warning(
                "checkpointer.keep_latest_k=%d and interval=%d retain about "
                "%d steps of rollback history. A spike whose onset predates "
                "that has no checkpoint to roll back to and will be reported "
                "without acting on it.",
                checkpointer.keep_latest_k,
                checkpointer.interval,
                checkpointer.keep_latest_k * checkpointer.interval,
            )
        logger.info(
            "Anticipatory routing: delay %d step(s), %s active step(s) once "
            "armed, %s.",
            anticipatory.delay_steps,
            "unlimited" if anticipatory.always_on else anticipatory.active_steps,
            "armed from step 1"
            if anticipatory.always_on
            else f"armed on a loss spike (max {anticipatory.max_rollbacks} rollbacks)",
        )

    # -- data ------------------------------------------------------------

    def microbatch_generator(
        self, data_iterable: Iterable[TrainingMicrobatch]
    ) -> Iterator[TrainingMicrobatch]:
        """Yield microbatches without counting their tokens at fetch time.

        Anticipatory routing fetches a microbatch ``delay_steps`` before it is
        trained on, so the base implementation's fetch-time token counter would
        credit the tokens to the wrong step: throughput would spike during
        warmup and read as zero while the queue drains. ``train_step`` counts
        them instead, when the microbatch is actually trained.
        """
        data_iterator = iter(data_iterable)
        while True:
            data_load_start = time.perf_counter()
            try:
                microbatch = next(data_iterator)
            except StopIteration as ex:
                raise DataloaderExhaustedError() from ex
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )
            yield microbatch

    def _fetch_step_batches(
        self, data_iterator: Iterator[TrainingMicrobatch]
    ) -> list[list[TrainingMicrobatch]]:
        """Fetch exactly one optimizer step's worth of microbatches.

        The single call site that advances the data stream. There is only one
        stream to advance: ``GrainDataLoader.__iter__`` returns the same stored
        iterator every time, so a second ``iter()`` would be an alias, and a
        second dataloader would drift out of step with this one -- the indices
        cached for a batch have to be consumed by that same batch.
        """
        microbatch_groups: list[list[TrainingMicrobatch]] = []
        for _ in range(self.gradient_accumulation_steps):
            microbatch_group = []
            for _ in range(self.num_pp_microbatches):
                with sl.log_trace_span("fetching_batch"):
                    microbatch_group.append(next(data_iterator))
            microbatch_groups.append(microbatch_group)
        return microbatch_groups

    # -- prefetch ---------------------------------------------------------

    def _prefetch_and_capture(
        self, data_iterator: Iterator[TrainingMicrobatch]
    ) -> bool:
        """Fetch one step ahead and cache its routing indices.

        Returns ``False`` when the data ran out, which ends the prefetch and
        leaves the queue to drain.
        """
        try:
            microbatch_groups = self._fetch_step_batches(data_iterator)
        except DataloaderExhaustedError:
            logger.warning(
                "Anticipatory prefetch ran out of data; draining %d queued "
                "step(s) before stopping.",
                len(self._queue),
            )
            self._data_exhausted = True
            return False

        cache = self.routing_cache
        slots: list[RoutingSlot] = []
        cache.mode = RoutingMode.CAPTURE
        try:
            with sl.log_trace_span("anticipatory_prefetch"):
                for microbatch_group in microbatch_groups:
                    for microbatch in microbatch_group:
                        slot: RoutingSlot = {}
                        cache.slot = slot
                        self.engine.forward_only_microbatch(microbatch)
                        slots.append(slot)
        finally:
            cache.mode = RoutingMode.OFF
            cache.slot = None

        # forward_backward_microbatch indexes this list by accumulation index,
        # which holds because a microbatch group is one microbatch without
        # pipeline parallelism.
        assert len(slots) == self.gradient_accumulation_steps
        self._queue.append(
            _QueuedStep(microbatch_groups=microbatch_groups, slots=slots)
        )
        self._maybe_log_cache_size(slots)
        return True

    def _maybe_log_cache_size(self, slots: list[RoutingSlot]) -> None:
        """Report the cache footprint once, after the first capture."""
        if self._logged_cache_size or not slots:
            return
        self._logged_cache_size = True
        per_step = sum(slot_nbytes(slot) for slot in slots)
        delay_steps = self.config.anticipatory.delay_steps
        logger.info(
            "Routing-index cache: %.1f MiB per step, %.1f MiB resident at "
            "delay %d.",
            per_step / 1024**2,
            per_step * max(delay_steps, 1) / 1024**2,
            delay_steps,
        )

    # -- the step ---------------------------------------------------------

    def train_step(self, data_iterator: Iterator[TrainingMicrobatch]) -> None:
        if self._needs_initial_warmup:
            self._needs_initial_warmup = False
            self._enter_anticipatory(data_iterator)

        fetched = False
        if self._phase is _Phase.ACTIVE and not self._data_exhausted:
            # Capture before the update, so these indices are the ones computed
            # at the parameters this step starts from. They are consumed
            # delay_steps optimizer steps from now.
            fetched = self._prefetch_and_capture(data_iterator)

        queued = self._queue.popleft() if self._queue else None
        if queued is None:
            microbatch_groups = self._fetch_step_batches(data_iterator)
            slots = None
            fetched = True
        else:
            microbatch_groups = queued.microbatch_groups
            # Queued indices are only replayed while the mode is active; the
            # drain phase trains the leftovers with freshly computed routing.
            slots = queued.slots if self._phase is _Phase.ACTIVE else None

        if not fetched:
            # This step trained on data loaded earlier. The metrics processor
            # averages over the recorded load times and would divide by zero if
            # a whole logging window landed inside the drain.
            self.metrics_processor.data_loading_times.extend(
                [0.0] * self._microbatches_per_step
            )

        self.metrics_processor.ntokens_since_last_log += (
            self._microbatches_per_step
            * self.config.training.num_tokens_per_microbatch_per_dp_rank
        )

        supplied = _SuppliedMicrobatches(
            [mb for group in microbatch_groups for mb in group]
        )
        cache = self.routing_cache
        cache.mode = RoutingMode.REPLAY if slots is not None else RoutingMode.OFF
        self.engine.set_step_routing_slots(slots)
        try:
            super().train_step(supplied)
        finally:
            cache.mode = RoutingMode.OFF
            cache.slot = None
            self.engine.set_step_routing_slots(None)
        supplied.verify_consumed()

        self._advance_phase()
        self._maybe_handle_spike(data_iterator)

    # -- phase transitions -------------------------------------------------

    def _advance_phase(self) -> None:
        if self._phase is _Phase.ACTIVE:
            if self._active_steps_left != _UNLIMITED:
                self._active_steps_left -= 1
            budget_spent = self._active_steps_left == 0
            if budget_spent or self._data_exhausted:
                logger.info(
                    "Anticipatory routing finished; draining %d queued step(s).",
                    len(self._queue),
                )
                self._phase = _Phase.DRAIN
        elif self._phase is _Phase.DRAIN and not self._queue:
            logger.info(
                "Anticipatory routing reverted to standard training; "
                "checkpointing resumes."
            )
            self._phase = _Phase.NORMAL
            # The queue is empty, so the data stream is back in step with the
            # parameters and a checkpoint pairs them correctly again.
            self.engine.suppress_checkpoint_saves = False
            # The run continues on a different trajectory, so the loss history
            # collected before the rollback no longer describes it.
            self.detector.reset()

    def _maybe_handle_spike(
        self, data_iterator: Iterator[TrainingMicrobatch]
    ) -> None:
        """Feed the detector and, on a spike, roll back and arm the mode.

        Only observed during standard training: while the mode is active the run
        is deliberately on a different trajectory, and retriggering there would
        stack rollbacks on top of each other.
        """
        if self._phase is not _Phase.NORMAL:
            return
        loss = self.engine.last_global_loss
        if loss is None:
            return
        step = self.engine.num_completed_steps
        if not self.detector.should_observe(step):
            return

        onset = self.detector.observe(step, loss)
        if onset is None:
            return

        anticipatory = self.config.anticipatory
        if self._num_rollbacks >= anticipatory.max_rollbacks:
            logger.warning(
                "Loss spike at step %d (onset %d) but the rollback budget of "
                "%d is spent; continuing without rollback.",
                step,
                onset,
                anticipatory.max_rollbacks,
            )
            return

        target = self.engine.find_rollback_target(onset)
        if target is None:
            logger.warning(
                "Loss spike at step %d (onset %d) but no resumable checkpoint "
                "before step %d survives retention; continuing without "
                "rollback.",
                step,
                onset,
                onset,
            )
            return

        logger.warning(
            "Loss spike at step %d (onset %d, loss %.4f): rolling back to step "
            "%d and arming anticipatory routing.",
            step,
            onset,
            loss,
            target,
        )
        self.engine.rollback_to(target)
        self._num_rollbacks += 1
        self.detector.reset()
        # The next log window starts here; spanning the rollback would report a
        # throughput and token delta for steps that were undone.
        self.metrics_processor.step_last_log = None
        self.metrics_processor.ntokens_since_last_log = 0
        self.metrics_processor.data_loading_times.clear()
        self.metrics_processor.time_last_log = time.perf_counter()

        self._enter_anticipatory(data_iterator)

    def _enter_anticipatory(
        self, data_iterator: Iterator[TrainingMicrobatch]
    ) -> None:
        """Warm the index cache, then switch to anticipatory train steps.

        The warmup runs inline rather than as outer-loop iterations so that one
        iteration of ``Trainer.train`` stays exactly one optimizer step. No
        optimizer step happens here, so every index the warmup caches is
        computed at the same parameters.

        With ``delay_steps=0`` the warmup is empty and each active step captures
        and consumes the same batch, which reproduces standard training exactly.
        """
        anticipatory = self.config.anticipatory
        self._queue.clear()
        with sl.log_trace_span("anticipatory_warmup"):
            for _ in range(anticipatory.delay_steps):
                if not self._prefetch_and_capture(data_iterator):
                    break
        self._phase = _Phase.ACTIVE
        self._active_steps_left = (
            _UNLIMITED if anticipatory.always_on else anticipatory.active_steps
        )
        self.engine.suppress_checkpoint_saves = True
        logger.info(
            "Anticipatory routing armed: %d step(s) of indices cached, "
            "replaying routing from %d step(s) back. Checkpointing is paused "
            "until the queue drains, because the data stream now runs ahead of "
            "the training step.",
            len(self._queue),
            anticipatory.delay_steps,
        )
