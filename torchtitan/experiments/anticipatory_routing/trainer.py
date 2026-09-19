# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from torchtitan.components.data.loader import DataloaderExhaustedError
from torchtitan.components.data.types import TrainingMicrobatch
from torchtitan.trainer import Trainer

from .config import AnticipatoryRoutingConfig
from .engine import AnticipatoryTrainingEngine
from .router import build_routing_cache
from .schedule import AnticipatorySchedule


class AnticipatoryTrainer(Trainer):
    """Trainer that trains on stale MoE routing indices after a loss spike.

    The schedule decides what each step trains on and how it routes; see
    :class:`AnticipatorySchedule`. This class only wires it up and hands each
    step over, so every override here is a wrapper that calls ``super()`` and no
    base method is reimplemented.
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
        checkpointer = config.checkpointer
        assert checkpointer is not None  # enforced by Config.__post_init__

        self.routing_cache = build_routing_cache(
            self.engine.model_parts,
            index_store_dtype=config.anticipatory.index_store_dtype,
            offload_to_cpu=config.anticipatory.offload_indices_to_cpu,
            device=self.engine.device,
        )
        self.engine.set_routing_cache(self.routing_cache)
        # The engine skips the loss all-reduce on steps the detector will not
        # look at, so the two cadences have to match.
        self.engine.loss_reduce_every = config.anticipatory.detector.check_every

        microbatches_per_step = (
            self.gradient_accumulation_steps * self.num_pp_microbatches
        )
        self.schedule = AnticipatorySchedule(
            config.anticipatory,
            engine=self.engine,
            routing_cache=self.routing_cache,
            metrics=self.metrics_processor,
            microbatches_per_step=microbatches_per_step,
            tokens_per_step=(
                microbatches_per_step
                * config.training.num_tokens_per_microbatch_per_dp_rank
            ),
            checkpoint_interval=checkpointer.interval,
            keep_latest_k=checkpointer.keep_latest_k,
        )

    def microbatch_generator(
        self, data_iterable: Iterable[TrainingMicrobatch]
    ) -> Iterator[TrainingMicrobatch]:
        """Yield microbatches without charging them to the current step.

        The base implementation credits tokens and load time to whichever step
        is running when a microbatch is fetched. Under anticipatory routing that
        is ``delay_steps`` before the step that trains on it, so the schedule
        holds the cost until the batch is actually trained.
        """
        data_iterator = iter(data_iterable)
        while True:
            data_load_start = time.perf_counter()
            try:
                microbatch = next(data_iterator)
            except StopIteration as ex:
                raise DataloaderExhaustedError() from ex
            self.schedule.record_load_time(time.perf_counter() - data_load_start)
            yield microbatch

    def train_step(self, data_iterator: Iterator[TrainingMicrobatch]) -> None:
        with self.schedule.step(data_iterator) as microbatches:
            super().train_step(microbatches)
