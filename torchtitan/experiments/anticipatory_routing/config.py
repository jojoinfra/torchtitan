# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field

from .cache import IndexStoreDtype
from .detector import LossSpikeDetector


@dataclass(kw_only=True, slots=True)
class AnticipatoryRoutingConfig:
    """When and how long to train with stale routing indices."""

    delay_steps: int = 16
    """Optimizer steps between computing a batch's routing indices and training
    on that batch. Also the number of warmup steps: entering the mode
    pre-computes indices for this many steps' worth of data before the first
    anticipatory optimizer step. Zero means capture and consume a batch in the
    same step, which reproduces standard training exactly and exists for
    numerical validation."""

    active_steps: int = 500
    """Optimizer steps to spend in anticipatory mode before reverting."""

    always_on: bool = False
    """Skip spike detection and enter anticipatory mode at the first step. For
    ablations; the dynamic path is what keeps the average overhead low."""

    max_rollbacks: int = 3
    """Upper bound on rollbacks in one run, so a persistently unstable run
    cannot loop forever."""

    index_store_dtype: IndexStoreDtype = "auto"
    """Integer width the cached indices are stored in. ``auto`` picks the
    narrowest dtype that can hold an expert id."""

    offload_indices_to_cpu: bool = False
    """Keep the cached indices in host memory instead of on the accelerator.
    Worth it when ``delay_steps * gradient_accumulation_steps`` makes the cache
    large relative to spare device memory."""

    detector: LossSpikeDetector.Config = field(
        default_factory=LossSpikeDetector.Config
    )

    def __post_init__(self) -> None:
        if self.delay_steps < 0:
            raise ValueError("delay_steps cannot be negative.")
        if self.active_steps < 1:
            raise ValueError("active_steps must be at least 1.")
        if self.max_rollbacks < 0:
            raise ValueError("max_rollbacks cannot be negative.")
