# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Anticipatory Routing: decouple MoE routing decisions from the backbone.

At step ``t`` the backbone runs with the current parameters, but the top-k
expert indices are the ones computed earlier with ``theta_{t-dt}``. The data for
step ``t`` is fetched ``dt`` steps early and pushed through a forward-only pass
that caches its routing indices, so no second copy of the model is needed.

The mode is armed dynamically: on a loss spike the run rolls back to an earlier
checkpoint, warms the index cache, trains for a while with stale routing, then
reverts to standard training.
"""

from .config import AnticipatoryRoutingConfig
from .detector import LossSpikeDetector
from .engine import AnticipatoryTrainingEngine
from .trainer import AnticipatoryTrainer


__all__ = [
    "AnticipatoryRoutingConfig",
    "AnticipatoryTrainer",
    "AnticipatoryTrainingEngine",
    "LossSpikeDetector",
]
