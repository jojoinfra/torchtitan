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

How an index actually travels
-----------------------------
Both halves end inside the model's forward, reading shared state rather than an
argument, so no call graph shows the connection. Threading a tensor instead
would mean editing ``models/common/moe.py`` and the ``RoutedExperts`` sharding
boundary, which the experiments rule rules out. The two chains::

    capture, at theta_t
        AnticipatorySchedule._prefetch_and_capture   schedule.py
        -> RoutingIndexCache.capturing(slot)         cache.py
        -> AnticipatoryTrainingEngine
               .forward_only_microbatch              engine.py   (no grad, eval)
        -> the model's forward
        -> AnticipatoryRoutingMixin._select_experts  router.py
        -> RoutingIndexCache.capture(key, ids)       cache.py

    replay, delay_steps later, at theta_{t+dt}
        AnticipatorySchedule.step                    schedule.py
        -> RoutingIndexCache.replaying(slots)        cache.py
        -> Trainer.train_step                        torchtitan/trainer.py
        -> AnticipatoryTrainingEngine
               .forward_backward_microbatch          engine.py
        -> RoutingIndexCache.select(accum_index)     cache.py
        -> AnticipatoryRoutingMixin._select_experts  router.py
        -> RoutingIndexCache.replay(key)             cache.py

``key`` is the router's FQN, assigned once by ``build_routing_cache``, so a slot
captured in one forward is found again in a later one.

That the gap between the two is exactly ``delay_steps`` is not visible in either
chain -- it emerges from the queue, which warmup fills with ``delay_steps``
entries before every later step pushes one and pops one. It is asserted at the
pop, in ``AnticipatorySchedule._check_staleness``.
"""

from .config import AnticipatoryRoutingConfig
from .detector import LossSpikeDetector
from .engine import AnticipatoryTrainingEngine
from .schedule import AnticipatorySchedule, Phase
from .trainer import AnticipatoryTrainer


__all__ = [
    "AnticipatoryRoutingConfig",
    "AnticipatorySchedule",
    "AnticipatoryTrainer",
    "AnticipatoryTrainingEngine",
    "LossSpikeDetector",
    "Phase",
]
