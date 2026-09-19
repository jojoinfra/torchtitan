# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
import functools
import logging

import torch
from torch import nn

from torchtitan.config.override import derive, override
from torchtitan.models.common.moe import TokenChoiceTopKRouter

from .cache import IndexStoreDtype, resolve_store_dtype, RoutingIndexCache, RoutingMode


logger = logging.getLogger(__name__)

OVERRIDE_TARGET = (
    "torchtitan.experiments.anticipatory_routing.router.anticipatory_router"
)
"""``override.imports`` entry that installs the anticipatory routers."""


class AnticipatoryRoutingMixin:
    """Capture or replay a router's top-k expert ids.

    Mixed in front of a concrete router class so that ``super()._select_experts``
    still runs that model's own selection logic -- group-limited selection for
    DeepSeek-V3, the top-(k+1) cutoff for quantile balancing, and so on.

    ``_select_experts`` is the right seam because it is the single producer of
    ``topk_expert_ids_TK``: the gating scores, the routing map, the per-expert
    token counts, the dispatcher's permutation and the load-balancing statistics
    are all derived from its return value. It also already runs inside the
    ``routing_decision`` remat region with ``recompute=False``, so it executes
    exactly once per forward even under activation checkpointing.

    ``_routing_cache`` is wired up after the model is built, by
    :func:`build_routing_cache`. Until then the router behaves like its base
    class.
    """

    _routing_cache: RoutingIndexCache | None
    _routing_key: str

    def __init__(self, config) -> None:
        super().__init__(config)
        self._routing_cache = None
        self._routing_key = ""

    def _select_experts(
        self,
        scores_TE: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
        **router_kwargs,
    ) -> torch.Tensor:
        cache = self._routing_cache
        if cache is None or cache.mode is RoutingMode.OFF:
            return super()._select_experts(scores_TE, expert_bias_E, **router_kwargs)

        if cache.mode is RoutingMode.REPLAY:
            cached = cache.replay(self._routing_key)
            # Materialize the cached ids through ``scores_TE`` rather than
            # returning the stored tensor: the RoutedExperts boundary asserts a
            # layout on topk_expert_ids_TK, and a tensor read back from a plain
            # dict carries no SPMD type. Under CP -- and under TP when EP is on
            # -- the token dim is sharded, and deriving the result from a slice
            # of the router's own output is what makes that visible here.
            topk_expert_ids_TK = torch.zeros_like(
                scores_TE[..., : self.top_k], dtype=torch.int64
            )
            topk_expert_ids_TK.copy_(cached)
            return topk_expert_ids_TK

        topk_expert_ids_TK = super()._select_experts(
            scores_TE, expert_bias_E, **router_kwargs
        )
        cache.capture(self._routing_key, topk_expert_ids_TK)
        return topk_expert_ids_TK


@functools.cache
def anticipatory_router_class(base_cls: type) -> type:
    """Return ``base_cls`` with anticipatory capture/replay mixed in front.

    Generated rather than written out once because every model brings its own
    router subclass (``DeepSeekV3Router``, ``DeepSeekV4Router``,
    ``QuantileBalancedTopKRouter``), and the mixin has to sit ahead of whichever
    one the config names. ``Configurable.__init_subclass__`` wires ``_owner`` on
    the generated Config, so ``Config.build()`` constructs the generated class.
    Cached so repeated overrides of the same base share one class.
    """
    config_cls = dataclasses.make_dataclass(
        f"Anticipatory{base_cls.__name__}Config",
        [],
        bases=(base_cls.Config,),
        kw_only=True,
        slots=True,
    )
    return type(
        f"Anticipatory{base_cls.__name__}",
        (AnticipatoryRoutingMixin, base_cls),
        {"Config": config_cls},
    )


@override(
    target=TokenChoiceTopKRouter.Config,
    description="Cache and replay top-k routing indices for anticipatory routing",
)
def anticipatory_router(
    config: TokenChoiceTopKRouter.Config,
) -> TokenChoiceTopKRouter.Config:
    """Replace every router config with its anticipatory counterpart.

    ``@override`` matches subclasses by default, so this one entry covers the
    base router and every per-model subclass. ``derive`` copies all fields,
    including the ``sharding_config`` that ``set_moe_sharding_config`` attached
    before overrides ran.
    """
    base_cls = type(config)._owner
    if base_cls is None:
        raise ValueError(
            f"{type(config).__qualname__} has no owner class, so the "
            "anticipatory router cannot be derived from it."
        )
    return derive(config, anticipatory_router_class(base_cls).Config)


def build_routing_cache(
    model_parts: list[nn.Module],
    *,
    index_store_dtype: IndexStoreDtype,
    offload_to_cpu: bool,
    device: torch.device,
) -> RoutingIndexCache:
    """Create the shared cache and attach it to every anticipatory router.

    Each router gets a key that is stable for the life of the process, so a slot
    captured in one forward can be looked up in a later one.
    """
    routers: list[tuple[str, AnticipatoryRoutingMixin]] = []
    plain_routers = 0
    for part_index, part in enumerate(model_parts):
        for fqn, module in part.named_modules():
            if isinstance(module, AnticipatoryRoutingMixin):
                routers.append((f"{part_index}.{fqn}", module))
            elif isinstance(module, TokenChoiceTopKRouter):
                plain_routers += 1

    if plain_routers:
        raise ValueError(
            f"{plain_routers} MoE router(s) were not replaced by the "
            "anticipatory router. Every router must be replaced, or the model "
            "would mix stale and current routing across layers. Check that "
            f"'{OVERRIDE_TARGET}' is listed in override.imports and that no "
            "other override claims the router nodes."
        )
    if not routers:
        raise ValueError(
            "Anticipatory routing found no MoE routers in the model. Check that "
            "the model has MoE layers and that "
            f"'{OVERRIDE_TARGET}' is listed in override.imports."
        )

    num_experts = routers[0][1].num_experts
    cache = RoutingIndexCache(
        store_dtype=resolve_store_dtype(index_store_dtype, num_experts),
        device=device,
        offload_to_cpu=offload_to_cpu,
    )
    for key, router in routers:
        router._routing_cache = cache
        router._routing_key = key

    logger.info(
        "Anticipatory routing attached to %d router(s); indices stored as %s.",
        len(routers),
        cache.store_dtype,
    )
    return cache
