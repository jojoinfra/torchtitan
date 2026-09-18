# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import enum
from typing import Literal

import torch


# One microbatch's worth of routing indices, keyed by router FQN.
RoutingSlot = dict[str, torch.Tensor]

IndexStoreDtype = Literal["auto", "int16", "int32", "int64"]

_DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
}


class RoutingMode(enum.Enum):
    """What the routers do with their top-k index decision this forward."""

    OFF = "off"
    """Behave exactly like the unmodified router."""

    CAPTURE = "capture"
    """Compute indices normally and store them in the current slot."""

    REPLAY = "replay"
    """Ignore the scores and use the indices already in the current slot."""


def resolve_store_dtype(name: IndexStoreDtype, num_experts: int) -> torch.dtype:
    """Pick the narrowest integer dtype that can hold an expert id.

    A slot holds one ``(T, K)`` index tensor per MoE layer, and a delay of
    ``dt`` steps keeps ``dt`` slots alive at once, so the dtype is worth
    narrowing: int16 is a quarter of the int64 the top-k returns.
    """
    if name != "auto":
        dtype = _DTYPE_BY_NAME[name]
        if num_experts - 1 > torch.iinfo(dtype).max:
            raise ValueError(
                f"index_store_dtype={name} cannot represent expert ids up to "
                f"{num_experts - 1}."
            )
        return dtype
    if num_experts - 1 <= torch.iinfo(torch.int16).max:
        return torch.int16
    return torch.int32


class RoutingIndexCache:
    """Store the anticipatory routers read from and write to.

    One instance is shared by every router in the model. The trainer sets
    ``mode`` and ``slot`` around each microbatch forward; a router only ever
    looks at those two fields, so nothing has to thread an extra argument
    through the model.
    """

    def __init__(
        self,
        *,
        store_dtype: torch.dtype,
        offload_to_cpu: bool = False,
    ) -> None:
        self.mode = RoutingMode.OFF
        self.slot: RoutingSlot | None = None
        self.store_dtype = store_dtype
        self.offload_to_cpu = offload_to_cpu

    def capture(self, key: str, topk_expert_ids_TK: torch.Tensor) -> None:
        """Store one router's indices for the microbatch being executed."""
        if self.slot is None:
            raise RuntimeError(
                "RoutingIndexCache is in CAPTURE mode with no slot set. The "
                "trainer must assign a slot before each microbatch forward."
            )
        # copy=True even when the dtype already matches: without it ``to``
        # returns the caller's tensor, which is still attached to the forward
        # that produced it.
        stored = topk_expert_ids_TK.detach().to(dtype=self.store_dtype, copy=True)
        if self.offload_to_cpu:
            stored = stored.to("cpu")
        self.slot[key] = stored

    def replay(self, key: str) -> torch.Tensor:
        """Return the indices cached for this router in the current slot."""
        if self.slot is None:
            raise RuntimeError(
                "RoutingIndexCache is in REPLAY mode with no slot set. The "
                "trainer must assign a slot before each microbatch forward."
            )
        cached = self.slot.get(key)
        if cached is None:
            # Never fall back to freshly computed indices: that would silently
            # train some layers on current-parameter routing and some on stale
            # routing, which is not the technique.
            raise KeyError(
                f"No cached routing indices for router '{key}'. The capture "
                "pass must visit every router the training forward visits."
            )
        return cached


def slot_nbytes(slot: RoutingSlot) -> int:
    """Bytes held by one microbatch's cached indices."""
    return sum(t.numel() * t.element_size() for t in slot.values())
