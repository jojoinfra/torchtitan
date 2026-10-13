# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import call, patch

import spmd_types as spmd
import torch
import torch.nn as nn
import torch.nn.functional as F
from spmd_types import SpmdType
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import _per_axis_types
from torchtitan.models.common import moe_sharding
from torchtitan.models.common.activation import Sigmoid, SiTUGLU, Softmax, SqrtSoftplus
from torchtitan.models.common.config_utils import (
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
    make_shared_expert_ffn_config,
)
from torchtitan.models.common.decoder_sharding import (
    dense_param_placement,
    token_id_placement,
)
from torchtitan.models.common.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    Linear,
    RowParallelLinear,
)
from torchtitan.models.common.moe import (
    collect_moe_load_metrics,
    MoE,
    RoundRobinTokenChoiceTopKRouter,
    TokenChoiceTopKRouter,
)
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.kimi_k3.moe import KimiLatentMoE


class _PassthroughRoutedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_tokens_per_expert_E = None

    def forward(
        self,
        x_TD,
        topk_scores_TK,
        topk_expert_ids_TK,
        num_local_tokens_per_expert_E,
        *,
        padding_mask_T=None,
    ):
        self.num_tokens_per_expert_E = num_local_tokens_per_expert_E
        return x_TD


class _RecordingRoutedExperts:
    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def __call__(self, x_TD, scores_TK, expert_ids_TK, counts_E, **kwargs):
        self.kwargs = kwargs
        return x_TD


def _forward_with_routed_mask(
    moe_cls: type[MoE], experts: _RecordingRoutedExperts, padding_mask_T: torch.Tensor
) -> torch.Tensor:
    """Run ``moe_cls.forward`` on stubs; return the mask the routed branch sees."""
    routed_mask_T = padding_mask_T.clone()  # stands for the TP-sharded mask
    moe = SimpleNamespace(
        _maybe_shard_routed_branch_inputs_across_tp=lambda x, mask: (x, routed_mask_T),
        router=lambda x, bias, padding_mask_T=None, **kwargs: (
            torch.zeros(x.shape[0], 2),
            torch.zeros(x.shape[0], 2, dtype=torch.int64),
            torch.zeros(x.shape[0], 4, dtype=torch.bool),
        ),
        expert_bias_E=None,
        routed_experts=experts,
        # KimiLatentMoE projects into and out of the routed experts' latent space.
        routed_down=lambda x: x,
        routed_norm=lambda x: x,
        routed_up=lambda x: x,
        _maybe_zero_fill_routed_output_to_tp_partial=lambda out: out,
        shared_experts=None,
        _maybe_all_reduce_moe_output_across_tp=lambda out: out,
    )
    moe_cls.forward(cast(Any, moe), torch.zeros(4, 3), padding_mask_T=padding_mask_T)
    return routed_mask_T


class _CapturingAuxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.routing_map_TE = None

    def forward(
        self,
        scores_TE,
        routing_map_TE,
        *,
        carrier,
        padding_mask_T=None,
        denominator,
    ):
        del scores_TE, padding_mask_T, denominator
        self.routing_map_TE = routing_map_TE
        return carrier


class _IdentityDispatcher(nn.Module):
    def dispatch(
        self,
        x_TD,
        topk_scores_TK,
        topk_expert_ids_TK,
        num_local_tokens_per_expert_E,
    ):
        del topk_scores_TK, topk_expert_ids_TK
        return x_TD, num_local_tokens_per_expert_E, None

    def combine(self, routed_output_RD, metadata, x_TD):
        del metadata, x_TD
        return routed_output_RD


class _AddOneW13(nn.Module):
    def forward(self, x_RD, offsets_E):
        del offsets_E
        return torch.stack((x_RD + 1, x_RD), dim=-2)


class _SelectGate(nn.Module):
    def forward(self, gate_up_R2D, *, offsets):
        del offsets
        return gate_up_R2D[:, 0]


class _IdentityW2(nn.Module):
    def forward(self, hidden_RD, offsets_E):
        del offsets_E
        return hidden_RD


def _build_moe(num_experts: int = 2) -> MoE:
    dim = 4
    top_k = 1
    moe = make_moe_config(
        num_experts=num_experts,
        router=make_router_config(
            dim=dim,
            num_experts=num_experts,
            gate_param_init={"weight": nn.init.zeros_},
            score_func=Sigmoid.Config(),
            top_k=top_k,
        ),
        routed_experts=make_routed_experts_config(
            dim=dim,
            hidden_dim=8,
            num_experts=num_experts,
            top_k=top_k,
            param_init={},
            enable_ep=False,
        ),
        load_balance_coeff=1e-3,
        enable_sp=True,
    ).build()
    with torch.no_grad():
        moe.router.gate.weight.zero_()
    moe.routed_experts = _PassthroughRoutedExperts()
    return moe


def _model_part_with_expert_counts(
    counts_by_layer: dict[str, list[float]],
) -> nn.Module:
    """A model part whose ``layers.{key}.moe`` routers hold the given counts."""
    layers = nn.ModuleDict()
    for key, counts_E in counts_by_layer.items():
        block = nn.Module()
        block.moe = _build_moe(num_experts=len(counts_E))
        block.moe.router.tokens_per_expert_E.copy_(torch.tensor(counts_E))
        layers[key] = block
    return nn.ModuleDict({"layers": layers})


class TestMoE(unittest.TestCase):
    def test_make_router_config_requires_score_func(self):
        with self.assertRaisesRegex(TypeError, "score_func"):
            make_router_config(
                dim=4,
                num_experts=4,
                gate_param_init={"weight": nn.init.zeros_},
            )

    def test_token_choice_router_requires_score_func(self):
        with self.assertRaisesRegex(TypeError, "score_func"):
            TokenChoiceTopKRouter.Config(
                num_experts=4,
                gate=HiMidLoLinear.Config(in_features=4, out_features=4),
            )

    def test_round_robin_router_balances_assignments(self):
        router = RoundRobinTokenChoiceTopKRouter.Config(
            num_experts=4,
            gate=HiMidLoLinear.Config(in_features=4, out_features=4),
            score_func=Sigmoid.Config(),
            top_k=2,
        ).build()

        _, topk_expert_ids_TK, _ = router(torch.randn(4, 4))

        torch.testing.assert_close(
            topk_expert_ids_TK,
            torch.tensor([[0, 1], [2, 3], [0, 1], [2, 3]]),
        )

    def test_routed_experts_use_configured_activation(self):
        """Routed experts build and execute their configured binary activation."""
        activation_fn = SiTUGLU.Config(beta=4.0, linear_beta=25.0)
        config = make_routed_experts_config(
            dim=4,
            hidden_dim=8,
            num_experts=2,
            top_k=1,
            param_init={},
            enable_ep=False,
        )
        config.activation_fn = activation_fn
        experts = config.build()
        gate_up_R2F = torch.randn(3, 2, 8)

        expected_RF = activation_fn.build()(gate_up_R2F)
        actual_RF = experts.activation_fn(gate_up_R2F)
        torch.testing.assert_close(actual_RF, expected_RF)

    def test_routed_experts_own_postprocess_before_combine(self):
        config = replace(
            make_routed_experts_config(
                dim=4,
                hidden_dim=4,
                num_experts=2,
                top_k=1,
                param_init={},
                enable_ep=False,
            ),
            output_postprocess=RMSNorm.Config(normalized_shape=4),
        )
        routed_experts = config.build()
        routed_experts.w13 = _AddOneW13()
        routed_experts.activation_fn = _SelectGate()
        routed_experts.w2 = _IdentityW2()
        routed_experts.token_dispatcher = _IdentityDispatcher()
        assert isinstance(routed_experts.output_postprocess, RMSNorm)
        with torch.no_grad():
            routed_experts.output_postprocess.weight.fill_(3.0)
        x_TD = torch.arange(8, dtype=torch.float32).reshape(2, 4).requires_grad_()

        output_TD = routed_experts(
            x_TD,
            torch.ones(2, 1),
            torch.zeros(2, 1, dtype=torch.int64),
            torch.tensor([2, 0]),
        )
        expected_TD = F.rms_norm(
            x_TD + 1,
            (4,),
            routed_experts.output_postprocess.weight,
            routed_experts.output_postprocess.eps,
        )
        torch.testing.assert_close(output_TD, expected_TD)
        self.assertIn("output_postprocess.weight", routed_experts.state_dict())

        output_TD.sum().backward()
        self.assertIsNotNone(x_TD.grad)
        self.assertIsNotNone(routed_experts.output_postprocess.weight.grad)

    def test_token_choice_router_uses_normalization_epsilon(self):
        x_TD = torch.zeros(1, 4)
        expert_bias_E = torch.tensor([4.0, 3.0, 2.0, 1.0])
        route_norm_epsilon = 1.0
        route_scale = 4.0

        for route_norm in (False, True):
            with self.subTest(route_norm=route_norm):
                config = make_router_config(
                    dim=4,
                    num_experts=4,
                    score_func=Sigmoid.Config(),
                    gate_param_init={"weight": nn.init.zeros_},
                    top_k=2,
                    route_norm=route_norm,
                    route_norm_epsilon=route_norm_epsilon,
                    route_scale=route_scale,
                )
                router = config.build()
                with torch.no_grad():
                    router.gate.weight.zero_()

                (
                    actual_topk_scores_TK,
                    actual_topk_expert_ids_TK,
                    _,
                ) = router(x_TD, expert_bias_E=expert_bias_E)
                actual_scores_TE = torch.zeros_like(x_TD).scatter(
                    dim=-1,
                    index=actual_topk_expert_ids_TK,
                    src=actual_topk_scores_TK,
                )
                expected_scores_TE = torch.tensor(
                    [[1.0, 1.0, 0.0, 0.0]] if route_norm else [[2.0, 2.0, 0.0, 0.0]]
                )

                torch.testing.assert_close(
                    actual_scores_TE,
                    expected_scores_TE,
                    rtol=0,
                    atol=0,
                )

    def test_token_choice_router_uses_configured_score_functions(self):
        x_TD = torch.tensor([[-2.0, 0.0, 1.0, 3.0]], dtype=torch.bfloat16)
        x_fp32_TD = x_TD.float()
        cases = (
            ("sigmoid", Sigmoid.Config(), torch.sigmoid(x_fp32_TD)),
            ("softmax", Softmax.Config(), F.softmax(x_fp32_TD, dim=-1)),
            (
                "sqrtsoftplus",
                SqrtSoftplus.Config(),
                F.softplus(x_fp32_TD).sqrt(),
            ),
        )

        for name, score_func, expected_scores_TE in cases:
            with self.subTest(score_func=name):
                config = make_router_config(
                    dim=4,
                    num_experts=4,
                    gate_param_init={"weight": nn.init.zeros_},
                    score_func=score_func,
                    top_k=4,
                )
                router = config.build()
                with torch.no_grad():
                    router.gate.weight.copy_(torch.eye(4))

                (
                    actual_topk_scores_TK,
                    actual_topk_expert_ids_TK,
                    _,
                ) = router(x_TD)
                actual_scores_TE = torch.zeros_like(expected_scores_TE).scatter(
                    dim=-1,
                    index=actual_topk_expert_ids_TK,
                    src=actual_topk_scores_TK,
                )

                self.assertIs(actual_topk_scores_TK.dtype, torch.float32)
                torch.testing.assert_close(
                    actual_scores_TE,
                    expected_scores_TE,
                    rtol=0,
                    atol=0,
                )

    def test_eval_forward_does_not_accumulate_tokens_per_expert(self):
        dim = 4
        top_k = 1
        moe = _build_moe()

        x_TD = torch.randn(6, dim)
        moe.train()
        moe(x_TD)
        self.assertEqual(moe.router.tokens_per_expert_E.sum().item(), 2 * 3 * top_k)
        training_counts = moe.router.tokens_per_expert_E.clone()

        moe.eval()
        with torch.no_grad():
            moe(x_TD)

        torch.testing.assert_close(
            moe.router.tokens_per_expert_E,
            training_counts,
        )

    def test_padding_is_excluded_from_counts_but_still_dispatched(self):
        moe = _build_moe()
        x_TD = torch.randn(6, 4)
        padding_mask_T = torch.tensor([False, False, False, True, True, True])

        moe.train()
        moe(x_TD, padding_mask_T=padding_mask_T)

        self.assertEqual(moe.router.tokens_per_expert_E.sum().item(), 3)
        self.assertEqual(moe.routed_experts.num_tokens_per_expert_E.sum().item(), 6)

    def test_moe_hands_the_routed_padding_mask_to_the_experts(self):
        for moe_cls in (MoE, KimiLatentMoE):
            with self.subTest(moe_cls=moe_cls.__name__):
                experts = _RecordingRoutedExperts()
                routed_mask_T = _forward_with_routed_mask(
                    moe_cls, experts, torch.tensor([False, False, True, True])
                )
                self.assertEqual(experts.kwargs, {"padding_mask_T": routed_mask_T})
                self.assertIs(experts.kwargs["padding_mask_T"], routed_mask_T)

    def test_router_masks_padding_only_for_aux_loss(self):
        router = make_router_config(
            dim=4,
            num_experts=2,
            score_func=Sigmoid.Config(),
            gate_param_init={"weight": nn.init.zeros_},
            top_k=1,
        ).build()
        router.init_states()
        aux_loss = _CapturingAuxLoss()
        router.aux_loss = aux_loss
        router.train()

        padding_mask_T = torch.tensor([False, False, True, True])
        _, _, routing_map_TE = router(
            torch.randn(4, 4),
            padding_mask_T=padding_mask_T,
            aux_loss_denominator=torch.tensor(2),
        )

        self.assertTrue(routing_map_TE[padding_mask_T].any())
        self.assertIsNotNone(aux_loss.routing_map_TE)
        self.assertFalse(aux_loss.routing_map_TE[padding_mask_T].any())
        torch.testing.assert_close(
            aux_loss.routing_map_TE[~padding_mask_T],
            routing_map_TE[~padding_mask_T],
        )
        torch.testing.assert_close(
            router.tokens_per_expert_E,
            aux_loss.routing_map_TE.sum(dim=0).to(torch.float32),
        )

    def test_router_validates_padding_mask(self):
        router = make_router_config(
            dim=4,
            num_experts=2,
            score_func=Sigmoid.Config(),
            gate_param_init={"weight": nn.init.zeros_},
            top_k=1,
        ).build()
        x_TD = torch.randn(4, 4)

        with self.assertRaisesRegex(ValueError, "dtype bool"):
            router(x_TD, padding_mask_T=torch.zeros(4))
        with self.assertRaisesRegex(ValueError, "routing-map token axis"):
            router(x_TD, padding_mask_T=torch.zeros(3, dtype=torch.bool))

    def test_padding_mask_enters_moe_replicated_and_router_sharded(self):
        for enable_sp in (False, True):
            with self.subTest(enable_sp=enable_sp):
                moe_config = moe_sharding.moe_sharding_config(enable_sp=enable_sp)
                assert moe_config.in_src_shardings is not None
                self.assertIsNone(moe_config.in_dst_shardings)
                self.assertEqual(
                    _per_axis_types(moe_config.in_src_shardings["padding_mask_T"]),
                    _per_axis_types(token_id_placement()),
                )

                router_config = moe_sharding.router_sharding_config()
                router_inputs = router_config.in_src_shardings
                assert router_inputs is not None
                self.assertIsNone(router_config.in_dst_shardings)
                self.assertEqual(
                    _per_axis_types(router_inputs["x_TD"])[MeshAxisName.TP],
                    spmd.S(0),
                )
                self.assertEqual(
                    _per_axis_types(router_inputs["padding_mask_T"]),
                    _per_axis_types(token_id_placement(enable_sp=True)),
                )

    def test_shared_expert_uses_runtime_sp_aware_output_projection(self):
        config = make_shared_expert_ffn_config(
            dim=4,
            hidden_dim=8,
            w13_param_init={},
            w2_param_init={},
            enable_sp=True,
        )

        self.assertIs(type(config.w13), ColumnParallelLinear.Config)
        self.assertIs(type(config.w2), RowParallelLinear.Config)
        self.assertEqual(config.w13.num_linears, 2)

    def test_common_shared_expert_input_gather_is_owned_by_w13(self):
        moe_config = moe_sharding.moe_sharding_config(enable_sp=True)
        shared = make_shared_expert_ffn_config(
            dim=4, hidden_dim=8, w13_param_init={}, w2_param_init={}, enable_sp=True
        )
        shared_config, w13_config = shared.sharding_config, shared.w13.sharding_config

        assert moe_config.in_src_shardings is not None
        assert shared_config.in_src_shardings is not None
        assert w13_config.in_src_shardings is not None
        self.assertIsNone(moe_config.in_dst_shardings)
        self.assertIsNone(shared_config.in_dst_shardings)
        self.assertIsNone(w13_config.in_dst_shardings)
        self.assertEqual(
            _per_axis_types(moe_config.in_src_shardings["x_TD"])[MeshAxisName.TP],
            spmd.S(0),
        )
        self.assertEqual(
            _per_axis_types(shared_config.in_src_shardings["x"])[MeshAxisName.TP],
            spmd.S(0),
        )
        self.assertEqual(
            _per_axis_types(w13_config.in_src_shardings["input"])[MeshAxisName.TP],
            spmd.S(0),
        )

    def test_explicit_moe_tp_transitions_with_ep_without_sp(self):
        moe = MoE.__new__(MoE)
        x_TD = torch.randn(4, 8)
        padding_mask_T = torch.zeros(4, dtype=torch.bool)
        tp_group = object()

        with (
            patch(
                "torchtitan.models.common.moe.spmd_sparse_mesh",
                return_value=object(),
            ),
            patch(
                "torchtitan.models.common.moe.spmd_dense_sp_enabled",
                return_value=False,
            ),
            patch(
                "torchtitan.models.common.moe.spmd_mesh_group",
                return_value=tp_group,
            ),
            patch(
                "torchtitan.models.common.moe.spmd.redistribute",
                side_effect=lambda tensor, *_args, **_kwargs: tensor,
            ) as redistribute,
            patch(
                "torchtitan.models.common.moe.remat.region",
                side_effect=lambda function, *_args, **_kwargs: function,
            ) as region,
            patch(
                "torchtitan.models.common.moe.remat.recompute_needs_tensor"
            ) as recompute_needs_tensor,
        ):
            moe._maybe_shard_routed_branch_inputs_across_tp(x_TD, padding_mask_T)
            moe._maybe_zero_fill_routed_output_to_tp_partial(x_TD)
            moe._maybe_all_reduce_moe_output_across_tp(x_TD)
            self.assertEqual(
                redistribute.call_args_list,
                [
                    call(
                        x_TD,
                        tp_group,
                        src=spmd.I,
                        dst=spmd.S(0),
                        backward_options={"op_dtype": x_TD.dtype},
                    ),
                    call(
                        padding_mask_T,
                        tp_group,
                        src=spmd.R,
                        dst=spmd.S(0),
                        backward_options={"op_dtype": padding_mask_T.dtype},
                    ),
                    call(
                        x_TD,
                        tp_group,
                        src=spmd.S(0),
                        dst=spmd.P,
                        backward_options={"op_dtype": x_TD.dtype},
                    ),
                    call(
                        x_TD,
                        tp_group,
                        src=spmd.P,
                        dst=spmd.I,
                        backward_options={"op_dtype": x_TD.dtype},
                    ),
                ],
            )
            region.assert_called_once_with(
                spmd.redistribute,
                "tp_output_reduction",
                recompute=True,
            )
            # The zero-fill pins the routed output it reads; the block's residual
            # add pins the reduced output.
            recompute_needs_tensor.assert_called_once_with(x_TD)

    def test_routed_branch_rejects_tp_without_ep(self):
        moe = MoE.__new__(MoE)
        x_TD = torch.randn(4, 8)

        with (
            patch(
                "torchtitan.models.common.moe.spmd_sparse_mesh",
                return_value=None,
            ),
            patch(
                "torchtitan.models.common.moe.spmd_mesh_group",
                return_value=object(),
            ),
            self.assertRaisesRegex(
                AssertionError,
                "requires expert parallelism",
            ),
        ):
            moe._maybe_shard_routed_branch_inputs_across_tp(x_TD, None)

    def test_expert_branch_layouts_before_moe_boundary(self):
        for enable_sp, expected in (
            (False, spmd.P),
            (True, spmd.S(0)),
        ):
            with self.subTest(enable_sp=enable_sp):
                shared_experts = make_shared_expert_ffn_config(
                    dim=4,
                    hidden_dim=8,
                    w13_param_init={},
                    w2_param_init={},
                    enable_sp=enable_sp,
                )
                shared = shared_experts.sharding_config
                w2 = shared_experts.w2.sharding_config
                # Without SP, a plain Linear keeps the TP-Partial output for
                # the MoE combine; with SP, RowParallelLinear reduce-scatters.
                self.assertIs(
                    type(shared_experts.w2),
                    RowParallelLinear.Config if enable_sp else Linear.Config,
                )
                routed = make_routed_experts_config(
                    dim=4,
                    hidden_dim=8,
                    num_experts=2,
                    top_k=1,
                    param_init={},
                    enable_ep=True,
                ).sharding_config

                shared_output = shared.out_src_shardings
                w2_output = w2.out_src_shardings
                routed_output = routed.out_src_shardings
                assert isinstance(shared_output, SpmdType)
                assert isinstance(w2_output, SpmdType)
                assert isinstance(routed_output, SpmdType)
                self.assertEqual(
                    _per_axis_types(shared_output).get(MeshAxisName.TP),
                    expected,
                )
                self.assertEqual(
                    _per_axis_types(w2_output).get(MeshAxisName.TP), expected
                )
                self.assertIsNone(w2.out_dst_shardings)
                self.assertEqual(
                    _per_axis_types(routed_output).get(MeshAxisName.TP),
                    spmd.S(0),
                )
                self.assertIsNone(routed.out_dst_shardings)

    def test_explicit_padding_mask_sharding_with_sp(self):
        moe = MoE.__new__(MoE)
        x_TD = torch.randn(4, 8)
        padding_mask_T = torch.zeros(4, dtype=torch.bool)
        tp_group = object()

        with (
            patch(
                "torchtitan.models.common.moe.spmd_sparse_mesh",
                return_value=object(),
            ),
            patch(
                "torchtitan.models.common.moe.spmd_dense_sp_enabled",
                return_value=True,
            ),
            patch(
                "torchtitan.models.common.moe.spmd_mesh_group",
                return_value=tp_group,
            ),
            patch(
                "torchtitan.models.common.moe.spmd.redistribute",
                side_effect=lambda tensor, *_args, **_kwargs: tensor,
            ) as redistribute,
        ):
            (
                actual_x_TD,
                actual_padding_mask_T,
            ) = moe._maybe_shard_routed_branch_inputs_across_tp(x_TD, padding_mask_T)

        self.assertIs(actual_x_TD, x_TD)
        self.assertIs(actual_padding_mask_T, padding_mask_T)
        redistribute.assert_called_once_with(
            padding_mask_T,
            tp_group,
            src=spmd.R,
            dst=spmd.S(0),
            backward_options={"op_dtype": padding_mask_T.dtype},
        )

    def test_moe_without_ep_leaves_routed_weights_unsharded(self):
        routed = make_routed_experts_config(
            dim=4,
            hidden_dim=8,
            num_experts=2,
            top_k=1,
            param_init={},
            enable_ep=False,
        )
        for grouped_linear in (routed.w13, routed.w2):
            self.assertEqual(
                grouped_linear.sharding_config.state_shardings["weight"],
                dense_param_placement(tp=spmd.R),
            )


_SINGLE_RANK = SimpleNamespace(
    ep_enabled=False, tp=1, get_optional_mesh=lambda name: None
)


class TestCollectMoELoadMetrics(unittest.TestCase):
    def test_reports_summaries_and_per_layer_statistics(self):
        model_part = _model_part_with_expert_counts({"0": [3.0, 1.0], "1": [2.0, 0.0]})
        metrics = collect_moe_load_metrics(
            [model_part],
            ["layers.0.moe", "layers.1.moe"],
            _SINGLE_RANK,
            per_layer=True,
        )

        expected = {
            "moe_load/max_vio_mean": 0.75,
            "moe_load/max_vio_max": 1.0,
            "moe_load/cv_mean": 0.75,
            "moe_load/dead_experts": 1.0,
            "moe_load_per_layer/layers.0.moe/max_vio": 0.5,
            "moe_load_per_layer/layers.0.moe/cv": 0.5,
            "moe_load_per_layer/layers.0.moe/dead_experts": 0.0,
            "moe_load_per_layer/layers.1.moe/max_vio": 1.0,
            "moe_load_per_layer/layers.1.moe/cv": 1.0,
            "moe_load_per_layer/layers.1.moe/dead_experts": 1.0,
        }
        self.assertEqual(metrics.keys(), expected.keys())
        for key, value in expected.items():
            self.assertAlmostEqual(metrics[key], value, places=6, msg=key)
        # Reading copies the counts: the balancing hook still needs them.
        torch.testing.assert_close(
            model_part.layers["0"].moe.router.tokens_per_expert_E,
            torch.tensor([3.0, 1.0]),
        )

    def test_layer_without_tokens_reports_zeros(self):
        model_part = _model_part_with_expert_counts({"0": [0.0, 0.0]})
        metrics = collect_moe_load_metrics(
            [model_part], ["layers.0.moe"], _SINGLE_RANK, per_layer=False
        )
        self.assertEqual(
            metrics,
            {
                "moe_load/max_vio_mean": 0.0,
                "moe_load/max_vio_max": 0.0,
                "moe_load/cv_mean": 0.0,
                "moe_load/dead_experts": 2.0,
            },
        )

    def test_dense_model_reports_nothing(self):
        metrics = collect_moe_load_metrics(
            [nn.Linear(2, 2)], [], _SINGLE_RANK, per_layer=True
        )
        self.assertEqual(metrics, {})

    def test_rejects_moe_layer_missing_from_config(self):
        model_part = _model_part_with_expert_counts({"0": [1.0, 1.0]})
        with self.assertRaisesRegex(AssertionError, "layers.0.moe"):
            collect_moe_load_metrics(
                [model_part], ["layers.1.moe"], _SINGLE_RANK, per_layer=False
            )


class TestCollectMoELoadMetricsDistributed(DTensorTestBase):
    """dp2/cp2/pp2 on CPU + gloo: the four DP/CP ranks of a stage route
    different tokens, and each stage holds different MoE layers."""

    @property
    def world_size(self):
        return 8

    @property
    def device_type(self):
        return "cpu"

    def _build_dims(self):
        from torchtitan.distributed.parallelism_context import ParallelismContext

        with patch("torchtitan.distributed.parallelism_context.device_type", "cpu"):
            parallelism_context = ParallelismContext(
                dp_replicate=1,
                dp_shard=2,
                cp=2,
                tp=1,
                pp=2,
                ep=1,
                world_size=8,
                enable_sequence_parallel=False,
            )
            parallelism_context.build_mesh()
        return parallelism_context

    @with_comms
    def test_sums_loss_mesh_and_gathers_stages(self):
        parallelism_context = self._build_dims()
        pp_rank = parallelism_context.get_mesh("pp").get_local_rank()
        loss_rank = parallelism_context.get_mesh("loss").get_local_rank()
        # Stage 0 holds layer 0, stage 1 layers 1 and 2. Global counts:
        #   layer 0: one expert per loss rank -> [1, 1, 1, 1]
        #   layer 1: loss_rank + 1 on expert 0 -> [10, 0, 0, 0]
        #   layer 2: the same on every rank   -> [4, 4, 0, 0]
        if pp_rank == 0:
            counts = {"0": [float(e == loss_rank) for e in range(4)]}
        else:
            counts = {"1": [loss_rank + 1.0, 0.0, 0.0, 0.0], "2": [1.0, 1.0, 0.0, 0.0]}
        model_part = _model_part_with_expert_counts(counts)

        metrics = collect_moe_load_metrics(
            [model_part],
            ["layers.0.moe", "layers.1.moe", "layers.2.moe"],
            parallelism_context,
            per_layer=True,
        )

        sqrt3 = 3.0**0.5
        expected = {
            "moe_load/max_vio_mean": 4.0 / 3.0,
            "moe_load/max_vio_max": 3.0,
            "moe_load/cv_mean": (sqrt3 + 1.0) / 3.0,
            "moe_load/dead_experts": 5.0,
            "moe_load_per_layer/layers.0.moe/max_vio": 0.0,
            "moe_load_per_layer/layers.0.moe/cv": 0.0,
            "moe_load_per_layer/layers.0.moe/dead_experts": 0.0,
            "moe_load_per_layer/layers.1.moe/max_vio": 3.0,
            "moe_load_per_layer/layers.1.moe/cv": sqrt3,
            "moe_load_per_layer/layers.1.moe/dead_experts": 3.0,
            "moe_load_per_layer/layers.2.moe/max_vio": 1.0,
            "moe_load_per_layer/layers.2.moe/cv": 1.0,
            "moe_load_per_layer/layers.2.moe/dead_experts": 2.0,
        }
        self.assertEqual(metrics.keys(), expected.keys())
        for key, value in expected.items():
            self.assertAlmostEqual(metrics[key], value, places=5, msg=key)

    @with_comms
    def test_stage_without_moe_layers_joins_the_gather(self):
        """A dense first stage (e.g. DeepSeek's leading dense layers) must
        still join the pipeline reduction instead of deadlocking it."""
        parallelism_context = self._build_dims()
        pp_rank = parallelism_context.get_mesh("pp").get_local_rank()
        if pp_rank == 0:
            model_part = nn.ModuleDict(
                {"layers": nn.ModuleDict({"0": nn.Linear(2, 2)})}
            )
        else:
            model_part = _model_part_with_expert_counts({"1": [1.0, 0.0]})

        with patch("torchtitan.models.common.moe.device_type", "cpu"):
            metrics = collect_moe_load_metrics(
                [model_part], ["layers.1.moe"], parallelism_context, per_layer=False
            )

        # Four loss ranks each route one token to expert 0: [4, 0].
        self.assertAlmostEqual(metrics["moe_load/max_vio_max"], 1.0, places=6)
        self.assertEqual(metrics["moe_load/dead_experts"], 1.0)


if __name__ == "__main__":
    unittest.main()
