# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields, replace

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config.override import OverrideConfig
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_16b,
    deepseek_v3_debugmodel,
)
from torchtitan.trainer import Trainer

from .config import AnticipatoryRoutingConfig
from .detector import LossSpikeDetector
from .router import OVERRIDE_TARGET
from .trainer import AnticipatoryTrainer


def to_anticipatory_config(
    base_config: Trainer.Config,
    *,
    anticipatory: AnticipatoryRoutingConfig,
    checkpointer: CheckpointManager.Config | None = None,
) -> AnticipatoryTrainer.Config:
    """Adapt a base recipe for anticipatory routing.

    Copies every field, installs the router override, forces CUDA graphs off
    (a graph captured in one routing mode would be replayed in every other),
    and supplies a checkpointer when the base recipe has none, since rollback
    needs checkpoints written during the run.
    """
    values = {f.name: getattr(base_config, f.name) for f in fields(base_config)}

    values["training"] = replace(base_config.training, disable_cuda_graphs=True)

    # A fresh list: dataclasses.replace is shallow, and mutating the base
    # recipe's list would leak the override into unrelated configs built from
    # the same function.
    imports = list(base_config.override.imports)
    if OVERRIDE_TARGET not in imports:
        imports.append(OVERRIDE_TARGET)
    values["override"] = OverrideConfig(imports=imports)

    if checkpointer is not None:
        values["checkpointer"] = checkpointer
    elif values["checkpointer"] is None:
        raise ValueError(
            "Anticipatory routing needs a checkpointer, and the base recipe "
            "does not configure one. Pass checkpointer=... ."
        )

    values["anticipatory"] = anticipatory
    return AnticipatoryTrainer.Config(**values)


def anticipatory_deepseek_v3_debugmodel() -> AnticipatoryTrainer.Config:
    """Small config for validating the mechanism end to end.

    ``always_on`` skips spike detection so a short run exercises the warmup,
    active and drain phases without waiting for a spike. Set
    ``--anticipatory.delay_steps 0`` to reproduce the baseline loss exactly.
    """
    return to_anticipatory_config(
        deepseek_v3_debugmodel(),
        checkpointer=CheckpointManager.Config(interval=2, keep_latest_k=0),
        anticipatory=AnticipatoryRoutingConfig(
            delay_steps=2,
            active_steps=4,
            always_on=True,
        ),
    )


def anticipatory_deepseek_v3_debugmodel_spike() -> AnticipatoryTrainer.Config:
    """Debug config driven by the spike detector rather than ``always_on``.

    The detector thresholds are loosened so a short run can actually trigger;
    production values belong closer to the defaults.
    """
    config = anticipatory_deepseek_v3_debugmodel()
    config.anticipatory = AnticipatoryRoutingConfig(
        delay_steps=2,
        active_steps=4,
        detector=LossSpikeDetector.Config(
            warmup_steps=2,
            z_threshold=2.0,
            onset_z_threshold=1.0,
            cooldown_steps=4,
        ),
    )
    return config


def anticipatory_deepseek_v3_16b() -> AnticipatoryTrainer.Config:
    return to_anticipatory_config(
        deepseek_v3_16b(),
        checkpointer=CheckpointManager.Config(interval=100, keep_latest_k=10),
        anticipatory=AnticipatoryRoutingConfig(
            delay_steps=16,
            active_steps=500,
        ),
    )
