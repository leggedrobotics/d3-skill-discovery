# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrappers and utilities to configure an :class:`ManagerBasedRLEnv` for RSL-RL library."""

from .crl_cfg import (
    RslCRlOnPolicyRunnerCfg,
    RslRlContrastiveCriticCfg,
    RslRlCrlAlgorithmCfg,
    RslRlGoalConditionedActorCfg,
)

__all__ = [
    "RslCRlOnPolicyRunnerCfg",
    "RslRlContrastiveCriticCfg",
    "RslRlCrlAlgorithmCfg",
    "RslRlGoalConditionedActorCfg",
]
