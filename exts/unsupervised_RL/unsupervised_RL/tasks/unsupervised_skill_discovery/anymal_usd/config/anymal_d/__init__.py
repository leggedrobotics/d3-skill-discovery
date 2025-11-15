# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

"""Not supported yet."""

import gymnasium as gym

from unsupervised_RL.tasks.unsupervised_skill_discovery.anymal_usd.anymal_hl_usd_box_env_cfg import UsdAnymalHLBoxEnvCfg
from unsupervised_RL.tasks.unsupervised_skill_discovery.anymal_usd.anymal_hl_usd_env_cfg import UsdAnymalHLEnvCfg
from unsupervised_RL.tasks.unsupervised_skill_discovery.anymal_usd.anymal_usd_env_cfg import UsdAnymalEnvCfg

from . import agents

##
# USD
##


gym.register(
    id="Isaac-USD-Anymal-D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UsdAnymalEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_usd_cfg:AnymalMetraPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-USD-HL-Anymal-D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UsdAnymalHLEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_usd_cfg:AnymalHL_USD_PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-USD-HL-Box-Anymal-D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UsdAnymalHLBoxEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_usd_cfg:AnymalHL_USD_PPORunnerCfg",
    },
)
