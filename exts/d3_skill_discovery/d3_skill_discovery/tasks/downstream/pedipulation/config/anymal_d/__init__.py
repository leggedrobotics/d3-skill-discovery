# Copyright (c) 2025, ETH Zurich, Rafael Cathomen
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from d3_skill_discovery.tasks.downstream.pedipulation.box_manipulation_base_env_cfg import BoxPoseEnvCfg
from d3_skill_discovery.tasks.downstream.pedipulation.box_manipulation_HL_env_cfg import BoxPoseHLEnvCfg
from d3_skill_discovery.tasks.downstream.pedipulation.foot_and_base_tracking_base_env_cfg import FootBaseTrackingEnvCfg
from d3_skill_discovery.tasks.downstream.pedipulation.foot_and_base_tracking_HL_env_cfg import FootBaseTrackingHLEnvCfg
from d3_skill_discovery.tasks.downstream.pedipulation.foot_tracking_base_env_cfg import FootTrackingEnvCfg
from d3_skill_discovery.tasks.downstream.pedipulation.foot_tracking_HL_env_cfg import FootTrackingHLEnvCfg

from . import agents

##
# Register Gym environments.
##

# foot tracking
gym.register(
    id="Isaac-FootTracking-Anymal-D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FootTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDFootTrackingPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-FootTracking-HL-Anymal-D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FootTrackingHLEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDFootTrackingPPORunnerCfg",
    },
)

# foot base tracking
gym.register(
    id="Isaac-FootBaseTracking-Anymal-D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FootBaseTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDBaseFootTrackingPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-FootBaseTracking-HL-Anymal-D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FootBaseTrackingHLEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDBaseFootTrackingPPORunnerCfg",
    },
)

# box pose tracking
gym.register(
    id="Isaac-BoxPoseTracking-Anymal-D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BoxPoseEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDBoxPosePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-BoxPoseTracking-HL-Anymal-D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BoxPoseHLEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDBoxPosePPORunnerCfg",
    },
)
