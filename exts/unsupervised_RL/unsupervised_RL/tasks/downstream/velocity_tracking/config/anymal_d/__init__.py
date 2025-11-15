# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from unsupervised_RL.tasks.downstream.velocity_tracking.velocity_tracking_base_env_cfg import VelocityTrackingEnvCfg
from unsupervised_RL.tasks.downstream.velocity_tracking.velocity_tracking_HL_env_cfg import VelocityTrackingHLEnvCfg

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-VelocityTracking-Anymal-D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": VelocityTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDVelocityTrackingPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-VelocityTracking-HL-Anymal-D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": VelocityTrackingHLEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDVelocityTrackingPPORunnerCfg",
    },
)
