# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from unsupervised_RL.tasks.downstream.goal_tracking.goal_tracking_base_env_cfg import GoalReachingEnvCfg
from unsupervised_RL.tasks.downstream.goal_tracking.goal_tracking_HL_env_cfg import GoalReachingHLEnvCfg

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-GoalTracking-Anymal-D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": GoalReachingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDGoalTrackingPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-GoalTracking-HL-Anymal-D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": GoalReachingHLEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDGoalTrackingPPORunnerCfg",
    },
)
