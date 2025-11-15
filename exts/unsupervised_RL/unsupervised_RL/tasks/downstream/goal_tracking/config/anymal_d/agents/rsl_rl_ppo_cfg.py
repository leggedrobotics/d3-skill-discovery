# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlPpoRelationalActorCriticCfg,
)

##
# Downstream PPO
##


@configclass
class AnymalDGoalTrackingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 554
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 500
    experiment_name = "anymal_goal_tracking_downstream"
    run_name = "anymal_goal_tracking_downstream"
    wandb_project = "anymal_goal_tracking_rough_downstream"
    empirical_normalization = False
    policy = RslRlPpoRelationalActorCriticCfg(
        init_noise_std=1.0,
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO_OG",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
