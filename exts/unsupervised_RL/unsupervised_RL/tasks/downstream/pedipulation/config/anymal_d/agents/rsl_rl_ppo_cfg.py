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
class AnymalDFootTrackingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 500
    experiment_name = "anymal_foot_tracking_downstream"
    run_name = "anymal_foot_tracking_downstream"
    wandb_project = "anymal_foot_tracking_downstream"
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


@configclass
class AnymalDBaseFootTrackingPPORunnerCfg(AnymalDFootTrackingPPORunnerCfg):
    experiment_name = "anymal_base_foot_tracking_downstream"
    run_name = "anymal_base_foot_tracking_downstream"
    wandb_project = "anymal_base_foot_tracking_downstream"
    max_iterations = 10000


@configclass
class AnymalDBoxPosePPORunnerCfg(AnymalDFootTrackingPPORunnerCfg):
    experiment_name = "anymal_box_pose_tracking_downstream"
    run_name = "anymal_box_pose_tracking_downstream"
    wandb_project = "anymal_box_pose_tracking_downstream"
    max_iterations = 10000
