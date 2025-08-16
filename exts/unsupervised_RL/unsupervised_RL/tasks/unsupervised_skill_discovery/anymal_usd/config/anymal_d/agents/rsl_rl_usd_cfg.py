from unsupervised_RL.rsl_rl.rl_cfg import (
    DiaynCfg,
    MetraCfg,
    RandomNetworkDistillationCfg,
    RslRlFactorizedUSDAlgorithmCfg,
    RslRlMetraAlgorithmCfg,
    RslRlPpoAlgorithmCfg,
    RslRlPpoRecurrentActorCriticCfg,
    RslRlPpoRelationalActorCriticCfg,
    UsdModelCfg,
)

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


##
# METRA
##
@configclass
class AnymalMetraPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 5432001
    num_steps_per_env = 25  # 24 + 1 extra
    num_transitions_per_episode = None
    max_iterations = 10_000_000  # 10_000_000
    save_interval = 1000
    experiment_name = "metra_factors_ppo_anymal_test"
    run_name = "metra_ppo_anymal_test"
    wandb_project = "usd_ppo_anymal"
    policy = RslRlPpoRelationalActorCriticCfg(
        init_noise_std=1.0,
        activation="elu",
        actor_layers=[512, 256, 256],
        critic_layers=[512, 256, 256],
        architecture="MLP",
    )
    algorithm = RslRlPpoAlgorithmCfg(
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
        symmetry_augmentation=True,
        symmetry_loss_weight=0.0,
        use_symmetry_aug_for_surrogate=True,
        force_reward_symmetry=False,
        beta_advantage_UCB=0.0,
    )
    usd = RslRlFactorizedUSDAlgorithmCfg(
        value_decomposition=True,
        metra_reward_scale=1.0,
        diayn_reward_scale=1.0,
        extrinsic_reward_scale=1.0,
        reward_normalization=True,
        factor_weight_lr=1.0,
        target_steps_to_goal=1000,
        regularize_factor_weights=0.1,
        adaptive_regularization_weight=False,
        randomize_factor_weights=True,
        factor_weights_skew=1.0,
        disable_style_reward=False,
        disable_regularization=False,
        disable_factor_weighting=False,
        metra=MetraCfg(
            lr=1e-4,
            lr_tau=5e-4,
            lambda_exploration=100.0,
            sigma=10.0,
            ensemble_size=1,
            norm_matching=False,
            state_representation_args=UsdModelCfg(
                hidden_layers=[256] * 2,
                activation="elu",
            ),
        ),
        diayn=DiaynCfg(
            lr=1e-4,
            lambda_exploration=1.0,
            lambda_skill_disentanglement=0.1,
            num_discriminators=1,
            discriminator_args=UsdModelCfg(
                hidden_layers=[256] * 2,  # 4
                activation="elu",
            ),
            skill_distribution_type="dirichlet",  # uniform_sphere, dirichlet  # TODO this is slow, find out why and make faster
            skill_disentanglement=True,
            symmetry_loss_weight=0.0,
            symmetry_zero_pad_skills=False,
            rnd_weight=1.0,
        ),
        rnd=RandomNetworkDistillationCfg(
            num_outputs=4,
            layers=[256] * 4,  # 4
            lr=1e-4,
        ),
        visualizer_interval=1000,
    )


@configclass
class AnymalHL_USD_PPORunnerCfg(AnymalMetraPPORunnerCfg):
    def __post_init__(self):
        self.algorithm.symmetry_augmentation = False
