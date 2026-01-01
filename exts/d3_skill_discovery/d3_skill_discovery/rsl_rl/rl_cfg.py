# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

##
# - Actor-Critic
##


@configclass
class RslRlPpoRelationalActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "RelationalActorCriticTransformer"
    """The policy class name. Default is RelationalActorCriticTransformer."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    log_std_range: tuple[float, float] = (-5, 2)
    """The range of the log standard deviation for the policy."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""

    actor_layers: list[int] = []
    """The hidden dimensions of the actor network."""

    critic_layers: list[int] = []
    """The hidden dimensions of the critic network."""

    architecture: Literal["MLP", "Transformer", "SimBa", "BodyTransformer"] = "MLP"
    """The architecture of the actor."""


@configclass
class RslRlPpoRecurrentActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "RelationalActorCriticRecurrent"
    """The policy class name. Default is RelationalActorCriticRecurrent."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""


##
# - Algorithm
##
@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    symmetry_augmentation: bool = False
    """Whether to use symmetry augmentation."""

    symmetry_loss_weight: float = 0.0
    """The symmetry loss weight, only used if symmetry augmentation is enabled."""

    use_symmetry_aug_for_surrogate: bool = True
    """Whether to use symmetry augmentation for the surrogate loss."""

    force_reward_symmetry: bool = True
    """Whether to force skill discovery rewards to be symmetric."""

    beta_advantage_UCB: float = 0.0
    """The beta parameter for the advantage UCB algorithm."""


@configclass
class SAC_MetraCfg:
    gamma: float = MISSING
    alpha: float = MISSING
    polyak: float = MISSING
    lr: float = MISSING


@configclass
class UsdModelCfg:
    hidden_layers: list[int] = MISSING
    activation: str = "elu"
    layer_norm: bool = False


@configclass
class UsdAlgBaseCfg:
    """Base configuration for USD algorithms."""

    lr: float = MISSING
    """Learning rate for the algorithm."""
    lambda_exploration: float = MISSING
    """Exploration coefficient for the algorithm."""
    use_rnd: bool = False
    """Whether to use Random Network Distillation for exploration."""
    rnd_weight: float = 0.0
    """Scaling factor for the RND reward."""


@configclass
class DiaynCfg(UsdAlgBaseCfg):
    """Configuration for the DIAYN algorithm."""

    lambda_skill_disentanglement: float = MISSING
    """DUSDi lambda skill disentanglement coefficient."""
    num_discriminators: int = MISSING
    """For DISDAIN exploration"""
    skill_disentanglement: bool = False
    """Whether to use skill disentanglement DUSDI."""
    skill_distribution_type: str = "categorical"
    """The skill distribution type."""
    symmetry_loss_weight: float = 0.0
    """The symmetry loss weight, only used if symmetry augmentation is enabled."""
    symmetry_zero_pad_skills: bool = True
    """Whether to zero pad the skills for symmetry augmentation."""
    discriminator_args: UsdModelCfg = MISSING


@configclass
class MetraCfg(UsdAlgBaseCfg):
    """Configuration for the METRA algorithm."""

    lr_tau: float = MISSING
    """Learning rate for the dual variable."""

    # lambda_skill_disentanglement: float = MISSING
    ensemble_size: int = MISSING
    slack: float = 1e-3
    sigma: float = MISSING
    """Norm matching Metra reward coefficient."""
    skill_step_size: float = 0.17
    """Norm matching skill step size, as in the paper."""
    state_representation_args: UsdModelCfg = MISSING
    """Parameters for the state representation network."""
    norm_matching: bool = False
    """Whether to use the norm matching version."""


@configclass
class RandomNetworkDistillationCfg:
    """Configuration for the Random Network Distillation (RND) algorithm."""

    num_outputs: int = 128
    """The number of outputs for the RND networks."""

    layers: list[int] = MISSING
    """The hidden dimensions of the RND target and predictor network."""

    lr: float = MISSING
    """The learning rate for the RND algorithm."""

    perturb_target: bool = False
    """Whether to perturb the target network."""

    target_net_perturbation_scale: float = 0.001
    """The scale of the perturbation for the target network."""

    target_net_perturbation_interval: int = 10
    """The interval for perturbing the target network."""


@configclass
class RslRlMetraAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "METRA"
    """The algorithm class name. Default is METRA."""

    state_representation_args: UsdModelCfg = MISSING
    """The arguments for the state representation network.
    This depends on how the state representation network is implemented."""

    batch_size: int = MISSING
    """The batch size for the algorithm."""

    replay_buffer_size_per_env: int = MISSING
    """The replay buffer size per environment."""

    instructor_reward_scaling: bool = False
    """When an instructor net is available, set this to True. This will multiply the metra reward by the instructor reward.
    From the paper https://arxiv.org/abs/2406.00324"""

    replay_buffer_size_total: int = None
    """The replay buffer size total, will override the replay_buffer_size_per_env if provided."""

    replay_buffer_num_envs: int = None
    """The number of environments from which we store transitions in the replay buffer. If None, all environments are used.
    This is useful when we have a large number of environments and we want to store transitions from a subset of them."""

    num_metra_learning_epochs: int = MISSING
    """The number of epochs."""

    sigma: float = MISSING
    """Metra error to reward scaling factor."""

    num_sgd_steps_metra: int = MISSING
    """The number of SGD steps per epoch."""

    skill_dim: int = MISSING
    """The skill dimension."""

    lr: float = MISSING
    """The learning rate for the algorithm."""

    lr_tau: float = MISSING
    """The learning rate for the constrained."""

    skill_step_size: float = MISSING
    """The maximum skill step size. Skills have max norm of skill_step_size * Num_steps_per_episode."""

    visualizer_interval: int = MISSING
    """The visualizer interval."""

    sac_hyperparameters: SAC_MetraCfg = None
    """The hyperparameters for the SAC algorithm, if not provided, PPO is used."""

    max_info_exploration: bool = False
    """Whether to use maximum information exploration."""

    non_metra_reward_scale: float = 1.0
    """The reward scale for non-METRA rewards. All reward weights are multiplied by this value."""

    lambda_exploration: float = 1.0
    """Exploration coefficient for METRA ensemble."""

    init_info_max_coeff: float = 0.1
    """The initial coefficient for the maximum information exploration."""

    max_info_lr: float = 1e-3
    """The learning rate for the maximum information exploration."""

    ensemble_size: int = 5
    """The number of metra models."""

    metra_reward_scale: float = 1.0
    """The reward scale for METRA rewards"""

    diayn_reward_scale: float = 1.0
    """The reward scale for DIAYN rewards"""

    reward_normalization: bool = True
    """Whether to normalize the intrinsic rewards."""

    diayn: DiaynCfg = None


@configclass
class RslRlFactorizedUSDAlgorithmCfg:
    """Configuration for factorizes unsupervised skill discovery algorithm
    Algorithms with these configs provide a skill discovery reward to any RL algorithm (PPO)"""

    class_name: str = "FACTOR_USD"
    """The algorithm class name. Default is FACTOR_USD."""

    state_representation_args: UsdModelCfg = MISSING
    """The arguments for the state representation network.
    This depends on how the state representation network is implemented."""

    instructor_reward_scaling: bool = False
    """When an instructor net is available, set this to True. This will multiply the metra reward by the instructor reward.
    From the paper https://arxiv.org/abs/2406.00324"""

    sigma: float = MISSING
    """Metra error to reward scaling factor."""

    # skill_dim: int = MISSING
    # """The skill dimension."""

    value_decomposition: bool = False
    """Whether to use value decomposition (i.e., one value function per factor) for the USD algorithm."""

    visualizer_interval: int = MISSING
    """The visualizer interval."""

    metra_reward_scale: float = 1.0
    """The reward scale for METRA rewards"""

    diayn_reward_scale: float = 1.0
    """The reward scale for DIAYN rewards"""

    extrinsic_reward_scale: float = 1.0
    """The reward scale for extrinsic rewards"""

    reward_normalization: bool = True
    """Whether to normalize the intrinsic rewards."""

    factor_weight_lr: float = MISSING
    """The update rate for the factor weights."""

    randomize_factor_weights: bool = False
    """Whether to randomize the factor weights and learn them too."""

    factor_weights_skew: float = 1.0
    """The skew of the factor weights."""

    target_steps_to_goal: int = MISSING
    """The desired number of steps to reach the goal metric. Determines how to update the weights"""

    regularize_factor_weights: float = MISSING
    """Weight of the regularization reward"""

    adaptive_regularization_weight: bool = False
    """Whether to use adaptive regularization weight"""

    metra: MetraCfg = None
    """Configuration for the METRA factors."""

    diayn: DiaynCfg = None
    """Configuration for the DIAYN factors."""

    rnd: RandomNetworkDistillationCfg = None
    """Configuration for RND."""

    disable_style_reward: bool = False
    """Whether to disable the style reward."""

    disable_regularization: bool = False
    """Whether to disable the regularization reward."""

    disable_factor_weighting: bool = False
    """If true, all factor weights are set to the same value."""
