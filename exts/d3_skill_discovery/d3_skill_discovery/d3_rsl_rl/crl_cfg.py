# Copyright (c) 2025, ETH Zurich, Rafael Cathomen
#
# SPDX-License-Identifier: BSD-3-Clause


from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


##
# - Actor
##
@configclass
class RslRlGoalConditionedActorCfg:
    """Configuration for the Goal conditioned actor network. used for contrastive rl."""

    class_name: str = "GoalConditionedActorCritic"
    """The policy class name. Default is GoalConditionedActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""


##
# - Critic
##
@configclass
class RslRlContrastiveCriticCfg:
    """Configuration for the contrastive critic networks. used for contrastive rl."""

    class_name: str = "ContrastiveCritic"
    """The policy class name. Default is ContrastiveCritic."""

    representation_dim: int = MISSING
    """The dimension of the representation."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""


##
# - Algorithm
##
@configclass
class RslRlCrlAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "CRL"
    """The algorithm class name. Default is CRL."""

    mini_batch_size_and_num_inserts_per_sample: int = MISSING
    """The number of samples per learning step for the critic. This is equal to the batch size.
    IMPORTANT: This quantity has to be smaller than the number of environments."""
    # TODO should batch size be different from num inserts per sample?

    stack_N_critic_batches: int = MISSING
    """To increase the batch size, we can stack multiple contrastive batches. This is the number of batches to stack.
    The effective batch size is mini_batch_size_and_num_inserts_per_sample * stack_N_critic_batches."""

    actor_batch_size: int = MISSING
    """The number of samples per learning step for the actor."""

    replay_buffer_size_per_env: int = MISSING
    """The size of the replay buffer per env. Total size is replay_buffer_size_per_env * num_envs."""

    num_critic_learning_steps_per_update: int | float = MISSING
    """The number of critic learning steps per update step.
    Can be a fraction if less then one update per step is desired (e.g. 1/2 = one update every second step)
    This is necessary, because actor and critic may bigger or smaller updating rates than the env.
    Note, this gets multiplied by the number of update steps defined in the runner"""
    # TODO, do we even want a full epoch for the critic?

    num_actor_learning_steps_per_update: int | float = MISSING
    """The number of actor learning epochs per update step.
    Can be a fraction if less then one update per step is desired (e.g. 1/2 = one update every second step)
    This is necessary, because actor and critic may bigger or smaller updating rates than the env.
    Note, this gets multiplied by the number of update steps defined in the runner"""

    log_sum_exp_regularization_coef: float = 0.01
    """The regularization term used for critic learning."""

    info_nce_type: Literal["forward", "backward", "symmetric"] = "symmetric"

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss for actor learning.
    If target_entropy is set, this is the initial entropy coefficient."""

    tau: float = MISSING
    """The target network update rate. If 1.0, the target network is updated every step."""

    actor_learning_rate: float = MISSING
    """The learning rate for the policy."""

    critic_learning_rate: float = MISSING
    """The learning rate for the critic."""

    use_target_entropy: float | bool | None = None
    """If true, the entropy_coef is a learnable parameter and set to -0.5 * action dim.
    If desired, a float value can be set manually."""

    gamma: float = MISSING
    """The discount factor."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""


##
# - Runner
##
@configclass
class RslCRlOnPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of rollout steps per environment per update."""

    buffer_fill_steps: int = MISSING
    """The number of steps to fill the replay buffer before training."""

    num_learning_steps: int = MISSING
    """The number of learning steps per update."""

    update_actor_critic_simultaneously: bool = False
    """If true, the actor and critic are updated with the same data.
    This will ignore the num_critic_learning_steps_per_update and num_actor_learning_steps_per_update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    policy: RslRlGoalConditionedActorCfg = MISSING
    """The policy configuration."""

    critic: RslRlContrastiveCriticCfg = MISSING

    algorithm: RslRlCrlAlgorithmCfg = MISSING
    """The algorithm configuration."""

    ##
    # Checkpointing parameters
    ##

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    ##
    # Logging parameters
    ##

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    ##
    # Loading parameters
    ##

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """
