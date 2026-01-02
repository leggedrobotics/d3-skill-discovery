# Copyright (c) 2025, ETH Zurich, Rafael Cathomen
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from abc import ABC, abstractmethod


class BaseSkillDiscovery(ABC):
    """Base class for unsupervised skill discovery (USD) algorithms.

    This abstract base class defines the interface that all USD algorithms must implement
    to be compatible with the D3 framework. Subclasses learn skill-conditioned policies
    π(a|s,z) where z is a skill vector sampled from a learned or predefined distribution.

    The USD algorithm provides intrinsic rewards that encourage the policy to learn
    diverse, distinguishable behaviors without explicit task rewards. This enables
    the discovery of a repertoire of skills that can later be composed or fine-tuned
    for downstream tasks.

    Key Responsibilities:
        - Define intrinsic reward function for skill-conditioned learning
        - Sample skills from a distribution p(z) to condition the policy
        - Update algorithm components (e.g., discriminators) using on-policy data
        - Manage checkpointing and state persistence
        - Track performance metrics for monitoring and curriculum learning

    Example Subclass Implementation:
        >>> class MyUSD(BaseSkillDiscovery):
        ...     def __init__(self, skill_dim, ...):
        ...         self.discriminator = Network(...)
        ...         self.skill_distribution = torch.distributions.Uniform(-1, 1)
        ...
        ...     def reward(self, obs, skill, **kwargs):
        ...         # Compute intrinsic reward based on skill and observation
        ...         log_q_z_s = self.discriminator.log_prob(obs, skill)
        ...         return log_q_z_s - self.skill_distribution.log_prob(skill)
        ...
        ...     def sample_skill(self, envs_to_sample, **kwargs):
        ...         num_envs = envs_to_sample.sum()
        ...         return self.skill_distribution.sample((num_envs,))
        ...
        ...     def update(self, observation_batch, **kwargs):
        ...         # Update discriminator or other learnable components
        ...         loss = ...
        ...         return {"loss": loss.item()}

    See Also:
        - DIAYN: Diversity-based skill discovery using a discriminator
        - METRA: Metric-aware skill discovery with representation learning
        - FACTOR_USD: Manages multiple USD algorithms for factorized learning
    """

    @abstractmethod
    def reward(self, usd_observations, skill: torch.Tensor, **kwargs) -> torch.Tensor:
        """Calculate the intrinsic reward for skill-conditioned learning.

        This method computes rewards that encourage the policy to exhibit behaviors
        that are distinguishable based on the provided skill. Typically combines
        terms for skill predictability and diversity.

        Args:
            usd_observations: Dictionary mapping observation keys to tensors of shape
                (num_envs, obs_dim). Should NOT contain the "skill" key itself.
            skill: Skill vectors of shape (num_envs, skill_dim) that condition the policy.
            **kwargs: Algorithm-specific additional inputs (e.g., complementary observations).

        Returns:
            Intrinsic rewards of shape (num_envs,) for each environment.

        Note:
            Rewards should be on the same scale as extrinsic task rewards for proper
            balancing. Consider using reward normalization or scaling factors.

        Example:
            >>> obs = {"joint_pos": torch.randn(2048, 12)}
            >>> skills = torch.randn(2048, 8)
            >>> rewards = usd_algorithm.reward(obs, skills)
            >>> print(rewards.shape)  # torch.Size([2048])
        """
        pass

    @abstractmethod
    def sample_skill(self, envs_to_sample: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample skills for environments that are resetting or need new skills.

        Args:
            envs_to_sample: Boolean tensor of shape (num_envs,) indicating which
                environments need new skills sampled (True = sample, False = skip).
            **kwargs: Algorithm-specific parameters (e.g., temperature for sampling).

        Returns:
            Sampled skills of shape (num_sampled_envs, skill_dim) where
            num_sampled_envs = envs_to_sample.sum().

        Note:
            - Skills should be sampled from the prior distribution p(z)
            - Some implementations may include deterministic skills for debugging
            - Return only skills for environments where envs_to_sample is True

        Example:
            >>> # Sample new skills for environments 0, 5, and 10
            >>> envs_to_sample = torch.tensor([True, False, False, False, False,
            ...                                 True, False, False, False, False, True])
            >>> skills = usd_algorithm.sample_skill(envs_to_sample)
            >>> print(skills.shape)  # torch.Size([3, skill_dim])
        """
        pass

    @abstractmethod
    def update(self, observation_batch, **kwargs) -> dict:
        """Update learnable components of the USD algorithm.

        Called after each PPO update to train discriminators, representation networks,
        or other learnable modules using the collected rollout data.

        Args:
            observation_batch: Dictionary of batched observations from rollout buffer.
                Shape: (num_steps * num_envs, obs_dim) for each observation type.
                Should include the "skill" key with associated skills.
            **kwargs: Algorithm-specific inputs (e.g., number of update epochs,
                complementary observations for disentanglement).

        Returns:
            Dictionary of metrics for logging (losses, accuracies, etc.).
            Keys should follow format: "AlgorithmName/metric_name".

        Example:
            >>> batch = {
            ...     "joint_pos": torch.randn(10000, 12),
            ...     "skill": torch.randn(10000, 8)
            ... }
            >>> metrics = usd_algorithm.update(batch, num_epochs=3)
            >>> print(metrics)
            {'DIAYN/discriminator_loss': 0.42, 'DIAYN/accuracy': 0.87}
        """
        pass

    @abstractmethod
    def get_save_dict(self) -> dict:
        """Get state dictionary for checkpointing.

        Returns:
            Dictionary containing all stateful components needed to resume training.
            Typically includes:
                - Model state dicts (discriminators, networks)
                - Optimizer state dicts
                - Hyperparameters (e.g., temperature, concentration)
                - Training statistics (if needed for curriculum)

        Example:
            >>> state = usd_algorithm.get_save_dict()
            >>> torch.save(state, "checkpoint.pt")
        """
        pass

    @abstractmethod
    def load(self, state_dict: dict, **kwargs) -> None:
        """Load state from checkpoint.

        Args:
            state_dict: Dictionary from get_save_dict() containing saved state.
            **kwargs: Optional flags like load_optimizer=False to skip optimizer state.

        Example:
            >>> state = torch.load("checkpoint.pt")
            >>> usd_algorithm.load(state, load_optimizer=True)
        """
        pass

    ##
    # - Optional Methods
    ##

    def visualize(self, *args, **kwargs):
        """Generate visualizations for debugging and analysis (optional).

        Subclasses can implement this to create plots showing:
        - Skill space coverage
        - Discriminator predictions vs. true skills
        - Observation clustering by skill
        - Trajectory visualization colored by skill

        Args:
            *args, **kwargs: Implementation-specific visualization parameters.

        Note:
            Default implementation does nothing. Override to add visualization.
        """
        pass

    def update_skill_distribution(self, *args, **kwargs):
        """Update the skill prior distribution p(z) (optional).

        Some algorithms adaptively adjust the skill distribution during training
        for curriculum learning or to focus on promising skill regions.

        Example use cases:
            - DIAYN with Dirichlet: Adjust concentration parameter based on
              discriminator accuracy to control skill diversity
            - Adaptive temperature: Increase/decrease entropy of skill distribution
            - Curriculum: Start with discrete skills, transition to continuous

        Args:
            *args, **kwargs: Implementation-specific parameters.

        Note:
            Default implementation does nothing. Override if your algorithm
            uses an adaptive skill distribution.
        """
        pass

    ##
    # - Metrics and Monitoring
    ##

    @property
    def curriculum_metric(self) -> torch.Tensor | None:
        """Return a per-environment metric for curriculum learning (optional).

        Returns:
            Tensor of shape (num_envs,) with curriculum progress metrics, or None.
            Higher values typically indicate environments ready for harder tasks.

        Example:
            For DIAYN, could return per-environment discriminator accuracy.
            For METRA, could return representation quality scores.

        Note:
            Used by curriculum managers to adjust difficulty per environment.
            Default returns None (no curriculum).
        """
        pass

    @property
    @abstractmethod
    def performance_metric(self) -> float:
        """Return overall performance metric in range [0, 1].

        This metric indicates how well the USD algorithm is performing:
            - 0.0: Worst performance (e.g., random chance accuracy)
            - 1.0: Perfect performance (e.g., 100% discriminator accuracy)

        Used for:
            - Monitoring training progress
            - Adjusting factor weights in factorized USD
            - Triggering curriculum updates
            - Early stopping criteria

        Returns:
            Performance value between 0 and 1.

        Example Implementations:
            - DIAYN: Normalized discriminator accuracy
            - METRA: Representation learning loss (inverted/normalized)
            - RND: Prediction error (inverted/normalized)

        Note:
            Should be invariant to batch size and averaged over recent history
            to avoid noise. Typically computed from a running buffer of metrics.
        """
        pass

    performance: torch.Tensor
    """Per-environment cumulative performance metric.

    Shape: (num_envs,)

    Tracks cumulative USD performance for each environment independently.
    Used for per-environment curriculum learning and monitoring. Updated
    during reward() computation and reset when environments reset.
    """

    ##
    # - Symmetry Augmentation
    ##
    def symmetry_augmentation(self, skill: torch.Tensor) -> torch.Tensor:
        """Apply symmetry transformations to skills for data augmentation.

        For robots with morphological symmetries (e.g., quadrupeds with left-right
        and front-back symmetry), augment skills by applying corresponding
        transformations. This improves sample efficiency and enforces that
        symmetric observations should correspond to symmetric skills.

        Args:
            skill: Original skills of shape (batch_size, skill_dim).

        Returns:
            Augmented skills of shape (batch_size * num_symmetries, skill_dim)
            where num_symmetries depends on the robot (typically 4 for quadrupeds:
            original, left-right flip, front-back flip, 180° rotation).

        Example:
            >>> # Original skills for 4-legged robot (one skill per leg: FL, FR, RL, RR)
            >>> skills = torch.tensor([[1.0, 0.5, -0.5, -1.0]])  # shape: (1, 4)
            >>>
            >>> augmented = usd.symmetry_augmentation(skills)
            >>> print(augmented.shape)  # torch.Size([4, 4])
            >>> # augmented[0]: original = [1.0, 0.5, -0.5, -1.0]
            >>> # augmented[1]: LR flip  = [0.5, 1.0, -1.0, -0.5]  (swap L<->R)
            >>> # augmented[2]: FB flip  = [-0.5, -1.0, 1.0, 0.5]  (swap F<->B)
            >>> # augmented[3]: 180 rot  = [-1.0, -0.5, 0.5, 1.0]  (both flips)

        Note:
            - Must be implemented consistently with environment observation symmetries
            - Typically only needed for algorithms using discrete/categorical skills
            - For continuous skills, may negate/permute sub-vectors instead
            - Default implementation does nothing (returns input unchanged)

        See Also:
            - PPO config: symmetry_augmentation flag
            - Environment: symmetry_augmented_obs() method
        """
        pass
