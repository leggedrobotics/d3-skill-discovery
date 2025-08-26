import torch
from abc import ABC, abstractmethod


class BaseSkillDiscovery(ABC):
    """Base class for skill discovery algorithms.
    Child classes are intended to train a skill conditioned policy pi(a|s, z) where z is the skill.
    """

    @abstractmethod
    def reward(self, usd_observations, skill: torch.Tensor, **kwargs) -> torch.Tensor:
        """Method to calculate the intrinsic reward for the underlying rl algorithm."""
        pass

    @abstractmethod
    def sample_skill(self, envs_to_sample: torch.Tensor, **kwargs) -> torch.Tensor:
        """Method to sample a skill z."""
        pass

    @abstractmethod
    def update(self, observation_batch, **kwargs) -> dict:
        """Method to update the intrinsic motivation algorithm."""
        pass

    @abstractmethod
    def get_save_dict(self) -> dict:
        """Method to get the save dict of the intrinsic motivation algorithm."""
        pass

    @abstractmethod
    def load(self, state_dict: dict, **kwargs) -> None:
        """Method to load the intrinsic motivation algorithm."""
        pass

    ##
    # - optional
    ##

    def visualize(self, *args, **kwargs):
        """Optional visualization method for debugging."""
        pass

    def update_skill_distribution(self, *args, **kwargs):
        """Optional method to update the skill distribution."""
        pass

    ##
    # - metrics
    ##

    @property
    def curriculum_metric(self) -> torch.Tensor:
        """Optional method to return a metric for the curriculum."""
        return None  # type: ignore

    @property
    @abstractmethod
    def performance_metric(self) -> float:
        """Returns a value between 0 and 1 that represents the performance of the skill discovery algorithm.
        0 is the worst and 1 is the best performance.
        This value might be used to change the weight of the intrinsic reward.
        """
        pass

    performance: torch.Tensor
    """Cumulated Performance metric of the skill discovery algorithm for all envs"""

    ##
    # - symmetry augmentation
    ##
    def symmetry_augmentation(self, skill: torch.Tensor) -> torch.Tensor:
        """Augments the skill by mirroring or rotating it.
        This needs to be implemented in accordance with the environment."""
        pass
