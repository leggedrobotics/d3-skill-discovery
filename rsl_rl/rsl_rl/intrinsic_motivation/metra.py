# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, deque
from typing import Generator, Literal

from rsl_rl.modules import DictFlattener, SimBa, StateRepresentation
from rsl_rl.utils import TIMER_CUMULATIVE, detach, mean_gradient_norm, to_device
from rsl_rl.utils.mirroring import remove_symmetry_subspaces

from .base_skill_discovery import BaseSkillDiscovery
from .max_info import MaxInfo


class METRA(BaseSkillDiscovery):
    """Implements online version of METRA [1]_, a norm matching version [2]_, optimistic exploration [3]_ and symmetry augmentation [4]_.
    This implementation is intended to be used along side PPO, as a factor by the Factorized USD algorithm.

    References:
        .. [1] Park et al. "METRA: Scalable Unsupervised RL with Metric-Aware Abstraction" arXiv preprint https://arxiv.org/abs/2310.08887 (2024)
        .. [2] Atanassov et al. "Constrained Skill Discovery: Quadruped Locomotion with Unsupervised Reinforcement Learning" arXiv preprint https://arxiv.org/abs/2410.07877 (2024)
        .. [3] Strouse et al. "Learning more skills through optimistic exploration" arXiv preprint https://arxiv.org/abs/2107.14226 (2021)
        .. [4] Mittal et al. "Symmetry Considerations for Learning Task Symmetric Robot Policies" arXiv preprint https://arxiv.org/abs/1802.06070 (2024)

    """

    def __init__(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        N_steps: int,  # episode length
        skill_dim: int,
        ensemble_size: int,
        sigma: float,
        num_envs: int,
        state_representation_args: dict,
        initial_lagrange_multiplier: float = 30.0,
        slack: float = 1e-5,
        lr=1e-3,
        lr_tau=1e-3,
        skill_step_size: float = 0.17,  # as in the paper, 0.17 is the average latent dist achieved by LSD
        device: str = "cpu",
        num_deterministic_skills: int = 0,
        lambda_exploration: float = 1.0,
        norm_matching: bool = False,
        max_grad_norm: float = 1.0,
        objective_switching_range: tuple[float, float] = (0.5, 0.7),
        batch_norm: bool = True,
        symmetry_zero_pad_skills: bool = False,
        **kwargs,
    ):
        """
        Args:
            obs (torch.Tensor | dict[str, torch.Tensor]): The observations from the environment.
            N_steps (int): The number of steps in the episode.
            skill_dim (int): The dimensionality of the skill.
            ensemble_size (int): The number of state representations in the ensemble.
            sigma (float): The exploration parameter.
            num_envs (int): The number of environments.
            state_representation_args (dict, optional): The arguments for the state representation. Defaults to {}.
            initial_lagrange_multiplier (float, optional): The initial lagrange multiplier. Defaults to 30.0.
            slack (float, optional): The slack variable. Defaults to 1e-5.
            lr (float, optional): The learning rate. Defaults to 1e-3.
            lr_tau (float, optional): The learning rate for the lagrange multiplier. Defaults to 1e-3.
            skill_step_size (float, optional): The step size for the skill. Defaults to 0.17.
            device (str, optional): The device to use. Defaults to "cpu".
            num_deterministic_skills (int, optional): The number of deterministic skills. Defaults to 0.
            lambda_exploration (float, optional): The exploration parameter. Defaults to 1.0.
            norm_matching (bool, optional): Whether to use norm matching. Defaults to False.
            max_grad_norm (float, optional): The maximum gradient norm. Defaults to 1.0.
            objective_switching_range (tuple[float, float], optional): The range for the objective switching. Defaults to (0.5, 0.7).
            batch_norm (bool, optional): Whether to use batch normalization. Defaults to True.
            symmetry_zero_pad_skills (bool, optional): Whether to use symmetry zero padding for the skills. Defaults to False.
            **kwargs: Additional keyword arguments.

        """

        if kwargs:
            print(
                "METRA.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()])
            )

        self.device = device
        num_augs = 4  # TODO: fix Hardcoded 4
        self.num_augs = num_augs

        if symmetry_zero_pad_skills:
            assert skill_dim % num_augs == 0, "Skill dimension must be divisible by num_augs."
            skill_dim = skill_dim // num_augs  # only 1 / num_augs -th is commanded, rest is zero
            obs = remove_symmetry_subspaces(obs)
        self.symmetry_zero_pad_skills = symmetry_zero_pad_skills

        # - METRA ensemble components

        self.state_representations = torch.nn.ModuleList(
            [
                StateRepresentation(obs=obs, latent_dim=skill_dim, **state_representation_args)
                for _ in range(ensemble_size)
            ]
        )
        # self.state_representations = torch.nn.ModuleList(
        #     [
        #         DictFlattener(
        #             SimBa(obs=torch.cat(list(obs.values()), dim=1), out_dim=skill_dim, **state_representation_args)
        #         )
        #         for _ in range(ensemble_size)
        #     ]
        # )

        self.state_representations.to(self.device)
        self.log_lagrange_multipliers = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.tensor(math.log(initial_lagrange_multiplier), device=self.device))
                for _ in range(ensemble_size)
            ]
        )
        self.log_lagrange_multipliers.to(self.device)
        self.state_representation_optimizers = [
            optim.Adam(state_representation.parameters(), lr=lr) for state_representation in self.state_representations
        ]
        self.lagrange_multiplier_optimizers = [
            optim.Adam([log_lagrange_multiplier], lr=lr_tau)
            for log_lagrange_multiplier in self.log_lagrange_multipliers
        ]
        self.N_steps = N_steps

        # - METRA hyperparameters
        self.skill_dim = skill_dim
        self.sigma = sigma
        self.slack = slack
        self.lambda_exploration = lambda_exploration
        self.norm_matching = norm_matching
        self.max_grad_norm = max_grad_norm
        self.objective_switching_range = objective_switching_range
        self.alpha_alignment_objective = 1.0
        # self._performance_metric_target = performance_metric_target
        self.batch_norm = batch_norm

        self.previous_state: dict[str, torch.Tensor] = None  # type: ignore
        # - logging & visualization
        self.skill_step_size = skill_step_size
        self.num_deterministic_skills = num_deterministic_skills
        self.sgd_step_counter = 0
        self.N_envs_to_visualize = 30
        self.viz_skill_buffer = [[] for _ in range(self.N_envs_to_visualize)]
        self.viz_skill_full = deque(maxlen=self.N_envs_to_visualize)
        self.viz_obs_buffer = [defaultdict(list) for _ in range(self.N_envs_to_visualize)]
        self.viz_obs = deque(maxlen=self.N_envs_to_visualize)

        self.logging_skill_reward = deque(maxlen=100)
        self.logging_alignment_reward = deque(maxlen=100)
        self.logging_norm_matching_reward = deque(maxlen=100)
        self.logging_phi_diff = deque(maxlen=100)
        self.logging_cosine_similarity = deque(maxlen=100)
        self.logging_metra_disagreement_reward = deque(maxlen=100)
        self.logging_infomax_reward = deque(maxlen=100)
        self.logging_error = deque(maxlen=100)
        self.logging_angle = deque(maxlen=100)

        self.performance = torch.zeros(num_envs, device=self.device)
        self._deterministic_skills = self._sample_deterministic_skills(num_deterministic_skills)

        # - for curriculum
        self.curr_metric = [torch.zeros(num_envs, device=self.device), torch.zeros(num_envs, device=self.device)]
        self.num_envs = num_envs

    @property
    def deterministic_skills(self):
        # add some noise to the deterministic skills such they are not exactly the same
        return self._deterministic_skills + (torch.rand_like(self._deterministic_skills) - 0.5) * 1e-4

    @property
    def curriculum_metric(self):
        mean = self.curr_metric[0] / torch.clamp(self.curr_metric[1], min=1)
        return torch.cos(mean)

    @property
    def performance_metric(self):
        return torch.tensor(self.logging_cosine_similarity).mean().item()

    def save_previous_state(self, state: torch.Tensor | dict[str, torch.Tensor]):
        if isinstance(state, dict):
            for key in state.keys():
                self.previous_state[key].copy_(state[key])
        else:
            self.previous_state.copy_(state)

    def test_mode(self):
        self.state_representations.test()

    def train_mode(self):
        self.state_representations.train()

    def save_traj_for_visualization(self, obs: dict[str, torch.Tensor], skill: torch.Tensor, dones: torch.Tensor):
        """For metra, we need to save the trajectory for visualization."""
        # add to buffer
        for i in range(self.N_envs_to_visualize):
            env_id = i
            for k, v in obs.items():
                # move to cpu to not fill up gpu memory
                self.viz_obs_buffer[i][k].append(v[env_id].cpu())
            self.viz_skill_buffer[i].append(skill[env_id].cpu())

            if dones[env_id]:
                self.viz_obs.append({k: torch.stack(v, dim=0) for k, v in self.viz_obs_buffer[i].items()})
                self.viz_obs_buffer[i] = defaultdict(list)
                self.viz_skill_full.append(torch.stack(self.viz_skill_buffer[i], dim=0))
                self.viz_skill_buffer[i] = []

    def reward(
        self,
        usd_observations: dict[str, torch.Tensor],
        skill: torch.Tensor,
        done: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Method to calculate the intrinsic METRA reward for the underlying rl algorithm.
        The reward is either alignment based or norm matching based. Additionally, an ensemble disagreement term is added to the reward.

        Args:
            usd_observations (dict[str, torch.Tensor]): The observations from the environment, must NOT contain the key "skill".
            skill (torch.Tensor): The skill used to calculate the intrinsic reward.
            done (torch.Tensor): The done flag indicates where a new episode has started.

        Returns:
            torch.Tensor: The calculated intrinsic METRA reward.
        """

        # - split skill from observations
        skill = skill.clone()
        observations = {k: v.clone() for k, v in usd_observations.items() if k != "skill"}

        if self.symmetry_zero_pad_skills:
            # remove zeros
            skill = skill[..., : self.skill_dim]
            # we can also remove 1/num_augs -th of the state. this another indurive bias
            # enforcing which sub-skill controls which symmetry
            observations = remove_symmetry_subspaces(usd_observations)

        if self.previous_state is None:  # first step
            self.previous_state = observations
            return torch.zeros(done.shape, device=self.device)

        # safe for visualization
        self.save_traj_for_visualization(observations, skill, done)

        # - Calculate the METRA rewards
        alignment_rewards = []
        norm_matching_rewards = []
        for state_representation in self.state_representations:
            old_phi = state_representation(self.previous_state)
            next_phi = state_representation(observations)
            phi_diff = next_phi - old_phi

            # norm matching
            error = torch.sum(torch.square(phi_diff - skill), dim=-1)
            usd_reward_norm_matching_ = 1 / (1 + self.sigma * error)

            # alignment
            usd_reward_alignment_ = torch.linalg.vecdot(phi_diff, skill)

            alignment_rewards.append(usd_reward_alignment_)
            norm_matching_rewards.append(usd_reward_norm_matching_)

        usd_reward_alignment = torch.stack(alignment_rewards, dim=0)
        usd_reward_norm_matching = torch.stack(norm_matching_rewards, dim=0)

        # log rewards
        self.logging_alignment_reward.append(usd_reward_alignment.mean().item())
        self.logging_norm_matching_reward.append(usd_reward_norm_matching.mean().item())

        # - Normalize rewards
        if self.batch_norm:
            usd_reward_alignment = (usd_reward_alignment - usd_reward_alignment.mean()) / (
                usd_reward_alignment.std() + 1e-6
            )
            usd_reward_norm_matching = (usd_reward_norm_matching - usd_reward_norm_matching.mean()) / (
                usd_reward_norm_matching.std() + 1e-6
            )

        # - Combine rewards
        ensemble_reward = (
            self.alpha_alignment_objective * usd_reward_alignment
            + (1 - self.alpha_alignment_objective) * usd_reward_norm_matching
        )
        self.save_previous_state(observations)
        skill_reward = ensemble_reward.mean(dim=0)
        disagreement = ensemble_reward.var(dim=0) if len(ensemble_reward) > 1 else torch.zeros_like(skill_reward)
        reward = skill_reward + disagreement * self.lambda_exploration
        reward[done.bool()] = 0  # (done = true) = previous state is terminal, new state is start of new episode

        # more logging
        self.logging_skill_reward.append(torch.clamp(reward, min=-1.0, max=1.0).mean())
        self.logging_metra_disagreement_reward.append(disagreement.mean())
        self.logging_error.append(torch.sqrt(error).mean())
        self.logging_phi_diff.append(phi_diff.norm(dim=-1).mean())
        # error_norm = torch.linalg.vector_norm(phi_diff - skill, dim=-1)
        projection = torch.linalg.vecdot(phi_diff, skill) / torch.clamp(
            phi_diff.norm(dim=1) * skill.norm(dim=1), min=1e-7
        )
        angle = torch.acos(torch.clamp(projection, min=-1.0, max=1.0))
        self.logging_angle.append(angle.mean())
        cosine_similarity = torch.nn.functional.cosine_similarity(phi_diff, skill, dim=-1)
        self.performance += cosine_similarity
        self.logging_cosine_similarity.append(cosine_similarity.mean())
        self.curr_metric[0] += angle[: self.num_envs]
        self.curr_metric[1] += 1
        self.curr_metric[0][done[: self.num_envs].bool()] = 0
        self.curr_metric[1][done[: self.num_envs].bool()] = 0

        return torch.clamp(reward, min=-1000.0, max=1000.0)

    def _sample_deterministic_skills(self, num_envs: int) -> torch.Tensor:
        """Creates a one-hot encoding for the deterministic skills."""
        # one hot encoding
        skills = torch.eye(self.skill_dim, device=self.device)  # * radius
        # deterministic_skills = torch.cat(
        #     [skills, skills * 2 / 3, skills * 1 / 3, skills * 0, -skills * 1 / 3, -skills * 2 / 3, -skills], dim=0
        # )
        deterministic_skills = torch.cat([skills, -skills], dim=0)
        repeates = int(num_envs // deterministic_skills.shape[0]) + 1
        if self.norm_matching:
            for i in range(repeates):
                deterministic_skills = torch.cat(
                    [deterministic_skills, skills / (2 ** (1 + i)), -skills / (2 ** (1 + i))], dim=0
                )
        else:
            deterministic_skills = torch.cat([deterministic_skills] * repeates, dim=0)
        return deterministic_skills[:num_envs]

    def sample_skill(
        self,
        envs_to_sample: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Samples from a multi dimensional unit normal distribution.
        Or sample from the unit hypersphere.
        Args:
            envs_to_sample (torch.Tensor): The environments to sample from.
            area_uniform (bool, optional): Whether to account for the volume of the hypersphere. If true, skills with large norms
            are sampled more frequently. Defaults to False.

        """
        # radius = self.N_steps * self.skill_step_size
        radius = 1.5
        num_envs = int(envs_to_sample.sum().item())
        # - random skills
        skills = torch.randn(num_envs, self.skill_dim, device=self.device)
        skills = skills / skills.norm(dim=-1, keepdim=True)

        # - alignment and norm matching skills differ in radius distribution
        # We sample a proportion (same as for the reward) of skills from the alignment and norm matching distribution
        norm_matching_skills = torch.rand(num_envs, 1, device=self.device) > self.alpha_alignment_objective
        radii_alignment = torch.ones(num_envs, 1, device=self.device)
        radii_norm_matching = torch.rand(num_envs, 1, device=self.device) * radius
        radii = norm_matching_skills * radii_norm_matching + (~norm_matching_skills) * radii_alignment
        skills = skills * radii

        # - deterministic skills
        if envs_to_sample[: self.num_deterministic_skills].any():
            deterministic_skills = self.deterministic_skills[envs_to_sample[: self.num_deterministic_skills].to(bool)]
            # - combine
            skills[: deterministic_skills.shape[0]] = deterministic_skills

        if self.symmetry_zero_pad_skills:
            # we only command 1 / num_augs -th of the skills space the rest is zero padded here
            skills = torch.cat([skills, torch.zeros_like(skills).repeat(1, self.num_augs - 1)], dim=1)

        return skills

    def update(
        self,
        observation_batch: dict[str, torch.Tensor],
        next_observation_batch: dict[str, torch.Tensor],
        num_augs: int = 1,
        **kwargs,
    ) -> dict[str, float]:
        """Update the state representation network and the lagrange multiplier.
        The loss is a weighted sum of the norm matching loss and the alignment loss.
        """

        skill = observation_batch["skill"]
        next_skill = next_observation_batch["skill"]
        same_skill = (skill == next_skill).all(dim=-1)

        current_obs = {k: v[same_skill] for k, v in observation_batch.items() if k != "skill"}
        next_obs = {k: v[same_skill] for k, v in next_observation_batch.items() if k != "skill"}
        skill = skill[same_skill]

        if self.symmetry_zero_pad_skills:
            # remove symmetry augmentation and zero padding, this effectively shields the discriminator from the symmetry augmentation
            skill = skill.chunk(num_augs, dim=0)[0][:, : self.skill_dim]
            current_obs = remove_symmetry_subspaces({k: v.chunk(num_augs, dim=0)[0] for k, v in current_obs.items()})
            next_obs = remove_symmetry_subspaces({k: v.chunk(num_augs, dim=0)[0] for k, v in next_obs.items()})
        self.sgd_step_counter += 1

        for (
            state_representation_optimizer,
            lagrange_multiplier_optimizer,
            state_representation,
            log_lagrange_multiplier,
        ) in zip(
            self.state_representation_optimizers,
            self.lagrange_multiplier_optimizers,
            self.state_representations,
            self.log_lagrange_multipliers,
        ):

            # - Embed the current and next observation
            phi_cur = state_representation(current_obs)
            phi_next = state_representation(next_obs)
            phi_diff = phi_next - phi_cur  # = target_z

            # - Compute metra losses
            # norm matching:
            # metra_loss = torch.sum(torch.square(self.N_steps * phi_diff - skill), dim=-1)
            # metra_loss = torch.nn.functional.mse_loss(self.N_steps * phi_diff, skill, reduction="none").sum(dim=-1)
            metra_loss_norm_matching = torch.nn.functional.smooth_l1_loss(phi_diff, skill, reduction="none").sum(dim=-1)
            # alignment:
            metra_loss_alignment = -torch.linalg.vecdot(phi_diff, skill)
            metra_loss = (
                self.alpha_alignment_objective * metra_loss_alignment
                + (1 - self.alpha_alignment_objective) * metra_loss_norm_matching
            )

            # - update trajectory embedding loss
            # constraint
            cst_objective = 1 - torch.linalg.vector_norm(phi_diff, dim=1)  # l2 norm
            cst_objective = torch.clamp(cst_objective, max=self.slack)
            loss_te = metra_loss - (1 + log_lagrange_multiplier.exp().detach()) * cst_objective
            loss_te = loss_te.mean()

            # - sgd state representation
            state_representation_optimizer.zero_grad()
            loss_te.backward()
            mean_state_repr_grad_norm = mean_gradient_norm(state_representation)
            torch.nn.utils.clip_grad_value_(state_representation.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(state_representation.parameters(), self.max_grad_norm)
            state_representation_optimizer.step()

            # - update constraint regularization
            loss_lag = log_lagrange_multiplier * (cst_objective.detach()).mean()
            # - sgd lagrange multiplier
            lagrange_multiplier_optimizer.zero_grad()
            loss_lag.backward()
            # torch.nn.utils.clip_grad_norm_(log_lagrange_multiplier, self.max_grad_norm)
            lagrange_multiplier_optimizer.step()

        # TIMER_CUMULATIVE.stop("metra_update")

        # - update objective weight
        mean_cosine_similarity = torch.tensor(self.logging_cosine_similarity).mean().item()
        if mean_cosine_similarity <= self.objective_switching_range[0]:
            self.alpha_alignment_objective = 1.0
        elif mean_cosine_similarity > self.objective_switching_range[1]:
            self.alpha_alignment_objective = 0.0
        else:
            self.alpha_alignment_objective = 1 - (mean_cosine_similarity - self.objective_switching_range[0]) / (
                self.objective_switching_range[1] - self.objective_switching_range[0]
            )

        # - metrics
        metrics = {
            "Loss/loss_te": loss_te.item(),
            "Loss/loss_norm_matching": metra_loss_norm_matching.mean().item(),
            "Loss/loss_alignment": metra_loss_alignment.mean().item(),
            "Metra/reward_usd_mean": torch.tensor(self.logging_skill_reward).mean().item(),
            "Metra/reward_alignment_mean": torch.tensor(self.logging_alignment_reward).mean().item(),
            "Metra/reward_norm_matching_mean": torch.tensor(self.logging_norm_matching_reward).mean().item(),
            "Metra/reward_exploration_mean": torch.tensor(self.logging_metra_disagreement_reward).mean().item()
            * self.lambda_exploration,
            "Metra/cosine_similarity_mean": torch.tensor(self.logging_cosine_similarity).mean().item(),
            "Metra/phi_grad_norm": mean_state_repr_grad_norm,
            "Metra/error_mean": torch.tensor(self.logging_error).mean().item(),
            "Metra/angle_mean": torch.tensor(self.logging_angle).mean().rad2deg().item(),
            "Metra/constraint_mean": cst_objective.mean().item(),
            "Metra/lagrange_multiplier": self.log_lagrange_multipliers[0].exp().item(),
            "Metra/diffs_norm_mean": torch.tensor(self.logging_phi_diff).mean().item(),
            "Loss/metra_loss_lag_mean": loss_lag.item(),
            "Metra/total_sgd_steps": self.sgd_step_counter,
            "Metra/phi_mean_norm": phi_cur.norm(dim=1).mean().item(),
            "Metra/batch_size": len(skill),
            "Metra/alpha_alignment_objective": self.alpha_alignment_objective,
        }

        return metrics

    def symmetry_augmentation_new_but_worse(self, skill: torch.Tensor) -> torch.Tensor:
        """Mirrors the skill.
        Currently mirrors in 3 (left/right, front/back, 180deg rot) ways, resulting.
        The symmetry is implemented in a geometric way, i.e. the skill is mirrored in the skill space."""

        if self.symmetry_zero_pad_skills:
            # if zero padding, we simply permute the sub-skills
            assert (
                skill.shape[-1] % 4 == 0
            ), "Skill dimension must be a multiple of 4 for symmetry augmentation with zero padding."
            skill_quarters = skill.chunk(4, dim=1)
            lr_skill = torch.cat([skill_quarters[1], skill_quarters[0], skill_quarters[3], skill_quarters[2]], dim=-1)
            fb_skill = torch.cat([skill_quarters[2], skill_quarters[3], skill_quarters[0], skill_quarters[1]], dim=-1)
            rot_skill = torch.cat([skill_quarters[3], skill_quarters[2], skill_quarters[1], skill_quarters[0]], dim=-1)
        else:
            lr_skill = -skill.clone()

            idx = torch.arange(self.skill_dim).to(skill.device)
            shifted_idx = (idx - (self.skill_dim) // 2) % self.skill_dim

            fb_skill = skill[:, shifted_idx].clone()
            rot_skill = -fb_skill.clone()

        return torch.cat([skill, lr_skill, fb_skill, rot_skill], dim=0)

    def symmetry_augmentation(self, skill: torch.Tensor) -> torch.Tensor:
        """Mirrors the skill.
        Currently mirrors in 3 (left/right, front/back, 180deg rot) ways, resulting.
        The symmetry is implemented in a geometric way, i.e. the skill is mirrored in the skill space."""

        if self.symmetry_zero_pad_skills:
            # if zero padding, we simply permute the sub-skills
            assert (
                skill.shape[-1] % 4 == 0
            ), "Skill dimension must be a multiple of 4 for symmetry augmentation with zero padding."
            skill_quarters = skill.chunk(4, dim=1)
            lr_skill = torch.cat([skill_quarters[1], skill_quarters[0], skill_quarters[3], skill_quarters[2]], dim=-1)
            fb_skill = torch.cat([skill_quarters[2], skill_quarters[3], skill_quarters[0], skill_quarters[1]], dim=-1)
            rot_skill = torch.cat([skill_quarters[3], skill_quarters[2], skill_quarters[1], skill_quarters[0]], dim=-1)
        elif self.skill_dim >= 4:
            third_dim = self.skill_dim // 3
            lr_skill = skill.clone()
            lr_skill[:, third_dim : 2 * third_dim] = -lr_skill[:, third_dim : 2 * third_dim]
            lr_skill[:, 2 * third_dim :] = -lr_skill[:, 2 * third_dim :]

            fb_skill = skill.clone()
            fb_skill[:, :third_dim] = -fb_skill[:, :third_dim]
            fb_skill[:, 2 * third_dim :] = -fb_skill[:, 2 * third_dim :]

            rot_skill = skill.clone()
            rot_skill[:, : 2 * third_dim] = -rot_skill[:, : 2 * third_dim]

        elif self.skill_dim == 3:
            lr_skill = skill.clone()
            lr_skill[:, 1] = -lr_skill[:, 1]
            lr_skill[:, 2] = -lr_skill[:, 2]

            fb_skill = skill.clone()
            fb_skill[:, 0] = -fb_skill[:, 0]
            fb_skill[:, 2] = -fb_skill[:, 2]

            rot_skill = skill.clone()
            rot_skill[:, 0] = -rot_skill[:, 0]
            rot_skill[:, 1] = -rot_skill[:, 1]
        elif self.skill_dim == 1:
            # if skill_dim == 1, we can only mirror in one direction
            lr_skill = -skill.clone()
            fb_skill = -skill.clone()
            rot_skill = skill.clone()

        else:
            lr_skill = skill.clone()
            lr_skill[:, 1] = -lr_skill[:, 1]

            fb_skill = skill.clone()
            fb_skill[:, 0] = -fb_skill[:, 0]

            rot_skill = skill.clone()
            rot_skill[:, 0] = -rot_skill[:, 0]
            rot_skill[:, 1] = -rot_skill[:, 1]
        return torch.cat([skill, lr_skill, fb_skill, rot_skill], dim=0)

    def get_save_dict(self) -> dict:
        return {
            "model_state_dict": self.state_representations.state_dict(),
            "log_lagrange_multiplier_state_dict": self.log_lagrange_multipliers.state_dict(),
            "alpha_alignment_objective": self.alpha_alignment_objective,
            "optimizer_state_dict": [optimizer.state_dict() for optimizer in self.state_representation_optimizers],
            "lagrange_multiplier_optimizer_state_dict": [
                optimizer.state_dict() for optimizer in self.lagrange_multiplier_optimizers
            ],
        }

    def load(self, state_dict: dict, load_optimizer: bool = True) -> None:
        """Load the state of the model from a dictionary.
        Args:
            state_dict (dict): The state dictionary to load from.
            load_optimizer (bool, optional): Whether to load the optimizer state. Defaults to True.
        """
        self.state_representations.load_state_dict(state_dict["model_state_dict"])
        self.log_lagrange_multipliers.load_state_dict(state_dict["log_lagrange_multiplier_state_dict"])
        self.alpha_alignment_objective = state_dict["alpha_alignment_objective"]

        if load_optimizer:
            for optimizer, state in zip(self.state_representation_optimizers, state_dict["optimizer_state_dict"]):
                optimizer.load_state_dict(state)
            for optimizer, state in zip(
                self.lagrange_multiplier_optimizers, state_dict["lagrange_multiplier_optimizer_state_dict"]
            ):
                optimizer.load_state_dict(state)

    def visualize(
        self,
        save_path: str,
        file_name: str,
        factor_name: str = "",
        save_ensemble_plots: bool = False,
        max_lines: int = 50,
    ) -> str:
        """Plots the trajectories of the environment in the state space.
        Colors the trajectories according to the skill used.
        Saves the plot to the specified path."""

        try:

            def get_2d_projection(data):
                mean = data.mean(dim=0, keepdim=True)
                centered_data = data - mean
                covariance = torch.matmul(centered_data.T, centered_data) / (data.size(0) - 1)
                eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
                top2_eigenvectors = eigenvectors[:, -2:]  # Last two columns correspond to top eigenvalues
                del data, mean, centered_data, covariance, eigenvalues, eigenvectors
                return top2_eigenvectors

            def project_to_plane(data, projection_matrix):
                return torch.matmul(data, projection_matrix)

            import matplotlib
            import matplotlib.pyplot as plt
            import os

            matplotlib.use("Agg")

            # TODO figure out a good way to visualize the ensemble
            # - get phis
            with torch.no_grad():
                phis = [self.state_representations[0](to_device(obs, self.device)) for obs in self.viz_obs]
                phis_ensemble = [
                    [state_representations(to_device(obs, self.device)) for obs in self.viz_obs]
                    for state_representations in self.state_representations
                ]

            line_count = 0

            # pca if skill_dim > 2
            if self.skill_dim > 2:
                projection_matrix = get_2d_projection(torch.concat(phis))
            else:
                projection_matrix = torch.eye(2).to(self.device)

            colormap = plt.cm.hsv
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            fig.suptitle(f"METRA factor: {factor_name}")

            # plt_size = 20
            # axs[0].set_xlim(-plt_size, plt_size)
            # axs[0].set_ylim(-plt_size, plt_size)
            axs[0][0].set_aspect("equal")
            axs[1][0].set_aspect("equal")
            axs[0][1].set_aspect("equal")
            axs[1][1].set_aspect("equal")

            ensemble_fix, ensemble_axs = plt.subplots(2, len(self.state_representations) // 2 + 1, figsize=(18, 6))
            ensemble_fix.suptitle(f"METRA ensemble, factor: {factor_name}")
            for l in range(2):
                if len(self.state_representations) // 2 + 1 == 1:
                    ensemble_axs[l] = [ensemble_axs[l]]
                for c in range(len(self.state_representations) // 2 + 1):
                    ensemble_axs[l][c].set_aspect("equal")

            for i, (obs_traj, phi_traj, skill_traj) in enumerate(zip(self.viz_obs, phis, self.viz_skill_full)):

                obs_traj = to_device(obs_traj, self.device)
                phi_traj = phi_traj.to(self.device)
                skill_traj = skill_traj.to(self.device)

                unique_skills = skill_traj.unique(dim=0)
                for skill in unique_skills:
                    line_count += 1
                    if line_count > max_lines:
                        break
                    skill_mask = (skill_traj == skill).all(dim=-1)

                    skill_proj = (
                        project_to_plane(skill_traj[skill_mask][:1], projection_matrix).squeeze() * self.N_steps
                    )

                    phis_2d = project_to_plane(phi_traj[skill_mask], projection_matrix).detach().cpu().numpy()

                    if self.skill_dim > 2:
                        extra_str = f" (2d projection from {self.skill_dim}d)"
                    else:
                        extra_str = ""
                    skill_angle = (torch.atan2(skill_proj[0], skill_proj[1]) + math.pi) / (2 * math.pi)
                    color = colormap(skill_angle.item())

                    # get trajectory
                    if "my_pose" in obs_traj:
                        pos_traj = obs_traj["my_pose"][skill_mask][:, :2].cpu().numpy()
                    elif "box_pose" in obs_traj:
                        pos_traj = obs_traj["box_pose"][skill_mask][:, :2].cpu().numpy()
                    elif "origin" in obs_traj:
                        pos_traj = obs_traj["origin"][skill_mask][:, :2].cpu().numpy()
                    elif "foot_pos" in obs_traj:
                        pos_traj = obs_traj["foot_pos"][skill_mask][:, [0, 2]].cpu().numpy()
                    elif torch.tensor(["bbox" in k for k in obs_traj.keys()]).any():
                        bbox_key = [k for k in obs_traj.keys() if "bbox" in k][0]
                        pos_traj = obs_traj[bbox_key][skill_mask][:, :2].cpu().numpy()
                    else:
                        pos_traj = None

                    # plot
                    axs[0][1].plot(phis_2d[:, 0], phis_2d[:, 1], color=color, lw=1, alpha=0.5, zorder=1)
                    axs[0][1].scatter(phis_2d[:, 0], phis_2d[:, 1], color=color, s=2, alpha=0.5, zorder=2)
                    axs[0][1].set_title("Phi trajectories" + extra_str)

                    axs[1][0].plot(phis_2d[:, 0], phis_2d[:, 1], color=color, lw=1, alpha=0.5, zorder=1)
                    axs[1][0].scatter(phis_2d[:, 0], phis_2d[:, 1], color=color, s=2, alpha=0.5, zorder=2)
                    axs[1][0].set_title("Representation space and skills" + extra_str)
                    # axs[1].annotate("", xy=skill.cpu().numpy(), xytext=(0, 0),
                    #     arrowprops=dict(arrowstyle="->", lw=2, color=))
                    x, y = skill_proj.cpu().numpy()
                    axs[1][0].arrow(
                        0,
                        0,
                        x,
                        y,
                        head_width=skill_proj.norm().item() / 10,
                        head_length=skill_proj.norm().item() / 5,
                        fc=color,
                        ec=color,
                        zorder=3,
                    )
                    last_phi = phis_2d[-1]
                    axs[0][1].scatter(
                        last_phi[0], last_phi[1], edgecolor="k", color=color, s=100, marker="X", linewidth=2, zorder=4
                    )
                    axs[1][0].scatter(
                        last_phi[0], last_phi[1], edgecolor="k", color=color, s=100, marker="X", linewidth=2, zorder=4
                    )

                    # data specific to the environment
                    if pos_traj is not None:
                        axs[0][0].plot(pos_traj[:, 0], pos_traj[:, 1], color=color, lw=1, alpha=0.5, zorder=1)
                        axs[0][0].scatter(pos_traj[:, 0], pos_traj[:, 1], color=color, s=2, alpha=0.5, zorder=2)

                        last_state = pos_traj[-1]
                        axs[0][0].scatter(
                            last_state[0],
                            last_state[1],
                            edgecolor="k",
                            color=color,
                            s=100,
                            marker="X",
                            linewidth=2,
                            zorder=4,
                        )

                    axs[0][0].set_title("State space")

                # phi ensemble
                projected_ensemble_phis = []
                for j, individual_phi in enumerate(phis_ensemble):

                    phi_to_plot = individual_phi[i]
                    plot_r = j // 2
                    plot_c = j % 2
                    projected_phis = project_to_plane(phi_to_plot, projection_matrix).detach()
                    projected_ensemble_phis.append(projected_phis)
                    projected_phis = projected_phis.cpu().numpy()

                    ensemble_axs[plot_c][plot_r].plot(
                        projected_phis[:, 0],
                        projected_phis[:, 1],
                        color=color,
                        lw=1,
                        alpha=0.5,
                        zorder=1,
                    )
                    ensemble_axs[plot_c][plot_r].scatter(
                        projected_phis[:, 0],
                        projected_phis[:, 1],
                        color=color,
                        s=2,
                        alpha=0.5,
                        zorder=2,
                    )
                    ensemble_axs[plot_c][plot_r].set_title(f"Ensemble {j} Phi traj")
                for remove_idx in range(j + 1, len(ensemble_axs) * len(ensemble_axs[0])):
                    plot_r = remove_idx // 2
                    plot_c = remove_idx % 2
                    ensemble_axs[plot_c][plot_r].axis("off")

                if line_count > max_lines:
                    break

            fig_save_path = os.path.join(save_path, file_name + ".png")
            fig.savefig(fig_save_path, dpi=300, bbox_inches="tight")
            print(f"[INFO] Saved visualization to {fig_save_path}")

            if save_ensemble_plots:
                ensemble_save_path = os.path.join(save_path, file_name + "_ensemble.png")
                ensemble_fix.savefig(ensemble_save_path, dpi=300, bbox_inches="tight")
                print(f"[INFO] Saved ensemble visualization to {ensemble_save_path}")

            return save_path
        except Exception as e:
            print(f"Error during METRA visualization: {e}")
            return None
