# Copyright (c) 2025, ETH Zurich, Rafael Cathomen
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import torch.optim as optim
from collections import deque
from torch.distributions import Categorical, Dirichlet, Normal
from typing import Literal

from rsl_rl.modules import DictFlattener, SimBa
from rsl_rl.utils import mean_gradient_norm
from rsl_rl.utils.mirroring import remove_symmetry_subspaces

from .base_skill_discovery import BaseSkillDiscovery
from .distributions import vmf_log_prob


class DIAYN(BaseSkillDiscovery):
    """
    Implementation of DIAYN [1]_ with optimistic exploration [2]_ and symmetry augmentation [3]_.
    Option to choose from various skill distributions (categorical, dirichlet, normal, beta, uniform_sphere).

    References:
        .. [1] Eysenbach et al. "Diversity is All You Need: Learning Skills without a Reward Function" arXiv preprint https://arxiv.org/abs/1802.06070 (2018)
        .. [2] Strouse et al. "Learning more skills through optimistic exploration" arXiv preprint https://arxiv.org/abs/2107.14226 (2021)
        .. [3] Mittal et al. "Symmetry Considerations for Learning Task Symmetric Robot Policies" arXiv preprint https://arxiv.org/abs/1802.06070 (2024)
    """

    def __init__(
        self,
        skill_dim: int,
        sample_obs: dict[str, torch.Tensor],
        num_envs: int,
        device: str,
        discriminator_args: dict,
        skill_distribution_type: Literal[
            "uniform", "dirichlet", "normal", "categorical", "beta", "uniform_sphere"
        ] = "categorical",
        num_discriminators: int = 1,
        num_deterministic_skills: int = 0,
        lr=1e-3,
        lambda_exploration: float = 100.0,
        lambda_skill_disentanglement: float = 0.1,
        skill_disentanglement: bool = False,
        complement_sample_obs: dict[str, torch.Tensor] = {},
        max_grad_norm: float = 1.0,
        initial_dirichlet_param: float = 0.05,
        entropy_bonus: float = 0.0,
        symmetry_loss_weight: float = 0.0,
        max_dirichlet_param: float = 0.25,
        symmetry_zero_pad_skills: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            skill_dim: Number of skills for discrete skill space, dimension of the continuous skill space
            sample_obs: Sample observation to infer the input shape
            num_envs: Number of environments
            device: Device to run the algorithm
            skill_distribution_type: Type of skill distribution (uniform, dirichlet, normal, categorical)
            num_discriminators: Number of discriminator networks (DISDAIN)
            num_deterministic_skills: Number of deterministic skills for debugging
            lr: Learning rate for the discriminator networks
            lambda_exploration: Weight for the exploration reward
            skill_disentanglement: Whether to use skill disentanglement (DUSDI)
            complement_sample_obs: Observation of all other state factors (DUSDI)
            max_grad_norm: Maximum gradient norm for the discriminator networks
            entropy_bonus: Entropy bonus for q(z|s)
            symmetry_loss_weight: Weight for the symmetry loss
            symmetry_zero_pad_skills: Whether to zero pad the skills for symmetry augmentation, if true, we always command the same symmetry
        """

        if kwargs:
            print(
                "DIAYN.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()])
            )

        self.device = device
        num_augs = 4  # TODO: fix Hardcoded 4
        self.num_augs = num_augs
        self.num_envs = num_envs

        # - Components
        self.skill_dim = skill_dim
        self.num_discriminators = num_discriminators
        assert "skill" not in sample_obs, "The sample_obs must NOT contain the skill key."
        self.optimistic_exploration = num_discriminators > 1
        skill_disentanglement = skill_disentanglement and bool(complement_sample_obs)
        self.skill_disentanglement = skill_disentanglement
        # - discriminator ensemble
        if symmetry_zero_pad_skills:
            assert skill_dim % num_augs == 0, "Skill dimension must be divisible by num_augs."
            self.skill_dim = skill_dim // num_augs  # only 1 / num_augs -th is commanded, rest is zero
            sample_obs = remove_symmetry_subspaces(sample_obs)
        if skill_distribution_type == "categorical":
            num_q_dist_params = self.skill_dim
        elif skill_distribution_type in ["dirichlet", "uniform_sphere"]:
            num_q_dist_params = self.skill_dim + 1
        else:
            num_q_dist_params = self.skill_dim * 2
        self.discriminators = torch.nn.ModuleList(
            [
                DictFlattener(
                    SimBa(
                        obs=torch.cat(list(sample_obs.values()), dim=1),
                        out_dim=num_q_dist_params,
                        **discriminator_args,
                    )
                )
                for _ in range(num_discriminators)
            ]
        ).to(device)
        self.optimizers = [optim.Adam(discriminator.parameters(), lr=lr) for discriminator in self.discriminators]
        if skill_disentanglement:
            self.entanglement_discriminator = DictFlattener(
                SimBa(
                    obs=torch.cat(list(complement_sample_obs.values()), dim=1),
                    out_dim=num_q_dist_params,
                    **discriminator_args,
                )
            )

            self.entanglement_discriminator.to(device)
            self.entanglement_optimizer = optim.Adam(self.entanglement_discriminator.parameters(), lr=lr)

        self.sampled_skill_in_env = torch.zeros(num_envs, device=device).bool()

        # - hyperparameters
        self.dirichlet_param = initial_dirichlet_param  # only used for dirichlet distribution
        self.min_dirichlet_param = 0.05
        self.skill_distribution_type = skill_distribution_type
        self.symmetry_zero_pad_skills = symmetry_zero_pad_skills
        self.update_distribution(num_augs=num_augs if symmetry_zero_pad_skills else 1)
        self.max_grad_norm = max_grad_norm
        self.lambda_exploration = lambda_exploration
        self.lambda_skill_disentanglement = lambda_skill_disentanglement
        self.entropy_bonus = entropy_bonus
        self.max_dirichlet_param = 1.0 if skill_distribution_type == "beta" else max_dirichlet_param
        self.symmetry_loss_weight = symmetry_loss_weight
        self.not_symmetric = kwargs.get("not_symmetric", False)

        # - debugging, visualization & logging
        self.num_deterministic_skills = num_deterministic_skills
        self._deterministic_skills = self._sample_deterministic_skills(num_deterministic_skills)
        self.logging_diayn_accuracy = deque(maxlen=100)
        self.logging_cosine_similarity = deque(maxlen=100)
        self.logging_dusdi_accuracy = deque(maxlen=100) if self.skill_disentanglement else None
        self.logging_diayn_reward = deque(maxlen=100)
        self.logging_exploration_reward = deque(maxlen=100)
        self.logging_disentanglement_reward = deque(maxlen=100)
        self.performance = torch.zeros(num_envs, device=device)

    @property
    def deterministic_skills(self):
        # add some noise to the deterministic skills such they are not exactly the same
        if self.skill_distribution_type != "categorical":
            return self._deterministic_skills + (torch.rand_like(self._deterministic_skills) - 0.5) * 1e-4
        return self._deterministic_skills

    @property
    def performance_metric(self):
        """Performance metric, used to update factor weight"""
        return (
            torch.tensor(self.logging_diayn_accuracy).mean().item()
            if self.skill_distribution_type in ["categorical", "dirichlet"]
            else torch.tensor(self.logging_cosine_similarity).mean().item()
        )

    ##
    # - Reward
    ##
    def reward(
        self,
        usd_observations: dict[str, torch.Tensor],
        skill: torch.Tensor,
        complementary_obs: dict[str, torch.Tensor] = {},
        **kwargs,
    ) -> torch.Tensor:
        """Calculates the intrinsic reward.
        Here the intrinsic reward is how well the discriminator can predict the skill.
        The discriminator predicts a probability distribution over the skills given the observation
        and the reward is the negative log likelihood of the true skill in this distribution.

        Diayn Reward =  log q_phi(z|s) - log p(z)

        Optionally adds an exploration reward (DISDAIN) and a disentanglement reward (DUSDI)

        Args:
            usd_observations (dict[str, torch.Tensor]): The observations from the environment, must NOT contain the key "skill".
            skill (torch.Tensor): The skill used to calculate the intrinsic reward.
            complementary_obs (dict[str, torch.Tensor], optional): Additional observations for disentanglement reward. Defaults to {}.

        Returns:
            torch.Tensor: The calculated intrinsic DIAYN reward.
        """
        if self.symmetry_zero_pad_skills:
            # remove zeros
            skill = skill[..., : self.skill_dim]

            # we can also remove 1/num_augs -th of the state. this another indurive bias
            # enforcing which sub-skill controls which symmetry
            usd_observations = remove_symmetry_subspaces(usd_observations)

        # - skill discovery reward
        rewards = []
        logits_stack = []
        for discriminator in self.discriminators:
            logits = discriminator(usd_observations)
            logits_stack.append(logits)

            log_q_z_given_s = self.skill_log_prob_from_logits(logits, skill)

            # - skill prior
            log_p_z = self.skill_distribution.log_prob(skill)  # / self.skill_distribution.entropy().abs()
            # eq 3 in the diayn paper
            reward = log_q_z_given_s - log_p_z
            rewards.append(reward)

        diayn_reward = torch.stack(rewards, dim=-1).mean(dim=-1)

        # - exploration reward
        exploration_reward = (
            self._exploration_reward(logits_stack, skill=skill) if self.optimistic_exploration else torch.tensor(0.0)
        )

        # - disentanglement reward
        skill_disentanglement_reward = (
            self._disentanglement_reward(complementary_obs, skill) if self.skill_disentanglement else torch.tensor(0.0)
        )
        # - visualization
        self.save_state_for_visualization(observation=usd_observations, skill=skill, logits=logits)

        # - logging
        self.logging_diayn_reward.append(diayn_reward.mean().item())
        self.logging_exploration_reward.append(exploration_reward.mean().item())
        self.logging_disentanglement_reward.append(skill_disentanglement_reward.mean().item())

        if self.skill_distribution_type in "categorical":
            diayn_accuracy = torch.log_softmax(logits, dim=-1).argmax(dim=-1) == skill.argmax(dim=-1)
            cosine_similarity = torch.nn.functional.cosine_similarity(torch.softmax(logits, dim=-1), skill, dim=-1)

        elif self.skill_distribution_type == "dirichlet":
            diayn_accuracy = torch.log_softmax(logits[..., : skill.shape[1]], dim=-1).argmax(dim=-1) == skill.argmax(
                dim=-1
            )
            cosine_similarity = torch.nn.functional.cosine_similarity(
                torch.softmax(logits[..., : skill.shape[1]], dim=-1), skill, dim=-1
            )
        elif self.skill_distribution_type == "uniform_sphere":
            mean = logits[..., :-1]
            diayn_accuracy = (mean - skill).norm(dim=1) < 0.1
            cosine_similarity = torch.nn.functional.cosine_similarity(mean, skill, dim=-1)
        else:
            # mean likelihood per dimension
            diayn_accuracy = log_q_z_given_s / self.skill_dim
            cosine_similarity = torch.nn.functional.cosine_similarity(
                torch.softmax(logits[..., : self.skill_dim], dim=-1), skill, dim=-1
            )
        self.logging_diayn_accuracy.append(
            (diayn_accuracy.float().mean().item() - 1 / self.skill_dim) / (1 - 1 / self.skill_dim)
        )
        self.logging_cosine_similarity.append(cosine_similarity.mean().item())
        self.performance += cosine_similarity

        full_reward = (
            diayn_reward
            + self.lambda_exploration * exploration_reward
            + self.lambda_skill_disentanglement * skill_disentanglement_reward
        )
        return full_reward

    def _exploration_reward(
        self, logits_stack: list[torch.Tensor], skill: torch.Tensor, num_samples: int = 50
    ) -> torch.Tensor:
        """
        Exploration reward from:
        "Learning More Skills Through Optimistic Exploration" (DISDAIN)
        https://arxiv.org/abs/2107.14226

        computes:
        entropy_1 = H( (1/N) * sum_i q_i )
        entropy_2 = (1/N) * sum_i H(q_i)
        Then returns (entropy_1 - entropy_2).

        Arguments:
            logits_stack: list of length N (ensemble size) containing logits for each ensemble member.
        - For 'categorical', each [batch_size, num_skills] are logits.
        - For 'continuous', each [batch_size, 2*D], chunk -> (mean, log_std).
            num_samples: how many samples to use for the mixture-of-Gaussians entropy approximation.
        """
        # Number of ensemble members
        N = len(logits_stack)

        if self.skill_distribution_type == "categorical":
            """
            1) Categorical:
            - q_i(z) = softmax(logits_i)
            - average_distribution = (1/N) * sum_i q_i(z)
            - entropy_1 = entropy(average_distribution)
            - entropy_2 = (1/N) sum_i entropy(q_i)
            """
            # - Compute the average probability distribution (not the softmax of avg logits!)
            all_probs = []
            for logits in logits_stack:
                probs_i = torch.softmax(logits, dim=-1)
                all_probs.append(probs_i)

            all_probs_t = torch.stack(all_probs, dim=0)
            avg_probs = all_probs_t.mean(dim=0)

            # H( avg_probs )
            avg_probs_clamped = avg_probs.clamp_min(1e-9)
            entropy_1 = -torch.sum(avg_probs_clamped * avg_probs_clamped.log(), dim=-1)

            # - Mean of entropies of each distribution
            entropies = []
            for probs_i in all_probs:
                probs_i = probs_i.clamp_min(1e-9)
                ent_i = -torch.sum(probs_i * probs_i.log(), dim=-1)
                entropies.append(ent_i)
            entropies_t = torch.stack(entropies, dim=0)
            entropy_2 = entropies_t.mean(dim=0)

            exploration_reward = entropy_1 - entropy_2

        elif self.skill_distribution_type in ["normal", "beta"]:
            """
            2) Continuous (Gaussian):
            - each ensemble member is q_i(z) = Normal(mean_i, diag(std_i^2))
            - The first term is H( mixture_of_all_i ), which has no closed form.
                We approximate it by sampling from the mixture.
            - The second term is easy: (1/N)*sum_i H(q_i)
                because H(Normal(mu_i, Sigma_i)) has a closed form.
            """

            def diag_gaussian_entropy(log_std):
                D = log_std.shape[-1]
                return 0.5 * (
                    D * (1.0 + torch.log(torch.tensor(2.0 * 3.1415926535, device=log_std.device)))
                    + 2.0 * log_std.sum(dim=-1)
                )

            # - entropy 2, mean of entropies of each single Gaussian
            all_means = []
            all_log_stds = []

            for logits in logits_stack:
                mean_i, log_std_i = logits.chunk(2, dim=-1)
                all_means.append(mean_i)
                all_log_stds.append(log_std_i)

            all_means_t = torch.stack(all_means, dim=0)
            all_log_stds_t = torch.stack(all_log_stds, dim=0)
            N, B, D = all_means_t.shape

            flat_log_stds = all_log_stds_t.view(N * B, D)
            flat_entropy = diag_gaussian_entropy(flat_log_stds)
            entropies_per_disc = flat_entropy.view(N, B)
            entropy_2 = entropies_per_disc.mean(dim=0)

            # - entropy 1, H(mixture_of_Gaussians)
            # Approximate first term via sampling from the mixture
            idxs = torch.randint(low=0, high=N, size=(B * num_samples,), device=self.device)
            batch_indices = torch.arange(B, device=self.device).repeat_interleave(num_samples)

            chosen_means = all_means_t[idxs, batch_indices]
            chosen_log_stds = all_log_stds_t[idxs, batch_indices]

            # Sample z from each chosen Gaussian
            eps = torch.randn_like(chosen_means)
            chosen_stds = torch.exp(chosen_log_stds)
            z_samples = chosen_means + eps * chosen_stds

            # 3) For each sample, compute log of mixture pdf:
            #    mixture(z) = (1/N)*sum_j Normal_j(z)
            # We'll compute all log_probs under each j in the ensemble and do a log-sum-exp.
            #   log(mixture(z)) = logsumexp over j of [ log_prob_j(z) ] - log(N).

            # Expand z_samples for broadcasting: [1, B*num_samples, D]
            z_samples_exp = z_samples.unsqueeze(0)
            # For each j in [0..N-1], gather the relevant batch element => [N, B*num_samples, D]
            all_means_exp = all_means_t[:, batch_indices, :]
            all_log_stds_exp = all_log_stds_t[:, batch_indices, :]
            all_stds_exp = torch.exp(all_log_stds_exp)

            # log_probs: shape [N, B*num_samples]
            log_probs = self._log_prob_diag_gaussian(z_samples_exp, all_means_exp, all_stds_exp)

            # log( (1/N)*sum_j exp(log_probs[j]) ) = logsumexp(log_probs, dim=0) - log(N)
            # shape => [B*num_samples]
            log_mix_pdf = torch.logsumexp(log_probs, dim=0) - math.log(N)

            neg_log_mix_pdf = -log_mix_pdf  # [B*num_samples]
            # Group by batch index => shape [B, num_samples]
            neg_log_mix_pdf = neg_log_mix_pdf.view(B, num_samples)
            # Approximate mixture entropy => mean across samples => shape [B]
            entropy_1 = neg_log_mix_pdf.mean(dim=1)

            exploration_reward = entropy_1 - entropy_2  # shape [B]

        else:
            # q distribution is more complex, we simply do std of rewards
            rewards = []
            for logits in logits_stack:
                log_q_z_given_s = self.skill_log_prob_from_logits(logits, skill)
                rewards.append(log_q_z_given_s)
            rewards_t = torch.stack(rewards, dim=0)
            exploration_reward = rewards_t.std(dim=0)

        return exploration_reward

    # Helper function for continuous case
    def _log_prob_diag_gaussian(self, z, mu, std):
        """
        z, mu, std: shapes [N, B, D] or broadcasting compatible
        returns log N(z|mu,std) shaped [N, B]
        """
        D = z.shape[-1]
        var = std * std
        # (z - mu)^2 / (2*sigma^2)
        diff = z - mu
        exp_term = (diff * diff / (2.0 * var)).sum(dim=-1)  # sum over D
        log_det = std.log() * D  # but we must sum over D => std is shape [N,B,D]
        log_det = log_det.sum(dim=-1) if log_det.dim() > 1 else log_det
        # final log-prob
        # = - 0.5 * D * log(2*pi) - sum_d log(std_d) - sum_d (diff_d^2)/(2 sigma_d^2)
        return -0.5 * (D * torch.log(torch.tensor(2.0 * torch.pi, device=z.device))) - log_det - exp_term

    # - Disentanglement reward
    def _disentanglement_reward(
        self, complementary_observations: dict[str, torch.Tensor], skill: torch.Tensor
    ) -> torch.Tensor:
        """This reward is calculated the same way as in DIAYN, but as a penalty."""
        logits = self.entanglement_discriminator(complementary_observations)

        log_q_z_given_s = self.skill_log_prob_from_logits(logits, skill)

        if self.skill_distribution_type == "categorical":
            self.logging_dusdi_accuracy.append(
                (torch.log_softmax(logits, dim=-1).argmax(dim=-1) == skill.argmax(dim=-1)).float().mean().item()
            )
        elif self.skill_distribution_type == "dirichlet":
            self.logging_dusdi_accuracy.append(
                (torch.log_softmax(logits[..., :-1], dim=-1).argmax(dim=-1) == skill.argmax(dim=-1))
                .float()
                .mean()
                .item()
            )

        log_p_z = self.skill_distribution.log_prob(skill)  # / self.skill_distribution.entropy().abs()
        penalty = log_q_z_given_s - log_p_z
        return -penalty

    ##
    # - Update
    ##

    def update(
        self,
        observation_batch: dict[str, torch.Tensor],
        complementary_obs_batch: dict[str, torch.Tensor],
        num_augs: int = 1,
        **kwargs,
    ) -> dict[str, float]:
        """Method to update the discriminators"""
        skills = observation_batch["skill"].clone().detach()
        observations_no_skill = {
            key: value.clone().detach() for key, value in observation_batch.items() if key != "skill"
        }

        if self.symmetry_zero_pad_skills:
            # here the skills are symmetry augmented, so we need to be careful
            if self.skill_distribution_type not in ["categorical", "dirichlet", "uniform", "beta", "uniform_sphere"]:
                raise ValueError("Invalid skill distribution type for symmetry zero padding.")

            skills = skills[: skills.shape[0] // num_augs, : self.skill_dim]

            observations_no_skill = remove_symmetry_subspaces(
                {k: v.chunk(num_augs, dim=0)[0] for k, v in observations_no_skill.items()}
            )
            complementary_obs_batch = {k: v[: v.shape[0] // num_augs] for k, v in complementary_obs_batch.items()}

        losses = []
        nlls = []
        for optimizer, discriminator in zip(self.optimizers, self.discriminators):

            # discriminability loss
            optimizer.zero_grad()
            logits = discriminator(observations_no_skill)
            nll = -self.skill_log_prob_from_logits(logits, skills) / self.skill_dim

            loss = nll.mean()

            # symmetry augmentation loss
            sym_loss = 0
            if num_augs > 1 and self.symmetry_loss_weight > 0:
                sym_skills = skills.chunk(num_augs, dim=0)
                chunked_tensors = {key: value.chunk(num_augs, dim=0) for key, value in observations_no_skill.items()}
                sym_obs = tuple(
                    {k: chunked_tensors[k][i] for k in observations_no_skill.keys()} for i in range(num_augs)
                )
                org_ll = self.skill_log_prob_from_logits(discriminator(sym_obs[0]), sym_skills[0]).clone().detach()
                for sym_id in range(1, num_augs):
                    sym_ll = self.skill_log_prob_from_logits(discriminator(sym_obs[sym_id]), sym_skills[sym_id])
                    sym_loss += torch.nn.functional.mse_loss(org_ll, sym_ll) / (num_augs - 1)
            loss += sym_loss * self.symmetry_loss_weight

            if loss.isnan() or loss.isinf():
                print("[WARNING] invalid loss. DIAYN_Loss is NaN or Inf, skipping update")
                print(observation_batch.keys())
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), self.max_grad_norm)
            optimizer.step()
            losses.append(loss.item())
            nlls.append(nll.mean().item())

        # - update disentanglement discriminator
        dusdi_metrics = (
            self._update_disentanglement_discriminator(complementary_obs_batch, skills)
            if self.skill_disentanglement
            else {}
        )

        # - logging
        metrics = {
            "DIAYN/discriminator_loss": torch.tensor(losses).mean().item(),
            "DIAYN/mean_nll": torch.tensor(nlls).mean().item(),
            "DIAYN/mean_symmetry_loss": sym_loss.item() if num_augs > 1 and self.symmetry_loss_weight > 0 else 0,
            "DIAYN/q_entropy": self.entropy_from_logits(logits).item(),
            "DIAYN/mean_gradient_norm": mean_gradient_norm(self.discriminators),
            "DIAYN/mean_likelihood": torch.exp(-torch.tensor(nlls)).mean().item(),
            "DIAYN/mean_accuracy": torch.tensor(self.logging_diayn_accuracy).mean().item(),
            "DIAYN/mean_cosine_similarity": torch.tensor(self.logging_cosine_similarity).mean().item(),
            "DIAYN/reward_diayn": torch.tensor(self.logging_diayn_reward).mean().item(),
            "DIAYN/reward_exploration": (
                torch.tensor(self.logging_exploration_reward).mean().item() * self.lambda_exploration
                if self.logging_exploration_reward
                else 0
            ),
            "DIAYN/reward_disentanglement": (
                torch.tensor(self.logging_disentanglement_reward).mean().item() * self.lambda_skill_disentanglement
                if self.logging_disentanglement_reward
                else 0
            ),
        }
        if self.skill_distribution_type == "dirichlet":
            metrics["DIAYN/dirichlet_param"] = self.dirichlet_param
            metrics["DIAYN/q_concentration"] = torch.nn.functional.softplus(logits[..., -1]).mean().item() + 1.0
        elif self.skill_distribution_type == "uniform_sphere":
            metrics["DIAYN/q_kappa"] = torch.nn.functional.softplus(logits[..., -1]).mean().item()

        elif self.skill_distribution_type not in ["categorical", "dirichlet"]:
            mean, log_std = logits.chunk(2, dim=-1)
            metrics["DIAYN/dist_std"] = torch.exp(log_std).mean().item()
            metrics["DIAYN/l2_diff_dist_mean_skill"] = (mean - skills).norm(dim=1).mean()
            metrics["DIAYN/l1_diff_dist_mean_skill"] = (mean - skills).abs().mean()

        return metrics | dusdi_metrics  # type: ignore

    def update_skill_distribution(self, num_augs: int, **kwargs):
        # - update skill distribution
        if self.skill_distribution_type == "dirichlet" and self.sampled_skill_in_env.all():
            mean_cosin_similarity = torch.tensor(self.logging_cosine_similarity).mean()
            if mean_cosin_similarity > 0.7:
                self.dirichlet_param *= 1.01
            elif mean_cosin_similarity < 0.6:
                self.dirichlet_param *= 0.99
            self.dirichlet_param = min(max(self.dirichlet_param, self.min_dirichlet_param), self.max_dirichlet_param)
            self.sampled_skill_in_env[:] = False
            self.update_distribution(num_augs=num_augs)

    def _update_disentanglement_discriminator(
        self, complementary_obs_batch: dict[str, torch.Tensor], skills: torch.Tensor, **kwargs
    ) -> dict[str, float]:
        """Method to update the disentanglement discriminator"""
        # log_p_z = self.skill_distribution.log_prob(skills) / self.skill_distribution.entropy().abs()
        self.entanglement_discriminator.zero_grad()
        logits = self.entanglement_discriminator(complementary_obs_batch)
        nll = -self.skill_log_prob_from_logits(logits, skills) / self.skill_dim
        # weight = torch.exp(-log_p_z).detach()
        # loss = (nll * weight).mean() / weight.mean()
        loss = nll.mean()

        if loss.isnan() or loss.isinf():
            print("[WARNING] invalid loss. DUSDI_Loss is NaN or Inf, skipping update")
            print(complementary_obs_batch.keys())
            return {}

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.entanglement_discriminator.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.entanglement_discriminator.parameters(), self.max_grad_norm)
        self.entanglement_optimizer.step()

        # - logging
        metrics = {
            "DUSDI/disentanglement_discriminator_loss": loss.item(),
            "DUSDI/mean_disentanglement_nll": nll.mean().item(),
            "DUSDI/mean_disentanglement_gradient_norm": mean_gradient_norm(self.entanglement_discriminator),
            "DUSDI/mean_disentanglement_likelihood": torch.clamp(torch.exp(-nll).mean(), 0, 1000).item(),
            "DUSDI/mean_disentanglement_accuracy": (
                torch.tensor(self.logging_dusdi_accuracy).mean().item() if self.logging_dusdi_accuracy else 0
            ),
        }
        return metrics

    ##
    # - Discriminator
    ##

    def skill_log_prob_from_logits(self, logits: torch.Tensor, skill: torch.Tensor) -> torch.Tensor:
        """Calculates the log probability of the skill given the logits of the discriminator.
        q(z|s)
        This function implements how q(z|s) is parameterized by the logits of the discriminator.

        Args:
            logits (torch.Tensor): The logits of the discriminator.
            skill (torch.Tensor): The skill used to calculate the intrinsic reward.

        Returns:
            reward (torch.Tensor): The log probability of the skill given the logits of the discriminator.
        """

        if self.symmetry_zero_pad_skills:
            # remove the zeros
            skill = skill[..., : self.skill_dim]

        if self.skill_distribution_type == "categorical":
            # logits for categorical distribution
            log_q_z_given_s_all_skills = torch.log_softmax(logits, dim=-1)
            # - select the log probability of the true skill
            log_q_z_given_s = log_q_z_given_s_all_skills[
                torch.arange(len(skill)), torch.argmax(skill.to(int), dim=-1)  # type: ignore
            ]
        elif self.skill_distribution_type == "dirichlet":
            # q is also a dirichlet distribution
            ratio = torch.softmax(logits[..., :-1], dim=-1) + 1e-6
            concentration = torch.nn.functional.softplus(logits[..., -1]) + 1.0  # + 1 for unimodal distribution
            alphas = concentration.unsqueeze(1) * ratio * ratio.shape[-1]

            dist = Dirichlet(alphas)
            log_q_z_given_s = dist.log_prob(skill)
        elif self.skill_distribution_type == "uniform_sphere":
            # q is a von Mises-Fisher distribution
            # else:  # using pytorch vMF
            mu_logits = logits[..., :-1]  # d dim
            mu = torch.nn.functional.normalize(mu_logits, p=2, dim=-1)
            concentration = torch.clamp(torch.nn.functional.softplus(logits[..., -1]) + 1e-6, max=100)  # scalar

            log_q_z_given_s = vmf_log_prob(skill, mu, concentration, cutoff=50, max_series=100)

        else:
            # for any p(z) we can use a multivariate gaussian with diagonal covariance (with infinite support)
            mean, log_std = logits.chunk(2, dim=-1)
            dist = Normal(mean, torch.exp(log_std))
            log_q_z_given_s = dist.log_prob(skill).sum(dim=-1)  #

        if log_q_z_given_s.isnan().any() or log_q_z_given_s.isinf().any():
            print("[WARNING] invalid log_q_z_given_s. NaN or Inf, masking them out")
            print(f"num skill infs: {skill.isnan().sum()}")
            print(f"num skill nans: {skill.isinf().sum()}")
            print(f"num logits nans: {logits.isnan().sum()}")
            print(f"num logits infs: {logits.isinf().sum()}")
            print(f"num nans: {log_q_z_given_s.isnan().sum()}")
            print(f"num infs: {log_q_z_given_s.isinf().sum()}")

            invalid_mask = torch.logical_or(log_q_z_given_s.isnan(), log_q_z_given_s.isinf())
            max_valid = log_q_z_given_s[~invalid_mask].max()
            min_valid = log_q_z_given_s[~invalid_mask].min()
            print(f"max_valid: {max_valid}, min_valid: {min_valid}")
            log_q_z_given_s[log_q_z_given_s.isnan()] = 0.0
            return torch.clamp(log_q_z_given_s, min=min_valid, max=max_valid)

        return log_q_z_given_s

    def entropy_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Calculates the entropy of the skill distribution given the logits of the discriminator.
        H(q(z|s))
        This function implements how H(q(z|s)) is parameterized by the logits of the discriminator.
        Args:
            logits (torch.Tensor): The logits of the discriminator.
        Returns:
            entropy (torch.Tensor): The entropy of the skill distribution given the logits of the discriminator.
        """
        if self.skill_distribution_type == "categorical":
            entropy = torch.distributions.Categorical(logits=logits).entropy().mean()
        elif self.skill_distribution_type == "dirichlet":
            ratio = torch.softmax(logits[..., :-1], dim=-1)
            concentration = torch.nn.functional.softplus(logits[..., -1]) + 1.0  # + 1 for unimodal distribution
            entropy = Dirichlet(concentration.unsqueeze(1) * ratio * self.skill_dim).entropy().mean()
        elif self.skill_distribution_type == "uniform_sphere":
            kappa = torch.nn.functional.softplus(logits[..., -1]) + 1e-6
            entropy = -kappa.mean()  # not actually correct, but its only for logging
        else:
            mean, log_std = logits.chunk(2, dim=-1)
            dist = Normal(mean, torch.exp(log_std))
            entropy = dist.entropy().mean()
        return entropy

    ##
    # - Sampling
    ##

    def sample_skill(
        self,
        envs_to_sample: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Samples from the specified skill-distribution.

        Args:
            envs_to_sample (torch.Tensor): The environments to sample from.
                This is a boolean tensor of shape (num_envs,).
                The environments to sample from are the ones with a value of True.
        Returns:
            torch.Tensor: The sampled skills.
        """
        # uniform skill distribution
        num_envs = int(envs_to_sample.sum().item())
        # - random skills
        skills = self.skill_distribution.sample((num_envs,)).float()

        # - deterministic skills
        if envs_to_sample[: self.num_deterministic_skills].any():
            deterministic_skills = self.deterministic_skills[envs_to_sample[: self.num_deterministic_skills].to(bool)]
            # - combine
            skills[: deterministic_skills.shape[0]] = deterministic_skills

        self.sampled_skill_in_env[envs_to_sample.bool()] = True

        if self.symmetry_zero_pad_skills:
            # we only command 1 / num_augs -th of the skills space the rest is zero padded here
            skills = torch.cat([skills, torch.zeros_like(skills).repeat(1, self.num_augs - 1)], dim=1)

        return skills

    def _sample_deterministic_skills(self, num_envs: int) -> torch.Tensor:
        """Creates a one-hot encoding for the deterministic skills."""
        # one hot encoding
        skills = torch.eye(self.skill_dim, device=self.device)  # * radius
        # make sure the deterministic skills are in the support of the distribution

        if self.skill_distribution_type == "categorical":
            deterministic_skills = torch.cat([skills, skills], dim=0)
        elif self.skill_distribution_type == "dirichlet":
            # all values 0< x < 1
            eps = 1e-6
            skills[skills == 0] = eps
            skills[skills == 1] = 1 - (eps * (self.skill_dim - 1))
            deterministic_skills = torch.cat([skills, skills], dim=0)

        elif self.skill_distribution_type == "uniform":
            deterministic_skills = torch.cat([skills, torch.zeros_like(skills)], dim=0)
        else:
            deterministic_skills = torch.cat([skills, -skills], dim=0)
        repeats = int(num_envs // deterministic_skills.shape[0]) + 1

        if self.skill_distribution_type == "categorical":
            # make sure the deterministic skills are in the support of the distribution
            deterministic_skills = deterministic_skills.repeat(repeats, 1)
        else:
            for i in range(repeats):
                deterministic_skills = torch.cat(
                    [deterministic_skills, skills / (2 ** (1 + i)), -skills / (2 ** (1 + i))], dim=0
                )
        return deterministic_skills[:num_envs]

    def update_distribution(self, num_augs: int = 1):

        # - define the skill distribution
        if self.skill_distribution_type == "dirichlet":

            class CustomDirichlet:
                """Implements dirichlet distribution where all alphas are the same.
                If alpha is small, behaves like a categorical distribution."""

                def __init__(
                    self,
                    concentration: torch.Tensor,
                ):
                    if concentration.mean() < 0.01:
                        self._distribution = Categorical(torch.ones_like(concentration))
                        self.is_dirichlet = False
                    else:
                        self._distribution = Dirichlet(concentration)
                        self.is_dirichlet = True

                def sample(self, shape: torch.Size) -> torch.Tensor:
                    if self.is_dirichlet:
                        return self._distribution.sample(shape)
                    else:
                        indices = self._distribution.sample(shape)
                        num_classes = self._distribution.logits.size(-1)
                        return torch.nn.functional.one_hot(indices, num_classes=num_classes)

                def log_prob(self, value: torch.Tensor):
                    if self.is_dirichlet:
                        value[value == 0] = 1e-9
                        return self._distribution.log_prob(value)
                    else:
                        return self._distribution.log_prob(value.argmax(dim=-1))

                def entropy(self):
                    return self._distribution.entropy()

            self.skill_distribution = CustomDirichlet(
                torch.ones(self.skill_dim, device=self.device) * self.dirichlet_param
            )
        elif self.skill_distribution_type == "normal":

            class DiagMultivariateNormal(torch.distributions.Normal):
                def log_prob(self, value: torch.Tensor):
                    return super().log_prob(value).sum(dim=-1)

            self.skill_distribution = DiagMultivariateNormal(
                torch.zeros(self.skill_dim, device=self.device), torch.ones(self.skill_dim, device=self.device)
            )
        elif self.skill_distribution_type == "uniform":

            class CustomUniform(torch.distributions.Uniform):
                """Implements log_prob for a uniform distribution"""

                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    volume = torch.prod(self.high - self.low)
                    self.log_prob_const = -torch.log(volume)

                def log_prob(self, value: torch.Tensor):
                    in_support = ((value >= self.low) & (value <= self.high)).all(dim=-1)
                    return torch.where(in_support, self.log_prob_const, torch.tensor(-math.inf, device=value.device))

            self.skill_distribution = CustomUniform(
                torch.zeros(self.skill_dim, device=self.device), torch.ones(self.skill_dim, device=self.device)
            )
        elif self.skill_distribution_type == "categorical":

            class OneHotCategorical(torch.distributions.Categorical):
                """Handles one-hot encoding for the categorical distribution"""

                def sample(self, sample_shape=torch.Size):
                    # Call the original sample method to get indices
                    indices = super().sample(sample_shape)
                    # Convert indices to one-hot encodings
                    num_classes = self.logits.size(-1)
                    return torch.nn.functional.one_hot(indices, num_classes=num_classes)

                def log_prob(self, value: torch.Tensor):
                    return super().log_prob(value.argmax(dim=-1))

            self.skill_distribution = OneHotCategorical(torch.ones(self.skill_dim, device=self.device))
        elif self.skill_distribution_type == "beta":
            gamma = torch.ones(self.skill_dim).to(self.device) * self.dirichlet_param
            self.skill_distribution = torch.distributions.Independent(
                torch.distributions.Beta(gamma, gamma), reinterpreted_batch_ndims=1
            )
        elif self.skill_distribution_type == "uniform_sphere":

            class UniformSphere:
                def __init__(self, dim, device):
                    self.dim = dim
                    self.device = device

                def sample(self, shape: torch.Size) -> torch.Tensor:
                    samples = torch.randn(shape + (self.dim,), device=self.device)
                    return samples / samples.norm(dim=-1, keepdim=True)

                def log_prob(self, value: torch.Tensor):
                    # is constant 0
                    return torch.zeros(value.shape[:-1], device=self.device)

                def entropy(self):
                    return torch.tensor(math.log(self.dim), device=self.device)

            self.skill_distribution = UniformSphere(self.skill_dim, self.device)
        else:
            raise ValueError(f"Skill distribution {self.skill_distribution_type} not supported.")

    def symmetry_augmentation(self, skill: torch.Tensor) -> torch.Tensor:
        if self.not_symmetric:
            return torch.cat([skill, skill, skill, skill], dim=0)

        if skill.shape[-1] % 4 == 0:
            if (
                self.skill_distribution_type in ["categorical", "dirichlet", "uniform", "beta"]
                or self.symmetry_zero_pad_skills
            ):
                # This type of symmetry is required for zero padding

                # no negative skills
                # permute the skills

                skill_quarters = skill.chunk(4, dim=1)
                lr_skill = torch.cat(
                    [skill_quarters[1], skill_quarters[0], skill_quarters[3], skill_quarters[2]], dim=-1
                )
                fb_skill = torch.cat(
                    [skill_quarters[2], skill_quarters[3], skill_quarters[0], skill_quarters[1]], dim=-1
                )
                rot_skill = torch.cat(
                    [skill_quarters[3], skill_quarters[2], skill_quarters[1], skill_quarters[0]], dim=-1
                )
            elif self.skill_distribution_type in ["normal", "uniform_sphere"]:
                # negative skills
                # dont permute the skills
                skill_quarters = skill.chunk(4, dim=1)
                lr_skill = torch.cat(
                    [skill_quarters[0], skill_quarters[1], -skill_quarters[2], -skill_quarters[3]], dim=-1
                )
                fb_skill = torch.cat(
                    [-skill_quarters[0], -skill_quarters[1], skill_quarters[2], skill_quarters[3]], dim=-1
                )
                rot_skill = torch.cat(
                    [-skill_quarters[0], -skill_quarters[1], -skill_quarters[2], -skill_quarters[3]], dim=-1
                )
            else:
                raise NotImplementedError(
                    f"Symmetry augmentation is not implemented for {self.skill_distribution_type}"
                )
        elif skill.shape[-1] == 2:
            lr_skill = skill[:, [1, 0]].clone()
            fb_skill = skill[:, [1, 0]].clone()
            rot_skill = skill.clone()
        else:
            raise NotImplementedError(
                f"Symmetry augmentation is only implemented for skills with a multiple of 4, but got {skill.shape[-1]}"
            )

        return torch.cat([skill, lr_skill, fb_skill, rot_skill], dim=0)

    def save_state_for_visualization(
        self, observation: dict[str, torch.Tensor], skill: torch.Tensor, logits: torch.Tensor
    ):
        """Method to save the state for visualization"""
        self.viz_usd_observations = {k: v[: self.num_envs] for k, v in observation.items()}
        self.viz_skill = skill[: self.num_envs]
        self.viz_logits = logits[: self.num_envs]

    def get_save_dict(self) -> dict:
        """Method to save the state for visualization"""
        return {
            "dirichlet_param": self.dirichlet_param,
            "model_state_dict": self.discriminators.state_dict(),
            "optimizer_state_dict": [optimizer.state_dict() for optimizer in self.optimizers],
            "entanglement_state_dict": (
                self.entanglement_discriminator.state_dict() if self.skill_disentanglement else None
            ),
            "entanglement_optimizer_state_dict": (
                self.entanglement_optimizer.state_dict() if self.skill_disentanglement else None
            ),
        }

    def load(self, state_dict: dict, load_optimizer: bool = True):
        """Method to load the state for visualization"""
        self.dirichlet_param = state_dict["dirichlet_param"]
        self.discriminators.load_state_dict(state_dict["model_state_dict"])
        if self.skill_disentanglement:
            self.entanglement_discriminator.load_state_dict(state_dict["entanglement_state_dict"])
        if load_optimizer:
            for optimizer, optimizer_state in zip(self.optimizers, state_dict["optimizer_state_dict"]):
                optimizer.load_state_dict(optimizer_state)
            if self.skill_disentanglement:
                self.entanglement_optimizer.load_state_dict(state_dict["entanglement_optimizer_state_dict"])
        self.update_distribution()

    def visualize(
        self, save_path: str, file_name: str, factor_name: str = "", save_ensemble_plots: bool = False
    ) -> str:
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
            import numpy as np
            import os

            matplotlib.use("Agg")

            # Project to 2D if the data is higher dimensional
            flat_obs = torch.concat([v.flatten(1) for v in self.viz_usd_observations.values()], dim=1)
            if "foot_pos" in self.viz_usd_observations:
                # only look at xz of one foot
                if self.symmetry_zero_pad_skills:
                    # already only one leg
                    flat_obs = self.viz_usd_observations["foot_pos"][..., [0, 2]].flatten(1)
                else:
                    flat_obs = self.viz_usd_observations["foot_pos"].view(-1, 4, 3)[..., 0, [0, 2]].flatten(1)

            dim = flat_obs.size(1)
            if dim > 2:
                projection_matrix = get_2d_projection(flat_obs)
                flat_obs = project_to_plane(flat_obs, projection_matrix)
            logit_dim = self.viz_logits.size(1)
            if logit_dim > 2:
                projection_matrix = get_2d_projection(self.viz_logits)
                viz_logits = project_to_plane(self.viz_logits, projection_matrix)
            else:
                viz_logits = self.viz_logits

            flat_obs = flat_obs.cpu().numpy()
            viz_logits = viz_logits.cpu().numpy()

            # Plot the data
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))

            skill_colors = plt.get_cmap("hsv")(np.linspace(0, 1, self.skill_dim + 1))[:-1]
            point_colors = self.viz_skill.abs().cpu().numpy() @ skill_colors
            point_colors /= point_colors.max()

            ax[0].scatter(flat_obs[:, 0], flat_obs[:, 1], c=point_colors)
            ax[1].scatter(viz_logits[:, 0], viz_logits[:, 1], c=point_colors)

            # Create legend handles for each skill
            legend_handles = [
                matplotlib.patches.Patch(color=skill_colors[i], label=f"{i+1}") for i in range(self.skill_dim)
            ]
            # Add the legend to the plot
            ax[0].legend(handles=legend_handles, title="Skills", loc="upper right")

            ax[0].set_title(f"Observations, {dim}d")
            ax[1].set_title(f"Logits, {logit_dim}d")

            extra_str = f"param: {round(self.dirichlet_param,3)}" if self.skill_distribution_type == "dirichlet" else ""
            fig.suptitle(f"DIAYN, factor: {factor_name}, z type: {self.skill_distribution_type} {extra_str}")
            fig_save_path = os.path.join(save_path, file_name + ".png")
            fig.savefig(fig_save_path, dpi=300, bbox_inches="tight")
            print(f"[INFO] Saved visualization to {fig_save_path}")

        except Exception as e:
            print(f"Error during DIAYN visualization: {e}")
            pass
