# Copyright (c) 2025, ETH Zurich, Rafael Cathomen
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import math
import torch
from collections import defaultdict, deque
from typing import Any, Literal

from d3_rsl_rl.modules import ExponentialMovingAverageNormalizer
from d3_rsl_rl.utils import is_valid

from .base_skill_discovery import BaseSkillDiscovery
from .diayn import DIAYN
from .metra import METRA
from .rnd import RandomNetworkDistillation


class FACTOR_USD:
    """Implements factorized unsupervised skill discovery based on DUSDI [1]_ with variations of METRA [2]_ and DIAYN.
    Optionally, the algorithm can be used with RND and symmetry augmentation.
    Intended to run alongside an online RL algorithm like PPO.

    References:
        .. [1] Hu et al. "Disentangled Unsupervised Skill Discovery for Efficient Hierarchical Reinforcement Learning" arXiv preprint https://arxiv.org/abs/2410.11251 (2024)
        .. [2] Park et al. "METRA: Scalable Unsupervised RL with Metric-Aware Abstraction" arXiv preprint https://arxiv.org/abs/2310.08887 (2024)
        .. [3] Eysenbach et al. "Diversity is All You Need: Learning Skills without a Reward Function" arXiv preprint https://arxiv.org/abs/1802.06070 (2018)
    """

    def __init__(
        self,
        obs: dict[str, torch.Tensor],
        infos: dict,
        sample_action: torch.Tensor,
        factors: dict[str, tuple[list[str], Literal["metra", "diayn"]]],
        skill_dims: dict[str, int],
        resampling_intervals: dict[str, int],
        diayn: dict,
        metra: dict,
        num_envs: int,
        N_steps: int,
        usd_alg_extra_cfg: dict = {},
        metra_reward_scale: float = 1.0,
        diayn_reward_scale: float = 1.0,
        num_deterministic_skills: int = 0,
        reward_normalization: bool = True,
        value_decomposition: bool = True,
        factor_weight_lr: float = 0.1,
        target_steps_to_goal: int = 100,
        regularize_factor_weights: float = 1.0,
        min_max_weight_max_ratio: float = 100.0,
        adaptive_regularization_weight: bool = True,
        randomize_factor_weights: bool = False,
        factor_weights_skew: float = 1.0,
        adaptive_regularization_weight_params: tuple[float, float] = (0.5, 2),
        disable_style_reward: bool = False,
        disable_factor_weighting: bool = False,
        device: str = "cpu",
        rnd: dict = {},
        **kwargs,
    ):
        """
        Args:
            obs (dict[str, torch.Tensor]): Sample observation dictionary to infer the shapes of the state representations.
            sample_action (torch.Tensor): Sample action tensor to infer the shapes of the actor and critic networks.
            factors (dict[str, tuple[list[str], Literal["metra", "diayn"]]): Dictionary that maps factor names to a tuple of the observation keys and the algorithm name.
            skill_dims (dict[str, int]): Dictionary that maps factor names to the dimension of the skill.
            resampling_intervals (dict[str, int]): Dictionary that maps factor names to the interval at which the skill should be resampled.
            diayn (dict): Dictionary with the DIAYN hyperparameters, same for all factors with DIAYN.
            metra (dict): Dictionary with the METRA hyperparameters, same for all factors with METRA.
            num_envs (int): The number of environments.
            N_steps (int): The episode length.
            metra_reward_scale (float, optional): The (initial) reward scaling for METRA. Defaults to 1.0.
            diayn_reward_scale (float, optional): The (initial) reward scaling for DIAYN. Defaults to 1.0.
            num_deterministic_skills (int, optional): The number of deterministic skills. Defaults to 0, i.g, for video recording.
            reward_normalization (bool, optional): Whether to normalize the rewards. Defaults to True.
            value_decomposition (bool, optional): Whether to decompose the value function. Defaults to True.
            factor_weight_lr (float, optional): The learning rate for the factor weights. Defaults to 0.1.
            target_steps_to_goal (int, optional): Hyperparameter that controls how fast the factor weights are updated. Defaults to 100.
            regularize_factor_weights (float, optional): The regularization weight for the factor weights. Defaults to 1.0. # Not used
            min_max_weight_max_ratio (float, optional): The maximum ratio between the maximum and minimum factor weight. Defaults to 100.0.
            adaptive_regularization_weight (bool, optional): Whether to use adaptive regularization. Defaults to True.
            randomize_factor_weights (bool, optional): If true, the weights are random per env and passed to the observations as part of the skill.
            factor_weights_skew (float, optional): The skewness of the distribution for the factor weights. Defaults to 1.0. > 1 is more skewed to the corners.
            adaptive_regularization_weight_params (tuple[float, float], optional): The parameters for the adaptive regularization. Defaults to (0.5, 2).
            disable_style_reward (bool, optional): If true, the style reward is disabled. Defaults to False.
            disable_factor_weighting (bool, optional): If true, the factor weighting is disabled. Defaults to False.
            device (str, optional): The device to run the algorithm on. Defaults to "cpu".
            rnd (dict, optional): Dictionary with the RND hyperparameters. Defaults to {}.
            **kwargs: Additional keyword arguments.

        """
        self.num_envs = num_envs
        self.device = device

        # - Mappers
        self.factor_to_obs_key_map = {factor_name: factor_values[0] for factor_name, factor_values in factors.items()}
        self.factor_to_algo_map = {factor_name: factor_values[1] for factor_name, factor_values in factors.items()}
        self.algo_to_reward_weight_map = {"metra": metra_reward_scale, "diayn": diayn_reward_scale}
        self.named_factor_weights = {"extrinsic": 1} | {
            factor_name: self.algo_to_reward_weight_map[algo] for factor_name, algo in self.factor_to_algo_map.items()
        }  # type: ignore
        self.rnd_factors: dict[Any, RandomNetworkDistillation] = {}
        self.factor_to_num_critics = {
            factor_name: usd_alg_extra_cfg[factor_name].get("num_critics", 1) if factor_name in usd_alg_extra_cfg else 1
            for factor_name in factors.keys()
        }
        self.rnd_factor_keys = {}
        val_slices = [(0, 1)]
        for num_critics in self.factor_to_num_critics.values():
            val_slices.append((val_slices[-1][1], val_slices[-1][1] + num_critics))
        self.val_ensemble_slices = val_slices
        self.any_val_ensembles = torch.tensor([(b - a) > 1 for a, b in val_slices]).any().item()

        obs_key_set = set(obs.keys())
        factor_key_set = {item for sublist in list(self.factor_to_obs_key_map.values()) for item in sublist}
        if not factor_key_set.issubset(obs_key_set):
            extra_keys = factor_key_set - obs_key_set
            raise ValueError(f"FACTOR_USD.__init__ got unexpected factor keys: {extra_keys}")
        if obs_key_set != factor_key_set:
            diff = obs_key_set - factor_key_set
            print(f"FACTOR_USD.__init__ got observation keys, which will be ignored: {diff}")
        if kwargs:
            print(
                "FACTOR_USD.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        obs_factorized = {
            factor_name: {key: obs[key] for key in factor_keys}
            for factor_name, factor_keys in self.factor_to_obs_key_map.items()
        }

        # - Skill discovery components
        self.factor_skill_discovery_algs: dict[Any, BaseSkillDiscovery] = {}
        self.reward_normalizers: dict[str, ExponentialMovingAverageNormalizer] = {}
        for i, (factor_name, sample_input) in enumerate(obs_factorized.items()):
            skill_d = skill_dims[factor_name] if isinstance(skill_dims, dict) else skill_dims
            algo = self.factor_to_algo_map[factor_name]

            extra_cfg = usd_alg_extra_cfg.get(factor_name, {})

            default_alg_cfg = metra if algo == "metra" else diayn
            alg_cfg = copy.deepcopy(default_alg_cfg)
            alg_cfg.update(extra_cfg)

            if alg_cfg.get("use_rnd", False):
                # define rnd observations
                potential_rnd_obs = (
                    sample_input
                    | infos["observations"].get("rnd_extra", {})
                    | {"action": sample_action, "skill": torch.zeros(sample_action.shape[0], skill_d, device=device)}
                )
                self.rnd_factor_keys[factor_name] = alg_cfg.get("rnd_obs", list(potential_rnd_obs.keys()))
                rnd_obs = {k: v for k, v in potential_rnd_obs.items() if k in self.rnd_factor_keys[factor_name]}
                # rnd

                rnd_args = rnd | alg_cfg  # type: ignore

                self.rnd_factors[factor_name] = RandomNetworkDistillation(
                    input_sample=rnd_obs,
                    num_outputs=rnd_args.get("num_outputs", 128),
                    predictor_hidden_dims=rnd_args.get("layers", [128, 128, 128]),
                    target_hidden_dims=rnd_args.get("layers", [128, 128, 128]),
                    weight=rnd_args.get("rnd_weight", 1.0),
                    perturb_target=rnd_args.get("perturb_target", False),
                    reward_normalization=rnd_args.get("rnd_reward_normalization", False),
                    state_normalization=True,
                    device=device,
                )

            assert isinstance(
                sample_input, dict
            ), "The sample_input must be a dictionary for the current implementation"

            if algo == "metra":
                self.factor_skill_discovery_algs[factor_name] = METRA(
                    obs=sample_input,
                    skill_dim=skill_d,
                    N_steps=N_steps,
                    num_envs=num_envs,
                    device=device,
                    num_deterministic_skills=num_deterministic_skills,
                    **alg_cfg,
                )
            elif algo == "diayn":
                self.factor_skill_discovery_algs[factor_name] = DIAYN(
                    skill_dim=skill_d,
                    sample_obs=sample_input,
                    num_envs=num_envs,
                    device=device,
                    num_deterministic_skills=num_deterministic_skills,
                    complement_sample_obs={k: v for k, v in obs.items() if k not in sample_input},
                    **alg_cfg,
                )
            else:
                raise ValueError(f"Unknown skill discovery algorithm {algo}, must be 'metra' or 'diayn'")

            if reward_normalization:
                self.reward_normalizers[factor_name] = ExponentialMovingAverageNormalizer(
                    decay_factor=0.9,
                    shape=1,
                    device=device,
                    eps=1e-5,
                )

        # - Performance metrics
        self.extrinsic_metric = deque(maxlen=100)
        metric_score_buffer = {}
        weighted_metric_score_buffer = {}
        metric_score_buffer["extrinsic"] = deque(maxlen=100)
        weighted_metric_score_buffer["extrinsic"] = deque(maxlen=10)
        for factor_name in self.factor_skill_discovery_algs.keys():
            metric_score_buffer[factor_name] = deque(maxlen=100)
            weighted_metric_score_buffer[factor_name] = deque(maxlen=10)
        self.metric_score_buffer = metric_score_buffer
        self.weighted_metric_score_buffer = weighted_metric_score_buffer

        # - Hyperparameters
        self.skill_dims: dict[str, int] = skill_dims
        self.resampling_intervals = resampling_intervals
        self.value_decomposition = value_decomposition
        self.factor_weight_lr = factor_weight_lr
        self.target_steps_to_goal = target_steps_to_goal
        self.regularize_factor_weights = torch.tensor(regularize_factor_weights).to(device)
        self.adaptive_regularization_weight = adaptive_regularization_weight
        self.regularizer_m, self.regularizer_p = adaptive_regularization_weight_params
        self.weights_max_norm = abs(math.log(min_max_weight_max_ratio)) * math.sqrt(
            len(self.factor_skill_discovery_algs.keys()) / (len(self.factor_skill_discovery_algs.keys()) + 1)
        )  # derivation by deepseekR1 and gpto1. ln(p)*sqrt(d-1/(d))
        self.randomize_factor_weights = randomize_factor_weights
        self.factor_weights_skew = factor_weights_skew
        self.disable_style_reward = disable_style_reward
        self.disable_factor_weighting = disable_factor_weighting

        # - state
        self.previous_state: dict[str, torch.Tensor] = None  # type: ignore

        # - Diversity measure from DOMiNO
        self.running_mean_state_feature_per_skill = {
            obs_name: torch.zeros_like(obs[obs_name]) for obs_name in obs.keys()
        }
        self.trajectory_lengths = torch.zeros(num_envs).to(device)
        self.quantile_lower_list = {key: [] for key in self.factor_skill_discovery_algs.keys()}
        self.quantile_upper_list = {key: [] for key in self.factor_skill_discovery_algs.keys()}

        # - Logging & visualization
        self.num_deterministic_skills = num_deterministic_skills
        self.sgd_step_counter = 0
        self.N_envs_to_visualize = 10
        self.viz_skill_buffer = [[] for _ in range(self.N_envs_to_visualize)]
        self.viz_skill_full = deque(maxlen=10)
        self.viz_obs_buffer = [defaultdict(list) for _ in range(self.N_envs_to_visualize)]
        self.viz_obs = deque(maxlen=10)
        self.logging_reward = deque(maxlen=100)
        self.logging_metra_reward = deque(maxlen=100)
        self.logging_infomax_reward = deque(maxlen=100)
        self.logging_error = deque(maxlen=100)
        self.logging_projection = deque(maxlen=100)
        self.logging_factor_rewards = defaultdict(lambda: deque(maxlen=100))
        self.rew_counter = 0
        self.extrinsic_performance = torch.zeros(num_envs, device=device)
        self.unweighted_factor_performance = torch.zeros(len(self.named_factor_weights.keys()), device=device)
        self.weighted_factor_performance = torch.zeros(len(self.named_factor_weights.keys()), device=device)

        # - Skills
        self.deterministic_skills = self._sample_deterministic_skills(num_deterministic_skills)
        self._skill = self._sample_skill(torch.ones(num_envs, dtype=torch.bool, device=device))
        if randomize_factor_weights:
            self._factor_weights = self._sample_factor_weights(torch.ones(num_envs, dtype=torch.bool, device=device))
        else:
            self._factor_weights = torch.zeros(len(self.named_factor_weights))

    @property
    def curriculum_metric(self) -> dict[str, torch.Tensor]:
        """Returns a metric per factor that can be used for curriculum learning."""
        curr_metrics = {}
        for factor_name, factor_alg in self.factor_skill_discovery_algs.items():
            curr_metrics[factor_name] = factor_alg.curriculum_metric
        return curr_metrics

    @property
    def overall_metric(self) -> float:
        """Returns the overall performance metric of the skill discovery algorithm."""
        return torch.tensor(list(self.metric_score_buffer.values())).mean().item()

    def save_extrinsic_performance(self, extrinsic_performance: torch.Tensor):
        """Saves the extrinsic performance of the skill discovery algorithm."""
        self.extrinsic_performance += extrinsic_performance.squeeze(1)
        self.extrinsic_metric.append(extrinsic_performance.mean().item())

    def factor_weights(self, skill_batch: torch.Tensor) -> torch.Tensor:
        """Returns the weights per factor used to scale normalized advantages."""
        if self.disable_factor_weighting:
            return torch.nn.functional.normalize(
                torch.ones_like(skill_batch[:, -len(self.named_factor_weights) :]), dim=1
            )

        if self.randomize_factor_weights:
            # factor weights are part of the skill
            return skill_batch[:, -len(self.named_factor_weights) :]

        else:
            all_weights = torch.softmax(torch.tensor(list(self.named_factor_weights.values())).to(self.device), dim=0)
            normalized_weights = all_weights / all_weights.square().sum().sqrt()  # normalize
            self._factor_weights = normalized_weights
            # return torch.cat([normalized_weights, self.regularize_factor_weights.unsqueeze(0)], dim=0)
            return self._factor_weights

    @property
    def skill(self) -> torch.Tensor:
        return self._skill

    def save_previous_state(self, state: torch.Tensor | dict[str, torch.Tensor]):
        """Saves the previous state of the skill discovery algorithm.
        This is used for skill discovery algorithms that require consecutive states, such as METRA."""
        if isinstance(state, dict):
            for key in state.keys():
                self.previous_state[key].copy_(state[key])
        elif isinstance(state, torch.Tensor):
            self.previous_state.copy_(state)
        else:
            raise ValueError("State must be either a dictionary or a tensor.")

    def save_traj_for_visualization(self, obs: dict[str, torch.Tensor], skill: torch.Tensor, dones: torch.Tensor):
        """Save trajectories for visualization."""
        for i in range(self.N_envs_to_visualize):
            env_id = i
            for k, v in obs.items():
                self.viz_obs_buffer[i][k].append(v[env_id])
            self.viz_skill_buffer[i].append(skill[env_id])

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
        action: torch.Tensor = None,  # type: ignore
        rnd_extra: dict[str, torch.Tensor] = {},
    ) -> torch.Tensor:
        """Returns the intrinsic skill discovery reward.
        This method should be called by the rl algorithms 'process_env_step' method to
        calculate the intrinsic reward and combine it with extrinsic rewards.

        Args:
            usd_observations (dict[str, torch.Tensor]): The observations from the environment, must NOT contain the key "skill".
            skill (torch.Tensor): The used skill.
            done (torch.Tensor): The done flag indicates where a new episode has started.
            action (torch.Tensor, optional): The action taken in the environment. Defaults to None.
            rnd_extra (dict[str, torch.Tensor], optional): Extra observations for the RND algorithm. Defaults to {}.
        """
        # - Reward calculation

        if not (is_valid(skill) and is_valid(usd_observations) and is_valid(action)):
            print(f"skill validity: {is_valid(skill)}")
            print(f"obs validity: {is_valid(usd_observations)}")
            if not is_valid(usd_observations):
                for k, v in usd_observations.items():
                    print(f"{k}: {is_valid(v)}")
                    if not is_valid(v):
                        print(f"obs {k} infs: {v.isinf().sum()}")
                        print(f"obs {k} nans: {v.isnan().sum()}")
            print(f"action validity: {is_valid(action)}")
            raise ValueError("[ERROR] invalid skill, obs or action")

        # factorize the observations
        obs_factorized = {
            factor_name: {key: usd_observations[key] for key in factor_keys}
            for factor_name, factor_keys in self.factor_to_obs_key_map.items()
        }
        factor_rewards = []
        skill_index = 0
        for factor_name, obs in obs_factorized.items():
            next_skill_index = skill_index + (
                self.skill_dims[factor_name] if isinstance(self.skill_dims, dict) else self.skill_dims
            )
            # call the reward method
            usd_reward = self.factor_skill_discovery_algs[factor_name].reward(
                usd_observations=obs,
                skill=skill[:, skill_index:next_skill_index],
                done=done,
                action=action,
                complementary_obs={k: v for k, v in usd_observations.items() if k not in obs},
            )
            self.logging_factor_rewards[factor_name].append(usd_reward.mean().item())

            # normalize the reward
            if self.reward_normalizers:
                usd_reward = self.reward_normalizers[factor_name](usd_reward)

            # scale the reward if needed
            reward_scale = (
                self.algo_to_reward_weight_map[self.factor_to_algo_map[factor_name]]
                if not self.value_decomposition
                else 1.0
            )

            # add rnd reward
            if factor_name in self.rnd_factors:
                potential_rnd_obs = (
                    obs | rnd_extra | {"action": action, "skill": skill[:, skill_index:next_skill_index]}  # type: ignore
                )
                rnd_obs = {k: v for k, v in potential_rnd_obs.items() if k in self.rnd_factor_keys[factor_name]}
                usd_reward += self.rnd_factors[factor_name].get_intrinsic_reward(rnd_obs)

            # save the reward
            usd_reward *= reward_scale
            for _ in range(self.factor_to_num_critics[factor_name]):
                factor_rewards.append(usd_reward)
            skill_index = next_skill_index

        aggregate_reward = (
            torch.stack(factor_rewards, dim=1)
            if self.value_decomposition
            else torch.stack(factor_rewards, dim=1).mean(dim=1)
        )

        # - diversity measure from DOMiNO
        for obs_name, obs in usd_observations.items():
            self.running_mean_state_feature_per_skill[obs_name][done[: self.num_envs].bool()] = 0
            self.running_mean_state_feature_per_skill[obs_name] += obs[: self.num_envs]
        self.trajectory_lengths[done[: self.num_envs].bool()] = 0
        self.trajectory_lengths += 1
        self.rew_counter += 1

        if not is_valid(aggregate_reward):
            print(aggregate_reward.mean(dim=0))
            print(f"num nans: {aggregate_reward.isnan().sum()}")
            print(f"num infs: {aggregate_reward.isinf().sum()}")
            raise ValueError("[ERROR] invalid aggregate_reward")

        return aggregate_reward

    def _sample_deterministic_skills(self, num_envs: int) -> torch.Tensor:
        """Creates a one-hot encoding for the deterministic skills. Useful for debugging and visualization."""
        # one hot encoding
        skill_dim = self.skill_dims if isinstance(self.skill_dims, int) else sum(self.skill_dims.values())

        skills = torch.eye(skill_dim, device=self.device)  # * radius
        # deterministic_skills = torch.cat(
        #     [skills, skills * 2 / 3, skills * 1 / 3, skills * 0, -skills * 1 / 3, -skills * 2 / 3, -skills], dim=0
        # )
        deterministic_skills = torch.cat([skills, -skills], dim=0)
        repeats = int(num_envs // deterministic_skills.shape[0]) + 1
        for i in range(repeats):
            deterministic_skills = torch.cat(
                [deterministic_skills, skills / (2 ** (1 + i)), -skills / (2 ** (1 + i))], dim=0
            )
        return deterministic_skills[:num_envs]

    def _sample_skill(self, envs_to_sample: torch.Tensor) -> torch.Tensor:
        """Samples a skill for the specified environments.
        Args:
            envs_to_sample (torch.Tensor): A boolean tensor indicating which environments to sample.
        """
        # call the sample_skill method of each factor
        factor_skills = []
        for factor_name, factor_alg in self.factor_skill_discovery_algs.items():
            factor_skills.append(factor_alg.sample_skill(envs_to_sample))

        # append factor weights to the skill if they are randomized
        if self.randomize_factor_weights and not self.disable_factor_weighting:
            factor_weights = self._sample_factor_weights(envs_to_sample, skew=self.factor_weights_skew)
            factor_skills.append(factor_weights)
        return torch.concat(factor_skills, dim=-1)

    def _sample_factor_weights(self, envs_to_sample: torch.Tensor, skew: float = 1.0) -> torch.Tensor:
        """Samples factor weights randomly. weights are normalized
        Args:
            envs_to_sample (torch.Tensor): A boolean tensor indicating which environments to sample.
            skew (float): The skewness of the distribution. 1.0 is uniform, > 1.0 is more skewed to the corners.
        """
        randn_samples = torch.randn(
            int(envs_to_sample.sum().item()), len(self.named_factor_weights), device=self.device
        )

        if self.disable_style_reward:
            randn_samples[:, 0] = 0.0
        random_factor_weights = torch.nn.functional.normalize(randn_samples.abs() ** skew, dim=-1)
        return random_factor_weights

    def update_skill(self, done_envs: torch.Tensor, episode_lengths: torch.Tensor) -> torch.Tensor:
        """Updates the skill, should be called every step.
        All environments that are done will get a new skill.
        Skill with a defined resampling interval will be updated accordingly."""

        # resample skills for done environments
        dones = done_envs.bool()
        if dones.any():
            self._skill[done_envs.bool()] = self._sample_skill(dones)

        # resample skills if time to resample
        skill_index = 0
        for factor_name, factor_alg in self.factor_skill_discovery_algs.items():
            next_skill_index = skill_index + self.skill_dims[factor_name]
            if self.resampling_intervals[factor_name] > 0:
                time_to_resample = episode_lengths % self.resampling_intervals[factor_name] == 0
                self._skill[time_to_resample, skill_index:next_skill_index] = factor_alg.sample_skill(time_to_resample)
            skill_index = next_skill_index

        # # for debugging/ visualization
        # # hardcoded skills
        # ###
        # if done_envs.float().mean() > 0.5 or not hasattr(self, "test_skill"):
        #     self.test_skill = self._skill[dones][0] if dones.any() else self._skill[0]
        #     # self.id_1 = torch.randint(0, 3, (1,)).item()
        #     # self.id_2 = torch.randint(3, 12, (1,)).item()
        #     # self.id_3 = torch.randint(3 + 12, 12 + 6, (1,)).item()
        # self._skill[:] = self.test_skill
        # # self._skill[...] = 0
        # # self._skill[:, self.id_1] = 1
        # # self._skill[:, self.id_2 : self.id_2 + 3] = 0.333
        # # self._skill[:, self.id_3] = 1
        # self._skill[:, :3] = 0
        # self._skill[:, 1] = 1

        # self._skill[:, 3 : 12 + 3] = 0
        # self._skill[:, 5] = 1

        # augmented_skill = self.symmetry_augment_skill(self._skill).chunk(4, dim=0)
        # augmented_skill = torch.cat(
        #     [augmented_skill[0][::4], augmented_skill[1][::4], augmented_skill[2][::4], augmented_skill[3][::4]]
        # )
        # self._skill = augmented_skill
        # ###

        return self._skill.clone()

    def update(
        self,
        current_obs: dict[str, torch.Tensor],
        next_obs: dict[str, torch.Tensor],
        action_batch: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, float]:
        """
        Updates all skill discovery algorithms and returns the metrics for logging.
        This should be called within the training loop of the rl algorithm.
        Args:
            current_obs (dict[str, torch.Tensor]): The current observation batch.
            next_obs (dict[str, torch.Tensor]): The next observations batch.
            action_batch (torch.Tensor | None): The action batch.
            **kwargs: Additional keyword arguments.
        """

        # factorize observations
        current_obs_factorized = {
            factor_name: {key: current_obs[key] for key in factor_keys}
            for factor_name, factor_keys in self.factor_to_obs_key_map.items()
        }

        next_obs_factorized = {
            factor_name: {key: next_obs[key] for key in factor_keys}
            for factor_name, factor_keys in self.factor_to_obs_key_map.items()
        }

        # add skills to observations
        skill_index = 0
        for factor_name in self.factor_skill_discovery_algs.keys():
            next_skill_index = skill_index + self.skill_dims[factor_name]
            current_skill = current_obs["skill"][:, skill_index:next_skill_index]
            next_skill = next_obs["skill"][:, skill_index:next_skill_index]
            current_obs_factorized[factor_name]["skill"] = current_skill
            next_obs_factorized[factor_name]["skill"] = next_skill
            skill_index = next_skill_index

        # - Update networks
        metrics = {}
        for factor_name, factor_alg in self.factor_skill_discovery_algs.items():
            # usd factor
            metrics[factor_name] = factor_alg.update(
                observation_batch=current_obs_factorized[factor_name],
                next_observation_batch=next_obs_factorized[factor_name],
                action_batch=action_batch,
                complementary_obs_batch={
                    k: v
                    for k, v in current_obs.items()
                    if k not in current_obs_factorized[factor_name]
                    and k not in [k for rnd_factor_keys in self.rnd_factor_keys.values() for k in rnd_factor_keys]
                },
                **kwargs,
            )
            # rnd
            if factor_name in self.rnd_factors:
                potential_rnd_obs = current_obs | {"action": action_batch, "skill": current_skill}
                rnd_obs = {k: v for k, v in potential_rnd_obs.items() if k in self.rnd_factor_keys[factor_name]}
                # remove symmetry for rnd
                num_augs = kwargs.get("num_augs", 1)
                self.rnd_factors[factor_name].update({k: v[: v.shape[0] // num_augs] for k, v in rnd_obs.items()})

        # logging
        metrics_flat = {}
        for factor_name, factor_metrics in metrics.items():
            for log_metric, value in factor_metrics.items():
                if any(alg_str in log_metric.lower() for alg_str in ["metra", "diayn", "dusdi"]):
                    split_metrics = log_metric.split("/")
                    log_metric_name = "/".join([split_metrics[0] + f"_{factor_name}"] + split_metrics[1:])
                else:
                    log_metric_name = f"{log_metric}/{factor_name}"
                metrics_flat[log_metric_name] = value

        return metrics_flat

    def get_metrics(self) -> dict[str, float]:
        """Returns the metrics of the factorized USD algorithm for logging."""
        # - Reward metrics
        reward_metrics = {
            "USD_Reward/"
            + (f"{self.factor_to_algo_map[factor_name]}/" if factor_name in self.factor_to_algo_map else "")
            + factor_name: torch.tensor(value).mean().item()
            for factor_name, value in self.logging_factor_rewards.items()
        }

        reward_normalizer_metrics = {
            f"Reward_Normalizer/Mean_{factor_name}": self.reward_normalizers[factor_name].mean.mean().item()
            for factor_name in self.reward_normalizers
        }
        reward_normalizer_metrics |= {
            f"Reward_Normalizer/Std_{factor_name}": torch.sqrt(self.reward_normalizers[factor_name].var).mean().item()
            for factor_name in self.reward_normalizers
        }  # type: ignore

        rnd_metrics = {
            f"RND/{k}/{factor_name}": v
            for factor_name, rnd in self.rnd_factors.items()
            for k, v in rnd.get_metrics().items()
        }

        # - Factor metrics and weights
        if not self.randomize_factor_weights:
            factor_weights_metric = {
                f"Factor_Weights/{factor_name}": self._factor_weights[i].item()
                for i, factor_name in enumerate(self.named_factor_weights.keys())
            } | {
                "Factor_Weights/Regularizer": self.regularize_factor_weights.item()
            }  # type: ignore
        else:
            factor_weights_metric = {}

        if len(self.metric_score_buffer["extrinsic"]) > 0:
            factor_metrics = {
                f"Factor_Metrics/{factor_name}": metric_buffer[-1]
                for factor_name, metric_buffer in self.metric_score_buffer.items()
            }
            if self.randomize_factor_weights:
                performance_weighted_unweighed = {}
                for i, factor_name in enumerate(self.named_factor_weights.keys()):
                    performance_weighted_unweighed[f"Factor_Metrics/Weighted_{factor_name}"] = (
                        self.weighted_factor_performance[i].item()
                    )
                    performance_weighted_unweighed[f"Factor_Metrics/Unweighted_{factor_name}"] = (
                        self.unweighted_factor_performance[i].item()
                    )
                    performance_weighted_unweighed[f"Factor_Metrics/Diff_{factor_name}"] = (
                        self.weighted_factor_performance[i].item() - self.unweighted_factor_performance[i].item()
                    )
                factor_metrics |= performance_weighted_unweighed

        else:
            factor_metrics = {}

        # - Diversity
        diversity_metric = {}
        quantile = 1 / 3
        skill_index = 0
        for factor_name in self.factor_skill_discovery_algs.keys():
            next_skill_index = skill_index + self.skill_dims[factor_name]
            factor_skill = self._skill[:, skill_index:next_skill_index]
            skill_index = next_skill_index
            max_num_envs = min(self.num_envs, 512)

            # pairwise dist between states with close skills, 10% of the closest
            skill_dist = torch.cdist(factor_skill[:max_num_envs], factor_skill[:max_num_envs], p=2)
            if len(self.quantile_lower_list[factor_name]) < 100:
                # we calculate the quantile for the first 100 steps to get a good estimate
                thresh_lower = torch.quantile(skill_dist[~torch.eye(max_num_envs).bool()], quantile)
                thresh_upper = torch.quantile(skill_dist[~torch.eye(max_num_envs).bool()], 1 - quantile)
                self.quantile_lower_list[factor_name].append(thresh_lower.item())
                self.quantile_upper_list[factor_name].append(thresh_upper.item())
            else:
                # we use the mean of the quantiles since the quantile computation is expensive
                thresh_lower = torch.mean(torch.tensor(self.quantile_lower_list[factor_name]))
                thresh_upper = torch.mean(torch.tensor(self.quantile_upper_list[factor_name]))
            close_skill_mask = skill_dist <= thresh_lower
            close_skill_mask[torch.eye(max_num_envs).bool()] = False
            # pairwise dist between states with faw away skills 10% of the farthest
            far_skill_mask = skill_dist >= thresh_upper
            for obs_name, obs_sum in self.running_mean_state_feature_per_skill.items():
                obs_mean = obs_sum / self.trajectory_lengths.unsqueeze(1)
                # pairwise dist between states
                pairwise_dist = torch.cdist(obs_mean[:max_num_envs], obs_mean[:max_num_envs], p=2)
                pairwise_mean_dist = pairwise_dist.sum() / (self.num_envs * (self.num_envs - 1))
                pairwise_close_dist = pairwise_dist[close_skill_mask].sum() / close_skill_mask.sum()
                pairwise_far_dist = pairwise_dist[far_skill_mask].sum() / far_skill_mask.sum()
                diversity_metric[f"Diversity_{factor_name}_all/{obs_name}"] = pairwise_mean_dist.item()
                diversity_metric[f"Diversity_{factor_name}_close_skill/{obs_name}"] = pairwise_close_dist.item()
                diversity_metric[f"Diversity_{factor_name}_far_skill/{obs_name}"] = pairwise_far_dist.item()

        return (
            reward_metrics
            | reward_normalizer_metrics
            | factor_weights_metric
            | factor_metrics
            | diversity_metric
            | rnd_metrics
        )

    def update_skill_distribution(self, **kwargs):
        """
        Updates skill distribution for each factor.
        This method has to be called before collecting new data, such that the
        new skills are sampled from the updated distribution.
        """
        for factor_alg in self.factor_skill_discovery_algs.values():
            factor_alg.update_skill_distribution(**kwargs)

    def update_factor_weights(self):
        """Updates the reward / advantage weights for each factor based on the performance metrics.
        The weights are only updated if the factor weights are not randomized (i.e. randomize_factor_weights is False).
        """

        # update metric buffers
        factor_performance = [self.extrinsic_performance]
        for factor_name, factor_algo in self.factor_skill_discovery_algs.items():
            self.metric_score_buffer[factor_name].append(factor_algo.performance_metric)
            factor_performance.append(factor_algo.performance.clone())
            factor_algo.performance[...] = 0

        self.metric_score_buffer["extrinsic"].append(torch.tensor(self.extrinsic_metric).mean().item())

        factor_performance_weighted = torch.stack(factor_performance, dim=1) * self.factor_weights(self._skill)
        self.weighted_factor_performance = factor_performance_weighted.mean(dim=0) / (
            self.factor_weights(self._skill).mean(dim=0) * self.rew_counter
        )
        self.unweighted_factor_performance = torch.stack(factor_performance, dim=1).mean(dim=0) / self.rew_counter

        self.rew_counter = 0
        self.extrinsic_performance[...] = 0

        if self.randomize_factor_weights:
            # no need to update weights since they are random
            return

        # get metric slopes
        metric_slopes = {}
        desired_slopes = {}
        for factor_name, metric_buffer in self.metric_score_buffer.items():
            if len(metric_buffer) < 2:
                metric_slopes[factor_name] = 0.0
            else:
                x_values = torch.arange(len(metric_buffer)).float()
                y_values = torch.tensor(metric_buffer)
                x_mean = x_values.mean()
                y_mean = y_values.mean()

                numerator = torch.sum((x_values - x_mean) * (y_values - y_mean))
                denominator = torch.sum((x_values - x_mean) ** 2)
                metric_slopes[factor_name] = (numerator / denominator).item()

            desired_slopes[factor_name] = (1 - metric_buffer[-1]) / self.target_steps_to_goal

        # update factor weights
        for factor_name, factor_weight in self.named_factor_weights.items():
            diff_to_target = desired_slopes[factor_name] - metric_slopes[factor_name]
            self.named_factor_weights[factor_name] += self.factor_weight_lr * diff_to_target

        # squash, such that after softmax of weights, the ratio of smallest and biggest possible weights is min_max_weight_max_ratio
        value_tensor = torch.tensor(list(self.named_factor_weights.values()))
        value_norm = value_tensor.norm()
        factor = torch.clamp(value_norm, max=self.weights_max_norm) / value_norm
        weights = value_tensor * factor
        for i, factor_name in enumerate(self.named_factor_weights.keys()):
            self.named_factor_weights[factor_name] = weights[i].item()

        # update regularizer weight
        if self.adaptive_regularization_weight:
            # update regularizer weight based on weighted average of metric scores
            metrics = torch.clamp(
                torch.tensor([torch.tensor(v).mean() for v in self.metric_score_buffer.values()]),
                min=0.0,
                max=1.0,
            )
            regul_w = torch.sum(torch.softmax(weights, dim=0) * metrics)

            # sigmoid like with shift m and steepness p
            regularize_factor_weights = (regul_w / self.regularizer_m).pow(self.regularizer_p) / (
                (regul_w / self.regularizer_m).pow(self.regularizer_p)
                + ((1 - regul_w) / (1 - self.regularizer_m)).pow(self.regularizer_p)
            )
            self.regularize_factor_weights = regularize_factor_weights.to(self.device)

    def symmetry_augment_skill(self, skill: torch.Tensor) -> torch.Tensor:
        """Augments the skill by adding symmetric versions of the skill.
        Calls the symmetry_augmentation method of the factorized skill discovery algorithms."""

        augmented_skills = []
        skill_index = 0
        for factor_name in self.factor_skill_discovery_algs.keys():
            next_skill_index = skill_index + self.skill_dims[factor_name]
            factor_skill = skill[:, skill_index:next_skill_index]
            augmented_factor_skill = self.factor_skill_discovery_algs[factor_name].symmetry_augmentation(factor_skill)
            augmented_skills.append(augmented_factor_skill)

            skill_index = next_skill_index

        if self.randomize_factor_weights:
            # weights are not mirrored:
            repeated_factor_weights = skill[:, next_skill_index:].repeat(
                augmented_skills[0].shape[0] // skill.shape[0], 1
            )
            augmented_skills.append(repeated_factor_weights)

        return torch.cat(augmented_skills, dim=1)

    def save(self) -> dict:
        """Method to save the intrinsic motivation algorithm."""
        save_dict = {
            factor_name: factor_alg.get_save_dict()
            for factor_name, factor_alg in self.factor_skill_discovery_algs.items()
        }
        return save_dict

    def load(self, save_dict: dict, load_optimizer: bool) -> None:
        """Method to load the intrinsic motivation algorithm."""
        for factor_name, factor_alg in self.factor_skill_discovery_algs.items():
            if factor_name in save_dict:
                factor_alg.load(save_dict[factor_name], load_optimizer=load_optimizer)
            else:
                raise ValueError(f"Factor {factor_name} not found in save dict.")

        self._skill = self._sample_skill(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))

    def visualize(self, save_path: str, file_name: str) -> None:
        """Calls the visualize method of each factor skill discovery algorithm.
        Args:
            save_path (str): The path to save the visualization.
            file_name (str): The name of the file to save the visualization.
        """

        for factor_name, factor_alg in self.factor_skill_discovery_algs.items():
            factor_alg.visualize(save_path, f"{file_name}_{factor_name}", factor_name=factor_name)
