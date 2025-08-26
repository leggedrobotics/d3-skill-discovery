#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, deque

from rsl_rl.intrinsic_motivation import FACTOR_USD
from rsl_rl.modules import ActorCritic, ExponentialMovingAverageNormalizer
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import TIMER_CUMULATIVE, augment_anymal_action, augment_anymal_obs, mean_gradient_norm


class PPO:
    """PPO adapted for USD"""

    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic,
        usd: FACTOR_USD,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        reward_normalization=False,
        scene_cfg=None,
        symmetry_augmentation=False,
        use_symmetry_aug_for_surrogate=True,
        symmetry_loss_weight=0.0,
        force_reward_symmetry=False,
        rnd_only_keys: list[str] = [],
        # factor_weights=[],
        extrinsic_reward_scale=1.0,
        beta_advantage_UCB=1.0,
        warmup_steps=0,
        # termination_penalty=-50.0,
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.warmup_steps = warmup_steps
        self.step = 0

        # Skill discovery
        self.usd = usd
        if (
            reward_normalization and len(usd._factor_weights) == 1
        ):  # TODO remove this, we normalize the advantages per reward factor so this is not needed
            self.extrinsic_reward_normalizer = ExponentialMovingAverageNormalizer(
                decay_factor=0.999, shape=1, device=self.device, eps=1e-6
            )
        else:
            self.extrinsic_reward_normalizer = None
        self.extrinsic_reward_scale = extrinsic_reward_scale
        self.rnd_only_keys = rnd_only_keys

        # factor_weights = [extrinsic_reward_scale] + factor_weights
        # self.factor_weights = (torch.tensor(factor_weights) / torch.tensor(factor_weights).sum()).to(self.device)
        # self.factor_weights /= self.factor_weights.square().sum().sqrt()
        # self.terminate_penalty = termination_penalty

        # symmetry augmentation
        if symmetry_augmentation and scene_cfg is not None:
            self.symmetry_augment_obs_without_skill = augment_anymal_obs(scene_cfg)
            self.symmetry_augment_action = augment_anymal_action
            self.symmetry_augmentation = True
            self.use_symmetry_loss = True  # symmetry_loss_weight > 0
            self.use_symmetry_aug_for_surrogate = use_symmetry_aug_for_surrogate
            self.symmetry_loss_weight = symmetry_loss_weight
            self.force_reward_symmetry = force_reward_symmetry
        else:
            self.symmetry_augmentation = False
            self.use_symmetry_loss = False
            self.use_symmetry_aug_for_surrogate = False
            self.force_reward_symmetry = False

        # advantage UCB
        self.beta_advantage_UCB = beta_advantage_UCB

        # logging
        self.intrinsic_rewards_mean = deque(maxlen=100)
        self.extrinsic_rewards_mean = deque(maxlen=100)
        self.intrinsic_rewards_std = deque(maxlen=100)
        self.extrinsic_rewards_std = deque(maxlen=100)
        self.regularization_rewards_mean = deque(maxlen=100)
        self.regularization_rewards_std = deque(maxlen=100)

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: list[int] | dict[str, torch.Tensor],
        critic_obs_shape: list[int] | dict[str, torch.Tensor],
        usd_obs_shape: dict[str, torch.Tensor],
        action_shape: list[int],
        num_rewards: int = 1,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            usd_obs_shape,
            action_shape,
            self.device,
            num_rewards=num_rewards,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, usd_obs):
        """Calls the actor to get actions"""

        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()

        # Compute the actions and values
        # store the actions, values, log_probs, and action_mean and action_sigma
        actions, pre_tanh_actions = self.actor_critic.act(obs, return_pre_tanh_actions=True)

        self.transition.actions = pre_tanh_actions.detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            pre_tanh_actions, is_pre_tanh_action=True
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.usd_observations = usd_obs
        self.transition.critic_observations = critic_obs
        return actions.detach()

    def process_env_step(self, rewards, regularization_reward, dones, infos):
        """Add the environment step to the storage, adds intrinsic usd reward to the extrinsic reward"""

        # normalize extrinsic rewards such that they are in the same range as (unscaled) intrinsic rewards
        if self.extrinsic_reward_normalizer is not None:
            rewards = self.extrinsic_reward_normalizer.normalize(rewards)

        # usd reward:
        num_augs = 1
        if self.force_reward_symmetry:
            # symmetry augmentation
            sym_obs = self.symmetry_augment_obs_without_skill(self.transition.usd_observations)
            sym_skill = self.usd.symmetry_augment_skill(self.transition.observations["skill"])
            num_augs = sym_skill.shape[0] // dones.shape[0]
            sym_dones = dones.repeat(num_augs)
            sym_actions = self.symmetry_augment_action(self.actor_critic.squash_action(self.transition.actions))
            # reward calculation
            sym_usd_reward = self.usd.reward(
                sym_obs,
                sym_skill,
                sym_dones,
                sym_actions,
                infos["observations"].get("rnd_extra", None),
            )
            usd_reward = torch.stack(sym_usd_reward.chunk(num_augs, dim=0), dim=0).mean(dim=0)
        else:
            usd_reward = self.usd.reward(
                self.transition.usd_observations,
                self.transition.observations["skill"],
                dones,
                self.actor_critic.squash_action(self.transition.actions),
                infos["observations"].get("rnd_extra", {}),
            )
        # add skill to usd buffer
        self.transition.usd_observations["skill"] = self.transition.observations["skill"]
        # add usd observations to the storage
        if "rnd_extra" in infos["observations"]:
            self.transition.usd_observations |= infos["observations"]["rnd_extra"]

        # transition rewards
        if usd_reward.dim() == 1:
            rewards *= self.extrinsic_reward_scale
            self.transition.rewards = rewards.clone() + usd_reward.clone() + regularization_reward.squeeze().clone()
        else:
            self.transition.rewards = torch.cat([rewards.unsqueeze(1), usd_reward], dim=1)
            self.transition.rewards += regularization_reward
            # due to per factor advantage normalization we need to add the termination reward to each factor
            terminated = dones.bool() & ~infos["time_outs"]
            if terminated.any():
                termination_value = self.transition.rewards[terminated][:, 0]
                self.transition.rewards[terminated] = termination_value.unsqueeze(1)

        # logging
        self.intrinsic_rewards_mean.append(usd_reward.mean())
        self.extrinsic_rewards_mean.append(rewards.mean())
        self.intrinsic_rewards_std.append(usd_reward.std())
        self.extrinsic_rewards_std.append(rewards.std())
        self.regularization_rewards_mean.append(regularization_reward.mean())
        self.regularization_rewards_std.append(regularization_reward.std())

        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            time_out_mask = infos["time_outs"].unsqueeze(1).to(self.device)
            if self.transition.values.dim() == 3:
                time_out_mask = time_out_mask.unsqueeze(2)
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * time_out_mask, 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def symmetry_augment_obs(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        augmented_obs = self.symmetry_augment_obs_without_skill(obs, ignore_keys=self.rnd_only_keys)
        augmented_skill = self.usd.symmetry_augment_skill(obs["skill"])
        augmented_obs["skill"] = augmented_skill
        return augmented_obs

    def update(self):

        if self.step < self.warmup_steps:
            self.storage.clear()
            self.step += 1

            return 0, 0, {}

        mean_value_loss = 0
        mean_factor_value_loss = None
        mean_surrogate_loss = 0
        mean_ratio = 0
        mean_ratio_clipped = 0
        mean_symmetry_loss = 0
        mean_usd_metrics = dict()
        mean_weighted_return_sum = 0
        mean_unweighted_return_sum = 0
        value_gradient_norm = torch.zeros(len(self.actor_critic.critics))
        advantage_ucb_mean = torch.zeros(len(self.usd.val_ensemble_slices))

        log_metrics = defaultdict(list)
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            raise NotImplementedError("Recurrent usd PPO not implemented")
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            next_critic_obs_batch,
            usd_obs_batch,
            next_usd_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            rewards_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:

            # symmetry augmentation
            num_augs = 1
            if self.symmetry_augmentation:
                obs_batch_sym = self.symmetry_augment_obs(obs_batch)
                critic_obs_batch_sym = self.symmetry_augment_obs(critic_obs_batch)

                usd_obs_batch_sym = self.symmetry_augment_obs(usd_obs_batch)
                next_usd_obs_batch_sym = self.symmetry_augment_obs(next_usd_obs_batch)
                actions_batch_sym = self.symmetry_augment_action(actions_batch)
                og_batch_size = actions_batch.shape[0]
                num_augs = actions_batch_sym.shape[0] // og_batch_size
                if self.use_symmetry_aug_for_surrogate:
                    obs_batch = obs_batch_sym
                    critic_obs_batch = critic_obs_batch_sym
                    usd_obs_batch = usd_obs_batch_sym
                    next_usd_obs_batch = next_usd_obs_batch_sym
                    actions_batch = actions_batch_sym
                    target_values_batch = target_values_batch.repeat(4, 1)
                    advantages_batch = advantages_batch.repeat(4, 1)
                    returns_batch = returns_batch.repeat(4, 1)
                    old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(4, 1)

            # update action distribution to get log_probs of the actions in the batch
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch, is_pre_tanh_action=True)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            if value_batch.dim() == 3:
                value_batch = value_batch.squeeze(1)

            # Skill discovery update
            usd_metrics = self.usd.update(usd_obs_batch, next_usd_obs_batch, actions_batch, num_augs=num_augs)

            # Entropy bonus
            entropy_batch = self.actor_critic.entropy

            # KL
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            if self.symmetry_augmentation and self.use_symmetry_aug_for_surrogate:
                org_sigma_batch = sigma_batch[:og_batch_size]
                org_mu_batch = mu_batch[:og_batch_size]
            else:
                org_sigma_batch = sigma_batch
                org_mu_batch = mu_batch
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(org_sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - org_mu_batch))
                        / (2.0 * torch.square(org_sigma_batch))
                        - 0.5,
                        axis=-1,
                    )  # type: ignore
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate clipped loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))

            if self.usd.any_val_ensembles:
                # mean of value ensemble

                advantage_ensemble_means = torch.stack(
                    [advantages_batch[:, start:end].mean(dim=1) for start, end in self.usd.val_ensemble_slices],
                    dim=1,
                )
                # std of value ensemble
                advantage_ensemble_stds = torch.stack(
                    [
                        (
                            advantages_batch[:, start:end].std(dim=1)
                            if (end - start) > 1
                            else torch.zeros(advantages_batch.shape[0], device=advantages_batch.device)
                        )
                        for start, end in self.usd.val_ensemble_slices
                    ],
                    dim=1,
                )

                # aggregate by factor weights
                aggregated_advantages = (
                    torch.sum(advantage_ensemble_means * self.usd.factor_weights(obs_batch["skill"]), dim=1)
                    if advantage_ensemble_means.shape[1] > 1
                    else advantage_ensemble_means.squeeze(1)
                )

                # add ucb term
                aggregated_advantages += self.beta_advantage_UCB * torch.sum(
                    advantage_ensemble_stds * self.usd.factor_weights(obs_batch["skill"]), dim=1
                )
            else:
                # aggregate by factor weights without ucb term
                aggregated_advantages = (
                    torch.sum(advantages_batch * self.usd.factor_weights(obs_batch["skill"]), dim=1)
                    if advantages_batch.shape[1] > 1
                    else advantages_batch.squeeze(1)
                )

            surrogate = -aggregated_advantages * ratio
            surrogate_clipped = -aggregated_advantages * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss (critic loss, MSE)
            if self.use_clipped_value_loss:

                # mean of ensemble for targets to reduce variance
                if self.usd.any_val_ensembles:
                    # mean of value ensemble
                    target_values_batch = torch.cat(
                        [
                            (target_values_batch[:, start:end].mean(dim=1).unsqueeze(1).repeat(1, end - start))
                            for start, end in self.usd.val_ensemble_slices
                        ],
                        dim=1,
                    )

                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean(dim=0)
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean().mean(dim=0)

            loss = surrogate_loss + self.value_loss_coef * value_loss.mean() - self.entropy_coef * entropy_batch.mean()

            if loss.isnan() or loss.isinf():
                print(" surrogate_loss: ", surrogate_loss)
                print(" value_loss: ", value_loss)
                print(" entropy_loss: ", entropy_batch.mean())
                print(" ratio: ", ratio.isnan().any().item(), ratio.isinf().any().item())
                print(
                    "aggregated_advantages: ",
                    aggregated_advantages.isnan().any().item(),
                    aggregated_advantages.isinf().any().item(),
                )
                print(
                    "advantages_batch: ", advantages_batch.isnan().any().item(), advantages_batch.isinf().any().item()
                )
                print("returns_batch: ", returns_batch.isnan().any().item(), returns_batch.isinf().any().item())
                print("value_batch: ", value_batch.isnan().any().item(), value_batch.isinf().any().item())
                print(
                    "actions_log_prob_batch: ",
                    actions_log_prob_batch.isnan().any().item(),
                    actions_log_prob_batch.isinf().any().item(),
                )
                print(
                    "old_actions_log_prob_batch: ",
                    old_actions_log_prob_batch.isnan().any().item(),
                    old_actions_log_prob_batch.isinf().any().item(),
                )
                print("old_mu_batch: ", old_mu_batch.isnan().any().item(), old_mu_batch.isinf().any().item())
                print("old_sigma_batch: ", old_sigma_batch.isnan().any().item(), old_sigma_batch.isinf().any().item())
                print("mu_batch: ", mu_batch.isnan().any().item(), mu_batch.isinf().any().item())
                raise ValueError("Loss is NaN or Inf")

            # Symmetry loss
            if self.symmetry_augmentation and self.use_symmetry_loss:
                # actions predicted by the actor
                obs_batch_sym = {k: v.detach().clone() for k, v in obs_batch_sym.items()}
                # pi(mirrored(s))
                mean_pred_sym_actions_batch = self.actor_critic.act_inference(obs_batch_sym)[og_batch_size:]
                # mirrored(pi(s))
                mean_sym_og_actions_batch = self.symmetry_augment_action(org_mu_batch)[og_batch_size:].detach()
                # loss = MSE(pi(mirrored(s)), mirrored(pi(s)).detach()), both means and not samples
                symmetry_loss = torch.nn.functional.mse_loss(mean_pred_sym_actions_batch, mean_sym_og_actions_batch)
                # add the symmetry loss to the total loss
                loss += symmetry_loss * self.symmetry_loss_weight
                mean_symmetry_loss += symmetry_loss.item()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # - Logging
            # # info max soft update
            # if self.usd.max_info is not None:
            #     self.usd.max_info.soft_update_target_actor(self.actor_critic.actor)
            if mean_factor_value_loss is None:
                mean_factor_value_loss = value_loss
            else:
                mean_factor_value_loss += value_loss

            mean_usd_metrics = {key: mean_usd_metrics.get(key, 0) + value for key, value in usd_metrics.items()}

            mean_value_loss += value_loss.mean().item()
            mean_surrogate_loss += surrogate_loss.item()

            value_gradient_norm += torch.tensor([mean_gradient_norm(critic) for critic in self.actor_critic.critics])

            if self.usd.any_val_ensembles:
                advantage_ucb_mean += advantage_ensemble_stds.mean(dim=0).cpu().detach()

            if self.usd.randomize_factor_weights:
                # check if weights with higher values have higher rewards
                mean_ensemble_returns = torch.stack(
                    [returns_batch[:, start:end].mean(dim=1) for start, end in self.usd.val_ensemble_slices], dim=1
                )
                mean_weighted_return_sum += torch.mean(
                    mean_ensemble_returns * torch.softmax(self.usd.factor_weights(obs_batch["skill"]), dim=-1)
                ).item()
                mean_unweighted_return_sum += mean_ensemble_returns.mean().item() / self.usd._factor_weights.shape[1]

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_ratio /= num_updates
        mean_ratio_clipped /= num_updates
        mean_symmetry_loss /= num_updates
        mean_weighted_return_sum /= num_updates
        mean_unweighted_return_sum /= num_updates
        value_gradient_norm /= num_updates
        advantage_ucb_mean /= num_updates

        self.storage.clear()
        self.usd.update_skill_distribution(num_augs=num_augs)
        self.usd.update_factor_weights()

        usd_metrics = {key: value / num_updates for key, value in mean_usd_metrics.items()}
        usd_metrics.update(self.usd.get_metrics())

        reward_metric = {
            "Reward/intrinsic_mean": torch.tensor(self.intrinsic_rewards_mean).mean().item(),
            "Reward/extrinsic_mean": torch.tensor(self.extrinsic_rewards_mean).mean().item(),
            "Reward/intrinsic_std": torch.tensor(self.intrinsic_rewards_std).mean().item(),
            "Reward/extrinsic_std": torch.tensor(self.extrinsic_rewards_std).mean().item(),
            "Reward/regularization_mean": torch.tensor(self.regularization_rewards_mean).mean().item(),
            "Reward/regularization_std": torch.tensor(self.regularization_rewards_std).mean().item(),
        }

        if self.usd.randomize_factor_weights:
            reward_metric.update(
                {
                    "Factor_Weights/Lambda_Weighted_Return_Sum": mean_weighted_return_sum,
                    "Factor_Weights/Unweighted_Return_Sum": mean_unweighted_return_sum,
                    "Factor_Weights/Diff_WeightedToUnweighted": mean_weighted_return_sum - mean_unweighted_return_sum,
                }
            )

        if mean_factor_value_loss.dim() > 0 and len(mean_factor_value_loss) > 1:
            mean_factor_value_loss /= num_updates
            factor_dim_iter = iter(self.usd.factor_to_num_critics.items())
            k = 0
            factor_name = None
            log_name = None
            ucb_iter = 1
            for i, loss in enumerate(mean_factor_value_loss):
                if i == 0:
                    # the first one is always extrinsic
                    log_name = "extrinsic"
                    if self.usd.val_ensemble_slices[0][1] > 1:
                        reward_metric["UCB/Extrinsic"] = advantage_ucb_mean[0].item()
                else:
                    if k == 0:
                        factor_name, k = next(factor_dim_iter)
                        if self.usd.val_ensemble_slices[ucb_iter][1] - self.usd.val_ensemble_slices[ucb_iter][0] > 1:
                            reward_metric[f"UCB/{factor_name}"] = advantage_ucb_mean[ucb_iter].item()
                        ucb_iter += 1
                    log_name = f"{factor_name}_{k}"
                    k -= 1

                reward_metric[f"Loss/Value_{log_name}"] = loss.item()
                reward_metric[f"Loss/Value_{log_name}_Gradient_Norm"] = value_gradient_norm[i].item()
        if self.extrinsic_reward_normalizer is not None:
            reward_metric.update(
                {
                    "Reward_Normalizer/Mean_Extrinsic_Reward": self.extrinsic_reward_normalizer.mean.mean().item(),
                    "Reward_Normalizer/Std_Extrinsic_Reward": torch.sqrt(self.extrinsic_reward_normalizer.var)
                    .mean()
                    .item(),
                }
            )

        symmetry_metrics = {"Loss/Symmetry": mean_symmetry_loss}

        return (
            mean_value_loss,
            mean_surrogate_loss,
            usd_metrics | reward_metric | symmetry_metrics | log_metrics,
        )
