# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch

from rsl_rl.utils import split_and_pad_trajectories


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.usd_observations: dict[str, torch.Tensor] = None
            self.actions: torch.Tensor = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shape: list[int] | dict[str, torch.Tensor],
        privileged_obs_shape: list[int] | dict[str, torch.Tensor],
        usd_obs_shape: dict[str, torch.Tensor],
        actions_shape: list[int],
        device="cpu",
        num_rewards: int = 1,
    ):
        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape
        self.num_rewards = num_rewards
        # Core
        if isinstance(obs_shape, list):
            self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
            if privileged_obs_shape[0] is not None:
                self.privileged_observations = torch.zeros(
                    num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
                )
            else:
                self.privileged_observations = None
        else:
            # observation is a dict
            self.observations = {
                key: torch.zeros(num_transitions_per_env, num_envs, *obs_shape[key].shape[1:], device=self.device)
                for key in obs_shape
            }

            if privileged_obs_shape[0] is not None:
                self.privileged_observations = {
                    key: torch.zeros(
                        num_transitions_per_env, num_envs, *privileged_obs_shape[key].shape[1:], device=self.device
                    )
                    for key in privileged_obs_shape
                }
            else:
                self.privileged_observations = None

        # usd obs is always a dict
        self.usd_observations = {
            k: torch.zeros(num_transitions_per_env, num_envs, *v.shape[1:], device=self.device)
            for k, v in usd_obs_shape.items()
        }

        self.rewards = torch.zeros(num_transitions_per_env, num_envs, num_rewards, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, num_rewards, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, num_rewards, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, num_rewards, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        self._add_transition_to_buffer(transition.observations, self.observations)
        self._add_transition_to_buffer(transition.critic_observations, self.privileged_observations)
        self._add_transition_to_buffer(transition.usd_observations, self.usd_observations)

        # if isinstance(self.observations, dict):
        #     # observation is a dict
        #     for key in self.observations:
        #         self.observations[key][self.step].copy_(transition.observations[key])
        #     if self.privileged_observations is not None:
        #         for key in self.privileged_observations:
        #             self.privileged_observations[key][self.step].copy_(transition.critic_observations[key])
        # else:
        #     self.observations[self.step].copy_(transition.observations)
        #     if self.privileged_observations is not None:
        #         self.privileged_observations[self.step].copy_(transition.critic_observations)

        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, self.num_rewards))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values.view(-1, self.num_rewards))
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _add_transition_to_buffer(
        self,
        new_transition_element: torch.Tensor | dict[str, torch.Tensor] | None,
        buffer: torch.Tensor | dict[str, torch.Tensor],
    ) -> None:
        if new_transition_element is not None and buffer is not None:
            if isinstance(buffer, dict) and isinstance(new_transition_element, dict):
                for key in buffer:
                    buffer[key][self.step].copy_(new_transition_element[key])
            elif isinstance(buffer, torch.Tensor) and isinstance(new_transition_element, torch.Tensor):
                buffer[self.step].copy_(new_transition_element)
            else:
                raise ValueError("Buffer and new_transition_element must have the same type")

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # initialize if needed
        if self.saved_hidden_states_a is None or self.saved_hidden_states_c is None:

            num_steps_per_env = (
                self.observations.shape[0]
                if isinstance(self.observations, torch.Tensor)
                else self.observations[list(self.observations.keys())[0]].shape[0]
            )  # TODO: check this

            self.saved_hidden_states_a = [
                torch.zeros(num_steps_per_env, *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(num_steps_per_env, *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        # generalized advantage estimation (GAE):
        # A_gae(s_t, a_t) = delta_t + gamma * lam * A_gae(s_{t+1}, a_{t+1}), where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values.view(-1, self.num_rewards)
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean(dim=(0, 1))) / (
            self.advantages.std(dim=(0, 1)) + 1e-8
        )

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0])
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def flatten_buffer(self, buffer: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
        if isinstance(buffer, dict):
            return {key: buffer[key].flatten(0, 1) for key in buffer}
        else:
            return buffer.flatten(0, 1)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * (self.num_transitions_per_env - 1)  # to make sure we get states and next states
        mini_batch_size = batch_size // num_mini_batches
        # indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
        indices = torch.randperm(batch_size, requires_grad=False, device=self.device)

        observations = self.flatten_buffer(self.observations)
        if self.privileged_observations is not None:
            critic_observations = self.flatten_buffer(self.privileged_observations)
        else:
            critic_observations = observations

        usd_observations = self.flatten_buffer(self.usd_observations)

        # if isinstance(self.observations, dict):
        #     observations = {key: self.observations[key].flatten(0, 1) for key in self.observations}
        #     if self.privileged_observations is not None:
        #         critic_observations = {
        #             key: self.privileged_observations[key].flatten(0, 1) for key in self.privileged_observations
        #         }
        #     else:
        #         critic_observations = observations
        # else:

        #     observations = self.observations.flatten(0, 1)
        #     if self.privileged_observations is not None:
        #         critic_observations = self.privileged_observations.flatten(0, 1)
        #     else:
        #         critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        rewards = self.rewards.flatten(0, 1)
        # dones = self.dones.flatten(0, 1).to(torch.bool).squeeze(-1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]
                future_idx = batch_idx + self.num_envs

                if isinstance(self.observations, dict):
                    obs_batch = {key: observations[key][batch_idx] for key in observations}
                    critic_observations_batch = {
                        key: critic_observations[key][batch_idx] for key in critic_observations
                    }
                    next_critic_obs_batch = {key: critic_observations[key][future_idx] for key in observations}

                    # same_skills = (next_obs_batch["skill"] == obs_batch["skill"]).all(dim=1)
                    # diff = torch.norm((next_obs_batch["my_pose"] - obs_batch["my_pose"])[:, :2], dim=1)
                    # dones_batch = dones[batch_idx]
                    # if not (same_skills == ~dones_batch).all():
                    #     print("ERROR: same_skills and dones_batch are not consistent")

                else:
                    obs_batch = observations[batch_idx]
                    critic_observations_batch = critic_observations[batch_idx]
                    next_critic_obs_batch = critic_observations[future_idx]

                usd_obs_batch = {key: usd_observations[key][batch_idx] for key in usd_observations}
                next_usd_obs_batch = {key: usd_observations[key][future_idx] for key in usd_observations}

                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                rewards_batch = rewards[batch_idx]
                yield obs_batch, critic_observations_batch, next_critic_obs_batch, usd_obs_batch, next_usd_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, rewards_batch, (
                    None,
                    None,
                ), None

    # for RNNs only
    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):

        raise NotImplementedError("The function reccurent_mini_batch_generator is not yet adapted for skill discovery")

        if isinstance(self.observations, dict):
            padded_obs_trajectories = {}
            for key, value in self.observations.items():
                padded_obs_trajectories[key], trajectory_masks = split_and_pad_trajectories(value, self.dones)
            if self.privileged_observations is not None:
                padded_critic_obs_trajectories = {}
                for key, value in self.privileged_observations.items():
                    padded_critic_obs_trajectories[key], _ = split_and_pad_trajectories(value, self.dones)
            else:
                padded_critic_obs_trajectories = padded_obs_trajectories
        else:
            padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
            if self.privileged_observations is not None:
                padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
            else:
                padded_critic_obs_trajectories = padded_obs_trajectories

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]  # shift by one
                last_was_done[0] = True  # first element is always a new trajectory
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])  # number of trajectories in the batch
                last_traj = first_traj + trajectories_batch_size  # index of the last trajectory in the batch
                masks_batch = trajectory_masks[:, first_traj:last_traj]

                if isinstance(self.observations, dict):  # TODO: check this
                    obs_batch = {key: value[:, first_traj:last_traj] for key, value in padded_obs_trajectories.items()}
                    critic_obs_batch = {
                        key: value[:, first_traj:last_traj] for key, value in padded_critic_obs_trajectories.items()
                    }

                else:
                    obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                    critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

                yield obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    hid_a_batch,
                    hid_c_batch,
                ), masks_batch

                first_traj = last_traj
