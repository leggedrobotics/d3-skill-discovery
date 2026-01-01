# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import EmpiricalNormalization

EPS = 1e-8


class MaxInfo:
    """Basic implementation of the maximum information gain exploration strategy, as described in the paper
    https://arxiv.org/abs/2412.12098
    """

    def __init__(
        self,
        sample_obs: dict[str, torch.Tensor] | torch.Tensor,
        sample_action: torch.Tensor,
        actor: nn.Module,
        ensemble_size: int = 5,
        layers: list[int] = [512, 256, 128],
        tau: float = 0.01,
        lr: float = 1e-3,
        max_grad_norm: float = 1.0,
        init_alpha: float = 1.0,
    ):
        """
        Args:
            sample_obs (dict or torch.Tensor): Sample observation tensor or dictionary of observation tensors.
            sample_action (torch.Tensor): Sample action tensor.
            actor (nn.Module): Actor network for the target policy. IMPORTANT: The actor must be a different instance from  the one used for the agent.
            ensemble_size (int, optional): Number of forward models in the ensemble. Defaults to 5.
            layers (list, optional): List of layer sizes for the forward models. Defaults to [512, 256, 128].
            tau (float, optional): Soft update coefficient for the target actor. Defaults to 0.005.
            lr (float, optional): Learning rate for the forward models. Defaults to 1e-3.
            max_grad_norm (float, optional): Maximum gradient norm for the forward models. Defaults to 1.0.
            init_alpha (float, optional): Initial value for the entropy regularization coefficient. Defaults to 1.0.

        """
        sample_obs = flatten_obs(sample_obs)
        # - components
        self.ensemble = nn.ModuleList(
            [ForwardModel(sample_obs.shape[1], sample_action.shape[1], layers) for _ in range(ensemble_size)]
        )
        self.obs_normalizer = EmpiricalNormalization(sample_obs.shape[1], until=1e8)
        self.action_normalizer = EmpiricalNormalization(sample_action.shape[1], until=1e8)
        self.entropy_normalizer = EmpiricalNormalization(1, until=1e8)
        # learnable alpha
        self.log_alpha = nn.Parameter(torch.tensor(init_alpha).log())
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)

        # copy actor for target policy and reinitialize
        self.target_actor = actor
        self.target_actor.eval()
        if hasattr(self.target_actor, "critic"):
            del self.target_actor.critic

        # - hyperparameters
        self.max_grad_norm = max_grad_norm
        self.tau = tau

        # - optimizer
        self.optimizers = [optim.Adam(model.parameters(), lr=lr) for model in self.ensemble]

    def reward(self, obs: dict[str, torch.Tensor] | torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute intrinsic reward for the given observation and action.

        Args:
            obs (dict or torch.Tensor): Observation tensor or dictionary of observation tensors.
            action (torch.Tensor): Action tensor.

        Returns:
            torch.Tensor: Intrinsic reward tensor.
        """
        with torch.inference_mode():
            obs = flatten_obs(obs)
            normalized_obs = self.obs_normalizer(obs)
            normalized_action = self.action_normalizer(action)

            next_state_predictions = torch.stack(
                [model(torch.cat([normalized_obs, normalized_action], dim=1)) for model in self.ensemble]
            )

            # Compute intrinsic reward based on disagreement between ensemble members
            epistemic_var = next_state_predictions.var(dim=0)
            entropies = torch.log(epistemic_var + EPS).mean(dim=1)

            # normalize
            # return entropies
            return self.entropy_normalizer(entropies.unsqueeze(1)).squeeze(1) * torch.exp(self.log_alpha.detach())

    def update(
        self,
        current_obs_batch: dict[str, torch.Tensor] | torch.Tensor,
        action_batch: torch.Tensor,
        next_obs_batch: dict[str, torch.Tensor] | torch.Tensor,
        skill_batch: torch.Tensor,
    ) -> dict[str, float]:
        """
        Update the ensemble of forward models
        Contrary to the original implementation, we do not predict rewards, only next states.
        The reason is that we do skill discovery in a separate module resulting in a non-stationary reward function.
        """
        # - update alpha
        self.train()
        self.disable_normalizer_update()
        dyn_entropy = self.reward(current_obs_batch, action_batch)
        target_dyn_entropy = self.reward(
            current_obs_batch, self.target_actor.act(current_obs_batch | {"skill": skill_batch})  # type: ignore
        )
        entropy_diff = target_dyn_entropy - dyn_entropy
        alpha_loss = (self.log_alpha * entropy_diff.detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # - prepare data
        normalized_current_obs = self.obs_normalizer(flatten_obs(current_obs_batch))
        normalized_next_obs = self.obs_normalizer(flatten_obs(next_obs_batch))
        delta_state = normalized_next_obs - normalized_current_obs
        normalized_action = self.action_normalizer(action_batch)

        # - update ensemble
        losses = []
        for optimizer, fw_model in zip(self.optimizers, self.ensemble):
            optimizer.zero_grad()
            delta_state_prediction = fw_model(torch.cat([normalized_current_obs, normalized_action], dim=1))
            loss = nn.functional.mse_loss(delta_state_prediction, delta_state)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fw_model.parameters(), self.max_grad_norm)
            optimizer.step()
        self.eval()
        self.enable_normalizer_update()

        # - metrics
        metrics = {
            "InfoMax/loss": np.mean(losses),
            "InfoMax/loss_std": np.std(losses),
            "InfoMax/entropy": dyn_entropy.mean().item(),
            "InfoMax/target_entropy": target_dyn_entropy.mean().item(),
            "InfoMax/alpha": torch.exp(self.log_alpha).item(),
            "InfoMax/alpha_loss": alpha_loss.item(),
        }

        return metrics

    def soft_update_target_actor(self, actor: nn.Module):
        for target_param, param in zip(self.target_actor.parameters(), actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    # - other stuff
    def train(self):
        """Set the models to training mode."""
        for model in self.ensemble:
            model.train()
        self.obs_normalizer.train()
        self.action_normalizer.train()
        self.entropy_normalizer.train()

    def disable_normalizer_update(self):
        self.obs_normalizer.eval()
        self.action_normalizer.eval()
        self.entropy_normalizer.eval()

    def enable_normalizer_update(self):
        self.obs_normalizer.train()
        self.action_normalizer.train()
        self.entropy_normalizer.train

    def eval(self):
        """Set the models to evaluation mode."""
        for model in self.ensemble:
            model.eval()
        self.obs_normalizer.eval()
        self.action_normalizer.eval()
        self.entropy_normalizer.eval()

    def to(self, device):
        """Move the models to the specified device."""
        for model in self.ensemble:
            model.to(device)
        self.obs_normalizer.to(device)
        self.action_normalizer.to(device)
        self.entropy_normalizer.to(device)
        self.target_actor.to(device)


class ForwardModel(nn.Module):
    """Implements a forward model, that predicts the next state given the current state and action."""

    def __init__(self, obs_dim: int, action_dim: int, layer_dims: list[int], activation: nn.Module = nn.ReLU()):
        """Initialize the forward model.

        Args:
            obs_dim (int): Dimension of observation space.
            action_dim (int): Dimension of action space.
            layer_dims (list): List of layer sizes for the forward model.
            activation (nn.Module, optional): Activation function for the forward model. Defaults to nn.ReLU().
        """
        super().__init__()
        in_dim = obs_dim + action_dim
        layers = []
        for out_dim in layer_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation)
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, obs_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# - helpers
def flatten_obs(obs: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
    """Flatten observation dictionary to a single tensor.

    Args:
        obs (dict): Dictionary of observations or a single observation tensor.

    Returns:
        torch.Tensor: Flattened observation tensor.
    """
    if isinstance(obs, torch.Tensor):
        return obs

    return torch.cat([obs[key].flatten(1) for key in obs.keys()], dim=1)
