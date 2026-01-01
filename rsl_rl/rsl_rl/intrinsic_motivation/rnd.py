# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from collections import deque

from rsl_rl.modules.normalizer import EmpiricalDiscountedVariationNormalization  # noqa: F401
from rsl_rl.modules.normalizer import EmpiricalNormalization, ExponentialMovingAverageNormalizer
from rsl_rl.utils import extract_batch_shape, flatten_batch, unflatten_batch


class RNDStateMLP(nn.Module):
    """Defines a simple multi-layer perceptron (MLP) network."""

    is_recurrent = False

    def __init__(
        self,
        state: dict[str, torch.Tensor] | torch.Tensor,
        hidden_layers: list[int],
        latent_dim: int,
        activation,
        layer_norm: bool = False,
    ):
        super().__init__()

        # state dim
        if isinstance(state, dict):
            in_dim = 0
            self.state_dim = {}
            for key, tensor in state.items():
                in_dim += tensor.flatten(1).shape[-1]
                self.state_dim[key] = tensor.flatten(1).shape[1:]

        else:
            in_dim = state.flatten(1).shape[-1]
            self.state_dim = state.flatten(1).shape[1:]

        layers = []
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation)
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, state: dict[str, torch.Tensor] | torch.Tensor):
        batch_dim = extract_batch_shape(state, self.state_dim)
        state = flatten_batch(state, self.state_dim)
        if isinstance(state, dict):
            state = torch.cat([state[key].flatten(1) for key in sorted(state.keys())], dim=-1)
        latent = self.mlp(state)

        return unflatten_batch(latent, batch_dim)


class RandomNetworkDistillation(nn.Module):
    """Implementation of Random Network Distillation (RND) [1]_

    References:
        .. [1] Burda, Yuri, et al. "Exploration by random network distillation." arXiv preprint arXiv:1810.12894 (2018).
    """

    def __init__(
        self,
        input_sample: torch.Tensor | dict[str, torch.Tensor],
        num_outputs: int,
        predictor_hidden_dims: list[int],
        target_hidden_dims: list[int],
        activation: str = "elu",
        weight: float = 0.0,
        reward_normalization: bool = False,
        state_normalization: bool = False,
        device: str = "cpu",
        lr: float = 1e-4,
        perturb_target: bool = False,
        target_net_perturbation_scale: float = 0.001,
        target_net_perturbation_interval: int = 250,
        max_param_norm: float = 10.0,
        **kwargs,
    ):
        """Initialize the RND module.

        - If :attr:`state_normalization` is True, then the input state is normalized using an Empirical Normalization layer.
        - If :attr:`reward_normalization` is True, then the intrinsic reward is normalized using an Empirical Discounted
          Variation Normalization layer.

        .. note::
            If the hidden dimensions are -1 in the predictor and target networks configuration, then the number of states
            is used as the hidden dimension.

        Args:
            input_sample: Sample input to determine the number of states.
            num_outputs: Number of outputs (embedding size) of the predictor and target networks.
            predictor_hidden_dims: List of hidden dimensions of the predictor network.
            target_hidden_dims: List of hidden dimensions of the target network.
            activation: Activation function. Defaults to "elu".
            weight: Scaling factor of the intrinsic reward. Defaults to 0.0.
            reward_normalization: Whether to normalize the intrinsic reward. Defaults to False.
            state_normalization: Whether to normalize the input state. Defaults to False.
            lr: Learning rate of the predictor network. Defaults to 1e-4.
            perturb_target: Whether to perturb the target network. Defaults to False.
            target_net_perturbation_scale: Standard deviation of the Gaussian noise added to the target network. Defaults to 0.0001.
            target_net_perturbation_interval: Interval of perturbation of the target network. Defaults to 10.
            max_param_norm: Maximum parameter norm of the target network. Defaults to 10.0.
            device: Device to use. Defaults to "cpu".

        Keyword Args:

            max_num_steps (int): Maximum number of steps per episode. Used for the weight schedule of type "step".
            final_value (float): Final value of the weight parameter. Used for the weight schedule of type "step".
        """
        super().__init__()

        # Store parameters
        self.weight = weight
        self.device = device
        self.reward_normalization = reward_normalization
        # Normalization of intrinsic reward
        if reward_normalization:
            # self.reward_normalizer = EmpiricalDiscountedVariationNormalization(shape=[], until=1.0e8).to(self.device)
            self.reward_normalizer = ExponentialMovingAverageNormalizer(decay_factor=0.995, shape=1, device=device)
        else:
            self.reward_normalizer = torch.nn.Identity()
        if state_normalization:
            self.state_normalizer = EmpiricalNormalization(input_sample).to(device)
        else:

            class DummyNormalizer(nn.Module):
                normalize = staticmethod(torch.nn.Identity())

                def forward(self, x):
                    return x

            self.state_normalizer = DummyNormalizer()

        # Counter for the number of updates
        self.update_counter = 0

        # Create network architecture
        self.predictor = RNDStateMLP(input_sample, predictor_hidden_dims, num_outputs, get_activation(activation)).to(
            device
        )
        self.target = RNDStateMLP(input_sample, target_hidden_dims, num_outputs, get_activation(activation)).to(device)
        # freeze target network
        self.target.requires_grad_(False)
        self.target.eval()

        # optimizer for the predictor network
        self.predictor_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)

        # perturbation of the target network
        self.target_net_perturbation = perturb_target
        self.target_net_perturbation_scale = target_net_perturbation_scale
        self.target_net_perturbation_interval = target_net_perturbation_interval
        self.max_param_norm = max_param_norm

        # logging
        self.log_reward_mean = deque(maxlen=25)
        self.log_reward_std = deque(maxlen=25)
        self.log_loss = deque(maxlen=25)

    def get_intrinsic_reward(self, rnd_state: dict[str, torch.Tensor]) -> torch.Tensor:
        # note: the counter is updated number of env steps per learning iteration
        self.update_counter += 1
        # Obtain the embedding of the gated state from the target and predictor networks
        rnd_state = self.state_normalizer.normalize(rnd_state)
        target_embedding = self.target(rnd_state).detach()
        predictor_embedding = self.predictor(rnd_state).detach()
        # Compute the intrinsic reward as the distance between the embeddings
        intrinsic_reward = torch.linalg.norm(target_embedding - predictor_embedding, dim=1)
        # Normalize intrinsic reward
        intrinsic_reward = self.reward_normalizer(intrinsic_reward) * self.weight
        # Log intrinsic reward
        self.log_reward_mean.append(intrinsic_reward.mean().item())
        self.log_reward_std.append(intrinsic_reward.std().item())
        return intrinsic_reward

    def forward(self, *args, **kwargs):
        raise RuntimeError("Forward method is not implemented. Use get_intrinsic_reward instead.")

    def train(self, mode: bool = True):
        # sets module into training mode
        self.predictor.train(mode)
        if self.state_normalization:
            self.state_normalizer.train(mode)
        if self.reward_normalization:
            self.reward_normalizer.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def update(self, rnd_state_batch: dict[str, torch.Tensor] | torch.Tensor) -> None:
        """Update the predictor network.
        Args:
            rnd_state_batch: Batch of states to update the target network.
        """
        self.update_counter += 1

        rnd_state_batch = self.state_normalizer(rnd_state_batch)

        target_embedding = self.target(rnd_state_batch).detach()
        predictor_embedding = self.predictor(rnd_state_batch)

        loss = torch.nn.functional.mse_loss(predictor_embedding, target_embedding)

        self.predictor_optimizer.zero_grad()
        loss.backward()
        self.predictor_optimizer.step()

        # perturb target network
        if self.target_net_perturbation and self.update_counter % self.target_net_perturbation_interval == 0:
            self.perturb_target()

        # logging
        self.log_loss.append(loss.item())

    def perturb_target(self):
        """Perturb the target network with Gaussian noise.
        This is useful to prevent the policy from forgetting old states.

        Args:
            noise: Standard deviation of the Gaussian noise.
        """
        for param in self.target.parameters():
            param.data += torch.randn_like(param) * self.target_net_perturbation_scale
            param.data = torch.clamp(param.data, -self.max_param_norm, self.max_param_norm)

    def get_metrics(self):
        metrics = {
            "intrinsic_reward": torch.mean(torch.tensor(self.log_reward_mean)).item(),
            "intrinsic_reward_std": torch.mean(torch.tensor(self.log_reward_std)).item(),
            "loss": torch.mean(torch.tensor(self.log_loss)).item(),
        }

        if self.reward_normalization:
            metrics["reward_normalizer_std"] = self.reward_normalizer.var.sqrt().item()
            metrics["reward_normalizer_mean"] = self.reward_normalizer.mean.item()

        return metrics


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
