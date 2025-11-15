# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualFeedForwardBlock(nn.Module):
    """
    Residual MLP block for SimBA:
      x_{l+1} = x_l + MLP(LN(x_l)),
    where the MLP is a two-layer feedforward network with activation.
    """

    def __init__(self, dim: int, expansion: int = 1, activation=F.relu):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        hidden_dim = expansion * dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN
        out = self.ln(x)
        # Feedforward
        out = self.activation(self.fc1(out))
        out = self.fc2(out)
        # Residual connection
        return x + out


class SimBa(nn.Module):
    """
        Implements the SimBa [1]_ architecture for efficient RL.

        Architecture:
            Input -> Linear -> Residual Block -> ... -> Residual Block -> LN -> Linear -> Output

        References:
            .. [1] Lee et al. "SimBa: Simplicity Bias for Scaling Up Parameters in Deep Reinforcement Learning
    " arXiv preprint https://arxiv.org/abs/2410.09754 (2024).
    """

    is_recurrent = False

    def __init__(
        self,
        obs: torch.Tensor,
        hidden_layers: list[int],
        out_dim: int,
        activation,
        expansion: int = 1,
        **kwargs,
    ):
        super().__init__()

        if kwargs:
            print(f"Simba got unexpected keyword arguments: {kwargs}")

        # 1) Determine total input dimension
        in_dim = obs.flatten(1).shape[1]

        if isinstance(activation, str):
            activation = get_activation(activation)

        # 2) Buffers for running mean/variance (not learned)
        self.register_buffer("running_mean", torch.zeros(in_dim))
        self.register_buffer("running_var", torch.ones(in_dim))
        self.register_buffer("count", torch.zeros((), dtype=torch.float))

        # 3) First linear to map inputs to the hidden dimension
        if not hidden_layers:
            raise ValueError("hidden_layers cannot be empty for SimBa.")
        self.input_dim = in_dim
        self.output_dim = out_dim
        self.initial_linear = nn.Linear(in_dim, hidden_layers[0])

        # 4) Build a stack of residual blocks (with the same hidden size)
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            if hidden_layers[i] != hidden_layers[i + 1]:
                raise ValueError(
                    f"All hidden_layers must match for a residual block, but got {hidden_layers[i]} -> {hidden_layers[i + 1]}."
                )
            self.blocks.append(ResidualFeedForwardBlock(hidden_layers[i], expansion=expansion, activation=activation))

        # 5) Final LN and projection
        self.final_ln = nn.LayerNorm(hidden_layers[-1])
        self.output_linear = nn.Linear(hidden_layers[-1], out_dim)

    def forward_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        x = self._rsnorm(obs)
        return self._forward_core(x)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward_tensor(obs)

    def _forward_core(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_linear(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        x = self.output_linear(x)
        return x

    def _rsnorm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the input tensor x using running statistics (mean and variance) and
        updates the running statistics based on the current batch.
        """
        with torch.no_grad():
            batch_count = x.shape[0]
            if batch_count > 1:
                batch_mean = x.mean(dim=0)
                batch_var = x.var(dim=0, unbiased=False)

                new_count = self.count + batch_count
                delta = batch_mean - self.running_mean

                # Update mean
                self.running_mean += delta * (batch_count / new_count)

                # Update var
                m_a = self.running_var * self.count
                m_b = batch_var * batch_count
                m2 = m_a + m_b + delta.pow(2) * (self.count * batch_count / new_count)
                self.running_var = m2 / new_count
                self.count = new_count

        # Standardize
        return (x - self.running_mean) / torch.sqrt(self.running_var + 1e-5)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "silu":
        return nn.SiLU()
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
