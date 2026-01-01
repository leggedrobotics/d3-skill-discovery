# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

#  Copyright (c) 2020 Preferred Networks, Inc.
#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, eps=1e-2, until=None):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the link learns input values until the sum of batch sizes
            exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.until = until

        self.is_dict = isinstance(shape, dict)
        if self.is_dict:
            self._mean = {key: torch.zeros(tensor.shape[1:]).unsqueeze(0) for key, tensor in shape.items()}
            self._var = {key: torch.ones(tensor.shape[1:]).unsqueeze(0) for key, tensor in shape.items()}
            self._std = {key: torch.ones(tensor.shape[1:]).unsqueeze(0) for key, tensor in shape.items()}
        else:
            self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
            self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
            self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.count = 0

    def state_dict(self):
        return {
            "_mean": self._mean,
            "_var": self._var,
            "_std": self._std,
            "count": self.count,
        }

    def to(self, device):
        if self.is_dict:
            self._mean = {key: tensor.to(device) for key, tensor in self._mean.items()}
            self._var = {key: tensor.to(device) for key, tensor in self._var.items()}
            self._std = {key: tensor.to(device) for key, tensor in self._std.items()}
        else:
            self._mean = self._mean.to(device)
            self._var = self._var.to(device)
            self._std = self._std.to(device)
        return self

    def load_state_dict(self, state_dict):
        self._mean = state_dict["_mean"]
        self._var = state_dict["_var"]
        self._std = state_dict["_std"]
        self.count = state_dict["count"]

    @property
    def mean(self):
        if self.is_dict:
            return {key: tensor.squeeze(0).clone() for key, tensor in self._mean.items()}
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        if self.is_dict:
            return {key: tensor.squeeze(0).clone() for key, tensor in self._std.items()}
        return self._std.squeeze(0).clone()

    def normalize(self, x):
        if self.is_dict:
            return {key: (tensor - self._mean[key]) / (self._std[key] + self.eps) for key, tensor in x.items()}
        return (x - self._mean) / (self._std + self.eps)

    def forward(self, x):
        """Normalize mean and variance of values based on empirical values.

        Args:
            x (ndarray or Variable): Input values

        Returns:
            ndarray or Variable: Normalized output values
        """

        if self.training:
            self.update(x)
        return self.normalize(x)

    @torch.jit.unused
    def update(self, x):
        """Learn input values without computing the output values of them"""

        if self.until is not None and self.count >= self.until:
            return
        if self.is_dict:
            count_x = x[list(x.keys())[0]].shape[0]
        else:
            count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count

        if self.is_dict:
            for key in x.keys():
                var_x = torch.var(x[key], dim=0, unbiased=False, keepdim=True)
                mean_x = torch.mean(x[key], dim=0, keepdim=True)
                delta_mean = mean_x - self._mean[key]
                self._mean[key] += rate * delta_mean
                self._var[key] += rate * (var_x - self._var[key] + delta_mean * (mean_x - self._mean[key]))
                self._std[key] = torch.sqrt(self._var[key])
        else:
            var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
            mean_x = torch.mean(x, dim=0, keepdim=True)
            delta_mean = mean_x - self._mean
            self._mean += rate * delta_mean
            self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
            self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean


class ExponentialMovingAverageNormalizer:
    r"""
    Normalizer with exponential moving averages (EMA) for mean and variance.

        mean_{t+1} = decay_factor * mean_t + (1 - decay_factor) * mean(x)
        var_{t+1}  = decay_factor * var_t  + (1 - decay_factor) * var(x)

    Args:
        decay_factor (float): EMA decay factor in [0,1].
                              Smaller -> faster updates,
                              Larger -> more smoothing (slower to change).
        shape (int): Dimensionality of the data (excluding batch axis).
        eps (float): Small constant added to std to avoid division by zero.
    """

    def __init__(
        self,
        decay_factor: float,
        shape: int,
        device: str,
        eps: float = 1e-5,
        train_mode: bool = True,
        until: int | None = None,
    ):
        # Decay factor for exponential moving average; must be in [0,1]
        self.decay_factor = decay_factor
        self.eps = eps

        # Initialize mean and var as [1, shape] for easy broadcasting
        self.mean = torch.zeros(shape).unsqueeze(0).to(device)
        self.var = torch.ones(shape).unsqueeze(0).to(device)
        self.train_mode = train_mode

        # Track how many samples we've processed (useful if you want to do any burn-in logic)
        self.count = 0
        self.until = until

    def train(self, mode: bool):
        self.train_mode = mode

    def update(self, x: torch.Tensor):
        """
        x must have shape [batch_size, shape].
        """

        if self.count == 0:
            # First update: set mean/var to that of x
            self.mean = x.mean(dim=0, keepdim=True)
            self.var = x.var(dim=0, unbiased=False, keepdim=True) + 1.0
            self.first_call = False
        else:
            # Clamp x to avoid numerical instability
            x = torch.clamp(x, min=self.mean - 100 * torch.sqrt(self.var), max=self.mean + 100 * torch.sqrt(self.var))

            # Compute batch statistics
            mean_x = x.mean(dim=0, keepdim=True)
            var_x = x.var(dim=0, unbiased=False, keepdim=True)

            # EMA updates
            self.mean = (self.decay_factor) * self.mean + (1 - self.decay_factor) * mean_x
            self.var = (self.decay_factor) * self.var + (1 - self.decay_factor) * var_x

        self.count += 1

        # Stop training if we've reached our sample limit
        if self.until is not None and self.count >= self.until:
            self.train_mode = False

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.mean) / (torch.sqrt(self.var) + self.eps), min=-25, max=25)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input x using the current EMA statistics.
        Update the statistics if in training mode.
        Returns normalized x of shape [batch_size, shape].
        """

        if self.train_mode:
            self.update(x)

        return self.normalize(x)


class EmpiricalDiscountedVariationNormalization(nn.Module):
    """Reward normalization from Pathak's large scale study on PPO.

    Reward normalization. Since the reward function is non-stationary, it is useful to normalize
    the scale of the rewards so that the value function can learn quickly. We did this by dividing
    the rewards by a running estimate of the standard deviation of the sum of discounted rewards.
    """

    def __init__(self, shape, eps=1e-2, gamma=0.99, until=None):
        super().__init__()

        self.emp_norm = EmpiricalNormalization(shape, eps, until)
        self.disc_avg = DiscountedAverage(gamma)

    def forward(self, rew):
        if self.training:
            # update discounected rewards
            avg = self.disc_avg.update(rew)

            # update moments from discounted rewards
            self.emp_norm.update(avg)

        return self.normalize(rew)

    def normalize(self, rew):
        if self.emp_norm._std > 0:
            return rew / self.emp_norm._std
        else:
            return rew


class DiscountedAverage:
    r"""Discounted average of rewards.

    The discounted average is defined as:

    .. math::

        \bar{R}_t = \gamma \bar{R}_{t-1} + r_t

    Args:
        gamma (float): Discount factor.
    """

    def __init__(self, gamma):
        self.avg = None
        self.gamma = gamma

    def update(self, rew: torch.Tensor) -> torch.Tensor:
        if self.avg is None:
            self.avg = rew
        else:
            self.avg = self.avg * self.gamma + rew
        return self.avg


class DictFlattener(nn.Module):
    def __init__(self, model: nn.Module):
        """
        A wrapper that flattens a dictionary of tensors into a single tensor
        before passing it to the wrapped model.

        Args:
            model (nn.Module): The model that accepts a flat tensor input.
        """
        super().__init__()
        self.model = model

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass. Converts a dict of tensors to a flat tensor.

        Args:
            x (dict[str, torch.Tensor]): Dictionary of tensors to be flattened and concatenated.

        Returns:
            torch.Tensor: Output from the wrapped model.
        """

        flat_tensor = torch.cat([x[key].flatten(1) for key in sorted(x.keys())], dim=-1)

        return self.model(flat_tensor)

    def __repr__(self):
        return f"DictFlattener({self.model})"
