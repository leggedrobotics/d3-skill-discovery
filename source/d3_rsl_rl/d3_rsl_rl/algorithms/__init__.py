# Copyright (c) 2025, ETH Zurich, Rafael Cathomen
#
# SPDX-License-Identifier: BSD-3-Clause

#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .ppo import PPO
from .ppo_og import PPO_OG

__all__ = ["PPO", "PPO_OG"]
