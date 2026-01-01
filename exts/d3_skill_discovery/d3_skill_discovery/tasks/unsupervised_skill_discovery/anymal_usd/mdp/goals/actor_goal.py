# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.envs import ManagerBasedRLEnv


def generated_goal(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Returns the generated goal for the given command.
    Note, the command needs to implement the goal property.
    This goal is not dependent on the current state, but is the goal for conditioning."""
    return env.command_manager._terms[command_name].goal  # type: ignore
