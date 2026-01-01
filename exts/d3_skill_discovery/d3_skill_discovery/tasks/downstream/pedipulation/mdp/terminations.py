# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp import UniformPose2dCommand


def goal_reached(
    env: ManagerBasedRLEnv,
    goal_cmd_name: str,
    distance_threshold: float = 0.5,
    angle_threshold: float = 0.1,
) -> torch.Tensor:
    """Terminate the episode when the goal is reached.
    Args:
        env: The learning environment.
        distance_threshold: The distance threshold to the goal.

    Returns:
        Boolean tensor indicating whether the goal is reached.
    """
    # extract the used quantities (to enable type-hinting)
    goal_cmd_geneator: UniformPose2dCommand = env.command_manager._terms[goal_cmd_name]
    # check for termination
    distance_goal = torch.linalg.vector_norm(goal_cmd_geneator.pos_command_b, dim=1)

    angle_error = torch.abs(goal_cmd_geneator.heading_command_b)
    # Check conditions
    at_goal = (distance_goal < distance_threshold) & (angle_error < angle_threshold)
    return at_goal


def too_far_from_goal(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 50.0,
    goal_cmd_name: str = "robot_goal",
) -> torch.Tensor:
    """Terminate the episode when the robot is too far from the goal.
    Args:
        env: The learning environment.
        distance_threshold: The distance threshold to the goal.

    Returns:
        Boolean tensor indicating whether the robot is too far from the goal.
    """
    # extract the used quantities (to enable type-hinting)
    goal_cmd_geneator: UniformPose2dCommand = env.command_manager._terms[goal_cmd_name]
    # check for termination
    distance_goal = torch.linalg.vector_norm(goal_cmd_geneator.pos_command_b, dim=1)
    # Check conditions
    too_far = distance_goal > distance_threshold
    return too_far
