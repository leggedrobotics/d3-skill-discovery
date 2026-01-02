# Copyright (c) 2025, ETH Zurich, Rafael Cathomen
#
# SPDX-License-Identifier: BSD-3-Clause


"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from d3_skill_discovery.tasks.unsupervised_skill_discovery.anymal_usd.mdp.commands import GoalCommand
from d3_skill_discovery.tasks.unsupervised_skill_discovery.anymal_usd.mdp.utils import get_robot_pos

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def too_far_away(
    env: ManagerBasedRLEnv,
    max_dist: float,
    command_name: str = "robot_goal",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate if the robot is too far away from the goal."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject | Articulation = env.scene[asset_cfg.name]
    goal_cmd_geneator: GoalCommand = env.command_manager._terms[command_name]

    robot_pos = get_robot_pos(robot)
    goal_pos = goal_cmd_geneator.goal_pos_w

    diff = torch.linalg.norm(robot_pos - goal_pos, dim=-1)
    return diff > max_dist


def goal_reached(
    env: ManagerBasedRLEnv,
    threshold_dist: float,
    command_name: str = "robot_goal",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate if the robot is at the goal."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject | Articulation = env.scene[asset_cfg.name]
    goal_cmd_geneator: GoalCommand = env.command_manager._terms[command_name]

    robot_pos = get_robot_pos(robot)
    goal_pos = goal_cmd_geneator.goal_pos_w

    diff = torch.linalg.norm(robot_pos - goal_pos, dim=-1)
    return diff < threshold_dist
