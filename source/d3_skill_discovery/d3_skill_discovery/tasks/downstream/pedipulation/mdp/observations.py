# Copyright (c) 2025, ETH Zurich, Rafael Cathomen
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.command_manager import CommandManager
from isaaclab.utils.math import quat_rotate_inverse, wrap_to_pi, yaw_quat


def foot_tracking_commands(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Commands are given to the policy in an inertial (non moving) frame, here the environment origin. To transform
    that vector into the robot base frame, we first subtract the vector that points from the env_origin to the robot
    base and then rotate into the robot frame. The asset_cfg is defined in pedipulation environment config file.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command_manager: CommandManager = env.command_manager
    command_w: torch.tensor = command_manager.get_command("foot_position")  # commands given in world frame

    robot_base_pos_w: torch.tensor = asset.data.root_pos_w
    robot_base_quat_q: torch.tensor = asset.data.root_quat_w
    # Transform desired foot position from env_origin in current robot base frame
    desired_foot_pos_b: torch.tensor = quat_rotate_inverse(robot_base_quat_q, command_w - robot_base_pos_w)
    return desired_foot_pos_b


def box_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Get the position of the box in the robot base frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    box_pos_w = asset.data.root_pos_w
    robot_pos_w = env.scene["robot"].data.root_pos_w
    robot_base_quat_q = env.scene["robot"].data.root_quat_w
    # Transform desired foot position from env_origin in current robot base frame
    vec = box_pos_w - robot_pos_w
    box_pos_b = quat_rotate_inverse(yaw_quat(robot_base_quat_q), vec)
    heading = wrap_to_pi(asset.data.heading_w - env.scene["robot"].data.heading_w)
    return torch.cat((box_pos_b, heading.unsqueeze(1)), dim=1)  # (x, y, heading)
