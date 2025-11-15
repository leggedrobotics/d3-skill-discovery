# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from d3_skill_discovery.tasks.downstream.pedipulation.mdp.commands import FootBasePositionCommand

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.commands import UniformPose2dCommand
from isaaclab.managers import CommandManager, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_from_euler_xyz, quat_rotate_inverse, wrap_to_pi, yaw_quat


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def moving_towards_goal(
    env: ManagerBasedRLEnv, command_name: str, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for moving towards the goal. This reward is computed as the dot product of the velocity of the robot
    and the unit vector pointing towards the goal."""
    robot: Articulation = env.scene[robot_cfg.name]
    robot_vel_w = robot.data.root_lin_vel_w
    robot_pos = robot.data.root_pos_w

    goal_command: FootBasePositionCommand | UniformPose2dCommand = env.command_manager._terms[command_name]
    if isinstance(goal_command, FootBasePositionCommand):
        goal_pos_w = goal_command.base_pos_command_w
    else:
        goal_pos_w = goal_command.pos_command_w

    to_goal_vec = goal_pos_w - robot_pos
    to_goal_vec = to_goal_vec / (torch.linalg.norm(to_goal_vec, dim=-1, keepdim=True) + 1e-6)

    vel_to_goal = torch.linalg.vecdot(robot_vel_w, to_goal_vec, dim=-1)
    reward = torch.clamp(vel_to_goal, -1, 1)
    return reward


def close_to_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for being close to the goal"""

    robot: Articulation = env.scene[robot_cfg.name]
    robot_pos = robot.data.root_pos_w

    goal_command: FootBasePositionCommand | UniformPose2dCommand = env.command_manager._terms[command_name]
    if isinstance(goal_command, FootBasePositionCommand):
        goal_pos_w = goal_command.base_pos_command_w
    else:
        goal_pos_w = goal_command.pos_command_w

    dist_to_goal = torch.linalg.vector_norm(robot_pos - goal_pos_w, dim=-1)

    reward = torch.exp(-dist_to_goal / sigma)

    return reward


def correct_heading_if_close(
    env: ManagerBasedRLEnv,
    command_name: str,
    dist_threshold: float = 1.0,
    sigma: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for heading in the right direction at the goal.
    Only given if the robot is close enough to the goal."""

    robot: Articulation = env.scene[robot_cfg.name]
    robot_pos = robot.data.root_pos_w

    goal_command: FootBasePositionCommand | UniformPose2dCommand = env.command_manager._terms[command_name]
    if isinstance(goal_command, FootBasePositionCommand):
        goal_pos_w = goal_command.base_pos_command_w
    else:
        goal_pos_w = goal_command.pos_command_w

    heading_error = goal_command.heading_command_b.abs()

    close_enough = torch.linalg.vector_norm(robot_pos - goal_pos_w, dim=-1) < dist_threshold

    reward = torch.where(close_enough, torch.exp(-heading_error / sigma), torch.zeros_like(heading_error))
    return reward


def foot_tracking(
    env: ManagerBasedRLEnv, sigma: float, asset_cfg: SceneEntityCfg
):  # TODO: here and below, we should switch to the ManagerBasedRLEnv. We don't have a ManagerBasedRLEnv though, just the cfg. Maybe it's worth adding one just for pylance?
    """
    Commands are given in an inertial (non moving) frame, here the environment origin. To transform that vector into
    the robot base frame, we subtract the vector that points from the env_origin to the robot base. The asset_cfg is
    defined in pedipulation environment config file.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    command_manager: CommandManager = env.command_manager  # commands given in world frame
    if asset_cfg.body_ids is None:
        raise ValueError("The body_ids of the robot are not defined in the environment config.")
    body_ids: list[int] = asset_cfg.body_ids[env.cfg.foot_index]
    foot_pos_w = asset.data.body_state_w[:, body_ids, :3]
    desired_foot_pos_w = command_manager.get_command("foot_position")  # commands given in world frame
    return torch.exp(-torch.norm(foot_pos_w - desired_foot_pos_w, dim=1) / sigma)


def foot_tracking_b(
    env: ManagerBasedRLEnv,
    sigma: float,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    dist_threshold: float,
    angle_threshold: float,
):  # TODO: here and below, we should switch to the ManagerBasedRLEnv. We don't have a ManagerBasedRLEnv though, just the cfg. Maybe it's worth adding one just for pylance?
    """
    Commands are given in an inertial (non moving) frame, here the environment origin. To transform that vector into
    the robot base frame, we subtract the vector that points from the env_origin to the robot base. The asset_cfg is
    defined in pedipulation environment config file.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    command: FootBasePositionCommand = env.command_manager._terms[command_name]
    if asset_cfg.body_ids is None:
        raise ValueError("The body_ids of the robot are not defined in the environment config.")
    body_ids: list[int] = asset_cfg.body_ids[env.cfg.foot_index]

    foot_pos_w = asset.data.body_state_w[:, body_ids, :3]
    foot_pos_b = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), foot_pos_w - asset.data.root_pos_w[:, :3])

    desired_foot_pos_b = command.foot_pos_commands_b

    # check if at goal:
    close_enough = torch.linalg.vector_norm(asset.data.root_pos_w - command.base_pos_command_w, dim=-1) < dist_threshold
    aligned_enough = torch.abs(wrap_to_pi(command.heading_command_b - asset.data.root_quat_w[:, 2])) < angle_threshold

    reward = torch.where(
        close_enough & aligned_enough, torch.exp(-torch.norm(foot_pos_b - desired_foot_pos_b, dim=1) / sigma), 0.0
    )

    return reward
