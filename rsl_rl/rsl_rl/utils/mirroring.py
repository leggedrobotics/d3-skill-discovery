# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
from typing import Callable, Literal

# symmetry augmentation for anymal
# symmetries in body and world frame are used independently


##
# symmetry augmentation for anymal
##
class augment_anymal_obs:
    def __init__(self, scene_cfg=None) -> None:
        """Augments the observation and actions of the Anymal-D robot with symmetries.
        Adds left-right, front-back and 180 degree rotation symmetries."""
        if scene_cfg is None or not hasattr(scene_cfg, "height_scanner"):
            self.mirror_height_scan_left_right: Callable = None  # type: ignore
            self.mirror_height_scan_front_back: Callable = None  # type: ignore
            self.rotate_height_scan_180: Callable = None  # type: ignore
        else:
            self.mirror_height_scan_left_right = mirror_height_scan_left_right(scene_cfg.height_scanner.pattern_cfg)
            self.mirror_height_scan_front_back = mirror_height_scan_front_back(scene_cfg.height_scanner.pattern_cfg)
            self.rotate_height_scan_180 = rotate_height_scan_180(scene_cfg.height_scanner.pattern_cfg)

    def __call__(self, obs_dict: dict[str, torch.Tensor], ignore_keys: list[str] = []) -> dict[str, torch.Tensor]:
        """
        Args:
            obs_dict (dict): Dictionary of observations.
            ignore_keys (list): List of keys to ignore for augmentation. These keys will be repeated
                instead of augmented.
        Returns:
            dict: Augmented observations.
        """
        augmented_obs = {}
        for obs_name, obs in obs_dict.items():
            if obs_name in ["origin", "box_pose"]:
                if obs.shape[-1] == 5:
                    lr_obs = mirror_origin_left_right(obs)
                    fb_obs = mirror_origin_front_back(obs)
                    rot_obs = rotate_origin_180(obs)
                else:
                    lr_obs = mirror_origin_2d_left_right(obs)
                    fb_obs = mirror_origin_2d_front_back(obs)
                    rot_obs = rotate_origin_2d_180(obs)
                # no augmentation because origin is in world frame (heading would need to be mirrored)
                augmented_obs[obs_name] = torch.cat([obs, lr_obs, fb_obs, rot_obs], dim=0)
            elif obs_name == "heading":
                lr_obs = mirror_origin_2d_left_right(obs)
                fb_obs = mirror_origin_2d_front_back(obs)
                rot_obs = rotate_origin_2d_180(obs)
                # no augmentation because origin is in world frame (heading would need to be mirrored)
                augmented_obs[obs_name] = torch.cat([obs, lr_obs, fb_obs, rot_obs], dim=0)
            elif obs_name == "height_scan":
                lr_obs = self.mirror_height_scan_left_right(obs)
                fb_obs = self.mirror_height_scan_front_back(obs)
                rot_obs = self.rotate_height_scan_180(obs)
                augmented_obs[obs_name] = torch.cat([obs, lr_obs, fb_obs, rot_obs], dim=0)
            elif obs_name == "base_lin_vel":
                lr_obs = mirror_base_lin_vel_left_right(obs)
                fb_obs = mirror_base_lin_vel_front_back(obs)
                rot_obs = rotate_base_lin_vel_180(obs)
                augmented_obs[obs_name] = torch.cat([obs, lr_obs, fb_obs, rot_obs], dim=0)
            elif obs_name == "base_ang_vel":
                lr_obs = mirror_base_ang_vel_left_right(obs)
                fb_obs = mirror_base_ang_vel_front_back(obs)
                rot_obs = rotate_base_ang_vel_180(obs)
                augmented_obs[obs_name] = torch.cat([obs, lr_obs, fb_obs, rot_obs], dim=0)
            elif obs_name == "projected_gravity":
                lr_obs = mirror_projected_gravity_left_right(obs)
                fb_obs = mirror_projected_gravity_front_back(obs)
                rot_obs = rotate_projected_gravity_180(obs)
                augmented_obs[obs_name] = torch.cat([obs, lr_obs, fb_obs, rot_obs], dim=0)
            elif obs_name == "foot_pos":
                lr_obs = mirror_foot_pos_left_right(obs)
                fb_obs = mirror_foot_pos_front_back(obs)
                rot_obs = rotate_foot_pos_gravity_180(obs)
                augmented_obs[obs_name] = torch.cat([obs, lr_obs, fb_obs, rot_obs], dim=0)
            elif obs_name in ["actions", "joint_pos", "joint_vel"]:
                lr_obs = mirror_joints_left_right(obs)
                fb_obs = mirror_joints_front_back(obs)
                rot_obs = rotate_180_joints(obs)
                augmented_obs[obs_name] = torch.cat([obs, lr_obs, fb_obs, rot_obs], dim=0)
            elif obs_name == "lidar_scan":
                lr_obs = mirror_lidar_scan_left_right(obs)
                fb_obs = mirror_lidar_scan_front_back(obs)
                rot_obs = rotate_lidar_scan_180(obs)
                augmented_obs[obs_name] = torch.cat([obs, lr_obs, fb_obs, rot_obs], dim=0)
            elif "bbox" in obs_name:
                lr_obs = mirror_object_bbox_left_right(obs)
                fb_obs = mirror_object_bbox_front_back(obs)
                rot_obs = rotate_object_bbox_180(obs)
                augmented_obs[obs_name] = torch.cat([obs, lr_obs, fb_obs, rot_obs], dim=0)
            elif obs_name == "roll_pitch":
                lr_obs = mirror_base_roll_pitch_left_right(obs)
                fb_obs = mirror_base_roll_pitch_front_back(obs)
                rot_obs = rotate_base_roll_pitch_180(obs)
                augmented_obs[obs_name] = torch.cat([obs, lr_obs, fb_obs, rot_obs], dim=0)
            elif obs_name in ["heading_rate", "heading_cumulative"]:
                lr_obs = mirror_base_heading_rate_left_right(obs)
                fb_obs = mirror_base_heading_rate_front_back(obs)
                rot_obs = rotate_base_heading_rate_180(obs)
                augmented_obs[obs_name] = torch.cat([obs, lr_obs, fb_obs, rot_obs], dim=0)
            elif obs_name == "skill":
                # skills are augmented in the skill discovery module
                continue
            elif obs_name in (["time_left", "base_height", "is_active"] + ignore_keys):
                # non-augmented observations are just repeated
                augmented_obs[obs_name] = obs.repeat(4, 1)
            else:
                raise ValueError(f"Unknown observation name: {obs_name}, No symmetry augmentation implemented.")
        return augmented_obs


def augment_anymal_action(action: torch.Tensor) -> torch.Tensor:
    """Augments the action with symmetries.
    Adds left-right, front-back and 180 degree rotation symmetries to the action."""
    og_action = action
    lr_action = mirror_joints_left_right(og_action)
    fb_action = mirror_joints_front_back(og_action)
    rot_action = rotate_180_joints(og_action)
    augmented_action = torch.cat([og_action, lr_action, fb_action, rot_action], dim=0)
    return augmented_action


##
# low-level transforms
##

# - body


# height scan
class mirror_height_scan_left_right:
    def __init__(self, grid_pattern_cfg) -> None:
        self.nx = 1 + int(np.round(grid_pattern_cfg.size[0] / grid_pattern_cfg.resolution))
        self.ny = 1 + int(np.round(grid_pattern_cfg.size[1] / grid_pattern_cfg.resolution))

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        mirrored_obs = obs.clone()
        mirrored_obs = mirrored_obs.view(-1, self.ny, self.nx).flip(dims=[1]).view(-1, self.nx * self.ny)
        return mirrored_obs


class mirror_height_scan_front_back:
    def __init__(self, grid_pattern_cfg) -> None:
        self.nx = 1 + int(np.round(grid_pattern_cfg.size[0] / grid_pattern_cfg.resolution))
        self.ny = 1 + int(np.round(grid_pattern_cfg.size[1] / grid_pattern_cfg.resolution))

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        mirrored_obs = obs.clone()
        mirrored_obs = mirrored_obs.view(-1, self.ny, self.nx).flip(dims=[2]).view(-1, self.nx * self.ny)
        return mirrored_obs


class rotate_height_scan_180:
    def __init__(self, grid_pattern_cfg) -> None:
        self.nx = 1 + int(np.round(grid_pattern_cfg.size[0] / grid_pattern_cfg.resolution))
        self.ny = 1 + int(np.round(grid_pattern_cfg.size[1] / grid_pattern_cfg.resolution))

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        mirrored_obs = obs.clone()
        mirrored_obs = mirrored_obs.view(-1, self.ny, self.nx).flip(dims=[1, 2]).view(-1, self.nx * self.ny)
        return mirrored_obs


# lidar scan (assumes 1d 360 degree scan with even spacing and even number of points)
def mirror_lidar_scan_left_right(obs: torch.Tensor) -> torch.Tensor:
    # obs = 360 degree scan
    N = obs.shape[1]
    indices = torch.arange(N, device=obs.device)
    mirrored_indices = (N - indices) % N

    mirrored = obs[:, mirrored_indices].clone()
    return mirrored


def mirror_lidar_scan_front_back(obs: torch.Tensor) -> torch.Tensor:
    # obs = 360 degree scan
    num_scans, N = obs.shape
    indices = torch.arange(N, device=obs.device)
    mirrored_indices = (N // 2 - indices) % N
    mirrored = obs[:, mirrored_indices].clone()
    return mirrored


def rotate_lidar_scan_180(obs: torch.Tensor) -> torch.Tensor:
    # obs.shape = [num_scans, N]
    num_scans, N = obs.shape
    indices = torch.arange(N, device=obs.device)
    rotated_indices = (indices + N // 2) % N
    rotated = obs[:, rotated_indices].clone()
    return rotated


# base lin vel
def mirror_base_lin_vel_left_right(obs: torch.Tensor):
    # obs = x,y,z velocity
    mirrored_obs = obs.clone()
    mirrored_obs[..., 1] = -obs[..., 1]
    return mirrored_obs


def mirror_base_lin_vel_front_back(obs: torch.Tensor):
    # obs = x,y,z velocity
    mirrored_obs = obs.clone()
    mirrored_obs[..., 0] = -obs[..., 0]
    return mirrored_obs


def rotate_base_lin_vel_180(obs: torch.Tensor):
    # obs = x,y,z velocity
    mirrored_obs = obs.clone()
    mirrored_obs[..., :2] = -obs[..., :2]
    return mirrored_obs


# heading rate
def mirror_base_heading_rate_left_right(obs: torch.Tensor):
    # obs = angular velocity in z (scalar)
    return -obs.clone()


def mirror_base_heading_rate_front_back(obs: torch.Tensor):
    # obs = angular velocity in z (scalar)
    return -obs.clone()


def rotate_base_heading_rate_180(obs: torch.Tensor):
    # obs = angular velocity in z (scalar)
    return obs.clone()


# roll pitch
def mirror_base_roll_pitch_left_right(obs: torch.Tensor):
    # obs = roll, pitch
    mirrored_obs = obs.clone()
    mirrored_obs[..., 0] = -obs[..., 0]
    return mirrored_obs


def mirror_base_roll_pitch_front_back(obs: torch.Tensor):
    # obs = roll, pitch
    mirrored_obs = obs.clone()
    mirrored_obs[..., 1] = -obs[..., 1]
    return mirrored_obs


def rotate_base_roll_pitch_180(obs: torch.Tensor):
    # obs = roll, pitch
    mirrored_obs = -obs.clone()
    return mirrored_obs


# base ang vel
def mirror_base_ang_vel_left_right(obs: torch.Tensor):
    # obs = x,y,z angular velocity
    mirrored_obs = obs.clone()
    mirrored_obs[..., 0] = -obs[..., 0]
    mirrored_obs[..., 2] = -obs[..., 2]
    return mirrored_obs


def mirror_base_ang_vel_front_back(obs: torch.Tensor):
    # obs = x,y,z angular velocity
    mirrored_obs = obs.clone()
    mirrored_obs[..., 1] = -obs[..., 1]
    mirrored_obs[..., 2] = -obs[..., 2]
    return mirrored_obs


def rotate_base_ang_vel_180(obs: torch.Tensor):
    # obs = x,y,z angular velocity
    mirrored_obs = obs.clone()
    mirrored_obs[..., :2] = -obs[..., :2]
    return mirrored_obs


# projected gravity
def mirror_projected_gravity_left_right(obs: torch.Tensor):
    # obs = x,y,z projected gravity
    mirrored_obs = obs.clone()
    mirrored_obs[..., 1] = -obs[..., 1]
    return mirrored_obs


def mirror_projected_gravity_front_back(obs: torch.Tensor):
    # obs = x,y,z projected gravity
    mirrored_obs = obs.clone()
    mirrored_obs[..., 0] = -obs[..., 0]
    return mirrored_obs


def rotate_projected_gravity_180(obs: torch.Tensor):
    # obs = x,y,z projected gravity
    mirrored_obs = obs.clone()
    mirrored_obs[..., :2] = -obs[..., :2]
    return mirrored_obs


# joints (angles, velocities, actions)
@torch.jit.script
def mirror_joints_left_right(obs: torch.Tensor):
    # obs = joint angles, velocities, actions, 12 dim
    mirrored_obs = obs.clone()
    # front left <-> front right
    mirrored_obs[..., 0] = -obs[..., 2]
    mirrored_obs[..., 2] = -obs[..., 0]
    mirrored_obs[..., 4] = obs[..., 6]
    mirrored_obs[..., 6] = obs[..., 4]
    mirrored_obs[..., 8] = obs[..., 10]
    mirrored_obs[..., 10] = obs[..., 8]

    # back left <-> back right
    mirrored_obs[..., 1] = -obs[..., 3]
    mirrored_obs[..., 3] = -obs[..., 1]
    mirrored_obs[..., 5] = obs[..., 7]
    mirrored_obs[..., 7] = obs[..., 5]
    mirrored_obs[..., 9] = obs[..., 11]
    mirrored_obs[..., 11] = obs[..., 9]
    return mirrored_obs


@torch.jit.script
def mirror_joints_front_back(obs: torch.Tensor):
    # obs = joint angles, velocities, actions, 12 dim
    mirrored_obs = obs.clone()
    # front left <-> back left
    mirrored_obs[..., 0] = obs[..., 1]
    mirrored_obs[..., 1] = obs[..., 0]
    mirrored_obs[..., 4] = -obs[..., 5]
    mirrored_obs[..., 5] = -obs[..., 4]
    mirrored_obs[..., 8] = -obs[..., 9]
    mirrored_obs[..., 9] = -obs[..., 8]

    # front right <-> back right
    mirrored_obs[..., 2] = obs[..., 3]
    mirrored_obs[..., 3] = obs[..., 2]
    mirrored_obs[..., 6] = -obs[..., 7]
    mirrored_obs[..., 7] = -obs[..., 6]
    mirrored_obs[..., 10] = -obs[..., 11]
    mirrored_obs[..., 11] = -obs[..., 10]
    return mirrored_obs


@torch.jit.script
def rotate_180_joints(obs: torch.Tensor):
    # obs = joint angles, velocities, actions, 12 dim
    mirrored_obs = obs.clone()
    # front left <-> back right
    mirrored_obs[..., 0] = -obs[..., 3]
    mirrored_obs[..., 3] = -obs[..., 0]
    mirrored_obs[..., 4] = -obs[..., 7]
    mirrored_obs[..., 7] = -obs[..., 4]
    mirrored_obs[..., 8] = -obs[..., 11]
    mirrored_obs[..., 11] = -obs[..., 8]

    # front right <-> back left
    mirrored_obs[..., 1] = -obs[..., 2]
    mirrored_obs[..., 2] = -obs[..., 1]
    mirrored_obs[..., 5] = -obs[..., 6]
    mirrored_obs[..., 6] = -obs[..., 5]
    mirrored_obs[..., 9] = -obs[..., 10]
    mirrored_obs[..., 10] = -obs[..., 9]
    return mirrored_obs


def mirror_foot_pos_left_right(obs: torch.Tensor):
    # obs = x,y,z for each foot
    obs_unflattened = obs.view(-1, 4, 3)
    mirrored_obs = obs_unflattened.clone()
    # switch legs
    mirrored_obs[..., 0, :] = obs_unflattened[..., 2, :]
    mirrored_obs[..., 2, :] = obs_unflattened[..., 0, :]
    mirrored_obs[..., 1, :] = obs_unflattened[..., 3, :]
    mirrored_obs[..., 3, :] = obs_unflattened[..., 1, :]
    # mirror at xz plane
    mirrored_obs[..., 1] = -mirrored_obs[..., 1]

    return mirrored_obs.view(-1, 12)


def mirror_foot_pos_front_back(obs: torch.Tensor):
    # obs = x,y,z for each foot
    obs_unflattened = obs.view(-1, 4, 3)
    mirrored_obs = obs_unflattened.clone()
    # switch legs
    mirrored_obs[..., 0, :] = obs_unflattened[..., 1, :]
    mirrored_obs[..., 1, :] = obs_unflattened[..., 0, :]
    mirrored_obs[..., 2, :] = obs_unflattened[..., 3, :]
    mirrored_obs[..., 3, :] = obs_unflattened[..., 2, :]
    # mirror at yz plane
    mirrored_obs[..., 0] = -mirrored_obs[..., 0]

    return mirrored_obs.view(-1, 12)


def rotate_foot_pos_gravity_180(obs: torch.Tensor):
    # obs = x,y,z for each foot
    obs_unflattened = obs.view(-1, 4, 3)
    mirrored_obs = obs_unflattened.clone()
    # switch legs
    mirrored_obs[..., 0, :] = obs_unflattened[..., 3, :]
    mirrored_obs[..., 1, :] = obs_unflattened[..., 2, :]
    mirrored_obs[..., 2, :] = obs_unflattened[..., 1, :]
    mirrored_obs[..., 3, :] = obs_unflattened[..., 0, :]
    # rotate 180 degrees around z axis
    mirrored_obs[..., 0] = -mirrored_obs[..., 0]
    mirrored_obs[..., 1] = -mirrored_obs[..., 1]

    return mirrored_obs.view(-1, 12)


# - global


# origin
@torch.jit.script
def mirror_origin_left_right(obs: torch.Tensor):
    # obs = x, y, z, cos(yaw), sin(yaw)
    mirrored_obs = obs.clone()
    mirrored_obs[..., 1] = -obs[..., 1]
    mirrored_obs[..., 4] = -obs[..., 4]
    return mirrored_obs


@torch.jit.script
def mirror_origin_front_back(obs: torch.Tensor):
    # obs = x, y, z, cos(yaw), sin(yaw)
    mirrored_obs = obs.clone()
    mirrored_obs[..., 0] = -obs[..., 0]
    mirrored_obs[..., 3] = -obs[..., 3]
    return mirrored_obs


@torch.jit.script
def rotate_origin_180(obs: torch.Tensor):
    # obs = x, y, z, cos(yaw), sin(yaw)
    mirrored_obs = -obs.clone()
    mirrored_obs[..., 2] = obs[..., 2]

    return mirrored_obs


@torch.jit.script
def mirror_origin_2d_left_right(obs: torch.Tensor):
    # obs = x, y, z, cos(yaw), sin(yaw)
    mirrored_obs = obs.clone()
    mirrored_obs[..., 1] = -obs[..., 1]
    return mirrored_obs


@torch.jit.script
def mirror_origin_2d_front_back(obs: torch.Tensor):
    # obs = x, y, z, cos(yaw), sin(yaw)
    mirrored_obs = obs.clone()
    mirrored_obs[..., 0] = -obs[..., 0]
    return mirrored_obs


@torch.jit.script
def rotate_origin_2d_180(obs: torch.Tensor):
    # obs = x, y, z, cos(yaw), sin(yaw)
    mirrored_obs = -obs.clone()
    return mirrored_obs


# bbox
def mirror_object_bbox_left_right(obs: torch.Tensor) -> torch.Tensor:
    # obs shape: [batch_size, 8*3]
    obs_unflat = obs.view(-1, 8, 3).clone()
    # Flip y
    obs_unflat[..., 1] = -obs_unflat[..., 1]
    # Flatten back
    return obs_unflat.view(-1, 24)


def mirror_object_bbox_front_back(obs: torch.Tensor) -> torch.Tensor:
    # obs shape: [batch_size, 8*3]
    obs_unflat = obs.view(-1, 8, 3).clone()
    # Flip x
    obs_unflat[..., 0] = -obs_unflat[..., 0]
    return obs_unflat.view(-1, 24)


def rotate_object_bbox_180(obs: torch.Tensor) -> torch.Tensor:
    # obs shape: [batch_size, 8*3]
    obs_unflat = obs.view(-1, 8, 3).clone()
    # Flip x and y
    obs_unflat[..., 0:2] = -obs_unflat[..., 0:2]
    return obs_unflat.view(-1, 24)


##
# State removing functions
##


class remove_symmetry_subspaces_cls:
    """If skill to state symmetry is enforced, this function removes all except for the default
    symmetry subspace."""

    def __init__(self) -> None:
        self.extract_leg = extract_leg()
        self.extract_foot_pos = extract_foot_pos()

    def __call__(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        cleared_obs = {}
        for key in obs.keys():
            if key in ["joint_pos"]:  # or any other obs ordered by joint_ordering
                cleared_obs[key] = self.extract_leg(obs[key])
            elif key == "foot_pos":
                cleared_obs[key] = self.extract_foot_pos(obs[key])

            else:
                raise ValueError(f"Unknown observation name: {key}, No symmetry removal implemented.")

        return cleared_obs


class extract_foot_pos:
    """Assumes ordering: LF, LH, RF, RH"""

    def __init__(self, leg_id: Literal["LF", "RF", "LH", "RH"] = "RF") -> None:

        id_to_idx_map = {
            "LF": 0,
            "RF": 2,
            "LH": 1,
            "RH": 3,
        }

        self.leg_index = id_to_idx_map[leg_id]

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.view(-1, 4, 3)[:, self.leg_index, :]


class extract_leg:
    """Assumes joint ordering of Anymal D"""

    def __init__(self, leg_id: Literal["LF", "RF", "LH", "RH"] = "RF") -> None:

        id_to_idx_map = {
            "LF": [0, 4, 8],
            "LH": [1, 5, 9],
            "RF": [2, 6, 10],
            "RH": [3, 7, 11],
        }

        self.leg_indices = id_to_idx_map[leg_id]

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[..., self.leg_indices]


remove_symmetry_subspaces = remove_symmetry_subspaces_cls()
