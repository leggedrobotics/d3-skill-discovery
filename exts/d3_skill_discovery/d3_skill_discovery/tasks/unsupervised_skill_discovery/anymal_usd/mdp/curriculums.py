# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CurriculumTermCfg, SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> float:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.lab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    move_down = torch.zeros(len(env_ids), device=env.device, dtype=torch.bool)
    move_up = torch.zeros(len(env_ids), device=env.device, dtype=torch.bool)

    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float()).item()


class distance_traveled(ManagerTermBase):
    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.dist_traveled = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        buffer_size = self.cfg.params.get("buffer_size", 10)
        self.prev_pos_buffer = torch.zeros((env.num_envs, buffer_size, 2), device=env.device, dtype=torch.float32)
        self.prev_mean_pos = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)
        self.was_reset = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        distance_thresholds: tuple[float, float],
        p_random_move_up: float = 0.1,
        p_random_move_down: float = 0.1,
        buffer_size: int = 10,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):

        robot: Articulation = env.scene[robot_cfg.name]
        if not hasattr(robot.data, "spawn_pose"):
            return 0.0

        # since we might do skill resampling, we need to integrate the distance traveled.
        # we do this in a smooth way by using a buffer of previous positions.
        # get positions:
        spawn_poses_xy = robot.data.spawn_pose[env_ids, :2]
        robot: Articulation = env.scene[robot_cfg.name]
        robot_poses_xy = robot.data.root_pos_w[:, :2]

        # update buffer:
        self.prev_pos_buffer[torch.arange(env.num_envs), env.episode_length_buf % buffer_size] = robot_poses_xy
        # if buffer full, safe mean position:
        buffer_full = ((env.episode_length_buf + 1) % buffer_size == 0) & (env.common_step_counter > buffer_size)
        buffer_full[env_ids] = (env.episode_length_buf[env_ids] > buffer_size) & (env.common_step_counter > buffer_size)
        if buffer_full.any():
            new_mean_pos = self.prev_pos_buffer[buffer_full].mean(dim=1)
            self.dist_traveled[buffer_full] += torch.norm(new_mean_pos - self.prev_mean_pos[buffer_full], dim=-1)
            self.prev_mean_pos[buffer_full] = new_mean_pos
            self.dist_traveled[self.was_reset] = 0.0

            self.was_reset[buffer_full] = False

        # reset buffer:
        self.prev_pos_buffer[env_ids, :] = spawn_poses_xy.unsqueeze(1).expand(-1, buffer_size, -1)
        self.prev_mean_pos[env_ids] = 0
        self.was_reset[env_ids] = True

        # check if we need to move up or down
        move_up = self.dist_traveled[env_ids] > max(distance_thresholds)
        move_down = self.dist_traveled[env_ids] < min(distance_thresholds)

        # add random moves
        random_move_up = torch.rand(len(env_ids), device=env.device) < p_random_move_up
        random_move_down = torch.rand(len(env_ids), device=env.device) < p_random_move_down

        move_up |= random_move_up & ~move_down
        move_down |= random_move_down & ~move_up

        # update terrain levels
        terrain: TerrainImporter = env.scene.terrain
        terrain.update_env_origins(env_ids, move_up, move_down)

        # return the mean terrain level
        return torch.mean(terrain.terrain_levels.float()).item()


def terrain_levels_usd_perf(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    p_random_move_up: float,
    p_random_move_down: float,
    factor_name: str,
    thresholds: tuple[float, float],
    maximize: bool,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> float:
    """Curriculum based on the performance of the unsupervised skill discovery.
    Args:
        p_random_move_up: The probability of randomly moving up.
        p_random_move_down: The probability of randomly moving down.
        factor_name: The name of the factor to be used.
        thresholds: The thresholds to decide wether to move up or down
        maximize: If true, we move up if the metric is above the upper threshold.
                  If false, we move up if the metric is below the lower threshold.

    Returns:
        The mean terrain level for the given environment ids.
    """
    if not hasattr(env, "usd_metrics"):
        return 0.0

    usd_curr_metric = env.usd_metrics[factor_name][env_ids]
    lower_than_threshold = usd_curr_metric < min(thresholds)
    bigger_than_threshold = usd_curr_metric > max(thresholds)

    move_up = bigger_than_threshold if maximize else lower_than_threshold
    move_down = lower_than_threshold if maximize else bigger_than_threshold

    # add random moves
    random_move_up = torch.rand(len(env_ids), device=env.device) < p_random_move_up
    random_move_down = torch.rand(len(env_ids), device=env.device) < p_random_move_down

    move_up = move_up | (random_move_up & ~move_down)
    move_down = move_down | (random_move_down & ~move_up)

    # update terrain levels
    terrain: TerrainImporter = env.scene.terrain
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level

    return torch.mean(terrain.terrain_levels.float()).item()


def num_boxes_curriculum(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> float:
    """Curriculum based on the number of boxes in the scene.

    This term is used to increase the number of boxes in the scene when the robot walks far enough and decrease the
    number of boxes when the robot walks less than half of the distance required by the commanded velocity.

    Returns:
        The mean number of boxes for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)

    # return the mean terrain level

    num_range = env.cfg.data_container.num_obstacles_range
    return (num_range[0] + num_range[1]) / 2


class anneal_reward_weight(ManagerTermBase):
    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.initial_weights = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_names: list[str],
        ratio: float,
        start_step: int,
        num_steps: int,
    ):
        """Curriculum that modifies a reward weight a given number of steps.

        Args:
            env: The learning environment.
            env_ids: Not used since all environments are affected.
            term_names: The name of the reward term.
            ratio: The final reward will have weight initial weight * ratio.
            start_step: The number of steps after which the change should be applied.
            num_steps: The number of over which the weight should be annealed.
        """

        if self.initial_weights is None:
            self.initial_weights = {
                term_name: env.reward_manager.get_term_cfg(term_name).weight for term_name in term_names
            }

        if env.common_step_counter > start_step:
            # interpolate ratios from 1 to ratio
            current_ratio = 1 + (ratio - 1) * (env.common_step_counter - start_step) / num_steps
            if env.common_step_counter > start_step + num_steps:
                current_ratio = ratio

            for term_name in term_names:
                # obtain term settings
                term_cfg = env.reward_manager.get_term_cfg(term_name)
                # update term settings
                term_cfg.weight = self.initial_weights[term_name] * current_ratio
                env.reward_manager.set_term_cfg(term_name, term_cfg)

            return current_ratio
        return 1.0


class decay_extrinsic_rewards(ManagerTermBase):
    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.initial_weights = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_names: list[str],
        ratio: float,
        factor_name: str,
        thresholds: tuple[float, float],
        maximize: bool,
    ):
        """Curriculum that modifies a reward weight based on the performance of the skill discovery.

        Args:
            env: The learning environment.
            env_ids: Not used since all environments are affected.
            term_names: The name of the reward term.
            ratio: The final reward will have weight initial weight * ratio.
            start_step: The number of steps after which the change should be applied.
            factor_name: The name of the factor to be used to decide the reward weight.
            thresholds: The thresholds to decide in which we linearly interpolate the reward weight.
            maximize: If true, the reward weights will start to decrease if the metric is above the lower threshold,
                      Until the upper threshold is reached. If false, the reward weights will start to decrease if the
                        metric is below the upper threshold, until the lower threshold is reached.
        """

        if self.initial_weights is None:
            self.initial_weights = {
                term_name: env.reward_manager.get_term_cfg(term_name).weight for term_name in term_names
            }

        if not hasattr(env, "usd_metrics"):
            return 0.0

        usd_curr_metric = torch.mean(env.usd_metrics[factor_name][env_ids])
        lower_bound = min(thresholds)
        upper_bound = max(thresholds)
        reward_scale = (usd_curr_metric - lower_bound) / (upper_bound - lower_bound)
        reward_scale = torch.clamp(reward_scale, 0.0, 1.0)
        if maximize:
            reward_scale = 1.0 - reward_scale

        # interpolate ratios from 1 to ratio
        reward_scale = ratio + (1 - ratio) * reward_scale

        for term_name in term_names:
            # obtain term settings
            term_cfg = env.reward_manager.get_term_cfg(term_name)
            # update term settings
            term_cfg.weight = self.initial_weights[term_name] * reward_scale
            env.reward_manager.set_term_cfg(term_name, term_cfg)

        return reward_scale.item()
