from __future__ import annotations

import functools
import numpy as np
import torch
from typing import TYPE_CHECKING, Literal

from unsupervised_RL.tasks.unsupervised_skill_discovery.anymal_usd.mdp.commands import GoalCommand

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import EventTermCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainImporter
from isaaclab.terrains.utils import find_flat_patches
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def reset_multiple_instances_decorator(reset_func: callable) -> callable:
    """Decorator to reset multiple instances of an asset at once."""

    @functools.wraps(reset_func)
    def wrapper(*args, **kwargs):
        asset_configs = kwargs.get("asset_configs", None)
        asset_config = kwargs.get("asset_cfg", None)
        if asset_configs is None and asset_config is None:
            asset_config = SceneEntityCfg("robot")
        if asset_configs is not None and asset_config is not None:
            raise ValueError(
                "The decorator 'reset_multiple_instances_decorator' requires either 'asset_cfg' or 'asset_configs' to be provided, not both."
            )
        if asset_configs is None and asset_config is not None:
            asset_configs = [asset_config]
        for i, asset_cfg in enumerate(asset_configs):
            kwargs["asset_cfg"] = asset_cfg
            kwargs["reset_id"] = i
            reset_func(*args, **kwargs)

    return wrapper


@reset_multiple_instances_decorator
def reset_root_state_uniform_on_terrain_aware(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    offset: list = [0.0, 0.0, 0.0],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lowest_level: bool = False,
    reset_used_patches_ids: bool = False,
    asset_configs: list[SceneEntityCfg] | None = None,
    reset_id: int = 0,
):
    """Reset the asset root state to a random position at the lowest level of the scene.
    This might be called multiple times to reset the root state of multiple assets.
    If assets must not be placed on the same position, the reset_used_patches_ids flag must be set to False
    for all but the first function call per reset."""

    # reset the used ids if required
    if reset_id == 0:
        # resample number of obstacles per env
        num_obs_range = env.cfg.data_container.num_obstacles_range
        env.cfg.data_container.num_obstacles = torch.randint(
            num_obs_range[0], num_obs_range[1] + 1, (len(env_ids),), dtype=torch.float
        ).to(env.device)

    # check if the asset should be removed from the scene
    spawn_lowest_terrain = reset_id < env.cfg.data_container.num_obstacles
    all_reset_env_ids = env_ids

    if reset_used_patches_ids:
        # reset the used patches ids, should be done only once per reset
        env.scene.terrain.terrain_used_flat_patches = {
            "lowest_pos": torch.zeros((len(env_ids), 1), dtype=torch.int64).to(env.device) - 1,
            "init_pos": torch.zeros((len(env_ids), 1), dtype=torch.int64).to(env.device) - 1,
            "not_lowest_pos": torch.zeros((len(env_ids), 1), dtype=torch.int64).to(env.device) - 1,
        }

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # sample random positions
    positions = torch.zeros((len(env_ids), 3), device=env.device)
    flat_patch_type = "lowest_pos" if lowest_level else "init_pos"
    positions[spawn_lowest_terrain] = _sample_terrain_pos(env, asset, env_ids, flat_patch_type)[spawn_lowest_terrain]
    positions[~spawn_lowest_terrain] = _sample_terrain_pos(env, asset, env_ids, "not_lowest_pos")[~spawn_lowest_terrain]
    positions += torch.tensor(offset, device=asset.device)

    # sample random orientations
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)

    # convert to quaternions
    orientations = math_utils.quat_from_euler_xyz(rand_samples[:, 0], rand_samples[:, 1], rand_samples[:, 2])

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)


def _sample_terrain_pos(
    env: ManagerBasedEnv,
    asset: RigidObject | Articulation,
    env_ids: torch.Tensor,
    flat_patch_type: str = "init_pos",
) -> torch.Tensor:
    """sample position that is on the terrain."""

    terrain: TerrainImporter = env.scene.terrain
    valid_positions: torch.Tensor = terrain.flat_patches.get(flat_patch_type)
    if valid_positions is None:
        raise ValueError(
            "The event term 'reset_root_state_uniform_on_terrain_aware' requires valid flat patches under 'init_pos'."
            f" Found: {list(terrain.flat_patches.keys())}"
        )

    # sample random valid poses
    used_ids = terrain.terrain_used_flat_patches[flat_patch_type]
    ids = torch.zeros((len(env_ids),), dtype=torch.int64, device=env.device) - 1
    all_valid_per_env = torch.zeros((len(env_ids),), dtype=torch.bool, device=env.device)
    while not all_valid_per_env.all():
        ids[~all_valid_per_env] = torch.randint(
            0, valid_positions.shape[2], size=(int((~all_valid_per_env).sum()),), device=env.device
        )
        all_valid_per_env = torch.all(used_ids != ids.unsqueeze(1), dim=1)

    # add the used ids
    terrain.terrain_used_flat_patches[flat_patch_type] = torch.cat([used_ids, ids.unsqueeze(1)], dim=1)

    # get the positions
    positions = valid_positions[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids], ids]
    positions += asset.data.default_root_state[env_ids, :3]
    return positions


def reset_id_joints_by_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    joint_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset specific robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # find joint ids:
    joint_ids, _ = asset.find_joints(joint_names)
    joint_ids = tuple(joint_ids)

    # get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids[:, None], joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids[:, None], joint_ids].clone()

    # bias these values randomly
    joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids[:, None], joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids[:, None], joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids)


def reset_random_dist_from_goal(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    radius_range: tuple[float, float],
    z_offset: float = 0.0,
    command_name: str = "robot_goal",
):

    # extract the used quantities (to enable type-hinting)
    robot: RigidObject | Articulation = env.scene[asset_cfg.name]
    goal_cmd_geneator: GoalCommand = env.command_manager._terms[command_name]

    # sample random vector from the goal
    goal_pos_w = goal_cmd_geneator.goal_pos_w[env_ids]
    random_radius = math_utils.sample_uniform(*radius_range, (len(env_ids),), device=robot.device)
    random_angle = math_utils.sample_uniform(0, 2 * np.pi, (len(env_ids),), device=robot.device)
    random_2d_offset = torch.stack(
        [random_radius * torch.cos(random_angle), random_radius * torch.sin(random_angle)], dim=-1
    )

    # set orientation to 0
    orientations = torch.zeros((len(env_ids), 4), device=robot.device)
    orientations[:, 0] = 1.0

    # set the new position
    new_pos = goal_pos_w + torch.cat([random_2d_offset, torch.zeros((len(env_ids), 1), device=robot.device)], dim=-1)
    new_pos[:, 2] += z_offset
    robot.write_root_pose_to_sim(torch.cat([new_pos, orientations], dim=-1), env_ids=env_ids)


def reset_save_root_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    initial_pose = torch.cat([positions, orientations], dim=-1)
    asset.write_root_pose_to_sim(initial_pose, env_ids=env_ids)
    if not hasattr(asset.data, "spawn_pose"):
        asset.data.spawn_pose = initial_pose
    else:
        asset.data.spawn_pose[env_ids] = initial_pose

    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


class reset_joints_by_scale_one_leg_up(ManagerTermBase):
    """Randomize the external forces and torques applied to the bodies.
    This function creates a set of random forces and torques sampled from the given ranges. The number of forces
    and torques is equal to the number of bodies times the number of environments. The forces and torques are
    applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
    applied when ``asset.write_data_to_sim()`` is called in the environment.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        leg_id: Literal["LF", "RF", "LH", "RH"] = "RF"

        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
        asset: Articulation = env.scene[asset_cfg.name]
        joint_leg_list = [name for name in asset.data.joint_names if leg_id in name]
        self.joint_ids = asset.find_joints(joint_leg_list)[0]

        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        position_range: tuple[float, float],
        velocity_range: tuple[float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        leg_up_probability: float = 0.5,
    ):
        """Reset the robot joints by scaling the default position and velocity by the given ranges.

        This function samples random values from the given ranges and scales the default joint positions and velocities
        by these values. The scaled values are then set into the physics simulation.
        One leg is moved up to help exploration.
        """
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # get default joint state
        joint_pos = asset.data.default_joint_pos[env_ids].clone()
        joint_vel = asset.data.default_joint_vel[env_ids].clone()

        # scale these values randomly
        joint_pos *= math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
        joint_vel *= math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

        # move one leg up
        random_move_up = torch.rand(len(env_ids), device=joint_pos.device) < leg_up_probability
        joint_pos[random_move_up, self.joint_ids[1]] -= 1.5  # TODO the sign might be different for other legs
        # TODO manipulate other joints of that leg

        # clamp joint pos to limits
        joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
        joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
        # clamp joint vel to limits
        joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
        joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

        # set into the physics simulation
        asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


class apply_external_force_torque_interval(ManagerTermBase):
    """Randomize the external forces and torques applied to the bodies.
    This function creates a set of random forces and torques sampled from the given ranges. The number of forces
    and torques is equal to the number of bodies times the number of environments. The forces and torques are
    applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
    applied when ``asset.write_data_to_sim()`` is called in the environment.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.duration = int(cfg.cls_cfg["duration_seconds"] / env.step_dt)

        self.applied_num_steps = torch.zeros((env.scene.num_envs,), dtype=torch.int64, device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        force_range: tuple[float, float],
        torque_range: tuple[float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        interval_size_seconds: float = 1,
    ):

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=asset.device)
        # resolve number of bodies
        num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

        # sample random forces and torques
        size = (len(env_ids), num_bodies, 3)
        forces = math_utils.sample_uniform(*force_range, size, asset.device)
        torques = math_utils.sample_uniform(*torque_range, size, asset.device)
        # set the forces and torques into the buffers
        # note: these are only applied when you call: `asset.write_data_to_sim()`
        asset.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids)


def disable_joint_effort_sim(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    rest_duration_s: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Disable the joint actions for the given asset.

    This allows the robot to not take any actions during the rest phase.
    It should be called before the simulation step.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # check if asset has implicit actuators
    if asset._has_implicit_actuators:  # type: ignore
        raise NotImplementedError("Disable joint effort actions is not implemented for implicit actuators.")

    # check which environments are in the rest phase
    env_in_rest_phase = env.episode_length_buf < int(rest_duration_s / env.step_dt)
    rest_env_ids = env_in_rest_phase.nonzero().flatten()

    # create zero joint effort target
    asset._joint_effort_target_sim[rest_env_ids, :] = 0.0  # type: ignore
    # disable the joint actions
    asset.root_physx_view.set_dof_actuation_forces(asset._joint_effort_target_sim, rest_env_ids)  # type: ignore
