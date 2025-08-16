import numpy as np
import torch
from typing import Literal

from rsl_rl.utils import TIMER_CUMULATIVE
from unsupervised_RL.tasks.unsupervised_skill_discovery.anymal_usd.mdp.commands import GoalCommand
from unsupervised_RL.tasks.unsupervised_skill_discovery.anymal_usd.mdp.utils import (
    get_robot_lin_vel_w,
    get_robot_pos,
    get_robot_quat,
    get_robot_rot_vel_w,
)

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import ObservationTermCfg, RewardTermCfg, SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCaster, RayCasterCfg, SensorBase, TiledCamera, patterns
from isaaclab.utils import math as math_utils
from isaaclab.utils.timer import Timer
from isaaclab.utils.warp import raycast_mesh

from .obs_utils import compute_asset_aabb


##
# - lidar
##
def lidar_obs_dist_2d(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """lidar scan from the given sensor w.r.t. the sensor's frame."""
    sensor: SensorBase = env.scene.sensors[sensor_cfg.name]
    distances = torch.linalg.vector_norm(
        sensor.data.ray_hits_w[..., :2] - sensor.data.pos_w[..., :2].unsqueeze(1), dim=2
    )

    return torch.log(distances + 1e-6)


def lidar_height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """lidar scan from the given sensor w.r.t. the sensor's frame."""
    sensor: SensorBase = env.scene.sensors[sensor_cfg.name]

    height_diffs = sensor.data.ray_hits_w[..., 2] - sensor.data.pos_w[..., 2].unsqueeze(1)

    return height_diffs


def lidar_obs_dist_2d_log(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """lidar scan from the given sensor w.r.t. the sensor's frame."""
    sensor: SensorBase = env.scene.sensors[sensor_cfg.name]
    distances = torch.linalg.vector_norm(sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1), dim=2)

    return torch.log(distances + 1e-6)


##
# - positions
##
def origin_env(env: ManagerBasedEnv, entity_cfg: SceneEntityCfg) -> torch.Tensor:
    """Returns the pose of the entity relative to the terrain origin.
    x,y position and heading in the form of cos(theta), sin(theta)."""
    entity: RigidObject | Articulation = env.scene[entity_cfg.name]

    # - position
    pos = get_robot_pos(entity)
    terrain = env.scene.terrain
    terrain_origins = terrain.env_origins
    rel_pos = pos.squeeze(1) - terrain_origins

    # - heading
    quat = get_robot_quat(entity).squeeze(1)
    yaw = math_utils.euler_xyz_from_quat(quat)[2].unsqueeze(1)
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)

    pose_2d = torch.cat([rel_pos, cos_yaw, sin_yaw], dim=-1)
    return pose_2d


def origin_b(env: ManagerBasedEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Returns the vector from the robot to the origin in the robots yaw frame."""
    robot: Articulation = env.scene[robot_cfg.name]

    robot_pos = robot.data.root_pos_w
    robot_quat = robot.data.root_quat_w
    terrain_origins = env.scene.terrain.env_origins

    rel_pos = terrain_origins - robot_pos

    # Rotate the vector to the robot's frame
    rel_pos_rot = math_utils.quat_rotate_inverse(math_utils.yaw_quat(robot_quat), rel_pos)

    return rel_pos_rot


def pose2d_b(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Returns the 2d pose of the entity in the robots frame."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # - position
    pos_w = asset.data.root_pos_w
    robot = env.scene[robot_cfg.name]
    robot_pos_w = robot.data.root_pos_w
    rel_pos = pos_w.squeeze(1) - robot_pos_w

    # - heading
    quat = asset.data.root_quat_w.squeeze(1)
    robot_quat = robot.data.root_quat_w
    yaw = math_utils.euler_xyz_from_quat(quat)[2].unsqueeze(1)
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)

    # Rotate the vector to the robot's frame
    rel_pos_rot = math_utils.quat_rotate_inverse(math_utils.yaw_quat(robot_quat), rel_pos)

    pose_2d = torch.cat([rel_pos_rot, cos_yaw, sin_yaw], dim=-1)
    return pose_2d


class object_bbox_in_robot_base(ManagerTermBase):
    """Provides the bounding box of the object in the robot base frame.

    This term uses the AABB bounds of the object to compute the bounding box.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        """Initialize the object's bounding box observation term.

        Args:
            cfg: The configuration for the observation term.
            env: The environment.
        """
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        object_cfg: SceneEntityCfg = cfg.params["object_cfg"]
        self.cube: RigidObject = env.scene[object_cfg.name]
        self.robot: Articulation = env.scene["robot"]

        if not isinstance(self.cube, RigidObject):
            raise ValueError(
                f"Observation term {self.__class__.__name__} expects a RigidObject asset, but got {type(self.cube)}"
            )

        # parse asset aabb of the object in object frame
        self.aabb = compute_asset_aabb(self.cube.cfg.prim_path, device=self.device)
        # create keypoints in the object frame
        corners = torch.tensor(
            [[int((i >> k) & 1) * 2 - 1 for k in range(3)] for i in range(8)], device=self.device, dtype=torch.float32
        )
        self.keypoints = corners.unsqueeze(0) * self.aabb.unsqueeze(1) / 2

        # setup debug visualization
        if env.sim.has_gui():
            self._set_debug_vis_impl()

    def __call__(self, env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, control_frame: bool = False) -> torch.Tensor:
        # transform points to world frame
        self.keypoints_w = math_utils.transform_points(
            self.keypoints, self.cube.data.root_pos_w, self.cube.data.root_quat_w
        )
        # debug visualization
        if env.sim.has_gui():
            self._debug_vis_callback()

        # resolve the control frame
        if control_frame:
            quat_w = math_utils.yaw_quat(self.robot.data.root_quat_w)
        else:
            quat_w = self.robot.data.root_quat_w

        # compute transform from base to world
        pos_b = -math_utils.quat_rotate_inverse(quat_w, self.robot.data.root_pos_w)
        quat_b = math_utils.quat_conjugate(quat_w)

        # transform points to base frame
        keypoints_b = math_utils.transform_points(self.keypoints_w, pos_b, quat_b)
        return keypoints_b.reshape(self.num_envs, -1)

    """
    Internals.
    """

    def _set_debug_vis_impl(self):
        """Set up the debug visualization."""
        cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/ObjectBbox",
            markers={},
        )
        colors = [
            (0.052, 0.176, 0.422),
            (0.747, 0.164, 0.219),
            (0.477, 0.759, 0.237),
            (0.759, 0.759, 0.759),
            (0.238, 0.575, 0.911),
            (0.852, 0.403, 0.769),
            (0.133, 0.151, 0.156),
            (0.926, 0.937, 0.957),
        ]
        for i, color in enumerate(colors):
            cfg.markers[f"bbox_{i}"] = sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
        self.visualization_marker = VisualizationMarkers(cfg)
        self.marker_indices = torch.arange(8, device=self.device).repeat(self.num_envs, 1).view(-1)

    def _debug_vis_callback(self):
        """Debug visualization."""
        self.visualization_marker.visualize(self.keypoints_w.reshape(-1, 3), marker_indices=self.marker_indices)


class object_bbox_spawn(ManagerTermBase):
    """Provides the bounding box of the object in the robot base frame.

    This term uses the AABB bounds of the object to compute the bounding box.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        """Initialize the object's bounding box observation term.

        Args:
            cfg: The configuration for the observation term.
            env: The environment.
        """
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        object_cfg: SceneEntityCfg = cfg.params["object_cfg"]
        self.cube: RigidObject = env.scene[object_cfg.name]
        self.robot: Articulation = env.scene["robot"]

        if not isinstance(self.cube, RigidObject):
            raise ValueError(
                f"Observation term {self.__class__.__name__} expects a RigidObject asset, but got {type(self.cube)}"
            )

        # parse asset aabb of the object in object frame
        self.aabb = compute_asset_aabb(self.cube.cfg.prim_path, device=self.device)
        # create keypoints in the object frame
        corners = torch.tensor(
            [[int((i >> k) & 1) * 2 - 1 for k in range(3)] for i in range(8)], device=self.device, dtype=torch.float32
        )
        self.keypoints = corners.unsqueeze(0) * self.aabb.unsqueeze(1) / 2

        # setup debug visualization
        if env.sim.has_gui():
            self._set_debug_vis_impl()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg,
        spawn_asset_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
        # transform points to world frame
        self.keypoints_w = math_utils.transform_points(
            self.keypoints, self.cube.data.root_pos_w, self.cube.data.root_quat_w
        )

        if spawn_asset_cfg is None:
            # keypoints in the terrain frame
            terrain_origins = env.scene.terrain.env_origins
            keypoints_b = self.keypoints_w - terrain_origins.unsqueeze(1)
            return keypoints_b.reshape(self.num_envs, -1)

        spawn_asset: Articulation | RigidObject = env.scene[spawn_asset_cfg.name]
        if not hasattr(spawn_asset.data, "spawn_pose"):
            return torch.zeros(env.num_envs, 8 * 3).to(env.device)
        spawn_pos_w, spawn_rot_w = spawn_asset.data.spawn_pose[:, :3], spawn_asset.data.spawn_pose[:, 3:]  # type: ignore

        # debug visualization
        if env.sim.has_gui():
            self._debug_vis_callback()

        # compute transform from base to world
        pos_b = -math_utils.quat_rotate_inverse(spawn_rot_w, spawn_pos_w)
        quat_b = math_utils.quat_conjugate(spawn_rot_w)

        # transform points to base frame
        keypoints_b = math_utils.transform_points(self.keypoints_w, pos_b, quat_b)
        return keypoints_b.reshape(self.num_envs, -1)

    """
    Internals.
    """

    def _set_debug_vis_impl(self):
        """Set up the debug visualization."""
        cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/ObjectBbox",
            markers={},
        )
        colors = [
            (0.052, 0.176, 0.422),
            (0.747, 0.164, 0.219),
            (0.477, 0.759, 0.237),
            (0.759, 0.759, 0.759),
            (0.238, 0.575, 0.911),
            (0.852, 0.403, 0.769),
            (0.133, 0.151, 0.156),
            (0.926, 0.937, 0.957),
        ]
        for i, color in enumerate(colors):
            cfg.markers[f"bbox_{i}"] = sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
        self.visualization_marker = VisualizationMarkers(cfg)
        self.marker_indices = torch.arange(8, device=self.device).repeat(self.num_envs, 1).view(-1)

    def _debug_vis_callback(self):
        """Debug visualization."""
        self.visualization_marker.visualize(self.keypoints_w.reshape(-1, 3), marker_indices=self.marker_indices)


class origin_spawn(ManagerTermBase):
    """Returns the pose of the robot wrt to a spawn frame.
    Args:
        env: The environment.
        asset_cfg: The asset (e.g, robot) configuration.
        pos_2d_only: If True, only return the 2D position (x, y).
        noise_scale: If > 0, add noise to the position.
        robot_frame: If True, return the position in the robot's frame, else in the spawn frame, default is False.
        return_yaw: If True, return the heading as (cos(theta), sin(theta)).
        heading_only: If True, return only the heading (cos(theta), sin(theta)).
        reset_every_step: If True, reset the spawn pose every step. ONLY ENABLE THIS FOR DEPLOYMENT.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        spawn_asset_cfg: SceneEntityCfg = cfg.params.get("spawn_asset_cfg", SceneEntityCfg("robot"))
        self.spawn_asset: Articulation | RigidObject = env.scene[spawn_asset_cfg.name]

        self._internal_spawn_pos = torch.cat(
            [self.spawn_asset.data.root_pos_w, self.spawn_asset.data.root_quat_w], dim=-1
        )
        self.use_internal_spawn_pos = cfg.params.get("reset_every_n_steps", -1) != -1

    @property
    def spawn_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the spawn pose."""
        if self.use_internal_spawn_pos:
            return self._internal_spawn_pos[:, :3], self._internal_spawn_pos[:, 3:]
        else:
            return self.spawn_asset.data.spawn_pose[:, :3], self.spawn_asset.data.spawn_pose[:, 3:]  # type: ignore

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        spawn_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        pos_2d_only: bool = False,
        noise_scale: float = 0,
        robot_frame: bool = False,
        return_yaw: bool = True,
        heading_only: bool = False,
        reset_every_n_steps: int = -1,
    ) -> torch.Tensor:

        robot: Articulation | RigidObject = env.scene[asset_cfg.name]
        spawn_asset: Articulation | RigidObject = env.scene[spawn_asset_cfg.name]

        if not hasattr(spawn_asset.data, "spawn_pose"):
            return torch.zeros(env.num_envs, 5).to(env.device)

        robot_pos = robot.data.root_pos_w
        robot_quat = robot.data.root_quat_w
        spawn_pos, spawn_rot = self.spawn_pose

        # Rotate the vector to the robot's frame
        if robot_frame:
            # where is the spawn in the robot's frame
            pos, heading = math_utils.subtract_frame_transforms(
                robot_pos, math_utils.yaw_quat(robot_quat), spawn_pos, math_utils.yaw_quat(spawn_rot)
            )
        else:
            # where is the robot in the spawn frame
            pos, heading = math_utils.subtract_frame_transforms(
                spawn_pos, math_utils.yaw_quat(spawn_rot), robot_pos, math_utils.yaw_quat(robot_quat)
            )

        if reset_every_n_steps != -1:
            time_to_reset = env.episode_length_buf % reset_every_n_steps == 0
            if time_to_reset.any():
                self._internal_spawn_pos[time_to_reset] = torch.cat([robot_pos, robot_quat], dim=-1)[time_to_reset]

        # add noise
        if noise_scale > 0:
            # noise relative to distance
            noise_xy = torch.randn_like(pos[..., :2]) * noise_scale * torch.linalg.norm(pos, dim=-1, keepdim=True)
            noise_z = torch.randn_like(pos[..., 2:]) * noise_scale * torch.linalg.norm(pos, dim=-1, keepdim=True)
            pos += torch.cat([noise_xy, noise_z], dim=-1)
        if pos_2d_only:
            return pos[..., :2]
        if heading_only:
            yaw = math_utils.euler_xyz_from_quat(heading)[2].unsqueeze(1)
            return torch.cat([torch.cos(yaw), torch.sin(yaw)], dim=-1)
        if return_yaw:
            # yaw = math_utils.euler_xyz_from_quat(heading)[2].unsqueeze(1) * 0
            # return torch.cat([pos * 0, torch.cos(yaw), torch.sin(yaw)], dim=-1)

            yaw = math_utils.euler_xyz_from_quat(heading)[2].unsqueeze(1)
            return torch.cat([pos, torch.cos(yaw), torch.sin(yaw)], dim=-1)
        return pos


class origin_spawn_quadrant(ManagerTermBase):
    """Same as origin_spawn, but we can set only one quadrant visible.
    Useful to test factor weights
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        spawn_asset_cfg: SceneEntityCfg = cfg.params.get("spawn_asset_cfg", SceneEntityCfg("robot"))
        self.spawn_asset: Articulation | RigidObject = env.scene[spawn_asset_cfg.name]

        self._internal_spawn_pos = torch.cat(
            [self.spawn_asset.data.root_pos_w, self.spawn_asset.data.root_quat_w], dim=-1
        )
        self.use_internal_spawn_pos = cfg.params.get("reset_every_n_steps", -1) != -1

    @property
    def spawn_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the spawn pose."""
        if self.use_internal_spawn_pos:
            return self._internal_spawn_pos[:, :3], self._internal_spawn_pos[:, 3:]
        else:
            return self.spawn_asset.data.spawn_pose[:, :3], self.spawn_asset.data.spawn_pose[:, 3:]  # type: ignore

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        quadrant: Literal["NE", "NW", "SE", "SW"],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        spawn_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        reset_every_n_steps: int = -1,
    ) -> torch.Tensor:

        robot: Articulation | RigidObject = env.scene[asset_cfg.name]
        spawn_asset: Articulation | RigidObject = env.scene[spawn_asset_cfg.name]

        if not hasattr(spawn_asset.data, "spawn_pose"):
            return torch.zeros(env.num_envs, 5).to(env.device)

        robot_pos = robot.data.root_pos_w
        robot_quat = robot.data.root_quat_w
        spawn_pos, spawn_rot = self.spawn_pose

        # where is the robot in the spawn frame
        pos, heading = math_utils.subtract_frame_transforms(
            spawn_pos, math_utils.yaw_quat(spawn_rot), robot_pos, math_utils.yaw_quat(robot_quat)
        )

        if reset_every_n_steps != -1:
            time_to_reset = env.episode_length_buf % reset_every_n_steps == 0
            if time_to_reset.any():
                self._internal_spawn_pos[time_to_reset] = torch.cat([robot_pos, robot_quat], dim=-1)[time_to_reset]

        # check if the position is in the correct quadrant
        if quadrant == "NE":
            visible = (pos[..., 0] > 0) & (pos[..., 1] > 0)
        elif quadrant == "NW":
            visible = (pos[..., 0] < 0) & (pos[..., 1] > 0)
        elif quadrant == "SE":
            visible = (pos[..., 0] > 0) & (pos[..., 1] < 0)
        elif quadrant == "SW":
            visible = (pos[..., 0] < 0) & (pos[..., 1] < 0)

        # mask invisible positions
        pos[~visible] = 0

        return pos[..., :2]


def base_lin_vel_quadrant(
    env: ManagerBasedEnv, quadrant: Literal["NE", "NW", "SE", "SW"], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    vel_b = asset.data.root_lin_vel_b

    # check if the position is in the correct quadrant
    if quadrant == "NE":
        visible = (vel_b[..., 0] > 0) & (vel_b[..., 1] > 0)
    elif quadrant == "NW":
        visible = (vel_b[..., 0] < 0) & (vel_b[..., 1] > 0)
    elif quadrant == "SE":
        visible = (vel_b[..., 0] > 0) & (vel_b[..., 1] < 0)
    elif quadrant == "SW":
        visible = (vel_b[..., 0] < 0) & (vel_b[..., 1] < 0)

    # mask invisible positions
    vel_b[~visible] = 0

    return vel_b


def time_left(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns the remaining time in the episode."""
    try:
        return (env.episode_length_buf / env.max_episode_length).unsqueeze(1)
    except AttributeError:
        return torch.zeros(env.num_envs, 1).to(env.device)


def pose_3d_env(env: ManagerBasedEnv, entity_cfg: SceneEntityCfg) -> torch.Tensor:
    """Position and Quaternion in environment frame"""
    entity: RigidObject | Articulation = env.scene[entity_cfg.name]

    # - position
    pos = get_robot_pos(entity)
    terrain = env.scene.terrain
    terrain_origins = terrain.env_origins
    rel_pos = pos.squeeze(1) - terrain_origins

    # - quaternion
    quat = get_robot_quat(entity).squeeze(1)

    return torch.cat([rel_pos, quat], dim=-1)


def pose_2d_w(env: ManagerBasedEnv, entity_cfg: SceneEntityCfg) -> torch.Tensor:
    """Returns the pose of the entity in the terrain frame."""

    entity: RigidObject | Articulation = env.scene[entity_cfg.name]

    # - position
    pos = get_robot_pos(entity)
    terrain = env.scene.terrain
    terrain_origins = terrain.env_origins
    rel_pos = pos.squeeze(1) - terrain_origins

    # - heading
    quat = get_robot_quat(entity).squeeze(1)
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat)
    cos_yaw, sin_yaw = torch.cos(yaw).unsqueeze(1), torch.sin(yaw).unsqueeze(1)

    pose_2d = torch.cat([rel_pos[:, :2], cos_yaw, sin_yaw], dim=-1)

    return pose_2d


def pose_2d_spawn(
    env: ManagerBasedEnv, entity_cfg: SceneEntityCfg, spawn_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Returns the 2D pose of the entity wrt to the spawn frame of the robot.
    x, y positions and heading in the form of cos(theta), sin(theta)."""

    spawn_asset = env.scene[spawn_asset_cfg.name]
    if not hasattr(spawn_asset.data, "spawn_pose"):
        return torch.zeros(env.num_envs, 4).to(env.device)

    spawn_pos_w, spawn_rot_w = spawn_asset.data.spawn_pose[:, :3], spawn_asset.data.spawn_pose[:, 3:]
    entity: RigidObject | Articulation = env.scene[entity_cfg.name]

    # - get entity pose
    entity_pos_w = get_robot_pos(entity)
    entity_quat_w = get_robot_quat(entity).squeeze(1)

    # - transform to spawn frame
    pos, quat = math_utils.subtract_frame_transforms(
        spawn_pos_w, math_utils.yaw_quat(spawn_rot_w), entity_pos_w, math_utils.yaw_quat(entity_quat_w)
    )

    # - heading
    yaw = math_utils.euler_xyz_from_quat(quat)[2].unsqueeze(1)
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)

    pose_2d = torch.cat([pos[:, :2], cos_yaw, sin_yaw], dim=-1)

    return pose_2d


def box_pose_2d(env: ManagerBasedEnv, entity_str: str, pov_entity: SceneEntityCfg) -> torch.Tensor:
    """Returns the 2d pose of all entities relative to the robot's frame.
    x, y positions and heading in the form of cos(theta), sin(theta)."""

    # - box poses
    box_ids = [asset for asset in list(env.scene.rigid_objects.keys()) if entity_str in asset]
    box_poses = []
    box_quats = []

    for box_id in box_ids:
        box_poses.append(env.scene.rigid_objects[box_id].data.root_pos_w)
        box_quats.append(env.scene.rigid_objects[box_id].data.root_quat_w)

    boxes_positions_w = torch.stack(box_poses, dim=1)
    boxes_quats_w = torch.stack(box_quats, dim=1)

    # - robot pose
    robot = env.scene[pov_entity.name]
    robot_pos_w = get_robot_pos(robot)
    robot_quat_w = get_robot_quat(robot)

    # Expand robot pose to match the number of boxes
    robot_pos_w_expanded = robot_pos_w.unsqueeze(1).expand_as(boxes_positions_w)
    robot_quat_w_expanded = robot_quat_w.unsqueeze(1).expand_as(boxes_quats_w)

    # - calculate pose of boxes in robot frame
    t_box_robot, q_box_robot = math_utils.subtract_frame_transforms(
        robot_pos_w_expanded, robot_quat_w_expanded, boxes_positions_w, boxes_quats_w
    )

    # Extract x, y positions
    x = t_box_robot[..., 0]
    y = t_box_robot[..., 1]

    # Compute yaw angle from the quaternion
    w = q_box_robot[..., 0]
    x_q = q_box_robot[..., 1]
    y_q = q_box_robot[..., 2]
    z_q = q_box_robot[..., 3]

    # Yaw angle computation
    sin_yaw = 2 * (w * z_q + x_q * y_q)
    cos_yaw = 1 - 2 * (y_q * y_q + z_q * z_q)

    # yaw = torch.atan2(sin_yaw, cos_yaw)
    # Stack the results into a single tensor
    pose = torch.concat([x, y, cos_yaw, sin_yaw], dim=1)

    return pose


def box_pose_3d(
    env: ManagerBasedEnv, entity_str: str, pov_entity: SceneEntityCfg, yaw_only: bool = True
) -> torch.Tensor:
    """Returns the full pose of all entities relative to the robot's frame.
    x, y, z positions and quaternion."""

    # - box poses
    box_ids = [asset for asset in list(env.scene.rigid_objects.keys()) if entity_str in asset]
    box_poses = []
    box_quats = []

    for box_id in box_ids:
        box_poses.append(env.scene.rigid_objects[box_id].data.root_pos_w)
        box_quats.append(env.scene.rigid_objects[box_id].data.root_quat_w)

    boxes_positions_w = torch.stack(box_poses, dim=1)
    boxes_quats_w = torch.stack(box_quats, dim=1)

    # - robot pose
    robot = env.scene[pov_entity.name]
    robot_pos_w = get_robot_pos(robot)
    robot_quat_w = get_robot_quat(robot)
    if yaw_only:
        robot_quat_w = math_utils.yaw_quat(robot_quat_w)

    # Expand robot pose to match the number of boxes
    robot_pos_w_expanded = robot_pos_w.unsqueeze(1).expand_as(boxes_positions_w)
    robot_quat_w_expanded = robot_quat_w.unsqueeze(1).expand_as(boxes_quats_w)

    # - calculate pose of boxes in robot frame
    t_box_robot, q_box_robot = math_utils.subtract_frame_transforms(
        robot_pos_w_expanded, robot_quat_w_expanded, boxes_positions_w, boxes_quats_w
    )
    pose = torch.concat([t_box_robot, q_box_robot], dim=-1).squeeze(1)
    return pose


class foot_pos_b(ManagerTermBase):
    """
    Returns the foot position in the robots root frame
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.default_feet_pos_b = torch.tensor(
            [
                [0.4730, 0.3178, -0.6775],
                [-0.4730, 0.3178, -0.6775],
                [0.4730, -0.3178, -0.6775],
                [-0.4730, -0.3178, -0.6775],
            ],
            device=env.device,
        ).unsqueeze(0)

    def __call__(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, foot_index: int | None = None) -> torch.Tensor:
        asset: RigidObject = env.scene[asset_cfg.name]

        if asset_cfg.body_ids is None:
            raise ValueError("The body_ids of the robot are not defined in the environment config.")
        body_ids: list[int] = asset_cfg.body_ids  # type: ignore
        foot_pos_w = asset.data.body_pos_w[:, body_ids]
        root_pose_w = asset.data.root_pos_w.unsqueeze(1)
        root_quat = math_utils.yaw_quat(asset.data.root_quat_w.unsqueeze(1))
        foot_pos_b = math_utils.quat_rotate_inverse(root_quat, foot_pos_w - root_pose_w)
        # foot_pos_b -= self.default_feet_pos_b
        if foot_index is not None:
            return foot_pos_b[:, foot_index, :]
        return foot_pos_b.flatten(start_dim=1)


class foot_pos_b_xy(ManagerTermBase):
    """
    Returns the foot position in the robots root xy frame
    The z coordinate is wrt to the terrain
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.default_feet_pos_b = torch.tensor(
            [
                [0.4730, 0.3178, -0.6775],
                [-0.4730, 0.3178, -0.6775],
                [0.4730, -0.3178, -0.6775],
                [-0.4730, -0.3178, -0.6775],
            ],
            device=env.device,
        ).unsqueeze(0)

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg,
        height_scanner: SceneEntityCfg,
        foot_index: int | None = None,
    ) -> torch.Tensor:
        asset: RigidObject = env.scene[asset_cfg.name]

        if asset_cfg.body_ids is None:
            raise ValueError("The body_ids of the robot are not defined in the environment config.")

        sensor: RayCaster = env.scene.sensors[height_scanner.name]  # type: ignore
        terrain_z = sensor.data.ray_hits_w[..., 2].mean(dim=1)

        body_ids: list[int] = asset_cfg.body_ids  # type: ignore
        foot_pos_w = asset.data.body_pos_w[:, body_ids]
        root_pose_w = asset.data.root_pos_w
        root_pose_w[..., 2] = terrain_z
        root_quat = math_utils.yaw_quat(asset.data.root_quat_w.unsqueeze(1))
        foot_pos_b = math_utils.quat_rotate_inverse(root_quat, foot_pos_w - root_pose_w.unsqueeze(1))
        # foot_pos_b -= self.default_feet_pos_b
        if foot_index is not None:
            return foot_pos_b[:, foot_index, :]
        return foot_pos_b.flatten(start_dim=1)


def base_height_metric(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    radius: float = 0.5,
) -> torch.Tensor:
    """
    Returns a score in [0, 1] based on the height of the robot base above the terrain.
    """

    # create the ray casting pattern
    N_rays = 100
    angles = torch.linspace(0, 2 * torch.pi, N_rays + 1, device=env.device)[:-1]
    offset = 25.0
    ray_starts = torch.stack([torch.cos(angles), torch.sin(angles), torch.zeros_like(angles) + offset], dim=1) * radius
    ray_directions = torch.stack([torch.zeros_like(angles), torch.zeros_like(angles), -torch.ones_like(angles)], dim=1)

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    robot_base_positions = asset.data.root_pos_w

    # offset the ray starts by the robot base positions
    ray_starts_w = ray_starts.unsqueeze(0) + robot_base_positions.unsqueeze(1)
    ray_dirs_w = ray_directions.unsqueeze(0).expand_as(ray_starts_w)

    # raycast the terrain mesh
    terrain_mesh = env.scene.terrain.warp_meshes["terrain"]
    ray_hits = raycast_mesh(
        ray_starts_w.float().contiguous(),
        ray_dirs_w.float().contiguous(),
        terrain_mesh.id,
    )[
        0
    ].squeeze(0)

    # calculate score
    dist_std = target_height / 4
    dist_tol = target_height / 3
    mean_heights = robot_base_positions[:, 2] - ray_hits[..., 2].mean(dim=1)
    dist_above_target = mean_heights - target_height
    score = (torch.tanh((dist_above_target + dist_tol) / dist_std) + 1) / 2
    return score


def base_height_flatness_metric(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Returns a score in [0, 1] based on the height of the robot base above the terrain.
    """

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # calculate score
    dist_std = target_height / 4
    dist_tol = target_height / 3
    mean_heights = base_height(env, asset_cfg)
    dist_above_target = mean_heights - target_height
    base_height_score = (torch.tanh((dist_above_target + dist_tol) / dist_std) + 1) / 2

    # calculate flatness score
    torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    angle = torch.acos(
        torch.cosine_similarity(
            asset.data.projected_gravity_b, torch.tensor([0, 0, -1]).unsqueeze(0).to(asset.device), dim=1
        )
    )
    flatness_score = 1 - torch.sqrt(torch.sin(angle))

    return base_height_score * (flatness_score).unsqueeze(1)


##
# - velocities
##
def velocity_2d_b(env: ManagerBasedEnv, entity_cfg: SceneEntityCfg, pov_entity_cfg: SceneEntityCfg) -> torch.Tensor:
    """Returns the velocity vector of the entity rotated to the robot's frame (only yaw considered).
    The robots velocity is neglected."""
    entity: RigidObject | Articulation = env.scene[entity_cfg.name]
    robot: RigidObject | Articulation = env.scene[pov_entity_cfg.name]

    robot_quat_w = math_utils.yaw_quat(get_robot_quat(robot))
    if entity == robot:
        lin_vel_w = get_robot_lin_vel_w(robot)
        lin_vel_b = math_utils.quat_rotate_inverse(robot_quat_w, lin_vel_w)
        ang_vel_z_w = get_robot_rot_vel_w(robot)[..., 2]
        return torch.cat([lin_vel_b, ang_vel_z_w.unsqueeze(1)], dim=-1)

    entity_vel_w = entity.data.body_lin_vel_w.squeeze(1)
    entity_ang_vel_z = entity.data.body_ang_vel_w.squeeze(1)[..., 2]
    entity_vel_b = math_utils.quat_rotate_inverse(robot_quat_w, entity_vel_w)
    return torch.cat([entity_vel_b[..., :2], entity_ang_vel_z.unsqueeze(1)], dim=-1)


def rotation_velocity_2d_b(
    env: ManagerBasedEnv, entity_cfg: SceneEntityCfg, pov_entity_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Returns the angular velocity in z direction (yaw rotation)"""
    entity: RigidObject | Articulation = env.scene[entity_cfg.name]
    robot: RigidObject | Articulation = env.scene[pov_entity_cfg.name]

    if entity == robot:
        ang_vel_z_w = get_robot_rot_vel_w(robot)[..., 2]
        return ang_vel_z_w.unsqueeze(1)

    entity_ang_vel_z = entity.data.body_ang_vel_w.squeeze(1)[..., 2]
    return entity_ang_vel_z.unsqueeze(1)


def heading_rate(env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Returns the heading rate of the robot in the terrain frame."""
    robot: RigidObject | Articulation = env.scene[robot_cfg.name]
    robot_ang_vel_z = robot.data.root_ang_vel_w[:, 2]
    return robot_ang_vel_z.unsqueeze(1)


def base_height(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """Get the base height from the raycast sensors in the scene."""
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    hits = sensor.data.ray_hits_w[..., 2].mean(dim=1)
    return (asset.data.root_pos_w[:, 2] - hits).unsqueeze(1)


def roll_pitch(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Returns the roll and pitch of the robot in the terrain frame."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    asset_quat = asset.data.root_quat_w
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(asset_quat)
    return torch.cat([math_utils.wrap_to_pi(roll).unsqueeze(1), math_utils.wrap_to_pi(pitch).unsqueeze(1)], dim=-1)


class heading_cumulative(ManagerTermBase):
    """
    Returns the heading rate integrated over time.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._internal_heading = torch.zeros(env.num_envs, 1).to(env.device)

    def __call__(self, env: ManagerBasedEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """Returns the heading angle of the robot in the terrain frame."""
        robot: RigidObject | Articulation = env.scene[robot_cfg.name]
        robot_ang_vel_z = robot.data.root_ang_vel_w[:, 2]

        # integrate the angular velocity to get the heading
        self._internal_heading += robot_ang_vel_z.unsqueeze(1) * env.step_dt
        return self._internal_heading

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Resets the internal heading to zero for the given environment ids."""
        self._internal_heading[env_ids] = 0.0


def velocity_2d_w(env: ManagerBasedEnv, entity_cfg: SceneEntityCfg) -> torch.Tensor:
    """Returns the velocity vector of the entity in the terrain frame."""
    entity: RigidObject | Articulation = env.scene[entity_cfg.name]
    entity_vel_w = get_robot_lin_vel_w(entity)
    entity_ang_vel_z = get_robot_rot_vel_w(entity)[..., 2]
    return torch.cat([entity_vel_w[..., :2], entity_ang_vel_z.unsqueeze(1)], dim=-1)


def velocity_3d_w(env: ManagerBasedEnv, entity_cfg: SceneEntityCfg) -> torch.Tensor:
    """Returns the linear and angular velocity vector in the world frame"""
    entity: RigidObject | Articulation = env.scene[entity_cfg.name]
    entity_vel_w = get_robot_lin_vel_w(entity)
    entity_angvel_w = get_robot_rot_vel_w(entity)
    return torch.cat([entity_vel_w, entity_angvel_w], dim=-1)


##
# - High level actions as observations
##


def action_command(env: ManagerBasedEnv, action_name: str) -> torch.Tensor:
    """Returns the action command as an observation."""
    return env.action_manager._terms[action_name].processed_actions


def last_low_level_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    return env.action_manager._terms[action_name].prev_low_level_actions  # type: ignore


##
# - goal
##


def dist_to_goal(env: ManagerBasedRLEnv, entity_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
    """Returns the distance to the goal for the given entity."""
    entity: RigidObject | Articulation = env.scene[entity_cfg.name]
    goal_cmd_geneator: GoalCommand = env.command_manager._terms[command_name]  # type: ignore

    entity_pos = get_robot_pos(entity)
    goal_pos = goal_cmd_geneator.goal_pos_w

    diff = torch.linalg.norm(entity_pos - goal_pos, dim=-1).unsqueeze(1)
    return diff


##
# - video
##


class video_recorder(ManagerTermBase):
    """
    Dummy observation term that records videos of the environment.
    These videos are not intended to be used as observations but for debugging purposes.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):

        super().__init__(cfg, env)
        self.video_intervall = 12500

        self.record_video = False
        self.attach_pos = True
        self.attach_yaw = True
        self.video_dict = {}
        self.video_counter = 0
        self.step_counter = 0
        self.num_frames_dict = {}

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        camera: str = "tiled_camera",
    ) -> torch.Tensor:
        """S
        Args:
            env: The learning environment.
            env_ids: The list of reset environments.
            camera: The name of the camera sensor to record.
        """
        # get sensor
        sensor: TiledCamera = env.scene.sensors[camera]  # type: ignore

        cam_env_ids = sensor.cam_env_ids
        cam_ids = sensor._ALL_INDICES

        # check if we should start to record videos
        if self.step_counter % self.video_intervall == 10:
            self.record_video = True

            # clear image stack
            for env_id in cam_env_ids.cpu().numpy():  # type: ignore
                self.num_frames_dict[env_id] = 0

        if self.record_video:
            # update camera positions
            if self.attach_pos:
                cam_positions = get_robot_pos(env.scene["robot"])[sensor.cam_env_ids] + torch.tensor(
                    sensor.cfg.offset.pos
                ).to(
                    sensor.cam_env_ids.device  # type: ignore
                )  # type: ignore
            else:
                cam_positions = sensor.data.pos_w
            if self.attach_yaw:
                cam_quat = math_utils.quat_mul(
                    math_utils.yaw_quat(get_robot_quat(env.scene["robot"])[sensor.cam_env_ids]),
                    torch.tensor(sensor.cfg.offset.rot)
                    .repeat(len(sensor.cam_env_ids), 1)
                    .to(sensor.cam_env_ids.device),  # type: ignore
                )

            else:
                cam_quat = (
                    torch.tensor(sensor.cfg.offset.rot).repeat(len(sensor.cam_env_ids), 1).to(sensor.cam_env_ids.device)  # type: ignore
                )

            sensor.set_world_poses(positions=cam_positions, orientations=cam_quat, convention="world")

            # record such that we start at a new episode
            start_env_ids = env.termination_manager.dones.nonzero()
            isfull = True
            env_frames = sensor.data.output["rgb"].cpu().numpy()
            for env_id, cam_id in zip(cam_env_ids.cpu().numpy(), cam_ids.cpu().numpy()):  # type: ignore
                self.video_dict[env_id] = None
                # start recording if env was reset
                if env_id in start_env_ids and self.num_frames_dict[env_id] == 0:
                    self.video_dict[env_id] = env_frames[cam_id]
                    self.num_frames_dict[env_id] += 1
                # if we started before, keep recording until the video length is reached
                elif 0 < self.num_frames_dict[env_id] < env.max_episode_length:
                    self.video_dict[env_id] = env_frames[cam_id]
                    self.num_frames_dict[env_id] += 1

                # check if we are done
                isfull &= self.num_frames_dict[env_id] == env.max_episode_length

            if isfull:
                self.record_video = False
                self.video_counter += 1

            env.eval_video_frame = self.video_counter, self.video_dict  # type: ignore
        else:
            env.eval_video_frame = None  # type: ignore

        self.step_counter += 1
        return torch.tensor(self.video_counter)


##
# - regularizers
##
class regularization_reward_obs(ManagerTermBase):
    """
    Returns regularization rewards
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.regularization_rewards_sum = {
            term: torch.zeros(env.num_envs).to(env.device) for term in cfg.params["terms"].keys()  # type: ignore
        }

        # resolve entities
        for name, term_config in cfg.params["terms"].items():  # type: ignore
            for key, value in term_config.params.items():
                if isinstance(value, SceneEntityCfg):
                    # load the entity
                    try:
                        value.resolve(self._env.scene)
                    except ValueError as e:
                        raise ValueError(f"Error while parsing '{name}:{key}'. {e}")

    def __call__(self, env: ManagerBasedRLEnv, terms: dict[str, RewardTermCfg]) -> torch.Tensor:
        """
        Returns regularization rewards
        """
        reward = torch.zeros(env.num_envs).to(env.device)
        for term_name, term_cfg in terms.items():
            if term_cfg.weight == 0.0:
                continue
            value = term_cfg.weight * term_cfg.func(env, **term_cfg.params) * env.step_dt
            reward += value
            # logging
            self.regularization_rewards_sum[term_name] += value.mean().item()
        return reward.unsqueeze(1)

    def reset(self, env_ids: slice = slice(None)):
        # as in reward manager
        if not hasattr(self._env, "extras"):
            return

        extras = {}
        for term, reward_sum in self.regularization_rewards_sum.items():
            episodic_sum_avg = torch.mean(reward_sum[env_ids]) / self._env.max_episode_length_s  # type: ignore
            extras["Episode_Regularizer_Reward/" + term] = episodic_sum_avg
            self.regularization_rewards_sum[term] *= 0

        if "log" in self._env.extras:
            self._env.extras["log"].update(extras)


def is_env_inactive(env: ManagerBasedRLEnv, rest_duration_s: float) -> torch.Tensor:
    """Check if the environment is in the rest phase."""
    # hacks the fact that episode length is initialized in the env after managers are initialized
    if hasattr(env, "episode_length_buf"):
        return (env.episode_length_buf < int(rest_duration_s / env.step_dt)).float().unsqueeze(1)
    else:
        return torch.ones(env.num_envs, 1, device=env.device)
