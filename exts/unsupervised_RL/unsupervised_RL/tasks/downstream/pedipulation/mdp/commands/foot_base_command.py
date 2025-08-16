from __future__ import annotations

import copy
import torch
from typing import TYPE_CHECKING, Sequence

from omni.usd.commands import DeletePrimsCommand

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.assets.articulation import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import (
    BLUE_ARROW_X_MARKER_CFG,
    CUBOID_MARKER_CFG,
    FRAME_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
)
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply_yaw,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
    subtract_frame_transforms,
    wrap_to_pi,
    yaw_quat,
)

if TYPE_CHECKING:
    from unsupervised_RL.tasks.downstream.pedipulation.mdp import FootBasePositionCommandCfg

    from isaaclab.envs import ManagerBasedRLEnv


CYLINDER_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "cylinder": sim_utils.CylinderCfg(
            radius=0.02,
            height=1.0,  # needs to be adjusted to the sizes of edges of the command space
            axis="X",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 1.0)),
        )
    }
)


class FootBasePositionCommand(CommandTerm):
    """TODO: Description"""

    cfg: FootBasePositionCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: FootBasePositionCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        super().__init__(cfg, env)

        self.env_cfg = env.cfg  # type: ignore
        # - robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # - commands
        self.base_pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_pos_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.heading_command_b = torch.zeros(self.num_envs, device=self.device)
        self.foot_pos_commands_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.foot_pos_commands_w = torch.zeros(self.num_envs, 3, device=self.device)

        # - misc
        self.tracking_error_sum = torch.zeros(self.num_envs, 3, device=self.device)
        self.log_step_counter = torch.zeros(self.num_envs, device=self.device)
        self.feet_ids, _ = self.robot.find_bodies(".*FOOT")

    def __str__(self) -> str:
        msg = "FootPositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired commands in body frame"""
        return torch.cat(
            [self.foot_pos_commands_b, self.base_pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1
        )

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        """Randomly select commands of some environments."""
        if len(env_ids) == 0:
            return

        # set tracking error to zero
        self.tracking_error_sum[env_ids] = 0.0
        self.log_step_counter[env_ids] = 0.0

        # sample random base position and heading
        self.base_pos_command_w[env_ids] = self._env.scene.env_origins[env_ids]
        r = torch.empty(len(env_ids), device=self.device)
        self.base_pos_command_w[env_ids, 0] += r.uniform_(*self.cfg.base_ranges.pos_x)
        self.base_pos_command_w[env_ids, 1] += r.uniform_(*self.cfg.base_ranges.pos_y)
        self.base_pos_command_w[env_ids, 2] += self.robot.data.default_root_state[env_ids, 2]
        self.base_heading_command_w[env_ids] = r.uniform_(*self.cfg.base_ranges.heading)

        # sample random foot position
        self.foot_pos_commands_b[env_ids, 0] = r.uniform_(*self.cfg.foot_ranges.pos_x)
        self.foot_pos_commands_b[env_ids, 1] = r.uniform_(*self.cfg.foot_ranges.pos_y)
        self.foot_pos_commands_b[env_ids, 2] = r.uniform_(*self.cfg.foot_ranges.pos_z)

    def _update_command(self):
        # position of goal in body frame
        target_vec = self.base_pos_command_w - self.robot.data.root_pos_w[:, :3]
        self.base_pos_command_b[:] = quat_rotate_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)
        # heading command in body frame
        self.heading_command_b = wrap_to_pi(self.base_heading_command_w - self.robot.data.heading_w)
        # foot position is already in body frame

        self._log_data()

    def _log_data(self):
        # log data used in the curriculum
        foot_pos_w = self.robot.data.body_state_w[:, self.feet_ids[self.env_cfg.foot_index], :3]
        foot_pos_b = quat_rotate_inverse(self.robot.data.root_quat_w, foot_pos_w - self.robot.data.root_pos_w[:, :3])

        self.tracking_error_sum += torch.abs(self.foot_pos_commands_b - foot_pos_b)
        self.log_step_counter += 1.0

    def _update_metrics(self):
        self.metrics["error_base_pos_2d"] = torch.norm(
            self.base_pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1
        )
        self.metrics["error_base_heading"] = torch.abs(
            wrap_to_pi(self.base_heading_command_w - self.robot.data.heading_w)
        )
        self.metrics["foot_tracking_error"] = torch.where(
            self.log_step_counter.bool(),
            torch.norm(self.tracking_error_sum / self.log_step_counter.unsqueeze(1), dim=1),
            self.log_step_counter,
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            # create the visualization markers if they do not exist
            if not hasattr(self, "foot_command_space_visualizers"):
                self._create_foot_command_space_visualizers()
            if not hasattr(self, "foot_position_command_visualizer"):
                self._create_foot_position_command_visualizer()
            if not hasattr(self, "foot_position_visualizer"):
                self._create_foot_position_visualizer()
            if not hasattr(self, "goal_pose_visualizer"):
                self._create_goal_pose_visualizer()
            # set their visibility to true

            # make the visualization markers visible
            self.foot_position_command_visualizer.set_visibility(self.cfg.commanded_foot_position_debug_vis)
            self.foot_position_visualizer.set_visibility(self.cfg.foot_position_debug_vis)
            for marker in self.command_space_visualizers:
                marker.set_visibility(self.cfg.foot_command_space_debug_vis)
        else:
            # make all visualization markers to invisible
            if hasattr(self, "command_space_visualizers"):
                for marker in self.command_space_visualizers:
                    marker.set_visibility(False)
            if hasattr(self, "foot_position_command_visualizer"):
                self.foot_position_command_visualizer.set_visibility(False)
            if hasattr(self, "foot_position_visualizer"):
                self.foot_position_visualizer.set_visibility(False)

    def _create_foot_command_space_visualizers(self):

        # delete existing command space visualization markers
        if hasattr(self, "command_space_visualizers"):
            for marker in self.command_space_visualizers:
                # turn off the visibility of the visualization marker
                marker.set_visibility(False)
                # delete the visualization marker
                DeletePrimsCommand(paths=[marker.prim_path])
            del self.command_space_visualizers
        # a cuboid has 12 edges, which we need to find the length, center position and orientation of
        self.command_space_visualizers = []
        self.edge_positions = torch.zeros(12, self.num_envs, 3, device=self.device)
        self.edge_orientations = torch.zeros(12, self.num_envs, 4, device=self.device)

        # compute the 8 corners of the command space
        corners: torch.tensor = torch.zeros([2, 2, 2, 3], device=self.device)  # min and max values for x, y, z
        for i in [0, 1]:
            for j in [0, 1]:
                for k in [0, 1]:
                    # Use only the first environment for now (current_command_space[0, ...])
                    corners[i, j, k, :] = torch.tensor(
                        [
                            self.cfg.foot_ranges.pos_x[i],
                            self.cfg.foot_ranges.pos_y[j],
                            self.cfg.foot_ranges.pos_z[k],
                        ],
                        device=self.device,
                    )

        # compute the 12 edges of the command space
        edge_lengths = torch.zeros(12, device=self.device)
        edge_positions = torch.zeros(12, 3, device=self.device)
        edge_orientations = torch.zeros(12, 4, device=self.device)  # quaternions

        # compute the 4 edges along x
        for i in range(2):
            for j in range(2):
                edge_lengths[i * 2 + j] = torch.linalg.norm(corners[i, j, 0] - corners[i, j, 1], dim=-1)
                edge_positions[i * 2 + j] = (corners[i, j, 0] + corners[i, j, 1]) / 2.0
                angle = torch.acos(
                    torch.dot(corners[i, j, 0] - corners[i, j, 1], torch.tensor([1.0, 0.0, 0.0], device=self.device))
                    / edge_lengths[i * 2 + j]
                )
                axis = torch.linalg.cross(
                    corners[i, j, 0] - corners[i, j, 1], torch.tensor([1.0, 0.0, 0.0], device=self.device)
                )
                edge_orientations[i * 2 + j] = quat_from_angle_axis(angle, axis)
        # compute the 4 edges along y
        for i in range(2):
            for j in range(2):
                edge_lengths[4 + i * 2 + j] = torch.linalg.norm(corners[i, 0, j] - corners[i, 1, j], dim=-1)
                edge_positions[4 + i * 2 + j] = (corners[i, 0, j] + corners[i, 1, j]) / 2.0
                angle = torch.acos(
                    torch.dot(corners[i, 0, j] - corners[i, 1, j], torch.tensor([1.0, 0.0, 0.0], device=self.device))
                    / edge_lengths[4 + i * 2 + j]
                )
                axis = torch.linalg.cross(
                    corners[i, 0, j] - corners[i, 1, j], torch.tensor([1.0, 0.0, 0.0], device=self.device)
                )
                edge_orientations[4 + i * 2 + j] = quat_from_angle_axis(angle, axis)
        # compute the 4 edges along z
        for i in range(2):
            for j in range(2):
                edge_lengths[8 + i * 2 + j] = torch.linalg.norm(corners[0, i, j] - corners[1, i, j], dim=-1)
                edge_positions[8 + i * 2 + j] = (corners[0, i, j] + corners[1, i, j]) / 2.0
                angle = torch.acos(
                    torch.dot(corners[0, i, j] - corners[1, i, j], torch.tensor([1.0, 0.0, 0.0], device=self.device))
                    / edge_lengths[8 + i * 2 + j]
                )
                axis = torch.linalg.cross(
                    corners[0, i, j] - corners[1, i, j], torch.tensor([1.0, 0.0, 0.0], device=self.device)
                )
                edge_orientations[8 + i * 2 + j] = quat_from_angle_axis(angle, axis)

        # we assume that the command space is the same for all environments
        self.edge_positions = edge_positions
        self.edge_orientations = edge_orientations

        # create and visualize the 12 markers
        marker_cfg = copy.copy(CYLINDER_MARKER_CFG)
        cylinder_marker: sim_utils.CylinderCfg = marker_cfg.markers["cylinder"]
        cylinder_marker.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0))
        for i in range(12):
            marker_cfg.prim_path = f"/Visuals/Command/Command_space_edge_{i}"
            cylinder_marker.height = edge_lengths[i].item()
            marker = VisualizationMarkers(marker_cfg)
            self.command_space_visualizers.append(marker)

    def _create_foot_position_command_visualizer(self):
        marker_cfg = copy.copy(CUBOID_MARKER_CFG)
        marker_cfg.prim_path = "/Visuals/Command/position_goal"
        cuboid_marker: sim_utils.CuboidCfg = marker_cfg.markers["cuboid"]
        cuboid_marker.size = (0.1, 0.1, 0.1)
        self.foot_position_command_visualizer = VisualizationMarkers(marker_cfg)

    def _create_goal_pose_visualizer(self):
        marker_cfg = copy.copy(GREEN_ARROW_X_MARKER_CFG)
        marker_cfg.prim_path = "/Visuals/Command/pose_goal"
        marker_cfg.markers["arrow"].scale = (0.2, 0.2, 0.8)
        self.goal_pose_visualizer = VisualizationMarkers(marker_cfg)

    def _create_foot_position_visualizer(self):
        marker_cfg = copy.copy(CUBOID_MARKER_CFG)
        marker_cfg.prim_path = "/Visuals/Command/foot_position"
        cuboid_marker: sim_utils.CuboidCfg = marker_cfg.markers["cuboid"]
        cuboid_marker.size = (0.1, 0.1, 0.1)
        cuboid_marker.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
        self.foot_position_visualizer = VisualizationMarkers(marker_cfg)

    def _debug_vis_callback(self, event):
        # update the box marker
        self.goal_pose_visualizer.visualize(
            translations=self.base_pos_command_w,
            orientations=quat_from_euler_xyz(
                torch.zeros_like(self.base_heading_command_w),
                torch.zeros_like(self.base_heading_command_w),
                self.base_heading_command_w,
            ),
        )
        # visualize the position of the pedipulating foot
        asset: RigidObject = self._env.scene["robot"]
        foot_pos_w = asset.data.body_state_w[:, self.feet_ids[self.env_cfg.foot_index], :3]
        self.foot_position_visualizer.visualize(foot_pos_w)

        # visualize the actual foot position
        self.foot_pos_commands_w = (
            quat_rotate(yaw_quat(asset.data.root_quat_w), self.foot_pos_commands_b)
        ) + asset.data.root_pos_w[:, :3]

        self.foot_position_command_visualizer.visualize(self.foot_pos_commands_w)
        # visualize the command space
        for i, marker in enumerate(self.command_space_visualizers):

            edge_position_w, edge_orientation_w = combine_frame_transforms(
                asset.data.root_pos_w,
                yaw_quat(asset.data.root_quat_w),
                self.edge_positions[i].repeat(self.num_envs, 1),
                self.edge_orientations[i].repeat(self.num_envs, 1),
            )

            marker.visualize(
                translations=edge_position_w,
                orientations=edge_orientation_w,
            )
