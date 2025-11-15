# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from .foot_base_command import FootBasePositionCommand


@configclass
class FootBasePositionCommandCfg(CommandTermCfg):
    """Configuration for the foot position and base pose command generator."""

    class_type: type = FootBasePositionCommand

    asset_name: str = ""

    @configclass
    class BaseRanges:
        """Uniform distribution ranges for the base position commands.
        Defined in the environment frame."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""
        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""
        heading: tuple[float, float] = MISSING
        """Heading range for the position commands (in rad)."""

    @configclass
    class FootRanges:
        """Uniform distribution ranges for the base position commands
        Defines a box in body frame."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""
        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""
        pos_z: tuple[float, float] = MISSING
        """Range for the z position (in m)."""

    base_ranges: BaseRanges = MISSING
    """Distribution ranges for the base position commands."""

    foot_ranges: FootRanges = MISSING
    """Distribution ranges for the foot position commands."""

    # debug visualization options
    foot_command_space_debug_vis: bool = True
    """Creates a box in the body frame to visualize the foot command space."""
    commanded_foot_position_debug_vis: bool = True
    foot_position_debug_vis: bool = True
    base_pose_debug_vis: bool = True

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose_goal"
    )
    goal_pose_visualizer_cfg.markers["arrow"].scale = (0.2, 0.2, 0.8)
