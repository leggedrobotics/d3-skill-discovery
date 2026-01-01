# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause


from dataclasses import MISSING

from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.utils import configclass

from . import hf_terrains


@configclass
class CellBorderCfg(HfTerrainBaseCfg):
    """Configuration for cell border wall."""

    function = hf_terrains.cell_border

    height: float = 2.0

    corner_witdh: float = 0.0


@configclass
class RandomPyramid(HfTerrainBaseCfg):
    """Configuration for cell border wall."""

    function = hf_terrains.random_pyramid

    wall_height: float = 2.0

    step_height: float = 0.5

    min_width: float = 0.5
    """minimal step with"""

    force_to_corner: bool = True
    """if true, all upper levels will be forced to the corner of the lower level.
    This will omit the min_width parameter"""

    origin_z: float = 0.0


@configclass
class HfRandomUniformTerrainDifficultyCfg(HfTerrainBaseCfg):
    """Configuration for a random uniform height field terrain."""

    function = hf_terrains.random_uniform_terrain_difficulty

    noise_range: tuple[float, float] = MISSING
    """The minimum and maximum height noise (i.e. along z) of the terrain (in m)."""
    noise_step: float = MISSING
    """The minimum height (in m) change between two points."""
    downsampled_scale: float | None = None
    """The distance between two randomly sampled points on the terrain. Defaults to None,
    in which case the :obj:`horizontal scale` is used.

    The heights are sampled at this resolution and interpolation is performed for intermediate points.
    This must be larger than or equal to the :obj:`horizontal scale`.
    """


@configclass
class HfRandomBoxesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a random uniform height field terrain."""

    function = hf_terrains.random_uniform_boxes_terrain

    noise_range: tuple[float, float] = (0.02, 0.05)
    """The minimum and maximum height noise (i.e. along z) of the terrain (in m)."""
    noise_step: float = 0.025
    """The minimum height (in m) change between two points."""
    downsampled_scale: float | None = None
    """The distance between two randomly sampled points on the terrain. Defaults to None,
    in which case the :obj:`horizontal scale` is used.

    The heights are sampled at this resolution and interpolation is performed for intermediate points.
    This must be larger than or equal to the :obj:`horizontal scale`.
    """

    num_boxes_range: tuple[int, int] = (10, 10)
    """The minimum and maximum number of boxes to place on the terrain."""

    box_height_range: tuple[float, float] = (0.1, 0.5)
    """Min and max height (in meters) for each 'box' placed on top of the terrain."""

    box_width_range: tuple[float, float] = (0.2, 1.0)
    """Min and max width (in meters, along X) of each box's footprint."""

    box_depth_range: tuple[float, float] = (0.2, 1.0)
    """Min and max depth (in meters, along Y) of each box's footprint."""

    platform_width: float = 1.0
    """The width of the central square platform area (in meters) where no boxes are spawned."""


# @configclass
# class HfRandomUniformWallsTerrainCfg(HfTerrainBaseCfg):
#     """Configuration for a random uniform height-field terrain with 4 walls along each edge."""

#     function = hf_terrains.random_uniform_walls_terrain

#     noise_range: tuple[float, float] = (0.02, 0.05)
#     """The minimum and maximum height noise (i.e. along z) of the terrain (in m)."""
#     noise_step: float = 0.025
#     """The minimum height (in m) change between two points."""
#     downsampled_scale: float | None = None
#     """The distance between two randomly sampled points on the terrain. Defaults to None,
#     in which case the :obj:`horizontal scale` is used.

#     The heights are sampled at this resolution and interpolation is performed for intermediate points.
#     This must be larger than or equal to the :obj:`horizontal scale`.
#     """
#     wall_height: float = 1.0
#     """The total height of each wall (in meters) above the local terrain surface."""

#     wall_width: float = 0.2
#     """The thickness of the walls in the inward direction (in meters)."""

#     wall_length: tuple[float, float] = (1.0, 3.0)
#     """The length of each wall (in meters). If this is smaller than the terrain size,
#     corners remain open. If equal or larger, the terrain is fully enclosed."""


@configclass
class HfRandomUniformWallsTerrainCfg(HfTerrainBaseCfg):
    """
    Configuration for a random uniform height-field terrain with only a north and east wall.

    The terrain is square, and the north wall is randomly sized and shifted along the top edge,
    while the east wall is randomly sized and shifted along the right edge.
    """

    function = hf_terrains.random_uniform_walls_terrain

    noise_range: tuple[float, float] = (0.02, 0.05)
    """Minimum and maximum height noise for the random uniform terrain base (in m)."""

    noise_step: float = 0.025
    """Minimum height change between sampled points (in m)."""

    downsampled_scale: float | None = None
    """Sampling resolution for generating the rough terrain. Defaults to None,
    in which case the :obj:`horizontal_scale` is used.
    Must be >= horizontal_scale."""

    wall_height: float = 1.0
    """Height of each wall above local terrain (in meters)."""

    wall_width: float = 0.2
    """Thickness of each wall into the terrain (in meters)."""

    wall_length_range: tuple[float, float] = (1.0, 3.0)
    """Min and max possible length of the wall in meters along its main axis."""

    wall_long_shift_range: tuple[float, float] = (0.0, 1.0)
    """Range (in meters) for shifting the wall along its length axis."""

    wall_thickness_shift_range: tuple[float, float] = (0.0, 0.2)
    """Range (in meters) for how far inside from the terrain boundary
    the wall is placed (shift along the thickness axis)."""
