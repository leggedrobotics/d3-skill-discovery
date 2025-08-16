# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

import isaaclab.terrains as terrain_gen
import isaaclab.terrains.trimesh.utils as mesh_utils_terrains

from . import mesh_terrains as mesh_terrains


@configclass
class MeshPyramidTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a pyramid stair mesh terrain."""

    function = mesh_terrains.pyramid_terrain

    border_width: float = 0.0
    """The width of the border around the terrain (in m). Defaults to 0.0.

    The border is a flat terrain with the same height as the terrain.
    """
    step_height: float = MISSING
    """The height of the steps (in m)."""
    step_width: tuple[float, float] = MISSING
    """The width of the steps (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    walls: bool = False
    """If True, each terrain is surrounded by walls. Defaults to False."""
    wall_height: float = 2.0
    """The height of the walls (in m). Defaults to 2.0."""
    wall_thickness: float = 0.1
    """The thickness of the walls (in m). Defaults to 0.1."""
    type: Literal["random", "spiral", "pyramid"] = "random"
    """The type of the terrain. Defaults to "random"."""

    holes: bool = False
    """If True, the terrain will have holes in the steps. Defaults to False.
    
    
    

    If :obj:`holes` is True, the terrain will have pyramid stairs of length or width
    :obj:`platform_width` (depending on the direction) with no steps in the remaining area. Additionally,
    no border will be added.
    """


@configclass
class MeshRandomBoxesTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain consisting of a flat ground with random boxes."""

    function = mesh_terrains.random_boxes_terrain

    num_boxes_range: tuple[int, int] = (10, 10)
    """
    The number of boxes to place on the terrain.
    """

    box_height_range: tuple[float, float] = (0.1, 0.5)
    """
    Min and max height of each randomly generated box.
    """

    box_width_range: tuple[float, float] = (0.2, 1.0)
    """
    Min and max width (size in X direction) of each randomly generated box.
    """

    box_depth_range: tuple[float, float] = (0.2, 1.0)
    """
    Min and max depth (size in Y direction) of each randomly generated box.
    """

    platform_width: float = 1.0
    """
    The width of the central square platform area (in meters) where no boxes are spawned.
    """


@configclass
class MeshFourWallsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a square, flat terrain with four walls in the middle of each side."""

    function = mesh_terrains.four_walls_terrain

    floor_thickness: float = 0.0
    """Thickness of the flat floor in meters."""

    wall_height: float = 1.0
    """Height of each wall in meters."""

    wall_length: float = 1.0
    """Length of each wall segment in meters.

    If equal to ``size[0]`` (the terrain size), the terrain is fully enclosed. If less than ``size[0]``,
    the corners remain open.
    """

    wall_thickness: float = 0.1
    """Thickness (width) of the wall segments in meters."""
