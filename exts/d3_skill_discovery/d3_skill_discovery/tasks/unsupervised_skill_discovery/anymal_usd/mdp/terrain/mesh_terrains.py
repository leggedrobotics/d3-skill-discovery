# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause


"""Functions to generate different terrains using the ``trimesh`` library."""

from __future__ import annotations

import numpy as np
import random
import trimesh
from typing import TYPE_CHECKING

from isaaclab.terrains.trimesh.utils import *  # noqa: F401, F403
from isaaclab.terrains.trimesh.utils import make_border

if TYPE_CHECKING:
    from . import mesh_terrains_cfg


def pyramid_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshPyramidTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a pyramid stair pattern where each level is placed at a random corner of the lower level.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # Resolve the terrain configuration
    step_height = cfg.step_height  # + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])

    # Compute number of steps
    step_width = cfg.step_width[0] + difficulty * (cfg.step_width[1] - cfg.step_width[0])

    max_steps_x = (cfg.size[0] - 2 * cfg.border_width) // (2 * step_width)
    max_steps_y = (cfg.size[1] - 2 * cfg.border_width) // (2 * step_width)
    num_steps = int(min(max_steps_x, max_steps_y))

    # Initialize list of meshes
    meshes_list = []

    if cfg.walls:
        wall_height = cfg.wall_height
        wall_thickness = cfg.wall_thickness
        # south wall
        center_south = [wall_thickness / 2, cfg.size[1] / 2, wall_height / 2]
        dims = [wall_thickness, cfg.size[1], wall_height]
        wall_box = trimesh.creation.box(dims, trimesh.transformations.translation_matrix(center_south))
        meshes_list.append(wall_box)
        # north wall
        center_east = [cfg.size[1] / 2, wall_thickness / 2, wall_height / 2]
        dims = [cfg.size[0], wall_thickness, wall_height]
        wall_box = trimesh.creation.box(dims, trimesh.transformations.translation_matrix(center_east))
        meshes_list.append(wall_box)

    # Generate the border if needed
    if cfg.border_width > 0.0:
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -step_height / 2]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        make_borders = make_border(cfg.size, border_inner_size, step_height, border_center)
        meshes_list += make_borders

    # Initialize variables for the base level
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    prev_box_size = terrain_size
    prev_box_pos = terrain_center
    prev_box_height = 0.0

    # Create the base level
    base_box_dims = (prev_box_size[0], prev_box_size[1], prev_box_height)
    base_box = trimesh.creation.box(base_box_dims, trimesh.transformations.translation_matrix(prev_box_pos))
    meshes_list.append(base_box)

    # Iterate through each level
    for k in range(1, num_steps + 1):
        # Reduce the size of the box for the next level
        box_size = (prev_box_size[0] - 2 * step_width, prev_box_size[1] - 2 * step_width)
        # Ensure the box size remains positive
        if box_size[0] <= cfg.platform_width or box_size[1] <= cfg.platform_width:
            break

        # Randomly select one of the four corners
        corners = [
            (-0.5, -0.5),  # Lower-left corner
            (-0.5, 0.5),  # Upper-left corner
            (0.5, -0.5),  # Lower-right corner
            (0.5, 0.5),  # Upper-right corner
        ]
        if cfg.type == "random":
            corner = random.choice(corners)
        elif cfg.type == "spiral":
            corner = corners[k % 4]
        else:
            corner = corners[0]

        # Calculate the position of the new box
        offset_x = corner[0] * (prev_box_size[0] - box_size[0])
        offset_y = corner[1] * (prev_box_size[1] - box_size[1])
        box_pos_x = prev_box_pos[0] + offset_x
        box_pos_y = prev_box_pos[1] + offset_y
        box_pos_z = prev_box_pos[2] + 0.5 * prev_box_height + 0.5 * step_height
        box_pos = (box_pos_x, box_pos_y, box_pos_z)

        # Create the new box
        box_dims = (box_size[0], box_size[1], step_height)
        new_box = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        meshes_list.append(new_box)

        # Update variables for the next iteration
        prev_box_size = box_size
        prev_box_pos = box_pos
        prev_box_height = step_height

    origin = np.array([terrain_center[0], terrain_center[1], 0])

    return meshes_list, origin


def random_boxes_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshRandomBoxesTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """
    Generate a flat terrain of size (cfg.size[0] x cfg.size[1]) with random boxes placed on top.
    No boxes will be spawned inside the central 'platform' region (square).
    Boxes do not overlap in the horizontal plane.

    The 'difficulty' can be used to scale box sizes or number of boxes, as desired.

    Args:
        difficulty: Float in [0, 1]. Higher means more/ larger boxes (depending on your design).
        cfg: MeshRandomBoxesTerrainCfg containing all necessary configuration.

    Returns:
        (meshes_list, origin)
          - meshes_list: A list of trimesh.Trimesh objects representing the flat plane plus all boxes.
          - origin: The origin (np.ndarray of shape (3,)) of the terrain in world coordinates.
    """

    size_x, size_y = cfg.size
    if size_x <= 0 or size_y <= 0:
        raise ValueError(f"Terrain size must be positive. Got size={cfg.size}")

    box_height_min, box_height_max = cfg.box_height_range
    box_width_min, box_width_max = cfg.box_width_range
    box_depth_min, box_depth_max = cfg.box_depth_range
    num_boxes = int(cfg.num_boxes_range[0] + difficulty * (cfg.num_boxes_range[1] - cfg.num_boxes_range[0]))
    terrain_thickness = 0.05

    plane_dim = (size_x, size_y, terrain_thickness)
    shift_plane = trimesh.transformations.translation_matrix([0.5 * size_x, 0.5 * size_y, -0.5 * terrain_thickness])
    plane_mesh = trimesh.creation.box(plane_dim, shift_plane)

    meshes_list = [plane_mesh]

    def boxes_overlap_2d(rect_a, rect_b):
        """Check if two axis-aligned rectangles in [x, y, w, d] form overlap."""
        ax, ay, aw, ad = rect_a
        bx, by, bw, bd = rect_b
        if (ax + aw <= bx) or (bx + bw <= ax):
            return False
        if (ay + ad <= by) or (by + bd <= ay):
            return False
        return True

    placed_rects = []
    half_plat = 0.5 * cfg.platform_width
    min_cx = 0.5 * size_x - half_plat
    max_cx = 0.5 * size_x + half_plat
    min_cy = 0.5 * size_y - half_plat
    max_cy = 0.5 * size_y + half_plat

    rng = np.random.default_rng()
    max_attempts = 1000

    for _ in range(num_boxes):
        attempt = 0
        placed = False
        while attempt < max_attempts and not placed:
            attempt += 1

            w = rng.uniform(box_width_min, box_width_max)
            d = rng.uniform(box_depth_min, box_depth_max)
            h = rng.uniform(box_height_min, box_height_max)

            x = rng.uniform(0.0, size_x - w)
            y = rng.uniform(0.0, size_y - d)

            box_min_x = x
            box_max_x = x + w
            box_min_y = y
            box_max_y = y + d

            overlap_center = not (
                (box_max_x <= min_cx) or (box_min_x >= max_cx) or (box_max_y <= min_cy) or (box_min_y >= max_cy)
            )
            if overlap_center:
                continue

            rect_proposed = (box_min_x, box_min_y, w, d)
            no_collision = True
            for rect in placed_rects:
                if boxes_overlap_2d(rect, rect_proposed):
                    no_collision = False
                    break

            if no_collision:
                placed_rects.append(rect_proposed)
                placed = True

                shift_box = trimesh.transformations.translation_matrix([x + 0.5 * w, y + 0.5 * d, 0.5 * h])
                box_mesh = trimesh.creation.box((w, d, h), shift_box)
                meshes_list.append(box_mesh)

    origin = np.array([0.5 * size_x, 0.5 * size_y, 0.0])

    return meshes_list, origin


def four_walls_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshFourWallsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Create a flat, square terrain with four walls centered on each edge.

    Args:
        difficulty: Unused in this terrain (kept for interface consistency).
        cfg: Configuration dataclass for the four-walls terrain.

    Returns:
        A tuple of:
         - A list of Trimesh objects for the floor and walls.
         - The terrain origin (np.ndarray) in [x, y, z].
    """
    if cfg.size[0] != cfg.size[1]:
        raise ValueError(f"The terrain must be square. Received size: {cfg.size}.")

    terrain_size = cfg.size[0]

    meshes_list = []

    floor_dim = (terrain_size, terrain_size, cfg.floor_thickness)
    # Place the floor so its top is at z = 0
    floor_pos = (0.5 * terrain_size, 0.5 * terrain_size, -0.5 * cfg.floor_thickness)
    floor_box = trimesh.creation.box(floor_dim, trimesh.transformations.translation_matrix(floor_pos))
    meshes_list.append(floor_box)

    half_wall_thk = 0.5 * cfg.wall_thickness
    half_terr = 0.5 * terrain_size
    half_wall_ht = 0.5 * cfg.wall_height

    top_wall_dim = (cfg.wall_length, cfg.wall_thickness, cfg.wall_height)
    top_wall_pos = (half_terr, terrain_size - half_wall_thk, half_wall_ht)
    top_wall = trimesh.creation.box(top_wall_dim, trimesh.transformations.translation_matrix(top_wall_pos))
    meshes_list.append(top_wall)

    bottom_wall_dim = (cfg.wall_length, cfg.wall_thickness, cfg.wall_height)
    bottom_wall_pos = (half_terr, half_wall_thk, half_wall_ht)
    bottom_wall = trimesh.creation.box(bottom_wall_dim, trimesh.transformations.translation_matrix(bottom_wall_pos))
    meshes_list.append(bottom_wall)

    left_wall_dim = (cfg.wall_thickness, cfg.wall_length, cfg.wall_height)
    left_wall_pos = (half_wall_thk, half_terr, half_wall_ht)
    left_wall = trimesh.creation.box(left_wall_dim, trimesh.transformations.translation_matrix(left_wall_pos))
    meshes_list.append(left_wall)

    right_wall_dim = (cfg.wall_thickness, cfg.wall_length, cfg.wall_height)
    right_wall_pos = (terrain_size - half_wall_thk, half_terr, half_wall_ht)
    right_wall = trimesh.creation.box(right_wall_dim, trimesh.transformations.translation_matrix(right_wall_pos))
    meshes_list.append(right_wall)

    origin = np.array([half_terr, half_terr, 0.0])

    return meshes_list, origin
