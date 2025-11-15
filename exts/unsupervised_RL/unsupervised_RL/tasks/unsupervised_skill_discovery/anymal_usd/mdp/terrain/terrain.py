# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainGenerator, TerrainGeneratorCfg

from .hf_terrains_cfg import (
    CellBorderCfg,
    HfRandomBoxesTerrainCfg,
    HfRandomUniformTerrainDifficultyCfg,
    HfRandomUniformWallsTerrainCfg,
    RandomPyramid,
)
from .mesh_terrains_cfg import MeshPyramidTerrainCfg

GAME_ARENA_BASE_CFG = TerrainGeneratorCfg(
    size=(16.0, 16.0),
    border_width=0.0,
    border_height=0.0,
    curriculum=True,
    num_rows=16,
    num_cols=16,
    horizontal_scale=0.1,
    vertical_scale=0.1,
    slope_threshold=None,
    use_cache=False,
    sub_terrains={
        # "rails": terrain_gen.MeshRailsTerrainCfg(
        #     proportion=0.2,
        #     rail_thickness_range=(0.05, 0.2),
        #     rail_height_range=(0.05, 0.2),
        # ),
        "wall": CellBorderCfg(
            border_width=0,
            height=2.0,
            flat_patch_sampling={
                "init_pos": FlatPatchSamplingCfg(num_patches=25, patch_radius=0.4, max_height_diff=5.0),
                "lowest_pos": FlatPatchSamplingCfg(
                    num_patches=25, patch_radius=0.4, max_height_diff=0.5, z_range=(0.0, 0.5)
                ),
            },
        ),
    },
)
GAME_ARENA_RANDOM_FLOORS_CFG = TerrainGeneratorCfg(
    size=(16.0, 16.0),
    border_width=20.0,
    border_height=0.0,
    curriculum=False,
    num_rows=16,  # difficulty levels
    num_cols=16,  # number of terrains per difficulty level
    horizontal_scale=0.25,
    vertical_scale=0.1,
    slope_threshold=0.75,
    use_cache=False,
    difficulty_range=(1, 1),  # number of steps
    sub_terrains={
        "wall": RandomPyramid(
            border_width=0,
            wall_height=2.0,
            step_height=1,
            min_width=0.5,
            flat_patch_sampling={
                "init_pos": FlatPatchSamplingCfg(num_patches=128, patch_radius=0.4, max_height_diff=5.0),
                "lowest_pos": FlatPatchSamplingCfg(
                    num_patches=128, patch_radius=0.4, max_height_diff=0.5, z_range=(0.0, 0.5)
                ),
                "not_lowest_pos": FlatPatchSamplingCfg(
                    num_patches=128, patch_radius=0.4, max_height_diff=5.0, z_range=(0.75, 5.0)
                ),
            },
        ),
    },
)


MESH_PYRAMID_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(20.0, 20.0),
    border_width=1.0,
    border_height=10.0,
    num_rows=16,  # difficulty levels
    num_cols=16,  # number of terrains per difficulty level
    horizontal_scale=0.5,
    vertical_scale=0.05,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    difficulty_range=(0, 0.75),
    sub_terrains={
        "pyramid_stairs": MeshPyramidTerrainCfg(
            proportion=1.0,
            step_height=1.0,
            step_width=(5.0, 1.5),
            platform_width=3.0,
            border_width=0.0,
            holes=False,
            walls=True,
            flat_patch_sampling={
                "init_pos": FlatPatchSamplingCfg(
                    num_patches=128,
                    patch_radius=1.5,
                    max_height_diff=5.0,
                ),
                "lowest_pos": FlatPatchSamplingCfg(
                    num_patches=128, patch_radius=1.0, max_height_diff=0.5, z_range=(0.0, 0.25)
                ),
                "not_lowest_pos": FlatPatchSamplingCfg(
                    num_patches=128,
                    patch_radius=0.3,
                    max_height_diff=50.0,
                    z_range=(0.75, 50.0),
                ),
            },
        )
    },
)


MESH_STEPPABLE_PYRAMID_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(25.0, 25.0),
    border_width=10.0,
    border_height=7.5,
    num_rows=32,  # difficulty levels
    num_cols=16,  # number of terrains per difficulty level
    horizontal_scale=0.5,
    vertical_scale=0.05,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    difficulty_range=(1.0, 1.0),
    sub_terrains={
        "pyramid_stairs": MeshPyramidTerrainCfg(
            proportion=1.0,
            step_height=0.5,
            step_width=(3.0, 1.75),
            platform_width=3.0,
            border_width=0.0,
            holes=False,
            walls=True,
            type="pyramid",
            wall_height=7.5,
            wall_thickness=1.25,
            flat_patch_sampling={
                "init_pos": FlatPatchSamplingCfg(
                    num_patches=8,
                    patch_radius=1.5,
                    max_height_diff=5.0,
                ),
                "lowest_pos": FlatPatchSamplingCfg(
                    num_patches=64, patch_radius=1.0, max_height_diff=0.25, z_range=(0.0, 0.25)
                ),
                "not_lowest_pos": FlatPatchSamplingCfg(
                    num_patches=256,
                    patch_radius=0.3,
                    max_height_diff=50.0,
                    z_range=(0.25, 6.0),
                ),
            },
        )
    },
)


MESH_FLAT_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    border_height=0.0,
    num_rows=8,
    num_cols=8,
    horizontal_scale=1.0,
    vertical_scale=1.0,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "pyramid_stairs": MeshPyramidTerrainCfg(
            proportion=1.0,
            step_height=0.0,
            step_width=(100.0, 100.5),
            platform_width=3.0,
            border_width=0.0,
            holes=False,
            flat_patch_sampling={
                "init_pos": FlatPatchSamplingCfg(
                    num_patches=128,
                    patch_radius=1.5,
                    max_height_diff=5.0,
                ),
                "lowest_pos": FlatPatchSamplingCfg(
                    num_patches=128,
                    patch_radius=0.5,
                    max_height_diff=0.5,
                ),
                "not_lowest_pos": FlatPatchSamplingCfg(
                    num_patches=128,
                    patch_radius=0.3,
                    max_height_diff=50.0,
                ),
            },
        )
    },
)


PYRAMID_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(16.0, 16.0),
    border_width=50.0,
    border_height=0.0,
    num_rows=10,  # difficulty levels
    num_cols=18,  # number of terrains per difficulty level
    horizontal_scale=0.2,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.3),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.00, 0.3),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pit": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.00, 0.6),
            step_width=4.0,
            platform_width=1.5,
            border_width=1.0,
            holes=False,
        ),
        # "random_rough2": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.3, noise_range=(0.005, 0.0875), noise_step=0.005, border_width=0.1
        # ),
        "random_rough3": HfRandomUniformTerrainDifficultyCfg(
            proportion=0.3, noise_range=(0.005, 0.3), noise_step=0.005, border_width=0.2
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.8), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.6), platform_width=2.0, border_width=0.25
        ),
        "boxes_small": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.3, grid_width=0.45, grid_height_range=(0.05, 0.35), platform_width=2.0
        ),
        "boxes_big": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.4, grid_width=2.5, grid_height_range=(0.0, 0.75), platform_width=2.5
        ),
    },
)
"""Rough terrains configuration."""

SLIGHTLY_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=50.0,
    border_height=0.0,
    num_rows=10,  # difficulty levels
    num_cols=10,  # number of terrains per difficulty level
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=90.2, noise_range=(0.02, 0.05), noise_step=0.025, border_width=0.25
        ),
    },
)

ONE_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=50.0,
    border_height=0.0,
    num_rows=1,  # difficulty levels
    num_cols=1,  # number of terrains per difficulty level
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=90.2, noise_range=(0.02, 0.05), noise_step=0.025, border_width=0.25
        ),
    },
)

"""Rough terrains configuration."""
LESS_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=50.0,
    border_height=0.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.02, 0.15),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.04, 0.20), platform_width=2.0
        ),
        "random_rough1": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "random_rough3": HfRandomUniformTerrainDifficultyCfg(
            proportion=0.2, noise_range=(0.005, 0.25), noise_step=0.005, border_width=0.2, horizontal_scale=0.2
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
    },
)
"""Rough terrains configuration."""

ONLY_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(32.0, 32.0),
    border_width=50.0,
    border_height=0.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough1": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
    },
)


TEST_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=50.0,
    border_height=0.0,
    num_rows=8,  # difficulty levels
    num_cols=8,  # number of terrains per difficulty level
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "boxes_big": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=2.5, grid_height_range=(0.3, 0.4), platform_width=2.5
        ),
    },
)


OBSTACLE_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(24.0, 24.0),
    border_width=50.0,
    border_height=0.0,
    num_rows=10,  # difficulty levels
    num_cols=10,  # number of terrains per difficulty level
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=True,
    use_cache=False,
    sub_terrains={
        "obstacles": HfRandomBoxesTerrainCfg(
            noise_range=(0.02, 0.05),
            noise_step=0.025,
            proportion=0.3,
            border_width=0.25,
            num_boxes_range=(8, 8 * 7),
            box_height_range=(0.1, 1.0),
            box_width_range=(0.25, 1.5),
            box_depth_range=(0.25, 1.5),
            platform_width=1.5,
        ),
        "obstacles_taller": HfRandomBoxesTerrainCfg(
            noise_range=(0.02, 0.05),
            noise_step=0.025,
            proportion=0.7,
            border_width=0.25,
            num_boxes_range=(4, 40),
            box_height_range=(1.0, 2.0),
            box_width_range=(0.5, 2.0),
            box_depth_range=(0.5, 2.0),
            platform_width=2.5,
        ),
    },
)


WALL_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(7.0, 7.0),
    border_width=50.0,
    border_height=0.0,
    num_rows=16,  # difficulty levels
    num_cols=16,  # number of terrains per difficulty level
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "wall_rough": HfRandomUniformWallsTerrainCfg(
            noise_range=(0.02, 0.05),
            noise_step=0.025,
            wall_height=2.0,
            wall_width=0.5,
            wall_length_range=(1.5, 4.0),
            wall_long_shift_range=(0.0, 5.5),
            wall_thickness_shift_range=(0.0, 1.0),
            proportion=0.4,
            border_width=0.1,
        ),
    },
)
