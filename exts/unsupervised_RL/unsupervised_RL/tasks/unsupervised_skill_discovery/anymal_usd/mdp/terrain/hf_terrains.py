# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to generate height fields for different terrains."""

from __future__ import annotations

import numpy as np
import scipy.interpolate as interpolate
from typing import TYPE_CHECKING

from isaaclab.terrains.height_field.utils import height_field_to_mesh

if TYPE_CHECKING:
    from . import hf_terrains_cfg


@height_field_to_mesh
def cell_border(difficulty: float, cfg: hf_terrains_cfg.CellBorderCfg) -> np.ndarray:
    """Generate a cell border wall."""

    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    hf_raw = np.zeros((width_pixels, length_pixels)).astype(bool)

    height = cfg.height / cfg.vertical_scale

    B = 1
    # -- border 1 pixels
    hf_raw[:B, :] = True
    hf_raw[-B:, :] = True
    hf_raw[:, :B] = True
    hf_raw[:, -B:] = True

    # cut corners
    if cfg.corner_witdh > 0:
        B = int(cfg.corner_witdh / cfg.horizontal_scale)
        # Top-left corner
        hf_raw[-B:, :B] |= np.tri(B, B, 0, dtype=bool)
        # Bottom-left corner
        hf_raw[:B, :B] |= np.tri(B, B, 0, dtype=bool)[::-1, :]
        # Top-right corner
        hf_raw[-B:, -B:] |= np.tri(B, B, 0, dtype=bool)[:, ::-1]
        # Bottom-right corner
        hf_raw[:B, -B:] |= np.tri(B, B, 0, dtype=bool)[::-1, ::-1]

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16) * height


@height_field_to_mesh
def random_pyramid(difficulty: float, cfg: hf_terrains_cfg.RandomPyramid) -> np.ndarray:
    """Generate a cell border wall."""

    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    hf_raw = np.zeros((width_pixels, length_pixels)).astype(bool)

    height = cfg.wall_height / cfg.vertical_scale

    # - border 1 pixels
    B = 1
    # -- border 1 pixels
    hf_raw[:B, :] = True
    hf_raw[-B:, :] = True
    hf_raw[:, :B] = True
    hf_raw[:, -B:] = True

    hf_raw = hf_raw.astype(float) * height

    # - pyramid

    # we generate a random pyramid by creating square levels of increasing width
    # with random xy offsets
    N_levels = int(round(difficulty))
    avg_level_width = width_pixels / (2 * N_levels + 1)
    step_height = int(cfg.step_height / cfg.vertical_scale)

    # random but smaller in bigger levels
    min_step_width_pixels = int(cfg.min_width / cfg.horizontal_scale)
    big_start_x = big_start_y = 0
    big_width = width_pixels
    for level_i in reversed(range(N_levels)):
        level_width = int((level_i * 2 + 1) * avg_level_width)

        if cfg.force_to_corner:
            start_x = big_start_x + int(np.random.random() > 0.5) * (big_width - level_width)
            start_y = big_start_y + int(np.random.random() > 0.5) * (big_width - level_width)

        else:
            start_x = big_start_x + np.random.randint(
                min_step_width_pixels, big_width - level_width - min_step_width_pixels
            )
            start_y = big_start_y + np.random.randint(
                min_step_width_pixels, big_width - level_width - min_step_width_pixels
            )

        hf_raw[start_x : start_x + level_width, start_y : start_y + level_width] += step_height
        big_start_x = start_x
        big_start_y = start_y
        big_width = level_width

    # round off the heights to the nearest vertical step

    if False:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("TkAgg")
        plt.imshow(hf_raw)
        plt.show()

    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def random_uniform_terrain_difficulty(
    difficulty: float, cfg: hf_terrains_cfg.HfRandomUniformTerrainDifficultyCfg
) -> np.ndarray:
    """Generate a terrain with height sampled uniformly from a specified range.

    .. image:: ../../_static/terrains/height_field/random_uniform_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.

    Raises:
        ValueError: When the downsampled scale is smaller than the horizontal scale.
    """
    # check parameters
    # -- horizontal scale
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = cfg.horizontal_scale
    elif cfg.downsampled_scale < cfg.horizontal_scale:
        raise ValueError(
            "Downsampled scale must be larger than or equal to the horizontal scale:"
            f" {cfg.downsampled_scale} < {cfg.horizontal_scale}."
        )

    # switch parameters to discrete units
    # -- horizontal scale
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- downsampled scale
    width_downsampled = int(cfg.size[0] / cfg.downsampled_scale)
    length_downsampled = int(cfg.size[1] / cfg.downsampled_scale)
    # -- height
    height_min = int(cfg.noise_range[0] * difficulty / cfg.vertical_scale)
    height_max = int(cfg.noise_range[1] * difficulty / cfg.vertical_scale)
    height_step = int(cfg.noise_step / cfg.vertical_scale)

    # create range of heights possible
    height_range = np.arange(height_min, height_max + height_step, height_step)
    # sample heights randomly from the range along a grid
    height_field_downsampled = np.random.choice(height_range, size=(width_downsampled, length_downsampled))
    # create interpolation function for the sampled heights
    x = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_downsampled)
    y = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_downsampled)
    func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)

    # interpolate the sampled heights to obtain the height field
    x_upsampled = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
    y_upsampled = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)
    z_upsampled = func(x_upsampled, y_upsampled)
    # round off the interpolated heights to the nearest vertical step
    return np.rint(z_upsampled).astype(np.int16)


@height_field_to_mesh
def random_uniform_boxes_terrain(difficulty: float, cfg: hf_terrains_cfg.HfRandomBoxesTerrainCfg) -> np.ndarray:
    """Generate a terrain with height sampled uniformly from a specified range.

    .. image:: ../../_static/terrains/height_field/random_uniform_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.

    Raises:
        ValueError: When the downsampled scale is smaller than the horizontal scale.
    """
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = cfg.horizontal_scale
    elif cfg.downsampled_scale < cfg.horizontal_scale:
        raise ValueError(
            "Downsampled scale must be >= horizontal_scale. " f"Got: {cfg.downsampled_scale} < {cfg.horizontal_scale}."
        )

    effective_size_x = cfg.size[0]
    effective_size_y = cfg.size[1]
    if effective_size_x < 0 or effective_size_y < 0:
        raise ValueError("Border width is too large compared to terrain size.")

    width_pixels = int(effective_size_x / cfg.horizontal_scale)
    length_pixels = int(effective_size_y / cfg.horizontal_scale)

    width_down = int(effective_size_x / cfg.downsampled_scale)
    length_down = int(effective_size_y / cfg.downsampled_scale)

    height_min = int(cfg.noise_range[0] / cfg.vertical_scale)
    height_max = int(cfg.noise_range[1] / cfg.vertical_scale)
    height_step = int(cfg.noise_step / cfg.vertical_scale)
    if height_step <= 0:
        raise ValueError("noise_step is too small relative to vertical_scale (must be >= one vertical step).")

    height_values = np.arange(height_min, height_max + height_step, height_step)
    height_field_down = np.random.choice(height_values, size=(width_down, length_down))

    x_down = np.linspace(0, effective_size_x, width_down)
    y_down = np.linspace(0, effective_size_y, length_down)
    func = interpolate.RectBivariateSpline(x_down, y_down, height_field_down)

    x_up = np.linspace(0, effective_size_x, width_pixels)
    y_up = np.linspace(0, effective_size_y, length_pixels)
    z_up = func(x_up, y_up)
    z_up = np.rint(z_up).astype(np.int16)

    height_field = z_up.astype(np.float32)

    nb_min, nb_max = cfg.num_boxes_range
    num_boxes = int(nb_min + difficulty * (nb_max - nb_min))

    wmin_m, wmax_m = cfg.box_width_range
    dmin_m, dmax_m = cfg.box_depth_range
    hmin_m, hmax_m = cfg.box_height_range

    wmin_px = int(np.floor(wmin_m / cfg.horizontal_scale))
    wmax_px = int(np.ceil(wmax_m / cfg.horizontal_scale))
    dmin_px = int(np.floor(dmin_m / cfg.horizontal_scale))
    dmax_px = int(np.ceil(dmax_m / cfg.horizontal_scale))

    def meters_to_hf_units(h_m: float) -> float:
        return h_m / cfg.vertical_scale

    half_plat_m = 0.5 * cfg.platform_width

    plat_center_x = 0.5 * effective_size_x / cfg.horizontal_scale
    plat_center_y = 0.5 * effective_size_y / cfg.horizontal_scale
    half_plat_px = half_plat_m / cfg.horizontal_scale

    plat_min_x = int(np.floor(plat_center_x - half_plat_px))
    plat_max_x = int(np.ceil(plat_center_x + half_plat_px))
    plat_min_y = int(np.floor(plat_center_y - half_plat_px))
    plat_max_y = int(np.ceil(plat_center_y + half_plat_px))

    def boxes_overlap_2d(ax, ay, aw, ad, bx, by, bw, bd):
        if (ax + aw <= bx) or (bx + bw <= ax):
            return False
        if (ay + ad <= by) or (by + bd <= ay):
            return False
        return True

    placed_rects = []
    rng = np.random.default_rng()

    max_attempts = 2000

    for _ in range(num_boxes):
        attempt = 0
        placed = False
        while attempt < max_attempts and not placed:
            attempt += 1

            w_px = rng.integers(wmin_px, max(wmin_px + 1, wmax_px + 1))  # avoid zero
            d_px = rng.integers(dmin_px, max(dmin_px + 1, dmax_px + 1))
            if w_px <= 0 or d_px <= 0:
                continue

            box_extra_h_m = rng.uniform(hmin_m, hmax_m)
            box_extra_h_units = meters_to_hf_units(box_extra_h_m)

            x_px = rng.integers(0, max(1, width_pixels - w_px))
            y_px = rng.integers(0, max(1, length_pixels - d_px))

            overlap_platform = not (
                (x_px + w_px <= plat_min_x)
                or (x_px >= plat_max_x)
                or (y_px + d_px <= plat_min_y)
                or (y_px >= plat_max_y)
            )
            if overlap_platform:
                continue

            # Check overlapwith previously placed boxes
            no_collision = True
            for ox, oy, ow, od in placed_rects:
                if boxes_overlap_2d(x_px, y_px, w_px, d_px, ox, oy, ow, od):
                    no_collision = False
                    break

            if no_collision:
                placed_rects.append((x_px, y_px, w_px, d_px))
                placed = True

                local_region = height_field[x_px : x_px + w_px, y_px : y_px + d_px]
                current_max = local_region.max()
                new_val = current_max + box_extra_h_units
                height_field[x_px : x_px + w_px, y_px : y_px + d_px] = new_val
    final_height_field = np.rint(height_field).astype(np.int16)
    return final_height_field


@height_field_to_mesh
def random_uniform_walls_terrain(
    difficulty: float,
    cfg: hf_terrains_cfg.HfRandomUniformWallsTerrainCfg,
) -> np.ndarray:
    """
    Generate a square random-uniform terrain with ONLY a north wall (top edge) and
    an east wall (right edge). Each wall's length is randomly sampled and its
    position is shifted randomly along its length axis and thickness axis, but
    constrained so it never goes out of the terrain bounds.

    Args:
        difficulty: A float in [0, 1]. If desired, you can further scale the random
                    parameters by difficulty (not shown here).
        cfg: HfRandomUniformWallsTerrainCfg with all relevant parameters.

    Returns:
        A 2D height field (shape [width_pixels, length_pixels]) that includes a
        random-uniform base plus two walls. This will be converted to a mesh
        downstream via the `height_field_to_mesh` decorator.
    """

    size_x, size_y = cfg.size
    if not np.isclose(size_x, size_y, atol=1e-6):
        raise ValueError(f"This terrain must be square, but got size={cfg.size}.")

    side_m = size_x
    if side_m <= 0:
        raise ValueError("Terrain side must be positive.")

    width_pixels = int(np.floor(side_m / cfg.horizontal_scale))
    length_pixels = width_pixels

    ds_scale = cfg.downsampled_scale
    if ds_scale is None:
        ds_scale = cfg.horizontal_scale
    elif ds_scale < cfg.horizontal_scale:
        raise ValueError(f"downsampled_scale ({ds_scale}) must be >= horizontal_scale ({cfg.horizontal_scale}).")
    width_down = int(np.floor(side_m / ds_scale))
    length_down = width_down

    h_min = int(cfg.noise_range[0] / cfg.vertical_scale)
    h_max = int(cfg.noise_range[1] / cfg.vertical_scale)
    h_step = int(cfg.noise_step / cfg.vertical_scale)
    if h_step <= 0:
        raise ValueError("noise_step is too small relative to vertical_scale.")

    possible_heights = np.arange(h_min, h_max + h_step, h_step)
    rng = np.random.default_rng()
    downsampled_field = rng.choice(possible_heights, size=(width_down, length_down))

    x_down = np.linspace(0, side_m, width_down)
    y_down = np.linspace(0, side_m, length_down)
    func = interpolate.RectBivariateSpline(x_down, y_down, downsampled_field)

    x_up = np.linspace(0, side_m, width_pixels)
    y_up = np.linspace(0, side_m, length_pixels)
    z_up = func(x_up, y_up)

    height_field = np.rint(z_up).astype(np.float32)

    h_wall_units = cfg.wall_height / cfg.vertical_scale
    wall_width_px = int(np.ceil(cfg.wall_width / cfg.horizontal_scale))

    def raise_region(x0, x1, y0, y1):
        """
        Raise [x0:x1, y0:y1] by adding h_wall_units above the local max.
        We clamp to array bounds, so if x1<x0 or y1<y0 after clamp => no effect.
        """
        if x1 <= x0 or y1 <= y0:
            return
        # clamp to array range
        x0c, x1c = max(0, x0), min(width_pixels, x1)
        y0c, y1c = max(0, y0), min(length_pixels, y1)
        if x1c <= x0c or y1c <= y0c:
            return
        region = height_field[x0c:x1c, y0c:y1c]
        region[...] = h_wall_units

    north_wall_length_m = rng.uniform(*cfg.wall_length_range)
    north_wall_length_m = min(north_wall_length_m, side_m)

    north_wall_len_px = int(np.floor(north_wall_length_m / cfg.horizontal_scale))
    north_long_shift_m = rng.uniform(*cfg.wall_long_shift_range)
    max_north_shift_m = max(0.0, side_m - north_wall_length_m)
    if north_long_shift_m > max_north_shift_m:
        north_long_shift_m = max_north_shift_m
    north_long_shift_px = int(np.floor(north_long_shift_m / cfg.horizontal_scale))

    north_thick_shift_m = rng.uniform(*cfg.wall_thickness_shift_range)
    max_north_thick_m = side_m - cfg.wall_width  # we want at least 'wall_width' near top
    if north_thick_shift_m > max_north_thick_m:
        north_thick_shift_m = max_north_thick_m
    north_thick_shift_px = int(np.floor(north_thick_shift_m / cfg.horizontal_scale))

    x0_n = north_long_shift_px
    x1_n = x0_n + north_wall_len_px
    y1_n = length_pixels - north_thick_shift_px
    y0_n = y1_n - wall_width_px

    raise_region(x0_n, x1_n, y0_n, y1_n)

    east_wall_length_m = rng.uniform(*cfg.wall_length_range)
    east_wall_length_m = min(east_wall_length_m, side_m)
    east_wall_len_px = int(np.floor(east_wall_length_m / cfg.horizontal_scale))

    east_long_shift_m = rng.uniform(*cfg.wall_long_shift_range)
    max_east_shift_m = max(0.0, side_m - east_wall_length_m)
    if east_long_shift_m > max_east_shift_m:
        east_long_shift_m = max_east_shift_m
    east_long_shift_px = int(np.floor(east_long_shift_m / cfg.horizontal_scale))

    east_thick_shift_m = rng.uniform(*cfg.wall_thickness_shift_range)
    max_east_thick_m = side_m - cfg.wall_width
    if east_thick_shift_m > max_east_thick_m:
        east_thick_shift_m = max_east_thick_m
    east_thick_shift_px = int(np.floor(east_thick_shift_m / cfg.horizontal_scale))

    x1_e = width_pixels - east_thick_shift_px
    x0_e = x1_e - wall_width_px
    y0_e = east_long_shift_px
    y1_e = y0_e + east_wall_len_px

    raise_region(x0_e, x1_e, y0_e, y1_e)

    final_height_field = np.rint(height_field).astype(np.int16)
    return final_height_field
