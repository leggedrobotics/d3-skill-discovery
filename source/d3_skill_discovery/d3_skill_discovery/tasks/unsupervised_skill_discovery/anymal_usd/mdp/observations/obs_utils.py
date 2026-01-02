# Copyright (c) 2025, ETH Zurich, Rafael Cathomen
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from pxr import Gf, Usd, UsdGeom

import isaaclab.sim as sim_utils


def compute_asset_aabb(prim_path_expr: str, device: str) -> torch.Tensor:
    """Compute the axis-aligned bounding box (AABB) of the given prim paths.

    Args:
        prim_path_expr: Expression to find the prim paths.
        device: Device to compute the bounding box on.

    Returns:
        The bounding box dimensions of the asset. Shape is (num_prims, 3).
    """
    # resolve prim paths for spawning and cloning
    prims = sim_utils.find_matching_prims(prim_path_expr)

    # Initialize scale tensor
    scale = torch.zeros(len(prims), 3, device=device)

    # Create a bbox cache
    bbox_cache = UsdGeom.BBoxCache(
        time=Usd.TimeCode.Default(), useExtentsHint=False, includedPurposes=[UsdGeom.Tokens.default_]
    )

    # Compute bounding box for each prim path
    for i, prim in enumerate(prims):
        bbox_bounds: Gf.BBox3d = bbox_cache.ComputeWorldBound(prim)
        bbox_range = bbox_bounds.GetRange()
        bbox_range_min, bbox_range_max = bbox_range.GetMin(), bbox_range.GetMax()

        scale[i] = torch.tensor([bbox_range_max[j] - bbox_range_min[j] for j in range(3)], device=device)

    return scale
