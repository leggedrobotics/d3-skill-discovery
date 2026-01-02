# Copyright (c) 2025, ETH Zurich, Rafael Cathomen
#
# SPDX-License-Identifier: BSD-3-Clause

#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Helper functions."""

from .mirroring import augment_anymal_action, augment_anymal_obs
from .timer import TIMER_CUMULATIVE
from .utils import (
    detach,
    extract_batch_shape,
    flatten_batch,
    is_valid,
    mean_gradient_norm,
    split_and_pad_trajectories,
    store_code_state,
    to_device,
    unflatten_batch,
    unpad_trajectories,
)
