# Copyright (c) 2025, ETH Zurich, Rafael Cathomen
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for various robotic environments."""

from isaaclab_tasks.utils import import_packages

##
# Register Gym environments.
##


# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = []
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
