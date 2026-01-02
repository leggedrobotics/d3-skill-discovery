# Copyright (c) 2025, ETH Zurich, Rafael Cathomen
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .skill_selection_action import SkillSelectionAction


@configclass
class SkillSelectionActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = SkillSelectionAction
    """ Class of the action term."""
    skill_space: dict[str : tuple[Literal["metra_align", "metra_norm_matching", "categorical", "dirichlet"], int]] = (
        MISSING
    )
    """Per factor, distribution type and dimension"""
    low_level_action: ActionTermCfg = MISSING
    """Configuration of the low level action term."""
    skill_policy_file_path: str = MISSING
    """Path to the low level skill policy file."""
    skill_policy_freq: float = MISSING
    """Frequency of the low level skill policy."""
    observation_group: str = MISSING
    """Observation group to use for the low level policy."""
