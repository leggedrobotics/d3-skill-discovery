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
    skill_space: dict[
        str : tuple[
            Literal[
                "metra_align",
                "metra_norm_matching",
                "metra_norm_matching_zero_pad",
                "categorical",
                "dirichlet",
                "unit_sphere_positive",
            ],
            int,
        ]
    ] = MISSING
    """Per factor, distribution type and dimension"""
    low_level_action: ActionTermCfg = MISSING
    """Configuration of the low level action term."""
    skill_policy_file_path: str | None = None
    """Absolute path to the low level skill policy file."""
    skill_policy_file_name: str | None = None
    """Name of the low level skill policy file.
    Either this or `skill_policy_file_path` must be set. 
    If skill_policy_file_name is set, the policy should be in ../nets/"""
    skill_policy_freq: float = MISSING
    """Frequency of the low level skill policy."""
    observation_group: str = MISSING
    """Observation group to use for the low level policy."""
    reorder_joint_list: list[str] = None
    """List of joints to reorder the action vector."""
