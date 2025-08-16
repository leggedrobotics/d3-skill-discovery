from __future__ import annotations

import os
import torch
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import math as math_utils
from isaaclab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from .skill_selection_action_cfg import SkillSelectionActionCfg

from ..nets import NET_PATH

TELEOP = False
if TELEOP:
    from isaaclab.devices import Se3Gamepad, Se3Keyboard, Se3SpaceMouse


class SkillSelectionAction(ActionTerm):

    cfg: SkillSelectionActionCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: SkillSelectionActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot_name = "robot"

        # -- load policy
        if cfg.skill_policy_file_name is not None:
            # load from the net path
            cfg.skill_policy_file_path = os.path.join(NET_PATH, cfg.skill_policy_file_name)

        if not check_file_path(cfg.skill_policy_file_path):
            raise FileNotFoundError(f"Policy file '{cfg.skill_policy_file_path}' does not exist.")
        # load policies
        file_bytes = read_file(self.cfg.skill_policy_file_path)
        self.skill_policy = torch.jit.load(file_bytes, map_location=self.device)
        self.skill_policy = torch.jit.freeze(self.skill_policy.eval())

        # calculate decimation
        self.low_level_policy_decimation = int(1 / (cfg.skill_policy_freq * env.physics_dt))

        # prepare joint position actions
        self.low_level_action_term: ActionTerm = self.cfg.low_level_action.class_type(cfg.low_level_action, env)

        self._action_dim = 0
        for name, (distr_type, dim) in cfg.skill_space.items():
            if "zero_pad" in distr_type:
                assert dim % 4 == 0, f"Zero padded skills must be divisible by 4, but got {dim}"
                self._action_dim += dim // 4 + 2  # 2 to select which symmetry of the 4 to use
            else:
                self._action_dim += dim

        self._processed_action_dim = sum([v[1] for v in cfg.skill_space.values()])

        # set up buffers
        self._init_buffers()

        # teleop:
        self.use_teleop = TELEOP
        if self.use_teleop:
            self.teleop_interface = Se3Keyboard(pos_sensitivity=1, rot_sensitivity=1)
            self.teleop_interface.add_callback("L", env.reset)
            print(self.teleop_interface)

        # joint reordering
        self.joint_mapping_gym_to_sim = (
            env.scene["robot"].find_joints(
                env.scene["robot"].joint_names, self.cfg.reorder_joint_list, preserve_order=True
            )[0]
            if self.cfg.reorder_joint_list
            else None
        )

        self.ll_action_range = (-10, 10)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_command_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """This returns the command for the low-level policies, predicted by the high-level policy."""
        return self._processed_command_actions

    @property
    def low_level_actions(self) -> torch.Tensor:
        return self._low_level_actions

    @property
    def prev_low_level_actions(self) -> torch.Tensor:
        return self._prev_low_level_actions

    """
    Operations.
    """

    def process_actions(self, actions):
        """Process high-level actions. This function is called with a frequency of 10Hz.

        Args:
            actions (torch.Tensor): The high level action command to condition the low-level policies.
        """

        self._raw_command_actions.copy_(actions)

        # depending ont the skill distribution, we need to transform the actions
        dim_idx = 0
        dim_prcd_idx = 0
        for skill_name, (distr_type, dim) in self.cfg.skill_space.items():
            next_dim_prcd_idx = dim_prcd_idx + dim
            if "zero_pad" in distr_type:
                # pad with zeros.
                self._processed_command_actions[:, dim_prcd_idx:next_dim_prcd_idx] = 0

                # select index to insert
                # we select the index based on the quadrant the action is in
                selection_action = actions[:, dim_idx : dim_idx + 2]
                sym_1 = (selection_action[:, 0] > 0) & (selection_action[:, 1] > 0)
                sym_2 = (selection_action[:, 0] > 0) & (selection_action[:, 1] < 0)
                sym_3 = (selection_action[:, 0] < 0) & (selection_action[:, 1] < 0)
                sym_4 = (selection_action[:, 0] < 0) & (selection_action[:, 1] > 0)

                # if invalid, set boolean values for symmetry
                invalid = (sym_1.int() + sym_2.int() + sym_3.int() + sym_4.int()) != 1
                sym_1[invalid] = True
                sym_2[invalid] = False
                sym_3[invalid] = False
                sym_4[invalid] = False

                sub_dim = dim // 4

                mask = torch.cat(
                    [
                        sym_1.repeat(sub_dim, 1).T,
                        sym_2.repeat(sub_dim, 1).T,
                        sym_3.repeat(sub_dim, 1).T,
                        sym_4.repeat(sub_dim, 1).T,
                    ],
                    dim=1,
                )

                dim_idx += 2
                next_dim_idx = dim_idx + dim // 4

                processed_action_slice = (slice(None), slice(dim_prcd_idx, next_dim_prcd_idx))

                if distr_type == "dirichlet":
                    # softmax the action
                    self._processed_command_actions[processed_action_slice][mask] = (
                        torch.softmax(actions[:, dim_idx:next_dim_idx], dim=1).float().flatten()
                    )
                elif "metra" in distr_type:
                    self._processed_command_actions[processed_action_slice][mask] = (
                        actions[:, dim_idx:next_dim_idx].float().flatten()
                    )

                else:
                    raise ValueError(f"Unknown distribution type {distr_type}")

            else:

                next_dim_idx = dim_idx + dim
                processed_action_slice = (slice(None), slice(dim_prcd_idx, next_dim_prcd_idx))

                if distr_type == "categorical":
                    # one-hot encode the action
                    self._processed_command_actions[processed_action_slice].copy_(
                        torch.nn.functional.one_hot(
                            torch.argmax(actions[:, dim_idx:next_dim_idx]).long(),
                            num_classes=dim,
                        ).float()
                    )
                elif distr_type == "dirichlet":
                    # softmax the action
                    self._processed_command_actions[processed_action_slice].copy_(
                        torch.softmax(actions[:, dim_idx:next_dim_idx], dim=1).float()
                    )
                elif "metra" in distr_type:
                    # TODO differencing between metra_align and metra_norm_matching
                    # copy the action
                    self._processed_command_actions[processed_action_slice].copy_(
                        actions[:, dim_idx:next_dim_idx].float()
                    )
                elif "unit_sphere_positive" in distr_type:

                    self._processed_command_actions[processed_action_slice].copy_(
                        torch.nn.functional.normalize(torch.nn.functional.softplus(actions[:, dim_idx:next_dim_idx]))
                    )
                else:
                    raise ValueError(f"Unknown distribution type {distr_type}")

            dim_idx = next_dim_idx
            dim_prcd_idx = next_dim_prcd_idx

        # transform the actions

        if self.use_teleop:
            delta_pose, gripper_command = self.teleop_interface.advance()

    def apply_actions(self):
        """Apply low-level actions for the simulator to the physics engine. This functions is called with the
        simulation frequency of 200Hz. Since low-level locomotion runs at 50Hz, we need to decimate the actions."""

        if self._counter % self.low_level_policy_decimation == 0:
            # update low-level action at 50Hz
            self._counter = 0
            # Get low level actions from low level policy

            self._low_level_actions.copy_(
                torch.clamp(
                    self.skill_policy(
                        self._flatten_dict_obs(
                            self._env.observation_manager.compute_group(group_name=self.cfg.observation_group)
                        )
                    ),
                    min=self.ll_action_range[0],
                    max=self.ll_action_range[1],
                )  # .chunk(2, dim=1)[0]
                # Chunking should happen in the exporter, ie the loaded policy should only return the means
            )
            self._prev_low_level_actions.copy_(self._low_level_actions.clone())

            if self.cfg.reorder_joint_list is not None:
                self._low_level_actions = self._low_level_actions[:, self.joint_mapping_gym_to_sim]

            # Process low level actions
            self.low_level_action_term.process_actions(
                self._low_level_actions
            )  # assuming all low level skills have the same action

        # Apply low level actions
        self.low_level_action_term.apply_actions()
        self._counter += 1

    """
    Helper functions
    """

    def _init_buffers(self):
        # Prepare buffers
        self._raw_command_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_command_actions = torch.zeros((self.num_envs, self._processed_action_dim), device=self.device)
        # self._skill_mask = torch.zeros(self.num_envs, self.num_skills, device=self.device).bool()
        self._low_level_actions = torch.zeros(self.num_envs, self.low_level_action_term.action_dim, device=self.device)
        self._prev_low_level_actions = torch.zeros_like(self._low_level_actions)
        self._counter = 0

    @staticmethod
    def _flatten_dict_obs(obs_dict: dict[torch.Tensor]) -> torch.Tensor:
        """Flatten the dictionary of tensors into a single tensor."""
        if isinstance(obs_dict, torch.Tensor):
            return obs_dict.flatten(1)
        return torch.cat([obs_dict[key].flatten(1) for key in sorted(obs_dict.keys())], dim=-1)

    """
    Debug visualization
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass
