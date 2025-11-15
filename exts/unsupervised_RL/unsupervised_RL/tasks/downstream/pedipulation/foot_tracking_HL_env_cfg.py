# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import unsupervised_RL.tasks.downstream.goal_tracking.mdp as mdp
import unsupervised_RL.tasks.unsupervised_skill_discovery.anymal_usd.mdp as usd_mdp

##
# Pre-defined configs
##
from unsupervised_RL.tasks.downstream.pedipulation.foot_tracking_base_env_cfg import FootTrackingEnvCfg


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    skill_action = mdp.SkillSelectionActionCfg(
        asset_name="robot",
        low_level_action=mdp.JointPositionActionCfg(  # copied from velocity_env & box_climb_env
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        ),
        # skill_policy_file_path="/home/rafael/Projects/MT/CRL/unsupervised_RL/logs/rsl_rl/metra_factors_ppo_anymal_test/metra_diayn_footpos2_mildest_export_2/exported/policy.pt",
        # skill_policy_file_path="/home/rafael/Projects/MT/CRL/unsupervised_RL/logs/rsl_rl/metra_factors_ppo_anymal_test/feet_metra_zero_pad_06_end/exported/policy.pt",
        skill_policy_file_path="/home/rafael/Projects/MT/CRL/unsupervised_RL/logs/rsl_rl/metra_factors_ppo_anymal_test/simba_sym_08/exported/policy.pt",
        # skill_policy_file_path="/home/rafael/Projects/MT/CRL/unsupervised_RL/logs/rsl_rl/metra_factors_ppo_anymal_test/metra_diayn_footpos2_export_6/exported/policy.pt",
        # skill_policy_file_path="/home/rafael/Projects/MT/CRL/unsupervised_RL/logs/rsl_rl/metra_factors_ppo_anymal_test/metra_diayn_footpos2_export_7/exported/policy.pt",
        skill_space={
            "position": ("metra_norm_matching", 2),
            "feet": ("metra_norm_matching_zero_pad", 12),
            "base_vel": ("dirichlet", 8),
            "skill_weights": ("unit_sphere_positive", 4),
        },
        observation_group="ll_skill_policy",
        skill_policy_freq=50.0,
        debug_vis=True,
    )


@configclass
class LowLevelSkillPolicyCfg(ObsGroup):
    """Observations for policy group."""

    # observation terms (order preserved)
    origin = ObsTerm(
        func=usd_mdp.origin_spawn,  # velocity_2d_b, rotation_velocity_2d_b
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    height_scan = ObsTerm(
        func=usd_mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        noise=Unoise(n_min=-0.1, n_max=0.1),
        clip=(-1.0, 1.0),
    )
    # time_left = ObsTerm(func=usd_mdp.time_left)

    # # proprioception
    base_lin_vel = ObsTerm(func=usd_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
    base_ang_vel = ObsTerm(func=usd_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(
        func=usd_mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    actions = ObsTerm(func=usd_mdp.last_low_level_action, params={"action_name": "skill_action"})
    joint_pos = ObsTerm(func=usd_mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=usd_mdp.joint_vel_rel)

    skill_command = ObsTerm(func=usd_mdp.action_command, params={"action_name": "skill_action"})

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = False


##
# Environment configuration
##


@configclass
class FootTrackingHLEnvCfg(FootTrackingEnvCfg):
    """Configuration for the downstream foot tracking environment."""

    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        """Post initialization."""
        # add low level observation group
        self.observations.ll_skill_policy = LowLevelSkillPolicyCfg()

        # general settings
        # -- frequency settings
        self.fz_planner = 10  # Hz
        self.sim.dt = 0.005  # 200 Hz
        self.decimation = int(1 / (self.sim.dt * self.fz_planner))

        self.episode_length_s = 20.0
        # simulation settings
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
