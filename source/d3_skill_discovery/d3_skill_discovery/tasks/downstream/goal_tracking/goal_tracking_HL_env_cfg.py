# Copyright (c) 2025, ETH Zurich, Rafael Cathomen
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import d3_skill_discovery.tasks.downstream.goal_tracking.mdp as mdp
import d3_skill_discovery.tasks.unsupervised_skill_discovery.anymal_usd.mdp as usd_mdp

##
# Pre-defined configs
##
from d3_skill_discovery.tasks.downstream.goal_tracking.goal_tracking_base_env_cfg import GoalReachingEnvCfg

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    skill_action = mdp.SkillSelectionActionCfg(
        asset_name="robot",
        low_level_action=mdp.JointPositionActionCfg(  # copied from velocity_env & box_climb_env
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        ),
        skill_policy_file_path="/home/rafael/Projects/MT/CRL/d3_skill_discovery/logs/d3_rsl_rl/metra_factors_ppo_anymal_test/policies_eval/oracle_velocity_tracking.pt",
        # skill_policy_file_path="/home/rafael/Projects/MT/CRL/d3_skill_discovery/logs/d3_rsl_rl/metra_factors_ppo_anymal_test/policies_eval/skill_policies/skill_policy_10_metra2d_diayn2d.pt",
        skill_space={
            # "position": ("metra_norm_matching", 2),
            # "heading": ("dirichlet", 2),
            # "weights": ("unit_sphere_positive", 3),
            "velocity_cmd": ("metra_norm_matching", 3),
        },
        observation_group="ll_oracle_policy",  # ll_skill_policy, ll_oracle_policy
        skill_policy_freq=50.0,
        debug_vis=True,
    )


@configclass
class LowLevelSkillPolicyCfg(ObsGroup):
    """Observations for policy group."""

    # observation terms (order preserved)
    origin = ObsTerm(
        func=usd_mdp.origin_spawn,  # velocity_2d_b, rotation_velocity_2d_b
        params={"asset_cfg": SceneEntityCfg("robot"), "reset_every_n_steps": 1},
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


@configclass
class OraclePolicyCfg(ObsGroup):
    """Observations for policy group."""

    # observation terms (order preserved)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    skill_command = ObsTerm(func=usd_mdp.action_command, params={"action_name": "skill_action"})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
    actions = ObsTerm(func=usd_mdp.last_low_level_action, params={"action_name": "skill_action"})
    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner_oracle")},
        noise=Unoise(n_min=-0.1, n_max=0.1),
        clip=(-1.0, 1.0),
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


##
# Environment configuration
##


@configclass
class GoalReachingHLEnvCfg(GoalReachingEnvCfg):
    """Configuration for the downstream goal-tracking env."""

    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        """Post initialization."""
        # add low level observation group
        self.observations.ll_skill_policy = LowLevelSkillPolicyCfg()
        self.observations.ll_oracle_policy = OraclePolicyCfg()

        # general settings
        # -- frequency settings
        self.fz_planner = 10  # Hz
        self.sim.dt = 0.005  # 200 Hz
        self.decimation = int(1 / (self.sim.dt * self.fz_planner))

        self.episode_length_s = 30.0
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

        self.observations.ll_skill_policy.origin.params["reset_every_n_steps"] = self.decimation

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
