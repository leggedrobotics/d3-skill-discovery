from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import unsupervised_RL.tasks.downstream.goal_tracking.mdp as mdp
import unsupervised_RL.tasks.unsupervised_skill_discovery.anymal_usd.mdp as usd_mdp

##
# Pre-defined configs
##
from unsupervised_RL.tasks.downstream.pedipulation.foot_and_base_tracking_base_env_cfg import (
    BaseSceneCfg,
    FootBaseTrackingEnvCfg,
)

ISAAC_GYM_JOINT_NAMES = [
    "LF_HAA",
    "LF_HFE",
    "LF_KFE",
    "LH_HAA",
    "LH_HFE",
    "LH_KFE",
    "RF_HAA",
    "RF_HFE",
    "RF_KFE",
    "RH_HAA",
    "RH_HFE",
    "RH_KFE",
]


# only required for Nikitas low-level policy
@configclass
class HLSceneCfg(BaseSceneCfg):

    height_scan_low_level = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[2.0, 1.0], ordering="yx"),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        # track_mesh_transforms=True,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    skill_action = mdp.SkillSelectionActionCfg(
        asset_name="robot",
        low_level_action=mdp.JointPositionActionCfg(  # copied from velocity_env & box_climb_env
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        ),
        # skill_policy_file_path="/home/rafael/Projects/MT/CRL/unsupervised_RL/logs/rsl_rl/metra_factors_ppo_anymal_test/metra_diayn_footpos2_mildest_export_2/exported/policy.pt",
        # skill_policy_file_path="/home/rafael/Projects/MT/CRL/unsupervised_RL/logs/rsl_rl/metra_factors_ppo_anymal_test/metra_diayn_footpos2_mild_export_2/exported/policy.pt",
        # skill_policy_file_path="/home/rafael/Projects/MT/CRL/unsupervised_RL/logs/rsl_rl/metra_factors_ppo_anymal_test/metra_diayn_footpos2_export_6/exported/policy.pt",
        # skill_policy_file_path="/home/rafael/Projects/MT/CRL/unsupervised_RL/logs/rsl_rl/metra_factors_ppo_anymal_test/metra_diayn_footpos2_export_7/exported/policy.pt",
        # skill_policy_file_path="/home/rafael/Projects/MT/CRL/unsupervised_RL/logs/rsl_rl/metra_factors_ppo_anymal_test/feet_metra_zero_pad_06/exported/policy.pt",
        skill_policy_file_path="/home/rafael/Projects/MT/CRL/unsupervised_RL/logs/rsl_rl/metra_factors_ppo_anymal_test/simba_sym_07/exported/policy.pt",
        # skill_policy_file_path="/home/rafael/Projects/MT/unsupervised_RL/exts/unsupervised_RL/unsupervised_RL/tasks/unsupervised_skill_discovery/jump_boxes/mdp/nets/policy_walk_0310.jit",
        skill_space={
            # "velocity_command": ("metra_norm_matching", 5),
            "position": ("metra_norm_matching", 2),
            "feet": ("metra_norm_matching", 12),
            "base_vel": ("dirichlet", 8),
            "skill_weights": ("unit_sphere_positive", 4),
        },
        observation_group="ll_skill_policy",
        skill_policy_freq=50.0,
        debug_vis=True,
        # reorder_joint_list=ISAAC_GYM_JOINT_NAMES,
    )


@configclass
class LowLevelSkillPolicyCfg(ObsGroup):
    """Observations for low-level skill conditioned policy."""

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


@configclass
class LowLevelPolicyCfg(ObsGroup):
    """Observations for Nikitas low-level policy."""

    # observation terms (order preserved)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    # velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    # command, 2d pos, sin cos heading, time left [0,1]
    pos_head_time_command = ObsTerm(func=usd_mdp.action_command, params={"action_name": "skill_action"})

    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=ISAAC_GYM_JOINT_NAMES, preserve_order=True)},
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=ISAAC_GYM_JOINT_NAMES, preserve_order=True)},
        noise=Unoise(n_min=-1.5, n_max=1.5),
    )
    actions = ObsTerm(func=usd_mdp.last_low_level_action, params={"action_name": "skill_action"})
    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scan_low_level")},
        noise=Unoise(n_min=-0.1, n_max=0.1),
        clip=(-1.0, 1.0),
    )


##
# Environment configuration
##


@configclass
class FootBaseTrackingHLEnvCfg(FootBaseTrackingEnvCfg):
    """Configuration for the downstream foot and base tracking environment."""

    scene: HLSceneCfg = HLSceneCfg(num_envs=4096, env_spacing=7.5)  # only required for Nikitas low-level policy

    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        """Post initialization."""
        # add low level observation group
        self.observations.ll_skill_policy = LowLevelSkillPolicyCfg()  # LowLevelSkillPolicyCfg LowLevelPolicyCfg

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
