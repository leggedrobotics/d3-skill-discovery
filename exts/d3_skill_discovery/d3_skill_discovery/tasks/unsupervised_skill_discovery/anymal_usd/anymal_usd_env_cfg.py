# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import Literal

import d3_skill_discovery.tasks.unsupervised_skill_discovery.anymal_usd.mdp as mdp

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # isort: skip


##
# Scene definition
##
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # - ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=mdp.terrain.LESS_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=8,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 2.0)),
        # attach_yaw_only=True,
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.0, 0.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """No commands for USD."""


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        origin = ObsTerm(func=mdp.origin_spawn, params={"asset_cfg": SceneEntityCfg("robot")})  # type: ignore

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class UsdStateCfg(ObsGroup):
        """Full observations for unsupervised skill discovery.
        Factorization is done by grouping the individual observations."""

        origin = ObsTerm(
            func=mdp.origin_spawn,  # type: ignore
            params={"asset_cfg": SceneEntityCfg("robot"), "noise_scale": 0.0, "pos_2d_only": True},
        )

        heading_rate = ObsTerm(func=mdp.heading_rate)

        base_height = ObsTerm(func=mdp.base_height, noise=Unoise(n_min=-0.15, n_max=0.15), clip=(-1.0, 2.0))

        roll_pitch = ObsTerm(func=mdp.roll_pitch, noise=Unoise(n_min=-0.15, n_max=0.15))

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.15, n_max=0.15))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class MetricCfg(ObsGroup):
        """Extrinsic metric for evaluating the extrinsic performance."""

        metric = ObsTerm(
            func=mdp.base_height_flatness_metric,
            params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.55},
        )

    @configclass
    class RegularizationRewards(ObsGroup):
        """Observations for the regularization rewards."""

        metric = ObsTerm(
            func=mdp.regularization_reward_obs,  # type: ignore
            params={
                "terms": {
                    "dof_torques_l2": RewTerm(func=mdp.joint_torques_l2, weight=-0.001),
                    "dof_acc_l2": RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7),
                    "action_rate_l2": RewTerm(func=mdp.action_rate_l2, weight=-0.05),
                    "torque_limits": RewTerm(func=mdp.applied_torque_limits, weight=-15.0),
                    "torque_limits_ratio": RewTerm(
                        func=mdp.applied_torque_limits_fraction, weight=-15.0, params={"fraction": 0.75}
                    ),
                    "joint_torques": RewTerm(func=mdp.joint_torques_l2, weight=-0.01),
                    "joint_vel_limits": RewTerm(func=mdp.joint_vel_limits, weight=-10.0, params={"soft_ratio": 1.0}),
                    "joint_vel": RewTerm(func=mdp.joint_vel_l2, weight=-0.01),
                    "joint_pos_limits": RewTerm(func=mdp.joint_pos_limits, weight=-10.0),
                }
            },
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    usd: UsdStateCfg = UsdStateCfg()
    metric: MetricCfg = MetricCfg()
    regularization_reward: RegularizationRewards = RegularizationRewards()


"""The factors dictionary defines the observations and algorithms used for each factor."""
factors: dict[str, tuple[list[str], Literal["metra", "diayn"]]] = {
    "position": (
        [
            "origin",
        ],
        "metra",
    ),
    "heading": (
        [
            "heading_rate",
        ],
        "diayn",
    ),
    "base_height": (
        [
            "base_height",
        ],
        "diayn",
    ),
    "roll_pitch": (
        [
            "roll_pitch",
        ],
        "diayn",
    ),
    "base_vel": (
        [
            "base_lin_vel",
        ],
        "diayn",
    ),
}

""" The skill_dims dictionary defines the dimensions of each skill."""
skill_dims: dict[str, int] = {
    "position": 2,
    "heading": 2,
    "base_height": 2,
    "roll_pitch": 4,
    "base_vel": 4,
}

""" The resampling_intervals dictionary defines the resampling intervals for each skill."""
resampling_intervals: dict[str, int] = {
    "position": 375,  # 7.5 seconds
    "heading": 375,  # 7.5 seconds
    "base_height": 375,  # 7.5 seconds
    "roll_pitch": 375,  # 7.5 seconds
    "base_vel": 375,  # 7.5 seconds
}

""" The usd_alg_extra_cfg dictionary can be used to define additional configuration
parameters for the USD algorithm, or to override the default parameters."""
usd_alg_extra_cfg = {
    "heading": {
        "max_dirichlet_param": 1.0,
    },
    "base_height": {
        "max_dirichlet_param": 1.0,
        "not_symmetric": True,
    },
    "roll_pitch": {
        "max_dirichlet_param": 1.0,
    },
    "base_vel": {
        "max_dirichlet_param": 1.0,
    },
}


reset_value = 0.1 * 1.0
reset_value_pos = 0.5 * 1.0


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 0.9),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_save_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-reset_value_pos, reset_value_pos),
                "y": (-reset_value_pos, reset_value_pos),
                "z": (0.01, 0.035),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-reset_value, reset_value),
                "y": (-reset_value, reset_value),
                "z": (-reset_value, reset_value),
                "roll": (-reset_value, reset_value),
                "pitch": (-reset_value, reset_value),
                "yaw": (-reset_value, reset_value),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.75, 1.25),
            "velocity_range": (0.0, 0.0),
        },
    )

    #

    # - interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 10.0),
        params={
            "velocity_range": {
                "x": (-0.75, 0.75),
                "y": (-0.75, 0.75),
                "z": (-0.75, 0.75),
                "roll": (-0.75, 0.75),
                "pitch": (-0.75, 0.75),
                "yaw": (-0.75, 0.75),
            }
        },
    )


@configclass
class RewardsCfg:
    """Style Rewards"""

    undesired_contacts_thigh = RewTerm(
        func=mdp.undesired_contacts,
        weight=-30.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )

    undesired_contacts_shank = RewTerm(
        func=mdp.undesired_contacts,
        weight=-30.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*SHANK"), "threshold": 1.0},
    )

    base_height = RewTerm(
        func=mdp.base_below_min_height,
        weight=-10.0,
        params={"target_height": 0.5},
    )

    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    terminated = RewTerm(
        func=mdp.is_terminated_term,  # type: ignore
        params={"term_keys": "upside_down"},
        weight=-5000.0,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    upside_down = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(100)})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(
        func=mdp.distance_traveled,  # type: ignore
        params={
            "distance_thresholds": (5.0, 10.0),
        },
    )


@configclass
class ViewerCfg:
    """Configuration of the scene viewport camera."""

    """Initial camera position (in m)"""
    eye: tuple[float, float, float] = (0.0, -30.0, 20.0)
    lookat: tuple[float, float, float] = (0.0, 0.0, -2.0)
    cam_prim_path: str = "/OmniverseKit_Persp"
    resolution: tuple[int, int] = (1280, 720)
    origin_type: Literal["world", "env", "asset_root"] = "world"
    """
    * ``"world"``: The origin of the world.
    * ``"env"``: The origin of the environment defined by :attr:`env_index`.
    * ``"asset_root"``: The center of the asset defined by :attr:`asset_name` in environment :attr:`env_index`.
    """
    env_index: int = 0
    asset_name: str | None = None  # "robot"


##
# Environment configuration
##


@configclass
class UsdAnymalEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for USD Anymal environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=7.5)
    viewer: ViewerCfg = ViewerCfg()

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # skill discover stuff
        self.factors = factors
        self.skill_dims = skill_dims
        self.resampling_intervals = resampling_intervals
        self.usd_alg_extra_cfg = usd_alg_extra_cfg
        # general settings
        self.decimation = 4  # 50 Hz
        self.episode_length_s = 30.0  #
        # simulation settings
        self.sim.dt = 0.005  # 200 Hz
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material

        # GPU settings
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**28
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**24
        # self.sim.physx.gpu_collision_stack_size = 2**28
        # self.sim.physx.gpu_found_lost_pairs_capacity = 2**28

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if hasattr(self.scene, "lidar") and self.scene.lidar is not None:  # type: ignore
            self.scene.lidar.update_period = self.decimation * self.sim.dt  # type: ignore
        if hasattr(self.scene, "height_scanner") and self.scene.height_scanner is not None:  # type: ignore
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt  # type: ignore

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
