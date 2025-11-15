# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import Literal

import d3_skill_discovery.tasks.unsupervised_skill_discovery.anymal_usd.mdp as mdp
from d3_skill_discovery.tasks.unsupervised_skill_discovery.anymal_usd.mdp.assets import CUBOID_FLAT_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
from isaaclab_assets.robots.ant import ANT_CFG

# isort: skip

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

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
    )

    robot: ArticulationCfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(2.0, 1.0)),
        debug_vis=False,
        mesh_prim_paths=[
            "/World/ground",
        ],
    )

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

    # box
    box1 = CUBOID_FLAT_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/Box_1",
        init_state=CUBOID_FLAT_CFG.InitialStateCfg(pos=(1.0, 0.0, 0.25)),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    skill_action = mdp.SkillSelectionActionCfg(
        asset_name="robot",
        low_level_action=mdp.JointPositionActionCfg(  # copied from velocity_env & box_climb_env
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        ),
        # skill_policy_file_path="/home/rafael/Projects/MT/CRL/d3_skill_discovery/logs/rsl_rl/metra_factors_ppo_anymal_test/usd_rough_sym_06/exported/policy.pt",
        skill_policy_file_name="usd_metra2d_diayn2d.pt",
        skill_space={
            "position": ("metra_norm_matching", 2),
            "heading": ("dirichlet", 2),
            "skill_weights": ("unit_sphere_positive", 3),
        },
        observation_group="ll_skill_policy",
        skill_policy_freq=50.0,
        debug_vis=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy.
        These observations need to be available from the robot's perspective.
        """

        origin = ObsTerm(
            func=mdp.origin_spawn,  # type: ignore
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        bbox_b = ObsTerm(
            func=mdp.object_bbox_in_robot_base,  # type: ignore
            params={
                "object_cfg": SceneEntityCfg("box1"),
                "control_frame": True,
            },
        )

        # proprioception
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
    class LowLevelSkillPolicyCfg(ObsGroup):
        """Observations for policy group."""

        origin = ObsTerm(
            func=mdp.origin_spawn,  # type: ignore
            params={"asset_cfg": SceneEntityCfg("robot"), "reset_every_n_steps": 1},
        )

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        # proprioception
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        actions = ObsTerm(func=mdp.last_low_level_action, params={"action_name": "skill_action"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        skill_command = ObsTerm(func=mdp.action_command, params={"action_name": "skill_action"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class UsdStateCfg(ObsGroup):
        """Observations for unsupervised skill discovery."""

        origin = ObsTerm(
            func=mdp.origin_spawn,  # type: ignore
            params={"asset_cfg": SceneEntityCfg("robot"), "noise_scale": 0.0, "pos_2d_only": True},
        )

        box_pose = ObsTerm(
            func=mdp.origin_spawn,  # type: ignore
            params={
                "asset_cfg": SceneEntityCfg("box1"),
                "spawn_asset_cfg": SceneEntityCfg("robot"),
                "pos_2d_only": True,
            },
        )

        box_bbox_spawn = ObsTerm(
            func=mdp.object_bbox_spawn,  # type: ignore
            params={
                "object_cfg": SceneEntityCfg("box1"),
                "spawn_asset_cfg": SceneEntityCfg("robot"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class MetricCfg(ObsGroup):
        """Observations for the metric that controls the extrinsic reward weight."""

        # # self
        metric = ObsTerm(
            func=mdp.base_height_flatness_metric,
            params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.55},
        )

    # - regularization
    @configclass
    class RegularizationRewards(ObsGroup):
        """Observations for the regularization rewards that are kept high."""

        # # self
        metric = ObsTerm(
            func=mdp.regularization_reward_obs,  # type: ignore
            params={
                "terms": {
                    "dof_torques_l2": RewTerm(func=mdp.joint_torques_l2, weight=-0.01),
                    "dof_acc_l2": RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7),
                    "action_rate_l2": RewTerm(func=mdp.action_rate_l2, weight=-0.05),
                    "torque_limits": RewTerm(func=mdp.applied_torque_limits, weight=-10.0),
                    "joint_vel_limits": RewTerm(func=mdp.joint_vel_limits, weight=-10.0, params={"soft_ratio": 1.0}),
                    "joint_pos_limits": RewTerm(func=mdp.joint_pos_limits, weight=-10.0),
                }
            },
        )

    @configclass
    class RNDextraCfg(ObsGroup):
        """Extra Observations for RND"""

        box_pose_box_spawn = ObsTerm(
            func=mdp.origin_spawn,  # type: ignore
            params={
                "asset_cfg": SceneEntityCfg("box1"),
                "spawn_asset_cfg": SceneEntityCfg("box1"),
            },
        )

        def __post_init__(self):
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    usd: UsdStateCfg = UsdStateCfg()
    rnd_extra: RNDextraCfg = RNDextraCfg()
    metric: MetricCfg = MetricCfg()
    regularization_reward: RegularizationRewards = RegularizationRewards()
    ll_skill_policy: LowLevelSkillPolicyCfg = LowLevelSkillPolicyCfg()


# factors for skill discovery
factors: dict[str, tuple[list[str], Literal["metra", "diayn"]]] = {
    "box": (
        [
            "box_pose",
            # "box_bbox_spawn",
        ],
        "metra",
    ),
}

skill_dims = {
    # "position" 3,
    "box": 3,  # 8
}

# resampling intervals in number of steps
resampling_intervals = {
    "box": -1,  # after termination
}

# overwrite default usd alg config
usd_alg_extra_cfg = {
    "box": {
        "use_rnd": True,
        "perturb_target": True,
        "rnd_weight": 100.0,
        "max_dirichlet_param": 1.0,
        "num_critics": 4,
        "rnd_obs": [
            "box_pose_box_spawn",
            # action,
            # skill,
            # ...
        ],
    },
}


reset_value = 0.1 * 0
reset_value_pos = 1.0


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 0.9),
            "dynamic_friction_range": (0.5, 0.7),
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

    physics_material_box = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("box1"),
            "static_friction_range": (0.5, 0.9),
            "dynamic_friction_range": (0.3, 0.7),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    add_box_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("box1"),
            "mass_distribution_params": (-2.0, 10.0),
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
                "z": (0.01, 0.01),
                "yaw": (-math.pi / 10, math.pi / 10),
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

    reset_box1 = EventTerm(
        func=mdp.reset_save_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.25, 0.25),
                "z": (0.01, 0.05),
                "yaw": (-math.pi / 10, math.pi / 10),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("box1"),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Style reward"""

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


@configclass
class ViewerCfg:
    """Configuration of the scene viewport camera."""

    eye: tuple[float, float, float] = (0.0, -30.0, 20.0)
    """Initial camera position (in m). Default is (7.5, 7.5, 7.5)."""
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
class UsdAnymalHLBoxEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for high-level unsupervised skill discovery with the ANYmal robot and boxes.
    Requires a low-level policy"""

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
        self.fz_planner = 10.0  # Hz
        self.sim.dt = 0.005  # 200 Hz
        self.decimation = int(1 / (self.sim.dt * self.fz_planner))
        self.episode_length_s = 60.0  #
        # simulation settings
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material

        # GPU settings
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**28
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**24
        # self.sim.physx.gpu_collision_stack_size = 2**28
        # self.sim.physx.gpu_found_lost_pairs_capacity = 2**28

        self.observations.ll_skill_policy.origin.params["reset_every_n_steps"] = self.decimation
        self.scene.contact_forces.history_length = self.decimation

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if hasattr(self.scene, "lidar") and self.scene.lidar is not None:  # type: ignore
            self.scene.lidar.update_period = self.decimation * self.sim.dt  # type: ignore
        if hasattr(self.scene, "height_scanner") and self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
