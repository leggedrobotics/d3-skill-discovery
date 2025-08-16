import torch

from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCaster, RayCasterCfg, SensorBase, patterns
from isaaclab.utils import math as math_utils
from isaaclab.utils.timer import Timer
from rsl_rl.utils import TIMER_CUMULATIVE


def generated_goal(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Returns the generated goal for the given command.
    Note, the command needs to implement the goal property.
    This goal is not dependent on the current state, but is the goal for conditioning."""
    return env.command_manager._terms[command_name].goal  # type: ignore
