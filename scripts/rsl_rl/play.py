# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint for and agent trained via unsupervised skill discovery"""

"""Launch Isaac Sim Simulator first."""

import argparse
import threading

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch

from skill_gui import SkillControlGUI

# Import extensions to set up environment tasks
import d3_skill_discovery.tasks  # noqa: F401
from rsl_rl.runners import OnPolicyRunner, UsdOnPolicyRunner  # noqa: F401

from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    ppo_runner = UsdOnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device
    )  # agent_cfg.device)
    ppo_runner.load(resume_path, load_optimizer=False)

    # obtain the trained policy for inference
    # policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # prepare for exporting
    if hasattr(ppo_runner.alg.actor_critic, "prepare_export"):
        ppo_runner.alg.actor_critic.prepare_export()

    # export policy
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )

    try:
        export_policy_as_onnx(ppo_runner.alg.actor_critic, export_model_dir, filename="policy.onnx")
    except Exception as e:
        print(f"[WARNING] Exporting to ONNX failed with error: {e}")
    # reset environment
    obs, _ = env.get_observations()

    skill = ppo_runner.alg.usd.skill

    obs |= {"skill": skill}

    # setup skill GUI
    skill_info = {factor: (dim, env_cfg.factors[factor][1]) for factor, dim in env_cfg.skill_dims.items()}
    gui = SkillControlGUI(skill_info)
    threading.Thread(target=gui.launch, daemon=True).start()

    # load saved policy if it exists
    policy_path = os.path.join(export_model_dir, "policy.pt")
    if os.path.exists(policy_path):
        print(f"[INFO] Loading policy from: {policy_path}")
        loaded_policy = torch.jit.load(policy_path, map_location=env.unwrapped.device)
        policy_loaded = torch.jit.freeze(loaded_policy.eval())
    else:
        print(f"[WARNING] Policy file not found at: {policy_path}")

    def flatten_dict_obs(obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten the dictionary of tensors into a single tensor."""
        return torch.cat([obs_dict[key].flatten(1) for key in sorted(obs_dict.keys())], dim=-1)

    resample_skill_interval = int(5 / env.unwrapped.step_dt)
    resampling_skill = False
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            # actions1 = policy(obs)
            actions2 = policy_loaded(flatten_dict_obs(obs))
            # env stepping
            obs, _, _, _ = env.step(actions2)
            obs |= {"skill": skill}
        # if args_cli.video:
        #     # Exit the play loop after recording one video
        #     if timestep == args_cli.video_length:
        #         break

        if resampling_skill and timestep % resample_skill_interval == 0:
            # resample skill
            skill = ppo_runner.alg.usd._sample_skill(torch.ones(skill.shape[0], dtype=bool, device=skill.device))
        if not resampling_skill:
            # skill from GUI
            skill = gui.skill.clone().to(skill.device).unsqueeze(0).expand(skill.shape[0], -1)

        timestep += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
