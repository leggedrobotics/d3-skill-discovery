# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
import zipfile
from collections import deque
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import imageio
from moviepy.editor import VideoFileClip, clips_array

import rsl_rl
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.intrinsic_motivation import FACTOR_USD
from rsl_rl.modules import ActorCritic  # noqa: F401
from rsl_rl.modules import EmpiricalNormalization, RelationalActorCriticRecurrent, RelationalActorCriticTransformer
from rsl_rl.utils import store_code_state


class UsdOnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        obs, infos = self.env.get_observations()
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # ActorCritic

        self.debug_mode = False
        self.exporting = False

        usd_obs = infos["observations"].get("usd", obs)

        if isinstance(obs, torch.Tensor):
            # flattened observation
            num_obs = obs.shape[1]
            if "critic" in infos["observations"]:
                num_critic_obs = infos["observations"]["critic"].shape[1]
            else:
                num_critic_obs = num_obs + skill_dim
            actor_critic: ActorCritic = actor_critic_class(
                num_obs + skill_dim, num_critic_obs + skill_dim, self.env.num_actions, **self.policy_cfg
            ).to(self.device)
        else:
            # dictionary observation

            # num_factors = len(env.unwrapped.cfg.factors) * int(self.cfg["usd"]["value_decomposition"])
            num_separate_rewards = 1  # for extrinsic reward
            for factor_name in env.unwrapped.cfg.factors.keys():
                if factor_name in env.unwrapped.cfg.usd_alg_extra_cfg:
                    num_separate_rewards += env.unwrapped.cfg.usd_alg_extra_cfg[factor_name].get("num_critics", 1)
                else:
                    num_separate_rewards += 1

            if not self.cfg["usd"]["value_decomposition"]:
                num_separate_rewards = 1

            full_skill_dim = sum(env.unwrapped.cfg.skill_dims.values()) + (len(env.unwrapped.cfg.factors) + 1) * int(
                self.cfg["usd"]["randomize_factor_weights"] and not self.cfg["usd"]["disable_factor_weighting"]
            )

            skill_dict = {"skill": torch.zeros(self.env.num_envs, full_skill_dim, device=self.device)}
            actor_critic: RelationalActorCriticTransformer | RelationalActorCriticRecurrent = actor_critic_class(
                actor_obs_dict=obs | skill_dict,
                critic_obs_dict=obs | skill_dict,
                num_actions=self.env.num_actions,
                num_critics=num_separate_rewards,
                **self.policy_cfg,
            ).to(self.device)

            num_obs = num_critic_obs = False
            critic_obs = [None]  # TODO critic from dict

        # intrinsic motivation and algorithm
        usd_class = eval(self.cfg["usd"].pop("class_name"))  # usd only
        sample_actor_obs = obs | skill_dict
        if self.exporting:
            sample_actor_obs = torch.cat([tensor.flatten(1) for _, tensor in sample_actor_obs.items()], dim=-1)

        usd: FACTOR_USD = usd_class(
            obs=usd_obs,
            infos=infos,
            device=self.device,
            N_steps=self.env.unwrapped.max_episode_length,
            num_deterministic_skills=env.cfg.num_videos if hasattr(env.cfg, "num_videos") else 0,
            sample_action=actor_critic.act(sample_actor_obs).detach(),
            # actor=actor_critic_copy,
            factors=env.unwrapped.cfg.factors,
            skill_dims=env.unwrapped.cfg.skill_dims,
            usd_alg_extra_cfg=env.unwrapped.cfg.usd_alg_extra_cfg,
            resampling_intervals=env.unwrapped.cfg.resampling_intervals,
            num_envs=self.env.num_envs,
            **self.cfg["usd"],
        )
        alg_class = eval(self.alg_cfg.pop("class_name"))  # PPO

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]

        self.alg: PPO = alg_class(
            actor_critic,
            device=self.device,
            usd=usd,
            scene_cfg=env.unwrapped.scene.cfg,
            reward_normalization=self.cfg["usd"]["reward_normalization"],
            extrinsic_reward_scale=self.cfg["usd"]["extrinsic_reward_scale"],
            rnd_only_keys=list(infos["observations"].get("rnd_extra", {}).keys()),
            warmup_steps=int(4 * self.env.max_episode_length // self.num_steps_per_env) if self.cfg["resume"] else 0,
            **self.alg_cfg,
        )

        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=obs, until=1e8).to(self.device)  # 1e8
            self.critic_obs_normalizer = EmpiricalNormalization(shape=obs, until=1e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity()  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity()  # no normalization

        self.video_recording = hasattr(self.env.env, "recording")  # a bit hacky

        # init storage and model

        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs + skill_dim] if num_obs else obs | skill_dict,
            [num_critic_obs] if num_critic_obs else critic_obs,  # TODO critic obs dict
            usd_obs | skill_dict | infos["observations"].get("rnd_extra", {}),
            [self.env.num_actions],
            num_rewards=num_separate_rewards,
        )
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.video_counter = -1
        self.video_writers = None
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, infos = self.env.get_observations()
        # obs, critic_obs = obs.to(self.device), critic_obs.to(self.device) # should be on device already
        self.train_mode()  # switch to train mode (for dropout for example)

        skill_command = self.alg.usd.skill
        obs = self.obs_normalizer(obs)

        # add skill to obs
        if isinstance(obs, torch.Tensor):
            obs = torch.cat((obs, skill_command), dim=1)
        else:
            obs["skill"] = skill_command

        if "critic" in infos["observations"]:
            critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"])
            if isinstance(critic_obs, torch.Tensor):
                critic_obs = torch.cat((critic_obs, skill_command), dim=1)
            else:
                critic_obs["skill"] = skill_command
        else:
            critic_obs = obs

        usd_obs = infos["observations"].get("usd", obs)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        instructor_reward_scaling = self.cfg["usd"]["instructor_reward_scaling"]

        time_last_check = time.time()
        run_hours = 16
        eval_metric_list = []
        usd_rew_list = []
        if self.debug_mode:
            self.env.unwrapped.episode_length_buf *= 0

        for it in range(start_iter, tot_iter):
            start = time.time()

            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs, critic_obs, usd_obs)
                    # Step environment
                    obs, rewards, dones, infos = self.env.step(actions)

                    # check if we have videos to save:
                    self.create_video_from_cameras()

                    regularization_reward = infos["observations"].get(
                        "regularization_reward", torch.zeros_like(rewards)
                    )
                    if self.cfg["usd"]["disable_regularization"]:
                        regularization_reward.zero_()

                    # store transition and add usd reward
                    if not self.debug_mode:
                        self.alg.process_env_step(rewards, regularization_reward, dones, infos)

                    # get extrinsic metric
                    self.alg.usd.save_extrinsic_performance(infos["observations"].get("metric", torch.tensor(1.0)))

                    # hacky way to pass usd metrics to the environment for curriculum
                    self.env.unwrapped.usd_metrics = self.alg.usd.curriculum_metric

                    # resample skill if needed
                    skill_command = self.alg.usd.update_skill(dones, self.env.episode_length_buf)

                    # normalize obs
                    obs = self.obs_normalizer(obs)
                    # add skill to obs
                    if isinstance(obs, torch.Tensor):
                        obs = torch.cat((obs, skill_command), dim=1)
                    else:
                        obs["skill"] = skill_command

                    # extract critic obs if available
                    if "critic" in infos["observations"]:
                        critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"])
                        if isinstance(critic_obs, torch.Tensor):
                            critic_obs = torch.cat((critic_obs, skill_command), dim=1)
                        else:
                            critic_obs["skill"] = skill_command
                    else:
                        critic_obs = obs

                    # usd observations
                    usd_obs = infos["observations"].get("usd", obs)

                    if self.log_dir is not None:
                        # Book keeping
                        # note: we changed logging to use "log" instead of "episode" to avoid confusion with
                        # different types of logging data (rewards, curriculum, etc.)
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                if self.debug_mode:
                    self.alg.storage.clear()
                    continue
                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss, usd_metrics = self.alg.update()

            usd_metrics["Curriculum/MaxEpisodeLength"] = self.env.unwrapped.max_episode_length
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if it % self.cfg["usd"]["visualizer_interval"] == self.cfg["usd"]["visualizer_interval"] // 2:
                imgs_path = os.path.join(self.log_dir, "visualizations", f"usd_{it}")
                os.makedirs(imgs_path, exist_ok=True)
                self.alg.usd.visualize(save_path=imgs_path, file_name=f"usd_{it}")
                for i, img in enumerate(os.listdir(imgs_path)):
                    img_path = os.path.join(imgs_path, img)
                    if self.logger_type in ["wandb"] and img_path is not None:
                        self.writer.log_image_from_file(f"usd_plots_{i}", img_path)

            if self.log_dir is not None:
                self.log(locals(), usd_metrics)
            if it % self.save_interval == 0:
                self.save(it, infos=infos, perf_metric=self.alg.usd.overall_metric)
            ep_infos.clear()
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def create_video_from_cameras(self):
        """
        Saves generated frames to video files. We merge the videos and zip them every time we have a new set of videos.
        For this to work, eval_video_frames must be set in the environment.
        eval_video_frames is a tuple with the first element being the video counter which counts the how many videos we made per env
        and the second element being a dictionary dict[any, np.ndarray] containing video frames as numpy arrays.
        """
        if not self.video_recording:
            return
        try:
            if hasattr(self.env.unwrapped, "eval_video_frame"):
                if self.env.unwrapped.eval_video_frame is None:
                    return
                video_counter, video_frames_dict = self.env.unwrapped.eval_video_frame
                if video_counter > self.video_counter:
                    # set is finished
                    if self.video_writers is not None:
                        # - close old video writers
                        for _, writer in self.video_writers.items():
                            writer.close()

                        # - merge videos:
                        clips = [VideoFileClip(writer._filename) for writer in self.video_writers.values()]
                        # Arrange them in a 2x4 grid
                        final_clip = clips_array(
                            [[clips[0], clips[1], clips[2], clips[3]], [clips[4], clips[5], clips[6], clips[7]]]
                        )
                        out_path = os.path.dirname(list(self.video_writers.values())[0]._filename)
                        final_clip.write_videofile(
                            os.path.join(out_path, f"merged_{self.video_counter}.mp4"), codec="libx264", audio=False
                        )

                        # - zip individual videos
                        zip_filename = os.path.join(out_path, f"videos_step_{self.video_counter}.zip")
                        with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
                            # Loop over each of the video files and add them to the zip
                            for _, writer in self.video_writers.items():
                                video_path = writer._filename
                                video_name = os.path.basename(video_path)
                                # Check if the file exists
                                if os.path.isfile(video_path):
                                    zipf.write(video_path, arcname=video_name)
                                else:
                                    print(f"Warning: {video_name} not found in {out_path}")
                        # - remove individual videos
                        for _, writer in self.video_writers.items():
                            os.remove(writer._filename)

                    # - new set of videos
                    video_path = os.path.join(self.log_dir, "cam_videos", f"videos_step_{video_counter}")
                    os.makedirs(video_path, exist_ok=True)
                    self.video_counter = video_counter
                    # create video writers
                    fps = 1.0 / self.env.unwrapped.step_dt
                    self.video_writers = {
                        vid_id: imageio.get_writer(
                            os.path.join(video_path, f"video_env_{vid_id}.mp4"), fps=fps, codec="libx264"
                        )
                        for vid_id in video_frames_dict.keys()
                    }

                # - append frames to video writers
                for vid_id, frame in video_frames_dict.items():
                    if frame is not None and self.video_writers is not None:
                        self.video_writers[vid_id].append_data(frame)
        except Exception as e:
            print(f"Error during video creation: {e}")

    def log(self, locs: dict, extras: dict = {}, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        for metric, value in extras.items():
            self.writer.add_scalar(metric, value, locs["it"])

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )
            if self.logger_type == "wandb":
                # upload video files
                self.writer.update_video_files(log_name="Video", fps=30)

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps:_}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )

        extras_string = ""
        for key, value in extras.items():
            extras_string += f"{f'{key}:':>{pad}} {value:.4f}\n"
        log_string += extras_string

        print(log_string)

    def save(self, it: int, infos=None, perf_metric: float = None):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }

        save_dict_usd = self.alg.usd.save()

        save_dict_all = {
            "actor_critic": saved_dict,
            "usd": save_dict_usd,
        }

        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()

        # we either save every model, or only the last and the best
        only_save_best_and_last = True  # TODO: make this a config option
        if only_save_best_and_last:
            if not hasattr(self, "best_performance"):
                self.best_performance = 0
            if perf_metric > self.best_performance:
                self.best_performance = perf_metric
                # save as the best model
                path = os.path.join(self.log_dir, "model_best.pt")  # type: ignore
            else:
                # save as the last model
                # we do not save the last model if its also the best model
                path = os.path.join(self.log_dir, "model_last.pt")  # type: ignore
        else:
            # save the model every time
            path = os.path.join(self.log_dir, f"model_{it}.pt")  # type: ignore

        torch.save(save_dict_all, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, weights_only=True)

        if "actor_critic" in loaded_dict:
            loaded_usd_dict = loaded_dict["usd"]
            loaded_dict = loaded_dict["actor_critic"]
        else:
            loaded_usd_dict = None

        # load actor and try to load critic
        try:
            self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        except Exception as e:
            print(f"[INFO] Error loading model state dict: {e}")
            print("[INFO] Trying to load model with strict=False")
            # at least actor needs to match
            non_matching_keys = set(loaded_dict["model_state_dict"].keys()) ^ set(
                self.alg.actor_critic.state_dict().keys()
            )
            actor_non_matching_keys = [key for key in non_matching_keys if "actor" in key]
            if len(actor_non_matching_keys) > 0:
                raise ValueError(
                    f"Actor state dict keys do not match: {actor_non_matching_keys}. "
                    "Please check the model architecture or the loaded weights."
                )
            # load the model with strict=False
            self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"], strict=False)
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])

        # load usd
        if loaded_usd_dict is not None:
            self.alg.usd.load(loaded_usd_dict, load_optimizer=load_optimizer)

        # load optimizer
        if load_optimizer:
            try:
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            except Exception as e:
                print(f"[INFO] Error loading optimizer state dict: {e}")
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        policy = self.alg.actor_critic.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.actor_critic.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self):
        self.alg.actor_critic.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        self.alg.actor_critic.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)
