from __future__ import annotations

import math
import numpy as np
import os
import time
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)
from whole_body_tracking.tasks.tracking.motion_pipeline import MotionSource, create_motion_source

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion_source: MotionSource = create_motion_source(
            mode=self.cfg.motion_source.mode,
            motion_file=self.cfg.motion_file,
            single_file=self.cfg.motion_source.single_file,
            library_file=self.cfg.motion_source.library_file,
            root_dir=self.cfg.motion_source.root_dir,
            normalize_weights=self.cfg.motion_source.normalize_weights,
            default_weight=self.cfg.motion_source.default_weight,
            body_indexes=self.body_indexes,
            body_names=self.cfg.body_names,
            device=self.device,
            target_fps=1.0 / float(env.cfg.decimation * env.cfg.sim.dt),
        )

        self.active_motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.active_frame_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Backward-compatible alias used by existing observation terms.
        self.time_steps = self.active_frame_ids

        motion_meta = self.motion_source.list_motion_meta()
        self._motion_num_frames = torch.tensor(
            [meta.num_frames for meta in motion_meta], dtype=torch.long, device=self.device
        )
        source_mode = str(getattr(self.cfg.motion_source, "mode", "single_npz")).lower()
        single_motion_file = motion_meta[0].file
        # Legacy MotionLoader only supports npz and is intended for the original single-file path.
        # For yaml libraries (even with one motion) and non-npz files (e.g. pkl), always use
        # MotionSource-backed cache path so pkl fallback remains valid.
        legacy_single_npz_compatible = (
            self.motion_source.num_motions() == 1
            and source_mode == "single_npz"
            and single_motion_file.lower().endswith(".npz")
        )
        self._multi_motion_enabled = not legacy_single_npz_compatible

        if self._multi_motion_enabled:
            self.motion = None
            self.bin_count = 1
            self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
            self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        else:
            self.motion = MotionLoader(single_motion_file, self.body_indexes, device=self.device)
            self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
            self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
            self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)

        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        if self._multi_motion_enabled:
            bootstrap_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            bootstrap_frames = torch.zeros_like(bootstrap_ids)
            bootstrap_batch = self.motion_source.fetch_frame_batch(bootstrap_ids, bootstrap_frames)
            dof_dim = int(bootstrap_batch.joint_pos.shape[1])
        else:
            dof_dim = int(self.motion.joint_pos.shape[1])

        self._joint_pos_cache = torch.zeros((self.num_envs, dof_dim), dtype=torch.float32, device=self.device)
        self._joint_vel_cache = torch.zeros_like(self._joint_pos_cache)
        self._body_pos_w_cache = torch.zeros((self.num_envs, len(cfg.body_names), 3), dtype=torch.float32, device=self.device)
        self._body_quat_w_cache = torch.zeros((self.num_envs, len(cfg.body_names), 4), dtype=torch.float32, device=self.device)
        self._body_lin_vel_w_cache = torch.zeros_like(self._body_pos_w_cache)
        self._body_ang_vel_w_cache = torch.zeros_like(self._body_pos_w_cache)

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        self._refresh_debug_counter = 0

        if self._multi_motion_enabled:
            self._refresh_motion_cache()

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        if self._multi_motion_enabled:
            return self._joint_pos_cache
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        if self._multi_motion_enabled:
            return self._joint_vel_cache
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        if self._multi_motion_enabled:
            return self._body_pos_w_cache + self._env.scene.env_origins[:, None, :]
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        if self._multi_motion_enabled:
            return self._body_quat_w_cache
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        if self._multi_motion_enabled:
            return self._body_lin_vel_w_cache
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        if self._multi_motion_enabled:
            return self._body_ang_vel_w_cache
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        if self._multi_motion_enabled:
            return self._body_pos_w_cache[:, self.motion_anchor_body_index] + self._env.scene.env_origins
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        if self._multi_motion_enabled:
            return self._body_quat_w_cache[:, self.motion_anchor_body_index]
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        if self._multi_motion_enabled:
            return self._body_lin_vel_w_cache[:, self.motion_anchor_body_index]
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        if self._multi_motion_enabled:
            return self._body_ang_vel_w_cache[:, self.motion_anchor_body_index]
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _normalize_env_ids(self, env_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    def _refresh_motion_cache(self) -> None:
        if not self._multi_motion_enabled:
            return
        t0 = time.perf_counter()
        batch = self.motion_source.fetch_frame_batch(self.active_motion_ids, self.active_frame_ids)
        self._joint_pos_cache = batch.joint_pos
        self._joint_vel_cache = batch.joint_vel
        self._body_pos_w_cache = batch.body_pos_w
        self._body_quat_w_cache = batch.body_quat_w
        self._body_lin_vel_w_cache = batch.body_lin_vel_w
        self._body_ang_vel_w_cache = batch.body_ang_vel_w
        if self._refresh_debug_counter < 5:
            unique_motion_count = int(torch.unique(self.active_motion_ids).numel())
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(
                f"[INFO] Motion cache refresh #{self._refresh_debug_counter + 1}: "
                f"envs={self.num_envs}, unique_motion_ids={unique_motion_count}, elapsed_ms={dt_ms:.2f}",
                flush=True,
            )
            self._refresh_debug_counter += 1

    def get_future_joint_obs(
        self,
        future_steps: Sequence[int],
        include_joint_pos: bool = True,
        include_joint_vel: bool = True,
    ) -> torch.Tensor:
        if not include_joint_pos and not include_joint_vel:
            return torch.zeros((self.num_envs, 0), device=self.device)

        step_offsets = torch.tensor(tuple(int(step) for step in future_steps), dtype=torch.long, device=self.device)
        if step_offsets.numel() == 0:
            return torch.zeros((self.num_envs, 0), device=self.device)

        if self._multi_motion_enabled:
            frame_indices = self.active_frame_ids.unsqueeze(1) + step_offsets.unsqueeze(0)
            max_frames = self._motion_num_frames[self.active_motion_ids].unsqueeze(1) - 1
            frame_indices = torch.minimum(torch.clamp(frame_indices, min=0), max_frames)
            future_joint_pos, future_joint_vel = self.motion_source.fetch_future_joint_batch(
                self.active_motion_ids, frame_indices
            )
        else:
            frame_indices = self.time_steps.unsqueeze(1) + step_offsets.unsqueeze(0)
            frame_indices = torch.clamp(frame_indices, min=0, max=self.motion.time_step_total - 1)
            future_joint_pos = self.motion.joint_pos[frame_indices]
            future_joint_vel = self.motion.joint_vel[frame_indices]

        obs_blocks: list[torch.Tensor] = []
        if include_joint_pos:
            obs_blocks.append(future_joint_pos.reshape(self.num_envs, -1))
        if include_joint_vel:
            obs_blocks.append(future_joint_vel.reshape(self.num_envs, -1))
        return torch.cat(obs_blocks, dim=-1)

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        if self._multi_motion_enabled:
            return
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = self._normalize_env_ids(env_ids)
        if env_ids.numel() == 0:
            return

        if self._multi_motion_enabled:
            motion_id_strategy = self.cfg.sampling.motion_id_strategy
            sampled_motion_ids = self.motion_source.sample_motion_ids(len(env_ids), device=self.device, strategy=motion_id_strategy)
            sampled_frame_ids = self.motion_source.sample_start_frames(
                sampled_motion_ids, random_start=self.cfg.sampling.random_start
            )

            self.active_motion_ids[env_ids] = sampled_motion_ids
            self.active_frame_ids[env_ids] = sampled_frame_ids

            # Fetch freshly sampled references before perturbation.
            sampled_batch = self.motion_source.fetch_frame_batch(sampled_motion_ids, sampled_frame_ids)
            root_pos = sampled_batch.body_pos_w[:, 0].clone() + self._env.scene.env_origins[env_ids]
            root_ori = sampled_batch.body_quat_w[:, 0].clone()
            root_lin_vel = sampled_batch.body_lin_vel_w[:, 0].clone()
            root_ang_vel = sampled_batch.body_ang_vel_w[:, 0].clone()

            range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
            ranges = torch.tensor(range_list, device=self.device)
            rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
            root_pos += rand_samples[:, 0:3]
            orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
            root_ori = quat_mul(orientations_delta, root_ori)

            range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
            ranges = torch.tensor(range_list, device=self.device)
            rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
            root_lin_vel += rand_samples[:, :3]
            root_ang_vel += rand_samples[:, 3:]

            joint_pos = sampled_batch.joint_pos.clone()
            joint_vel = sampled_batch.joint_vel.clone()
            joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
            soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
            joint_pos = torch.clip(joint_pos, soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1])

            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            self.robot.write_root_state_to_sim(
                torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1),
                env_ids=env_ids,
            )
            self._refresh_motion_cache()
            return

        self._adaptive_sampling(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        if self._multi_motion_enabled:
            self.active_frame_ids += 1
            max_frames = self._motion_num_frames[self.active_motion_ids]
            env_ids = torch.where(self.active_frame_ids >= max_frames)[0]
            self._resample_command(env_ids)
            self._refresh_motion_cache()
        else:
            self.time_steps += 1
            env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
            self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        if not self._multi_motion_enabled:
            self.bin_failed_count = (
                self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
            )
            self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionSourceCfg:
    """Configuration for selecting a motion source backend."""

    mode: str = "single_npz"
    single_file: str | None = None
    library_file: str | None = None
    root_dir: str | None = None
    normalize_weights: bool = True
    default_weight: float = 1.0

    def __post_init__(self):
        self.mode = self.mode.lower()
        if self.default_weight <= 0:
            raise ValueError(f"default_weight must be > 0, got: {self.default_weight}")


@configclass
class MotionSamplingCfg:
    """Configuration for motion-id and intra-motion frame sampling."""

    motion_id_strategy: str = "weighted"
    time_strategy: str = "legacy_adaptive"
    random_start: bool = True

    def __post_init__(self):
        self.motion_id_strategy = self.motion_id_strategy.lower()
        self.time_strategy = self.time_strategy.lower()


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING
    motion_source: MotionSourceCfg = MotionSourceCfg()
    sampling: MotionSamplingCfg = MotionSamplingCfg()

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
