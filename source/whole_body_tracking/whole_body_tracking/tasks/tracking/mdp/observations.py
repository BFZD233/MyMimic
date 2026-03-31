from __future__ import annotations

from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING

import isaaclab.envs.mdp as base_mdp
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def twist2_like_motion_obs(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Motion/mimic-style block used by the twist2-like observation pipeline.

    Output layout:
    - command (joint_pos + joint_vel)
    - anchor relative position
    - anchor relative orientation (first 2 rotation-matrix columns, flattened)
    """
    return torch.cat(
        [
            base_mdp.generated_commands(env, command_name=command_name),
            motion_anchor_pos_b(env, command_name=command_name),
            motion_anchor_ori_b(env, command_name=command_name),
        ],
        dim=-1,
    )


def twist2_like_proprio_obs(env: ManagerBasedEnv) -> torch.Tensor:
    """Proprioceptive block used by the twist2-like observation pipeline.

    Output layout:
    - base angular velocity
    - joint position
    - joint velocity
    - previous action
    """
    return torch.cat(
        [
            base_mdp.base_ang_vel(env),
            base_mdp.joint_pos_rel(env),
            base_mdp.joint_vel_rel(env),
            base_mdp.last_action(env),
        ],
        dim=-1,
    )


def _normalize_future_steps(future_steps: Sequence[int] | torch.Tensor | int) -> tuple[int, ...]:
    if isinstance(future_steps, int):
        return (future_steps,)
    if isinstance(future_steps, torch.Tensor):
        return tuple(int(step) for step in future_steps.tolist())
    return tuple(int(step) for step in future_steps)


def motion_future_joint_obs(
    env: ManagerBasedEnv,
    command_name: str,
    future_steps: Sequence[int] | torch.Tensor | int = (0,),
    include_joint_pos: bool = True,
    include_joint_vel: bool = True,
) -> torch.Tensor:
    """Future joint feature block sampled from motion command timeline.

    This term is intentionally explicit and mask-free:
    - steps are controlled by `future_steps`
    - components are toggled by `include_joint_pos` / `include_joint_vel`
    """
    if not include_joint_pos and not include_joint_vel:
        return torch.zeros((env.num_envs, 0), device=env.device)

    command: MotionCommand = env.command_manager.get_term(command_name)
    step_offsets = _normalize_future_steps(future_steps)
    if len(step_offsets) == 0:
        return torch.zeros((env.num_envs, 0), device=command.time_steps.device)

    offset_tensor = torch.tensor(step_offsets, dtype=torch.long, device=command.time_steps.device)
    frame_indices = command.time_steps.unsqueeze(1) + offset_tensor.unsqueeze(0)
    frame_indices = torch.clamp(frame_indices, min=0, max=command.motion.time_step_total - 1)

    obs_blocks: list[torch.Tensor] = []
    if include_joint_pos:
        obs_blocks.append(command.motion.joint_pos[frame_indices].reshape(env.num_envs, -1))
    if include_joint_vel:
        obs_blocks.append(command.motion.joint_vel[frame_indices].reshape(env.num_envs, -1))

    return torch.cat(obs_blocks, dim=-1)
