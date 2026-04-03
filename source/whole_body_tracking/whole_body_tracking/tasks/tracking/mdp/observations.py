from __future__ import annotations

from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING

import isaaclab.envs.mdp as base_mdp
from isaaclab.utils.math import euler_xyz_from_quat, matrix_from_quat, quat_apply_inverse, subtract_frame_transforms

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_lin_vel_w.view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_ang_vel_w.view(env.num_envs, -1)


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


def _roll_pitch_from_quat(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    roll, pitch, _ = euler_xyz_from_quat(quat_wxyz)
    return roll, pitch


def twist2_1432_motion_obs(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """TWIST2-compatible mimic block (35 dims per frame).

    Layout:
    - anchor linear velocity XY in anchor-local frame (2)
    - anchor position Z in world frame (1)
    - anchor roll/pitch (2)
    - anchor angular velocity yaw in anchor-local frame (1)
    - target joint position (29)
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    anchor_lin_vel_b = quat_apply_inverse(command.anchor_quat_w, command.anchor_lin_vel_w)
    anchor_ang_vel_b = quat_apply_inverse(command.anchor_quat_w, command.anchor_ang_vel_w)
    roll, pitch = _roll_pitch_from_quat(command.anchor_quat_w)
    return torch.cat(
        [
            anchor_lin_vel_b[:, :2],
            command.anchor_pos_w[:, 2:3],
            roll.unsqueeze(-1),
            pitch.unsqueeze(-1),
            anchor_ang_vel_b[:, 2:3],
            command.joint_pos,
        ],
        dim=-1,
    )


def twist2_1432_proprio_obs(env: ManagerBasedEnv) -> torch.Tensor:
    """TWIST2-compatible proprio block (92 dims per frame).

    Layout:
    - base angular velocity (3)
    - robot roll/pitch (2)
    - joint position relative (29)
    - joint velocity relative (29)
    - previous action (29)
    """
    command: MotionCommand = env.command_manager.get_term("motion")
    roll, pitch = _roll_pitch_from_quat(command.robot_anchor_quat_w)
    return torch.cat(
        [
            base_mdp.base_ang_vel(env),
            roll.unsqueeze(-1),
            pitch.unsqueeze(-1),
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


def twist2_1432_future_motion_obs(
    env: ManagerBasedEnv,
    command_name: str,
    future_steps: Sequence[int] | torch.Tensor | int = (0,),
) -> torch.Tensor:
    """TWIST2-compatible future mimic block.

    Each future step contributes 35 dims with the same semantic layout as
    :func:`twist2_1432_motion_obs`. Future anchor terms are approximated from the
    current anchor state, while future joint positions come from motion timeline.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    step_offsets = _normalize_future_steps(future_steps)
    num_steps = len(step_offsets)
    if num_steps == 0:
        step_offsets = (0,)
        num_steps = 1

    future_joint_pos = command.get_future_joint_obs(
        future_steps=step_offsets,
        include_joint_pos=True,
        include_joint_vel=False,
    )
    dof_dim = int(future_joint_pos.shape[1] // num_steps)
    future_joint_pos = future_joint_pos.view(env.num_envs, num_steps, dof_dim)

    anchor_lin_vel_b = quat_apply_inverse(command.anchor_quat_w, command.anchor_lin_vel_w)
    anchor_ang_vel_b = quat_apply_inverse(command.anchor_quat_w, command.anchor_ang_vel_w)
    roll, pitch = _roll_pitch_from_quat(command.anchor_quat_w)
    anchor_block = torch.cat(
        [
            anchor_lin_vel_b[:, :2],
            command.anchor_pos_w[:, 2:3],
            roll.unsqueeze(-1),
            pitch.unsqueeze(-1),
            anchor_ang_vel_b[:, 2:3],
        ],
        dim=-1,
    )
    anchor_block = anchor_block.unsqueeze(1).expand(-1, num_steps, -1)
    future_obs = torch.cat([anchor_block, future_joint_pos], dim=-1)
    return future_obs.reshape(env.num_envs, -1)


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
    command: MotionCommand = env.command_manager.get_term(command_name)
    step_offsets = _normalize_future_steps(future_steps)
    return command.get_future_joint_obs(
        future_steps=step_offsets,
        include_joint_pos=include_joint_pos,
        include_joint_vel=include_joint_vel,
    )
