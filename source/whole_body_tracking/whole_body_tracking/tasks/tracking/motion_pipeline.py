from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
import pickle
import sys

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional in minimal envs
    def tqdm(iterable, *args, **kwargs):
        del args, kwargs
        return iterable


REQUIRED_NPZ_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
PKL_MOTION_KEYS = ("fps", "root_pos", "root_rot", "dof_pos", "local_body_pos", "link_body_list")
PKL_FALLBACK_WARNING_EMITTED = False


@dataclass(frozen=True)
class MotionMeta:
    motion_id: int
    name: str
    file: str
    num_frames: int
    fps: float
    duration_s: float
    weight: float = 1.0
    tag: str | None = None
    extras: dict[str, Any] | None = None


@dataclass
class MotionFrameBatch:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_ang_vel_w: torch.Tensor


def _resolve_path(path_like: str | None) -> Path | None:
    if path_like is None or path_like == "":
        return None
    return Path(path_like).expanduser().resolve()


def _to_body_index_tensor(body_indexes: Sequence[int] | torch.Tensor, device: str | torch.device) -> torch.Tensor:
    return torch.as_tensor(body_indexes, dtype=torch.long, device=device)


def _fps_to_float(fps_value: np.ndarray | float | int) -> float:
    fps_arr = np.asarray(fps_value)
    if fps_arr.size == 0:
        raise ValueError("Invalid fps: empty value.")
    fps = float(fps_arr.reshape(-1)[0])
    if fps <= 0:
        raise ValueError(f"Invalid fps: {fps}")
    return fps


def _quat_rotate_xyzw(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternions in xyzw format."""
    q_xyz = quat[..., :3]
    q_w = quat[..., 3:4]
    uv = torch.cross(q_xyz, vec, dim=-1)
    uuv = torch.cross(q_xyz, uv, dim=-1)
    return vec + 2.0 * (q_w * uv + uuv)


def _quat_conjugate_xyzw(quat: torch.Tensor) -> torch.Tensor:
    out = quat.clone()
    out[..., :3] = -out[..., :3]
    return out


def _quat_mul_xyzw(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    qx, qy, qz, qw = q.unbind(dim=-1)
    rx, ry, rz, rw = r.unbind(dim=-1)
    x = qw * rx + qx * rw + qy * rz - qz * ry
    y = qw * ry - qx * rz + qy * rw + qz * rx
    z = qw * rz + qx * ry - qy * rx + qz * rw
    w = qw * rw - qx * rx - qy * ry - qz * rz
    return torch.stack((x, y, z, w), dim=-1)


def _quat_to_exp_map_xyzw(quat: torch.Tensor) -> torch.Tensor:
    quat = quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=1e-8)
    xyz = quat[..., :3]
    w = torch.clamp(quat[..., 3], min=-1.0, max=1.0)
    sin_half = torch.linalg.norm(xyz, dim=-1)
    angle = 2.0 * torch.atan2(sin_half, w)
    axis = xyz / torch.clamp(sin_half.unsqueeze(-1), min=1e-8)
    return axis * angle.unsqueeze(-1)


def _compute_ang_vel_from_quat_xyzw(quat: torch.Tensor, dt: float) -> torch.Tensor:
    if quat.shape[0] == 1:
        return torch.zeros((1, 3), dtype=quat.dtype, device=quat.device)
    if quat.shape[0] == 2:
        q_rel = _quat_mul_xyzw(quat[1:2], _quat_conjugate_xyzw(quat[0:1]))
        omega = _quat_to_exp_map_xyzw(q_rel) / max(dt, 1e-8)
        return omega.repeat(2, 1)

    # Central difference for interior points, matching batch_pkl_to_npz.py behavior.
    q_prev = quat[:-2]
    q_next = quat[2:]
    q_rel = _quat_mul_xyzw(q_next, _quat_conjugate_xyzw(q_prev))
    omega_mid = _quat_to_exp_map_xyzw(q_rel) / max(2.0 * dt, 1e-8)
    omega = torch.cat([omega_mid[:1], omega_mid, omega_mid[-1:]], dim=0)
    return omega


def _quat_normalize_xyzw(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=1e-8)


def _quat_slerp_xyzw(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Slerp for quaternions in xyzw format.

    Args:
        q0: shape [N, 4]
        q1: shape [N, 4]
        t: shape [N]
    """
    q0 = _quat_normalize_xyzw(q0)
    q1 = _quat_normalize_xyzw(q1)

    dot = torch.sum(q0 * q1, dim=-1)
    neg_mask = dot < 0.0
    q1 = torch.where(neg_mask.unsqueeze(-1), -q1, q1)
    dot = torch.where(neg_mask, -dot, dot)
    dot = torch.clamp(dot, -1.0, 1.0)

    # Linear fallback for tiny angles.
    linear_mask = dot > 0.9995
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * t
    sin_theta = torch.sin(theta)

    s0 = torch.cos(theta) - dot * sin_theta / torch.clamp(sin_theta_0, min=1e-8)
    s1 = sin_theta / torch.clamp(sin_theta_0, min=1e-8)
    out_slerp = s0.unsqueeze(-1) * q0 + s1.unsqueeze(-1) * q1

    out_lerp = (1.0 - t).unsqueeze(-1) * q0 + t.unsqueeze(-1) * q1
    out = torch.where(linear_mask.unsqueeze(-1), out_lerp, out_slerp)
    return _quat_normalize_xyzw(out)


def _lerp_tensor(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return a * (1.0 - blend) + b * blend


def _resample_pkl_tracks_to_target_fps(
    *,
    root_pos: torch.Tensor,
    root_rot_xyzw: torch.Tensor,
    dof_pos: torch.Tensor,
    local_body_pos: torch.Tensor,
    input_fps: float,
    target_fps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if target_fps <= 0:
        raise ValueError(f"target_fps must be > 0, got {target_fps}")
    if input_fps <= 0:
        raise ValueError(f"input_fps must be > 0, got {input_fps}")
    if root_pos.shape[0] < 2:
        return root_pos, root_rot_xyzw, dof_pos, local_body_pos

    duration = (root_pos.shape[0] - 1) / input_fps
    output_dt = 1.0 / target_fps
    times = torch.arange(0.0, duration, output_dt, dtype=torch.float32, device=root_pos.device)
    if times.numel() == 0:
        times = torch.zeros(1, dtype=torch.float32, device=root_pos.device)

    phase = torch.clamp(times / max(duration, 1e-8), 0.0, 1.0)
    index_0 = torch.floor(phase * (root_pos.shape[0] - 1)).long()
    index_1 = torch.minimum(index_0 + 1, torch.full_like(index_0, root_pos.shape[0] - 1))
    blend = phase * (root_pos.shape[0] - 1) - index_0.float()

    root_pos_out = _lerp_tensor(root_pos[index_0], root_pos[index_1], blend.unsqueeze(-1))
    root_rot_out = _quat_slerp_xyzw(root_rot_xyzw[index_0], root_rot_xyzw[index_1], blend)
    dof_pos_out = _lerp_tensor(dof_pos[index_0], dof_pos[index_1], blend.unsqueeze(-1))
    local_body_pos_out = _lerp_tensor(
        local_body_pos[index_0],
        local_body_pos[index_1],
        blend.view(-1, 1, 1),
    )
    return root_pos_out, root_rot_out, dof_pos_out, local_body_pos_out


class MotionSource(ABC):
    @abstractmethod
    def num_motions(self) -> int:
        pass

    @abstractmethod
    def get_motion_meta(self, motion_id: int) -> MotionMeta:
        pass

    @abstractmethod
    def list_motion_meta(self) -> list[MotionMeta]:
        pass

    @abstractmethod
    def sample_motion_ids(self, n: int, device: str | torch.device, strategy: str = "weighted") -> torch.Tensor:
        pass

    @abstractmethod
    def sample_start_frames(self, motion_ids: torch.Tensor, random_start: bool = True) -> torch.Tensor:
        pass

    @abstractmethod
    def fetch_frame_batch(self, motion_ids: torch.Tensor, frame_ids: torch.Tensor) -> MotionFrameBatch:
        pass

    @abstractmethod
    def fetch_future_joint_batch(
        self, motion_ids: torch.Tensor, frame_ids_2d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass


def _read_motion_payload(path: Path) -> dict[str, Any]:
    try:
        payload = np.load(path, allow_pickle=True)
        if isinstance(payload, dict):
            return payload
        if hasattr(payload, "files"):
            data = {key: payload[key] for key in payload.files}
            payload.close()
            return data
    except Exception:
        pass

    # Compatibility alias for pickles generated with numpy>=2 that reference `numpy._core`.
    if "numpy._core" not in sys.modules:
        try:
            import numpy._core as np_core  # type: ignore[attr-defined]
        except Exception:
            import numpy.core as np_core  # type: ignore[no-redef]
        sys.modules["numpy._core"] = np_core
    if "numpy._core.multiarray" not in sys.modules:
        try:
            import numpy.core.multiarray as np_multiarray
            sys.modules["numpy._core.multiarray"] = np_multiarray
        except Exception:
            pass
    if "numpy._core.numeric" not in sys.modules:
        try:
            import numpy.core.numeric as np_numeric
            sys.modules["numpy._core.numeric"] = np_numeric
        except Exception:
            pass

    with path.open("rb") as file:
        obj = pickle.load(file)
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported motion payload type in {path}: {type(obj).__name__}")
    return obj


def _load_motion_arrays_for_path(
    *,
    path: Path,
    body_indexes: torch.Tensor,
    body_names: Sequence[str] | None,
    device: str | torch.device,
    target_fps: float | None = None,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = _read_motion_payload(path)
    keys = set(payload.keys())

    # Native npz format path (preferred for compatibility with existing BeyondMimic behavior).
    if all(key in keys for key in REQUIRED_NPZ_KEYS):
        fps = _fps_to_float(payload["fps"])
        joint_pos = torch.tensor(payload["joint_pos"], dtype=torch.float32, device=device)
        joint_vel = torch.tensor(payload["joint_vel"], dtype=torch.float32, device=device)
        body_pos_w = torch.tensor(payload["body_pos_w"], dtype=torch.float32, device=device)[:, body_indexes]
        body_quat_w = torch.tensor(payload["body_quat_w"], dtype=torch.float32, device=device)[:, body_indexes]
        body_lin_vel_w = torch.tensor(payload["body_lin_vel_w"], dtype=torch.float32, device=device)[:, body_indexes]
        body_ang_vel_w = torch.tensor(payload["body_ang_vel_w"], dtype=torch.float32, device=device)[:, body_indexes]
        return fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w

    # TWIST2 pkl format fallback.
    if all(key in keys for key in PKL_MOTION_KEYS):
        global PKL_FALLBACK_WARNING_EMITTED
        if body_names is None:
            raise ValueError(
                f"Loading pkl motion requires `body_names` for link mapping. Missing for: {path}"
            )
        if not PKL_FALLBACK_WARNING_EMITTED:
            print(
                "[WARNING] Loading pkl motions through online fallback path. "
                "For best tracking fidelity, prefer offline conversion via scripts/batch_pkl_to_npz.py.",
                flush=True,
            )
            PKL_FALLBACK_WARNING_EMITTED = True
        fps = _fps_to_float(payload["fps"])
        original_fps = fps

        root_pos = torch.tensor(payload["root_pos"], dtype=torch.float32, device=device)
        root_rot = torch.tensor(payload["root_rot"], dtype=torch.float32, device=device)
        dof_pos = torch.tensor(payload["dof_pos"], dtype=torch.float32, device=device)
        local_body_pos = torch.tensor(payload["local_body_pos"], dtype=torch.float32, device=device)
        link_body_list = [str(name) for name in np.asarray(payload["link_body_list"]).tolist()]

        if dof_pos.ndim != 2:
            raise ValueError(f"Expected dof_pos shape [T, D], got {tuple(dof_pos.shape)} in {path}")
        if root_pos.shape[0] < 2:
            raise ValueError(f"Motion requires at least 2 frames: {path}")
        if root_pos.shape[0] != dof_pos.shape[0] or root_pos.shape[0] != local_body_pos.shape[0]:
            raise ValueError(
                f"Inconsistent frame count in {path}: root={root_pos.shape[0]}, "
                f"dof={dof_pos.shape[0]}, body={local_body_pos.shape[0]}"
            )

        missing_body_names = [name for name in body_names if name not in link_body_list]
        if missing_body_names:
            raise ValueError(
                f"Body names not found in pkl link_body_list for {path}: {missing_body_names[:8]}"
            )
        body_name_to_index = {name: idx for idx, name in enumerate(link_body_list)}
        selected_indices = torch.tensor([body_name_to_index[name] for name in body_names], dtype=torch.long, device=device)
        selected_local_body_pos = local_body_pos[:, selected_indices]

        if target_fps is not None and abs(float(target_fps) - float(fps)) > 1e-6:
            old_frames = int(root_pos.shape[0])
            root_pos, root_rot, dof_pos, local_body_pos = _resample_pkl_tracks_to_target_fps(
                root_pos=root_pos,
                root_rot_xyzw=root_rot,
                dof_pos=dof_pos,
                local_body_pos=local_body_pos,
                input_fps=fps,
                target_fps=float(target_fps),
            )
            fps = float(target_fps)
            selected_local_body_pos = local_body_pos[:, selected_indices]
            print(
                f"[INFO] Resampled pkl motion {path.name}: fps {original_fps:.3f} -> {fps:.3f}, "
                f"frames {old_frames} -> {int(root_pos.shape[0])}",
                flush=True,
            )

        dt = 1.0 / fps
        root_rot_rep = root_rot[:, None, :].expand(-1, selected_local_body_pos.shape[1], -1)
        body_pos_w = root_pos[:, None, :] + _quat_rotate_xyzw(root_rot_rep, selected_local_body_pos)
        body_quat_w = root_rot_rep.clone()

        joint_pos = dof_pos
        joint_vel = torch.gradient(joint_pos, spacing=dt, dim=0)[0]
        body_lin_vel_w = torch.gradient(body_pos_w, spacing=dt, dim=0)[0]
        root_ang_vel = _compute_ang_vel_from_quat_xyzw(root_rot, dt)
        body_ang_vel_w = root_ang_vel[:, None, :].expand(-1, selected_local_body_pos.shape[1], -1).clone()

        return fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w

    raise ValueError(
        f"Unsupported motion format in {path}. Expected keys {REQUIRED_NPZ_KEYS} or {PKL_MOTION_KEYS}, "
        f"got keys: {sorted(keys)[:20]}"
    )


class SingleNpzMotionSource(MotionSource):
    def __init__(
        self,
        motion_file: str | Path,
        body_indexes: Sequence[int] | torch.Tensor,
        device: str | torch.device,
        body_names: Sequence[str] | None = None,
        target_fps: float | None = None,
    ):
        motion_path = Path(motion_file).expanduser().resolve()
        if not motion_path.is_file():
            raise FileNotFoundError(f"Invalid single motion file: {motion_path}")

        self._device = device
        self._body_indexes = _to_body_index_tensor(body_indexes, device)
        self._body_names = tuple(body_names) if body_names is not None else None
        self._target_fps = target_fps
        self._motion_file = motion_path
        self._load_motion_file()

    def _load_motion_file(self) -> None:
        fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w = _load_motion_arrays_for_path(
            path=self._motion_file,
            body_indexes=self._body_indexes,
            body_names=self._body_names,
            device=self._device,
            target_fps=self._target_fps,
        )

        if joint_pos.ndim != 2:
            raise ValueError(f"Expected joint_pos shape [T, D], got {tuple(joint_pos.shape)} in {self._motion_file}")
        if joint_vel.shape != joint_pos.shape:
            raise ValueError(
                f"joint_vel shape {tuple(joint_vel.shape)} does not match joint_pos shape {tuple(joint_pos.shape)} "
                f"in {self._motion_file}"
            )
        if joint_pos.shape[0] < 2:
            raise ValueError(f"Motion requires at least 2 frames: {self._motion_file}")

        self.joint_pos = joint_pos
        self.joint_vel = joint_vel
        self.body_pos_w = body_pos_w
        self.body_quat_w = body_quat_w
        self.body_lin_vel_w = body_lin_vel_w
        self.body_ang_vel_w = body_ang_vel_w
        self.num_frames = int(joint_pos.shape[0])
        self.fps = fps

        self._meta = [
            MotionMeta(
                motion_id=0,
                name=self._motion_file.stem,
                file=str(self._motion_file),
                num_frames=self.num_frames,
                fps=fps,
                duration_s=(self.num_frames - 1) / fps,
            )
        ]

    def num_motions(self) -> int:
        return 1

    def get_motion_meta(self, motion_id: int) -> MotionMeta:
        if motion_id != 0:
            raise IndexError(f"Single source only supports motion_id=0, got {motion_id}")
        return self._meta[0]

    def list_motion_meta(self) -> list[MotionMeta]:
        return self._meta

    def sample_motion_ids(self, n: int, device: str | torch.device, strategy: str = "weighted") -> torch.Tensor:
        del strategy
        return torch.zeros(n, dtype=torch.long, device=device)

    def sample_start_frames(self, motion_ids: torch.Tensor, random_start: bool = True) -> torch.Tensor:
        if not random_start:
            return torch.zeros_like(motion_ids)
        frame_ids = torch.floor(torch.rand_like(motion_ids, dtype=torch.float32) * float(self.num_frames)).long()
        return torch.clamp(frame_ids, min=0, max=self.num_frames - 1)

    def fetch_frame_batch(self, motion_ids: torch.Tensor, frame_ids: torch.Tensor) -> MotionFrameBatch:
        del motion_ids
        frames = torch.clamp(frame_ids.long(), min=0, max=self.num_frames - 1)
        return MotionFrameBatch(
            joint_pos=self.joint_pos[frames],
            joint_vel=self.joint_vel[frames],
            body_pos_w=self.body_pos_w[frames],
            body_quat_w=self.body_quat_w[frames],
            body_lin_vel_w=self.body_lin_vel_w[frames],
            body_ang_vel_w=self.body_ang_vel_w[frames],
        )

    def fetch_future_joint_batch(
        self, motion_ids: torch.Tensor, frame_ids_2d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del motion_ids
        frames = torch.clamp(frame_ids_2d.long(), min=0, max=self.num_frames - 1)
        return self.joint_pos[frames], self.joint_vel[frames]


class YamlNpzMotionLibrarySource(MotionSource):
    def __init__(
        self,
        library_file: str | Path,
        body_indexes: Sequence[int] | torch.Tensor,
        device: str | torch.device,
        body_names: Sequence[str] | None = None,
        root_dir: str | None = None,
        normalize_weights: bool = True,
        default_weight: float = 1.0,
        target_fps: float | None = None,
    ):
        self._device = device
        self._body_indexes = _to_body_index_tensor(body_indexes, device)
        self._body_names = tuple(body_names) if body_names is not None else None
        self._library_file = Path(library_file).expanduser().resolve()
        if not self._library_file.is_file():
            raise FileNotFoundError(f"Motion library file does not exist: {self._library_file}")

        self._normalize_weights = normalize_weights
        self._default_weight = float(default_weight)
        self._target_fps = target_fps
        if self._default_weight <= 0:
            raise ValueError(f"default_weight must be > 0, got: {self._default_weight}")

        self._root_dir_override = _resolve_path(root_dir)

        self._meta: list[MotionMeta] = []
        # Temporary per-motion chunks during loading.
        self._joint_pos_chunks: list[torch.Tensor] = []
        self._joint_vel_chunks: list[torch.Tensor] = []
        self._body_pos_w_chunks: list[torch.Tensor] = []
        self._body_quat_w_chunks: list[torch.Tensor] = []
        self._body_lin_vel_w_chunks: list[torch.Tensor] = []
        self._body_ang_vel_w_chunks: list[torch.Tensor] = []
        # Concatenated storage for fast batched gather via global frame index.
        self._joint_pos_cat: torch.Tensor | None = None
        self._joint_vel_cat: torch.Tensor | None = None
        self._body_pos_w_cat: torch.Tensor | None = None
        self._body_quat_w_cat: torch.Tensor | None = None
        self._body_lin_vel_w_cat: torch.Tensor | None = None
        self._body_ang_vel_w_cat: torch.Tensor | None = None
        self._frame_offsets: torch.Tensor | None = None
        self._weights_list: list[float] = []
        self._num_frames_list: list[int] = []

        self._load_library()

    def _resolve_motion_path(self, motion_file: str, yaml_root: Path) -> Path:
        motion_path = Path(motion_file).expanduser()
        if motion_path.is_absolute():
            return motion_path.resolve()
        if self._root_dir_override is not None:
            return (self._root_dir_override / motion_path).resolve()
        return (yaml_root / motion_path).resolve()

    def _load_library(self) -> None:
        with self._library_file.open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file)

        if not isinstance(payload, dict):
            raise ValueError(f"Expected dict yaml payload in {self._library_file}, got {type(payload).__name__}")

        motion_entries = payload.get("motions", None)
        if not isinstance(motion_entries, list) or len(motion_entries) == 0:
            raise ValueError(f"Motion library must define non-empty 'motions' list: {self._library_file}")

        root_path_field = payload.get("root_path", None)
        if self._root_dir_override is not None:
            yaml_root = self._root_dir_override
        elif root_path_field:
            yaml_root = Path(root_path_field).expanduser().resolve()
        else:
            yaml_root = self._library_file.parent

        first_joint_dim: int | None = None
        first_body_dim: int | None = None

        print(
            f"[INFO] Loading motion library: {self._library_file} "
            f"(motions={len(motion_entries)}, root={yaml_root})",
            flush=True,
        )
        motion_iter = tqdm(
            enumerate(motion_entries),
            total=len(motion_entries),
            desc=f"[MotionLibrary] {self._library_file.name}",
            unit="motion",
            dynamic_ncols=True,
            file=sys.stdout,
        )
        for motion_id, entry in motion_iter:
            if not isinstance(entry, dict):
                raise ValueError(f"Each motion entry must be dict. Got: {type(entry).__name__} at index {motion_id}")

            file_value = entry.get("file", None)
            if not isinstance(file_value, str) or file_value == "":
                raise ValueError(f"Motion entry missing non-empty 'file' at index {motion_id}: {entry}")

            motion_path = self._resolve_motion_path(file_value, yaml_root)
            if not motion_path.is_file():
                raise FileNotFoundError(f"Motion file does not exist: {motion_path}")

            weight = float(entry.get("weight", self._default_weight))
            if weight <= 0:
                raise ValueError(f"Motion weight must be > 0 at index {motion_id}: {weight}")

            fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w = _load_motion_arrays_for_path(
                path=motion_path,
                body_indexes=self._body_indexes,
                body_names=self._body_names,
                device=self._device,
                target_fps=self._target_fps,
            )
            if "fps_override" in entry:
                fps = _fps_to_float(entry["fps_override"])

            if joint_pos.ndim != 2:
                raise ValueError(f"Expected joint_pos shape [T, D], got {tuple(joint_pos.shape)} in {motion_path}")
            if joint_vel.shape != joint_pos.shape:
                raise ValueError(
                    f"joint_vel shape {tuple(joint_vel.shape)} does not match joint_pos shape {tuple(joint_pos.shape)} "
                    f"in {motion_path}"
                )
            if joint_pos.shape[0] < 2:
                raise ValueError(f"Motion requires at least 2 frames: {motion_path}")

            if first_joint_dim is None:
                first_joint_dim = int(joint_pos.shape[1])
            if int(joint_pos.shape[1]) != first_joint_dim:
                raise ValueError(
                    f"Inconsistent dof dimension in library. Expected {first_joint_dim}, got {joint_pos.shape[1]} "
                    f"for {motion_path}"
                )

            if first_body_dim is None:
                first_body_dim = int(body_pos_w.shape[1])
            if int(body_pos_w.shape[1]) != first_body_dim:
                raise ValueError(
                    f"Inconsistent body dimension in library. Expected {first_body_dim}, got {body_pos_w.shape[1]} "
                    f"for {motion_path}"
                )

            num_frames = int(joint_pos.shape[0])
            name = str(entry.get("name", motion_path.stem))
            tag = entry.get("tag", None)
            extras = {k: v for k, v in entry.items() if k not in {"file", "weight", "name", "tag", "fps_override"}}

            self._meta.append(
                MotionMeta(
                    motion_id=motion_id,
                    name=name,
                    file=str(motion_path),
                    num_frames=num_frames,
                    fps=fps,
                    duration_s=(num_frames - 1) / fps,
                    weight=weight,
                    tag=tag,
                    extras=extras if extras else None,
                )
            )
            self._joint_pos_chunks.append(joint_pos)
            self._joint_vel_chunks.append(joint_vel)
            self._body_pos_w_chunks.append(body_pos_w)
            self._body_quat_w_chunks.append(body_quat_w)
            self._body_lin_vel_w_chunks.append(body_lin_vel_w)
            self._body_ang_vel_w_chunks.append(body_ang_vel_w)
            self._weights_list.append(weight)
            self._num_frames_list.append(num_frames)

        if len(self._meta) == 0:
            raise ValueError(f"No valid motions loaded from {self._library_file}")

        self._num_frames = torch.tensor(self._num_frames_list, dtype=torch.long, device=self._device)
        self._weights = torch.tensor(self._weights_list, dtype=torch.float32, device=self._device)
        if torch.any(self._weights <= 0):
            raise ValueError(f"All motion weights must be > 0 in {self._library_file}")

        if self._normalize_weights:
            weight_sum = torch.sum(self._weights)
            if weight_sum <= 0:
                raise ValueError(f"Motion weights must sum to > 0 in {self._library_file}")
            self._weights = self._weights / weight_sum

        # Build global frame offsets and concatenated tensors. This avoids per-step Python loops over
        # unique motion ids in fetch_frame_batch/fetch_future_joint_batch.
        self._frame_offsets = torch.zeros(len(self._meta), dtype=torch.long, device=self._device)
        if len(self._meta) > 1:
            self._frame_offsets[1:] = torch.cumsum(self._num_frames[:-1], dim=0)

        self._joint_pos_cat = torch.cat(self._joint_pos_chunks, dim=0)
        self._joint_vel_cat = torch.cat(self._joint_vel_chunks, dim=0)
        self._body_pos_w_cat = torch.cat(self._body_pos_w_chunks, dim=0)
        self._body_quat_w_cat = torch.cat(self._body_quat_w_chunks, dim=0)
        self._body_lin_vel_w_cat = torch.cat(self._body_lin_vel_w_chunks, dim=0)
        self._body_ang_vel_w_cat = torch.cat(self._body_ang_vel_w_chunks, dim=0)

        # Release temporary lists to limit memory overhead after build.
        self._joint_pos_chunks.clear()
        self._joint_vel_chunks.clear()
        self._body_pos_w_chunks.clear()
        self._body_quat_w_chunks.clear()
        self._body_lin_vel_w_chunks.clear()
        self._body_ang_vel_w_chunks.clear()

        print(
            f"[INFO] Motion library loaded: {len(self._meta)} motions, "
            f"dof_dim={first_joint_dim}, body_dim={first_body_dim}",
            flush=True,
        )

    def num_motions(self) -> int:
        return len(self._meta)

    def get_motion_meta(self, motion_id: int) -> MotionMeta:
        return self._meta[motion_id]

    def list_motion_meta(self) -> list[MotionMeta]:
        return self._meta

    def sample_motion_ids(self, n: int, device: str | torch.device, strategy: str = "weighted") -> torch.Tensor:
        strategy = strategy.lower()
        if strategy == "uniform":
            probs = torch.ones_like(self._weights) / float(self.num_motions())
        elif strategy == "weighted":
            if self._normalize_weights:
                probs = self._weights
            else:
                probs = self._weights / torch.sum(self._weights)
        else:
            raise ValueError(f"Unsupported motion_id sampling strategy: {strategy}")
        motion_ids = torch.multinomial(probs, num_samples=n, replacement=True)
        return motion_ids.to(device=device)

    def sample_start_frames(self, motion_ids: torch.Tensor, random_start: bool = True) -> torch.Tensor:
        if not random_start:
            return torch.zeros_like(motion_ids, dtype=torch.long)

        lengths = self._num_frames[motion_ids]
        frame_ids = torch.floor(torch.rand_like(motion_ids, dtype=torch.float32) * lengths.float()).long()
        return torch.minimum(frame_ids, lengths - 1)

    def fetch_frame_batch(self, motion_ids: torch.Tensor, frame_ids: torch.Tensor) -> MotionFrameBatch:
        if motion_ids.shape != frame_ids.shape:
            raise ValueError(f"motion_ids and frame_ids must have same shape, got {motion_ids.shape} vs {frame_ids.shape}")

        batch_size = int(motion_ids.shape[0])
        if batch_size == 0:
            raise ValueError("fetch_frame_batch requires non-empty batch")

        if (
            self._frame_offsets is None
            or self._joint_pos_cat is None
            or self._joint_vel_cat is None
            or self._body_pos_w_cat is None
            or self._body_quat_w_cat is None
            or self._body_lin_vel_w_cat is None
            or self._body_ang_vel_w_cat is None
        ):
            raise RuntimeError("Motion library tensors are not initialized.")

        # frame_ids are per-motion local frame indices.
        max_frames = self._num_frames[motion_ids] - 1
        local_frames = torch.minimum(torch.clamp(frame_ids.long(), min=0), max_frames)
        global_frames = self._frame_offsets[motion_ids] + local_frames

        return MotionFrameBatch(
            joint_pos=self._joint_pos_cat[global_frames],
            joint_vel=self._joint_vel_cat[global_frames],
            body_pos_w=self._body_pos_w_cat[global_frames],
            body_quat_w=self._body_quat_w_cat[global_frames],
            body_lin_vel_w=self._body_lin_vel_w_cat[global_frames],
            body_ang_vel_w=self._body_ang_vel_w_cat[global_frames],
        )

    def fetch_future_joint_batch(
        self, motion_ids: torch.Tensor, frame_ids_2d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if motion_ids.ndim != 1:
            raise ValueError(f"motion_ids must be 1D [B], got shape {motion_ids.shape}")
        if frame_ids_2d.ndim != 2:
            raise ValueError(f"frame_ids_2d must be 2D [B, K], got shape {frame_ids_2d.shape}")
        if motion_ids.shape[0] != frame_ids_2d.shape[0]:
            raise ValueError(
                f"motion_ids batch and frame_ids_2d batch mismatch: {motion_ids.shape[0]} vs {frame_ids_2d.shape[0]}"
            )

        if self._frame_offsets is None or self._joint_pos_cat is None or self._joint_vel_cat is None:
            raise RuntimeError("Motion library tensors are not initialized.")

        batch_size, future_count = frame_ids_2d.shape
        dof_dim = int(self._joint_pos_cat.shape[1])
        max_frames = self._num_frames[motion_ids].unsqueeze(1) - 1
        local_frames = torch.minimum(torch.clamp(frame_ids_2d.long(), min=0), max_frames)
        global_frames = self._frame_offsets[motion_ids].unsqueeze(1) + local_frames
        flat_frames = global_frames.reshape(-1)

        joint_pos = self._joint_pos_cat[flat_frames].reshape(batch_size, future_count, dof_dim)
        joint_vel = self._joint_vel_cat[flat_frames].reshape(batch_size, future_count, dof_dim)
        return joint_pos, joint_vel


def create_motion_source(
    *,
    mode: str,
    motion_file: str | None,
    single_file: str | None,
    library_file: str | None,
    root_dir: str | None,
    normalize_weights: bool,
    default_weight: float,
    body_indexes: Sequence[int] | torch.Tensor,
    body_names: Sequence[str] | None = None,
    device: str | torch.device,
    target_fps: float | None = None,
) -> MotionSource:
    mode = mode.lower().strip()
    resolved_single = _resolve_path(single_file)
    resolved_library = _resolve_path(library_file)
    resolved_legacy = _resolve_path(motion_file)

    if mode == "single_npz":
        if resolved_library is not None:
            raise ValueError("single_npz mode does not accept `library_file`.")
        motion_path = resolved_single if resolved_single is not None else resolved_legacy
        if motion_path is None:
            raise ValueError("single_npz mode requires `single_file` or legacy `motion_file`.")
        return SingleNpzMotionSource(
            motion_path,
            body_indexes=body_indexes,
            body_names=body_names,
            device=device,
            target_fps=target_fps,
        )

    if mode == "yaml_npz_library":
        if resolved_single is not None:
            raise ValueError("yaml_npz_library mode does not accept `single_file`.")
        if resolved_library is None:
            raise ValueError("yaml_npz_library mode requires `library_file`.")
        return YamlNpzMotionLibrarySource(
            resolved_library,
            body_indexes=body_indexes,
            device=device,
            body_names=body_names,
            root_dir=root_dir,
            normalize_weights=normalize_weights,
            default_weight=default_weight,
            target_fps=target_fps,
        )

    raise ValueError(f"Unsupported motion_source.mode: {mode}. Supported modes: single_npz, yaml_npz_library")
