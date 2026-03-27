"""Batch-convert G1 motion pickles into loader-compatible motion npz files.

This follows the same conversion path as ``scripts/csv_to_npz.py``:
1. Read root pose + joint positions from the source motion.
2. Interpolate to the requested output FPS and estimate velocities.
3. Write those states into the G1 articulation.
4. Read back the full articulation body/joint state and save it as ``.npz``.

Example:

    python scripts/batch_pkl_to_npz.py \
        --input_root /media/raid/workspace/huangyuming/TWIST2/TWIST2_full \
        --output_root /media/raid/workspace/huangyuming/TWIST2/TWIST2_full_npz \
        --input_glob '*/*.pkl' \
        --output_fps 50 \
        --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import pickle
import sys
import traceback
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

try:
    from tqdm.auto import tqdm
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, total=None, **kwargs):
            self.iterable = iterable
            self.total = total

        def __iter__(self):
            if self.iterable is None:
                return iter(range(self.total or 0))
            return iter(self.iterable)

        def set_postfix(self, *args, **kwargs):
            pass

        def write(self, message: str):
            print(message)


parser = argparse.ArgumentParser(description="Batch-convert motion pkl files into motion npz files.")
parser.add_argument(
    "--input_root",
    type=str,
    default="/media/raid/workspace/huangyuming/TWIST2/TWIST2_full",
    help="Root directory containing source pkl files.",
)
parser.add_argument(
    "--output_root",
    type=str,
    default="/media/raid/workspace/huangyuming/TWIST2/TWIST2_full_npz",
    help="Root directory where converted npz files will be written.",
)
parser.add_argument(
    "--input_glob",
    type=str,
    default="*/*.pkl",
    help="Glob pattern relative to input_root used to discover motion files.",
)
parser.add_argument(
    "--output_fps",
    type=int,
    default=50,
    help="FPS of the output motion npz files.",
)
parser.add_argument("--force", action="store_true", help="Overwrite existing output npz files.")
parser.add_argument("--max_files", type=int, default=None, help="Optional limit on number of files to convert.")
parser.add_argument(
    "--continue_on_error",
    action="store_true",
    help="Log errors and continue converting the remaining files.",
)
parser.add_argument(
    "--compressed",
    action="store_true",
    help="Save with np.savez_compressed instead of np.savez.",
)
parser.add_argument(
    "--allow_multi_gpu",
    action="store_true",
    help="Do not force Isaac Sim renderer multi-GPU settings to single-GPU mode.",
)
parser.add_argument(
    "--tmps_dir",
    type=str,
    default="tmps",
    help="Directory under the current repo used to store run-time intermediate logs and manifests.",
)
parser.add_argument(
    "--robot_urdf",
    type=str,
    default=None,
    help=(
        "Optional explicit URDF path for the G1 robot. "
        "If not set, the script will try the default project path and a few common local fallback locations."
    ),
)
parser.add_argument(
    "--isolate_visible_gpus",
    action="store_true",
    help=(
        "Strictly isolate to one physical GPU by setting CUDA_VISIBLE_DEVICES to the selected --device index "
        "before launching Isaac Sim. This remaps runtime device index to cuda:0."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def append_kit_args(existing: str | None, extra: str) -> str:
    if existing is None or not existing.strip():
        return extra
    return f"{existing} {extra}"


RUN_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
TMPS_ROOT = Path(args_cli.tmps_dir).expanduser().resolve()
RUN_DIR = TMPS_ROOT / f"batch_pkl_to_npz_{RUN_TIMESTAMP}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG_PATH = RUN_DIR / "run.log"
RUN_MANIFEST_PATH = RUN_DIR / "manifest.tsv"
RUN_SUMMARY_PATH = RUN_DIR / "summary.txt"
RUN_CONFIG_PATH = RUN_DIR / "config.txt"


def append_text(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(text)


def emit(message: str, progress: tqdm | None = None) -> None:
    if progress is not None:
        progress.write(message)
    else:
        print(message)
    append_text(RUN_LOG_PATH, f"{message}\n")


def write_config_line(key: str, value: str) -> None:
    append_text(RUN_CONFIG_PATH, f"{key}: {value}\n")


def record_manifest(status: str, input_path: Path | str, output_path: Path | str, note: str = "") -> None:
    safe_note = note.replace("\n", " ").replace("\t", " ")
    append_text(RUN_MANIFEST_PATH, f"{status}\t{input_path}\t{output_path}\t{safe_note}\n")


def parse_cuda_device_index(device: str) -> int | None:
    if not device.startswith("cuda"):
        return None
    if ":" not in device:
        return 0
    return int(device.split(":")[-1])


selected_device_original = str(getattr(args_cli, "device", "cuda:0"))
selected_device = selected_device_original
selected_device_id: int | None = None
selected_device_id = parse_cuda_device_index(selected_device)

if args_cli.isolate_visible_gpus:
    if selected_device_id is None:
        raise ValueError("--isolate_visible_gpus requires a CUDA device, for example: --device cuda:1")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_device_id)
    selected_device = "cuda:0"
    selected_device_id = 0
    args_cli.device = selected_device

if selected_device.startswith("cuda") and not args_cli.allow_multi_gpu:
    single_gpu_kit_args = (
        "--/renderer/multiGpu/enabled=false "
        "--/renderer/multiGpu/autoEnable=false "
        "--/renderer/multiGpu/maxGpuCount=1"
    )
    existing_kit_args = getattr(args_cli, "kit_args", None)
    if not existing_kit_args or "--/renderer/multiGpu/" not in existing_kit_args:
        args_cli.kit_args = append_kit_args(existing_kit_args, single_gpu_kit_args)
if selected_device_id is not None:
    existing_kit_args = getattr(args_cli, "kit_args", None)
    explicit_gpu_binding_kit_args = (
        f"--/renderer/activeGpu={selected_device_id} "
        f"--/physics/cudaDevice={selected_device_id}"
    )
    if not existing_kit_args or (
        "--/renderer/activeGpu=" not in existing_kit_args and "--/physics/cudaDevice=" not in existing_kit_args
    ):
        args_cli.kit_args = append_kit_args(existing_kit_args, explicit_gpu_binding_kit_args)

append_text(RUN_MANIFEST_PATH, "status\tinput_path\toutput_path\tnote\n")
write_config_line("run_timestamp_utc", RUN_TIMESTAMP)
write_config_line("cwd", str(Path.cwd()))
write_config_line("pid", str(os.getpid()))
write_config_line("tmps_root", str(TMPS_ROOT))
write_config_line("run_dir", str(RUN_DIR))
write_config_line("argv", " ".join(sys.argv))
write_config_line("requested_device_original", selected_device_original)
write_config_line("requested_device_effective", selected_device)
write_config_line("isolate_visible_gpus", str(bool(args_cli.isolate_visible_gpus)))
write_config_line("effective_kit_args", str(getattr(args_cli, "kit_args", "")))
write_config_line("cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
write_config_line("ld_library_path", os.environ.get("LD_LIBRARY_PATH", ""))
write_config_line("vk_icd_filenames", os.environ.get("VK_ICD_FILENAMES", ""))

if os.environ.get("CUDA_VISIBLE_DEVICES"):
    if args_cli.isolate_visible_gpus:
        emit(
            "[INFO] GPU isolation enabled via CUDA_VISIBLE_DEVICES. "
            "Effective runtime GPU index is remapped to cuda:0."
        )
    else:
        emit(
            "[WARN] Detected CUDA_VISIBLE_DEVICES. Omniverse warns this can cause device-enumeration issues. "
            "Prefer passing --device cuda:N directly."
        )
emit(f"[INFO] Intermediate run artifacts will be written under {RUN_DIR}")
emit(f"[INFO] Process PID: {os.getpid()}")
emit(f"[INFO] Requested Isaac device (original): {selected_device_original}")
emit(f"[INFO] Requested Isaac device (effective): {selected_device}")
if getattr(args_cli, "kit_args", None):
    emit(f"[INFO] Effective kit args: {args_cli.kit_args}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

kit_log_dir = Path(sys.prefix) / "lib/python3.11/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim/5.1"
if kit_log_dir.is_dir():
    kit_logs = sorted(kit_log_dir.glob("kit_*.log"), key=lambda path: path.stat().st_mtime)
    if kit_logs:
        write_config_line("latest_kit_log", str(kit_logs[-1]))
        emit(f"[INFO] Latest Kit log: {kit_logs[-1]}")

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG


G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

REQUIRED_KEYS = {"fps", "root_pos", "root_rot", "dof_pos"}


def install_numpy_pickle_compat() -> None:
    """Register numpy compatibility aliases required by some legacy pickle files."""

    import numpy.core.multiarray as multiarray
    import numpy.core.numeric as numeric

    numpy_core_module = types.ModuleType("numpy._core")
    numpy_core_module.multiarray = multiarray
    numpy_core_module.numeric = numeric

    sys.modules.setdefault("numpy._core", numpy_core_module)
    sys.modules.setdefault("numpy._core.multiarray", multiarray)
    sys.modules.setdefault("numpy._core.numeric", numeric)


def load_pickle_motion(path: Path) -> dict:
    install_numpy_pickle_compat()
    with path.open("rb") as file:
        data = pickle.load(file)

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict motion payload in {path}, got {type(data)!r}")

    missing_keys = REQUIRED_KEYS.difference(data.keys())
    if missing_keys:
        raise KeyError(f"Missing required keys in {path}: {sorted(missing_keys)}")

    return data


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class PklMotionLoader:
    """Load a source pkl motion and convert it into root/joint trajectories."""

    def __init__(self, motion_file: Path, output_fps: int, device: torch.device):
        self.motion_file = motion_file
        self.output_fps = output_fps
        self.output_dt = 1.0 / float(output_fps)
        self.device = device

        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self) -> None:
        data = load_pickle_motion(self.motion_file)

        self.input_fps = int(data["fps"])
        if self.input_fps <= 0:
            raise ValueError(f"Invalid input fps={self.input_fps} in {self.motion_file}")
        self.input_dt = 1.0 / float(self.input_fps)

        motion_base_pos = np.asarray(data["root_pos"], dtype=np.float32)
        motion_base_rot = np.asarray(data["root_rot"], dtype=np.float32)
        motion_dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)

        if motion_base_pos.ndim != 2 or motion_base_pos.shape[1] != 3:
            raise ValueError(f"Expected root_pos with shape (T, 3) in {self.motion_file}, got {motion_base_pos.shape}")
        if motion_base_rot.ndim != 2 or motion_base_rot.shape[1] != 4:
            raise ValueError(f"Expected root_rot with shape (T, 4) in {self.motion_file}, got {motion_base_rot.shape}")
        if motion_dof_pos.ndim != 2 or motion_dof_pos.shape[1] != len(G1_JOINT_NAMES):
            raise ValueError(
                f"Expected dof_pos with shape (T, {len(G1_JOINT_NAMES)}) in {self.motion_file}, got {motion_dof_pos.shape}"
            )
        if not (motion_base_pos.shape[0] == motion_base_rot.shape[0] == motion_dof_pos.shape[0]):
            raise ValueError(
                "root_pos, root_rot and dof_pos must share the same frame count in "
                f"{self.motion_file}: {motion_base_pos.shape[0]}, {motion_base_rot.shape[0]}, {motion_dof_pos.shape[0]}"
            )
        if motion_base_pos.shape[0] < 2:
            raise ValueError(f"Need at least 2 frames to estimate velocities in {self.motion_file}")

        self.motion_base_poss_input = torch.from_numpy(motion_base_pos).to(self.device)
        self.motion_base_rots_input = torch.from_numpy(motion_base_rot[:, [3, 0, 1, 2]]).to(self.device)
        self.motion_dof_poss_input = torch.from_numpy(motion_dof_pos).to(self.device)

        self.input_frames = motion_base_pos.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt

    def _interpolate_motion(self) -> None:
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        if times.numel() == 0:
            times = torch.zeros(1, device=self.device, dtype=torch.float32)
        self.output_frames = int(times.shape[0])

        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )

    def _compute_frame_blend(self, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        max_index = torch.full_like(index_0, self.input_frames - 1)
        index_1 = torch.minimum(index_0 + 1, max_index)
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1.0 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        slerped_quats = torch.zeros_like(a)
        for index in range(a.shape[0]):
            slerped_quats[index] = quat_slerp(a[index], b[index], blend[index])
        return slerped_quats

    def _compute_velocities(self) -> None:
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        if rotations.shape[0] == 1:
            return torch.zeros((1, 3), dtype=rotations.dtype, device=rotations.device)
        if rotations.shape[0] == 2:
            q_rel = quat_mul(rotations[1:2], quat_conjugate(rotations[0:1]))
            omega = axis_angle_from_quat(q_rel) / dt
            return omega.repeat(2, 1)

        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega


def discover_input_files(input_root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in input_root.glob(pattern) if path.is_file())


def resolve_robot_urdf_path(explicit_urdf: str | None, default_urdf: str) -> Path:
    def normalize(path_str: str) -> Path:
        return Path(path_str).expanduser().resolve()

    if explicit_urdf is not None:
        explicit_path = normalize(explicit_urdf)
        if not explicit_path.is_file():
            raise FileNotFoundError(f"--robot_urdf path does not exist: {explicit_path}")
        return explicit_path

    default_path = normalize(default_urdf)
    if default_path.is_file():
        return default_path

    repo_root = Path(__file__).resolve().parent.parent
    env_override = os.environ.get("WHOLE_BODY_TRACKING_G1_URDF", "").strip()
    candidate_paths: list[Path] = []
    if env_override:
        candidate_paths.append(normalize(env_override))
    candidate_paths.extend(
        [
            (repo_root / "source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf").resolve(),
            (repo_root.parent / "TWIST2_lzd/assets/g1/g1_custom_collision_29dof.urdf").resolve(),
            (repo_root.parent / "TWIST2/assets/g1/g1_custom_collision_29dof.urdf").resolve(),
            Path("/media/raid/workspace/huangyuming/lzd/TWIST2_lzd/assets/g1/g1_custom_collision_29dof.urdf"),
            Path("/media/raid/workspace/huangyuming/TWIST2/assets/g1/g1_custom_collision_29dof.urdf"),
            Path("/media/raid/workspace/huangyuming/unitree_rl_gym/resources/robots/g1_description/g1_29dof_rev_1_0.urdf"),
        ]
    )

    tried_paths: list[Path] = [default_path]
    for candidate in candidate_paths:
        resolved = candidate.expanduser().resolve()
        if resolved in tried_paths:
            continue
        tried_paths.append(resolved)
        if resolved.is_file():
            return resolved

    tried_text = "\n".join(f"  - {path}" for path in tried_paths)
    raise FileNotFoundError(
        "Could not resolve a valid G1 URDF file.\n"
        "Please provide --robot_urdf explicitly or set WHOLE_BODY_TRACKING_G1_URDF.\n"
        "Tried paths:\n"
        f"{tried_text}"
    )


def build_output_path(input_root: Path, output_root: Path, input_path: Path) -> Path:
    relative_path = input_path.relative_to(input_root)
    return output_root / relative_path.with_suffix(".npz")


def save_log_npz(output_path: Path, log: dict[str, np.ndarray], compressed: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(output_path, **log)
    else:
        np.savez(output_path, **log)


def convert_motion_file(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    robot_joint_indexes: torch.Tensor,
    input_path: Path,
    output_path: Path,
) -> None:
    motion = PklMotionLoader(input_path, output_fps=args_cli.output_fps, device=sim.device)

    log: dict[str, list[np.ndarray] | np.ndarray] = {
        "fps": np.array([args_cli.output_fps], dtype=np.int32),
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    for frame_index in range(motion.output_frames):
        motion_base_pos = motion.motion_base_poss[frame_index : frame_index + 1]
        motion_base_rot = motion.motion_base_rots[frame_index : frame_index + 1]
        motion_base_lin_vel = motion.motion_base_lin_vels[frame_index : frame_index + 1]
        motion_base_ang_vel = motion.motion_base_ang_vels[frame_index : frame_index + 1]
        motion_dof_pos = motion.motion_dof_poss[frame_index : frame_index + 1]
        motion_dof_vel = motion.motion_dof_vels[frame_index : frame_index + 1]

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()
        scene.update(sim.get_physics_dt())

        log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

    stacked_log = {"fps": log["fps"]}
    for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        stacked_log[key] = np.stack(log[key], axis=0)

    save_log_npz(output_path, stacked_log, compressed=args_cli.compressed)


def main() -> None:
    input_root = Path(args_cli.input_root).expanduser().resolve()
    output_root = Path(args_cli.output_root).expanduser().resolve()

    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root does not exist: {input_root}")

    input_files = discover_input_files(input_root, args_cli.input_glob)
    if args_cli.max_files is not None:
        input_files = input_files[: args_cli.max_files]
    if not input_files:
        raise FileNotFoundError(f"No pkl files found under {input_root} with pattern {args_cli.input_glob!r}")

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / float(args_cli.output_fps)
    sim = SimulationContext(sim_cfg)

    robot_urdf_path = resolve_robot_urdf_path(args_cli.robot_urdf, G1_CYLINDER_CFG.spawn.asset_path)
    write_config_line("resolved_robot_urdf", str(robot_urdf_path))
    emit(f"[INFO] Resolved robot URDF: {robot_urdf_path}")

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot.spawn.asset_path = str(robot_urdf_path)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot: Articulation = scene["robot"]
    robot_joint_indexes = robot.find_joints(G1_JOINT_NAMES, preserve_order=True)[0]

    converted_count = 0
    skipped_count = 0
    failed_count = 0

    write_config_line("input_root", str(input_root))
    write_config_line("output_root", str(output_root))
    write_config_line("input_glob", args_cli.input_glob)
    write_config_line("output_fps", str(args_cli.output_fps))
    write_config_line("input_files", str(len(input_files)))

    emit(f"[INFO] Found {len(input_files)} input pkl files under {input_root}")
    emit(f"[INFO] Writing converted npz files under {output_root}")

    progress = tqdm(input_files, total=len(input_files), desc="Converting motions", unit="file", dynamic_ncols=True)

    for file_index, input_path in enumerate(progress, start=1):
        output_path = build_output_path(input_root, output_root, input_path)
        if output_path.exists() and not args_cli.force:
            skipped_count += 1
            emit(
                f"[SKIP] ({file_index}/{len(input_files)}) {input_path} -> {output_path} already exists",
                progress=progress,
            )
            record_manifest("skipped", input_path, output_path, "already exists")
            progress.set_postfix(converted=converted_count, skipped=skipped_count, failed=failed_count)
            continue

        emit(
            f"[INFO] ({file_index}/{len(input_files)}) Converting {input_path} -> {output_path}",
            progress=progress,
        )
        try:
            convert_motion_file(sim, scene, robot, robot_joint_indexes, input_path, output_path)
            converted_count += 1
        except Exception as error:
            failed_count += 1
            emit(f"[ERROR] Failed converting {input_path}: {error}", progress=progress)
            record_manifest("failed", input_path, output_path, str(error))
            append_text(RUN_LOG_PATH, f"{traceback.format_exc()}\n")
            progress.set_postfix(converted=converted_count, skipped=skipped_count, failed=failed_count)
            if not args_cli.continue_on_error:
                raise
        else:
            record_manifest("converted", input_path, output_path)
            progress.set_postfix(converted=converted_count, skipped=skipped_count, failed=failed_count)

    summary_message = (
        "[INFO] Conversion complete: "
        f"converted={converted_count}, skipped={skipped_count}, failed={failed_count}, total={len(input_files)}"
    )
    emit(summary_message)
    append_text(
        RUN_SUMMARY_PATH,
        "\n".join(
            [
                f"run_dir={RUN_DIR}",
                f"input_root={input_root}",
                f"output_root={output_root}",
                f"total={len(input_files)}",
                f"converted={converted_count}",
                f"skipped={skipped_count}",
                f"failed={failed_count}",
            ]
        )
        + "\n",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        append_text(RUN_LOG_PATH, f"{traceback.format_exc()}\n")
        raise
    finally:
        simulation_app.close()
