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

    # Or consume a motion-library yaml:
    python scripts/batch_pkl_to_npz.py \
        --input_yaml config/twist2_dataset_debug.yaml \
        --output_root /media/raid/workspace/huangyuming/TWIST2/TWIST2_full_npz \
        --output_fps 50 \
        --cuda_visible_devices 6 \
        --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import pickle
import sys
import threading
import traceback
import types
from dataclasses import dataclass
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
    "--input_yaml",
    type=str,
    default=None,
    help=(
        "Optional motion library yaml. If set, conversion sources are resolved from yaml['motions'][*]['file'] "
        "relative to yaml['root_path'] (or root_dir/root)."
    ),
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
parser.add_argument(
    "--cuda_visible_devices",
    type=str,
    default=None,
    help=(
        "Optional explicit CUDA_VISIBLE_DEVICES value (for example: '6' or '6,7'). "
        "Applied before Isaac Sim launch. If a single visible GPU is provided, runtime device is remapped to cuda:0."
    ),
)
# Backward-compatible typo alias used in some local scripts.
parser.add_argument("--cuda_visibile_devices", dest="cuda_visible_devices", type=str, help=argparse.SUPPRESS)

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


def _resolve_yaml_root_prelaunch(yaml_path: Path, payload: dict) -> Path:
    root_value = payload.get("root_path", None)
    if root_value is None:
        root_value = payload.get("root_dir", None)
    if root_value is None:
        root_value = payload.get("root", None)
    if root_value is None:
        return yaml_path.parent.resolve()
    root_path = Path(str(root_value)).expanduser()
    if not root_path.is_absolute():
        root_path = (yaml_path.parent / root_path).resolve()
    else:
        root_path = root_path.resolve()
    return root_path


def _collect_prelaunch_items(input_root: Path, output_root: Path) -> list[tuple[Path, Path]]:
    items: list[tuple[Path, Path]] = []
    if args_cli.input_yaml is not None:
        try:
            import yaml
        except ImportError as error:
            raise ImportError("PyYAML is required for --input_yaml. Please install `pyyaml`.") from error
        input_yaml = Path(args_cli.input_yaml).expanduser().resolve()
        if not input_yaml.is_file():
            raise FileNotFoundError(f"Input yaml does not exist: {input_yaml}")
        with input_yaml.open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected mapping yaml in {input_yaml}, got {type(payload)!r}")
        motions = payload.get("motions", None)
        if not isinstance(motions, list):
            raise ValueError(f"'motions' in {input_yaml} must be a list")
        yaml_root = _resolve_yaml_root_prelaunch(input_yaml, payload)
        for index, motion_item in enumerate(motions):
            if isinstance(motion_item, str):
                raw_file = motion_item
            elif isinstance(motion_item, dict):
                raw_file = motion_item.get("file", None)
            else:
                raise TypeError(f"motions[{index}] in {input_yaml} must be a string or mapping")
            if raw_file is None:
                raise KeyError(f"Missing file in motions[{index}] of {input_yaml}")
            source_file = Path(str(raw_file)).expanduser()
            if source_file.is_absolute():
                src = source_file.resolve()
                if src.is_relative_to(yaml_root):
                    rel = src.relative_to(yaml_root)
                else:
                    rel = Path(src.name)
            else:
                rel = source_file
                src = (yaml_root / rel).resolve()
            dst = output_root / rel.with_suffix(".npz")
            items.append((src, dst))
    else:
        if not input_root.is_dir():
            raise NotADirectoryError(f"Input root does not exist: {input_root}")
        for src in sorted(path for path in input_root.glob(args_cli.input_glob) if path.is_file()):
            rel = src.relative_to(input_root)
            dst = output_root / rel.with_suffix(".npz")
            items.append((src, dst))
    return items


selected_device_original = str(getattr(args_cli, "device", "cuda:0"))
selected_device = selected_device_original
selected_device_id: int | None = None
selected_device_id = parse_cuda_device_index(selected_device)
skip_explicit_gpu_binding = False

if args_cli.cuda_visible_devices and args_cli.isolate_visible_gpus:
    raise ValueError("--cuda_visible_devices and --isolate_visible_gpus are mutually exclusive. Please use only one.")

if args_cli.cuda_visible_devices is not None:
    visible = args_cli.cuda_visible_devices.strip()
    if not visible:
        raise ValueError("--cuda_visible_devices was provided but empty.")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    visible_ids = [token.strip() for token in visible.split(",") if token.strip()]
    if len(visible_ids) == 1 and selected_device.startswith("cuda"):
        selected_device = "cuda:0"
        args_cli.device = selected_device
        selected_device_id = 0
    elif len(visible_ids) > 1:
        # Device ordinal semantics become relative to visible list.
        # Skip forcing /renderer/activeGpu to avoid binding to an invalid physical index.
        skip_explicit_gpu_binding = True

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
if selected_device_id is not None and not skip_explicit_gpu_binding:
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

precheck_input_root = Path(args_cli.input_root).expanduser().resolve()
precheck_output_root = Path(args_cli.output_root).expanduser().resolve()
precheck_items = _collect_prelaunch_items(precheck_input_root, precheck_output_root)
if args_cli.max_files is not None:
    precheck_items = precheck_items[: args_cli.max_files]
precheck_pending = [(src, dst) for (src, dst) in precheck_items if args_cli.force or (not dst.exists())]
write_config_line("precheck_total_items", str(len(precheck_items)))
write_config_line("precheck_pending_items", str(len(precheck_pending)))
if len(precheck_items) > 0 and len(precheck_pending) == 0:
    emit(
        "[INFO] Pre-check: all target npz files already exist and --force is not set. "
        "No conversion needed; exiting without launching Isaac Sim."
    )
    append_text(
        RUN_SUMMARY_PATH,
        "\n".join(
            [
                f"run_dir={RUN_DIR}",
                f"input_root={precheck_input_root}",
                f"output_root={precheck_output_root}",
                f"total={len(precheck_items)}",
                "converted=0",
                f"skipped={len(precheck_items)}",
                "failed=0",
                "note=precheck_no_pending_exit",
            ]
        )
        + "\n",
    )
    sys.exit(0)

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
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul

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


@dataclass(frozen=True)
class MotionConversionItem:
    """Source and relative path descriptor for a single conversion task."""

    input_path: Path
    relative_path: Path


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
        # Fast batched slerp path for a sequence of quaternions.
        # Inputs: a/b=(T, 4), blend=(T,).
        a = torch.nn.functional.normalize(a, p=2, dim=1)
        b = torch.nn.functional.normalize(b, p=2, dim=1)

        dot = torch.sum(a * b, dim=1)
        neg_mask = dot < 0.0
        b = torch.where(neg_mask.unsqueeze(1), -b, b)
        dot = torch.where(neg_mask, -dot, dot)
        dot = torch.clamp(dot, -1.0, 1.0)

        linear_mask = dot > 0.9995
        linear = a + blend.unsqueeze(1) * (b - a)
        linear = torch.nn.functional.normalize(linear, p=2, dim=1)

        theta_0 = torch.acos(dot)
        sin_theta_0 = torch.sin(theta_0)
        theta = theta_0 * blend
        sin_theta = torch.sin(theta)
        denom = torch.clamp(sin_theta_0, min=1e-8)
        s0 = torch.sin(theta_0 - theta) / denom
        s1 = sin_theta / denom
        spherical = s0.unsqueeze(1) * a + s1.unsqueeze(1) * b
        spherical = torch.nn.functional.normalize(spherical, p=2, dim=1)

        return torch.where(linear_mask.unsqueeze(1), linear, spherical)

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


def _resolve_yaml_root(yaml_path: Path, payload: dict) -> Path:
    root_value = payload.get("root_path", None)
    if root_value is None:
        root_value = payload.get("root_dir", None)
    if root_value is None:
        root_value = payload.get("root", None)
    if root_value is None:
        return yaml_path.parent.resolve()

    root_path = Path(str(root_value)).expanduser()
    if not root_path.is_absolute():
        root_path = (yaml_path.parent / root_path).resolve()
    else:
        root_path = root_path.resolve()
    return root_path


def load_conversion_items_from_yaml(yaml_path: Path) -> tuple[Path, list[MotionConversionItem]]:
    try:
        import yaml
    except ImportError as error:
        raise ImportError("PyYAML is required for --input_yaml. Please install `pyyaml`.") from error

    with yaml_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)

    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping yaml in {yaml_path}, got {type(payload)!r}")
    if "motions" not in payload:
        raise KeyError(f"Missing 'motions' key in {yaml_path}")
    motions = payload["motions"]
    if not isinstance(motions, list) or len(motions) == 0:
        raise ValueError(f"'motions' in {yaml_path} must be a non-empty list")

    yaml_root = _resolve_yaml_root(yaml_path, payload)
    items: list[MotionConversionItem] = []
    for index, motion_item in enumerate(motions):
        if isinstance(motion_item, str):
            raw_file = motion_item
        elif isinstance(motion_item, dict):
            raw_file = motion_item.get("file", None)
            if raw_file is None:
                raise KeyError(f"Missing 'file' in motions[{index}] of {yaml_path}")
        else:
            raise TypeError(f"motions[{index}] in {yaml_path} must be a string or mapping, got {type(motion_item)!r}")

        source_file = Path(str(raw_file)).expanduser()
        if source_file.is_absolute():
            input_path = source_file.resolve()
            if not input_path.is_relative_to(yaml_root):
                raise ValueError(
                    f"motions[{index}]={input_path} is outside yaml root {yaml_root}. "
                    "Please use files under root_path/root_dir/root for stable relative-path mirroring."
                )
            relative_path = input_path.relative_to(yaml_root)
        else:
            relative_path = source_file
            input_path = (yaml_root / source_file).resolve()

        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"motions[{index}] has invalid relative path {relative_path}. "
                "Please use a normalized relative path under root_path."
            )

        if input_path.suffix.lower() not in {".pkl", ".pickle"}:
            raise ValueError(
                f"motions[{index}]={input_path} is not a .pkl/.pickle file. "
                "This converter expects pickle motions as input."
            )

        items.append(MotionConversionItem(input_path=input_path, relative_path=relative_path.with_suffix(".npz")))

    return yaml_root, items


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


def build_output_path(output_root: Path, relative_path: Path) -> Path:
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

    num_frames = motion.output_frames
    joint_dim = int(robot.data.joint_pos.shape[1])
    body_dim = int(robot.data.body_pos_w.shape[1])

    joint_pos_log = torch.empty((num_frames, joint_dim), device=sim.device, dtype=robot.data.joint_pos.dtype)
    joint_vel_log = torch.empty((num_frames, joint_dim), device=sim.device, dtype=robot.data.joint_vel.dtype)
    body_pos_log = torch.empty((num_frames, body_dim, 3), device=sim.device, dtype=robot.data.body_pos_w.dtype)
    body_quat_log = torch.empty((num_frames, body_dim, 4), device=sim.device, dtype=robot.data.body_quat_w.dtype)
    body_lin_vel_log = torch.empty((num_frames, body_dim, 3), device=sim.device, dtype=robot.data.body_lin_vel_w.dtype)
    body_ang_vel_log = torch.empty((num_frames, body_dim, 3), device=sim.device, dtype=robot.data.body_ang_vel_w.dtype)

    default_root_state = robot.data.default_root_state.clone()
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    root_states = default_root_state.clone()
    joint_pos = default_joint_pos.clone()
    joint_vel = default_joint_vel.clone()

    with torch.inference_mode():
        for frame_index in range(num_frames):
            root_states.copy_(default_root_state)
            joint_pos.copy_(default_joint_pos)
            joint_vel.copy_(default_joint_vel)

            root_states[:, :3] = motion.motion_base_poss[frame_index : frame_index + 1]
            root_states[:, :2] += scene.env_origins[:, :2]
            root_states[:, 3:7] = motion.motion_base_rots[frame_index : frame_index + 1]
            root_states[:, 7:10] = motion.motion_base_lin_vels[frame_index : frame_index + 1]
            root_states[:, 10:] = motion.motion_base_ang_vels[frame_index : frame_index + 1]
            robot.write_root_state_to_sim(root_states)

            joint_pos[:, robot_joint_indexes] = motion.motion_dof_poss[frame_index : frame_index + 1]
            joint_vel[:, robot_joint_indexes] = motion.motion_dof_vels[frame_index : frame_index + 1]
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            sim.render()
            scene.update(sim.get_physics_dt())

            joint_pos_log[frame_index].copy_(robot.data.joint_pos[0])
            joint_vel_log[frame_index].copy_(robot.data.joint_vel[0])
            body_pos_log[frame_index].copy_(robot.data.body_pos_w[0])
            body_quat_log[frame_index].copy_(robot.data.body_quat_w[0])
            body_lin_vel_log[frame_index].copy_(robot.data.body_lin_vel_w[0])
            body_ang_vel_log[frame_index].copy_(robot.data.body_ang_vel_w[0])

    stacked_log = {
        "fps": np.array([args_cli.output_fps], dtype=np.int32),
        "joint_pos": joint_pos_log.cpu().numpy(),
        "joint_vel": joint_vel_log.cpu().numpy(),
        "body_pos_w": body_pos_log.cpu().numpy(),
        "body_quat_w": body_quat_log.cpu().numpy(),
        "body_lin_vel_w": body_lin_vel_log.cpu().numpy(),
        "body_ang_vel_w": body_ang_vel_log.cpu().numpy(),
    }

    save_log_npz(output_path, stacked_log, compressed=args_cli.compressed)


def main() -> None:
    output_root = Path(args_cli.output_root).expanduser().resolve()
    input_root: Path | None = None
    input_yaml: Path | None = None

    conversion_items: list[MotionConversionItem] = []
    if args_cli.input_yaml is not None:
        input_yaml = Path(args_cli.input_yaml).expanduser().resolve()
        if not input_yaml.is_file():
            raise FileNotFoundError(f"Input yaml does not exist: {input_yaml}")
        input_root, conversion_items = load_conversion_items_from_yaml(input_yaml)
    else:
        input_root = Path(args_cli.input_root).expanduser().resolve()
        if not input_root.is_dir():
            raise NotADirectoryError(f"Input root does not exist: {input_root}")
        input_files = discover_input_files(input_root, args_cli.input_glob)
        conversion_items = [
            MotionConversionItem(input_path=path, relative_path=path.relative_to(input_root).with_suffix(".npz"))
            for path in input_files
        ]

    if args_cli.max_files is not None:
        conversion_items = conversion_items[: args_cli.max_files]
    if not conversion_items:
        if input_yaml is not None:
            raise FileNotFoundError(f"No valid pkl files resolved from yaml: {input_yaml}")
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
    write_config_line("input_yaml", str(input_yaml) if input_yaml is not None else "")
    write_config_line("output_root", str(output_root))
    write_config_line("input_glob", args_cli.input_glob)
    write_config_line("output_fps", str(args_cli.output_fps))
    write_config_line("input_files", str(len(conversion_items)))

    if input_yaml is not None:
        emit(f"[INFO] Found {len(conversion_items)} input pkl files from yaml {input_yaml}")
    else:
        emit(f"[INFO] Found {len(conversion_items)} input pkl files under {input_root}")
    emit("[INFO] Relative paths from motion file entries will be mirrored under output_root.")
    emit(f"[INFO] Writing converted npz files under {output_root}")

    progress = tqdm(
        conversion_items,
        total=len(conversion_items),
        desc="Converting motions",
        unit="file",
        dynamic_ncols=True,
    )

    for file_index, item in enumerate(progress, start=1):
        input_path = item.input_path
        output_path = build_output_path(output_root, item.relative_path)
        if not input_path.is_file():
            failed_count += 1
            emit(
                f"[ERROR] ({file_index}/{len(conversion_items)}) Missing input file: {input_path}",
                progress=progress,
            )
            record_manifest("failed", input_path, output_path, "input file missing")
            progress.set_postfix(converted=converted_count, skipped=skipped_count, failed=failed_count)
            if not args_cli.continue_on_error:
                raise FileNotFoundError(f"Input file missing: {input_path}")
            continue

        if output_path.exists() and not args_cli.force:
            skipped_count += 1
            emit(
                f"[SKIP] ({file_index}/{len(conversion_items)}) {input_path} -> {output_path} already exists",
                progress=progress,
            )
            record_manifest("skipped", input_path, output_path, "already exists")
            progress.set_postfix(converted=converted_count, skipped=skipped_count, failed=failed_count)
            continue

        emit(
            f"[INFO] ({file_index}/{len(conversion_items)}) Converting {input_path} -> {output_path}",
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
        f"converted={converted_count}, skipped={skipped_count}, failed={failed_count}, total={len(conversion_items)}"
    )
    emit(summary_message)
    append_text(
        RUN_SUMMARY_PATH,
        "\n".join(
            [
                f"run_dir={RUN_DIR}",
                f"input_root={input_root}",
                f"input_yaml={input_yaml}",
                f"output_root={output_root}",
                f"total={len(conversion_items)}",
                f"converted={converted_count}",
                f"skipped={skipped_count}",
                f"failed={failed_count}",
            ]
        )
        + "\n",
    )


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        append_text(RUN_LOG_PATH, f"{traceback.format_exc()}\n")
        traceback.print_exc()
        exit_code = 1
    finally:
        app_obj = globals().get("simulation_app", None)
        if app_obj is not None:
            close_done = threading.Event()

            def _close_app_safely() -> None:
                try:
                    app_obj.close()
                finally:
                    close_done.set()

            close_thread = threading.Thread(target=_close_app_safely, name="isaac_app_close", daemon=True)
            close_thread.start()
            close_thread.join(timeout=8.0)
            if not close_done.is_set():
                append_text(
                    RUN_LOG_PATH,
                    "[WARN] simulation_app.close() timeout after 8s; forcing process exit to avoid hang.\n",
                )
                print("[WARN] simulation_app.close() timeout after 8s; forcing process exit to avoid hang.")

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
