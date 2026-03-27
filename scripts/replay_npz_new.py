"""Replay a motion npz on G1, from W&B registry or local file.

Examples:

    # from W&B registry
    CUDA_VISIBLE_DEVICES=1 python scripts/replay_npz_new.py \
        --registry_name my-org/wandb-registry-motions/my_motion \
        --headless --livestream 2

    # from local npz
    CUDA_VISIBLE_DEVICES=1 python scripts/replay_npz_new.py \
        --motion_file /path/to/motion.npz \
        --headless --livestream 2
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay converted motions.")
    parser.add_argument(
        "--registry_name",
        type=str,
        default=None,
        help="W&B registry artifact path, e.g. <org>/wandb-registry-motions/<motion_name>[:alias].",
    )
    parser.add_argument("--motion_file", type=str, default=None, help="Optional local motion npz path.")
    parser.add_argument(
        "--artifact_file",
        type=str,
        default="motion.npz",
        help="Preferred filename inside downloaded artifact directory.",
    )
    parser.add_argument(
        "--anchor_body_name",
        type=str,
        default="pelvis",
        help="Body name used as root pose source when replaying.",
    )
    parser.add_argument(
        "--disable_camera_follow",
        action="store_true",
        help="Disable camera follow mode.",
    )

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    if args.motion_file is None and args.registry_name is None:
        parser.error("Either --motion_file or --registry_name must be provided.")
    return args


args_cli = parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader


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


REQUIRED_KEYS = ("fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def pick_npz_from_artifact(download_dir: Path, preferred_name: str) -> Path:
    preferred = download_dir / preferred_name
    if preferred.is_file():
        return preferred

    npz_files = sorted(download_dir.rglob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz file found under downloaded artifact directory: {download_dir}")
    if len(npz_files) > 1:
        print(f"[WARN] Found multiple npz files under artifact dir, using the first one: {npz_files[0]}")
    return npz_files[0]


def resolve_motion_file() -> Path:
    if args_cli.motion_file is not None:
        path = Path(args_cli.motion_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--motion_file does not exist: {path}")
        return path

    registry_name = args_cli.registry_name
    assert registry_name is not None
    if ":" not in registry_name:
        registry_name += ":latest"

    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    download_dir = Path(artifact.download())
    motion_file = pick_npz_from_artifact(download_dir, args_cli.artifact_file)
    return motion_file.resolve()


def inspect_motion_npz(motion_file: Path) -> None:
    data = np.load(motion_file)
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise KeyError(f"Missing required keys in {motion_file}: {missing}")

    frames = int(data["joint_pos"].shape[0])
    if frames <= 1:
        print(f"[WARN] Motion has only {frames} frame(s). It will look static.")
        return

    body_pos = data["body_pos_w"]
    joint_pos = data["joint_pos"]
    root_span = float(np.linalg.norm(body_pos[:, 0, :].max(axis=0) - body_pos[:, 0, :].min(axis=0)))
    joint_std_mean = float(joint_pos.std(axis=0).mean())
    print(f"[INFO] Motion file: {motion_file}")
    print(f"[INFO] Motion frames: {frames}, fps: {int(np.array(data['fps']).reshape(-1)[0])}")
    print(f"[INFO] Root displacement span (body[0]): {root_span:.6f}")
    print(f"[INFO] Mean joint std: {joint_std_mean:.6f}")
    if root_span < 1e-3 and joint_std_mean < 1e-3:
        print("[WARN] Motion variation is extremely small. This motion is effectively static.")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()

    motion_file = resolve_motion_file()
    inspect_motion_npz(motion_file)

    try:
        anchor_body_index = robot.body_names.index(args_cli.anchor_body_name)
    except ValueError:
        anchor_body_index = 0
        print(
            f"[WARN] anchor body '{args_cli.anchor_body_name}' not found in robot.body_names. "
            f"Fallback to index 0 ({robot.body_names[0]})."
        )
    anchor_body_index_tensor = torch.tensor([anchor_body_index], dtype=torch.long, device=sim.device)

    motion = MotionLoader(str(motion_file), anchor_body_index_tensor, sim.device)
    print(f"[INFO] Motion time_step_total: {motion.time_step_total}")

    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0, :] + scene.env_origins[:, :]
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0, :]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0, :]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0, :]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        scene.write_data_to_sim()
        sim.render()
        scene.update(sim_dt)

        if not args_cli.disable_camera_follow:
            pos_lookat = root_states[0, :3].detach().cpu().numpy()
            sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
