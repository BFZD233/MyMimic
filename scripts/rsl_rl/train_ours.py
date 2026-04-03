# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL (local motion version)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib.metadata as metadata
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

DEFAULT_MOTION_ROOT = "/media/raid/workspace/huangyuming/TWIST2/TWIST2_full_npz/"

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--motion_root",
    type=str,
    default=DEFAULT_MOTION_ROOT,
    help="Root directory of local motion npz files. Used when --motion_file is not provided.",
)
parser.add_argument("--motion_file", type=str, default=None, help="Optional local motion npz file path.")
parser.add_argument(
    "--motion_library",
    type=str,
    default=None,
    help="Optional local motion library yaml path. When set, uses yaml_npz_library mode.",
)
parser.add_argument(
    "--motion_glob",
    type=str,
    default="**/*.npz",
    help="Glob pattern used under --motion_root to discover motion files.",
)
parser.add_argument(
    "--motion_index",
    type=int,
    default=0,
    help="Index into the sorted discovered motion files. Only used when --motion_file is not provided.",
)
parser.add_argument(
    "--debug_timing",
    action="store_true",
    default=False,
    help="Print first few timing probes for vec_env.get_observations() and vec_env.step() to locate startup stalls.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
_user_passed_livestream = any(arg == "--livestream" or arg.startswith("--livestream=") for arg in sys.argv[1:])
args_cli, hydra_args = parser.parse_known_args()


def _append_kit_args(existing: str | None, extra: str) -> str:
    if existing is None or existing.strip() == "":
        return extra
    return f"{existing} {extra}"


# In headless mode, long blocking tasks (e.g. large motion library load) can trigger Kit hang detector,
# which tries to open a zenity dialog and may stall in non-GUI environments.
if args_cli.headless:
    print(
        "[INFO] AppLauncher settings before adjustments: "
        f"headless={getattr(args_cli, 'headless', None)}, "
        f"livestream={getattr(args_cli, 'livestream', None)}, "
        f"LIVESTREAM_env={os.environ.get('LIVESTREAM', '<unset>')}",
        flush=True,
    )
    # Avoid accidentally enabling livestream via environment variables in training jobs.
    # In practice, this can add extra rendering/network extensions and destabilize headless runs.
    if not _user_passed_livestream and getattr(args_cli, "livestream", -1) != 0:
        args_cli.livestream = 0
        print("[INFO] Headless training: force livestream=0.")

    existing_kit_args = getattr(args_cli, "kit_args", None)
    if not existing_kit_args or "--/app/hangDetector/" not in existing_kit_args:
        headless_hang_guard = "--/app/hangDetector/enabled=false --/app/hangDetector/timeout=3600"
        args_cli.kit_args = _append_kit_args(existing_kit_args, headless_hang_guard)

    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        print(
            "[INFO] CUDA_VISIBLE_DEVICES is set. "
            "Keep IsaacSim device ordinals at default (do not force physical GPU index in kit_args).",
            flush=True,
        )
        # Under CUDA_VISIBLE_DEVICES remapping, the only safe runtime ordinal is usually cuda:0.
        # For example, CUDA_VISIBLE_DEVICES=6 means physical GPU6 becomes runtime cuda:0.
        if getattr(args_cli, "device", None) is None:
            args_cli.device = "cuda:0"
            print("[INFO] Auto-set --device cuda:0 because CUDA_VISIBLE_DEVICES is set.", flush=True)
        elif str(args_cli.device).startswith("cuda:") and str(args_cli.device) != "cuda:0":
            print(
                f"[WARNING] --device={args_cli.device} with CUDA_VISIBLE_DEVICES may cause ordinal mismatch. "
                "Recommended: --device cuda:0",
                flush=True,
            )

        # Align renderer/physics to the remapped runtime ordinal as well.
        existing_kit_args = getattr(args_cli, "kit_args", None)
        if not existing_kit_args or "--/renderer/activeGpu=" not in existing_kit_args:
            args_cli.kit_args = _append_kit_args(getattr(args_cli, "kit_args", None), "--/renderer/activeGpu=0")
        if not getattr(args_cli, "kit_args", None) or "--/physics/cudaDevice=" not in args_cli.kit_args:
            args_cli.kit_args = _append_kit_args(getattr(args_cli, "kit_args", None), "--/physics/cudaDevice=0")

    if getattr(args_cli, "kit_args", None):
        print(f"[INFO] Effective kit args: {args_cli.kit_args}")


def _parse_cuda_device_index(device: str | None) -> int | None:
    if not device:
        return None
    if not str(device).startswith("cuda"):
        return None
    if ":" not in str(device):
        return 0
    try:
        return int(str(device).split(":")[-1])
    except Exception:
        return None


def _append_cuda_binding_kit_args_if_needed() -> None:
    device_idx = _parse_cuda_device_index(getattr(args_cli, "device", None))
    if device_idx is None:
        return
    existing_kit_args = getattr(args_cli, "kit_args", None)
    if existing_kit_args and (
        "--/renderer/activeGpu=" in existing_kit_args or "--/physics/cudaDevice=" in existing_kit_args
    ):
        return
    explicit_gpu_binding_kit_args = (
        f"--/renderer/activeGpu={device_idx} "
        f"--/physics/cudaDevice={device_idx}"
    )
    args_cli.kit_args = _append_kit_args(existing_kit_args, explicit_gpu_binding_kit_args)
    print(
        f"[INFO] Auto-appended explicit GPU binding to kit args for --device cuda:{device_idx}.",
        flush=True,
    )


_append_cuda_binding_kit_args_if_needed()

# Backward-compatible override rewrite:
# The Hydra root config is {"env": ..., "agent": ...}, so env overrides should be under "env.*".
# We transparently map shorthand "obs_pipeline.*" and "commands.*" to "env.*".
def _rewrite_env_overrides(overrides: list[str]) -> list[str]:
    rewritten: list[str] = []
    for token in overrides:
        prefix = ""
        body = token
        # Preserve Hydra prefix operators (e.g. +, ++, ~).
        while body.startswith("+"):
            prefix += "+"
            body = body[1:]
        if body.startswith("~"):
            prefix += "~"
            body = body[1:]

        if body.startswith("obs_pipeline") and not body.startswith("env.obs_pipeline"):
            body = "env." + body
        if body.startswith("commands.") and not body.startswith("env.commands."):
            body = "env." + body
        rewritten.append(prefix + body)
    return rewritten

hydra_args = _rewrite_env_overrides(hydra_args)

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.tasks.tracking.obs_pipeline import apply_observation_pipeline
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def dump_pickle(filename: str, data: object) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as file:
        pickle.dump(data, file)


def resolve_local_motion_file() -> str:
    if args_cli.motion_file is not None:
        motion_file = Path(args_cli.motion_file).expanduser().resolve()
        if not motion_file.is_file():
            raise FileNotFoundError(f"--motion_file does not exist: {motion_file}")
        print(f"[INFO] Using local motion file from --motion_file: {motion_file}")
        return str(motion_file)

    motion_root = Path(args_cli.motion_root).expanduser().resolve()
    if not motion_root.is_dir():
        raise NotADirectoryError(f"--motion_root does not exist or is not a directory: {motion_root}")

    motion_files = sorted(
        path for path in motion_root.glob(args_cli.motion_glob) if path.is_file() and path.suffix.lower() == ".npz"
    )
    if not motion_files:
        raise FileNotFoundError(
            f"No .npz motion files found under --motion_root={motion_root} with --motion_glob={args_cli.motion_glob}"
        )
    if args_cli.motion_index < 0 or args_cli.motion_index >= len(motion_files):
        raise IndexError(
            f"--motion_index={args_cli.motion_index} is out of range for {len(motion_files)} motion files "
            f"(valid range: 0..{len(motion_files) - 1})"
        )

    motion_file = motion_files[args_cli.motion_index]
    print(f"[INFO] Discovered {len(motion_files)} motion files under {motion_root}")
    print(f"[INFO] Using local motion file ({args_cli.motion_index + 1}/{len(motion_files)}): {motion_file}")
    return str(motion_file)


def resolve_local_motion_library_file() -> str:
    if args_cli.motion_library is None:
        raise ValueError("resolve_local_motion_library_file requires --motion_library to be set.")
    library_file = Path(args_cli.motion_library).expanduser().resolve()
    if not library_file.is_file():
        raise FileNotFoundError(f"--motion_library does not exist: {library_file}")
    print(f"[INFO] Using motion library from --motion_library: {library_file}")
    return str(library_file)


def attach_vec_env_timing_probe(vec_env, max_calls: int = 5, step_print_every: int = 200) -> None:
    """Monkey-patch timing probes for early-stage stall diagnosis."""

    state = {"get_obs_calls": 0, "step_calls": 0}
    original_get_observations = vec_env.get_observations
    original_step = vec_env.step

    def timed_get_observations(*args, **kwargs):
        t0 = time.perf_counter()
        out = original_get_observations(*args, **kwargs)
        state["get_obs_calls"] += 1
        if state["get_obs_calls"] <= max_calls:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[INFO] Probe get_observations #{state['get_obs_calls']}: elapsed_ms={dt_ms:.2f}", flush=True)
        return out

    def timed_step(*args, **kwargs):
        t0 = time.perf_counter()
        out = original_step(*args, **kwargs)
        state["step_calls"] += 1
        should_print = state["step_calls"] <= max_calls
        if not should_print and step_print_every > 0 and state["step_calls"] % step_print_every == 0:
            should_print = True
        if should_print:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[INFO] Probe env.step #{state['step_calls']}: elapsed_ms={dt_ms:.2f}", flush=True)
        return out

    vec_env.get_observations = timed_get_observations
    vec_env.step = timed_step


def attach_runner_timing_probe(runner, max_calls: int = 5, update_print_every: int = 20) -> None:
    """Attach timing probes to learner-side stages to locate slow/hanging phase."""

    state = {"returns_calls": 0, "update_calls": 0}
    original_compute_returns = runner.alg.compute_returns
    original_update = runner.alg.update

    def timed_compute_returns(*args, **kwargs):
        t0 = time.perf_counter()
        out = original_compute_returns(*args, **kwargs)
        state["returns_calls"] += 1
        if state["returns_calls"] <= max_calls:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[INFO] Probe compute_returns #{state['returns_calls']}: elapsed_ms={dt_ms:.2f}", flush=True)
        return out

    def timed_update(*args, **kwargs):
        t0 = time.perf_counter()
        out = original_update(*args, **kwargs)
        state["update_calls"] += 1
        should_print = state["update_calls"] <= max_calls
        if not should_print and update_print_every > 0 and state["update_calls"] % update_print_every == 0:
            should_print = True
        if should_print:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[INFO] Probe alg.update #{state['update_calls']}: elapsed_ms={dt_ms:.2f}", flush=True)
        return out

    runner.alg.compute_returns = timed_compute_returns
    runner.alg.update = timed_update


def attach_post_update_timing_probe(runner, max_calls: int = 8) -> None:
    """Probe logger/save stages that happen after alg.update in each iteration."""

    state = {"log_calls": 0, "save_calls": 0}
    original_log = runner.logger.log
    original_save = runner.save

    def timed_log(*args, **kwargs):
        state["log_calls"] += 1
        call_idx = state["log_calls"]
        if call_idx <= max_calls:
            print(f"[INFO] Probe logger.log #{call_idx}: enter", flush=True)
        t0 = time.perf_counter()
        out = original_log(*args, **kwargs)
        if call_idx <= max_calls:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[INFO] Probe logger.log #{call_idx}: elapsed_ms={dt_ms:.2f}", flush=True)
        return out

    def timed_save(*args, **kwargs):
        state["save_calls"] += 1
        call_idx = state["save_calls"]
        if call_idx <= max_calls:
            print(f"[INFO] Probe runner.save #{call_idx}: enter", flush=True)
        t0 = time.perf_counter()
        out = original_save(*args, **kwargs)
        if call_idx <= max_calls:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[INFO] Probe runner.save #{call_idx}: elapsed_ms={dt_ms:.2f}", flush=True)
        return out

    runner.logger.log = timed_log
    runner.save = timed_save


def install_safe_excepthook(log_dir: str) -> None:
    """Install a robust excepthook to avoid secondary hook failures masking the original exception."""

    def _safe_hook(exc_type, exc, tb):
        crash_log = os.path.join(log_dir, "unhandled_exception.log")
        try:
            with open(crash_log, "w", encoding="utf-8") as f:
                f.write(f"exc_type={exc_type!r}\n")
                f.write(f"exc_repr={exc!r}\n\n")
                try:
                    traceback.print_exception(exc_type, exc, tb, file=f)
                except BaseException as print_err:
                    f.write(f"[safe_excepthook] traceback.print_exception failed: {print_err!r}\n")
        except BaseException:
            pass

        try:
            traceback.print_exception(exc_type, exc, tb)
        except BaseException as print_err:
            try:
                sys.__stderr__.write(
                    f"[safe_excepthook] failed to print original exception: {print_err!r}; "
                    f"original exc_type={exc_type!r}, exc_repr={exc!r}\n"
                )
            except BaseException:
                pass

    sys.excepthook = _safe_hook


def patch_wandb_exit_hooks(log_dir: str) -> None:
    """Patch wandb ExitHooks to avoid known traceback formatting crashes in excepthook."""
    try:
        from wandb.sdk.lib import exit_hooks as wandb_exit_hooks
    except Exception as err:
        print(f"[WARNING] Failed to import wandb exit hooks for patching: {err!r}", flush=True)
        return

    if getattr(wandb_exit_hooks, "_whole_body_tracking_safe_patch", False):
        return

    os.makedirs(log_dir, exist_ok=True)
    hook_log = os.path.join(log_dir, "wandb_exit_hook.log")

    def _safe_wandb_exc_handler(self, exc_type, exc, tb):
        self.exit_code = 1
        self.exception = exc
        if isinstance(exc, KeyboardInterrupt):
            self.exit_code = 255

        try:
            with open(hook_log, "a", encoding="utf-8") as f:
                f.write(f"\n===== wandb ExitHooks =====\nexc_type={exc_type!r}\nexc_repr={exc!r}\n")
                try:
                    traceback.print_exception(exc_type, exc, tb, file=f)
                except BaseException as log_err:
                    f.write(f"[wandb_exit_hooks] traceback.print_exception failed: {log_err!r}\n")
        except BaseException:
            pass

        try:
            traceback.print_exception(exc_type, exc, tb)
        except BaseException as print_err:
            try:
                sys.__stderr__.write(
                    "[wandb_exit_hooks] failed to print exception safely: "
                    f"{print_err!r}; original={exc_type!r}: {exc!r}\n"
                )
            except BaseException:
                pass

        orig = getattr(self, "_orig_excepthook", None)
        if orig:
            try:
                orig(exc_type, exc, tb)
            except BaseException as orig_err:
                try:
                    with open(hook_log, "a", encoding="utf-8") as f:
                        f.write(f"[wandb_exit_hooks] calling _orig_excepthook failed: {orig_err!r}\n")
                except BaseException:
                    pass

    def _safe_wandb_hook(self):
        self._orig_exit = sys.exit
        sys.exit = self.exit
        # Keep user/runtime excepthook untouched; wandb's default handler can crash on some exception states.
        self._orig_excepthook = None

    wandb_exit_hooks.ExitHooks.hook = _safe_wandb_hook
    wandb_exit_hooks.ExitHooks.exc_handler = _safe_wandb_exc_handler
    wandb_exit_hooks._whole_body_tracking_safe_patch = True
    print("[INFO] Patched wandb ExitHooks with safe handler.", flush=True)


def patch_rsl_rl_wandb_writer() -> None:
    """Patch rsl-rl WandbSummaryWriter to batch scalar logs and avoid deprecated wandb settings."""
    try:
        import wandb
        from rsl_rl.utils import wandb_utils as rsl_wandb_utils
    except Exception as err:
        print(f"[WARNING] Failed to patch rsl-rl wandb writer: {err!r}", flush=True)
        return

    if getattr(rsl_wandb_utils, "_whole_body_tracking_writer_patch", False):
        return

    class SafeWandbSummaryWriter(SummaryWriter):
        def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
            super().__init__(log_dir, flush_secs=flush_secs)

            run_name = os.path.split(log_dir)[-1]
            project = cfg.get("wandb_project", None)
            if project is None:
                raise KeyError("Please specify wandb_project in the runner config, e.g. legged_gym.")
            entity = os.environ.get("WANDB_USERNAME", None)

            # Avoid deprecated `start_method` argument in newer wandb versions.
            wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config={"log_dir": log_dir},
                settings=wandb.Settings(),
            )

            self.logged_videos: set[str] = set()
            self._pending_step: int | None = None
            self._pending_scalars: dict[str, float] = {}

        def _flush_pending_scalars(self) -> None:
            if not self._pending_scalars:
                return
            wandb.log(dict(self._pending_scalars), step=self._pending_step, commit=True)
            self._pending_scalars.clear()
            self._pending_step = None

        def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
            from dataclasses import asdict

            wandb.config.update({"train_cfg": train_cfg})
            try:
                wandb.config.update({"env_cfg": env_cfg.to_dict()})  # type: ignore
            except Exception:
                wandb.config.update({"env_cfg": asdict(env_cfg)})  # type: ignore

        def add_scalar(
            self,
            tag: str,
            scalar_value: float,
            global_step: int | None = None,
            walltime: float | None = None,
            new_style: bool = False,
        ) -> None:
            super().add_scalar(
                tag,
                scalar_value,
                global_step=global_step,
                walltime=walltime,
                new_style=new_style,
            )

            if global_step is None:
                wandb.log({tag: float(scalar_value)})
                return

            step = int(global_step)
            if self._pending_step is None:
                self._pending_step = step
            if step != self._pending_step:
                self._flush_pending_scalars()
                self._pending_step = step
            self._pending_scalars[tag] = float(scalar_value)

        def stop(self) -> None:
            self._flush_pending_scalars()
            wandb.finish()

        def save_model(self, model_path: str, it: int) -> None:
            self._flush_pending_scalars()
            wandb.save(model_path, base_path=os.path.dirname(model_path))

        def save_file(self, path: str) -> None:
            self._flush_pending_scalars()
            wandb.save(path, base_path=os.path.dirname(path))

        def save_video(self, video: Path, it: int) -> None:
            if video.name not in self.logged_videos:
                self._flush_pending_scalars()
                wandb.log({"video": wandb.Video(str(video), format="mp4")}, step=it)
                self.logged_videos.add(video.name)

    rsl_wandb_utils.WandbSummaryWriter = SafeWandbSummaryWriter
    rsl_wandb_utils._whole_body_tracking_writer_patch = True
    print("[INFO] Patched rsl-rl WandbSummaryWriter with batched scalar logging.", flush=True)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # Hydra overrides (e.g. env.obs_pipeline.*) are applied after cfg object construction.
    # Re-apply the observation pipeline here so overridden obs_pipeline settings truly affect
    # final policy/critic observation terms and dimensions.
    if hasattr(env_cfg, "obs_pipeline"):
        env_cfg.obs_pipeline.__post_init__()
        apply_observation_pipeline(env_cfg)
        policy_group = getattr(getattr(env_cfg, "observations", None), "policy", None)
        if policy_group is not None:
            candidate_terms = (
                "command",
                "motion_anchor_pos_b",
                "motion_anchor_ori_b",
                "base_lin_vel",
                "base_ang_vel",
                "joint_pos",
                "joint_vel",
                "actions",
                "twist2_motion",
                "twist2_proprio",
                "twist2_future",
                "twist2_1432_motion",
                "twist2_1432_proprio",
                "twist2_1432_future",
            )
            active_policy_terms = [name for name in candidate_terms if getattr(policy_group, name, None) is not None]
            print(
                "[INFO] Effective obs_pipeline: "
                f"mode={env_cfg.obs_pipeline.mode}, actor_mode={env_cfg.obs_pipeline.resolved_actor_mode()}, "
                f"critic_mode={env_cfg.obs_pipeline.resolved_critic_mode()}, "
                f"include_history={env_cfg.obs_pipeline.include_history}, "
                f"history_len={env_cfg.obs_pipeline.history_len}, "
                f"include_future={env_cfg.obs_pipeline.include_future}, "
                f"future_steps={env_cfg.obs_pipeline.future_steps}",
                flush=True,
            )
            print(f"[INFO] Active policy observation terms: {active_policy_terms}", flush=True)
            if env_cfg.obs_pipeline.resolved_actor_mode() == "twist2_1432":
                # Current frame dim is fixed by design:
                # - twist2_1432_motion: 35
                # - twist2_1432_proprio: 92
                # Future branch contributes 35 per configured step.
                history_frames = (env_cfg.obs_pipeline.history_len + 1) if env_cfg.obs_pipeline.include_history else 1
                future_steps = tuple(int(step) for step in env_cfg.obs_pipeline.future_steps)
                future_dim = (35 * len(future_steps)) if env_cfg.obs_pipeline.include_future else 0
                expected_dim = 127 * history_frames + future_dim
                print(
                    "[INFO] twist2_1432 dim estimate: "
                    f"history_frames={history_frames}, future_steps={future_steps}, "
                    f"expected_actor_obs_dim={expected_dim}",
                    flush=True,
                )
    print(f"[INFO] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    if torch.cuda.is_available():
        print(
            f"[INFO] torch.cuda.device_count()={torch.cuda.device_count()}, "
            f"torch.current_device={torch.cuda.current_device()}",
            flush=True,
        )
    else:
        print("[WARNING] torch.cuda.is_available() is False.", flush=True)

    motion_cfg = env_cfg.commands.motion
    motion_source_cfg = getattr(motion_cfg, "motion_source", None)
    source_mode = (
        str(getattr(motion_source_cfg, "mode", "single_npz")).lower()
        if motion_source_cfg is not None
        else "single_npz"
    )

    if args_cli.motion_file is not None and args_cli.motion_library is not None:
        raise ValueError("--motion_file and --motion_library cannot be set at the same time.")

    if args_cli.motion_library is not None:
        library_file = resolve_local_motion_library_file()
        # Keep legacy required field populated so config validation passes.
        motion_cfg.motion_file = library_file
        if motion_source_cfg is not None:
            motion_source_cfg.mode = "yaml_npz_library"
            motion_source_cfg.library_file = library_file
            motion_source_cfg.single_file = None
        print(f"[INFO] Motion source mode: yaml_npz_library ({library_file})")
    elif args_cli.motion_file is not None:
        motion_file = resolve_local_motion_file()
        motion_cfg.motion_file = motion_file
        if motion_source_cfg is not None:
            motion_source_cfg.mode = "single_npz"
            motion_source_cfg.single_file = motion_file
            motion_source_cfg.library_file = None
        print(f"[INFO] Motion source mode: single_npz ({motion_file})")
    elif source_mode == "yaml_npz_library":
        library_file = getattr(motion_source_cfg, "library_file", None) if motion_source_cfg is not None else None
        if not isinstance(library_file, str) or library_file == "":
            raise ValueError(
                "commands.motion.motion_source.mode=yaml_npz_library requires commands.motion.motion_source.library_file"
            )
        # Keep legacy required field populated so config validation passes.
        motion_cfg.motion_file = library_file
        print(f"[INFO] Motion source mode from config: yaml_npz_library ({library_file})")
    else:
        cfg_single_file = getattr(motion_source_cfg, "single_file", None) if motion_source_cfg is not None else None
        if isinstance(cfg_single_file, str) and cfg_single_file:
            motion_file = str(Path(cfg_single_file).expanduser().resolve())
            if not Path(motion_file).is_file():
                raise FileNotFoundError(f"Configured commands.motion.motion_source.single_file does not exist: {motion_file}")
            motion_cfg.motion_file = motion_file
            if motion_source_cfg is not None:
                motion_source_cfg.mode = "single_npz"
                motion_source_cfg.library_file = None
            print(f"[INFO] Motion source mode from config: single_npz ({motion_file})")
        else:
            motion_file = resolve_local_motion_file()
            motion_cfg.motion_file = motion_file
            if motion_source_cfg is not None:
                motion_source_cfg.mode = "single_npz"
                motion_source_cfg.single_file = motion_file
                motion_source_cfg.library_file = None
            print(f"[INFO] Motion source mode: single_npz ({motion_file})")

    # Disable debug visualizers for headless runs to avoid noisy Fabric marker warnings
    # and unnecessary rendering overhead.
    if args_cli.headless:
        if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
            if getattr(env_cfg.commands.motion, "debug_vis", False):
                env_cfg.commands.motion.debug_vis = False
                print("[INFO] Headless mode: commands.motion.debug_vis is forced to False.")
        if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "contact_forces"):
            if getattr(env_cfg.scene.contact_forces, "debug_vis", False):
                env_cfg.scene.contact_forces.debug_vis = False
                print("[INFO] Headless mode: scene.contact_forces.debug_vis is forced to False.")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Avoid wandb console redirection conflicts that can hide/interrupt traceback printing.
    if str(getattr(agent_cfg, "logger", "")).lower() == "wandb":
        if os.environ.get("WANDB_CONSOLE", None) is None:
            os.environ["WANDB_CONSOLE"] = "off"
            print("[INFO] WANDB_CONSOLE is set to 'off' for stable traceback/console behavior.", flush=True)
        patch_rsl_rl_wandb_writer()
        patch_wandb_exit_hooks(log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if args_cli.debug_timing:
        print("[INFO] Debug timing probe enabled.", flush=True)
        rollout_steps = int(getattr(agent_cfg, "num_steps_per_env", 24))
        probe_max_calls = max(24, min(128, rollout_steps + 8))
        attach_vec_env_timing_probe(env, max_calls=probe_max_calls, step_print_every=50)
        print(
            f"[INFO] Debug probe window: first {probe_max_calls} env.step calls "
            f"(num_steps_per_env={rollout_steps}).",
            flush=True,
        )

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=None)
    if args_cli.debug_timing:
        attach_runner_timing_probe(runner, max_calls=8, update_print_every=10)
        attach_post_update_timing_probe(runner, max_calls=8)
        print("[INFO] Debug timing probe attached to learner stages.", flush=True)
    if str(getattr(agent_cfg, "logger", "")).lower() == "wandb":
        install_safe_excepthook(log_dir)
        print("[INFO] Installed safe sys.excepthook for clearer uncaught exception diagnostics.", flush=True)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    print("[INFO] Entering runner.learn().", flush=True)
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    except BaseException:
        crash_log = os.path.join(log_dir, "fatal_exception.log")
        with open(crash_log, "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        print(f"[ERROR] Unhandled exception traceback saved to: {crash_log}", flush=True)
        raise

    # close the simulator
    env.close()


if __name__ == "__main__":
    try:
        # run the main function
        main()
    finally:
        # Always close Isaac Sim even if an exception/KeyboardInterrupt happens.
        # This reduces shutdown deadlocks and noisy secondary excepthook failures.
        simulation_app.close()
