# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to export an RSL-RL checkpoint to ONNX."""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib.metadata as metadata
import pathlib
import sys

import yaml
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Export a trained RSL-RL checkpoint to ONNX.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--ckpt_path", type=str, required=True, help="Absolute or relative path to checkpoint .pt file.")
parser.add_argument("--motion_file", type=str, default=None, help="Optional override motion npz file path.")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory for ONNX. Defaults to <ckpt_dir>/exported.")
parser.add_argument("--output_name", type=str, default="policy.onnx", help="Output ONNX filename.")
parser.add_argument("--metadata_run_path", type=str, default=None, help="Value written to ONNX metadata key 'run_path'.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to instantiate for export.")
parser.add_argument(
    "--obs_only",
    action="store_true",
    default=False,
    help="Export play-style ONNX that only takes `obs` and outputs `actions`.",
)
parser.add_argument(
    "--use_saved_obs_pipeline",
    action="store_true",
    default=False,
    help=(
        "If set, try to apply obs_pipeline from <ckpt_dir>/params/env.yaml before creating env. "
        "Keep disabled when checkpoint and saved env config may be out-of-sync."
    ),
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
raw_hydra_args = list(hydra_args)

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.io import load_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.tasks.tracking.obs_pipeline import apply_observation_pipeline
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx, export_obs_policy_as_onnx


def resolve_ckpt_path() -> str:
    ckpt_path = pathlib.Path(args_cli.ckpt_path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"--ckpt_path does not exist: {ckpt_path}")
    return str(ckpt_path)


def resolve_cli_motion_file() -> str | None:
    if args_cli.motion_file is None:
        return None
    motion_path = pathlib.Path(args_cli.motion_file).expanduser().resolve()
    if not motion_path.is_file():
        raise FileNotFoundError(f"--motion_file does not exist: {motion_path}")
    print(f"[INFO] Using motion file from CLI: {motion_path}")
    return str(motion_path)


def load_saved_env_yaml(ckpt_path: str) -> tuple[dict | None, pathlib.Path | None]:
    env_yaml_path = pathlib.Path(ckpt_path).resolve().parent / "params" / "env.yaml"
    if not env_yaml_path.is_file():
        return None, None

    env_yaml = None
    try:
        env_yaml = load_yaml(str(env_yaml_path))
    except Exception as error:
        print(f"[WARN] Failed to parse env.yaml with isaaclab loader: {error}")
        print("[WARN] Falling back to yaml.unsafe_load for legacy python-tagged YAML.")
        try:
            with env_yaml_path.open("r", encoding="utf-8") as file:
                env_yaml = yaml.unsafe_load(file)
        except Exception as fallback_error:
            print(f"[WARN] yaml.unsafe_load also failed: {fallback_error}")

    return (env_yaml if isinstance(env_yaml, dict) else None), env_yaml_path


def resolve_motion_file_from_saved_env_cfg(ckpt_path: str) -> str | None:
    env_yaml, env_yaml_path = load_saved_env_yaml(ckpt_path)
    if env_yaml_path is None:
        return None

    if env_yaml is not None:
        motion_file = env_yaml.get("commands", {}).get("motion", {}).get("motion_file", None)
        if isinstance(motion_file, str) and motion_file:
            motion_path = pathlib.Path(motion_file).expanduser().resolve()
            if motion_path.is_file():
                print(f"[INFO] Using motion file from saved run config: {motion_path}")
                return str(motion_path)

    # Final fallback: line-based parsing to avoid any YAML constructor requirements.
    motion_file = extract_motion_file_via_text_scan(env_yaml_path)
    if motion_file:
        motion_path = pathlib.Path(motion_file).expanduser().resolve()
        if motion_path.is_file():
            print(f"[INFO] Using motion file from saved run config (text fallback): {motion_path}")
            return str(motion_path)
    return None


def resolve_default_motion_file(env_cfg) -> str | None:
    motion_cfg = getattr(getattr(env_cfg, "commands", None), "motion", None)
    motion_file = getattr(motion_cfg, "motion_file", None)
    if isinstance(motion_file, str) and motion_file and pathlib.Path(motion_file).is_file():
        motion_path = pathlib.Path(motion_file).expanduser().resolve()
        print(f"[INFO] Using motion file from task default config: {motion_path}")
        return str(motion_path)
    return None


def resolve_motion_source_from_saved_env_cfg(ckpt_path: str) -> dict[str, object] | None:
    env_yaml, _ = load_saved_env_yaml(ckpt_path)
    if env_yaml is None:
        return None

    source = env_yaml.get("commands", {}).get("motion", {}).get("motion_source", None)
    if not isinstance(source, dict):
        return None

    resolved: dict[str, object] = {}
    supported_keys = ("mode", "single_file", "library_file", "root_dir", "normalize_weights", "default_weight")
    for key in supported_keys:
        if key in source:
            resolved[key] = source[key]
    return resolved if resolved else None


def apply_motion_source_cfg(env_cfg, motion_source_cfg: dict[str, object] | None) -> None:
    if not motion_source_cfg:
        return

    target = getattr(getattr(getattr(env_cfg, "commands", None), "motion", None), "motion_source", None)
    if target is None:
        return

    for key, value in motion_source_cfg.items():
        if hasattr(target, key):
            setattr(target, key, value)
    print(f"[INFO] Applied motion_source from saved run config: {motion_source_cfg}")


def resolve_output_dir(ckpt_path: str) -> str:
    if args_cli.output_dir is not None:
        output_dir = pathlib.Path(args_cli.output_dir).expanduser().resolve()
    else:
        output_dir = pathlib.Path(ckpt_path).resolve().parent / "exported"
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def extract_motion_file_via_text_scan(env_yaml_path: pathlib.Path) -> str | None:
    commands_indent = None
    motion_indent = None

    with env_yaml_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.rstrip("\n")
            stripped = line.lstrip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(line) - len(stripped)

            if commands_indent is None:
                if stripped == "commands:":
                    commands_indent = indent
                continue

            # Exit commands block when dedented to same/higher level and a new key starts.
            if indent <= commands_indent and stripped.endswith(":") and stripped != "commands:":
                commands_indent = None
                motion_indent = None
                continue

            if motion_indent is None:
                if indent == commands_indent + 2 and stripped == "motion:":
                    motion_indent = indent
                continue

            # Exit motion block when dedented to same/higher level and a new key starts.
            if indent <= motion_indent and stripped.endswith(":") and stripped != "motion:":
                motion_indent = None
                continue

            if indent == motion_indent + 2 and stripped.startswith("motion_file:"):
                value = stripped.split(":", 1)[1].strip()
                if not value:
                    return None
                return value.strip("\"'")

    return None


def _strip_hydra_prefix_operators(token: str) -> str:
    body = token
    while body.startswith("+"):
        body = body[1:]
    if body.startswith("~"):
        body = body[1:]
    return body


def has_cli_obs_pipeline_override() -> bool:
    for token in raw_hydra_args:
        body = _strip_hydra_prefix_operators(token)
        if body.startswith("env.obs_pipeline.") or body.startswith("obs_pipeline."):
            return True
    return False


def resolve_obs_pipeline_from_saved_env_cfg(ckpt_path: str) -> dict[str, object] | None:
    env_yaml, env_yaml_path = load_saved_env_yaml(ckpt_path)
    if env_yaml is None:
        return None

    obs_pipeline = env_yaml.get("obs_pipeline", None)
    if not isinstance(obs_pipeline, dict):
        print(f"[WARN] Saved env config missing obs_pipeline in: {env_yaml_path}")
        return None

    resolved: dict[str, object] = {}
    supported_keys = (
        "mode",
        "actor_mode",
        "critic_mode",
        "include_history",
        "history_len",
        "include_future",
        "future_steps",
        "future_include_joint_pos",
        "future_include_joint_vel",
    )

    for key in supported_keys:
        if key in obs_pipeline:
            resolved[key] = obs_pipeline[key]

    if not resolved:
        return None

    if isinstance(resolved.get("future_steps"), list):
        resolved["future_steps"] = tuple(resolved["future_steps"])
    return resolved


def apply_obs_pipeline_from_saved_env_cfg_if_needed(env_cfg, ckpt_path: str) -> None:
    if has_cli_obs_pipeline_override():
        print("[INFO] Keeping CLI/Hydra-provided obs_pipeline overrides for export.")
        return

    saved_obs_pipeline = resolve_obs_pipeline_from_saved_env_cfg(ckpt_path)
    if saved_obs_pipeline is None:
        return

    for key, value in saved_obs_pipeline.items():
        if hasattr(env_cfg.obs_pipeline, key):
            setattr(env_cfg.obs_pipeline, key, value)

    # Normalize and validate (lower-case modes, tuple conversion, constraints).
    env_cfg.obs_pipeline.__post_init__()
    apply_observation_pipeline(env_cfg)
    print(f"[INFO] Applied obs_pipeline from saved run config: {saved_obs_pipeline}")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Load a checkpoint and export the policy as ONNX."""
    del agent_cfg
    ckpt_path = resolve_ckpt_path()

    loaded_agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    installed_version = metadata.version("rsl-rl-lib")
    loaded_agent_cfg = handle_deprecated_rsl_rl_cfg(loaded_agent_cfg, installed_version)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.use_saved_obs_pipeline:
        apply_obs_pipeline_from_saved_env_cfg_if_needed(env_cfg, ckpt_path)
    else:
        print("[INFO] Using task/default obs_pipeline (set --use_saved_obs_pipeline to load from saved env.yaml).")

    # motion source priority:
    # 1) --motion_file
    # 2) saved run config at <ckpt_dir>/params/env.yaml
    # 3) task default motion_file
    cli_motion_file = resolve_cli_motion_file()
    motion_file = cli_motion_file
    if motion_file is None:
        motion_file = resolve_motion_file_from_saved_env_cfg(ckpt_path)
    if motion_file is None:
        motion_file = resolve_default_motion_file(env_cfg)
    if motion_file is None:
        raise ValueError(
            "Could not resolve motion file. Please pass --motion_file /path/to/motion.npz "
            "or ensure <ckpt_dir>/params/env.yaml contains commands.motion.motion_file."
        )
    env_cfg.commands.motion.motion_file = motion_file
    if cli_motion_file is None:
        apply_motion_source_cfg(env_cfg, resolve_motion_source_from_saved_env_cfg(ckpt_path))
    else:
        print("[INFO] Skipping saved motion_source override because --motion_file was provided.")

    output_dir = resolve_output_dir(ckpt_path)
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    print(f"[INFO] Export output directory: {output_dir}")
    print(f"[INFO] Export output filename: {args_cli.output_name}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        # wrap around environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=loaded_agent_cfg.clip_actions)

        # load previously trained model
        runner = OnPolicyRunner(env, loaded_agent_cfg.to_dict(), log_dir=None, device=loaded_agent_cfg.device)
        runner.load(ckpt_path)

        # export policy to onnx
        policy_for_export = runner.alg.get_policy() if hasattr(runner.alg, "get_policy") else runner.alg.policy
        if args_cli.obs_only:
            print("[INFO] Export mode: obs-only (play-style)")
            export_obs_policy_as_onnx(
                policy_for_export,
                normalizer=getattr(runner, "obs_normalizer", None),
                path=output_dir,
                filename=args_cli.output_name,
            )
        else:
            print("[INFO] Export mode: motion (obs + time_step)")
            export_motion_policy_as_onnx(
                env.unwrapped,
                policy_for_export,
                normalizer=getattr(runner, "obs_normalizer", None),
                path=output_dir,
                filename=args_cli.output_name,
            )

        run_path = args_cli.metadata_run_path if args_cli.metadata_run_path else ckpt_path
        attach_onnx_metadata(env.unwrapped, run_path, output_dir, filename=args_cli.output_name)
    finally:
        env.close()

    print(f"[INFO] ONNX export completed: {pathlib.Path(output_dir) / args_cli.output_name}")


if __name__ == "__main__":
    try:
        main()
    finally:
        # close sim app
        simulation_app.close()
