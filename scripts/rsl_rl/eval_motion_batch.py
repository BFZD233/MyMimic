"""Batch evaluate policy tracking quality on multiple motions from a motion-library yaml.

This script runs short rollouts on several motion files and records tracking metrics from
the `motion` command term (e.g. anchor/body/joint tracking errors).
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata as metadata
import json
import random
import sys
import traceback
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Batch evaluate policy tracking on motions from a dataset yaml.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--ckpt_path", type=str, default=None, help="Checkpoint path. If unset, use defaults/fallback.")
parser.add_argument("--motion_yaml", type=str, required=True, help="Motion-library yaml used during training.")
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory to store intermediate and summary outputs. Defaults to ./tmps/motion_eval/<timestamp>.",
)
parser.add_argument("--num_motions", type=int, default=8, help="How many motion files to evaluate.")
parser.add_argument(
    "--selection",
    type=str,
    choices=("head", "spaced", "random"),
    default="spaced",
    help="How to pick motions from the yaml list.",
)
parser.add_argument("--seed", type=int, default=1, help="Random seed for motion selection and env.")
parser.add_argument("--steps", type=int, default=400, help="Simulation steps per motion.")
parser.add_argument("--warmup_steps", type=int, default=20, help="Initial steps excluded from summary means.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs for each evaluation run.")
parser.add_argument("--disable_time_out", action="store_true", default=True, help="Disable timeout termination.")
parser.add_argument(
    "--disable_startup_rand",
    action="store_true",
    default=True,
    help="Disable startup randomization events for cleaner evaluation.",
)
parser.add_argument(
    "--disable_push",
    action="store_true",
    default=True,
    help="Disable interval push event for cleaner evaluation.",
)
parser.add_argument(
    "--disable_obs_corruption",
    action="store_true",
    default=True,
    help="Disable policy observation noise corruption for deterministic evaluation.",
)
parser.add_argument(
    "--obs_mode",
    type=str,
    default="twist2_1432",
    choices=("legacy", "twist2_like", "twist2_1432"),
    help="Observation pipeline actor mode for evaluation.",
)
parser.add_argument("--history_len", type=int, default=10, help="obs_pipeline.history_len")
parser.add_argument("--include_history", action="store_true", default=True, help="obs_pipeline.include_history")
parser.add_argument("--include_future", action="store_true", default=True, help="obs_pipeline.include_future")
parser.add_argument(
    "--future_steps",
    type=str,
    default="0",
    help='Comma-separated non-negative ints for future steps, e.g. "0" or "0,1,2".',
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.tasks.tracking.obs_pipeline import apply_observation_pipeline


TRACKING_METRIC_KEYS = (
    "error_anchor_pos",
    "error_anchor_rot",
    "error_body_pos",
    "error_body_rot",
    "error_joint_pos",
    "error_joint_vel",
)


def _parse_future_steps(csv_text: str) -> tuple[int, ...]:
    parts = [item.strip() for item in csv_text.split(",") if item.strip() != ""]
    if not parts:
        return (0,)
    values = tuple(int(item) for item in parts)
    if any(item < 0 for item in values):
        raise ValueError(f"--future_steps must be non-negative, got: {values}")
    return values


def _resolve_output_dir() -> Path:
    if args_cli.output_dir:
        out_dir = Path(args_cli.output_dir).expanduser().resolve()
    else:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("tmps") / "motion_eval" / f"batch_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _resolve_ckpt_path(agent_cfg: RslRlOnPolicyRunnerCfg) -> str:
    if args_cli.ckpt_path:
        ckpt = Path(args_cli.ckpt_path).expanduser().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"--ckpt_path does not exist: {ckpt}")
        return str(ckpt)

    default_candidates = (
        Path("logs/rsl_rl/g1_flat/2026-04-02_05-45-59_20260402_twist2_1432outputs/model_24500.pt"),
        Path("logs/rsl_rl/g1_flat/2026-04-02_05-45-59_20260402_twist2_1432outputs/model_20000.pt"),
    )
    for candidate in default_candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return str(candidate)

    log_root_path = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
    log_root_path = log_root_path.resolve()
    ckpt_path = get_checkpoint_path(str(log_root_path), agent_cfg.load_run, agent_cfg.load_checkpoint)
    return str(Path(ckpt_path).resolve())


def _load_motion_entries(motion_yaml: Path) -> list[dict[str, Any]]:
    with motion_yaml.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Motion yaml is not a mapping: {motion_yaml}")

    root_path = payload.get("root_path", None)
    if not isinstance(root_path, str) or root_path == "":
        raise ValueError(f"Motion yaml missing root_path: {motion_yaml}")
    root = Path(root_path).expanduser().resolve()

    motions = payload.get("motions", None)
    if not isinstance(motions, list) or len(motions) == 0:
        raise ValueError(f"Motion yaml has no motions list: {motion_yaml}")

    entries: list[dict[str, Any]] = []
    for index, item in enumerate(motions):
        if not isinstance(item, dict):
            continue
        file_rel = item.get("file", None)
        if not isinstance(file_rel, str) or file_rel == "":
            continue
        file_path = Path(file_rel).expanduser()
        if not file_path.is_absolute():
            file_path = (root / file_path).resolve()
        else:
            file_path = file_path.resolve()
        entries.append(
            {
                "yaml_index": index,
                "file_rel": file_rel,
                "file_abs": str(file_path),
                "weight": float(item.get("weight", 1.0)),
                "description": str(item.get("description", "")),
                "exists": file_path.is_file(),
            }
        )
    return entries


def _pick_motion_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [entry for entry in entries if entry["exists"]]
    if len(valid) == 0:
        raise ValueError("No existing motion files found from motion yaml.")

    num = max(1, min(args_cli.num_motions, len(valid)))
    if num == len(valid):
        return valid

    if args_cli.selection == "head":
        chosen = valid[:num]
    elif args_cli.selection == "random":
        rng = random.Random(args_cli.seed)
        chosen = rng.sample(valid, num)
        chosen = sorted(chosen, key=lambda item: int(item["yaml_index"]))
    else:
        # spaced
        if num == 1:
            indices = [0]
        else:
            indices = [round(i * (len(valid) - 1) / float(num - 1)) for i in range(num)]
        dedup: list[int] = []
        for idx in indices:
            if idx not in dedup:
                dedup.append(idx)
        while len(dedup) < num:
            for idx in range(len(valid)):
                if idx not in dedup:
                    dedup.append(idx)
                if len(dedup) >= num:
                    break
        chosen = [valid[idx] for idx in dedup[:num]]
    return chosen


def _apply_eval_overrides(env_cfg, motion_file: str) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Fix this run to a single motion file.
    env_cfg.commands.motion.motion_file = motion_file
    motion_source_cfg = getattr(env_cfg.commands.motion, "motion_source", None)
    if motion_source_cfg is not None:
        motion_source_cfg.mode = "single_npz"
        motion_source_cfg.single_file = motion_file
        motion_source_cfg.library_file = None

    # Keep playback clean/fast in headless mode.
    if getattr(args_cli, "headless", False):
        if hasattr(env_cfg.commands, "motion"):
            env_cfg.commands.motion.debug_vis = False
        if hasattr(env_cfg.scene, "contact_forces"):
            env_cfg.scene.contact_forces.debug_vis = False

    if args_cli.disable_time_out and hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    if args_cli.disable_push and hasattr(env_cfg, "events") and hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if args_cli.disable_startup_rand and hasattr(env_cfg, "events"):
        for term_name in ("physics_material", "add_joint_default_pos", "base_com"):
            if hasattr(env_cfg.events, term_name):
                setattr(env_cfg.events, term_name, None)

    if args_cli.disable_obs_corruption and hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.enable_corruption = False

    # Use a world-anchored camera to avoid tracking callbacks from affecting playback interaction.
    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.asset_name = None

    # Apply requested observation pipeline.
    if hasattr(env_cfg, "obs_pipeline"):
        env_cfg.obs_pipeline.mode = args_cli.obs_mode
        env_cfg.obs_pipeline.actor_mode = None
        env_cfg.obs_pipeline.critic_mode = None
        env_cfg.obs_pipeline.include_history = bool(args_cli.include_history)
        env_cfg.obs_pipeline.history_len = int(args_cli.history_len)
        env_cfg.obs_pipeline.include_future = bool(args_cli.include_future)
        env_cfg.obs_pipeline.future_steps = _parse_future_steps(args_cli.future_steps)
        env_cfg.obs_pipeline.future_include_joint_pos = True
        env_cfg.obs_pipeline.future_include_joint_vel = False
        env_cfg.obs_pipeline.__post_init__()
        apply_observation_pipeline(env_cfg)


def _mean_metric_value(command_term, key: str) -> float:
    value = command_term.metrics.get(key, None)
    if value is None:
        return float("nan")
    return float(value.mean().item())


def _write_step_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if len(rows) == 0:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_rows(rows: list[dict[str, Any]], warmup_steps: int) -> dict[str, float]:
    if len(rows) == 0:
        return {}
    start = max(0, min(warmup_steps, len(rows)))
    body = rows[start:] if start < len(rows) else rows

    summary: dict[str, float] = {}
    summary["evaluated_steps"] = float(len(rows))
    summary["warmup_steps"] = float(start)
    summary["post_warmup_steps"] = float(len(body))
    summary["done_events"] = float(sum(int(row["done_count"]) for row in rows))
    summary["done_events_post_warmup"] = float(sum(int(row["done_count"]) for row in body))
    summary["mean_reward"] = float(sum(float(row["reward_mean"]) for row in body) / max(len(body), 1))
    for key in TRACKING_METRIC_KEYS:
        k = f"mean_{key}"
        summary[k] = float(sum(float(row[key]) for row in body) / max(len(body), 1))
    return summary


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    output_dir = _resolve_output_dir()
    per_motion_dir = output_dir / "per_motion"
    per_motion_dir.mkdir(parents=True, exist_ok=True)

    motion_yaml = Path(args_cli.motion_yaml).expanduser().resolve()
    if not motion_yaml.is_file():
        raise FileNotFoundError(f"--motion_yaml does not exist: {motion_yaml}")

    random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device

    ckpt_path = _resolve_ckpt_path(agent_cfg)
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"Resolved checkpoint does not exist: {ckpt_path}")

    entries = _load_motion_entries(motion_yaml)
    selected = _pick_motion_entries(entries)

    run_meta = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "task": args_cli.task,
        "ckpt_path": ckpt_path,
        "motion_yaml": str(motion_yaml),
        "num_motions_requested": args_cli.num_motions,
        "num_motions_selected": len(selected),
        "selection": args_cli.selection,
        "seed": args_cli.seed,
        "steps": args_cli.steps,
        "warmup_steps": args_cli.warmup_steps,
        "device": args_cli.device,
        "num_envs": args_cli.num_envs,
        "obs_mode": args_cli.obs_mode,
        "history_len": args_cli.history_len,
        "include_history": bool(args_cli.include_history),
        "include_future": bool(args_cli.include_future),
        "future_steps": list(_parse_future_steps(args_cli.future_steps)),
        "disable_time_out": bool(args_cli.disable_time_out),
        "disable_startup_rand": bool(args_cli.disable_startup_rand),
        "disable_push": bool(args_cli.disable_push),
        "disable_obs_corruption": bool(args_cli.disable_obs_corruption),
    }
    with (output_dir / "run_meta.json").open("w", encoding="utf-8") as file:
        json.dump(run_meta, file, indent=2, ensure_ascii=True)
    with (output_dir / "selected_motions.json").open("w", encoding="utf-8") as file:
        json.dump(selected, file, indent=2, ensure_ascii=True)

    all_summaries: list[dict[str, Any]] = []
    for seq_idx, motion_entry in enumerate(selected):
        motion_file = motion_entry["file_abs"]
        print(
            f"[INFO] [{seq_idx + 1}/{len(selected)}] evaluating motion: {motion_entry['file_rel']}",
            flush=True,
        )

        env_cfg_i = deepcopy(env_cfg)
        _apply_eval_overrides(env_cfg_i, motion_file)

        env = None
        try:
            env = gym.make(args_cli.task, cfg=env_cfg_i, render_mode=None)
            if isinstance(env.unwrapped, DirectMARLEnv):
                env = multi_agent_to_single_agent(env)
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(ckpt_path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)

            # Wrapper APIs differ across versions: consume first element as observations.
            reset_out = env.get_observations()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            step_rows: list[dict[str, Any]] = []

            for step in range(args_cli.steps):
                with torch.inference_mode():
                    actions = policy(obs)
                    step_out = env.step(actions)
                    if not isinstance(step_out, tuple) or len(step_out) < 3:
                        raise RuntimeError(
                            f"Unexpected env.step output type/arity: {type(step_out)} / {len(step_out) if isinstance(step_out, tuple) else 'n/a'}"
                        )
                    obs = step_out[0]
                    reward = step_out[1]
                    dones = step_out[2]
                    try:
                        policy.reset(dones)
                    except Exception:
                        pass

                command_term = env.unwrapped.command_manager.get_term("motion")
                row = {
                    "step": step,
                    "reward_mean": float(reward.mean().item()),
                    "done_count": int(dones.sum().item()),
                }
                for key in TRACKING_METRIC_KEYS:
                    row[key] = _mean_metric_value(command_term, key)
                step_rows.append(row)
        except Exception as exc:
            print(
                f"[ERROR] motion evaluation failed for {motion_entry['file_rel']}: {exc}\n{traceback.format_exc()}",
                flush=True,
            )
            raise

        safe_name = f"{seq_idx:03d}__{Path(motion_file).name.replace(' ', '_')}"
        step_csv_path = per_motion_dir / f"{safe_name}.csv"
        _write_step_csv(step_csv_path, step_rows)

        summary = {
            "sequence_index": seq_idx,
            "yaml_index": int(motion_entry["yaml_index"]),
            "file_rel": motion_entry["file_rel"],
            "file_abs": motion_entry["file_abs"],
            "steps": int(args_cli.steps),
            "step_csv": str(step_csv_path.resolve()),
        }
        summary.update(_summarize_rows(step_rows, args_cli.warmup_steps))
        all_summaries.append(summary)

        with (per_motion_dir / f"{safe_name}__summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=True)

        if env is not None:
            env.close()

    # write summary table
    summary_csv_path = output_dir / "summary.csv"
    if all_summaries:
        fieldnames = list(all_summaries[0].keys())
        with summary_csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_summaries)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(all_summaries, file, indent=2, ensure_ascii=True)

    # aggregate headline
    headline = {}
    if all_summaries:
        denom = float(len(all_summaries))
        headline["num_sequences"] = len(all_summaries)
        headline["avg_done_events_post_warmup"] = sum(s["done_events_post_warmup"] for s in all_summaries) / denom
        headline["avg_mean_error_anchor_pos"] = sum(s["mean_error_anchor_pos"] for s in all_summaries) / denom
        headline["avg_mean_error_body_pos"] = sum(s["mean_error_body_pos"] for s in all_summaries) / denom
        headline["avg_mean_error_joint_pos"] = sum(s["mean_error_joint_pos"] for s in all_summaries) / denom
    with (output_dir / "headline.json").open("w", encoding="utf-8") as file:
        json.dump(headline, file, indent=2, ensure_ascii=True)

    print(f"[INFO] Evaluation finished. Output dir: {output_dir}", flush=True)
    if summary_csv_path.is_file():
        print(f"[INFO] Summary csv: {summary_csv_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
