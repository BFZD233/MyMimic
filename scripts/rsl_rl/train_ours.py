# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL (local motion version)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib.metadata as metadata
import pickle
import sys
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

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Backward-compatible override rewrite:
# The Hydra root config is {"env": ..., "agent": ...}, so env overrides should be under "env.*".
# We transparently map shorthand "obs_pipeline.*" to "env.obs_pipeline.*".
def _rewrite_obs_pipeline_overrides(overrides: list[str]) -> list[str]:
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
        rewritten.append(prefix + body)
    return rewritten

hydra_args = _rewrite_obs_pipeline_overrides(hydra_args)

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
import os
import torch
from datetime import datetime

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

    # load the motion file from local filesystem
    env_cfg.commands.motion.motion_file = resolve_local_motion_file()

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

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

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=None)
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
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
