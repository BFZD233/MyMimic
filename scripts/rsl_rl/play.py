"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib.metadata as metadata
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument(
    "--play_no_reset",
    action="store_true",
    default=False,
    help="Disable common training-time resets/disturbances for continuous motion playback.",
)
parser.add_argument(
    "--play_free_camera",
    action="store_true",
    default=False,
    help="Use world-anchored camera so manual viewport drag is not pulled back by asset tracking.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
raw_hydra_args = list(hydra_args)
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True


def _rewrite_env_overrides(overrides: list[str]) -> list[str]:
    rewritten: list[str] = []
    for token in overrides:
        prefix = ""
        body = token
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


hydra_args = _rewrite_env_overrides(hydra_args)

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import load_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.tasks.tracking.obs_pipeline import apply_observation_pipeline
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


def resolve_cli_motion_file() -> str | None:
    if args_cli.motion_file is None:
        return None
    motion_path = pathlib.Path(args_cli.motion_file).expanduser().resolve()
    if not motion_path.is_file():
        raise FileNotFoundError(f"--motion_file does not exist: {motion_path}")
    print(f"[INFO] Using motion file from CLI: {motion_path}")
    return str(motion_path)


def resolve_motion_file_from_saved_env_cfg(resume_path: str) -> str | None:
    env_yaml_path = pathlib.Path(resume_path).resolve().parent / "params" / "env.yaml"
    if not env_yaml_path.is_file():
        return None

    env_yaml = load_yaml(str(env_yaml_path))
    motion_file = (
        env_yaml.get("commands", {})
        .get("motion", {})
        .get("motion_file", None)
    )
    if not isinstance(motion_file, str) or not motion_file:
        return None
    motion_path = pathlib.Path(motion_file).expanduser().resolve()
    if motion_path.is_file():
        print(f"[INFO] Using motion file from saved run config: {motion_path}")
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


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    if args_cli.play_no_reset:
        # Keep playback continuous: remove short episode timeout and typical early-fail reset terms.
        env_cfg.episode_length_s = float(max(float(getattr(env_cfg, "episode_length_s", 10.0)), 1.0e6))
        if hasattr(env_cfg, "terminations"):
            for term_name in ("time_out", "anchor_pos", "anchor_ori", "ee_body_pos"):
                if hasattr(env_cfg.terminations, term_name):
                    setattr(env_cfg.terminations, term_name, None)
        if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "push_robot"):
            env_cfg.events.push_robot = None
        print("[INFO] play_no_reset enabled: disabled time_out/anchor terminations and push_robot event.", flush=True)

    if args_cli.play_free_camera:
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.asset_name = None
        print("[INFO] play_free_camera enabled: viewer.origin_type=world (no asset-follow snap-back).", flush=True)

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
                "[INFO] Effective obs_pipeline (play): "
                f"mode={env_cfg.obs_pipeline.mode}, actor_mode={env_cfg.obs_pipeline.resolved_actor_mode()}, "
                f"critic_mode={env_cfg.obs_pipeline.resolved_critic_mode()}, "
                f"include_history={env_cfg.obs_pipeline.include_history}, "
                f"history_len={env_cfg.obs_pipeline.history_len}, "
                f"include_future={env_cfg.obs_pipeline.include_future}, "
                f"future_steps={env_cfg.obs_pipeline.future_steps}",
                flush=True,
            )
            print(f"[INFO] Active policy observation terms (play): {active_policy_terms}", flush=True)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    artifact_motion_file = None

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        else:
            artifact_motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # motion source priority:
    # 1) --motion_file
    # 2) wandb artifact motion (if wandb_path is used)
    # 3) saved run config at <run>/params/env.yaml
    # 4) task default motion_file
    motion_file = resolve_cli_motion_file()
    if motion_file is None and artifact_motion_file is not None:
        motion_file = artifact_motion_file
        print(f"[INFO] Using motion file from wandb artifact: {motion_file}")
    if motion_file is None:
        motion_file = resolve_motion_file_from_saved_env_cfg(resume_path)
    if motion_file is None:
        motion_file = resolve_default_motion_file(env_cfg)
    if motion_file is None:
        raise ValueError(
            "Could not resolve motion file. Please pass --motion_file /path/to/motion.npz "
            "or use --wandb_path with a run that has a motions artifact."
        )
    env_cfg.commands.motion.motion_file = motion_file

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
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

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    policy_for_export = ppo_runner.alg.get_policy() if hasattr(ppo_runner.alg, "get_policy") else ppo_runner.alg.policy
    try:
        export_motion_policy_as_onnx(
            env.unwrapped,
            policy_for_export,
            normalizer=getattr(ppo_runner, "obs_normalizer", None),
            path=export_model_dir,
            filename="policy.onnx",
        )
        attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)
    except Exception as exc:
        # Video playback should still proceed even if export preconditions are not met.
        print(f"[WARN] Skipping ONNX export during play: {exc}", flush=True)
    # reset environment (wrapper API may return either obs or (obs, extras))
    reset_out = env.get_observations()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            # Wrapper APIs differ across versions: consume first element as observations.
            step_out = env.step(actions)
            obs = step_out[0] if isinstance(step_out, tuple) else step_out
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
