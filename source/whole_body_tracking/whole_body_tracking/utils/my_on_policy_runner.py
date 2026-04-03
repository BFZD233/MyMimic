import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.exporter import (
    attach_onnx_metadata,
    export_motion_policy_as_onnx,
    export_obs_policy_as_onnx,
)


def _is_wandb_logger(runner: OnPolicyRunner) -> bool:
    logger = getattr(runner, "logger", None)
    if logger is not None and getattr(logger, "writer", None) is None:
        return False
    logger_type = getattr(logger, "logger_type", None)
    if logger_type is None:
        cfg = getattr(runner, "cfg", None)
        if isinstance(cfg, dict):
            logger_type = cfg.get("logger", None)
    return isinstance(logger_type, str) and logger_type.lower() == "wandb"


def _get_policy_module(runner: OnPolicyRunner):
    if hasattr(runner.alg, "get_policy"):
        return runner.alg.get_policy()
    return getattr(runner.alg, "policy", None)


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if _is_wandb_logger(self):
            if wandb.run is None:
                return
            policy = _get_policy_module(self)
            if policy is None:
                return
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(policy, normalizer=getattr(self, "obs_normalizer", None), path=policy_path, filename=filename)
            run_name = wandb.run.name if wandb.run is not None else "none"
            attach_onnx_metadata(self.env.unwrapped, run_name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if _is_wandb_logger(self):
            if wandb.run is None:
                return
            policy = _get_policy_module(self)
            if policy is None:
                return
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            try:
                export_motion_policy_as_onnx(
                    self.env.unwrapped,
                    policy,
                    normalizer=getattr(self, "obs_normalizer", None),
                    path=policy_path,
                    filename=filename,
                )
            except Exception as motion_export_exc:
                # Multi-motion mode does not provide a single `cmd.motion` timeline for motion-head export.
                # Fall back to obs-only ONNX so checkpointing/logging never interrupts training.
                print(
                    "[WARNING] Motion ONNX export failed; falling back to obs-only ONNX export. "
                    f"reason={motion_export_exc}",
                    flush=True,
                )
                try:
                    export_obs_policy_as_onnx(
                        policy,
                        normalizer=getattr(self, "obs_normalizer", None),
                        path=policy_path,
                        filename=filename,
                    )
                except Exception as obs_export_exc:
                    print(
                        "[WARNING] Obs-only ONNX export also failed; skipping ONNX upload for this checkpoint. "
                        f"reason={obs_export_exc}",
                        flush=True,
                    )
                    return
            run_name = wandb.run.name if wandb.run is not None else "none"
            try:
                attach_onnx_metadata(self.env.unwrapped, run_name, path=policy_path, filename=filename)
            except Exception as metadata_exc:
                print(
                    f"[WARNING] Failed to attach ONNX metadata; continuing without metadata. reason={metadata_exc}",
                    flush=True,
                )
            try:
                wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
            except Exception as wandb_save_exc:
                print(
                    f"[WARNING] Failed to upload ONNX artifact to W&B; continuing training. reason={wandb_save_exc}",
                    flush=True,
                )

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
