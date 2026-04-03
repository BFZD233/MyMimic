#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -f "/etc/vulkan/icd.d/nvidia_icd.json" ]]; then
  export VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json"
fi

TARGET_CONDA_ENV="${TARGET_CONDA_ENV:-env_isaaclab}"
USE_TWIST2_1432_OBS="${USE_TWIST2_1432_OBS:-1}" # 1: use TWIST2-compatible 1432-dim actor obs
TWIST2_1432_INCLUDE_FUTURE="${TWIST2_1432_INCLUDE_FUTURE:-1}" # 1: include future branch (default target: 1432)
USER_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
RUN_NAME="${RUN_NAME:-20260402_twist2_1432outputs}"

detect_idle_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '{gsub(/ /,""); print $1" "$2" "$3}' \
    | sort -k2n -k3n \
    | awk 'NR==1{print $1}'
}

if [[ -z "${USER_CUDA_VISIBLE_DEVICES}" ]]; then
  AUTO_DETECTED_GPU="$(detect_idle_gpu || true)"
  CUDA_VISIBLE_DEVICES="${AUTO_DETECTED_GPU:-0}"
else
  CUDA_VISIBLE_DEVICES="${USER_CUDA_VISIBLE_DEVICES}"
fi

# Ensure expected conda env for IsaacLab.
if [[ "${CONDA_DEFAULT_ENV:-}" != "${TARGET_CONDA_ENV}" ]]; then
  if [[ -f "/home/huangyuming/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    set +u
    source /home/huangyuming/anaconda3/etc/profile.d/conda.sh
    conda activate "${TARGET_CONDA_ENV}"
    set -u
  else
    echo "[ERROR] conda.sh not found. Please activate ${TARGET_CONDA_ENV} manually." >&2
    exit 1
  fi
fi

# Keep runtime libs aligned with conda + system driver libs.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi

echo "[INFO] conda_env: ${CONDA_DEFAULT_ENV:-<none>}"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

# Fast-fail if current environment cannot see CUDA at all.
set +e
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -c 'import os, sys
try:
    import torch
except Exception as error:
    print(f"[ERROR] torch import failed: {error}")
    sys.exit(3)
print(f"[INFO] torch: version={torch.__version__}, cuda_build={torch.version.cuda}")
n = torch.cuda.device_count()
print(f"[INFO] torch.cuda.is_available={torch.cuda.is_available()}, visible_count={n}")
if n <= 0:
    sys.exit(2)
try:
    print(f"[INFO] visible_gpu0={torch.cuda.get_device_name(0)}")
except Exception as error:
    print(f"[WARN] unable to query gpu name: {error}")
sys.exit(0)'
CUDA_CHECK_RC=$?
set -e
if [[ ${CUDA_CHECK_RC} -ne 0 ]]; then
  if [[ ${CUDA_CHECK_RC} -eq 3 ]]; then
    echo "[ERROR] Python env is missing usable torch. Try: conda activate ${TARGET_CONDA_ENV}" >&2
  else
    echo "[ERROR] No CUDA GPU visible to torch in current shell." >&2
    echo "[ERROR] This indicates runtime/driver/container GPU passthrough issue (not obs config)." >&2
    echo "[ERROR] Quick checks: /dev/nvidia* exists, NVIDIA container runtime is enabled, and this shell can run CUDA torch." >&2
  fi
  exit 2
fi

OBS_ARGS=(
  obs_pipeline.mode=twist2_like
  obs_pipeline.include_history=true
  obs_pipeline.history_len=10
  obs_pipeline.include_future=false
)

if [[ "${USE_TWIST2_1432_OBS}" == "1" ]]; then
  if [[ "${TWIST2_1432_INCLUDE_FUTURE}" == "1" ]]; then
    OBS_ARGS=(
      obs_pipeline.mode=twist2_1432
      obs_pipeline.include_history=true
      obs_pipeline.history_len=10
      obs_pipeline.include_future=true
      obs_pipeline.future_steps=[0]
      obs_pipeline.future_include_joint_pos=true
      obs_pipeline.future_include_joint_vel=false
    )
    echo "[INFO] Using TWIST2-compatible obs pipeline (target actor input dim: 1432)."
  else
    OBS_ARGS=(
      obs_pipeline.mode=twist2_1432
      obs_pipeline.include_history=true
      obs_pipeline.history_len=10
      obs_pipeline.include_future=false
    )
    echo "[INFO] Using TWIST2-compatible obs pipeline without future branch (target actor input dim: 1397)."
  fi
else
  echo "[INFO] Using default obs pipeline (twist2_like)."
fi
echo "[INFO] OBS overrides: ${OBS_ARGS[*]}"

HYDRA_FULL_ERROR=1 \
PYTHONUNBUFFERED=1 \
WANDB_CONSOLE=off \
LIVESTREAM=0 \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python scripts/rsl_rl/train_ours.py --task=Tracking-Flat-G1-v0 \
  --motion_root /media/raid/workspace/huangyuming/TWIST2/TWIST2_full_npz/ \
  --device cuda:0 \
  --headless --logger wandb --log_project_name deepmimic --run_name "${RUN_NAME}" \
  --debug_timing \
  --kit_args "--/app/hangDetector/enabled=false --/app/hangDetector/timeout=3600 --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1 --/renderer/activeGpu=0 --/physics/cudaDevice=0" \
  "${OBS_ARGS[@]}" \
  --motion_library /media/raid/workspace/huangyuming/lzd/whole_body_tracking/config/twist2_dataset_npz.yaml
