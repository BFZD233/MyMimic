#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Usage:
#   bash to_onnx.sh [ckpt_path] [task]
#
# Defaults keep your current workflow unchanged.
DEFAULT_CKPT_PATH="/media/raid/workspace/huangyuming/lzd/whole_body_tracking/logs/rsl_rl/g1_flat/2026-04-02_05-45-59_20260402_twist2_1432outputs/model_24000.pt"
DEFAULT_TASK="Tracking-Flat-G1-v0"
DEFAULT_CONDA_ENV="env_isaaclab"
DEFAULT_CUDA_DEVICE="7"
DEFAULT_MOTION_FILE_OVERRIDE="/media/raid/workspace/huangyuming/TWIST2/TWIST2_full_npz/AMASS_g1_GMR8/ACCAD_Female1Running_c3d_C27_-_crouch_to_run1_stageii.npz"

CKPT_PATH="${1:-${DEFAULT_CKPT_PATH}}"
TASK_NAME="${2:-${DEFAULT_TASK}}"
EXPORT_OBS_ONLY="${EXPORT_OBS_ONLY:-1}" # 1: obs-only(play-style), 0: obs+time_step(motion export)
TARGET_CONDA_ENV="${TARGET_CONDA_ENV:-${DEFAULT_CONDA_ENV}}"
USE_CUDA_VISIBLE_DEVICES="${USE_CUDA_VISIBLE_DEVICES:-0}" # 1: use remapped ordinal mode (cuda:0), 0: use physical ordinal directly
USE_SAVED_OBS_PIPELINE="${USE_SAVED_OBS_PIPELINE:-1}" # 1: apply obs_pipeline from saved env.yaml
USER_MOTION_FILE_OVERRIDE="${MOTION_FILE_OVERRIDE:-}"
MOTION_FILE_OVERRIDE="${USER_MOTION_FILE_OVERRIDE:-${DEFAULT_MOTION_FILE_OVERRIDE}}" # optional single motion npz to avoid loading full motion library
ONNX_TIMEOUT_SECONDS="${ONNX_TIMEOUT_SECONDS:-240}" # default enabled to avoid stuck process

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "[ERROR] ckpt not found: ${CKPT_PATH}" >&2
  exit 1
fi

# Ensure we are in IsaacLab conda env when script is called directly.
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

# Keep runtime libs aligned with active conda env and system drivers.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi

detect_idle_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '{gsub(/ /,""); print $1" "$2" "$3}' \
    | sort -k2n -k3n \
    | awk 'NR==1{print $1}'
}

AUTO_DETECTED_GPU="$(detect_idle_gpu || true)"
CUDA_DEVICE="${CUDA_DEVICE:-${AUTO_DETECTED_GPU:-${DEFAULT_CUDA_DEVICE}}}"
CUDA_DEVICE="${CUDA_DEVICE%%,*}"

# Force non-interactive behavior so missing GUI dialog backends (e.g. zenity)
# don't block the workflow.
export HEADLESS=1
export LIVESTREAM=0
export ENABLE_CAMERAS=0

# Prefer NVIDIA ICD explicitly when present to avoid mixed ICD selection.
if [[ -f "/etc/vulkan/icd.d/nvidia_icd.json" ]]; then
  export VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json"
fi

if [[ -n "${MOTION_FILE_OVERRIDE}" && ! -f "${MOTION_FILE_OVERRIDE}" ]]; then
  if [[ -n "${USER_MOTION_FILE_OVERRIDE}" ]]; then
    echo "[ERROR] MOTION_FILE_OVERRIDE not found: ${MOTION_FILE_OVERRIDE}" >&2
    exit 1
  fi
  echo "[WARN] default motion file not found, fallback to saved run motion source." >&2
  MOTION_FILE_OVERRIDE=""
fi

if [[ "${USE_CUDA_VISIBLE_DEVICES}" == "1" ]]; then
  APP_DEVICE="cuda:0"
  ACTIVE_GPU="0"
  PHYSX_GPU="0"
  GPU_MODE_DESC="CUDA_VISIBLE_DEVICES remap mode"
else
  APP_DEVICE="cuda:${CUDA_DEVICE}"
  ACTIVE_GPU="${CUDA_DEVICE}"
  PHYSX_GPU="${CUDA_DEVICE}"
  GPU_MODE_DESC="physical GPU ordinal mode (recommended for IsaacSim)"
fi

KIT_ARGS="--/app/hangDetector/enabled=false --/app/hangDetector/timeout=3600 --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1 --/renderer/activeGpu=${ACTIVE_GPU} --/physics/cudaDevice=${PHYSX_GPU}"

echo "[INFO] ckpt: ${CKPT_PATH}"
echo "[INFO] task: ${TASK_NAME}"
echo "[INFO] export_obs_only: ${EXPORT_OBS_ONLY}"
echo "[INFO] conda_env: ${CONDA_DEFAULT_ENV:-<none>}"
echo "[INFO] cuda_device (physical): ${CUDA_DEVICE}"
echo "[INFO] gpu_mode: ${GPU_MODE_DESC}"
echo "[INFO] app_device: ${APP_DEVICE} | activeGpu=${ACTIVE_GPU} | physics.cudaDevice=${PHYSX_GPU}"
echo "[INFO] livestream: 0 (disabled)"
if [[ "${ONNX_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]] && (( ONNX_TIMEOUT_SECONDS > 0 )); then
  echo "[INFO] timeout: ${ONNX_TIMEOUT_SECONDS}s"
fi
if [[ -n "${MOTION_FILE_OVERRIDE}" ]]; then
  echo "[INFO] motion_file_override: ${MOTION_FILE_OVERRIDE}"
fi
echo "[INFO] running ONNX export..."

# Fast-fail when current CUDA visibility is invalid for the selected index/list.
set +e
if [[ "${USE_CUDA_VISIBLE_DEVICES}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python -c 'import sys
try:
    import torch
except Exception as error:
    print(f"[ERROR] Python env check failed: {error}")
    sys.exit(3)
n = torch.cuda.device_count()
print(f"[INFO] torch visible cuda devices: {n}")
sys.exit(0 if n > 0 else 2)'
else
  python -c "import sys
try:
    import torch
except Exception as error:
    print(f'[ERROR] Python env check failed: {error}')
    sys.exit(3)
n = torch.cuda.device_count()
idx = int('${CUDA_DEVICE}')
print(f'[INFO] torch visible cuda devices: {n}')
if n <= idx:
    print(f'[ERROR] requested physical cuda index {idx} out of range (0..{max(n - 1, -1)})')
    sys.exit(2)
sys.exit(0)"
fi
CUDA_CHECK_RC=$?
set -e
if [[ ${CUDA_CHECK_RC} -ne 0 ]]; then
  if [[ ${CUDA_CHECK_RC} -eq 3 ]]; then
    echo "[ERROR] Python env is missing torch or not activated. Try: conda activate env_isaaclab" >&2
  else
    echo "[ERROR] No CUDA device visible with CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}." >&2
    echo "[ERROR] Try: CUDA_DEVICE=<valid_gpu_index> bash to_onnx.sh" >&2
  fi
  exit 2
fi

EXTRA_EXPORT_ARGS=()
if [[ "${EXPORT_OBS_ONLY}" == "1" ]]; then
  EXTRA_EXPORT_ARGS+=(--obs_only)
fi

CMD=(
  python scripts/rsl_rl/save_onnx.py
  --task "${TASK_NAME}"
  --ckpt_path "${CKPT_PATH}"
  --headless
  --livestream 0
  --device "${APP_DEVICE}"
  --kit_args "${KIT_ARGS}"
  "${EXTRA_EXPORT_ARGS[@]}"
)

if [[ "${USE_SAVED_OBS_PIPELINE}" == "1" ]]; then
  CMD+=(--use_saved_obs_pipeline)
fi
if [[ -n "${MOTION_FILE_OVERRIDE}" ]]; then
  CMD+=(--motion_file "${MOTION_FILE_OVERRIDE}")
fi

if [[ "${ONNX_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]] && (( ONNX_TIMEOUT_SECONDS > 0 )); then
  CMD=(timeout --foreground "${ONNX_TIMEOUT_SECONDS}" "${CMD[@]}")
fi

set +e
if [[ "${USE_CUDA_VISIBLE_DEVICES}" == "1" ]]; then
  HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${CMD[@]}"
else
  HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 "${CMD[@]}"
fi
RC=$?
set -e

find_latest_kit_log() {
  local root=""
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    if [[ -d "${CONDA_PREFIX}/lib/python3.11/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim" ]]; then
      root="${CONDA_PREFIX}/lib/python3.11/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim"
    elif [[ -d "${CONDA_PREFIX}/lib/python3.10/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim" ]]; then
      root="${CONDA_PREFIX}/lib/python3.10/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim"
    fi
  fi
  if [[ -z "${root}" && -d "${HOME}/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim" ]]; then
    root="${HOME}/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim"
  fi
  if [[ -z "${root}" ]]; then
    return 1
  fi
  ls -1t "${root}"/*/kit_*.log 2>/dev/null | head -n 1
}

LATEST_LOG="$(find_latest_kit_log || true)"
if [[ -n "${LATEST_LOG}" ]]; then
  echo "[INFO] latest Kit log: ${LATEST_LOG}"
  echo "[INFO] key diagnostics:"
  rg -n "Hang detected|Showing message box|\\[Error\\]|CUDA being in bad state|Multiple Installable Client Drivers|Failed to create any GPU" "${LATEST_LOG}" | tail -n 40 || true
fi

if [[ ${RC} -ne 0 ]]; then
  if [[ ${RC} -eq 124 ]]; then
    echo "[ERROR] ONNX export timed out after ${ONNX_TIMEOUT_SECONDS}s" >&2
    exit "${RC}"
  fi
  echo "[ERROR] ONNX export failed (exit code: ${RC})" >&2
  exit "${RC}"
fi

echo "[INFO] ONNX export finished."
