#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Usage:
#   bash play.sh [load_run] [checkpoint] [motion_file] [task] [num_envs]
#
# Example:
#   bash play.sh \
#     2026-03-27_02-24-00_20260326-Tracking-Flat-G1-v0 \
#     model_20000.pt \
#     /path/to/motion.npz \
#     Tracking-Flat-G1-v0 \
#     2

DEFAULT_LOAD_RUN="2026-03-27_02-24-00_20260326-Tracking-Flat-G1-v0"
DEFAULT_CHECKPOINT="model_20000.pt"
DEFAULT_MOTION_FILE="/media/raid/workspace/huangyuming/TWIST2/TWIST2_full_npz/AMASS_g1_GMR8/ACCAD_Female1General_c3d_A1_-_Stand_stageii.npz"
DEFAULT_TASK="Tracking-Flat-G1-v0"
DEFAULT_NUM_ENVS="2"

LOAD_RUN="${1:-${DEFAULT_LOAD_RUN}}"
CHECKPOINT="${2:-${DEFAULT_CHECKPOINT}}"
MOTION_FILE="${3:-${DEFAULT_MOTION_FILE}}"
TASK_NAME="${4:-${DEFAULT_TASK}}"
NUM_ENVS="${5:-${DEFAULT_NUM_ENVS}}"

# Optional env overrides.
CUDA_DEVICE="${CUDA_DEVICE:-0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}" # e.g. g1_flat

if [[ ! -f "${MOTION_FILE}" ]]; then
  echo "[ERROR] motion_file not found: ${MOTION_FILE}" >&2
  exit 1
fi

# Keep execution non-interactive.
export HEADLESS=1
export LIVESTREAM=2
export ENABLE_CAMERAS=0

# Prefer NVIDIA ICD explicitly when present to reduce mixed ICD issues.
if [[ -f "/etc/vulkan/icd.d/nvidia_icd.json" ]]; then
  export VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json"
fi

# Avoid GUI hang popup path (zenity) and reduce mgpu-related startup instability.
KIT_ARGS="--/app/hangDetector/enabled=false --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1"

echo "[INFO] load_run: ${LOAD_RUN}"
echo "[INFO] checkpoint: ${CHECKPOINT}"
echo "[INFO] motion_file: ${MOTION_FILE}"
echo "[INFO] task: ${TASK_NAME}"
echo "[INFO] num_envs: ${NUM_ENVS}"
echo "[INFO] cuda_device: ${CUDA_DEVICE}"
if [[ -n "${EXPERIMENT_NAME}" ]]; then
  echo "[INFO] experiment_name: ${EXPERIMENT_NAME}"
fi
echo "[INFO] livestream: 1 (WebRTC public mode)"

CMD=(
  python scripts/rsl_rl/play.py
  --task "${TASK_NAME}"
  --num_envs "${NUM_ENVS}"
  --load_run "${LOAD_RUN}"
  --checkpoint "${CHECKPOINT}"
  --motion_file "${MOTION_FILE}"
  --headless
  --livestream 2
  --kit_args "${KIT_ARGS}"
)

if [[ -n "${EXPERIMENT_NAME}" ]]; then
  CMD+=(--experiment_name "${EXPERIMENT_NAME}")
fi

set +e
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${CMD[@]}"
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
  rg -n "Hang detected|Showing message box|\\[Error\\]|CUDA being in bad state|Multiple Installable Client Drivers|Failed to create any GPU" "${LATEST_LOG}" | tail -n 60 || true
fi

if [[ ${RC} -ne 0 ]]; then
  echo "[ERROR] play failed (exit code: ${RC})" >&2
  exit "${RC}"
fi

echo "[INFO] play finished."
