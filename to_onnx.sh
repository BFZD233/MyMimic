#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Usage:
#   bash to_onnx.sh [ckpt_path] [task]
#
# Defaults keep your current workflow unchanged.
DEFAULT_CKPT_PATH="/media/raid/workspace/huangyuming/lzd/whole_body_tracking/logs/rsl_rl/g1_flat/2026-03-27_02-24-00_20260326-Tracking-Flat-G1-v0/model_20000.pt"
DEFAULT_TASK="Tracking-Flat-G1-v0"

CKPT_PATH="${1:-${DEFAULT_CKPT_PATH}}"
TASK_NAME="${2:-${DEFAULT_TASK}}"
EXPORT_OBS_ONLY="${EXPORT_OBS_ONLY:-1}" # 1: obs-only(play-style), 0: obs+time_step(motion export)

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "[ERROR] ckpt not found: ${CKPT_PATH}" >&2
  exit 1
fi

# Force non-interactive behavior so missing GUI dialog backends (e.g. zenity)
# don't block the workflow.
export HEADLESS=1
export LIVESTREAM=0
export ENABLE_CAMERAS=0

# Prefer NVIDIA ICD explicitly when present to avoid mixed ICD selection.
if [[ -f "/etc/vulkan/icd.d/nvidia_icd.json" ]]; then
  export VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json"
fi

KIT_ARGS="--/app/hangDetector/enabled=false --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1"

echo "[INFO] ckpt: ${CKPT_PATH}"
echo "[INFO] task: ${TASK_NAME}"
echo "[INFO] export_obs_only: ${EXPORT_OBS_ONLY}"
echo "[INFO] running ONNX export..."

EXTRA_EXPORT_ARGS=()
if [[ "${EXPORT_OBS_ONLY}" == "1" ]]; then
  EXTRA_EXPORT_ARGS+=(--obs_only)
fi

set +e
CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/save_onnx.py \
  --task "${TASK_NAME}" \
  --ckpt_path "${CKPT_PATH}" \
  --headless \
  --livestream 0 \
  --device cuda:0 \
  --kit_args "${KIT_ARGS}" \
  "${EXTRA_EXPORT_ARGS[@]}"
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
  echo "[ERROR] ONNX export failed (exit code: ${RC})" >&2
  exit "${RC}"
fi

echo "[INFO] ONNX export finished."
