#!/usr/bin/env bash
set -e -o pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "env_isaaclab" ]]; then
  if [[ -f "/home/huangyuming/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source /home/huangyuming/anaconda3/etc/profile.d/conda.sh
    conda activate env_isaaclab
  fi
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES=7 exec python scripts/batch_pkl_to_npz.py \
  --input_yaml config/twist2_dataset.yaml \
  --output_root /media/raid/workspace/huangyuming/TWIST2/TWIST2_full_npz \
  --output_fps 50 \
  --headless \
  --force
