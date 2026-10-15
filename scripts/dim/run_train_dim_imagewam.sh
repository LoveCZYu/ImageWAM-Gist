#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

GPU_PER_NODE="${GPU_PER_NODE:-8}"
TASK_TYPE="${TASK_TYPE:-libero}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/checkpoints}"

imagewam_require_env DATA_ROOT
imagewam_require_env DIM_SRC
DIM_MODEL_PATH="${DIM_MODEL_PATH:-${MODEL_ROOT}/DIM/DIM-4.6B-Edit}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${MODEL_ROOT}/Qwen/Qwen2.5-VL-3B-Instruct}"
SANA_CONFIG_PATH="${SANA_CONFIG_PATH:-${DIM_SRC}/models/sana1-5_config/1024ms/Sana_1600M_1024px_allqknorm_bf16_lr2e5_channel_cond.yaml}"

if [ "${TASK_TYPE}" != "libero" ]; then
  echo "DIM release script currently supports TASK_TYPE=libero only." >&2
  exit 1
fi

ACTION_INIT="${ACTION_INIT:-checkpoints/action_dit_dim_libero_init.pt}"
if [ "${REBUILD_DIM_ACTION_INIT:-false}" = "true" ] || [ ! -f "${ACTION_INIT}" ]; then
  export PYTHONPATH="${REPO_ROOT}/src:${DIM_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
  imagewam_run imagewam_python scripts/dim/preprocess_action_dit_dim.py \
    --model-config configs/model/imagewam_dim.yaml \
    --dim-model-path "${DIM_MODEL_PATH}" \
    --sana-config-path "${SANA_CONFIG_PATH}" \
    --action-dim "${ACTION_DIM:-7}" \
    --output "${ACTION_INIT}"
fi

imagewam_print_config DATA_ROOT DIM_SRC DIM_MODEL_PATH QWEN_MODEL_PATH SANA_CONFIG_PATH ACTION_INIT
TASK=libero_dim_imagewam imagewam_run bash scripts/dim/train_dim_imagewam.sh "${GPU_PER_NODE}" \
  model.dim_model_path="${DIM_MODEL_PATH}" \
  model.sana_config_path="${SANA_CONFIG_PATH}" \
  model.qwen_path="${QWEN_MODEL_PATH}" \
  data.train.dataset_dirs="[${DATA_ROOT}/libero_spatial_no_noops_lerobot,${DATA_ROOT}/libero_object_no_noops_lerobot,${DATA_ROOT}/libero_goal_no_noops_lerobot,${DATA_ROOT}/libero_10_no_noops_lerobot]" \
  model.action_dit_pretrained_path="${ACTION_INIT}" \
  "$@"
