#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

GPU_PER_NODE="${GPU_PER_NODE:-8}"
TASK_TYPE="${TASK_TYPE:-libero}"
OVIS_U1_MODEL_PATH="${OVIS_U1_MODEL_PATH:-AIDC-AI/Ovis-U1-3B}"
imagewam_require_env DATA_ROOT

case "${TASK_TYPE}" in
  libero)
    ACTION_DIM=7
    ACTION_INIT="${ACTION_INIT:-checkpoints/action_dit_ovis_u1_libero_init.pt}"
    TASK_NAME="libero_ovis_u1_imagewam"
    DATASET_OVERRIDES=(
      "data.train.dataset_dirs=[${DATA_ROOT}/libero_spatial_no_noops_lerobot,${DATA_ROOT}/libero_object_no_noops_lerobot,${DATA_ROOT}/libero_goal_no_noops_lerobot,${DATA_ROOT}/libero_10_no_noops_lerobot]"
    )
    ;;
  robotwin)
    ACTION_DIM=14
    ACTION_INIT="${ACTION_INIT:-checkpoints/action_dit_ovis_u1_robotwin_init.pt}"
    TASK_NAME="robotwin_ovis_u1_imagewam"
    ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${DATA_ROOT}/robotwin2.0}"
    NONIDLE_FILTER_PATH="${NONIDLE_FILTER_PATH:-${ROBOTWIN_ROOT}/nonidle_ranges.json}"
    DATASET_OVERRIDES=(
      "data.train.dataset_dirs=[${ROBOTWIN_ROOT}]"
      "data.val.dataset_dirs=[${ROBOTWIN_ROOT}]"
      "data.train.nonidle_filter_path=${NONIDLE_FILTER_PATH}"
      "model.proprio_dim=14"
    )
    ;;
  *) echo "Invalid TASK_TYPE=${TASK_TYPE}; expected libero or robotwin" >&2; exit 1 ;;
esac

if [ "${REBUILD_ACTION_INIT:-false}" = "true" ] || [ ! -f "${ACTION_INIT}" ]; then
  export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/third_party${PYTHONPATH:+:${PYTHONPATH}}"
  imagewam_run imagewam_python scripts/ovis_u1/preprocess_action_dit_ovis_u1.py \
    --model-config configs/model/imagewam_ovis_u1.yaml \
    --ovis-u1-model-path "${OVIS_U1_MODEL_PATH}" \
    --action-dim "${ACTION_DIM}" \
    --output "${ACTION_INIT}"
fi

imagewam_print_config TASK_TYPE DATA_ROOT OVIS_U1_MODEL_PATH ACTION_INIT
TASK="${TASK_NAME}" imagewam_run bash scripts/ovis_u1/train_ovis_u1_imagewam.sh "${GPU_PER_NODE}" \
  "${DATASET_OVERRIDES[@]}" \
  model.action_dit_pretrained_path="${ACTION_INIT}" \
  "$@"
