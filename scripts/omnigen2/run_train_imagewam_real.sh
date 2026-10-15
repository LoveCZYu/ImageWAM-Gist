#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

GPU_PER_NODE="${GPU_PER_NODE:-8}"
TASK="${TASK:-hdf5_omnigen2_imagewam}"
QWEN_CACHE_DIR="${QWEN_CACHE_DIR:-${REPO_ROOT}/data/qwen_cache_hdf5}"
ACTION_INIT="${ACTION_INIT:-checkpoints/action_dit_omnigen2_init.pt}"

imagewam_require_env OMNIGEN2_SRC
imagewam_require_env OMNIGEN2_MODEL_PATH
imagewam_require_env QWEN_MODEL_PATH
export PYTHONPATH="${REPO_ROOT}/src:${OMNIGEN2_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

if [ "${REBUILD_ACTION_INIT:-false}" = "true" ] || [ ! -f "${ACTION_INIT}" ]; then
  imagewam_run imagewam_python scripts/omnigen2/preprocess_action_dit_omnigen2.py \
    --model-config configs/model/imagewam_omnigen2.yaml \
    --omnigen2-model-path "${OMNIGEN2_MODEL_PATH}" \
    --action-dim "${ACTION_DIM:-16}" \
    --output "${ACTION_INIT}"
fi

imagewam_print_config TASK OMNIGEN2_SRC OMNIGEN2_MODEL_PATH QWEN_MODEL_PATH QWEN_CACHE_DIR ACTION_INIT
TASK="${TASK}" imagewam_run bash scripts/omnigen2/train_omnigen2_imagewam.sh "${GPU_PER_NODE}" \
  model.omnigen2_model_path="${OMNIGEN2_MODEL_PATH}" \
  model.omnigen2_vae_path="${OMNIGEN2_MODEL_PATH}" \
  model.qwen_path="${QWEN_MODEL_PATH}" \
  data.train.qwen_text_cache_dir="${QWEN_CACHE_DIR}" \
  model.action_dit_pretrained_path="${ACTION_INIT}" \
  "$@"
