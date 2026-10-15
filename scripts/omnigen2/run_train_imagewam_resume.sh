#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

GPU_PER_NODE="${GPU_PER_NODE:-8}"
TASK_TYPE="${TASK_TYPE:-robotwin}"
TASK="${TASK:-robotwin_omnigen2_imagewam}"
ACTION_INIT="${ACTION_INIT:-checkpoints/action_dit_omnigen2_init.pt}"

imagewam_require_env DATA_ROOT
imagewam_require_env OMNIGEN2_SRC
imagewam_require_env OMNIGEN2_MODEL_PATH
imagewam_require_env QWEN_MODEL_PATH
imagewam_require_env RESUME_PATH

ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${DATA_ROOT}/robotwin2.0}"
QWEN_CACHE_DIR="${QWEN_CACHE_DIR:-${ROBOTWIN_ROOT}/qwen_cache}"
NONIDLE_FILTER_PATH="${NONIDLE_FILTER_PATH:-${ROBOTWIN_ROOT}/nonidle_ranges.json}"

imagewam_print_config TASK RESUME_PATH ROBOTWIN_ROOT QWEN_CACHE_DIR NONIDLE_FILTER_PATH
TASK="${TASK}" imagewam_run bash scripts/omnigen2/train_omnigen2_imagewam.sh "${GPU_PER_NODE}" \
  model.omnigen2_model_path="${OMNIGEN2_MODEL_PATH}" \
  model.omnigen2_vae_path="${OMNIGEN2_MODEL_PATH}" \
  model.qwen_path="${QWEN_MODEL_PATH}" \
  data.train.dataset_dirs="[${ROBOTWIN_ROOT}]" \
  data.val.dataset_dirs="[${ROBOTWIN_ROOT}]" \
  data.train.qwen_text_cache_dir="${QWEN_CACHE_DIR}" \
  data.val.qwen_text_cache_dir="${QWEN_CACHE_DIR}" \
  data.train.nonidle_filter_path="${NONIDLE_FILTER_PATH}" \
  data.val.nonidle_filter_path="${NONIDLE_FILTER_PATH}" \
  model.action_dit_pretrained_path="${ACTION_INIT}" \
  model.proprio_dim=14 \
  resume="${RESUME_PATH}" \
  "$@"
