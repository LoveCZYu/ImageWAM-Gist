#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

GPU_PER_NODE="${GPU_PER_NODE:-16}"
MEMORY_MODE="${MEMORY_MODE:-fixed}" # fixed | adaptive_span | adaptive_gist | adaptive_both
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:?Set ROBOTWIN_ROOT to the LeRobot dataset root}"
QWEN_CACHE_DIR="${QWEN_CACHE_DIR:?Set QWEN_CACHE_DIR to the Qwen3 FLUX.2 cache}"
FLUX2_MODEL_PATH="${FLUX2_MODEL_PATH:?Set FLUX2_MODEL_PATH}"
FLUX2_AE_MODEL_PATH="${FLUX2_AE_MODEL_PATH:?Set FLUX2_AE_MODEL_PATH}"
ACTION_INIT="${ACTION_INIT:?Set ACTION_INIT to the Stage1/action checkpoint}"
RESUME="${RESUME:-null}"
export IMAGEWAM_REPLAN_STEPS="${REPLAN_STEPS:-16}"

case "${MEMORY_MODE}" in
  fixed|adaptive_span|adaptive_gist|adaptive_both) ;;
  *) echo "Invalid MEMORY_MODE=${MEMORY_MODE}" >&2; exit 1 ;;
esac

export TASK="robotwin_flux2_klein_4b_base_memorywam"
export ZERO_STAGE="${ZERO_STAGE:-1}"
imagewam_print_config TASK GPU_PER_NODE MEMORY_MODE IMAGEWAM_REPLAN_STEPS ROBOTWIN_ROOT QWEN_CACHE_DIR RESUME

imagewam_run bash scripts/flux2/train_flux2_klein_imagewam.sh "${GPU_PER_NODE}" \
  memory="${MEMORY_MODE}" \
  data.train.dataset_dirs="[${ROBOTWIN_ROOT}]" \
  data.val.dataset_dirs="[${ROBOTWIN_ROOT}]" \
  data.train.qwen_text_cache_dir="${QWEN_CACHE_DIR}" \
  data.val.qwen_text_cache_dir="${QWEN_CACHE_DIR}" \
  model.flux2_model_path="${FLUX2_MODEL_PATH}" \
  model.ae_model_path="${FLUX2_AE_MODEL_PATH}" \
  model.action_dit_pretrained_path="${ACTION_INIT}" \
  batch_size=1 \
  gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-2}" \
  num_workers="${NUM_WORKERS:-8}" \
  resume="${RESUME}" \
  "$@"
