#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cache_path="${IMAGEWAM_VLM_SEMANTIC_CACHE:?IMAGEWAM_VLM_SEMANTIC_CACHE is required}"
if [[ ! -f "${cache_path}" ]]; then
  echo "VLM semantic cache does not exist: ${cache_path}" >&2
  exit 2
fi

export IMAGEWAM_VLM_SEMANTIC_CACHE="${cache_path}"
export RUN_ID="${RUN_ID:-memorywam_from_flux2_adaptive_vlm_$(date +%Y%m%d_%H%M%S)}"
exec bash "${SCRIPT_DIR}/run_train_memorywam_from_flux2_rmbench_8gpu.sh" \
  memory=adaptive_gist_vlm \
  data.train.edit_memory_drop_incomplete_current=true \
  data.val.edit_memory_drop_incomplete_current=true \
  "$@"
