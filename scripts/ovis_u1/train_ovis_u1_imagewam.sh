#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

TASK="${TASK:-robotwin_ovis_u1_imagewam}"
NPROC_PER_NODE="${1:?Usage: bash scripts/ovis_u1/train_ovis_u1_imagewam.sh <nproc_per_node> [hydra_overrides...]}"
shift

OVIS_U1_MODEL_PATH="${OVIS_U1_MODEL_PATH:-AIDC-AI/Ovis-U1-3B}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/third_party${PYTHONPATH:+:${PYTHONPATH}}"
export IMAGEWAM_MEM_TRIM_EVERY="${IMAGEWAM_MEM_TRIM_EVERY:-50}"
export IMAGEWAM_MEM_TRIM_GC="${IMAGEWAM_MEM_TRIM_GC:-1}"

imagewam_print_config TASK OVIS_U1_MODEL_PATH NPROC_PER_NODE
imagewam_run bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
  task="${TASK}" \
  model.ovis_u1_model_path="${OVIS_U1_MODEL_PATH}" \
  "$@"
