#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

TASK="${TASK:-libero_omnigen2_imagewam}"
NPROC_PER_NODE="${1:?Usage: bash scripts/omnigen2/train_omnigen2_imagewam.sh <nproc_per_node> [hydra_overrides...]}"
shift

imagewam_require_env OMNIGEN2_SRC
export PYTHONPATH="${REPO_ROOT}/src:${OMNIGEN2_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
export IMAGEWAM_MEM_TRIM_EVERY="${IMAGEWAM_MEM_TRIM_EVERY:-50}"
export IMAGEWAM_MEM_TRIM_GC="${IMAGEWAM_MEM_TRIM_GC:-1}"
export IMAGEWAM_DEBUG_OMNIGEN2_FORWARD_EVERY="${IMAGEWAM_DEBUG_OMNIGEN2_FORWARD_EVERY:-0}"

imagewam_print_config TASK OMNIGEN2_SRC NPROC_PER_NODE
imagewam_run bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
  task="${TASK}" \
  "$@"
