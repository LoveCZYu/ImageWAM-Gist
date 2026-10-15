#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

TASK="${TASK:-libero_flux2_klein_4b_base_imagewam}"
NPROC_PER_NODE="${1:?Usage: bash scripts/flux2/train_flux2_klein_imagewam.sh <nproc_per_node> [hydra_overrides...]}"
shift

imagewam_require_env FLUX2_SRC
ZERO_STAGE="${ZERO_STAGE:-1}"
export PYTHONPATH="${REPO_ROOT}/src:${FLUX2_SRC}/src:${FLUX2_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
export IMAGEWAM_MEM_TRIM_EVERY="${IMAGEWAM_MEM_TRIM_EVERY:-50}"
export IMAGEWAM_MEM_TRIM_GC="${IMAGEWAM_MEM_TRIM_GC:-1}"

case "${ZERO_STAGE}" in
  1|zero1) TRAIN_SCRIPT="scripts/train_zero1.sh" ;;
  2|zero2) TRAIN_SCRIPT="scripts/train_zero2.sh" ;;
  3|zero3) TRAIN_SCRIPT="scripts/train_zero3.sh" ;;
  *) echo "Invalid ZERO_STAGE=${ZERO_STAGE}; expected 1, zero1, 2, zero2, 3, or zero3" >&2; exit 1 ;;
esac

imagewam_print_config TASK ZERO_STAGE FLUX2_SRC NPROC_PER_NODE
imagewam_run bash "${TRAIN_SCRIPT}" "${NPROC_PER_NODE}" \
  task="${TASK}" \
  model.flux2_src_path="${FLUX2_SRC}" \
  "$@"
