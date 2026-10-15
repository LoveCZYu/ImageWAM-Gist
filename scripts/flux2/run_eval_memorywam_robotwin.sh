#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEMORY_MODE="${MEMORY_MODE:-fixed}" # fixed | adaptive_span | adaptive_gist | adaptive_gist_vlm_infer | adaptive_both
export IMAGEWAM_REPLAN_STEPS="${REPLAN_STEPS:-16}"
case "${MEMORY_MODE}" in
  fixed|adaptive_span|adaptive_gist|adaptive_gist_vlm_infer|adaptive_both) ;;
  *) echo "Invalid MEMORY_MODE=${MEMORY_MODE}" >&2; exit 1 ;;
esac

export TASK="robotwin_flux2_klein_4b_base_memorywam"
exec bash "${SCRIPT_DIR}/run_eval_flux2_robotwin.sh" memory="${MEMORY_MODE}" "$@"
