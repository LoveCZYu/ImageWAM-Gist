#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

imagewam_require_env ROBOTWIN_ROOT
NONIDLE_FILTER_PATH="${NONIDLE_FILTER_PATH:-${ROBOTWIN_ROOT}/nonidle_ranges.json}"

imagewam_print_config ROBOTWIN_ROOT NONIDLE_FILTER_PATH
imagewam_run imagewam_python scripts/data/compute_robotwin_nonidle_ranges.py \
  "${ROBOTWIN_ROOT}" \
  --output "${NONIDLE_FILTER_PATH}" \
  "$@"
