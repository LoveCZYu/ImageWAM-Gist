#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/checkpoints}"
DIM_MODEL_PATH="${DIM_MODEL_PATH:-${MODEL_ROOT}/DIM/DIM-4.6B-Edit}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${MODEL_ROOT}/Qwen/Qwen2.5-VL-3B-Instruct}"

imagewam_print_config MODEL_ROOT DIM_MODEL_PATH QWEN_MODEL_PATH
cat <<'EOF'
DIM assets are not downloaded automatically by this helper.
Download them according to the upstream DIM/Qwen licenses, then rerun the training script with:
  DIM_MODEL_PATH=/path/to/DIM-4.6B-Edit
  QWEN_MODEL_PATH=/path/to/Qwen2.5-VL-3B-Instruct
EOF
