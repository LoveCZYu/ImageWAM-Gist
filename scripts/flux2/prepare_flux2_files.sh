#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/checkpoints}"
FLUX2_ROOT="${FLUX2_ROOT:-${MODEL_ROOT}/flux2}"
DOWNLOAD_9B="${DOWNLOAD_9B:-true}"

FLUX2_4B_DIR="${FLUX2_4B_DIR:-${FLUX2_ROOT}/FLUX.2-klein-base-4B}"
FLUX2_9B_DIR="${FLUX2_9B_DIR:-${FLUX2_ROOT}/FLUX.2-klein-base-9B}"
FLUX2_AE_DIR="${FLUX2_AE_DIR:-${FLUX2_ROOT}/FLUX.2-dev}"

mkdir -p "${FLUX2_4B_DIR}" "${FLUX2_AE_DIR}"
imagewam_print_config FLUX2_4B_DIR FLUX2_AE_DIR FLUX2_9B_DIR DOWNLOAD_9B

imagewam_run huggingface-cli download black-forest-labs/FLUX.2-klein-base-4B \
  flux-2-klein-base-4b.safetensors \
  --local-dir "${FLUX2_4B_DIR}"

# Gated repo: requires Hugging Face access.
imagewam_run huggingface-cli download black-forest-labs/FLUX.2-dev \
  ae.safetensors \
  --local-dir "${FLUX2_AE_DIR}"

if [ "${DOWNLOAD_9B}" = "true" ]; then
  mkdir -p "${FLUX2_9B_DIR}"
  # Gated repo: requires Hugging Face access.
  imagewam_run huggingface-cli download black-forest-labs/FLUX.2-klein-base-9B \
    flux-2-klein-base-9b.safetensors \
    --local-dir "${FLUX2_9B_DIR}"
fi

cat <<EOF
Suggested environment variables:
  FLUX2_MODEL_PATH=${FLUX2_4B_DIR}/flux-2-klein-base-4b.safetensors
  FLUX2_AE_MODEL_PATH=${FLUX2_AE_DIR}/ae.safetensors
EOF
