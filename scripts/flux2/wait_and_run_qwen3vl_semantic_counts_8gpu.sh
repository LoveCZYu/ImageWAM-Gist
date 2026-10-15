#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLL_SECONDS="${POLL_SECONDS:-60}"

while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]]; do
  sleep "${POLL_SECONDS}"
done

exec bash "${SCRIPT_DIR}/resume_qwen3vl_semantic_counts_8gpu.sh"
