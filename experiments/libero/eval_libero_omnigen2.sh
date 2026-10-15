#!/usr/bin/env bash
set -euo pipefail

CKPT="${1:?Usage: bash experiments/libero/eval_libero_omnigen2.sh <checkpoint.pt> [hydra_overrides...]}"
shift

OMNIGEN2_SRC="/apdcephfs_nj7/share_305204761/alixzhang/omnigen2_ft/OmniGen2"
export PYTHONPATH="$(pwd)/src:${OMNIGEN2_SRC}:${PYTHONPATH:-}"

python3 experiments/libero/eval_libero_single.py \
  --config-name sim_libero_omnigen2 \
  "ckpt=${CKPT}" \
  "$@"
