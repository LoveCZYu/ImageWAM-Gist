#!/usr/bin/env bash
set -euo pipefail

# Full-FLUX KV8 episode-TBPTT trial for the 8xA800 host.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GPU_PER_NODE=8
export ZERO_STAGE=2
export ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-scripts/accelerate_configs/accelerate_zero2_memorywam_ds.yaml}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=1
export TBPTT_STEPS=4
export FLUX2_LORA_ENABLED="${FLUX2_LORA_ENABLED:-false}"
export ASSET_ROOT="${ASSET_ROOT:-/home/xucx2606/ImageWAM}"
export PYTHON_BIN="${PYTHON_BIN:-${ASSET_ROOT}/.venv/bin/python}"
export VLM_CACHE="${VLM_CACHE:-/data41/xucx2606/vlm_semantic_labels_20260724/qwen3vl8b_full/semantic_change_curve_full_8gpu.pt}"
export NORM_STATS_PATH="${NORM_STATS_PATH:-${ASSET_ROOT}/runs/rmbench_flux2_klein_4b_base_imagewam/2026-06-26_19-31-23/dataset_stats.json}"
export FLUX2_QWEN3_MODEL_SPEC="${FLUX2_QWEN3_MODEL_SPEC:-/home/xucx2606/models/Qwen3-4B-Instruct-2507}"
export WANDB_ENABLED="${WANDB_ENABLED:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
flux_train_mode="fullflux"
if [[ "${FLUX2_LORA_ENABLED}" == "true" ]]; then
  flux_train_mode="fluxlora"
fi
export RUN_ID="${RUN_ID:-memorywam_kv8_episode_tbptt4_${flux_train_mode}_8gpu_zero2_bs1_ep5_$(date +%Y%m%d_%H%M%S)}"

exec bash "${SCRIPT_DIR}/run_train_memorywam_softk_vlm_rmbench_16gpu_dlc.sh" "$@"
