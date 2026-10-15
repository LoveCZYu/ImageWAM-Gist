#!/usr/bin/env bash
set -euo pipefail

# Dedicated 16-GPU DLC entry point for the EventVLA-style memory training path:
#   - shared observation/language/proprio memory latent
#   - VLM hard + soft Adaptive-k supervision
#   - sample-level teacher-to-predict curriculum
#   - terminal episode samples with chronological KV8 replay
#   - TBPTT=4 with MemoryBank value preservation across detach boundaries
#
# Required DLC assets can be supplied through the same environment variables as
# run_train_memorywam_softk_vlm_rmbench_16gpu_dlc.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DLC_ROOT="${DLC_ROOT:-/mnt/workspace/xcx}"

export GPU_PER_NODE=16
export ZERO_STAGE="${ZERO_STAGE:-2}"
if [[ "${ZERO_STAGE}" == "2" ]]; then
  export ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-scripts/accelerate_configs/accelerate_zero2_memorywam_ds.yaml}"
fi
export BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=1
export TBPTT_STEPS=4
export FLUX2_LORA_ENABLED="${FLUX2_LORA_ENABLED:-false}"
export ASSET_ROOT="${ASSET_ROOT:-${DLC_ROOT}/ImageWAM}"
export PYTHON_BIN="${PYTHON_BIN:-${ASSET_ROOT}/.venv/bin/python}"
export VLM_CACHE="${VLM_CACHE:-${DLC_ROOT}/vlm_semantic_labels_20260724/qwen3vl8b_full/semantic_change_curve_full_8gpu.pt}"
export NORM_STATS_PATH="${NORM_STATS_PATH:-${ASSET_ROOT}/runs/rmbench_flux2_klein_4b_base_imagewam/2026-06-26_19-31-23/dataset_stats.json}"
export FLUX2_QWEN3_MODEL_SPEC="${FLUX2_QWEN3_MODEL_SPEC:-${ASSET_ROOT}/checkpoints/Qwen3-4B-Instruct-2507}"
export WANDB_ENABLED="${WANDB_ENABLED:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
flux_train_mode="fullflux"
if [[ "${FLUX2_LORA_ENABLED}" == "true" ]]; then
  flux_train_mode="fluxlora"
fi
export RUN_ID="${RUN_ID:-memorywam_kv8_episode_tbptt4_${flux_train_mode}_zero${ZERO_STAGE}_16gpu_bs1_ga1_ep5_$(date +%Y%m%d_%H%M%S)}"

exec bash "${SCRIPT_DIR}/run_train_memorywam_softk_vlm_rmbench_16gpu_dlc.sh" "$@"
