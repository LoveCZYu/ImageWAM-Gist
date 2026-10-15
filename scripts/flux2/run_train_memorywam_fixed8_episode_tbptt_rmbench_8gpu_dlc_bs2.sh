#!/usr/bin/env bash
set -euo pipefail

# Fixed-8 gist KV MemoryWAM with EATB K=8. The memory bank stays chronological
# across optimizer steps and is detached only at EATB segment boundaries.
# Raw replans 0/1 and the latest four replans keep full K/V; every other replan
# keeps exactly eight gist tokens.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DLC_ROOT="${DLC_ROOT:-/mnt/workspace/xcx}"

export GPU_PER_NODE=8
export ZERO_STAGE=3
export ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-scripts/accelerate_configs/accelerate_zero3_memorywam_cpu_ds.yaml}"
export BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=1
export NUM_EPOCHS="${NUM_EPOCHS:-10}"
export TBPTT_STEPS=8
export MEMORY_SOURCE=eatb
export ADAPTIVE_GIST_ENABLED=false
export FIXED_GIST_WRITE_HEAD=false
export VLM_COUNT_SUPERVISION_ENABLED=false
export VLM_WRITE_ONLY=false
export MEMORY_CURRICULUM_ENABLED=false
export TEACHER_ONLY_UNTIL_STEP=14
export TEACHER_TRANSITION_START_STEP=15
export TEACHER_TRANSITION_END_STEP=57
export FLUX2_LORA_ENABLED="${FLUX2_LORA_ENABLED:-false}"
export ASSET_ROOT="${ASSET_ROOT:-${DLC_ROOT}/ImageWAM}"
export PYTHON_BIN="${PYTHON_BIN:-${ASSET_ROOT}/.venv/bin/python}"
export VLM_CACHE="${VLM_CACHE:-${DLC_ROOT}/vlm_semantic_labels_20260724/qwen3vl8b_full/semantic_change_curve_full_8gpu.pt}"
export NORM_STATS_PATH="${NORM_STATS_PATH:-${ASSET_ROOT}/runs/rmbench_flux2_klein_4b_base_imagewam/2026-06-26_19-31-23/dataset_stats.json}"
export FLUX2_QWEN3_MODEL_SPEC="${FLUX2_QWEN3_MODEL_SPEC:-${ASSET_ROOT}/checkpoints/Qwen3-4B-Instruct-2507}"
export WANDB_ENABLED="${WANDB_ENABLED:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export RUN_ID="${RUN_ID:-memorywam_fixed8_eatb8_fullflux_zero3_8gpu_bs1_ep10_$(date +%Y%m%d_%H%M%S)}"

exec bash "${SCRIPT_DIR}/run_train_memorywam_softk_vlm_rmbench_16gpu_dlc.sh" "$@"
