#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

GPU_PER_NODE="${GPU_PER_NODE:-8}"
if [ "${GPU_PER_NODE}" -ne 8 ]; then
  echo "This entrypoint requires exactly 8 GPUs; got GPU_PER_NODE=${GPU_PER_NODE}." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/home/xucx2606/ImageWAM/.venv/bin/python}"
FLUX2_SRC="${FLUX2_SRC:-/home/xucx2606/ImageWAM/third_party/flux2}"
MODEL_ROOT="${MODEL_ROOT:-/home/xucx2606/ImageWAM/checkpoints}"
FLUX2_MODEL_PATH="${FLUX2_MODEL_PATH:-${MODEL_ROOT}/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors}"
FLUX2_AE_MODEL_PATH="${FLUX2_AE_MODEL_PATH:-${MODEL_ROOT}/flux2/FLUX.2-dev/ae.safetensors}"
ACTION_INIT="${ACTION_INIT:-${MODEL_ROOT}/action_dit_flux2_4b_rmbench_init.pt}"
FLUX2_QWEN3_MODEL_SPEC="${FLUX2_QWEN3_MODEL_SPEC:-/home/xucx2606/models/Qwen3-4B-Instruct-2507}"
RMBENCH_LEROBOT_ROOT="${RMBENCH_LEROBOT_ROOT:-/home/xucx2606/ImageWAM/data/rmbench_lerobot/rmbench_official9_50demo}"
QWEN_CACHE_DIR="${QWEN_CACHE_DIR:-/home/xucx2606/ImageWAM/data/rmbench_lerobot/flux2_qwen3_cache_4b_official9}"
export IMAGEWAM_RMBENCH_QWEN_CACHE="${QWEN_CACHE_DIR}"
export IMAGEWAM_REPLAN_STEPS="${REPLAN_STEPS:-16}"
export PYTHON_BIN FLUX2_SRC FLUX2_MODEL_PATH FLUX2_AE_MODEL_PATH
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONPATH="${REPO_ROOT}/src:${FLUX2_SRC}/src:${FLUX2_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export ZERO_STAGE="${ZERO_STAGE:-2}"
export RUN_ID="${RUN_ID:-memorywam_from_flux2_fixed8_$(date +%Y%m%d_%H%M%S)}"
HISTORY_MAX_STEPS="${HISTORY_MAX_STEPS:-16}"
HISTORY_STEPS_PER_UPDATE="${HISTORY_STEPS_PER_UPDATE:-1}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python environment is unavailable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [ ! -d "${FLUX2_SRC}/src/flux2" ]; then
  echo "FLUX.2 source checkout is unavailable: ${FLUX2_SRC}" >&2
  exit 2
fi

if [ ! -f "${ACTION_INIT}" ]; then
  imagewam_run imagewam_python scripts/flux2/preprocess_action_dit_flux2.py \
    --model-config configs/model/imagewam_flux2_klein_4b_base.yaml \
    --flux2-src-path "${FLUX2_SRC}" \
    --flux2-model-path "${FLUX2_MODEL_PATH}" \
    --variant klein-base-4b \
    --action-dim 14 \
    --output "${ACTION_INIT}" \
    --apply-alpha-scaling true
fi

imagewam_run imagewam_python scripts/flux2/validate_memorywam_from_flux2_assets.py \
  --flux2 "${FLUX2_MODEL_PATH}" \
  --ae "${FLUX2_AE_MODEL_PATH}" \
  --action-init "${ACTION_INIT}" \
  --dataset "${RMBENCH_LEROBOT_ROOT}" \
  --qwen-cache "${QWEN_CACHE_DIR}" \
  --context-len 512

mapfile -t GPU_FREE_MIB < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
if [ "${#GPU_FREE_MIB[@]}" -ne 8 ]; then
  echo "Expected exactly 8 visible GPUs, found ${#GPU_FREE_MIB[@]}." >&2
  exit 2
fi
MIN_FREE_MIB="${MIN_FREE_MIB:-50000}"
for index in "${!GPU_FREE_MIB[@]}"; do
  free_mib="${GPU_FREE_MIB[$index]//[[:space:]]/}"
  if [ "${free_mib}" -lt "${MIN_FREE_MIB}" ]; then
    echo "GPU ${index} has ${free_mib} MiB free; require at least ${MIN_FREE_MIB} MiB." >&2
    exit 3
  fi
done

imagewam_print_config GPU_PER_NODE ZERO_STAGE RUN_ID FLUX2_SRC FLUX2_MODEL_PATH \
  FLUX2_AE_MODEL_PATH ACTION_INIT FLUX2_QWEN3_MODEL_SPEC RMBENCH_LEROBOT_ROOT \
  QWEN_CACHE_DIR IMAGEWAM_REPLAN_STEPS CUDA_VISIBLE_DEVICES HISTORY_MAX_STEPS \
  HISTORY_STEPS_PER_UPDATE

export TASK="rmbench_flux2_klein_4b_base_memorywam_from_flux2"
imagewam_run bash scripts/flux2/train_flux2_klein_imagewam.sh "${GPU_PER_NODE}" \
  memory=fixed \
  data.train.dataset_dirs="[${RMBENCH_LEROBOT_ROOT}]" \
  data.val.dataset_dirs="[${RMBENCH_LEROBOT_ROOT}]" \
  data.train.qwen_text_cache_dir="${QWEN_CACHE_DIR}" \
  data.val.qwen_text_cache_dir="${QWEN_CACHE_DIR}" \
  data.train.qwen_context_len=512 \
  data.val.qwen_context_len=512 \
  data.train.edit_memory_history_max_steps="${HISTORY_MAX_STEPS}" \
  data.val.edit_memory_history_max_steps="${HISTORY_MAX_STEPS}" \
  model.flux2_src_path="${FLUX2_SRC}" \
  model.flux2_model_path="${FLUX2_MODEL_PATH}" \
  model.ae_model_path="${FLUX2_AE_MODEL_PATH}" \
  model.qwen3_model_spec="${FLUX2_QWEN3_MODEL_SPEC}" \
  model.qwen_context_len=512 \
  model.action_dit_pretrained_path="${ACTION_INIT}" \
  model.flux2_lora_config.enabled=true \
  model.flux2_lora_config.save_trainable_only=false \
  model.edit_memory_config.training.history_steps_per_update="${HISTORY_STEPS_PER_UPDATE}" \
  batch_size="${BATCH_SIZE:-1}" \
  gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-4}" \
  num_workers="${NUM_WORKERS:-8}" \
  num_epochs="${NUM_EPOCHS:-5}" \
  learning_rate="${LEARNING_RATE:-1e-4}" \
  save_every="${SAVE_EVERY:-500}" \
  eval_every=0 \
  resume=null \
  wandb.enabled="${WANDB_ENABLED:-false}" \
  "$@"
