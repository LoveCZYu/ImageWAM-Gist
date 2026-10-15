#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

GPU_PER_NODE=16
export ZERO_STAGE="${ZERO_STAGE:-3}"
if [[ "${ZERO_STAGE}" != "3" ]]; then
  echo "Full-episode DLC MemoryWAM training requires ZERO_STAGE=3; got ${ZERO_STAGE}." >&2
  exit 2
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -d /mnt/workspace/xcx/ImageWAM ]]; then
  ASSET_ROOT="${ASSET_ROOT:-/mnt/workspace/xcx/ImageWAM}"
elif [[ -d /mnt/data/xcx/ImageWAM ]]; then
  ASSET_ROOT="${ASSET_ROOT:-/mnt/data/xcx/ImageWAM}"
else
  echo "Cannot find the shared ImageWAM assets under /mnt/workspace/xcx or /mnt/data/xcx." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
FLUX2_SRC="${FLUX2_SRC:-${ASSET_ROOT}/third_party/flux2}"
MODEL_ROOT="${MODEL_ROOT:-${ASSET_ROOT}/checkpoints}"
FLUX2_MODEL_PATH="${FLUX2_MODEL_PATH:-${MODEL_ROOT}/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors}"
FLUX2_AE_MODEL_PATH="${FLUX2_AE_MODEL_PATH:-${MODEL_ROOT}/flux2/FLUX.2-dev/ae.safetensors}"
ACTION_INIT="${ACTION_INIT:-${MODEL_ROOT}/action_dit_flux2_4b_rmbench_init.pt}"
FLUX2_QWEN3_MODEL_SPEC="${FLUX2_QWEN3_MODEL_SPEC:-${MODEL_ROOT}/Qwen3-4B-Instruct-2507}"
RMBENCH_LEROBOT_ROOT="${RMBENCH_LEROBOT_ROOT:-${ASSET_ROOT}/data/rmbench_lerobot/rmbench_official9_50demo}"
QWEN_CACHE_DIR="${QWEN_CACHE_DIR:-${ASSET_ROOT}/data/rmbench_lerobot/flux2_qwen3_cache_4b_official9}"
NORM_STATS_PATH="${NORM_STATS_PATH:-${RMBENCH_LEROBOT_ROOT}/dataset_stats_memorywam.json}"

export PYTHON_BIN FLUX2_SRC FLUX2_MODEL_PATH FLUX2_AE_MODEL_PATH
export IMAGEWAM_RMBENCH_QWEN_CACHE="${QWEN_CACHE_DIR}"
export IMAGEWAM_REPLAN_STEPS="${REPLAN_STEPS:-16}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONPATH="${REPO_ROOT}/src:${FLUX2_SRC}/src:${FLUX2_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export RUN_ID="${RUN_ID:-memorywam_from_flux2_fixed8_full_episode_ar_dlc16_zero${ZERO_STAGE}_$(date +%Y%m%d_%H%M%S)}"

HISTORY_MAX_STEPS="${HISTORY_MAX_STEPS:-0}"
HISTORY_STEPS_PER_UPDATE="${HISTORY_STEPS_PER_UPDATE:-4}"
if [[ "${HISTORY_MAX_STEPS}" -ne 0 || "${HISTORY_STEPS_PER_UPDATE}" -lt 1 ]]; then
  echo "MemoryWAM DLC training requires full history replay and at least one trainable history step." >&2
  exit 2
fi

for required in "${PYTHON_BIN}" "${FLUX2_MODEL_PATH}" "${FLUX2_AE_MODEL_PATH}" "${ACTION_INIT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required DLC asset is unavailable: ${required}" >&2
    exit 2
  fi
done
if [[ ! -f "${NORM_STATS_PATH}" ]]; then
  echo "Precomputed normalization stats are unavailable: ${NORM_STATS_PATH}" >&2
  echo "Generate them once before launching distributed training." >&2
  exit 2
fi
"${PYTHON_BIN}" -m json.tool "${NORM_STATS_PATH}" >/dev/null
for required_dir in "${FLUX2_SRC}/src/flux2" "${FLUX2_QWEN3_MODEL_SPEC}" "${RMBENCH_LEROBOT_ROOT}" "${QWEN_CACHE_DIR}"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "Required DLC directory is unavailable: ${required_dir}" >&2
    exit 2
  fi
done

gpu_info="$(${PYTHON_BIN} -c 'import torch; n=torch.cuda.device_count(); m=[torch.cuda.get_device_properties(i).total_memory//(1024*1024) for i in range(n)]; print(n, min(m, default=0), ",".join(map(str,m)))')"
read -r gpu_count min_memory_mib gpu_memory_list <<< "${gpu_info}"
if [[ "${gpu_count}" -ne 16 || "${min_memory_mib}" -lt "${MIN_GPU_MEMORY_MIB:-61440}" ]]; then
  echo "Exactly 16 GPUs with at least ${MIN_GPU_MEMORY_MIB:-61440} MiB each are required; detected count=${gpu_count} memory_mib=[${gpu_memory_list}]." >&2
  exit 3
fi

imagewam_run imagewam_python scripts/flux2/validate_memorywam_from_flux2_assets.py \
  --flux2 "${FLUX2_MODEL_PATH}" \
  --ae "${FLUX2_AE_MODEL_PATH}" \
  --action-init "${ACTION_INIT}" \
  --dataset "${RMBENCH_LEROBOT_ROOT}" \
  --qwen-cache "${QWEN_CACHE_DIR}" \
  --context-len 512

imagewam_print_config GPU_PER_NODE ZERO_STAGE RUN_ID ASSET_ROOT PYTHON_BIN FLUX2_SRC \
  FLUX2_MODEL_PATH FLUX2_AE_MODEL_PATH ACTION_INIT FLUX2_QWEN3_MODEL_SPEC \
  RMBENCH_LEROBOT_ROOT QWEN_CACHE_DIR NORM_STATS_PATH IMAGEWAM_REPLAN_STEPS \
  CUDA_VISIBLE_DEVICES

export TASK="rmbench_flux2_klein_4b_base_memorywam_from_flux2"
imagewam_run bash scripts/flux2/train_flux2_klein_imagewam.sh 16 \
  memory=fixed \
  data.train.dataset_dirs="[${RMBENCH_LEROBOT_ROOT}]" \
  data.val.dataset_dirs="[${RMBENCH_LEROBOT_ROOT}]" \
  data.train.qwen_text_cache_dir="${QWEN_CACHE_DIR}" \
  data.val.qwen_text_cache_dir="${QWEN_CACHE_DIR}" \
  data.train.qwen_context_len=512 \
  data.val.qwen_context_len=512 \
  data.train.pretrained_norm_stats="${NORM_STATS_PATH}" \
  data.val.pretrained_norm_stats="${NORM_STATS_PATH}" \
  data.train.edit_memory_history_max_steps=0 \
  data.val.edit_memory_history_max_steps=0 \
  data.train.edit_memory_episode_terminal_samples_only=true \
  data.val.edit_memory_episode_terminal_samples_only=true \
  model.flux2_src_path="${FLUX2_SRC}" \
  model.flux2_model_path="${FLUX2_MODEL_PATH}" \
  model.ae_model_path="${FLUX2_AE_MODEL_PATH}" \
  model.qwen3_model_spec="${FLUX2_QWEN3_MODEL_SPEC}" \
  model.qwen_context_len=512 \
  model.action_dit_pretrained_path="${ACTION_INIT}" \
  model.flux2_lora_config.enabled=false \
  model.flux2_lora_config.save_trainable_only=false \
  model.edit_memory_config.training.history_steps_per_update="${HISTORY_STEPS_PER_UPDATE}" \
  batch_size="${BATCH_SIZE:-1}" \
  gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-1}" \
  num_workers="${NUM_WORKERS:-4}" \
  num_epochs="${NUM_EPOCHS:-5}" \
  learning_rate="${LEARNING_RATE:-2e-4}" \
  save_every="${SAVE_EVERY:-50}" \
  eval_every=0 \
  resume=null \
  wandb.enabled="${WANDB_ENABLED:-false}" \
  "$@"
