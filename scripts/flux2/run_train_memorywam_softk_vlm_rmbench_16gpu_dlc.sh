#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

GPU_PER_NODE="${GPU_PER_NODE:-16}"
# EATB carries detached episode memory across optimizer steps and works with ZeRO-2.
export ZERO_STAGE="${ZERO_STAGE:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${ASSET_ROOT:-}" && -d "${ASSET_ROOT}" ]]; then
  ASSET_ROOT="$(cd "${ASSET_ROOT}" && pwd)"
elif [[ -d /mnt/data/xcx/ImageWAM ]]; then
  ASSET_ROOT="${ASSET_ROOT:-/mnt/data/xcx/ImageWAM}"
elif [[ -d /mnt/workspace/xcx/ImageWAM ]]; then
  ASSET_ROOT="${ASSET_ROOT:-/mnt/workspace/xcx/ImageWAM}"
else
  echo "Cannot find ImageWAM assets under /mnt/workspace/xcx or /mnt/data/xcx." >&2
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
VLM_CACHE="${VLM_CACHE:-/mnt/data/xcx/memoryWAM/data/vlm_semantic_labels_20260724/qwen3vl8b_full/semantic_change_curve_full_8gpu.pt}"
EXPECTED_VLM_SHA256="${EXPECTED_VLM_SHA256:-f9f9ce431fc856f43c416dd01ff6baae10638d84cc229beed1be90a93d463042}"
TBPTT_STEPS="${TBPTT_STEPS:-4}"
MEMORY_SOURCE="${MEMORY_SOURCE:-eatb}"
FLUX2_LORA_ENABLED="${FLUX2_LORA_ENABLED:-false}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
ADAPTIVE_GIST_ENABLED="${ADAPTIVE_GIST_ENABLED:-true}"
FIXED_GIST_WRITE_HEAD="${FIXED_GIST_WRITE_HEAD:-false}"
VLM_COUNT_SUPERVISION_ENABLED="${VLM_COUNT_SUPERVISION_ENABLED:-true}"
VLM_WRITE_ONLY="${VLM_WRITE_ONLY:-false}"
MEMORY_CURRICULUM_ENABLED="${MEMORY_CURRICULUM_ENABLED:-true}"
TEACHER_ONLY_UNTIL_STEP="${TEACHER_ONLY_UNTIL_STEP:-28}"
TEACHER_TRANSITION_START_STEP="${TEACHER_TRANSITION_START_STEP:-29}"
TEACHER_TRANSITION_END_STEP="${TEACHER_TRANSITION_END_STEP:-114}"

export PYTHON_BIN FLUX2_SRC FLUX2_MODEL_PATH FLUX2_AE_MODEL_PATH
export IMAGEWAM_RMBENCH_QWEN_CACHE="${QWEN_CACHE_DIR}"
export IMAGEWAM_VLM_SEMANTIC_CACHE="${VLM_CACHE}"
export IMAGEWAM_REPLAN_STEPS="${REPLAN_STEPS:-16}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONPATH="${REPO_ROOT}/src:${FLUX2_SRC}/src:${FLUX2_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((GPU_PER_NODE - 1))")"
fi
export CUDA_VISIBLE_DEVICES
flux_train_mode="fullflux"
if [[ "${FLUX2_LORA_ENABLED}" == "true" ]]; then
  flux_train_mode="fluxlora"
fi
export RUN_ID="${RUN_ID:-memorywam_kv8_${MEMORY_SOURCE}_${flux_train_mode}_${GPU_PER_NODE}gpu_zero${ZERO_STAGE}_$(date +%Y%m%d_%H%M%S)}"
OSS_OUTPUT_ROOT="${OSS_OUTPUT_ROOT:-/mnt/oss_models/models_train/motus_robochallenge}"
OUTPUT_DIR="${OUTPUT_DIR:-${OSS_OUTPUT_ROOT}/fastwam_runs/rmbench_flux2_klein_4b_base_memorywam_from_flux2/${RUN_ID}}"
export OSS_OUTPUT_ROOT OUTPUT_DIR

if ! mountpoint -q /mnt/oss_models; then
  echo "OSS checkpoint mount is unavailable: /mnt/oss_models" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}"
if [[ ! -w "${OUTPUT_DIR}" ]]; then
  echo "OSS output directory is not writable: ${OUTPUT_DIR}" >&2
  exit 2
fi

for required in "${PYTHON_BIN}" "${FLUX2_MODEL_PATH}" "${FLUX2_AE_MODEL_PATH}" "${ACTION_INIT}" "${NORM_STATS_PATH}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required DLC asset is unavailable: ${required}" >&2
    exit 2
  fi
done
for required_dir in "${FLUX2_SRC}/src/flux2" "${FLUX2_QWEN3_MODEL_SPEC}" "${RMBENCH_LEROBOT_ROOT}" "${QWEN_CACHE_DIR}"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "Required DLC directory is unavailable: ${required_dir}" >&2
    exit 2
  fi
done
"${PYTHON_BIN}" -m json.tool "${NORM_STATS_PATH}" >/dev/null

if [[ "${VLM_COUNT_SUPERVISION_ENABLED}" == "true" ]]; then
  if [[ ! -e "${VLM_CACHE}" ]]; then
    echo "Required VLM cache is unavailable: ${VLM_CACHE}" >&2
    exit 2
  fi
  actual_vlm_sha256="$(sha256sum "${VLM_CACHE}" | awk '{print $1}')"
  if [[ "${actual_vlm_sha256}" != "${EXPECTED_VLM_SHA256}" ]]; then
    echo "VLM cache SHA256 mismatch: expected=${EXPECTED_VLM_SHA256} actual=${actual_vlm_sha256}" >&2
    exit 2
  fi
fi

if [[ "${ADAPTIVE_GIST_ENABLED}" == "true" ]]; then
  imagewam_run imagewam_python scripts/flux2/preflight_memorywam_softk_dlc.py \
    --cache "${VLM_CACHE}" \
    --dataset "${RMBENCH_LEROBOT_ROOT}" \
    --configs "${REPO_ROOT}/configs" \
    --memory-source "${MEMORY_SOURCE}"
else
  imagewam_run imagewam_python scripts/flux2/preflight_memorywam_fixed8_dlc.py \
    --cache "${VLM_CACHE}" \
    --dataset "${RMBENCH_LEROBOT_ROOT}" \
    --configs "${REPO_ROOT}/configs" \
    --gpus "${GPU_PER_NODE}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${NUM_EPOCHS}" \
    --tbptt-steps "${TBPTT_STEPS}" \
    --memory-source "${MEMORY_SOURCE}"
fi

gpu_info="$(${PYTHON_BIN} -c 'import torch; n=torch.cuda.device_count(); m=[torch.cuda.get_device_properties(i).total_memory//(1024*1024) for i in range(n)]; print(n, min(m, default=0), ",".join(map(str,m)))')"
read -r gpu_count min_memory_mib gpu_memory_list <<< "${gpu_info}"
if [[ "${gpu_count}" -ne "${GPU_PER_NODE}" || "${min_memory_mib}" -lt "${MIN_GPU_MEMORY_MIB:-61440}" ]]; then
  echo "Exactly ${GPU_PER_NODE} GPUs with at least ${MIN_GPU_MEMORY_MIB:-61440} MiB each are required; detected count=${gpu_count} memory_mib=[${gpu_memory_list}]." >&2
  exit 3
fi

imagewam_run imagewam_python scripts/flux2/validate_memorywam_from_flux2_assets.py \
  --flux2 "${FLUX2_MODEL_PATH}" \
  --ae "${FLUX2_AE_MODEL_PATH}" \
  --action-init "${ACTION_INIT}" \
  --dataset "${RMBENCH_LEROBOT_ROOT}" \
  --qwen-cache "${QWEN_CACHE_DIR}" \
  --context-len 512

imagewam_print_config GPU_PER_NODE ZERO_STAGE RUN_ID ASSET_ROOT FLUX2_MODEL_PATH \
  ACTION_INIT RMBENCH_LEROBOT_ROOT QWEN_CACHE_DIR NORM_STATS_PATH VLM_CACHE \
  IMAGEWAM_REPLAN_STEPS MEMORY_SOURCE TBPTT_STEPS FLUX2_LORA_ENABLED BATCH_SIZE \
  GRADIENT_ACCUMULATION_STEPS NUM_EPOCHS ADAPTIVE_GIST_ENABLED FIXED_GIST_WRITE_HEAD \
  VLM_COUNT_SUPERVISION_ENABLED VLM_WRITE_ONLY MEMORY_CURRICULUM_ENABLED \
  CUDA_VISIBLE_DEVICES OSS_OUTPUT_ROOT OUTPUT_DIR

export TASK="rmbench_flux2_klein_4b_base_memorywam_from_flux2"
imagewam_run bash scripts/flux2/train_flux2_klein_imagewam.sh "${GPU_PER_NODE}" \
  memory=adaptive_gist_vlm \
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
  data.train.edit_memory_drop_incomplete_current=true \
  data.val.edit_memory_drop_incomplete_current=true \
  data.train.edit_memory_episode_terminal_samples_only=true \
  data.val.edit_memory_episode_terminal_samples_only=true \
  model.flux2_src_path="${FLUX2_SRC}" \
  model.flux2_model_path="${FLUX2_MODEL_PATH}" \
  model.ae_model_path="${FLUX2_AE_MODEL_PATH}" \
  model.qwen3_model_spec="${FLUX2_QWEN3_MODEL_SPEC}" \
  model.qwen_context_len=512 \
  model.action_dit_pretrained_path="${ACTION_INIT}" \
  model.flux2_lora_config.enabled="${FLUX2_LORA_ENABLED}" \
  model.flux2_lora_config.save_trainable_only=false \
  model.edit_memory_config.adaptive_gist.enabled="${ADAPTIVE_GIST_ENABLED}" \
  model.edit_memory_config.fixed_gist_write_head="${FIXED_GIST_WRITE_HEAD}" \
  model.edit_memory_config.vlm_count_supervision.enabled="${VLM_COUNT_SUPERVISION_ENABLED}" \
  model.edit_memory_config.vlm_count_supervision.write_only="${VLM_WRITE_ONLY}" \
  model.edit_memory_config.training.memory_source="${MEMORY_SOURCE}" \
  model.edit_memory_config.training.replan_steps="${IMAGEWAM_REPLAN_STEPS}" \
  model.edit_memory_config.training.tbptt_steps="${TBPTT_STEPS}" \
  model.edit_memory_config.training.memory_curriculum.enabled="${MEMORY_CURRICULUM_ENABLED}" \
  model.edit_memory_config.training.memory_curriculum.teacher_only_until_step="${TEACHER_ONLY_UNTIL_STEP}" \
  model.edit_memory_config.training.memory_curriculum.transition_start_step="${TEACHER_TRANSITION_START_STEP}" \
  model.edit_memory_config.training.memory_curriculum.transition_end_step="${TEACHER_TRANSITION_END_STEP}" \
  batch_size="${BATCH_SIZE}" \
  gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
  num_workers="${NUM_WORKERS:-4}" \
  num_epochs="${NUM_EPOCHS}" \
  learning_rate="${LEARNING_RATE:-2e-4}" \
  save_every="${SAVE_EVERY:-1000}" \
  eval_every=0 \
  resume=null \
  output_dir="${OUTPUT_DIR}" \
  wandb.enabled="${WANDB_ENABLED:-false}" \
  "$@"
