#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

SUITE="libero_plus"
CONFIG_NAME="sim_libero_omnigen2"
TASK="${TASK:-libero_omnigen2_imagewam}"

imagewam_require_env OMNIGEN2_SRC
imagewam_require_env OMNIGEN2_MODEL_PATH
imagewam_require_env QWEN_MODEL_PATH
imagewam_ckpt_from_exp
imagewam_require_env CKPT_PATH
imagewam_require_env DATASET_STATS_PATH

export PYTHONPATH="${REPO_ROOT}/src:${OMNIGEN2_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
export WORKER_PYTHONPATH="${PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
LIBERO_WORKER_ENV_SOURCE="${LIBERO_WORKER_ENV_SOURCE:-}"
export LIBERO_WORKER_ENV_SOURCE
imagewam_prepare_eval_ckpt

COMMON=(
  --config-name "${CONFIG_NAME}"
  task="${TASK}"
  ckpt="${CKPT_PATH}"
  EVALUATION.dataset_stats_path="${DATASET_STATS_PATH}"
  model.omnigen2_model_path="${OMNIGEN2_MODEL_PATH}"
  model.omnigen2_vae_path="${OMNIGEN2_MODEL_PATH}"
  model.qwen_path="${QWEN_MODEL_PATH}"
  model.omnigen2_online_text_cache_compatible="${OMNIGEN2_ONLINE_TEXT_CACHE_COMPATIBLE:-true}"
  MULTIRUN.num_gpus="${NUM_GPUS:-8}"
  MULTIRUN.max_tasks_per_gpu="${MAX_TASKS_PER_GPU:-4}"
  EVALUATION.action_horizon="${ACTION_HORIZON:-16}"
  EVALUATION.replan_steps="${REPLAN_STEPS:-12}"
  model.proprio_dim="${PROPRIO_DIM:-8}"
)

COMMON+=(
  model.pack_proprio_after_text="${PACK_PROPRIO_AFTER_TEXT:-true}"
  MULTIRUN.task_sample_ratio="${TASK_SAMPLE_RATIO:-0.15}"
  MULTIRUN.task_suite_names="${TASK_SUITE_NAMES:-[libero_10,libero_goal,libero_spatial,libero_object]}"
  EVALUATION.num_trials="${NUM_TRIALS:-1}"
  MULTIRUN.chunk_size="${CHUNK_SIZE:-20}"
)

imagewam_print_config SUITE TASK CKPT_PATH DATASET_STATS_PATH OMNIGEN2_SRC OMNIGEN2_MODEL_PATH QWEN_MODEL_PATH
imagewam_run imagewam_python experiments/libero/run_libero_manager.py "${COMMON[@]}" "$@"
