#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

SUITE="robotwin"
CONFIG_NAME="sim_robotwin"
TASK="${TASK:-robotwin_omnigen2_imagewam}"

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
  MULTIRUN.max_tasks_per_gpu="${MAX_TASKS_PER_GPU:-3}"
  EVALUATION.action_horizon="${ACTION_HORIZON:-16}"
  EVALUATION.replan_steps="${REPLAN_STEPS:-16}"
  model.proprio_dim="${PROPRIO_DIM:-14}"
)

COMMON+=(
  EVALUATION.eval_num_episodes="${EVAL_NUM_EPISODES:-50}"
  MULTIRUN.phases="${PHASES:-[clean,random]}"
  EVALUATION.skip_get_obs_within_replan="${SKIP_GET_OBS_WITHIN_REPLAN:-true}"
  EVALUATION.robotwin_camera_layout="${ROBOTWIN_CAMERA_LAYOUT:-compact_288x256}"
  EVALUATION.timing_enabled="${TIMING_ENABLED:-false}"
)

imagewam_print_config SUITE TASK CKPT_PATH DATASET_STATS_PATH OMNIGEN2_SRC OMNIGEN2_MODEL_PATH QWEN_MODEL_PATH
imagewam_run imagewam_python experiments/robotwin/run_robotwin_manager.py "${COMMON[@]}" "$@"
