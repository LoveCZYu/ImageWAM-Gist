#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

TASK="${TASK:-libero_ovis_u1_imagewam}"
OVIS_U1_MODEL_PATH="${OVIS_U1_MODEL_PATH:-AIDC-AI/Ovis-U1-3B}"
imagewam_ckpt_from_exp
imagewam_require_env CKPT_PATH
imagewam_require_env DATASET_STATS_PATH

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
LIBERO_WORKER_ENV_SOURCE="${LIBERO_WORKER_ENV_SOURCE:-}"
export LIBERO_WORKER_ENV_SOURCE
imagewam_prepare_eval_ckpt

imagewam_print_config TASK CKPT_PATH DATASET_STATS_PATH OVIS_U1_MODEL_PATH
imagewam_run imagewam_python experiments/libero/run_libero_manager.py \
  --config-name sim_libero_omnigen2 \
  task="${TASK}" \
  ckpt="${CKPT_PATH}" \
  EVALUATION.dataset_stats_path="${DATASET_STATS_PATH}" \
  model.ovis_u1_model_path="${OVIS_U1_MODEL_PATH}" \
  MULTIRUN.num_gpus="${NUM_GPUS:-8}" \
  MULTIRUN.max_tasks_per_gpu="${MAX_TASKS_PER_GPU:-4}" \
  MULTIRUN.task_suite_names="${TASK_SUITE_NAMES:-[libero_10,libero_goal,libero_spatial,libero_object]}" \
  EVALUATION.num_trials="${NUM_TRIALS:-10}" \
  EVALUATION.action_horizon="${ACTION_HORIZON:-16}" \
  EVALUATION.replan_steps="${REPLAN_STEPS:-8}" \
  model.proprio_dim="${PROPRIO_DIM:-8}" \
  "$@"
