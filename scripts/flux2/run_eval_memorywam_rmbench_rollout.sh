#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data41/xucx2606/memoryWAM_softk}"
PYTHON_BIN="${PYTHON_BIN:-/home/xucx2606/ImageWAM/.venv/bin/python}"
RMBENCH_ROOT="${RMBENCH_ROOT:-/home/xucx2606/RMBench}"
FLUX2_SRC="${FLUX2_SRC:-/home/xucx2606/ImageWAM/third_party/flux2}"
FLUX2_MODEL_PATH="${FLUX2_MODEL_PATH:-/home/xucx2606/ImageWAM/checkpoints/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors}"
FLUX2_AE_MODEL_PATH="${FLUX2_AE_MODEL_PATH:-/home/xucx2606/ImageWAM/checkpoints/flux2/FLUX.2-dev/ae.safetensors}"
FLUX2_QWEN3_MODEL_SPEC="${FLUX2_QWEN3_MODEL_SPEC:-/home/xucx2606/models/Qwen3-4B-Instruct-2507}"
RMBENCH_LEROBOT_ROOT="${RMBENCH_LEROBOT_ROOT:-/home/xucx2606/ImageWAM/data/rmbench_lerobot/rmbench_official9_50demo}"
QWEN_CACHE_DIR="${QWEN_CACHE_DIR:-/home/xucx2606/ImageWAM/data/rmbench_lerobot/flux2_qwen3_cache_4b_official9}"
DATASET_STATS_PATH="${DATASET_STATS_PATH:-/home/xucx2606/ImageWAM/runs/rmbench_flux2_klein_4b_base_imagewam/2026-06-26_19-31-23/dataset_stats.json}"
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/runs/rmbench_flux2_klein_4b_base_memorywam_from_flux2/memorywam_kv8_episode_tbptt4_fullflux_8gpu_zero2_bs1_ep5_20260725_175203/checkpoints/weights/step_000285.pt}"

GPU_ID="${GPU_ID:-1}"
TASK_NAME="${TASK_NAME:-observe_and_pickup}"
ENV_SEED_CONFIG="${ENV_SEED_CONFIG:--1}"  # RMBench maps -1 to simulator seed 0.
ACTION_SEED="${ACTION_SEED:-42}"
ACTION_HORIZON="${ACTION_HORIZON:-16}"
REPLAN_STEPS="${REPLAN_STEPS:-16}"
# The paper evaluates simulator rollouts with 50 denoising steps. Real-time
# deployments may still override this to 10 through the environment.
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
CUDA_MEMORY_FRACTION="${CUDA_MEMORY_FRACTION:-0.19}"
RUN_ID="${RUN_ID:-memorywam_step285_${TASK_NAME}_seed0_$(date +%Y%m%d_%H%M%S)}"

export PYTHONPATH="${REPO_ROOT}/src:${FLUX2_SRC}/src:${FLUX2_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
export IMAGEWAM_RMBENCH_LEROBOT_ROOT="${RMBENCH_LEROBOT_ROOT}"
export IMAGEWAM_RMBENCH_QWEN_CACHE="${QWEN_CACHE_DIR}"
export IMAGEWAM_REPLAN_STEPS="${REPLAN_STEPS}"
export IMAGEWAM_ACTION_SEED="${ACTION_SEED}"
export IMAGEWAM_CUDA_MEMORY_FRACTION="${CUDA_MEMORY_FRACTION}"
export IMAGEWAM_EDIT_MEMORY_DEBUG=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

OUTPUT_HINT="${REPO_ROOT}/evaluate_results/robotwin/placeholder/${RUN_ID}"
CMD=(
  "${PYTHON_BIN}"
  experiments/robotwin/eval_robotwin_single.py
  --config-name sim_robotwin
  task=rmbench_flux2_klein_4b_base_memorywam_from_flux2
  memory=adaptive_gist_vlm_infer
  "ckpt=${CKPT_PATH}"
  "gpu_id=${GPU_ID}"
  "seed=${ENV_SEED_CONFIG}"
  "EVALUATION.robotwin_root=${RMBENCH_ROOT}"
  "EVALUATION.task_name=${TASK_NAME}"
  EVALUATION.task_config=demo_clean
  EVALUATION.instruction_type=unseen
  EVALUATION.eval_num_episodes=1
  "EVALUATION.output_dir=${OUTPUT_HINT}"
  "EVALUATION.dataset_stats_path=${DATASET_STATS_PATH}"
  "EVALUATION.action_horizon=${ACTION_HORIZON}"
  "EVALUATION.replan_steps=${REPLAN_STEPS}"
  "EVALUATION.num_inference_steps=${NUM_INFERENCE_STEPS}"
  EVALUATION.timing_enabled=true
  EVALUATION.skip_get_obs_within_replan=false
  EVALUATION.robotwin_camera_layout=compact_288x256
  "model.flux2_src_path=${FLUX2_SRC}"
  "model.flux2_model_path=${FLUX2_MODEL_PATH}"
  "model.ae_model_path=${FLUX2_AE_MODEL_PATH}"
  model.variant=klein-base-4b
  "model.qwen3_model_spec=${FLUX2_QWEN3_MODEL_SPEC}"
  model.qwen_context_len=512
  model.load_text_encoder=true
  model.pack_proprio_after_text=true
  model.proprio_dim=14
)

cd "${REPO_ROOT}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exec "${CMD[@]}" --cfg job --resolve
fi
exec "${CMD[@]}"
