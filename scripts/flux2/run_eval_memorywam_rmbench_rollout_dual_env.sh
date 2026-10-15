#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data41/xucx2606/memoryWAM_softk}"
RMBENCH_ROOT="${RMBENCH_ROOT:-/home/xucx2606/RMBench}"
IMAGEWAM_PYTHON="${IMAGEWAM_PYTHON:-/home/xucx2606/ImageWAM/.venv/bin/python}"
RMBENCH_PYTHON="${RMBENCH_PYTHON:-/home/xucx2606/miniconda3/envs/RMBench/bin/python}"
FLUX2_SRC="${FLUX2_SRC:-/home/xucx2606/ImageWAM/third_party/flux2}"
SERVER_CONFIG="${SERVER_CONFIG:-${REPO_ROOT}/scripts/flux2/rmbench_memorywam_step285_server.yml}"
CLIENT_POLICY_SOURCE="${REPO_ROOT}/experiments/rmbench/imagewam_remote_client"
SERVER_POLICY_SOURCE="${REPO_ROOT}/experiments/robotwin/imagewam_policy"
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/runs/rmbench_flux2_klein_4b_base_memorywam_from_flux2/memorywam_kv8_episode_tbptt4_fullflux_8gpu_zero2_bs1_ep5_20260725_175203/checkpoints/weights/step_000285.pt}"

GPU_ID="${GPU_ID:-1}"
PORT="${PORT:-29685}"
TASK_NAME="${TASK_NAME:-observe_and_pickup}"
RUN_ID="${RUN_ID:-memorywam_step285_${TASK_NAME}_seed0_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/evaluate_results/rmbench_rollout/${RUN_ID}}"
CUDA_MEMORY_FRACTION="${CUDA_MEMORY_FRACTION:-0.19}"

mkdir -p "${OUTPUT_DIR}"
ln -sfn "${SERVER_POLICY_SOURCE}" "${RMBENCH_ROOT}/policy/imagewam_policy"
ln -sfn "${CLIENT_POLICY_SOURCE}" "${RMBENCH_ROOT}/policy/imagewam_remote_client"

SERVER_LOG="${OUTPUT_DIR}/model_server.log"
CLIENT_LOG="${OUTPUT_DIR}/rollout_client.log"
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

(
  cd "${RMBENCH_ROOT}"
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  export PYTHONPATH="${REPO_ROOT}/src:${FLUX2_SRC}/src:${FLUX2_SRC}:${RMBENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  export IMAGEWAM_RMBENCH_LEROBOT_ROOT=/home/xucx2606/ImageWAM/data/rmbench_lerobot/rmbench_official9_50demo
  export IMAGEWAM_RMBENCH_QWEN_CACHE=/home/xucx2606/ImageWAM/data/rmbench_lerobot/flux2_qwen3_cache_4b_official9
  export IMAGEWAM_QWEN_TEXT_CACHE_DIR=/home/xucx2606/ImageWAM/data/rmbench_lerobot/flux2_qwen3_cache_4b_official9
  export IMAGEWAM_REQUIRE_TEXT_CACHE=true
  export IMAGEWAM_REPLAN_STEPS=16
  export IMAGEWAM_ACTION_SEED=42
  export IMAGEWAM_CUDA_MEMORY_FRACTION="${CUDA_MEMORY_FRACTION}"
  export IMAGEWAM_EDIT_MEMORY_DEBUG=1
  export IMAGEWAM_ROLLOUT_DIAGNOSTICS_PATH="${OUTPUT_DIR}/rollout_diagnostics.jsonl"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export PYTHONUNBUFFERED=1
  exec "${IMAGEWAM_PYTHON}" -u script/policy_model_server.py \
    --port "${PORT}" \
    --config "${SERVER_CONFIG}"
) >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "${SERVER_PID}" >"${OUTPUT_DIR}/model_server.pid"

for _ in $(seq 1 600); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "ImageWAM model server exited during initialization." >&2
    tail -n 200 "${SERVER_LOG}" >&2
    exit 1
  fi
  if grep -q "Model server started" "${SERVER_LOG}" 2>/dev/null; then
    break
  fi
  sleep 1
done
if ! grep -q "Model server started" "${SERVER_LOG}" 2>/dev/null; then
  echo "Timed out waiting for ImageWAM model server." >&2
  exit 1
fi

(
  cd "${RMBENCH_ROOT}"
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  export IMAGEWAM_INSTRUCTION_OVERRIDE="Initially, there is one target object on the shelf and five random objects on the table. Then, a screen obscures the target object. Pick up the corresponding target object from the table and lift it up."
  export PYTHONUNBUFFERED=1
  exec "${RMBENCH_PYTHON}" -u script/eval_policy.py \
    --config policy/imagewam_remote_client/deploy_policy.yml \
    --overrides \
    --task_name "${TASK_NAME}" \
    --task_config demo_clean \
    --ckpt_setting "${CKPT_PATH}" \
    --seed -1 \
    --policy_name imagewam_remote_client \
    --instruction_type unseen \
    --eval_num_episodes 1 \
    --eval_start_episode 0 \
    --eval_output_dir "${OUTPUT_DIR}" \
    --port "${PORT}"
) 2>&1 | tee "${CLIENT_LOG}"
