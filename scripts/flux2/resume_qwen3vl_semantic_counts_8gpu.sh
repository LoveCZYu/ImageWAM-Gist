#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xucx2606/ImageWAM/.venv/bin/python}"
QWEN3VL_PYDEPS="${QWEN3VL_PYDEPS:-/data41/xucx2606/vlm_semantic_labels_20260724/qwen3vl_pydeps}"
DATASET_ROOT="${DATASET_ROOT:-/home/xucx2606/ImageWAM/data/rmbench_lerobot/rmbench_official9_50demo}"
MODEL_ROOT="${MODEL_ROOT:-/data41/xucx2606/models/Qwen3-VL-8B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data41/xucx2606/vlm_semantic_labels_20260729/qwen3vl8b_episode_replan16_3view_sequence}"
GPU_WORKERS="${GPU_WORKERS:-1,1,1,1,1,1,1,1}"
IFS=',' read -r -a WORKER_COUNTS <<<"${GPU_WORKERS}"
WORLD_SIZE=0
for worker_count in "${WORKER_COUNTS[@]}"; do
  if [[ ! "${worker_count}" =~ ^[0-9]+$ ]]; then
    echo "GPU_WORKERS must be comma-separated non-negative integers: ${GPU_WORKERS}" >&2
    exit 2
  fi
  WORLD_SIZE=$((WORLD_SIZE + worker_count))
done
if [[ "${WORLD_SIZE}" -le 0 ]]; then
  echo "GPU_WORKERS must request at least one worker." >&2
  exit 2
fi
SHARD_DIR="${SHARD_DIR:-${OUTPUT_ROOT}/shards_${WORLD_SIZE}worker}"
RAW_OUTPUT="${RAW_OUTPUT:-${OUTPUT_ROOT}/semantic_change_curve_replan16_3view_sequence_raw_${WORLD_SIZE}worker.pt}"
AUDIT_OUTPUT="${AUDIT_OUTPUT:-${OUTPUT_ROOT}/audit_raw_${WORLD_SIZE}worker}"

if [[ -d "${QWEN3VL_PYDEPS}" ]]; then
  export PYTHONPATH="${QWEN3VL_PYDEPS}${PYTHONPATH:+:${PYTHONPATH}}"
fi

for required in "${PYTHON_BIN}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required input is unavailable: ${required}" >&2
    exit 2
  fi
done
for required_dir in "${DATASET_ROOT}" "${MODEL_ROOT}"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "Required directory is unavailable: ${required_dir}" >&2
    exit 2
  fi
done
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "${#WORKER_COUNTS[@]}" -ne "${GPU_COUNT}" ]]; then
  echo "GPU_WORKERS has ${#WORKER_COUNTS[@]} entries but the host has ${GPU_COUNT} GPUs." >&2
  exit 3
fi
if pgrep -f '[g]enerate_qwen3vl_semantic_counts.py' >/dev/null; then
  echo "A Qwen3-VL semantic-count generator is already running; refusing a duplicate launch." >&2
  exit 4
fi

mkdir -p "${SHARD_DIR}"
cd "${REPO_ROOT}"

pids=()
limit_args=()
if [[ -n "${LIMIT_EPISODES:-}" ]]; then
  limit_args+=(--limit-episodes "${LIMIT_EPISODES}")
fi
rank=0
for gpu in $(seq 0 $((GPU_COUNT - 1))); do
  for worker_slot in $(seq 1 "${WORKER_COUNTS[gpu]}"); do
    shard="${SHARD_DIR}/semantic_change_curve_rank_$(printf '%02d' "${rank}")_of_$(printf '%02d' "${WORLD_SIZE}").pt"
    log="${SHARD_DIR}/rank_$(printf '%02d' "${rank}").log"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u scripts/flux2/generate_qwen3vl_semantic_counts.py \
      --dataset-root "${DATASET_ROOT}" \
      --model "${MODEL_ROOT}" \
      --output "${shard}" \
      --rank "${rank}" \
      --world-size "${WORLD_SIZE}" \
      --action-chunk-size "${ACTION_CHUNK_SIZE:-16}" \
      --frame-stride "${FRAME_STRIDE:-16}" \
      --cpu-threads "${CPU_THREADS:-4}" \
      "${limit_args[@]}" \
      >"${log}" 2>&1 &
    pids+=("$!")
    echo "gpu=${gpu} slot=${worker_slot} rank=${rank} pid=${pids[-1]} output=${shard} log=${log}"
    rank=$((rank + 1))
  done
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
  echo "At least one semantic-label worker failed; shards and logs are preserved." >&2
  exit "${status}"
fi
if [[ "${PROBE_ONLY:-false}" == "true" ]]; then
  echo "Probe complete: workers=${WORLD_SIZE} limit_episodes=${LIMIT_EPISODES:-none}"
  exit 0
fi

"${PYTHON_BIN}" scripts/flux2/shard_qwen3vl_semantic_counts.py merge \
  --output-dir "${SHARD_DIR}" --output "${RAW_OUTPUT}" --world-size "${WORLD_SIZE}"
"${PYTHON_BIN}" scripts/flux2/validate_qwen3vl_replan_cache.py \
  --cache "${RAW_OUTPUT}" --dataset-root "${DATASET_ROOT}" \
  --action-chunk-size "${ACTION_CHUNK_SIZE:-16}" --frame-stride "${FRAME_STRIDE:-16}"
"${PYTHON_BIN}" scripts/flux2/audit_qwen3vl_semantic_counts.py \
  --cache "${RAW_OUTPUT}" --dataset-root "${DATASET_ROOT}" --output-dir "${AUDIT_OUTPUT}"
echo "Semantic cache complete: workers=${WORLD_SIZE} raw=${RAW_OUTPUT}"
