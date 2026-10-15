#!/usr/bin/env python3
"""Precompute and cache Qwen2.5-VL text embeddings.

Single-GPU (default):
    python scripts/omnigen2/precompute_qwen_embeds.py \
        task=robotwin_omnigen2_imagewam \
        data.train.dataset_dirs=[...] \
        data.train.qwen_text_cache_dir=./qwen_cache \
        qwen_cache_batch_size=32

Multi-GPU via torchrun (each rank gets one GPU, prompts are sharded):
    torchrun --nproc_per_node=8 scripts/omnigen2/precompute_qwen_embeds.py \
        task=robotwin_omnigen2_imagewam \
        data.train.dataset_dirs=[...] \
        data.train.qwen_text_cache_dir=./qwen_cache \
        qwen_cache_batch_size=32

Cache files are named by SHA-256 hash so writes from different ranks never
conflict — no distributed coordination is needed.

File saving is done asynchronously in a ThreadPoolExecutor so the GPU starts
the next batch while the previous batch's .pt files are being written to disk.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen2_5_VLModel

from imagewam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from imagewam.utils.config_resolvers import register_default_resolvers
from imagewam.utils.logging_config import get_logger, setup_logging

register_default_resolvers()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Multi-GPU rank helpers (torchrun sets LOCAL_RANK / WORLD_SIZE automatically)
# ---------------------------------------------------------------------------

def _get_rank_info() -> tuple[int, int]:
    """Return (local_rank, world_size) from env vars set by torchrun/mpirun."""
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return local_rank, world_size


def _is_main_rank(local_rank: int) -> bool:
    return local_rank == 0


# ---------------------------------------------------------------------------
# Config / dataset helpers
# ---------------------------------------------------------------------------

def _iter_dataset_nodes(node: Any, path: str = "data"):
    if isinstance(node, DictConfig):
        if "dataset_dirs" in node and node.get("dataset_dirs") is not None:
            yield path, node
        for key, value in node.items():
            yield from _iter_dataset_nodes(value, f"{path}.{key}")
    elif isinstance(node, ListConfig):
        for idx, value in enumerate(node):
            yield from _iter_dataset_nodes(value, f"{path}[{idx}]")


def _instruction_to_prompt(instruction: Any) -> str | None:
    task = str(instruction).strip()
    if not task:
        return None
    return DEFAULT_PROMPT.format(task=task)


def _configured_prompts(node: DictConfig) -> list[str]:
    prompts: list[str] = []
    override_instruction = node.get("override_instruction")
    if override_instruction is not None:
        prompt = _instruction_to_prompt(override_instruction)
        return [prompt] if prompt is not None else []

    fallback_instructions = node.get("fallback_instructions")
    if not fallback_instructions:
        return prompts
    for instruction in fallback_instructions.values():
        prompt = _instruction_to_prompt(instruction)
        if prompt is not None:
            prompts.append(prompt)
    return prompts


def _collect_dataset_settings(data_cfg: DictConfig) -> tuple[list[str], list[Path], set[int], list[str]]:
    dataset_dirs: list[str] = []
    cache_dirs: list[Path] = []
    context_lens: set[int] = set()
    config_prompts: list[str] = []
    seen_prompts: set[str] = set()
    for node_path, node in _iter_dataset_nodes(data_cfg):
        raw_dirs = node.get("dataset_dirs")
        if raw_dirs is None:
            continue
        cache_dir = node.get("qwen_text_cache_dir")
        if cache_dir is None or not str(cache_dir).strip():
            raise ValueError(f"Missing `qwen_text_cache_dir` for dataset node `{node_path}`.")
        node_prompts = _configured_prompts(node)
        if not node_prompts:
            for ds in raw_dirs:
                ds_str = str(ds)
                if ds_str not in dataset_dirs:
                    dataset_dirs.append(ds_str)
        cache_path = Path(str(cache_dir)).expanduser()
        if cache_path not in cache_dirs:
            cache_dirs.append(cache_path)
        context_lens.add(int(node.get("qwen_context_len", 128)))
        for prompt in node_prompts:
            if prompt not in seen_prompts:
                seen_prompts.add(prompt)
                config_prompts.append(prompt)
        if node_prompts:
            logger.info(
                "Using %d configured prompt(s) from `%s`; skipping tasks.jsonl for this node.",
                len(node_prompts),
                node_path,
            )
    return dataset_dirs, cache_dirs, context_lens, config_prompts


def _read_unique_prompts(dataset_dirs: list[str], initial_prompts: list[str] | None = None) -> list[str]:
    prompts: list[str] = []
    seen = set()
    for prompt in initial_prompts or []:
        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
    for ds_dir in dataset_dirs:
        tasks_path = Path(ds_dir) / "meta" / "tasks.jsonl"
        if not tasks_path.exists():
            raise FileNotFoundError(f"Missing tasks file: {tasks_path}")
        with tasks_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "task" not in record:
                    raise KeyError(f"Missing `task` field at {tasks_path}:{line_idx}")
                prompt = DEFAULT_PROMPT.format(task=str(record["task"]))
                if prompt not in seen:
                    seen.add(prompt)
                    prompts.append(prompt)
    return prompts


def _atomic_torch_save(payload: dict[str, torch.Tensor], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{os.getpid()}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging(log_level=logging.INFO)
    if cfg.data is None:
        raise ValueError("`cfg.data` is required.")
    if cfg.model is None:
        raise ValueError("`cfg.model` is required.")

    local_rank, world_size = _get_rank_info()
    is_main = _is_main_rank(local_rank)

    # ------------------------------------------------------------------
    # GPU / device assignment
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_id = local_rank % num_gpus
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        gpu_id = -1
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if is_main:
        logger.info(
            "Multi-GPU cache: world_size=%d  local_rank=%d → device=%s  dtype=%s",
            world_size, local_rank, device, dtype,
        )

    # ------------------------------------------------------------------
    # Collect prompts & settings
    # ------------------------------------------------------------------
    dataset_dirs, cache_dirs, context_lens, config_prompts = _collect_dataset_settings(cfg.data)
    if len(context_lens) != 1:
        raise ValueError(f"Expected one qwen_context_len, got {sorted(context_lens)}")
    context_len = next(iter(context_lens))

    all_prompts = _read_unique_prompts(dataset_dirs, initial_prompts=config_prompts)
    if not all_prompts:
        if is_main:
            logger.warning("No prompts found; nothing to cache.")
        return

    # Shard prompts across ranks (interleaved so each rank gets a mix of tasks)
    my_prompts = all_prompts[local_rank::world_size]

    if is_main:
        logger.info(
            "Total unique prompts: %d  →  %d per rank (world_size=%d)",
            len(all_prompts), len(my_prompts), world_size,
        )
    else:
        logger.info(
            "[rank%d] Assigned %d / %d prompts",
            local_rank, len(my_prompts), len(all_prompts),
        )

    # ------------------------------------------------------------------
    # Load model (once per rank / GPU)
    # ------------------------------------------------------------------
    qwen_path = str(cfg.model.qwen_path)
    if is_main:
        logger.warning(
            "ImageWAM-OmniGen2 Qwen cache uses `%s`. This follows OmniGen2 official training configs "
            "(`pretrained_text_encoder_model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct`). "
            "It is not yet confirmed whether OmniGen2 inference's `mllm/` subfolder is the correct or "
            "a separately tuned text path for this ImageWAM use case.",
            qwen_path,
        )
    logger.info(
        "[rank%d] Loading Qwen text encoder from %s → %s dtype=%s",
        local_rank, qwen_path, device, dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(qwen_path)
    tokenizer.padding_side = "right"
    text_encoder = Qwen2_5_VLModel.from_pretrained(qwen_path, torch_dtype=dtype).to(device).eval()

    batch_size = int(cfg.get("qwen_cache_batch_size", 32))
    overwrite = bool(cfg.get("qwen_cache_overwrite", False))
    save_workers = int(cfg.get("qwen_cache_save_workers", 4))
    for cache_dir in cache_dirs:
        cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Cache loop
    # GPU forward results are moved to CPU then saved asynchronously via a
    # thread pool so the GPU can start the next batch while files are written.
    # On a fast local disk (ext4 /data) this is fine; on slow CephFS fuse you
    # can reduce save_workers to 1 to avoid metadata storms.
    # ------------------------------------------------------------------
    written = 0
    skipped = 0
    total_fwd_s = 0.0
    total_save_s = 0.0
    rank_desc = f"rank{local_rank}" if world_size > 1 else "Caching Qwen text"

    def _save_one(prompt: str, hidden_i: torch.Tensor, mask_i: torch.Tensor) -> None:
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        payload = {
            "text_hidden_states": hidden_i.clone(),
            "text_attention_mask": mask_i.to(dtype=torch.bool).clone(),
        }
        for cache_dir in cache_dirs:
            _atomic_torch_save(payload, cache_dir / f"{hashed}.qwen2_5_vl_len{context_len}.pt")

    with torch.no_grad(), concurrent.futures.ThreadPoolExecutor(max_workers=save_workers) as pool:
        futs: list[concurrent.futures.Future] = []

        pbar = tqdm(
            range(0, len(my_prompts), batch_size),
            desc=rank_desc,
            disable=not is_main,
        )
        for start in pbar:
            batch_prompts = my_prompts[start : start + batch_size]
            needed = []
            for prompt in batch_prompts:
                hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                exists_everywhere = all(
                    (cache_dir / f"{hashed}.qwen2_5_vl_len{context_len}.pt").exists()
                    for cache_dir in cache_dirs
                )
                if exists_everywhere and not overwrite:
                    skipped += 1
                else:
                    needed.append(prompt)
            if not needed:
                continue

            encoded = tokenizer(
                needed,
                padding="max_length",
                truncation=True,
                max_length=context_len,
                return_tensors="pt",
            )
            input_ids = encoded.input_ids.to(device)
            attention_mask = encoded.attention_mask.to(device)

            t0 = time.perf_counter()
            hidden = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            ).last_hidden_state
            # Bulk GPU→CPU transfer (one cudaMemcpy for the batch) then release GPU.
            hidden_cpu = hidden.detach().to(device="cpu", dtype=torch.bfloat16)
            mask_cpu = attention_mask.detach().cpu()
            total_fwd_s += time.perf_counter() - t0

            t1 = time.perf_counter()
            for i, prompt in enumerate(needed):
                futs.append(pool.submit(_save_one, prompt, hidden_cpu[i], mask_cpu[i]))
                written += 1
            total_save_s += time.perf_counter() - t1  # dispatch time only

        # Drain: wait for all saves and surface any write errors.
        t2 = time.perf_counter()
        for fut in concurrent.futures.as_completed(futs):
            fut.result()
        total_save_s += time.perf_counter() - t2

    logger.info(
        "[rank%d] Finished: written=%d skipped=%d gpu_fwd=%.1fs save_wall=%.1fs  dirs=%s",
        local_rank, written, skipped, total_fwd_s, total_save_s,
        [str(p) for p in cache_dirs],
    )


if __name__ == "__main__":
    main()
