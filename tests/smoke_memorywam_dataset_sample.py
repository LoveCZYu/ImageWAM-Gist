#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode one complete MemoryWAM training sample.")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--qwen-cache", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=100)
    args = parser.parse_args()

    os.environ["IMAGEWAM_RMBENCH_LEROBOT_ROOT"] = str(args.dataset)
    os.environ["IMAGEWAM_RMBENCH_QWEN_CACHE"] = str(args.qwen_cache)
    os.environ["IMAGEWAM_REPLAN_STEPS"] = "16"
    with initialize_config_dir(config_dir=str((args.repo / "configs").resolve()), version_base=None):
        config = compose(
            config_name="train",
            overrides=[
                "task=rmbench_flux2_klein_4b_base_memorywam_from_flux2",
                f"data.train.pretrained_norm_stats={args.stats}",
            ],
        )
    dataset = instantiate(config.data.train)
    sample = dataset[args.sample_index % len(dataset)]
    required = {
        "video",
        "action",
        "proprio",
        "text_hidden_states",
        "text_attention_mask",
        "edit_memory_video",
        "edit_memory_action",
        "edit_memory_proprio",
        "edit_memory_valid",
    }
    missing = required - set(sample)
    if missing:
        raise RuntimeError(f"Training sample is missing fields: {sorted(missing)}")
    if sample["text_hidden_states"].shape[0] != 512:
        raise RuntimeError(f"Unexpected Qwen context shape: {tuple(sample['text_hidden_states'].shape)}")
    shapes = {
        key: tuple(int(dim) for dim in value.shape)
        for key, value in sample.items()
        if key in required and isinstance(value, torch.Tensor)
    }
    print(f"PASS training sample index={args.sample_index % len(dataset)} length={len(dataset)} shapes={shapes}")


if __name__ == "__main__":
    main()
