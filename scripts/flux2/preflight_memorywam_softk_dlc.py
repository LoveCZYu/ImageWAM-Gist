#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument(
        "--memory-source",
        choices=("eatb", "episode_tbptt", "episode_autoregressive"),
        default="eatb",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with initialize_config_dir(config_dir=str(args.configs.resolve()), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "task=rmbench_flux2_klein_4b_base_memorywam_from_flux2",
                "memory=adaptive_gist_vlm",
                "data.train.edit_memory_drop_incomplete_current=true",
                "data.val.edit_memory_drop_incomplete_current=true",
                "model.flux2_lora_config.enabled=false",
                "model.edit_memory_config.adaptive_gist.enabled=true",
                f"model.edit_memory_config.training.memory_source={args.memory_source}",
                "model.edit_memory_config.training.memory_curriculum.enabled=true",
                "model.edit_memory_config.training.memory_curriculum.teacher_only_until_step=28",
                "model.edit_memory_config.training.memory_curriculum.transition_start_step=29",
                "model.edit_memory_config.training.memory_curriculum.transition_end_step=114",
            ],
        )
    memory = cfg.model.edit_memory_config
    curriculum = memory.training.memory_curriculum
    checks = {
        "memory_enabled": bool(memory.enabled),
        "adaptive_gist": bool(memory.adaptive_gist.enabled),
        "no_action_attention_teacher": not bool(memory.selector.use_action_attention_teacher),
        "vlm_supervision": bool(memory.vlm_count_supervision.enabled),
        "direct_vlm_soft_label_loss": float(memory.vlm_count_supervision.count_weight) > 0.0,
        "full_flux": not bool(cfg.model.flux2_lora_config.enabled),
        "flux_gradient_checkpointing": bool(cfg.model.mot_checkpoint_mixed_attn),
        "action_gradient_checkpointing": bool(cfg.model.action_dit_config.use_gradient_checkpointing),
        "full_history": int(cfg.data.train.edit_memory_history_max_steps) == 0,
        "replan_16": int(cfg.data.train.edit_memory_history_stride) == 16,
        "terminal_episode_samples": bool(
            cfg.data.train.edit_memory_episode_terminal_samples_only
        ),
        "five_epochs": int(cfg.num_epochs) == 5,
        "clean_current_frame": cfg.data.train.video_augmentation is None,
        "batch_one": int(cfg.batch_size) == 1,
        "ga_one": int(cfg.gradient_accumulation_steps) == 1,
        "anchor_two": int(memory.bank.anchor_frames) == 2,
        "recent_four_including_current": int(memory.bank.recent_frames) == 4,
        "gist_capacity_eight": int(memory.bank.gist_tokens_per_frame) == 8,
        "unified_softk_zero_to_eight": (
            int(memory.adaptive_gist.min_tokens) == 0
            and int(memory.adaptive_gist.max_tokens) == 8
        ),
        "no_competing_token_cost": float(memory.adaptive_gist.token_cost) == 0.0,
        "episode_replay": str(memory.training.memory_source)
        == args.memory_source,
        "backward_span": (
            args.memory_source in {"eatb", "episode_autoregressive"}
            or int(memory.training.tbptt_steps) == 4
        ),
        "history_length_bucketing": bool(cfg.history_length_bucketing),
        "history_payload_stays_on_host": bool(cfg.manual_batch_device_placement),
        "clean_prefix_memory_head": bool(memory.clean_prefix_memory_head),
        "write_is_unified_k0": not bool(memory.fixed_gist_write_head),
        "teacher_to_predict_curriculum": (
            bool(curriculum.enabled)
            and int(curriculum.teacher_only_until_step) == 28
            and int(curriculum.transition_start_step) == 29
            and int(curriculum.transition_end_step) == 114
            and str(curriculum.granularity) == "sample"
        ),
        "cache_path": Path(str(memory.vlm_count_supervision.cache_path)).resolve()
        == args.cache.resolve(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Soft-k composed-config preflight failed: {failed}")

    payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    if payload.get("format") != "samta_semantic_change_curve_v1" or int(payload.get("k_max", -1)) != 8:
        raise ValueError("VLM cache must contain N=0..8 semantic curves in samta_semantic_change_curve_v1 format.")
    records = payload.get("records", [])
    actual: set[tuple[int, int]] = set()
    histogram = [0] * 9
    for record in records:
        key = (int(record["episode_id"]), int(record["timestep"]))
        if key in actual:
            raise ValueError(f"Duplicate VLM cache key: {key}")
        actual.add(key)
        probabilities = torch.as_tensor(
            record["teacher_probabilities"], dtype=torch.float32
        ).reshape(-1)
        teacher_count = int(record["teacher_count"])
        if (
            probabilities.numel() != 9
            or not bool(torch.isfinite(probabilities).all())
            or bool((probabilities < 0).any())
            or not bool(torch.isclose(probabilities.sum(), torch.tensor(1.0), atol=1.0e-5))
            or teacher_count != int(probabilities.argmax().item())
        ):
            raise ValueError(f"Invalid direct VLM count label at {key}")
        histogram[teacher_count] += 1

    expected: set[tuple[int, int]] = set()
    required_future = int(cfg.data.train.num_frames) - 1
    replan_stride = int(cfg.data.train.edit_memory_history_stride)
    episodes_path = args.dataset / "meta" / "episodes.jsonl"
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        episode = json.loads(line)
        episode_id = int(episode["episode_index"])
        valid_current_count = max(int(episode["length"]) - required_future, 0)
        expected.update(
            (episode_id, timestep)
            for timestep in range(0, valid_current_count, replan_stride)
        )
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(
            f"VLM cache coverage mismatch: expected={len(expected)} actual={len(actual)} "
            f"missing={len(missing)} extra={len(extra)}"
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "config": checks,
                "records": len(actual),
                "episodes": len({episode_id for episode_id, _ in actual}),
                "histogram_n0_to_n8": histogram,
                "cache_path": str(args.cache),
                "replan_steps": int(os.environ.get("IMAGEWAM_REPLAN_STEPS", "16")),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
