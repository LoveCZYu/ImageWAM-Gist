#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from hydra import compose, initialize_config_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--tbptt-steps", type=int, required=True)
    parser.add_argument(
        "--memory-source",
        choices=("eatb", "episode_tbptt", "episode_autoregressive"),
        default="eatb",
    )
    args = parser.parse_args()

    with initialize_config_dir(config_dir=str(args.configs.resolve()), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "task=rmbench_flux2_klein_4b_base_memorywam_from_flux2",
                "memory=adaptive_gist_vlm",
                f"batch_size={args.batch_size}",
                "data.train.edit_memory_episode_terminal_samples_only=true",
                "data.train.edit_memory_drop_incomplete_current=true",
                "model.flux2_lora_config.enabled=false",
                "model.edit_memory_config.adaptive_gist.enabled=false",
                "model.edit_memory_config.fixed_gist_write_head=false",
                "model.edit_memory_config.vlm_count_supervision.enabled=false",
                "model.edit_memory_config.vlm_count_supervision.write_only=false",
                "model.edit_memory_config.training.memory_curriculum.enabled=false",
                f"model.edit_memory_config.training.memory_source={args.memory_source}",
                f"model.edit_memory_config.training.tbptt_steps={args.tbptt_steps}",
                "model.edit_memory_config.training.memory_curriculum.teacher_only_until_step=14",
                "model.edit_memory_config.training.memory_curriculum.transition_start_step=15",
                "model.edit_memory_config.training.memory_curriculum.transition_end_step=57",
            ],
        )
    memory = cfg.model.edit_memory_config
    episode_records = [
        json.loads(line)
        for line in (args.dataset / "meta" / "episodes.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    episode_count = len(episode_records)
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be positive, got {args.epochs}")
    if args.tbptt_steps <= 0:
        raise ValueError(f"--tbptt-steps must be positive, got {args.tbptt_steps}")
    stride = int(cfg.data.train.edit_memory_history_stride)
    current_horizon = (
        int(cfg.data.train.num_frames) - 1
        if bool(cfg.data.train.edit_memory_drop_incomplete_current)
        else 0
    )
    replan_counts = [
        len(range(0, max(0, int(record["length"]) - current_horizon), stride))
        for record in episode_records
    ]
    window_counts = [
        math.ceil(count / args.tbptt_steps) for count in replan_counts if count > 0
    ]
    global_batch_size = args.gpus * args.batch_size
    steps_per_epoch = sum(
        max(window_counts[start : start + global_batch_size])
        for start in range(0, len(window_counts), global_batch_size)
    )
    expected_steps = steps_per_epoch * args.epochs
    checks = {
        "positive_data_parallelism": args.gpus > 0 and args.batch_size > 0,
        "fixed_eight_gist": (
            not bool(memory.adaptive_gist.enabled)
            and int(memory.bank.gist_tokens_per_frame) == 8
        ),
        "no_adaptive_span": not bool(memory.adaptive_span.enabled),
        "paper_hybrid_windows": (
            int(memory.bank.anchor_frames) == 2
            and int(memory.bank.recent_frames) == 4
        ),
        "no_write_gate": not bool(memory.fixed_gist_write_head),
        "no_vlm_write_gate": not bool(memory.vlm_count_supervision.enabled),
        "no_memory_curriculum": not bool(memory.training.memory_curriculum.enabled),
        "memory_source_matches": (
            str(memory.training.memory_source) == args.memory_source
        ),
        "terminal_episode_samples": bool(
            cfg.data.train.edit_memory_episode_terminal_samples_only
        ),
        "tbptt_steps_match": (
            int(memory.training.tbptt_steps) == args.tbptt_steps
        ),
        "batch_size": int(cfg.batch_size) == int(args.batch_size),
        "full_flux": not bool(cfg.model.flux2_lora_config.enabled),
        "paper_video_action_loss": (
            float(cfg.model.loss.lambda_video) == 1.0
            and float(cfg.model.loss.lambda_action) == 1.0
        ),
        "video_logit_normal_sampling": (
            str(cfg.model.video_scheduler.sampling_distribution) == "logit_normal"
            and float(cfg.model.video_scheduler.logit_mean) == 0.0
            and float(cfg.model.video_scheduler.logit_std) == 1.0
            and str(cfg.model.video_scheduler.loss_weighting) in {"none", "constant"}
            and float(cfg.model.video_scheduler.train_shift) == 5.0
        ),
        "action_logit_normal_sampling": (
            str(cfg.model.action_scheduler.sampling_distribution) == "logit_normal"
            and float(cfg.model.action_scheduler.logit_mean) == 0.0
            and float(cfg.model.action_scheduler.logit_std) == 1.0
            and str(cfg.model.action_scheduler.loss_weighting) in {"none", "constant"}
            and float(cfg.model.action_scheduler.train_shift) == 1.0
            and float(cfg.model.action_scheduler.infer_shift) == 5.0
        ),
        "clean_conditioning_augmentation": (
            bool(memory.training.clean_conditioning_augmentation.enabled)
            and float(memory.training.clean_conditioning_augmentation.probability) == 1.0
            and float(memory.training.clean_conditioning_augmentation.ratio_min) == 0.0
            and float(memory.training.clean_conditioning_augmentation.ratio_max) == 1.0
        ),
        "gist_uses_rope_not_learned_time_embedding": (
            str(memory.gist.temporal_embedding).lower() == "rope"
        ),
        "episode_count_450": episode_count == 450,
        "epochs_positive": args.epochs > 0,
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Fixed-8 memory preflight failed: {failed}")

    print(
        json.dumps(
            {
                "status": "PASS",
                "config": checks,
                "episodes": episode_count,
                "epochs": args.epochs,
                "memory_source": args.memory_source,
                "optimizer_steps": expected_steps,
                "optimizer_steps_per_epoch": steps_per_epoch,
                "replans_per_epoch": sum(replan_counts),
                "windows_per_epoch": sum(window_counts),
                "tbptt_steps": args.tbptt_steps,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
