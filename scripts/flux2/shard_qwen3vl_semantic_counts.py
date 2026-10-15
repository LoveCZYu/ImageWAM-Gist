#!/usr/bin/env python3
"""Split or merge resumable Qwen3-VL semantic-count caches by episode."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


IDENTITY_KEYS = (
    "format",
    "teacher_model",
    "teacher_prompt_hash",
    "preprocess_hash",
    "semantic_scorer_hash",
    "dataset_episodes_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("split", "merge"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--world-size", type=int, default=8)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "samta_semantic_change_curve_v1":
        raise ValueError(f"Unexpected cache format: {path}")
    return payload


def shard_path(output_dir: Path, rank: int, world_size: int) -> Path:
    return output_dir / f"semantic_change_curve_rank_{rank:02d}_of_{world_size:02d}.pt"


def atomic_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def assert_same_identity(left: dict[str, Any], right: dict[str, Any]) -> None:
    for key in IDENTITY_KEYS:
        if left.get(key) != right.get(key):
            raise ValueError(f"Cache identity mismatch for {key}")


def split(args: argparse.Namespace) -> None:
    if args.input is None or not args.input.is_file():
        raise FileNotFoundError(f"Split input is unavailable: {args.input}")
    source = load(args.input)
    source_records = {
        (int(record["episode_id"]), int(record["timestep"])): record
        for record in source["records"]
    }
    for rank in range(args.world_size):
        path = shard_path(args.output_dir, rank, args.world_size)
        records: dict[tuple[int, int], dict[str, Any]] = {}
        if path.is_file():
            existing = load(path)
            assert_same_identity(source, existing)
            records.update(
                ((int(record["episode_id"]), int(record["timestep"])), record)
                for record in existing["records"]
            )
        records.update(
            (key, record)
            for key, record in source_records.items()
            if key[0] % args.world_size == rank
        )
        payload = dict(source)
        payload["rank"] = rank
        payload["world_size"] = args.world_size
        payload["records"] = [records[key] for key in sorted(records)]
        atomic_save(path, payload)
        print(f"rank={rank} records={len(records)} output={path}")


def merge(args: argparse.Namespace) -> None:
    if args.output is None:
        raise ValueError("--output is required for merge")
    payloads = [load(shard_path(args.output_dir, rank, args.world_size)) for rank in range(args.world_size)]
    first = payloads[0]
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for rank, payload in enumerate(payloads):
        assert_same_identity(first, payload)
        for record in payload["records"]:
            key = (int(record["episode_id"]), int(record["timestep"]))
            if key[0] % args.world_size != rank:
                raise ValueError(f"Record {key} is in the wrong shard {rank}")
            if key in records:
                raise ValueError(f"Duplicate record across shards: {key}")
            records[key] = record
    merged = dict(first)
    merged["rank"] = 0
    merged["world_size"] = 1
    merged["records"] = [records[key] for key in sorted(records)]
    atomic_save(args.output, merged)
    print(f"merged_records={len(records)} output={args.output}")


def main() -> None:
    args = parse_args()
    if args.world_size <= 0:
        raise ValueError("--world-size must be positive")
    if args.mode == "split":
        split(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
