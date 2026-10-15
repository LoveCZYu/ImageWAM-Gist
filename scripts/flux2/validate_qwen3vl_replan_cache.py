#!/usr/bin/env python3
"""Validate exact ImageWAM replan coverage and K=0..8 probability records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--action-chunk-size", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    if payload.get("format") != "samta_semantic_change_curve_v1":
        raise ValueError("Unexpected semantic cache format")
    preprocess = payload.get("preprocess", {})
    for key, expected in (
        ("action_chunk_size", int(args.action_chunk_size)),
        ("frame_stride", int(args.frame_stride)),
    ):
        if int(preprocess.get(key, -1)) != expected:
            raise ValueError(f"Cache {key}={preprocess.get(key)!r}, expected {expected}")
    episodes = [
        json.loads(line)
        for line in (args.dataset_root / "meta" / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    expected = {
        (int(episode["episode_index"]), timestep)
        for episode in episodes
        for timestep in range(
            0,
            max(int(episode["length"]) - int(args.action_chunk_size), 0),
            int(args.frame_stride),
        )
    }
    records = payload.get("records", [])
    actual: set[tuple[int, int]] = set()
    histogram: Counter[int] = Counter()
    for record in records:
        key = (int(record["episode_id"]), int(record["timestep"]))
        if key in actual:
            raise ValueError(f"Duplicate cache record: {key}")
        actual.add(key)
        probabilities = torch.as_tensor(
            record["teacher_probabilities"], dtype=torch.float32
        ).reshape(-1)
        if probabilities.numel() != 9 or not bool(torch.isfinite(probabilities).all()):
            raise ValueError(f"Invalid probability vector at {key}")
        if not torch.isclose(probabilities.sum(), torch.tensor(1.0), atol=1.0e-5):
            raise ValueError(f"Probabilities do not sum to one at {key}")
        count = int(record["teacher_count"])
        if count != int(probabilities.argmax().item()):
            raise ValueError(f"teacher_count disagrees with argmax at {key}")
        planned_evidence = record.get("planned_evidence")
        is_planned = isinstance(planned_evidence, dict)
        if not is_planned and count >= 3:
            raise ValueError(f"K>=3 record has no evidence plan at {key}")
        if is_planned:
            importance = int(planned_evidence.get("importance", -1))
            if not 3 <= importance <= 8:
                raise ValueError(f"Planned importance is outside 3-8 at {key}")
            if importance <= 4 and not 3 <= count <= 4:
                raise ValueError(f"Medium evidence record is outside K3-4 at {key}")
            if importance >= 5 and count < 5:
                raise ValueError(f"High evidence record is below K5 at {key}")
            if not str(planned_evidence.get("reason", "")).strip():
                raise ValueError(f"Planned evidence record has no reason at {key}")
        histogram[count] += 1
    missing = sorted(expected.difference(actual))
    extra = sorted(actual.difference(expected))
    if missing or extra:
        raise ValueError(
            f"Replan coverage mismatch: missing={len(missing)} {missing[:8]} "
            f"extra={len(extra)} {extra[:8]}"
        )
    expected_records = sum(
        len(
            range(
                0,
                max(int(episode["length"]) - int(args.action_chunk_size), 0),
                int(args.frame_stride),
            )
        )
        for episode in episodes
    )
    if expected_records != len(records):
        raise AssertionError(f"Expected {expected_records} records, found {len(records)}")
    print(
        json.dumps(
            {
                "status": "PASS",
                "episodes": len(episodes),
                "records": len(records),
                "missing": 0,
                "extra": 0,
                "histogram_n0_to_n8": [histogram[index] for index in range(9)],
                "teacher_method": payload.get("teacher_method"),
                "preprocess": preprocess,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
