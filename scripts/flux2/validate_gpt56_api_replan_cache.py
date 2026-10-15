#!/usr/bin/env python3
"""Validate direct GPT-5.6 API K=0..8 labels and exact replan coverage."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--allow-subset", action="store_true")
    args = parser.parse_args()
    payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    if payload.get("teacher_method") not in {
        "gpt56_api_episode_direct_free_k0_to_k8_v1",
        "gpt56_api_episode_direct_free_k0_to_k8_v2",
    }:
        raise ValueError("Unexpected teacher method")
    preprocess = payload["preprocess"]
    action_chunk_size = int(preprocess["action_chunk_size"])
    frame_stride = int(preprocess["frame_stride"])
    episodes = [
        json.loads(line)
        for line in (args.dataset_root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    expected = {
        (int(episode["episode_index"]), step)
        for episode in episodes
        for step in range(0, max(int(episode["length"]) - action_chunk_size, 0), frame_stride)
    }
    actual: set[tuple[int, int]] = set()
    histogram: Counter[int] = Counter()
    for record in payload["records"]:
        key = (int(record["episode_id"]), int(record["timestep"]))
        if key in actual:
            raise ValueError(f"Duplicate record {key}")
        actual.add(key)
        probabilities = torch.as_tensor(record["teacher_probabilities"]).float().reshape(-1)
        count = int(record["teacher_count"])
        if probabilities.numel() != 9 or count != int(probabilities.argmax().item()):
            raise ValueError(f"Invalid K/probabilities at {key}")
        if not torch.equal(probabilities, torch.nn.functional.one_hot(torch.tensor(count), 9).float()):
            raise ValueError(f"Expected one-hot API probability at {key}")
        evidence = record.get("api_evidence")
        if count >= 5 and not isinstance(evidence, dict):
            raise ValueError(f"K5-8 record lacks audit evidence at {key}")
        if count < 5 and evidence is not None:
            raise ValueError(f"K0-4 record unexpectedly has high evidence at {key}")
        if isinstance(evidence, dict) and not str(evidence.get("reason", "")).strip():
            raise ValueError(f"Empty audit evidence at {key}")
        histogram[count] += 1
    if not actual.issubset(expected):
        raise ValueError(f"Unexpected keys: {sorted(actual - expected)[:8]}")
    if not args.allow_subset and actual != expected:
        raise ValueError(f"Coverage mismatch: missing={len(expected - actual)} extra={len(actual - expected)}")
    print(json.dumps({
        "status": "PASS",
        "records": len(actual),
        "expected_records": len(expected),
        "missing": len(expected - actual),
        "histogram_n0_to_n8": [histogram[index] for index in range(9)],
        "teacher_model": payload.get("teacher_model"),
        "teacher_method": payload.get("teacher_method"),
        "preprocess": preprocess,
    }, indent=2))


if __name__ == "__main__":
    main()
