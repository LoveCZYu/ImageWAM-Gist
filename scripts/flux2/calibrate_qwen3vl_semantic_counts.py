#!/usr/bin/env python3
"""Calibrate VLM K=0..8 probabilities to a target global hard-count mean."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-mean", type=float, default=4.5)
    parser.add_argument("--preserve-initial", type=int, default=1)
    return parser.parse_args()


def atomic_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def calibrated(log_probabilities: torch.Tensor, beta: float) -> torch.Tensor:
    counts = torch.arange(log_probabilities.shape[1], dtype=torch.float64)
    return torch.softmax(log_probabilities + float(beta) * counts[None, :], dim=1)


def main() -> None:
    args = parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    if payload.get("format") != "samta_semantic_change_curve_v1" or int(payload.get("k_max", -1)) != 8:
        raise ValueError("Input must be a samta_semantic_change_curve_v1 K=0..8 cache.")
    records = payload.get("records", [])
    if not records:
        raise ValueError("Input cache is empty.")
    raw = torch.stack(
        [torch.as_tensor(record["teacher_probabilities"], dtype=torch.float64) for record in records]
    )
    if raw.shape[1] != 9 or bool((raw < 0).any()) or not torch.isfinite(raw).all():
        raise ValueError("Invalid teacher_probabilities in input cache.")
    raw = raw / raw.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    log_raw = raw.clamp_min(1.0e-12).log()
    initial = torch.tensor(
        [int(record["timestep"]) < int(args.preserve_initial) for record in records],
        dtype=torch.bool,
    )

    def evaluate(beta: float) -> tuple[float, torch.Tensor, torch.Tensor]:
        probabilities = calibrated(log_raw, beta)
        probabilities[initial] = raw[initial]
        counts = probabilities.argmax(dim=1)
        counts[initial] = 0
        return float(counts.double().mean().item()), probabilities, counts

    low, high = -32.0, 32.0
    candidates: list[tuple[float, float]] = []
    for _ in range(80):
        mid = (low + high) / 2.0
        mean, probabilities, counts = evaluate(mid)
        candidates.append((abs(mean - args.target_mean), mid))
        if mean < args.target_mean:
            low = mid
        else:
            high = mid
    for beta in (low, high):
        mean, probabilities, counts = evaluate(beta)
        candidates.append((abs(mean - args.target_mean), beta))
    _, beta = min(candidates, key=lambda value: (value[0], abs(value[1])))
    _, probabilities, counts = evaluate(beta)

    raw_counts = raw.argmax(dim=1)
    raw_counts[initial] = 0
    count_values = torch.arange(9, dtype=torch.float64)
    expected_mean = float((probabilities * count_values[None, :]).sum(dim=1).mean().item())
    mean_kl = float(
        (probabilities * (probabilities.clamp_min(1.0e-12).log() - log_raw)).sum(dim=1).mean().item()
    )
    metadata = {
        "method": "ordinal_log_probability_bias_v1",
        "beta": float(beta),
        "target_hard_mean": float(args.target_mean),
        "preserve_initial_timesteps": int(args.preserve_initial),
        "raw_hard_mean": float(raw_counts.double().mean().item()),
        "calibrated_hard_mean": float(counts.double().mean().item()),
        "calibrated_expected_mean": expected_mean,
        "raw_histogram_n0_to_n8": [int((raw_counts == index).sum().item()) for index in range(9)],
        "calibrated_histogram_n0_to_n8": [int((counts == index).sum().item()) for index in range(9)],
        "flip_rate": float((counts != raw_counts).double().mean().item()),
        "mean_kl_calibrated_to_raw": mean_kl,
    }
    metadata["calibration_hash"] = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    output_records = []
    for record, raw_probability, probability, count in zip(records, raw, probabilities, counts):
        output = dict(record)
        output["raw_teacher_probabilities"] = raw_probability.float().clone()
        output["raw_teacher_count"] = int(raw_probability.argmax().item())
        if "semantic_loss" in output:
            output["raw_semantic_loss"] = torch.as_tensor(output["semantic_loss"]).float().clone()
        output["teacher_probabilities"] = probability.float().clone()
        output["teacher_count"] = int(count.item())
        output["semantic_loss"] = -probability.clamp_min(1.0e-12).log().float()
        output_records.append(output)
    result = dict(payload)
    result["records"] = output_records
    result["calibration"] = metadata
    atomic_save(args.output, result)
    print(json.dumps({"records": len(records), **metadata}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
