#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import torch
from safetensors import safe_open


def _safetensor_keys(path: Path) -> set[str]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return set(handle.keys())


def _tensor_shapes(value: object, prefix: str = "") -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    if isinstance(value, torch.Tensor):
        shapes[prefix or "<root>"] = tuple(int(dim) for dim in value.shape)
    elif isinstance(value, dict):
        for key, item in value.items():
            shapes.update(_tensor_shapes(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            shapes.update(_tensor_shapes(item, f"{prefix}[{index}]"))
    return shapes


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate fixed-8 Memory-ImageWAM initialization assets.")
    parser.add_argument("--flux2", type=Path, required=True)
    parser.add_argument("--ae", type=Path, required=True)
    parser.add_argument("--action-init", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--qwen-cache", type=Path, required=True)
    parser.add_argument("--context-len", type=int, default=512)
    args = parser.parse_args()

    for path in (args.flux2, args.ae, args.action_init):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing or empty model asset: {path}")

    flux_keys = _safetensor_keys(args.flux2)
    if not any("double_blocks" in key for key in flux_keys):
        raise RuntimeError("FLUX.2 checkpoint has no double-block weights.")
    if not any("single_blocks" in key for key in flux_keys):
        raise RuntimeError("FLUX.2 checkpoint has no single-block weights.")
    ae_keys = _safetensor_keys(args.ae)
    if not ae_keys:
        raise RuntimeError("AE checkpoint contains no tensors.")

    payload = torch.load(args.action_init, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise RuntimeError("Action initialization is not an official FLUX.2-derived state payload.")
    state = payload["state_dict"]
    meta = payload.get("meta", {})
    if not any(key.startswith("double_blocks.") for key in state):
        raise RuntimeError("Action initialization has no copied FLUX.2 double-block weights.")
    if not any(key.startswith("single_blocks.") for key in state):
        raise RuntimeError("Action initialization has no copied FLUX.2 single-block weights.")
    if any(key.startswith(("action_encoder.", "head.")) for key in state):
        raise RuntimeError("Action encoder/head must remain newly initialized in the official conversion policy.")
    source_name = Path(str(meta.get("source", ""))).name
    if source_name != args.flux2.name:
        raise RuntimeError(
            f"Action initialization source mismatch: meta={source_name!r}, flux2={args.flux2.name!r}"
        )

    info_path = args.dataset / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    episodes = int(info.get("total_episodes", 0))
    frames = int(info.get("total_frames", 0))
    if episodes <= 0 or frames <= 0:
        raise RuntimeError(f"Dataset metadata is incomplete: episodes={episodes}, frames={frames}")
    video_files = sorted((args.dataset / "videos").rglob("*.mp4"))
    if not video_files:
        raise RuntimeError("Dataset has no MP4 videos.")
    with av.open(str(video_files[0])) as container:
        first_frame = next(container.decode(video=0), None)
    if first_frame is None:
        raise RuntimeError(f"PyAV could not decode a frame from {video_files[0]}")
    expected_suffix = f".qwen3_flux2_len{args.context_len}.pt"
    cache_files = sorted(args.qwen_cache.glob(f"*{expected_suffix}"))
    tasks = int(info.get("total_tasks", 0))
    if len(cache_files) < tasks:
        raise RuntimeError(
            f"Qwen cache coverage is incomplete: files={len(cache_files)} tasks={tasks} suffix={expected_suffix}"
        )
    cache_sample = torch.load(cache_files[0], map_location="cpu", mmap=True, weights_only=False)
    cache_shapes = _tensor_shapes(cache_sample)
    if not cache_shapes or not any(args.context_len in shape for shape in cache_shapes.values()):
        raise RuntimeError(
            f"Qwen cache tensor contract does not contain context length {args.context_len}: {cache_shapes}"
        )

    print(
        "PASS assets "
        f"flux_tensors={len(flux_keys)} ae_tensors={len(ae_keys)} "
        f"action_tensors={len(state)} copied={meta.get('copied')} resized={meta.get('resized')} "
        f"episodes={episodes} frames={frames} qwen_cache_files={len(cache_files)} "
        f"qwen_sample_shapes={cache_shapes} pyav_frame={first_frame.width}x{first_frame.height}"
    )


if __name__ == "__main__":
    main()
