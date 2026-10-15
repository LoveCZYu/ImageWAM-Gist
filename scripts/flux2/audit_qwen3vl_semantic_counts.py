#!/usr/bin/env python3
"""Audit cached VLM K=0..8 labels and render K=0 plus K=5..8 examples."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import av
import torch
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples-per-group", type=int, default=48)
    parser.add_argument("--lookback", type=int, default=16)
    parser.add_argument("--minimum-spacing", type=int, default=32)
    return parser.parse_args()


CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def video_path(root: Path, camera_key: str, episode_id: int) -> Path:
    return (
        root
        / "videos"
        / f"chunk-{episode_id // 1000:03d}"
        / camera_key
        / f"episode_{episode_id:06d}.mp4"
    )


def compose_imagewam_views(
    head: Image.Image,
    left_wrist: Image.Image,
    right_wrist: Image.Image,
) -> Image.Image:
    head = head.resize((256, 192), Image.Resampling.BILINEAR)
    left_wrist = left_wrist.resize((128, 96), Image.Resampling.BILINEAR)
    right_wrist = right_wrist.resize((128, 96), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (256, 288), "black")
    canvas.paste(head, (0, 0))
    canvas.paste(left_wrist, (0, 192))
    canvas.paste(right_wrist, (128, 192))
    return canvas


def task_text(episode: dict) -> str:
    tasks = episode.get("tasks", [])
    if len(tasks) >= 2:
        return str(tasks[1])
    return str(tasks[0]) if tasks else "unknown"


def frame_triplet(
    root: Path, episode_id: int, timestep: int, episode_length: int, lookback: int
) -> Image.Image:
    indices = [max(0, timestep - lookback), timestep, min(episode_length - 1, timestep + lookback)]
    wanted = set(indices)
    camera_frames: list[dict[int, Image.Image]] = [dict() for _ in CAMERA_KEYS]
    containers = [av.open(str(video_path(root, key, episode_id))) for key in CAMERA_KEYS]
    try:
        for output, container in zip(camera_frames, containers):
            for index, frame in enumerate(container.decode(video=0)):
                if index in wanted:
                    output[index] = frame.to_image().convert("RGB")
                if len(output) == len(wanted):
                    break
    finally:
        for container in containers:
            container.close()
    images = [
        compose_imagewam_views(*(frames[index] for frames in camera_frames))
        for index in indices
    ]
    width, height = images[0].size
    canvas = Image.new("RGB", (width * 3, height + 24), "black")
    draw = ImageDraw.Draw(canvas)
    for column, (index, image) in enumerate(zip(indices, images)):
        canvas.paste(image, (column * width, 24))
        draw.text((column * width + 4, 4), f"t={index}", fill="white")
    return canvas


def render_group(items: list[dict], root: Path, output: Path, title: str) -> None:
    if not items:
        return
    tile_width, tile_height, columns = 720, 330, 3
    rendered: list[Image.Image] = []
    for item in items:
        triplet = frame_triplet(
            root,
            item["episode_id"],
            item["timestep"],
            item["episode_length"],
            item["lookback"],
        )
        triplet.thumbnail((tile_width, 220), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_width, tile_height), "white")
        tile.paste(triplet, (0, 34))
        draw = ImageDraw.Draw(tile)
        draw.text(
            (5, 5),
            f"K={item['count']} ep={item['episode_id']} t={item['timestep']} "
            f"p={item['confidence']:.3f} margin={item['margin']:.3f} H={item['entropy']:.3f}",
            fill="black",
        )
        draw.text((5, 265), item["task"][:105], fill="black")
        if item.get("planned_reason"):
            draw.text((5, 285), f"evidence: {item['planned_reason'][:100]}", fill="black")
        rendered.append(tile)
    rows = (len(rendered) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height + 30), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((5, 5), title, fill="black")
    for index, tile in enumerate(rendered):
        row, column = divmod(index, columns)
        sheet.paste(tile, (column * tile_width, 30 + row * tile_height))
    sheet.save(output, quality=92)


def select_examples(items: list[dict], count: int, limit: int, spacing: int) -> list[dict]:
    eligible = [
        item
        for item in items
        if item["count"] == count and (count != 0 or item["timestep"] > 0)
    ]
    half = max(limit // 2, 1)
    high = sorted(
        eligible,
        key=lambda item: (-item["margin"], item["task"], item["episode_id"], item["timestep"]),
    )[: half * 8]
    low = sorted(
        eligible,
        key=lambda item: (item["margin"], item["task"], item["episode_id"], item["timestep"]),
    )[: (limit - half) * 8]
    ordered = [item for pair in zip(high, low) for item in pair]
    selected: list[dict] = []
    for item in ordered:
        if item in selected:
            continue
        if any(
            previous["episode_id"] == item["episode_id"]
            and abs(previous["timestep"] - item["timestep"]) < spacing
            for previous in selected
        ):
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def aggregate(items: list[dict]) -> dict:
    histogram = Counter(item["count"] for item in items)
    return {
        "records": len(items),
        "histogram_n0_to_n8": [histogram[index] for index in range(9)],
        "hard_mean": sum(item["count"] for item in items) / max(len(items), 1),
        "expected_mean": sum(item["expected_count"] for item in items) / max(len(items), 1),
        "mean_entropy": sum(item["entropy"] for item in items) / max(len(items), 1),
        "mean_top1_margin": sum(item["margin"] for item in items) / max(len(items), 1),
    }


def main() -> None:
    args = parse_args()
    payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    if payload.get("format") != "samta_semantic_change_curve_v1":
        raise ValueError("Unexpected semantic cache format.")
    episodes = {
        int(row["episode_index"]): row
        for row in (
            json.loads(line)
            for line in (args.dataset_root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    items = []
    count_values = torch.arange(9, dtype=torch.float32)
    for record in payload.get("records", []):
        episode_id = int(record["episode_id"])
        probabilities = torch.as_tensor(record["teacher_probabilities"], dtype=torch.float32).reshape(-1)
        count = int(record["teacher_count"])
        if probabilities.numel() != 9 or count != int(probabilities.argmax().item()):
            raise ValueError(f"Invalid direct VLM label at episode={episode_id}, timestep={record['timestep']}")
        top = probabilities.topk(2).values
        items.append(
            {
                "episode_id": episode_id,
                "episode_length": int(episodes[episode_id]["length"]),
                "timestep": int(record["timestep"]),
                "count": count,
                "task": task_text(episodes[episode_id]),
                "lookback": int(args.lookback),
                "confidence": float(probabilities[count].item()),
                "margin": float((top[0] - top[1]).item()),
                "entropy": float(-(probabilities * probabilities.clamp_min(1.0e-12).log()).sum().item()),
                "expected_count": float((probabilities * count_values).sum().item()),
                "planned_reason": str(
                    (
                        record.get("api_evidence")
                        if isinstance(record.get("api_evidence"), dict)
                        else record.get("planned_evidence")
                        if isinstance(record.get("planned_evidence"), dict)
                        else {}
                    ).get("reason", "")
                ),
            }
        )
    if not items:
        raise ValueError("Cache is empty.")

    by_task: dict[str, list[dict]] = defaultdict(list)
    by_episode: dict[int, list[dict]] = defaultdict(list)
    for item in items:
        by_task[item["task"]].append(item)
        by_episode[item["episode_id"]].append(item)
    no_history = [item for item in items if item["timestep"] == 0]
    summary = {
        **aggregate(items),
        "episodes": len(by_episode),
        "tasks": len(by_task),
        "no_history_k0_fraction": sum(item["count"] == 0 for item in no_history) / max(len(no_history), 1),
        "k0_fraction": sum(item["count"] == 0 for item in items) / len(items),
        "k8_fraction": sum(item["count"] == 8 for item in items) / len(items),
        "k5_to_k8_fraction": sum(item["count"] >= 5 for item in items) / len(items),
        "teacher_model": payload.get("teacher_model"),
        "calibration": payload.get("calibration"),
        "per_task": {task: aggregate(values) for task, values in sorted(by_task.items())},
        "per_episode": {str(episode): aggregate(values) for episode, values in sorted(by_episode.items())},
    }
    selected = {
        f"k{count}": select_examples(items, count, args.examples_per_group, args.minimum_spacing)
        for count in (0, 5, 6, 7, 8)
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "examples.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    for count in (0, 5, 6, 7, 8):
        render_group(
            selected[f"k{count}"],
            args.dataset_root,
            args.output_dir / f"k{count}_exact_examples.jpg",
            f"Exact K={count} labels: semantic importance audit",
        )
    print(json.dumps({key: value for key, value in summary.items() if key not in {"per_task", "per_episode"}}, indent=2))


if __name__ == "__main__":
    main()
