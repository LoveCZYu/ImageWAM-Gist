#!/usr/bin/env python3
"""Label every ImageWAM replan frame with a direct GPT-5.6 K=0..8 decision."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import av
import torch
from PIL import Image


CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
IMAGEWAM_LAYOUT = "head_top_left_wrist_bottom_right_wrist"
MODEL_RUBRIC = """You are assigning visual-memory token counts to every replan frame in one
successful robot episode. Every displayed frame is both global chronological context
and an independent target that must receive exactly one integer K from 0 through 8.

Use the complete 0-8 scale directly for every frame:
- K0: completely redundant; no new task-relevant visual information.
- K1: near-duplicate with only a tiny task-relevant change.
- K2: ordinary motion or weak evidence with some task relevance.
- K3: useful local progress, approach, or alignment.
- K4: a clear local state change or transition that is useful but recoverable.
- K5: distinctly important state that may help a later task phase.
- K6: important stable evidence or transition that later behavior likely depends on.
- K7: critical evidence that is difficult to recover once changed or hidden.
- K8: irreplaceable task-defining evidence such as identity, number, ordering, pose,
  or location immediately before it is lost, hidden, or irreversibly changed.

Rules:
- Choose among all nine values independently for every frame. There is no planner,
  band, quota, interpolation, class balancing, or required histogram.
- Compare neighboring and distant frames across the full episode. Do not copy a K
  merely because adjacent frames look similar.
- Camera/robot motion, visual novelty, and a busy image alone do not justify high K.
- A high K requires concrete task relevance. Ordinary movement, preparation, and a
  robot merely being positioned should normally remain K0-K3.
- K5-K8 must preserve either a visibly stable/completed task state or the last clear
  evidence immediately before important information is lost. Pickup, transport,
  holding, rotation, contact, alignment, or placement still in progress should not
  exceed K4 unless that exact transient event is uniquely required and cannot be
  recovered from the following stable frame.
- Do not automatically score both sides of the same transition as high. Use K3-K4
  for recoverable intermediate motion and reserve K5-K8 for the unique evidence.
- Initial frames are high only when their specific visible content must be remembered.
- First fix the complete labels array. Only afterward write concise evidence for the
  frames already assigned K5-K8. Evidence is for audit and must not gate or change
  which K values were available during labeling.
- Do not force every K value to appear.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-base-url", default=os.environ.get("OPENAI_BASE_URL", "https://jojocode.com/v1"))
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh", "max"), default="high")
    parser.add_argument("--image-detail", choices=("low", "high", "original", "auto"), default="original")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--action-chunk-size", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=16)
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--episode-ids", type=str, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def video_path(root: Path, camera_key: str, episode_id: int) -> Path:
    return root / "videos" / f"chunk-{episode_id // 1000:03d}" / camera_key / f"episode_{episode_id:06d}.mp4"


def compose_imagewam_views(head: Image.Image, left: Image.Image, right: Image.Image) -> Image.Image:
    head = head.resize((256, 192), Image.Resampling.BILINEAR)
    left = left.resize((128, 96), Image.Resampling.BILINEAR)
    right = right.resize((128, 96), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (256, 288), "black")
    canvas.paste(head, (0, 0))
    canvas.paste(left, (0, 192))
    canvas.paste(right, (128, 192))
    return canvas


def replan_timesteps(total_frames: int, action_chunk_size: int, frame_stride: int) -> list[int]:
    return list(range(0, max(int(total_frames) - int(action_chunk_size), 0), int(frame_stride)))


def decode_episode_frames(root: Path, episode_id: int, timesteps: list[int]) -> dict[int, Image.Image]:
    if not timesteps:
        return {}
    paths = [video_path(root, camera_key, episode_id) for camera_key in CAMERA_KEYS]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing episode camera videos: {missing}")
    wanted = set(timesteps)
    outputs: list[dict[int, Image.Image]] = [dict() for _ in CAMERA_KEYS]
    containers = [av.open(str(path)) for path in paths]
    try:
        for output, container in zip(outputs, containers, strict=True):
            for index, frame in enumerate(container.decode(video=0)):
                if index in wanted:
                    output[index] = frame.to_image().convert("RGB")
                if index >= timesteps[-1]:
                    break
    finally:
        for container in containers:
            container.close()
    for camera_key, output in zip(CAMERA_KEYS, outputs, strict=True):
        absent = wanted.difference(output)
        if absent:
            raise RuntimeError(f"episode {episode_id} camera {camera_key} missing {sorted(absent)[:8]}")
    return {
        step: compose_imagewam_views(*(camera[step] for camera in outputs))
        for step in timesteps
    }


def task_text(episode: dict[str, Any]) -> str:
    tasks = episode.get("tasks", [])
    if len(tasks) >= 2:
        return str(tasks[1])
    if tasks:
        return str(tasks[0])
    return "Complete the demonstrated robot manipulation task."


def image_data_url(image: Image.Image, jpeg_quality: int) -> str:
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=jpeg_quality, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def output_schema(timesteps: list[int]) -> dict[str, Any]:
    step_schema = {"type": "integer", "enum": timesteps}
    label_item = {
        "type": "object",
        "properties": {
            "step": step_schema,
            "k": {"type": "integer", "enum": list(range(9))},
        },
        "required": ["step", "k"],
        "additionalProperties": False,
    }
    evidence_item = {
        "type": "object",
        "properties": {
            "step": step_schema,
            "reason": {"type": "string"},
        },
        "required": ["step", "reason"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": label_item,
                "minItems": len(timesteps),
                "maxItems": len(timesteps),
            },
            "high_evidence": {"type": "array", "items": evidence_item},
        },
        "required": ["labels", "high_evidence"],
        "additionalProperties": False,
    }


def make_request(
    *,
    model: str,
    reasoning_effort: str,
    image_detail: str,
    episode_id: int,
    total_frames: int,
    task: str,
    timesteps: list[int],
    frames: dict[int, Image.Image],
    jpeg_quality: int,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                f"Episode id={episode_id}; total_frames={total_frames}.\n"
                f"Task instruction: {task}\n"
                f"Label all {len(timesteps)} replan observations at exact steps {timesteps}.\n"
                "The observations follow in chronological order."
            ),
        }
    ]
    for index, step in enumerate(timesteps, start=1):
        content.append({"type": "input_text", "text": f"Replan {index}/{len(timesteps)}; step={step}"})
        content.append(
            {
                "type": "input_image",
                "image_url": image_data_url(frames[step], jpeg_quality),
                "detail": image_detail,
            }
        )
    content.append(
        {
            "type": "input_text",
            "text": (
                "Return every step exactly once in chronological order. Assign labels first using the full 0-8 scale. "
                "Then include high_evidence for exactly the steps assigned K5-K8."
            ),
        }
    )
    return {
        "model": model,
        "reasoning": {"effort": reasoning_effort},
        "instructions": MODEL_RUBRIC,
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "rmbench_replan_k_labels",
                "strict": True,
                "schema": output_schema(timesteps),
            }
        },
        "max_output_tokens": max(2048, 64 * len(timesteps)),
    }


def response_text(response: dict[str, Any]) -> str:
    texts = [
        str(content.get("text", ""))
        for output in response.get("output", [])
        for content in output.get("content", [])
        if content.get("type") == "output_text"
    ]
    if not texts:
        raise ValueError(f"API response has no output_text; status={response.get('status')!r}")
    return "".join(texts)


def validate_annotation(payload: dict[str, Any], timesteps: list[int]) -> tuple[dict[int, int], dict[int, str]]:
    labels = payload.get("labels")
    evidence = payload.get("high_evidence")
    if not isinstance(labels, list) or not isinstance(evidence, list):
        raise ValueError("labels and high_evidence must be arrays")
    label_map: dict[int, int] = {}
    for item in labels:
        step = int(item["step"])
        k = int(item["k"])
        if step in label_map or step not in timesteps or not 0 <= k <= 8:
            raise ValueError(f"invalid or duplicate label step={step} k={k}")
        label_map[step] = k
    if list(label_map) != timesteps:
        raise ValueError(f"label coverage/order mismatch: got={list(label_map)} expected={timesteps}")
    evidence_map: dict[int, str] = {}
    for item in evidence:
        step = int(item["step"])
        reason = str(item["reason"]).strip()
        if step in evidence_map or step not in label_map or not reason:
            raise ValueError(f"invalid high evidence step={step}")
        evidence_map[step] = reason
    expected_high = {step for step, k in label_map.items() if k >= 5}
    if set(evidence_map) != expected_high:
        raise ValueError(
            f"high evidence mismatch: missing={sorted(expected_high - set(evidence_map))} "
            f"extra={sorted(set(evidence_map) - expected_high)}"
        )
    return label_map, evidence_map


def call_api(
    *,
    api_key: str,
    api_base_url: str,
    request_payload: dict[str, Any],
    timesteps: list[int],
    retries: int,
    timeout_seconds: int,
) -> tuple[dict[int, int], dict[int, str], dict[str, Any], str]:
    body = json.dumps(request_payload, separators=(",", ":")).encode("utf-8")
    endpoint = api_base_url.rstrip("/") + "/responses"
    last_error: Exception | None = None
    for attempt in range(max(retries, 0) + 1):
        request = urllib.request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as result:
                response = json.loads(result.read().decode("utf-8"))
            output_text = response_text(response)
            annotation = json.loads(output_text)
            labels, evidence = validate_annotation(annotation, timesteps)
            return labels, evidence, response, output_text
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, TypeError, ValueError) as error:
            last_error = error
            if attempt >= max(retries, 0):
                break
            delay = min(60.0, (2.0 ** attempt) + random.random())
            time.sleep(delay)
    raise RuntimeError(f"API annotation failed after {max(retries, 0) + 1} attempts: {last_error}")


def atomic_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_existing(path: Path, metadata: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    if not path.is_file():
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("format", "teacher_model", "teacher_prompt_hash", "preprocess_hash", "semantic_scorer_hash"):
        if payload.get(key) != metadata[key]:
            raise ValueError(f"Cannot resume cache with mismatched {key}")
    return {(int(record["episode_id"]), int(record["timestep"])): record for record in payload["records"]}


def annotate_episode(
    episode: dict[str, Any],
    *,
    args: argparse.Namespace,
    api_key: str,
) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    episode_id = int(episode["episode_index"])
    total_frames = int(episode["length"])
    timesteps = replan_timesteps(total_frames, args.action_chunk_size, args.frame_stride)
    frames = decode_episode_frames(args.dataset_root, episode_id, timesteps)
    task = task_text(episode)
    request_payload = make_request(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        image_detail=args.image_detail,
        episode_id=episode_id,
        total_frames=total_frames,
        task=task,
        timesteps=timesteps,
        frames=frames,
        jpeg_quality=args.jpeg_quality,
    )
    started = time.monotonic()
    labels, evidence, response, output_text = call_api(
        api_key=api_key,
        api_base_url=args.api_base_url,
        request_payload=request_payload,
        timesteps=timesteps,
        retries=args.retries,
        timeout_seconds=args.timeout_seconds,
    )
    usage = response.get("usage", {})
    response_id = str(response.get("id", ""))
    response_model = str(response.get("model", args.model))
    sequence_hash = sha256_json(
        {"episode_id": episode_id, "task": task, "timesteps": timesteps, "output_text": output_text}
    )
    records: list[dict[str, Any]] = []
    for step in timesteps:
        k = labels[step]
        probabilities = torch.nn.functional.one_hot(torch.tensor(k), num_classes=9).float()
        api_evidence = None
        if step in evidence:
            api_evidence = {"step": step, "reason": evidence[step], "source": "post_label_audit"}
        records.append(
            {
                "episode_id": episode_id,
                "timestep": step,
                "semantic_loss": -probabilities.clamp_min(1.0e-12).log(),
                "teacher_probabilities": probabilities,
                "teacher_count": k,
                "planned_evidence": None,
                "api_evidence": api_evidence,
                "api_response_id": response_id,
                "api_response_model": response_model,
                "api_input_tokens": int(usage.get("input_tokens", 0) or 0),
                "api_output_tokens": int(usage.get("output_tokens", 0) or 0),
                "change_text_sha256": sha256_json(
                    {"episode_sequence_hash": sequence_hash, "episode_id": episode_id, "timestep": step}
                ),
            }
        )
    info = {
        "episode_id": episode_id,
        "labels": len(records),
        "counts": [labels[step] for step in timesteps],
        "high_evidence": evidence,
        "api_response_id": response_id,
        "api_response_model": response_model,
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "elapsed_seconds": time.monotonic() - started,
    }
    return episode_id, records, info


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    if not 0 <= args.rank < args.world_size or args.workers <= 0:
        raise ValueError("Invalid rank/world-size/workers")
    if args.frame_stride <= 0 or args.action_chunk_size <= 0 or not 1 <= args.jpeg_quality <= 100:
        raise ValueError("Invalid stride, action chunk size, or JPEG quality")
    episodes_path = args.dataset_root / "meta" / "episodes.jsonl"
    preprocess = {
        "camera_keys": list(CAMERA_KEYS),
        "imagewam_layout": IMAGEWAM_LAYOUT,
        "annotation_unit": "episode_direct_free_k0_to_k8",
        "context_sampling": "all_imagewam_replan_timesteps",
        "action_chunk_size": int(args.action_chunk_size),
        "frame_stride": int(args.frame_stride),
        "image_size": [256, 288],
        "image_detail": args.image_detail,
        "jpeg_quality": int(args.jpeg_quality),
        "probability_source": "one_hot_from_structured_k",
    }
    metadata = {
        "format": "samta_semantic_change_curve_v1",
        "k_max": 8,
        "teacher_model": args.model,
        "teacher_prompt_hash": hashlib.sha256(MODEL_RUBRIC.encode("utf-8")).hexdigest(),
        "probe_seeds": [0],
        "source_checkpoint_sha256": sha256_json({"api_base_url": args.api_base_url, "model": args.model}),
        "preprocess_hash": sha256_json(preprocess),
        "semantic_scorer_hash": sha256_json(
            {"method": "gpt56_api_episode_direct_free_k0_to_k8_v2", "model": args.model, "reasoning_effort": args.reasoning_effort}
        ),
        "teacher_method": "gpt56_api_episode_direct_free_k0_to_k8_v2",
        "dataset_episodes_sha256": sha256_file(episodes_path),
        "preprocess": preprocess,
        "api_base_url": args.api_base_url,
        "reasoning_effort": args.reasoning_effort,
    }
    records = load_existing(args.output, metadata)
    requested = None
    if args.episode_ids:
        requested = {int(value.strip()) for value in args.episode_ids.split(",") if value.strip()}
    episodes = [
        episode
        for episode in read_jsonl(episodes_path)
        if (
            int(episode["episode_index"]) in requested
            if requested is not None
            else int(episode["episode_index"]) % args.world_size == args.rank
        )
    ]
    if requested is not None:
        found = {int(episode["episode_index"]) for episode in episodes}
        if requested != found:
            raise ValueError(f"Requested episode IDs unavailable: {sorted(requested - found)}")
    if args.limit_episodes is not None:
        episodes = episodes[: max(args.limit_episodes, 0)]
    pending = []
    for episode in episodes:
        episode_id = int(episode["episode_index"])
        expected = {
            (episode_id, step)
            for step in replan_timesteps(int(episode["length"]), args.action_chunk_size, args.frame_stride)
        }
        if not expected.issubset(records):
            pending.append(episode)
    started = time.monotonic()

    def save() -> None:
        payload = dict(metadata)
        payload["rank"] = int(args.rank)
        payload["world_size"] = int(args.world_size)
        payload["elapsed_seconds"] = time.monotonic() - started
        payload["records"] = [records[key] for key in sorted(records)]
        atomic_save(args.output, payload)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(annotate_episode, episode, args=args, api_key=api_key): episode for episode in pending}
        for future in as_completed(futures):
            episode_id, episode_records, info = future.result()
            for record in episode_records:
                key = (int(record["episode_id"]), int(record["timestep"]))
                records[key] = record
            save()
            print(json.dumps({"status": "episode_complete", **info}, separators=(",", ":")), flush=True)
    save()
    print(f"rank={args.rank} complete records={len(records)} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
