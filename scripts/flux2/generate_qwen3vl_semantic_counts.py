#!/usr/bin/env python3
"""Create task-semantic K=0..8 labels for every ImageWAM replan frame.

Each episode is annotated with one multimodal prefill. The VLM sees every
replan-aligned timestep in chronological order, with the same three-camera
composite used by ImageWAM, then emits one constrained K=0..8 decision per
displayed frame. This preserves direct nine-class probabilities while avoiding
re-encoding the same episode context for every target frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import av
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
IMAGEWAM_LAYOUT = "head_top_left_wrist_bottom_right_wrist"

RUBRIC = """You are an expert robot-video keyframe and visual-memory-token annotator.

You will receive every replan-aligned observation from one robot episode in
chronological order. Annotate EVERY displayed observation. Compare frames across
the complete episode before assigning counts.

CRITICAL REQUIREMENTS:
- Follow the task instruction as the primary objective definition.
- Judge whether information in each frame must be preserved for later task success.
- Camera motion, robot motion, visual novelty, or a busy image alone do NOT justify
  a high count.
- Keep high counts sparse. Counts 5-8 require concrete task-critical evidence
  that affects a later task decision; a visually salient frame is not enough.
- A visually static frame can be important only when it clearly preserves task state
  needed after an occlusion, placement, comparison, or repeated cycle.

Ordinal memory-token rubric:
- 0: redundant observation; no task-relevant information needs to be preserved.
- 1-2: minor motion or weak evidence; no meaningful state transition.
- 3-4: useful approach/alignment, local progress, or a task-relevant transition
  that does not need long-term preservation.
- 5-6: stable evidence or a completed state transition that a later task phase
  must use after the state becomes hidden, changed, or ambiguous.
- 7-8: irreplaceable identity, number, ordering, pose, location, or outcome
  evidence whose loss would plausibly cause a later task decision to fail.

Sequence-level calibration:
- Ordinary movement and repeated unchanged states should usually be 0-2.
- Use 5-8 only when you can identify the specific task state being preserved.
- Do not inflate counts to distribute labels evenly across 0-8.
- Apply the same standard throughout the episode.
- Do not invent K=5-8 merely to ensure that an episode contains a high label.
- Preparation, approach, alignment, a robot merely being positioned, motion in
  progress, and a terminal state with no later use are never K=5-8.
- Judge every output independently; do not copy the preceding digit merely because
  adjacent frames look similar.

Task-grounded examples:
- If an object identity, number, original pose, ordering, or arrangement will later
  be hidden or changed, its last clear observation before that loss is K=7-8.
- A stable grasp, completed placement, reveal, button confirmation, or completed
  cycle is K=5-6 only when later task steps depend on remembering that state.
- Screen motion, free-space arm motion, repeated unchanged views, and approach
  motion without a completed transition remain K=0-2.
- For repeated trials, preserve the successful result or newly established state,
  not every attempt.

Visual layout:
- Each image is one ImageWAM three-camera composite.
- The global head camera is the large top panel.
- The left- and right-wrist cameras are the lower-left and lower-right panels.

Output constraints:
- Return exactly one ASCII digit 0-8 for each displayed replan observation.
- Preserve the displayed chronological order.
- Separate adjacent digits with one comma.
- Return no JSON, spaces, markdown, explanation, or extra text.
- The independent evidence review is exhaustive for K>=3: events planned at
  importance 3-4 must receive K=3-4, events planned at importance 5-8 must receive
  K=5-8, and all unlisted frames must receive K=0-2.
"""

PLANNER_SYSTEM = """You are an expert visual-evidence planner for robot episodes.
Select a sparse set of displayed frames with concrete task-relevant visual evidence.
Use importance 3-4 for useful local progress, approach/alignment, or a visible state
transition that does not need long-term preservation. Use importance 5-6 only for a
stable state or completed transition that a later task phase must use after it becomes
hidden, changed, or ambiguous. Use importance 7-8 only for irreplaceable identity,
number, ordering, pose, location, or outcome evidence whose loss could make a later
task decision fail. Prefer the last clear evidence before information loss. Never
assign 5-8 to preparation, approach, a robot merely being positioned, motion in
progress, generic initial scene context, or a terminal state with no later use.
Every selected frame needs a concrete reason tied to the task instruction. Select an
initial state at 5-8 only when its specific content must be remembered later. Do not
invent a 5-8 event merely to include a high-importance frame."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--action-chunk-size", type=int, default=16)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=16,
        help="Replan stride. ImageWAM RMBench training uses 16.",
    )
    parser.add_argument("--max-pixels", type=int, default=160 * 224)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--planner-retries", type=int, default=2)
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument(
        "--episode-ids",
        type=str,
        default=None,
        help="Optional comma-separated exact episode IDs for deterministic probes.",
    )
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
    return (
        root
        / "videos"
        / f"chunk-{episode_id // 1000:03d}"
        / camera_key
        / f"episode_{episode_id:06d}.mp4"
    )


def resize_for_teacher(image: Image.Image, max_pixels: int) -> Image.Image:
    width, height = image.size
    scale = min(1.0, math.sqrt(float(max_pixels) / float(width * height)))
    new_width = max(28, int(width * scale) // 28 * 28)
    new_height = max(28, int(height * scale) // 28 * 28)
    if (new_width, new_height) == image.size:
        return image
    return image.resize((new_width, new_height), Image.Resampling.BILINEAR)


def compose_imagewam_views(
    head: Image.Image,
    left_wrist: Image.Image,
    right_wrist: Image.Image,
) -> Image.Image:
    """Match RobotVideoDataset's compact_288x256 camera layout."""
    head = head.resize((256, 192), Image.Resampling.BILINEAR)
    left_wrist = left_wrist.resize((128, 96), Image.Resampling.BILINEAR)
    right_wrist = right_wrist.resize((128, 96), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (256, 288), "black")
    canvas.paste(head, (0, 0))
    canvas.paste(left_wrist, (0, 192))
    canvas.paste(right_wrist, (128, 192))
    return canvas


def replan_timesteps(total_frames: int, action_chunk_size: int, frame_stride: int) -> list[int]:
    usable_length = int(total_frames) - int(action_chunk_size)
    if usable_length <= 0:
        return []
    return list(range(0, usable_length, int(frame_stride)))


def decode_episode_frames(
    root: Path,
    episode_id: int,
    timesteps: list[int],
    *,
    max_pixels: int,
) -> dict[int, Image.Image]:
    if not timesteps:
        return {}
    paths = [video_path(root, camera_key, episode_id) for camera_key in CAMERA_KEYS]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing episode camera videos: {missing}")
    wanted = set(int(timestep) for timestep in timesteps)
    last_wanted = max(wanted)
    camera_frames: list[dict[int, Image.Image]] = [dict() for _ in CAMERA_KEYS]
    containers = [av.open(str(path)) for path in paths]
    try:
        for output, container in zip(camera_frames, containers, strict=True):
            for index, frame in enumerate(container.decode(video=0)):
                if index in wanted:
                    output[index] = frame.to_image().convert("RGB")
                if index >= last_wanted:
                    break
    finally:
        for container in containers:
            container.close()
    for camera_key, decoded in zip(CAMERA_KEYS, camera_frames, strict=True):
        missing_timesteps = wanted.difference(decoded)
        if missing_timesteps:
            raise RuntimeError(
                f"episode {episode_id} camera {camera_key} is missing frames "
                f"{sorted(missing_timesteps)[:8]}"
            )
    return {
        timestep: resize_for_teacher(
            compose_imagewam_views(*(frames[timestep] for frames in camera_frames)),
            max_pixels,
        )
        for timestep in timesteps
    }


def task_text(episode: dict[str, Any]) -> str:
    tasks = episode.get("tasks", [])
    if len(tasks) >= 2:
        return str(tasks[1])
    if tasks:
        return str(tasks[0])
    return "Complete the demonstrated robot manipulation task."


def make_conversation(
    observations: list[tuple[int, Image.Image]],
    *,
    episode_id: int,
    total_frames: int,
    task: str,
    evidence_plan: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Episode metadata: id={episode_id}; task={task}; total_frames={total_frames}.\n"
                f"There are {len(observations)} replan observations. Annotate all of them.\n"
                "Ordered replan observations follow:"
            ),
        }
    ]
    for position, (timestep, image) in enumerate(observations):
        content.append(
            {
                "type": "text",
                "text": f"Replan {position + 1}/{len(observations)}: frame {timestep}",
            }
        )
        content.append({"type": "image", "image": image})
    content.append(
        {
            "type": "text",
            "text": (
                "Independent evidence review:\n"
                f"{json.dumps({'events': evidence_plan}, ensure_ascii=True, separators=(',', ':'))}\n"
                "Use this review to calibrate the high-K frames.\n"
                f"Return exactly {len(observations)} comma-separated digits in the same order.\n"
                "Memory-token counts:"
            ),
        }
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": RUBRIC}]},
        {"role": "user", "content": content},
    ]


def make_planner_conversation(
    observations: list[tuple[int, Image.Image]],
    *,
    episode_id: int,
    total_frames: int,
    task: str,
    repair_message: str | None = None,
) -> list[dict[str, Any]]:
    timesteps = [timestep for timestep, _image in observations]
    max_events = min(
        len(observations),
        max(8, min(16, math.ceil(len(observations) / 8))),
    )
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Episode metadata: id={episode_id}; task={task}; total_frames={total_frames}.\n"
                f"Select 1-{max_events} sparse task-relevant evidence frames from these exact "
                f"replan timesteps: {timesteps}.\n"
                "Return JSON only with this schema: "
                '{"events":[{"step":int,"importance":3,"reason":"specific visible evidence and later use"}]}.\n'
                "importance must be an integer from 3 through 8. step must be one of the "
                "displayed timesteps."
            ),
        }
    ]
    for position, (timestep, image) in enumerate(observations):
        content.append(
            {
                "type": "text",
                "text": f"Replan {position + 1}/{len(observations)}: frame {timestep}",
            }
        )
        content.append({"type": "image", "image": image})
    if repair_message:
        content.append(
            {
                "type": "text",
                "text": f"Your previous output was invalid: {repair_message}. Return corrected JSON only.",
            }
        )
    return [
        {"role": "system", "content": [{"type": "text", "text": PLANNER_SYSTEM}]},
        {"role": "user", "content": content},
    ]


def parse_evidence_plan(
    output_text: str,
    valid_timesteps: list[int],
) -> list[dict[str, Any]]:
    start = output_text.find("{")
    end = output_text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object found")
    payload = json.loads(output_text[start : end + 1])
    raw_events = payload.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError("events must be a non-empty list")
    max_events = min(
        len(valid_timesteps),
        max(8, min(16, math.ceil(len(valid_timesteps) / 8))),
    )
    if len(raw_events) > max_events:
        raise ValueError("too many evidence events")
    valid_set = set(valid_timesteps)
    events_by_step: dict[int, dict[str, Any]] = {}
    for raw_event in raw_events:
        if not isinstance(raw_event, dict):
            raise ValueError("each event must be an object")
        step = int(raw_event.get("step"))
        importance = int(raw_event.get("importance"))
        reason = str(raw_event.get("reason", "")).strip()
        if step not in valid_set:
            raise ValueError(f"event step {step} is not a displayed replan timestep")
        if not 3 <= importance <= 8:
            raise ValueError(f"event importance {importance} is outside 3-8")
        if not reason:
            raise ValueError(f"event step {step} has no reason")
        event = {"step": step, "importance": importance, "reason": reason}
        previous = events_by_step.get(step)
        if previous is None or importance > int(previous["importance"]):
            events_by_step[step] = event
    return [events_by_step[step] for step in sorted(events_by_step)]


def plan_episode(
    model: Qwen3VLForConditionalGeneration,
    processor: Any,
    observations: list[tuple[int, Image.Image]],
    *,
    episode_id: int,
    total_frames: int,
    task: str,
    retries: int,
) -> tuple[list[dict[str, Any]], str]:
    repair_message = None
    last_output = ""
    for _attempt in range(max(int(retries), 0) + 1):
        conversation = make_planner_conversation(
            observations,
            episode_id=episode_id,
            total_frames=total_frames,
            task=task,
            repair_message=repair_message,
        )
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                use_cache=True,
                max_new_tokens=512,
                pad_token_id=(
                    processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
                ),
            )
        last_output = processor.tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        try:
            return (
                parse_evidence_plan(
                    last_output,
                    [timestep for timestep, _image in observations],
                ),
                last_output,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            repair_message = str(error)
    raise RuntimeError(
        f"episode {episode_id} planner failed after {max(int(retries), 0) + 1} attempts; "
        f"last_output={last_output!r}"
    )


def single_token_ids(processor: Any, values: list[str], kind: str) -> list[int]:
    ids: list[int] = []
    for value in values:
        token_ids = processor.tokenizer.encode(value, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"{kind} {value!r} is not one tokenizer token: {token_ids}")
        ids.append(int(token_ids[0]))
    if len(set(ids)) != len(ids):
        raise ValueError(f"{kind} token IDs must be unique: {ids}")
    return ids


def annotate_episode(
    model: Qwen3VLForConditionalGeneration,
    processor: Any,
    observations: list[tuple[int, Image.Image]],
    *,
    episode_id: int,
    total_frames: int,
    task: str,
    evidence_plan: list[dict[str, Any]],
    temperature: float,
    digit_ids: list[int],
    separator_id: int,
) -> tuple[list[torch.Tensor], str]:
    conversation = make_conversation(
        observations,
        episode_id=episode_id,
        total_frames=total_frames,
        task=task,
        evidence_plan=evidence_plan,
    )
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    prompt_length = int(inputs["input_ids"].shape[1])
    generated_length = 2 * len(observations) - 1
    planned_importance = {
        int(event["step"]): int(event["importance"]) for event in evidence_plan
    }

    def allowed_tokens(_batch_id: int, input_ids: torch.Tensor) -> list[int]:
        output_position = int(input_ids.shape[-1]) - prompt_length
        if output_position % 2 == 1:
            return [separator_id]
        label_index = output_position // 2
        timestep = int(observations[label_index][0])
        importance = planned_importance.get(timestep)
        if importance is None:
            return digit_ids[:3]
        if importance <= 4:
            return digit_ids[3:5]
        return digit_ids[5:]

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            use_cache=True,
            max_new_tokens=generated_length,
            min_new_tokens=generated_length,
            prefix_allowed_tokens_fn=allowed_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=pad_token_id,
        )
    output_ids = generated.sequences[0, prompt_length:].detach().cpu().tolist()
    if len(output_ids) != generated_length or len(generated.scores) != generated_length:
        raise RuntimeError(
            f"episode {episode_id} generated {len(output_ids)} tokens and "
            f"{len(generated.scores)} score rows, expected {generated_length}"
        )
    digit_ids_cpu = torch.tensor(digit_ids, dtype=torch.long)
    probabilities: list[torch.Tensor] = []
    for position in range(generated_length):
        token_id = int(output_ids[position])
        if position % 2 == 1:
            if token_id != separator_id:
                raise RuntimeError(
                    f"episode {episode_id} output position {position} is not the separator"
                )
            continue
        digit_logits = generated.scores[position][0].detach().float().cpu().index_select(
            0, digit_ids_cpu
        )
        digit_probabilities = F.softmax(digit_logits / float(temperature), dim=0)
        predicted_id = digit_ids[int(digit_probabilities.argmax().item())]
        if token_id != predicted_id:
            raise RuntimeError(
                f"episode {episode_id} constrained digit disagrees with score argmax at "
                f"output position {position}: token={token_id} argmax={predicted_id}"
            )
        probabilities.append(digit_probabilities)
    if len(probabilities) != len(observations):
        raise RuntimeError(
            f"episode {episode_id} produced {len(probabilities)} labels for "
            f"{len(observations)} observations"
        )
    output_text = processor.tokenizer.decode(output_ids, skip_special_tokens=False)
    return probabilities, output_text


def load_existing(path: Path, metadata: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    if not path.is_file():
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "samta_semantic_change_curve_v1":
        raise ValueError(f"Cannot resume unexpected label cache format in {path}")
    for key in ("teacher_model", "teacher_prompt_hash", "preprocess_hash", "semantic_scorer_hash"):
        if payload.get(key) != metadata[key]:
            raise ValueError(f"Cannot resume cache with mismatched {key}")
    return {(int(record["episode_id"]), int(record["timestep"])): record for record in payload["records"]}


def atomic_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if not 0 <= args.rank < args.world_size:
        raise ValueError("Invalid rank/world-size")
    if args.temperature <= 0 or args.frame_stride <= 0 or args.action_chunk_size <= 0:
        raise ValueError("temperature, frame-stride, and action-chunk-size must be positive")
    torch.set_num_threads(max(int(args.cpu_threads), 1))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    model_config = args.model / "config.json"
    model_index = args.model / "model.safetensors.index.json"
    if not model_config.is_file() or not model_index.is_file():
        raise FileNotFoundError(f"Incomplete Qwen3-VL model directory: {args.model}")
    episodes_path = args.dataset_root / "meta" / "episodes.jsonl"
    preprocess = {
        "camera_keys": list(CAMERA_KEYS),
        "imagewam_layout": IMAGEWAM_LAYOUT,
        "annotation_unit": "episode_importance_plan_plus_three_band_replan_sequence",
        "context_sampling": "all_imagewam_replan_timesteps",
        "action_chunk_size": int(args.action_chunk_size),
        "frame_stride": int(args.frame_stride),
        "max_pixels": int(args.max_pixels),
        "temperature": float(args.temperature),
    }
    metadata = {
        "format": "samta_semantic_change_curve_v1",
        "k_max": 8,
        "teacher_model": "Qwen/Qwen3-VL-8B-Instruct",
        "teacher_prompt_hash": hashlib.sha256(
            (PLANNER_SYSTEM + "\n" + RUBRIC).encode("utf-8")
        ).hexdigest(),
        "probe_seeds": [0],
        "source_checkpoint_sha256": sha256_json(
            {"label_source": "qwen3vl_episode_replan_sequence"}
        ),
        "preprocess_hash": sha256_json(preprocess),
        "semantic_scorer_hash": sha256_json(
            {
                "config_sha256": sha256_file(model_config),
                "weight_index_sha256": sha256_file(model_index),
                "method": "qwen3vl_8b_hierarchical_replan_sequence_v4",
            }
        ),
        "teacher_method": "qwen3vl_8b_hierarchical_replan_sequence_v4",
        "dataset_episodes_sha256": sha256_file(episodes_path),
        "preprocess": preprocess,
    }
    records = load_existing(args.output, metadata)
    requested_episode_ids = None
    if args.episode_ids:
        requested_episode_ids = {
            int(value.strip()) for value in args.episode_ids.split(",") if value.strip()
        }
    episodes = [
        episode
        for episode in read_jsonl(episodes_path)
        if (
            int(episode["episode_index"]) in requested_episode_ids
            if requested_episode_ids is not None
            else int(episode["episode_index"]) % args.world_size == args.rank
        )
    ]
    if requested_episode_ids is not None:
        found_ids = {int(episode["episode_index"]) for episode in episodes}
        missing_ids = sorted(requested_episode_ids.difference(found_ids))
        if missing_ids:
            raise ValueError(f"Requested episode IDs are unavailable: {missing_ids}")
    if args.limit_episodes is not None:
        episodes = episodes[: max(int(args.limit_episodes), 0)]

    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    digit_ids = single_token_ids(processor, [str(count) for count in range(9)], "digit")
    separator_id = single_token_ids(processor, [","], "separator")[0]
    if separator_id in digit_ids:
        raise ValueError("Separator token overlaps a digit token")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda").eval()

    started = time.monotonic()
    processed_now = 0
    processed_episodes = 0

    def save() -> None:
        payload = dict(metadata)
        payload["rank"] = int(args.rank)
        payload["world_size"] = int(args.world_size)
        payload["elapsed_seconds"] = float(time.monotonic() - started)
        payload["records"] = [records[key] for key in sorted(records)]
        atomic_save(args.output, payload)

    for episode in episodes:
        episode_id = int(episode["episode_index"])
        total_frames = int(episode["length"])
        target_timesteps = replan_timesteps(
            total_frames,
            action_chunk_size=args.action_chunk_size,
            frame_stride=args.frame_stride,
        )
        expected_keys = {(episode_id, timestep) for timestep in target_timesteps}
        if not expected_keys or expected_keys.issubset(records):
            continue
        frames = decode_episode_frames(
            args.dataset_root,
            episode_id,
            target_timesteps,
            max_pixels=args.max_pixels,
        )
        observations = [(timestep, frames[timestep]) for timestep in target_timesteps]
        task = task_text(episode)
        evidence_plan, planner_output = plan_episode(
            model,
            processor,
            observations,
            episode_id=episode_id,
            total_frames=total_frames,
            task=task,
            retries=args.planner_retries,
        )
        episode_probabilities, output_text = annotate_episode(
            model,
            processor,
            observations,
            episode_id=episode_id,
            total_frames=total_frames,
            task=task,
            evidence_plan=evidence_plan,
            temperature=args.temperature,
            digit_ids=digit_ids,
            separator_id=separator_id,
        )
        sequence_hash = sha256_json(
            {
                "episode_id": episode_id,
                "task": task,
                "replan_timesteps": target_timesteps,
                "output_text": output_text,
                "planner_output": planner_output,
                "evidence_plan": evidence_plan,
            }
        )
        added = 0
        evidence_by_step = {int(event["step"]): event for event in evidence_plan}
        for timestep, probabilities in zip(
            target_timesteps, episode_probabilities, strict=True
        ):
            key = (episode_id, timestep)
            if key in records:
                continue
            teacher_count = int(probabilities.argmax().item())
            records[key] = {
                "episode_id": episode_id,
                "timestep": timestep,
                "semantic_loss": (-probabilities.clamp_min(1.0e-12).log()).clone(),
                "teacher_probabilities": probabilities.clone(),
                "teacher_count": teacher_count,
                "planned_evidence": evidence_by_step.get(timestep),
                "change_text_sha256": sha256_json(
                    {
                        "episode_sequence_hash": sequence_hash,
                        "episode_id": episode_id,
                        "timestep": timestep,
                        "task": task,
                        "replan_timesteps": target_timesteps,
                    }
                ),
            }
            added += 1
        processed_now += added
        processed_episodes += 1
        save()
        elapsed = max(time.monotonic() - started, 1.0e-6)
        print(
            f"rank={args.rank} episode={episode_id} episodes={processed_episodes} "
            f"new={processed_now} episode_labels={len(target_timesteps)} total={len(records)} "
            f"rate={processed_now / elapsed:.4f}_labels_per_second "
            f"peak_gpu_gib={torch.cuda.max_memory_allocated() / (1024 ** 3):.3f} "
            f"plan={json.dumps(evidence_plan, ensure_ascii=True, separators=(',', ':'))} "
            f"counts={output_text}",
            flush=True,
        )
    save()
    print(f"rank={args.rank} complete records={len(records)} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
