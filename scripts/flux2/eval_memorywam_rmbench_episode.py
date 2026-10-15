#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from imagewam.utils.video_io import save_mp4
from imagewam.utils.video_metrics import video_psnr, video_ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inspect-dataset-only", action="store_true")
    parser.add_argument("--memory-fraction", type=float, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_fingerprint(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "min": float(value.float().min().item()) if value.numel() else None,
        "max": float(value.float().max().item()) if value.numel() else None,
        "mean": float(value.float().mean().item()) if value.numel() else None,
    }


def tensor_to_pil(tensor: torch.Tensor, label: str | None = None) -> Image.Image:
    value = tensor.detach().float().cpu().clamp(-1.0, 1.0)
    array = ((value + 1.0) * 127.5).round().to(torch.uint8)
    image = Image.fromarray(array.permute(1, 2, 0).numpy())
    if label:
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 20), fill=(0, 0, 0))
        draw.text((5, 4), label, fill=(255, 255, 255))
    return image


def pil_to_chw01(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def denormalize_action(
    processor: Any,
    action: torch.Tensor,
    proprio: torch.Tensor,
    embodiment: Any,
) -> torch.Tensor:
    action_btd = action.detach().cpu().float()
    if action_btd.ndim == 2:
        action_btd = action_btd.unsqueeze(0)
    state_btd = proprio.detach().cpu().float()
    if state_btd.ndim == 2:
        state_btd = state_btd.unsqueeze(0)
    batch = {
        "action": action_btd,
        "state": state_btd,
        "embodiment": embodiment,
    }
    batch = processor.action_state_merger.backward(batch)
    batch = processor.normalizer.backward(batch)
    action_meta = processor.shape_meta["action"]
    state_meta = processor.shape_meta["state"]
    merged = {
        "action": {
            meta["key"]: batch["action"][meta["key"]].squeeze(0)
            for meta in action_meta
        },
        "state": {
            meta["key"]: batch["state"][meta["key"]].squeeze(0)
            for meta in state_meta
        },
    }
    return processor.action_state_merger.forward(merged)["action"].float()


def action_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float | bool]:
    pred = pred.detach().cpu().float()
    target = target.detach().cpu().float()
    diff = pred - target
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    return {
        "finite": bool(torch.isfinite(pred).all().item()),
        "l1": float(diff.abs().mean().item()),
        "rmse": float(diff.square().mean().sqrt().item()),
        "max_abs": float(diff.abs().max().item()),
        "cosine": float(
            F.cosine_similarity(pred_flat[None], target_flat[None]).item()
        ),
        "pred_min": float(pred.min().item()),
        "pred_max": float(pred.max().item()),
        "pred_std": float(pred.std().item()),
        "gt_min": float(target.min().item()),
        "gt_max": float(target.max().item()),
        "gt_std": float(target.std().item()),
    }


def step_payload(sample: dict[str, Any], history_idx: int | None) -> dict[str, Any]:
    if history_idx is None:
        return {
            "video": sample["video"],
            "action": sample["action"],
            "proprio": sample["proprio"],
            "episode_id": int(sample["edit_memory_episode_id"].item()),
            "timestep": int(sample["edit_memory_timestep"].item()),
        }
    return {
        "video": sample["edit_memory_video"][history_idx],
        "action": sample["edit_memory_action"][history_idx],
        "proprio": sample["edit_memory_proprio"][history_idx],
        "episode_id": int(sample["edit_memory_history_episode_id"][history_idx].item()),
        "timestep": int(sample["edit_memory_history_timestep"][history_idx].item()),
    }


def infer_kwargs(
    sample: dict[str, Any],
    step: dict[str, Any],
    *,
    num_inference_steps: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "prompt": None,
        "input_image": step["video"][:, 0].unsqueeze(0),
        "action_horizon": int(step["action"].shape[0]),
        "proprio": step["proprio"][0],
        "context": sample["text_hidden_states"],
        "context_mask": sample["text_attention_mask"],
        "num_inference_steps": int(num_inference_steps),
        "seed": int(seed),
        "rand_device": "cpu",
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False

    cfg = OmegaConf.load(args.config)
    dataset = instantiate(cfg.data.val)
    if args.inspect_dataset_only:
        lengths = sorted(
            (
                int(dataset.edit_memory_history_length(index)),
                int(index),
            )
            for index in range(len(dataset))
        )
        print(
            json.dumps(
                {
                    "dataset_length": len(dataset),
                    "shortest": lengths[:10],
                    "median": lengths[len(lengths) // 2],
                    "longest": lengths[-10:],
                },
                sort_keys=True,
            )
        )
        return
    if args.memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(
            float(args.memory_fraction), torch.device(args.device)
        )
    model = instantiate(
        cfg.model,
        model_dtype=torch.bfloat16,
        device=args.device,
    )
    payload = model.load_checkpoint(
        str(args.checkpoint),
        strict_model_weights=True,
    )
    checkpoint_step = int(payload.get("step", -1))
    checkpoint_contract = dict(payload.get("edit_memory_contract", {}))
    del payload
    model.eval()

    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(
            f"sample-index={args.sample_index} is outside dataset length={len(dataset)}"
        )
    sample = dataset[args.sample_index]
    history_valid = sample["edit_memory_valid"].bool()
    history_indices = torch.nonzero(history_valid, as_tuple=False).flatten().tolist()
    processor = dataset.lerobot_dataset.processor
    model.reset_edit_memory(batch_size=1, device=model.device)

    torch.save(
        {
            "sample_index": int(args.sample_index),
            "episode_id": sample["edit_memory_episode_id"],
            "timestep": sample["edit_memory_timestep"],
            "video": sample["video"],
            "action": sample["action"],
            "proprio": sample["proprio"],
            "text_hidden_states": sample["text_hidden_states"],
            "text_attention_mask": sample["text_attention_mask"],
            "history_episode_id": sample["edit_memory_history_episode_id"],
            "history_timestep": sample["edit_memory_history_timestep"],
            "history_valid": history_valid,
        },
        args.output_dir / "parity_inputs.pt",
    )

    predicted_normalized = []
    target_normalized = []
    predicted_denormalized = []
    target_denormalized = []
    observation_frames = []
    memory_trace = []
    embodiment = sample.get("embodiment")

    for replan_idx, history_idx in enumerate(history_indices):
        step = step_payload(sample, history_idx)
        kwargs = infer_kwargs(
            sample,
            step,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
        )
        pred = model.infer_action_flux2(**kwargs)["action"]
        gt = step["action"].detach().cpu().float()
        pred_denorm = denormalize_action(
            processor, pred, step["proprio"], embodiment
        )
        gt_denorm = denormalize_action(
            processor, gt, step["proprio"], embodiment
        )
        predicted_normalized.append(pred.detach().cpu().float())
        target_normalized.append(gt)
        predicted_denormalized.append(pred_denorm)
        target_denormalized.append(gt_denorm)
        bank = model.edit_memory_bank
        gist_decision = getattr(model, "_edit_memory_last_gist_decision", None)
        memory_trace.append(
            {
                "replan": int(replan_idx),
                "episode_id": step["episode_id"],
                "timestep": step["timestep"],
                "entries": int(len(bank.entries)),
                "write": bool(bank.last_write_happened),
                "write_confidence": float(bank.last_write_confidence),
                "adaptive_gist_hard_count": (
                    None
                    if gist_decision is None
                    else int(gist_decision["hard_count"].item())
                ),
            }
        )
        observation_frames.append(
            tensor_to_pil(
                step["video"][:, 0],
                f"replan {replan_idx:03d}  t={step['timestep']}",
            )
        )
        print(
            json.dumps(
                {
                    "status": "history",
                    **memory_trace[-1],
                    "action_l1_denorm": action_metrics(pred_denorm, gt_denorm)["l1"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    terminal = step_payload(sample, None)
    terminal_kwargs = infer_kwargs(
        sample,
        terminal,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
    )
    bank_before_terminal = copy.deepcopy(model.edit_memory_bank)
    infer_step_before_terminal = int(model._edit_memory_infer_step_idx)
    joint_kwargs = {
        **terminal_kwargs,
        "num_frames": int(terminal["video"].shape[1]),
        "action": terminal["action"],
        "text_cfg_scale": 1.0,
        "action_cfg_scale": 1.0,
        "tiled": False,
    }
    first = model.infer(**joint_kwargs)
    model.edit_memory_bank = copy.deepcopy(bank_before_terminal)
    model._edit_memory_infer_step_idx = infer_step_before_terminal
    second = model.infer(**joint_kwargs)

    pred_action = first["action"].detach().cpu().float()
    pred_action_repeat = second["action"].detach().cpu().float()
    gt_action = terminal["action"].detach().cpu().float()
    pred_action_denorm = denormalize_action(
        processor, pred_action, terminal["proprio"], embodiment
    )
    pred_action_repeat_denorm = denormalize_action(
        processor, pred_action_repeat, terminal["proprio"], embodiment
    )
    gt_action_denorm = denormalize_action(
        processor, gt_action, terminal["proprio"], embodiment
    )
    predicted_normalized.append(pred_action)
    target_normalized.append(gt_action)
    predicted_denormalized.append(pred_action_denorm)
    target_denormalized.append(gt_action_denorm)

    pred_image = first["video"][0].convert("RGB")
    pred_image_repeat = second["video"][0].convert("RGB")
    current_image = tensor_to_pil(terminal["video"][:, 0], "CURRENT OBS")
    gt_image = tensor_to_pil(terminal["video"][:, -1], "GT NEXT")
    labeled_pred = pred_image.copy()
    draw = ImageDraw.Draw(labeled_pred)
    draw.rectangle((0, 0, labeled_pred.width, 20), fill=(0, 0, 0))
    draw.text((5, 4), "PRED NEXT", fill=(255, 255, 255))

    comparison = Image.new(
        "RGB",
        (current_image.width * 3, current_image.height),
    )
    comparison.paste(current_image, (0, 0))
    comparison.paste(labeled_pred, (current_image.width, 0))
    comparison.paste(gt_image, (current_image.width * 2, 0))
    comparison_frames = [comparison] * 32
    save_mp4(
        comparison_frames,
        str(args.output_dir / "terminal_current_pred_gt.mp4"),
        fps=8,
    )
    observation_frames.append(
        tensor_to_pil(
            terminal["video"][:, 0],
            f"terminal t={terminal['timestep']}",
        )
    )
    observation_frames.extend([labeled_pred] * 8)
    observation_frames.extend([gt_image] * 8)
    save_mp4(
        observation_frames,
        str(args.output_dir / "episode_replay_then_prediction.mp4"),
        fps=4,
    )
    labeled_pred.save(args.output_dir / "pred_next.png")
    current_image.save(args.output_dir / "current_obs.png")
    gt_image.save(args.output_dir / "gt_next.png")

    pred_video_tensor = pil_to_chw01(pred_image).unsqueeze(1)
    gt_video_tensor = (
        (terminal["video"][:, -1].detach().cpu().float().clamp(-1, 1) + 1.0)
        * 0.5
    ).unsqueeze(1)
    normalized_metrics = action_metrics(pred_action, gt_action)
    denormalized_metrics = action_metrics(pred_action_denorm, gt_action_denorm)
    repeat_action_diff = (pred_action - pred_action_repeat).abs()
    repeat_video_a = np.asarray(pred_image, dtype=np.int16)
    repeat_video_b = np.asarray(pred_image_repeat, dtype=np.int16)
    repeat_video_diff = np.abs(repeat_video_a - repeat_video_b)

    np.savez_compressed(
        args.output_dir / "actions.npz",
        predicted_normalized=torch.stack(predicted_normalized).numpy(),
        target_normalized=torch.stack(target_normalized).numpy(),
        predicted_denormalized=torch.stack(predicted_denormalized).numpy(),
        target_denormalized=torch.stack(target_denormalized).numpy(),
    )
    report = {
        "scope": {
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "checkpoint_step": checkpoint_step,
            "checkpoint_contract": checkpoint_contract,
            "sample_index": int(args.sample_index),
            "episode_id": terminal["episode_id"],
            "terminal_timestep": terminal["timestep"],
            "history_replans": len(history_indices),
            "seed": int(args.seed),
            "num_inference_steps": int(args.num_inference_steps),
            "dtype": "torch.bfloat16",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": torch.cuda.get_device_name(torch.device(args.device)),
        },
        "inputs": {
            key: tensor_fingerprint(sample[key])
            for key in (
                "video",
                "action",
                "proprio",
                "text_hidden_states",
                "text_attention_mask",
                "edit_memory_video",
                "edit_memory_action",
                "edit_memory_proprio",
                "edit_memory_valid",
            )
        },
        "memory_trace": memory_trace,
        "terminal": {
            "action_normalized": normalized_metrics,
            "action_denormalized": denormalized_metrics,
            "video_psnr": float(
                video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
            ),
            "video_ssim": float(
                video_ssim(pred=pred_video_tensor, target=gt_video_tensor)
            ),
            "repeat_action_exact": bool(torch.equal(pred_action, pred_action_repeat)),
            "repeat_action_max_abs": float(repeat_action_diff.max().item()),
            "repeat_action_mean_abs": float(repeat_action_diff.mean().item()),
            "repeat_denorm_action_exact": bool(
                torch.equal(pred_action_denorm, pred_action_repeat_denorm)
            ),
            "repeat_video_exact": bool(np.array_equal(repeat_video_a, repeat_video_b)),
            "repeat_video_max_pixel_diff": int(repeat_video_diff.max()),
            "first_pred_action_denormalized": pred_action_denorm[0].tolist(),
            "first_gt_action_denormalized": gt_action_denorm[0].tolist(),
        },
        "artifacts": {
            "comparison_video": str(args.output_dir / "terminal_current_pred_gt.mp4"),
            "episode_video": str(args.output_dir / "episode_replay_then_prediction.mp4"),
            "actions": str(args.output_dir / "actions.npz"),
            "inputs": str(args.output_dir / "parity_inputs.pt"),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report["terminal"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
