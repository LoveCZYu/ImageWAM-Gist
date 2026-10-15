#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from imagewam.models.backbones.action_dit_sana import ActionDiTSana
from imagewam.models.backbones.dim_video_expert import DimVideoExpert


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_tensor_to_shape(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(src.shape) == target_shape:
        return src
    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(f"Cannot reduce tensor rank: src={tuple(src.shape)} target={target_shape}")
        out = out.squeeze(0)
    for dim, new_size in enumerate(target_shape):
        if out.shape[dim] == new_size:
            continue
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        inv_perm = [0] * out.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        out = _interpolate_last_dim(out.permute(*perm).contiguous(), new_size).permute(*inv_perm).contiguous()
    return out.to(dtype=src.dtype)


def _is_unresolved(value: Any) -> bool:
    return isinstance(value, str) and "${" in value and "}" in value


def _load_action_cfg(path: str, action_dim: int, use_gradient_checkpointing: bool) -> dict[str, Any]:
    cfg = OmegaConf.load(path)
    action_cfg = cfg.model.action_dit_config if "model" in cfg else cfg.action_dit_config
    action_cfg = OmegaConf.to_container(action_cfg, resolve=False)
    if not isinstance(action_cfg, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_cfg)}")
    if _is_unresolved(action_cfg.get("action_dim")):
        action_cfg["action_dim"] = int(action_dim)
    if _is_unresolved(action_cfg.get("use_gradient_checkpointing")):
        action_cfg["use_gradient_checkpointing"] = bool(use_gradient_checkpointing)
    unresolved = {key: value for key, value in action_cfg.items() if _is_unresolved(value)}
    if unresolved:
        raise ValueError(f"Unresolved action_dit_config values remain: {unresolved}")
    return action_cfg


def _candidate_sana_keys(action_key: str) -> list[str]:
    if not action_key.startswith("blocks."):
        return []
    parts = action_key.split(".")
    if len(parts) < 3:
        return []
    block_prefix = ".".join(parts[:2])
    tail = ".".join(parts[2:])
    mapping = {
        "qkv.weight": ["attn.qkv.weight"],
        "q_norm.weight": ["attn.q_norm.weight"],
        "k_norm.weight": ["attn.k_norm.weight"],
        "proj.weight": ["attn.proj.weight", "attn.proj_out.weight"],
        "cross_q.weight": ["cross_attn.q_linear.weight"],
        "cross_q.bias": ["cross_attn.q_linear.bias"],
        "cross_kv.weight": ["cross_attn.kv_linear.weight"],
        "cross_kv.bias": ["cross_attn.kv_linear.bias"],
        "cross_q_norm.weight": ["cross_attn.q_norm.weight"],
        "cross_k_norm.weight": ["cross_attn.k_norm.weight"],
        "cross_proj.weight": ["cross_attn.proj.weight"],
        "cross_proj.bias": ["cross_attn.proj.bias"],
        "scale_shift_table": ["scale_shift_table"],
        # SANA configs use either a standard MLP or a GLUMBConv-style 1x1 conv
        # stack. Both can provide a useful initialization after shape resize.
        "mlp.0.weight": [
            "mlp.fc1.weight",
            "mlp.inverted_conv.conv.weight",
            "mlp.inverted_conv.weight",
        ],
        "mlp.2.weight": [
            "mlp.fc2.weight",
            "mlp.point_conv.conv.weight",
            "mlp.point_conv.weight",
        ],
    }
    return [f"{block_prefix}.{src_tail}" for src_tail in mapping.get(tail, [])]


def _prepare_source_tensor(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    out = src
    # Convert 1x1 Conv weights used by SANA GLUMBConv into linear weights.
    while out.ndim > len(target_shape) and out.shape[-1] == 1:
        out = out.squeeze(-1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize SANA-shaped ActionDiT from DIM/SANA weights.")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--dim-model-path", required=True)
    parser.add_argument("--sana-config-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--residual-dim", type=int, default=None)
    parser.add_argument("--use-gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    video_expert = DimVideoExpert.from_pretrained(
        dim_model_path=args.dim_model_path,
        sana_config_path=args.sana_config_path,
        device=args.device,
        torch_dtype=torch.float32,
    )
    action_cfg = _load_action_cfg(
        args.model_config,
        action_dim=args.action_dim,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
    )
    if args.residual_dim is not None:
        action_cfg["residual_dim"] = int(args.residual_dim)
    action_cfg["attn_dim"] = int(video_expert.hidden_dim)
    action_cfg["context_dim"] = int(video_expert.hidden_dim)
    action_cfg["num_heads"] = int(video_expert.num_heads)
    action_cfg["num_layers"] = len(video_expert.blocks)
    action_expert = ActionDiTSana(**action_cfg).to(args.device)

    src_state = video_expert.model.state_dict()
    dst_state = action_expert.state_dict()
    out_state = {}
    copied = 0
    resized = 0
    initialized = 0
    initialized_keys = []
    for key, dst in dst_state.items():
        if key.startswith(("action_encoder.", "head.", "head_norm.", "time_proj.", "timestep_embedder.")):
            out_state[key] = dst.detach().cpu().contiguous()
            initialized += 1
            initialized_keys.append(key)
            continue
        src = None
        for src_key in _candidate_sana_keys(key):
            if src_key in src_state:
                src = _prepare_source_tensor(src_state[src_key], tuple(dst.shape))
                break
        if src is None:
            out_state[key] = dst.detach().cpu().contiguous()
            initialized += 1
            initialized_keys.append(key)
            continue
        if tuple(src.shape) == tuple(dst.shape):
            value = src
            copied += 1
        else:
            value = _resize_tensor_to_shape(src, tuple(dst.shape))
            resized += 1
        out_state[key] = value.detach().cpu().to(dtype=dst.dtype).contiguous()

    payload = {
        "state_dict": out_state,
        "meta": {
            "source": args.dim_model_path,
            "copied": copied,
            "resized": resized,
            "initialized": initialized,
            "initialized_keys": initialized_keys,
            "policy": "copy/resize SANA attention, cross-attention, modulation, and MLP where available; initialize action-specific and unmapped weights from ActionDiTSana defaults",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"[ok] saved {output} copied={copied} resized={resized} initialized={initialized} keys={len(out_state)}")
    if initialized_keys:
        print("[initialized_keys]")
        for key in initialized_keys:
            print(key)


if __name__ == "__main__":
    main()
