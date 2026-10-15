#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from imagewam.models.backbones.action_dit_omnigen2 import ActionDiTOmnigen2


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_tensor_to_shape(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(src.shape) == tuple(target_shape):
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
        raise ValueError(
            "Unresolved action_dit_config values remain. Pass explicit values or use a resolved config: "
            f"{unresolved}"
        )
    return action_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize OmniGen2-style ActionDiT from an OmniGen2 transformer.")
    parser.add_argument("--model-config", required=True, help="Hydra model/task yaml containing action_dit_config.")
    parser.add_argument("--omnigen2-model-path", required=True)
    parser.add_argument("--transformer-subfolder", default="transformer")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--use-gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel

    transformer = OmniGen2Transformer2DModel.from_pretrained(
        args.omnigen2_model_path,
        subfolder=args.transformer_subfolder,
    ).to(args.device)
    action_cfg = _load_action_cfg(
        args.model_config,
        action_dim=args.action_dim,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
    )
    action_cfg["num_layers"] = int(transformer.config.num_layers)
    action_cfg["num_heads"] = int(transformer.config.num_attention_heads)
    action_cfg["num_kv_heads"] = int(transformer.config.num_kv_heads)
    action_cfg["attn_head_dim"] = int(transformer.config.hidden_size) // int(transformer.config.num_attention_heads)
    action_expert = ActionDiTOmnigen2(**action_cfg).to(args.device)

    src_state = transformer.state_dict()
    dst_state = action_expert.state_dict()
    out_state = {}
    copied = 0
    resized = 0
    for key, dst in dst_state.items():
        if key.startswith(("action_encoder.", "head.", "action_freqs_cis")):
            continue
        src_key = key
        if key.startswith("blocks."):
            src_key = "layers." + key[len("blocks.") :]
        elif key.startswith(("time_proj.", "timestep_embedder.")):
            src_key = "time_caption_embed." + key
        if src_key not in src_state:
            continue
        src = src_state[src_key]
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
            "source": args.omnigen2_model_path,
            "copied": copied,
            "resized": resized,
            "policy": "copy matching OmniGen2 blocks, interpolate mismatched tensor axes",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"[ok] saved {output} copied={copied} resized={resized} keys={len(out_state)}")


if __name__ == "__main__":
    main()
