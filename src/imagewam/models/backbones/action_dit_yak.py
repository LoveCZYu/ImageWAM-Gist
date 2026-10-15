from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from imagewam.utils.logging_config import get_logger

from .ovis_u1_imports import ensure_ovis_u1_remote_code_importable
from .ovis_u1_video_expert import OvisU1VideoExpert

ensure_ovis_u1_remote_code_importable()

from ovis_u1_hf.modeling_yak import (  # noqa: E402
    DoubleStreamXBlock,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)

logger = get_logger(__name__)


class ActionDiTYak(nn.Module):
    """Action expert shaped like Ovis-U1's Yak MMDiT."""

    block_protocol = "yak"

    def __init__(
        self,
        action_dim: int,
        residual_dim: int = 1536,
        num_heads: int = 12,
        num_layers_double: int = 6,
        num_layers_single: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        max_action_horizon: int = 64,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.hidden_dim = int(residual_dim)
        self.num_heads = int(num_heads)
        self.num_kv_heads = self.num_heads
        self.attn_head_dim = self.hidden_dim // self.num_heads
        self.double_layers = int(num_layers_double)
        self.single_layers = int(num_layers_single)
        self.max_action_horizon = int(max_action_horizon)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        self.action_encoder = nn.Linear(self.action_dim, self.hidden_dim)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_dim)
        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamXBlock(
                    self.hidden_dim,
                    self.num_heads,
                    mlp_ratio=float(mlp_ratio),
                    qkv_bias=bool(qkv_bias),
                )
                for _ in range(self.double_layers)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_dim,
                    self.num_heads,
                    mlp_ratio=float(mlp_ratio),
                )
                for _ in range(self.single_layers)
            ]
        )
        self.head = LastLayer(self.hidden_dim, 1, self.action_dim)

    @property
    def blocks(self):
        return list(self.double_blocks) + list(self.single_blocks)

    @classmethod
    def from_pretrained(
        cls,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ActionDiTYak":
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required for ActionDiTYak.from_pretrained().")
        model = cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        if skip_dit_load_from_pretrain or not action_dit_pretrained_path:
            logger.info("Initializing ActionDiTYak without pretrained action weights.")
            return model

        payload = torch.load(action_dit_pretrained_path, map_location="cpu")
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if not isinstance(state_dict, dict):
            raise ValueError(f"Invalid ActionDiTYak checkpoint type: {type(payload)}")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("ActionDiTYak missing keys when loading: %s", missing[:20])
        if unexpected:
            logger.warning("ActionDiTYak unexpected keys when loading: %s", unexpected[:20])
        return model

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> Dict[str, Any]:
        if action_tokens.ndim != 3:
            raise ValueError(f"`action_tokens` must be [B,T,D], got {tuple(action_tokens.shape)}")
        batch_size, seq_len, action_dim = action_tokens.shape
        if action_dim != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {action_dim}")
        if seq_len > self.max_action_horizon:
            raise ValueError(f"Action length {seq_len} exceeds max_action_horizon={self.max_action_horizon}")
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be [B], got {tuple(timestep.shape)}")
        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        if timestep.shape[0] != batch_size:
            raise ValueError(f"`timestep` length must match batch size {batch_size}, got {timestep.shape[0]}")

        tokens = self.action_encoder(action_tokens)
        vec = self.time_in(timestep_embedding(timestep, 256)).to(dtype=tokens.dtype)
        ids = OvisU1VideoExpert.build_action_ids(
            batch_size,
            seq_len,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return {
            "tokens": tokens,
            "ids": ids,
            "t_mod": vec,
            "context": None,
            "context_mask": context_mask,
            "meta": {"batch_size": batch_size, "seq_len": seq_len},
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        return self.head(tokens, pre_state["t_mod"])
