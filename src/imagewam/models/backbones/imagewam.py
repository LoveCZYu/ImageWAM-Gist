import os
import time
from contextlib import nullcontext
from typing import Any, Optional, Sequence, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

from imagewam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .edit_memory import (
    AdaptiveGistAllocator,
    ActionAttentionCapture,
    DynamicHistorySpanController,
    EditKVMemoryBank,
    PerLayerTopKEditKVSelector,
    VLMSemanticCountCache,
    apply_gist_allocation,
    combine_memory_entries,
    evict_nonpersistent_frame_kv,
    memory_item_confidence,
    memory_kv_tokens_total,
    memory_tokens_per_layer,
    mix_teacher_predicted_gist_decision,
    select_hybrid_memory_entries,
    teacher_memory_probability,
    vlm_supervised_adaptive_k_loss,
)
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


class ImageWAM(torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        video_train_sampling_distribution: str = "uniform",
        video_train_loss_weighting: str = "gaussian",
        video_train_logit_mean: float = 0.0,
        video_train_logit_std: float = 1.0,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        action_train_sampling_distribution: str = "uniform",
        action_train_loss_weighting: str = "gaussian",
        action_train_logit_mean: float = 0.0,
        action_train_logit_std: float = 1.0,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        stack: str = "wan22",
        omnigen2_online_text_cache_compatible: bool = False,
        qwen_context_len: int = 128,
        pack_proprio_after_text: bool = False,
        edit_memory_config: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
            sampling_distribution=video_train_sampling_distribution,
            loss_weighting=video_train_loss_weighting,
            logit_mean=video_train_logit_mean,
            logit_std=video_train_logit_std,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
            sampling_distribution=action_train_sampling_distribution,
            loss_weighting=action_train_loss_weighting,
            logit_mean=action_train_logit_mean,
            logit_std=action_train_logit_std,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.stack = str(stack)
        self.omnigen2_online_text_cache_compatible = bool(omnigen2_online_text_cache_compatible)
        self.qwen_context_len = int(qwen_context_len)
        self.pack_proprio_after_text = bool(pack_proprio_after_text)
        self._configure_edit_memory(edit_memory_config)

        self.to(self.device)

    @staticmethod
    def _default_edit_memory_config() -> dict[str, Any]:
        return {
            "enabled": False,
            "clean_prefix_memory_head": True,
            "support_stack": "flux2",
            "version": "v0.2-memorywam-hybrid-kv",
            "selector": {
                "type": "per_layer_topk",
                "topk_tokens_per_layer": "all",
                "use_action_attention_teacher": True,
                "select_image_tokens_only": True,
                "detach_stored_memory": False,
                "store_dtype": "bfloat16",
                "learned": False,
            },
            "gist": {
                "learned": True,
                "init_std": 0.02,
                "rope_marker": "after_grid",
                "temporal_embedding": "rope",
                "max_time_steps": 2048,
            },
            "adaptive_span": {
                "enabled": False,
                "min_frames": 1,
                "max_frames": 64,
                "hidden_dim": 256,
                "gate_temperature": 1.0,
                "span_cost": 0.0001,
            },
            "adaptive_gist": {
                "enabled": False,
                "min_tokens": 1,
                "max_tokens": 8,
                "hidden_dim": 256,
                "gap_temperature": 0.5,
                "token_cost": 0.001,
                "target_tokens": 4.0,
                "no_cut_gap": 0.05,
            },
            "vlm_count_supervision": {
                "enabled": False,
                "cache_path": None,
                "require_complete": True,
                "count_weight": 1.0,
                "class_balance": False,
                "class_balance_max_weight": 4.0,
            },
            "bank": {
                "max_entries": 0,
                "anchor_frames": 2,
                "recent_frames": 4,
                "gist_tokens_per_frame": 8,
                "write_threshold": 0.0,
                "cooldown_steps": 0,
                "consolidation": "none",
            },
            "injection": {
                "order": "memory_video_action",
                "native_kv": True,
                "use_projection": False,
                "allow_action_attend_memory": True,
                "allow_video_attend_memory": True,
            },
            "retrieval": {
                "enabled": True,
                "use_time_metadata": True,
                "gate_fusion": True,
            },
            "training": {
                "memory_source": "episode_window",
                "history_required": True,
                "history_teacher_no_grad": False,
                "history_steps_per_update": 0,
                "history_credit_sampling": "uniform",
                "old_history_memory_only": True,
                "static_encode_chunk_size": 8,
                "two_phase_local_history_credit": True,
                "clean_conditioning_augmentation": {
                    "enabled": False,
                    "probability": 0.0,
                    "ratio_min": 0.0,
                    "ratio_max": 1.0,
                },
                "memory_curriculum": {
                    "enabled": False,
                    "teacher_only_until_step": 1066,
                    "transition_start_step": 1067,
                    "transition_end_step": 4267,
                    "granularity": "sample",
                },
            },
            "losses": {
                "lambda_attn": 0.1,
                "lambda_memory_action": 1.0,
                "lambda_distill": 0.0,
                "lambda_diversity": 0.0,
            },
        }

    @classmethod
    def _merge_edit_memory_config(cls, user_config: Optional[dict[str, Any]]) -> dict[str, Any]:
        merged = cls._default_edit_memory_config()
        if not user_config:
            return merged
        for key, value in dict(user_config).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        return merged

    def _configure_edit_memory(self, edit_memory_config: Optional[dict[str, Any]]) -> None:
        cfg = self._merge_edit_memory_config(edit_memory_config)
        selector_cfg = cfg.get("selector", {})
        gist_cfg = cfg.get("gist", {})
        bank_cfg = cfg.get("bank", {})
        adaptive_gist_cfg = cfg.get("adaptive_gist", {})
        injection_cfg = cfg.get("injection", {})
        version = str(cfg.get("version"))
        if version not in {"v0.1-full-cache-memory", "v0.2-memorywam-hybrid-kv"}:
            raise ValueError(
                "Only edit_memory.version in {'v0.1-full-cache-memory', "
                "'v0.2-memorywam-hybrid-kv'} is supported by this implementation."
            )
        if str(selector_cfg.get("type")) != "per_layer_topk":
            raise ValueError("v0.1-full edit memory requires selector.type='per_layer_topk'.")
        if str(injection_cfg.get("order")) != "memory_video_action":
            raise ValueError("v0.1-full edit memory requires injection.order='memory_video_action'.")
        if not bool(injection_cfg.get("native_kv", True)):
            raise ValueError("v0.1-full edit memory requires injection.native_kv=true.")
        if bool(injection_cfg.get("use_projection", False)):
            raise ValueError(
                "last-layer selection plus per-layer projection is not implemented unless explicitly added as a fallback."
            )

        self.edit_memory_config = cfg
        self.edit_memory_enabled = bool(cfg.get("enabled", False))
        self.edit_memory_selector = None
        self.edit_memory_bank = None
        self.edit_memory_capture = None
        self.edit_memory_gist_tokens = None
        self.edit_memory_gist_time_embed = None
        self.edit_memory_gist_type_embed = None
        self.edit_memory_span_controller = None
        self.edit_memory_gist_allocator = None
        self.edit_memory_vlm_count_cache = None
        self.edit_memory_vlm_class_weights = None
        self.edit_memory_task_projection = None
        self.edit_memory_prefix_projection = None
        self.edit_memory_write_head = None
        self._edit_memory_last_span_decision = None
        self._edit_memory_last_gist_decision = None
        self._edit_memory_infer_step_idx = 0
        self._edit_memory_training_step = 0
        if not self.edit_memory_enabled:
            return
        if str(cfg.get("support_stack", "flux2")) != "flux2" or str(self.stack) != "flux2":
            raise ValueError("edit_memory v0.2-memorywam-hybrid-kv is only supported for stack='flux2'.")
        if bool(selector_cfg.get("learned", False)):
            raise NotImplementedError("edit_memory.selector.learned=true is reserved for a later learned-selector path.")
        gist_temporal_embedding = str(
            gist_cfg.get("temporal_embedding", "rope")
        ).strip().lower()
        if gist_temporal_embedding not in {"learned", "rope", "none"}:
            raise ValueError(
                "edit_memory.gist.temporal_embedding must be 'learned', 'rope', or 'none', "
                f"got {gist_temporal_embedding!r}."
            )
        self.edit_memory_selector = PerLayerTopKEditKVSelector(
            topk_tokens_per_layer=selector_cfg.get("topk_tokens_per_layer", "all"),
            select_image_tokens_only=bool(selector_cfg.get("select_image_tokens_only", True)),
            detach_stored_memory=bool(selector_cfg.get("detach_stored_memory", True)),
            store_dtype=selector_cfg.get("store_dtype", "bfloat16"),
        )
        self.edit_memory_bank = EditKVMemoryBank(
            max_entries=int(bank_cfg.get("max_entries", 0)),
            write_threshold=float(bank_cfg.get("write_threshold", 0.55)),
            cooldown_steps=int(bank_cfg.get("cooldown_steps", 10)),
            consolidation=str(bank_cfg.get("consolidation", "none")),
            merge_until_under_limit=bool(bank_cfg.get("merge_until_under_limit", True)),
            anchor_frames=int(bank_cfg.get("anchor_frames", 2)),
            # The live current observation is the fourth recent frame.
            recent_frames=max(int(bank_cfg.get("recent_frames", 4)) - 1, 0),
            gist_tokens_per_frame=int(bank_cfg.get("gist_tokens_per_frame", 8)),
            detach_on_write=bank_cfg.get("detach_on_write", "auto"),
        )
        gist_tokens_per_frame = int(bank_cfg.get("gist_tokens_per_frame", 8))
        if bool(gist_cfg.get("learned", True)) and gist_tokens_per_frame > 0:
            img_in = getattr(getattr(self.video_expert, "transformer", None), "img_in", None)
            gist_input_dim = getattr(img_in, "in_features", None)
            if gist_input_dim is None:
                raise ValueError("Cannot initialize FLUX.2 learnable gist tokens: transformer.img_in.in_features missing.")
            init_std = float(gist_cfg.get("init_std", 0.02))
            gist = torch.empty(
                gist_tokens_per_frame,
                int(gist_input_dim),
                device=self.device,
                dtype=self.torch_dtype,
            )
            torch.nn.init.normal_(gist, mean=0.0, std=init_std)
            self.edit_memory_gist_tokens = torch.nn.Parameter(gist)
            if gist_temporal_embedding == "learned":
                max_time_steps = max(int(gist_cfg.get("max_time_steps", 2048)), 1)
                time_embed = torch.empty(
                    max_time_steps,
                    int(gist_input_dim),
                    device=self.device,
                    dtype=self.torch_dtype,
                )
                type_embed = torch.empty(
                    1,
                    int(gist_input_dim),
                    device=self.device,
                    dtype=self.torch_dtype,
                )
                torch.nn.init.normal_(time_embed, mean=0.0, std=init_std)
                torch.nn.init.normal_(type_embed, mean=0.0, std=init_std)
                self.edit_memory_gist_time_embed = torch.nn.Parameter(time_embed)
                self.edit_memory_gist_type_embed = torch.nn.Parameter(type_embed)
            span_cfg = cfg.get("adaptive_span", {})
            if bool(span_cfg.get("enabled", False)):
                self.edit_memory_span_controller = DynamicHistorySpanController(
                    input_dim=int(gist_input_dim),
                    min_frames=int(span_cfg.get("min_frames", 1)),
                    max_frames=int(span_cfg.get("max_frames", 64)),
                    hidden_dim=int(span_cfg.get("hidden_dim", 256)),
                    gate_temperature=float(span_cfg.get("gate_temperature", 1.0)),
                    span_cost=float(span_cfg.get("span_cost", 0.0001)),
                ).to(device=self.device, dtype=self.torch_dtype)
            if bool(adaptive_gist_cfg.get("enabled", False)):
                min_tokens = int(adaptive_gist_cfg.get("min_tokens", 1))
                max_tokens = int(adaptive_gist_cfg.get("max_tokens", gist_tokens_per_frame))
                if min_tokens not in {0, 1}:
                    raise ValueError("Adaptive-k gist allocation requires min_tokens in {0,1}.")
                if max_tokens != gist_tokens_per_frame:
                    raise ValueError(
                        "adaptive_gist.max_tokens must equal bank.gist_tokens_per_frame "
                        f"({max_tokens} != {gist_tokens_per_frame})."
                    )
                self.edit_memory_gist_allocator = AdaptiveGistAllocator(
                    input_dim=int(gist_input_dim),
                    min_tokens=min_tokens,
                    max_tokens=max_tokens,
                    hidden_dim=int(adaptive_gist_cfg.get("hidden_dim", 256)),
                    gap_temperature=float(adaptive_gist_cfg.get("gap_temperature", 0.5)),
                    token_cost=float(adaptive_gist_cfg.get("token_cost", 0.001)),
                    target_tokens=float(adaptive_gist_cfg.get("target_tokens", 4.0)),
                    no_cut_gap=float(adaptive_gist_cfg.get("no_cut_gap", 0.05)),
                ).to(device=self.device, dtype=self.torch_dtype)
            if self.edit_memory_span_controller is not None or self.edit_memory_gist_allocator is not None:
                self.edit_memory_task_projection = nn.Linear(
                    int(self.text_dim), int(gist_input_dim), bias=False
                ).to(device=self.device, dtype=self.torch_dtype)
            prefix_dim = getattr(img_in, "out_features", None)
            fixed_gist_write_head = bool(cfg.get("fixed_gist_write_head", False))
            if self.edit_memory_gist_allocator is not None or fixed_gist_write_head:
                if prefix_dim is None:
                    raise ValueError("transformer.img_in.out_features is required for memory-prefix heads.")
                self.edit_memory_prefix_projection = nn.Linear(
                    int(prefix_dim), int(gist_input_dim), bias=False
                ).to(device=self.device, dtype=self.torch_dtype)
            needs_separate_write_head = fixed_gist_write_head or (
                self.edit_memory_gist_allocator is not None
                and int(getattr(self.edit_memory_gist_allocator, "min_tokens", 1)) > 0
            )
            if needs_separate_write_head:
                write_hidden_dim = int(adaptive_gist_cfg.get("hidden_dim", 256))
                self.edit_memory_write_head = nn.Sequential(
                    nn.LayerNorm(int(gist_input_dim)),
                    nn.Linear(int(gist_input_dim), write_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(write_hidden_dim, 1),
                ).to(device=self.device, dtype=self.torch_dtype)
        vlm_cfg = cfg.get("vlm_count_supervision", {})
        if bool(vlm_cfg.get("enabled", False)):
            write_only = bool(vlm_cfg.get("write_only", False))
            if self.edit_memory_gist_allocator is None and not (
                write_only and self.edit_memory_write_head is not None
            ):
                raise ValueError("VLM count supervision requires adaptive_gist.enabled=true.")
            if not write_only and int(getattr(self.edit_memory_gist_allocator, "min_tokens", 1)) != 0:
                raise ValueError(
                    "Direct VLM distribution supervision requires a unified Adaptive-k K=0..K head."
                )
            self.edit_memory_vlm_count_cache = VLMSemanticCountCache(
                vlm_cfg.get("cache_path"),
                max_tokens=int(adaptive_gist_cfg.get("max_tokens", gist_tokens_per_frame)),
            )
            if (
                not write_only
                and len(self.edit_memory_vlm_count_cache) > 0
                and bool(
                vlm_cfg.get("class_balance", False)
                )
            ):
                self.edit_memory_vlm_class_weights = self.edit_memory_vlm_count_cache.class_weights(
                    max_weight=float(vlm_cfg.get("class_balance_max_weight", 4.0)),
                )
            if len(self.edit_memory_vlm_count_cache) > 0:
                logger.info(
                    "Loaded %d VLM semantic count curves with fingerprint=%s.",
                    len(self.edit_memory_vlm_count_cache),
                    self.edit_memory_vlm_count_cache.fingerprint,
                )
        self.edit_memory_capture = ActionAttentionCapture()
        self.mot.action_attention_capture = self.edit_memory_capture

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        mot_gqa_implementation: str = "repeat",
        mot_force_flash_attention: bool = False,
        omnigen2_online_text_cache_compatible: bool = False,
        qwen_context_len: int = 128,
        pack_proprio_after_text: bool = False,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for ImageWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for ImageWAM.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            gqa_implementation=mot_gqa_implementation,
            force_flash_attention=mot_force_flash_attention,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            stack="wan22",
            pack_proprio_after_text=bool(pack_proprio_after_text),
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    @classmethod
    def from_omnigen2_pretrained(
        cls,
        omnigen2_model_path: str,
        omnigen2_vae_path: str,
        qwen_path: str | None,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        proprio_dim: Optional[int] = None,
        transformer_subfolder: str | None = "transformer",
        vae_subfolder: str | None = "vae",
        load_text_encoder: bool = True,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        mot_gqa_implementation: str = "repeat",
        mot_force_flash_attention: bool = False,
        omnigen2_online_text_cache_compatible: bool = False,
        qwen_context_len: int = 128,
        pack_proprio_after_text: bool = False,
    ):
        from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
        from transformers import AutoTokenizer, Qwen2_5_VLModel

        from .action_dit_omnigen2 import ActionDiTOmnigen2
        from .omnigen2_video_expert import OmniGen2VideoExpert

        video_expert = OmniGen2VideoExpert.from_pretrained(
            omnigen2_model_path,
            subfolder=transformer_subfolder,
            torch_dtype=torch_dtype,
        )
        action_cfg = dict(action_dit_config)
        expected_action_shape = {
            "num_layers": len(video_expert.blocks),
            "num_heads": int(video_expert.num_heads),
            "num_kv_heads": int(video_expert.num_kv_heads),
            "attn_head_dim": int(video_expert.attn_head_dim),
        }
        for key, expected_value in expected_action_shape.items():
            if key in action_cfg and int(action_cfg[key]) != int(expected_value):
                logger.warning(
                    "Overriding action_dit_config.%s=%s to match OmniGen2 video expert value %s.",
                    key,
                    action_cfg[key],
                    expected_value,
                )
            action_cfg[key] = expected_value
        action_expert = ActionDiTOmnigen2.from_pretrained(
            action_dit_config=action_cfg,
            action_dit_pretrained_path=action_dit_pretrained_path,
            device=device,
            torch_dtype=torch_dtype,
        )
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            gqa_implementation=mot_gqa_implementation,
            force_flash_attention=mot_force_flash_attention,
        )
        vae = AutoencoderKL.from_pretrained(
            omnigen2_vae_path,
            subfolder=vae_subfolder,
            torch_dtype=torch_dtype,
        )
        tokenizer = None
        text_encoder = None
        if load_text_encoder:
            logger.warning(
                "ImageWAM-OmniGen2 is loading Qwen from `%s`. This follows OmniGen2 official training configs "
                "using upstream Qwen2.5-VL-3B-Instruct. It is not yet confirmed whether the official "
                "inference `mllm/` subfolder should be used instead for this ImageWAM path.",
                qwen_path,
            )
            if qwen_path is None:
                raise ValueError("`qwen_path` is required when `load_text_encoder=True`.")
            tokenizer = AutoTokenizer.from_pretrained(qwen_path)
            tokenizer.padding_side = "right"
            text_encoder = Qwen2_5_VLModel.from_pretrained(qwen_path, torch_dtype=torch_dtype)

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_dim=2048,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            stack="omnigen2",
            omnigen2_online_text_cache_compatible=bool(omnigen2_online_text_cache_compatible),
            qwen_context_len=int(qwen_context_len),
            pack_proprio_after_text=bool(pack_proprio_after_text),
        )
        model.model_paths = {
            "omnigen2_transformer": omnigen2_model_path,
            "omnigen2_vae": omnigen2_vae_path,
            "qwen": qwen_path,
            "action_dit": action_dit_pretrained_path,
        }
        return model

    @classmethod
    def from_ovis_u1_pretrained(
        cls,
        ovis_u1_model_path: str,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        proprio_dim: Optional[int] = None,
        load_condition_encoder: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        mot_gqa_implementation: str = "repeat",
        mot_force_flash_attention: bool = False,
    ):
        from transformers import AutoModelForCausalLM

        from .action_dit_yak import ActionDiTYak
        from .mot import MoT
        from .ovis_u1_video_expert import OvisU1VideoExpert

        ovis_model = AutoModelForCausalLM.from_pretrained(
            ovis_u1_model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).eval()
        if load_condition_encoder:
            ovis_model = ovis_model.to(device=device, dtype=torch_dtype)
        visual_generator = ovis_model.get_visual_generator()
        video_expert = OvisU1VideoExpert.from_visual_generator(visual_generator)

        action_cfg = dict(action_dit_config)
        expected_action_shape = {
            "residual_dim": int(video_expert.hidden_dim),
            "num_heads": int(video_expert.num_heads),
            "num_layers_double": int(video_expert.double_layers),
            "num_layers_single": int(video_expert.single_layers),
        }
        for key, expected_value in expected_action_shape.items():
            if key in action_cfg and int(action_cfg[key]) != int(expected_value):
                logger.warning(
                    "Overriding action_dit_config.%s=%s to match Ovis-U1 Yak value %s.",
                    key,
                    action_cfg[key],
                    expected_value,
                )
            action_cfg[key] = expected_value
        action_expert = ActionDiTYak.from_pretrained(
            action_dit_config=action_cfg,
            action_dit_pretrained_path=action_dit_pretrained_path,
            device=device,
            torch_dtype=torch_dtype,
        )
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            gqa_implementation=mot_gqa_implementation,
            force_flash_attention=mot_force_flash_attention,
        )
        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=visual_generator.get_vae(),
            text_encoder=None,
            tokenizer=None,
            text_dim=4096,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            stack="ovis_u1",
        )
        object.__setattr__(model, "ovis_condition_encoder", ovis_model if load_condition_encoder else None)
        model.model_paths = {
            "ovis_u1": ovis_u1_model_path,
            "action_dit": action_dit_pretrained_path,
        }
        return model

    @classmethod
    def from_flux2_klein_pretrained(
        cls,
        flux2_model_path: str,
        ae_model_path: str,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        flux2_src_path: str | None = None,
        variant: str = "klein-base-4b",
        proprio_dim: Optional[int] = None,
        load_text_encoder: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        video_train_sampling_distribution: str = "uniform",
        video_train_loss_weighting: str = "gaussian",
        video_train_logit_mean: float = 0.0,
        video_train_logit_std: float = 1.0,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        action_train_sampling_distribution: str = "uniform",
        action_train_loss_weighting: str = "gaussian",
        action_train_logit_mean: float = 0.0,
        action_train_logit_std: float = 1.0,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        mot_gqa_implementation: str = "repeat",
        mot_force_flash_attention: bool = False,
        pack_proprio_after_text: bool = True,
        flux2_lora_config: Optional[dict[str, Any]] = None,
        edit_memory_config: Optional[dict[str, Any]] = None,
        qwen3_model_spec: str | None = None,
        qwen_context_len: int = 512,
    ):
        from safetensors.torch import load_file as load_sft

        from .action_dit_flux2 import ActionDiTFlux2
        from .flux2_imports import ensure_flux2_importable
        from .flux2_video_expert import Flux2VideoExpert
        from .mot import MoT

        ensure_flux2_importable(flux2_src_path)
        from flux2.autoencoder import AutoEncoder, AutoEncoderParams

        key = str(variant).lower().replace("_", "-")
        if key in {"klein-base-4b", "flux.2-klein-base-4b", "4b", "base-4b"}:
            text_dim = 7680
            default_qwen3_model_spec = "Qwen/Qwen3-4B"
        elif key in {"klein-base-9b", "flux.2-klein-base-9b", "9b", "base-9b"}:
            text_dim = 12288
            default_qwen3_model_spec = "Qwen/Qwen3-8B"
        else:
            raise ValueError(f"Unsupported FLUX.2 Klein variant: {variant!r}")

        video_expert = Flux2VideoExpert.from_pretrained(
            flux2_model_path=flux2_model_path,
            variant=key,
            flux2_src_path=flux2_src_path,
            device=device,
            torch_dtype=torch_dtype,
        )
        flux2_lora_config = dict(flux2_lora_config or {})
        if bool(flux2_lora_config.get("enabled", False)):
            from .lora import apply_lora_to_linear_suffixes

            target_suffixes = flux2_lora_config.get(
                "target_suffixes",
                [
                    "qkv",
                    "proj",
                    "linear1",
                    "linear2",
                    "img_mlp.0",
                    "img_mlp.2",
                    "txt_mlp.0",
                    "txt_mlp.2",
                ],
            )
            apply_lora_to_linear_suffixes(
                video_expert.transformer,
                target_suffixes=target_suffixes,
                rank=int(flux2_lora_config.get("rank", 16)),
                alpha=float(flux2_lora_config.get("alpha", 16.0)),
                dropout=float(flux2_lora_config.get("dropout", 0.0)),
            )
            video_expert.flux2_lora_enabled = True
            video_expert.flux2_lora_target_suffixes = tuple(str(item) for item in target_suffixes)
        else:
            video_expert.flux2_lora_enabled = False
        action_cfg = dict(action_dit_config)
        expected_action_shape = {
            "num_heads": int(video_expert.num_heads),
            "attn_head_dim": int(video_expert.attn_head_dim),
            "num_layers_double": int(video_expert.double_layers),
            "num_layers_single": int(video_expert.single_layers),
        }
        action_cfg.setdefault("hidden_dim", 1024)
        for key_name, expected_value in expected_action_shape.items():
            if key_name in action_cfg and int(action_cfg[key_name]) != int(expected_value):
                logger.warning(
                    "Overriding action_dit_config.%s=%s to match FLUX.2 value %s.",
                    key_name,
                    action_cfg[key_name],
                    expected_value,
                )
            action_cfg[key_name] = expected_value
        action_expert = ActionDiTFlux2.from_pretrained(
            action_dit_config=action_cfg,
            action_dit_pretrained_path=action_dit_pretrained_path,
            device=device,
            torch_dtype=torch_dtype,
        )
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            gqa_implementation=mot_gqa_implementation,
            force_flash_attention=mot_force_flash_attention,
        )
        with torch.device("meta"):
            ae = AutoEncoder(AutoEncoderParams())
        ae_state = load_sft(str(ae_model_path), device=str(device))
        ae.load_state_dict(ae_state, strict=True, assign=True)
        ae = ae.to(device=device, dtype=torch_dtype).eval()
        if load_text_encoder:
            from types import SimpleNamespace
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_spec = qwen3_model_spec or default_qwen3_model_spec
            qwen3_model = AutoModelForCausalLM.from_pretrained(
                model_spec,
                torch_dtype=torch_dtype,
            ).to(device).eval()
            text_encoder = SimpleNamespace(
                model=qwen3_model,
                tokenizer=AutoTokenizer.from_pretrained(model_spec),
                max_length=int(qwen_context_len),
            )
        else:
            text_encoder = None

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=ae,
            text_encoder=text_encoder,
            tokenizer=None,
            text_dim=text_dim,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            video_train_sampling_distribution=video_train_sampling_distribution,
            video_train_loss_weighting=video_train_loss_weighting,
            video_train_logit_mean=video_train_logit_mean,
            video_train_logit_std=video_train_logit_std,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            action_train_sampling_distribution=action_train_sampling_distribution,
            action_train_loss_weighting=action_train_loss_weighting,
            action_train_logit_mean=action_train_logit_mean,
            action_train_logit_std=action_train_logit_std,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            stack="flux2",
            qwen_context_len=int(qwen_context_len),
            pack_proprio_after_text=bool(pack_proprio_after_text),
            edit_memory_config=edit_memory_config,
        )
        model.model_paths = {
            "flux2": flux2_model_path,
            "flux2_src": flux2_src_path,
            "ae": ae_model_path,
            "action_dit": action_dit_pretrained_path,
            "qwen3": qwen3_model_spec or default_qwen3_model_spec,
        }
        model.flux2_qwen3_model_spec = qwen3_model_spec or default_qwen3_model_spec
        model.save_lora_merged = bool(flux2_lora_config.get("save_lora_merged", bool(flux2_lora_config.get("enabled", False))))
        model.save_trainable_only = bool(flux2_lora_config.get("save_trainable_only", False))
        return model

    @classmethod
    def from_dim_pretrained(
        cls,
        dim_model_path: str,
        sana_config_path: str,
        qwen_path: str,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        proprio_dim: Optional[int] = None,
        max_condition_length: int = 8192,
        with_latents_condition: bool = True,
        load_mllm: bool = True,
        qwen_attn_implementation: str = "flash_attention_2",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        mot_gqa_implementation: str = "repeat",
        mot_force_flash_attention: bool = False,
        pack_proprio_after_text: bool = True,
    ):
        from safetensors.torch import load_file
        from transformers import AutoProcessor

        from .action_dit_sana import ActionDiTSana
        from .dim_video_expert import DimVideoExpert
        from .mot import MoT

        from models.modeling_dim import Qwen2_5_VLForConditionalGeneration
        from models.multimodal.multimodal_projector.builder import build_projector

        video_expert = DimVideoExpert.from_pretrained(
            dim_model_path=dim_model_path,
            sana_config_path=sana_config_path,
            max_condition_length=int(max_condition_length),
            with_latents_condition=bool(with_latents_condition),
            device=device,
            torch_dtype=torch_dtype,
        )
        action_cfg = dict(action_dit_config)
        expected_action_shape = {
            "attn_dim": int(video_expert.hidden_dim),
            "context_dim": int(video_expert.hidden_dim),
            "num_heads": int(video_expert.num_heads),
            "num_layers": len(video_expert.blocks),
        }
        for key, expected_value in expected_action_shape.items():
            if key in action_cfg and int(action_cfg[key]) != int(expected_value):
                logger.warning(
                    "Overriding action_dit_config.%s=%s to match DIM/SANA value %s.",
                    key,
                    action_cfg[key],
                    expected_value,
                )
            action_cfg[key] = expected_value
        action_expert = ActionDiTSana.from_pretrained(
            action_dit_config=action_cfg,
            action_dit_pretrained_path=action_dit_pretrained_path,
            device=device,
            torch_dtype=torch_dtype,
        )
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            gqa_implementation=mot_gqa_implementation,
            force_flash_attention=mot_force_flash_attention,
        )
        processor = AutoProcessor.from_pretrained(qwen_path, padding_side="left")
        processor.tokenizer.padding_side = "left"
        mllm = None
        if load_mllm:
            mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                qwen_path,
                torch_dtype=torch_dtype,
                attn_implementation=qwen_attn_implementation,
            ).eval()
            for param in mllm.parameters():
                param.requires_grad = False
        projector = build_projector(proj_in=2048, proj_out=int(video_expert.caption_dim), projector_type="mlp2x_gelu")
        ckpt = os.path.join(dim_model_path, "model.safetensors") if os.path.isdir(dim_model_path) else dim_model_path
        state = load_file(ckpt)
        projector_state = {k[len("projector.") :]: v for k, v in state.items() if k.startswith("projector.")}
        missing, unexpected = projector.load_state_dict(projector_state, strict=False)
        if missing:
            logger.warning("DIM projector missing keys: %s", missing[:20])
        if unexpected:
            logger.warning("DIM projector unexpected keys: %s", unexpected[:20])
        projector = projector.to(device=device, dtype=torch_dtype)

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=video_expert.vae,
            text_encoder=mllm,
            tokenizer=processor.tokenizer,
            text_dim=int(video_expert.caption_dim),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            stack="dim",
            pack_proprio_after_text=bool(pack_proprio_after_text),
        )
        model.dim_processor = processor
        model.dim_projector = projector
        model.dim_max_condition_length = int(max_condition_length)
        model.dim_with_latents_condition = bool(with_latents_condition)
        model.model_paths = {
            "dim": dim_model_path,
            "sana_config": sana_config_path,
            "qwen": qwen_path,
            "action_dit": action_dit_pretrained_path,
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            if hasattr(self.text_encoder, "to"):
                self.text_encoder.to(*args, **kwargs)
            elif hasattr(self.text_encoder, "model") and hasattr(self.text_encoder.model, "to"):
                self.text_encoder.model.to(*args, **kwargs)
        if hasattr(self, "dim_projector"):
            self.dim_projector.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    @staticmethod
    def _scheduler_timestep_to_unit(timestep: torch.Tensor, scheduler) -> torch.Tensor:
        """Convert ImageWAM scheduler timesteps back to Yak's [0, 1] time domain."""
        num_train_timesteps = float(getattr(scheduler, "num_train_timesteps", 1000))
        if num_train_timesteps <= 0:
            raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}.")
        return timestep / num_train_timesteps

    @staticmethod
    def _augment_clean_conditioning_tokens(
        tokens: torch.Tensor,
        *,
        enabled: bool,
        probability: float,
        ratio_min: float,
        ratio_max: float,
        ratio: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Apply MemoryWAM's train-only clean-conditioning noise interpolation."""
        if tokens.ndim < 2:
            raise ValueError(f"`tokens` must include a batch dimension, got {tuple(tokens.shape)}")
        if not enabled or probability <= 0.0:
            return tokens
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"`probability` must be in [0, 1], got {probability}")
        if not 0.0 <= ratio_min <= ratio_max <= 1.0:
            raise ValueError(
                "`ratio_min` and `ratio_max` must satisfy 0 <= min <= max <= 1, "
                f"got {ratio_min}, {ratio_max}."
            )
        batch_size = int(tokens.shape[0])
        if ratio is None:
            ratio = torch.rand(
                (batch_size,),
                device=tokens.device,
                dtype=torch.float32,
                generator=generator,
            )
            ratio = ratio_min + (ratio_max - ratio_min) * ratio
        else:
            ratio = ratio.to(device=tokens.device, dtype=torch.float32).reshape(-1)
            if ratio.numel() == 1:
                ratio = ratio.expand(batch_size)
            if ratio.numel() != batch_size:
                raise ValueError(
                    "`ratio` must be scalar or one value per batch item, "
                    f"got shape {tuple(ratio.shape)} for batch_size={batch_size}."
                )
        if bool(((ratio < 0.0) | (ratio > 1.0)).any().item()):
            raise ValueError("`ratio` values must be in [0, 1].")
        if noise is None:
            noise = torch.randn(
                tokens.shape,
                device=tokens.device,
                dtype=tokens.dtype,
                generator=generator,
            )
        else:
            noise = noise.to(device=tokens.device, dtype=tokens.dtype)
            if noise.shape != tokens.shape:
                raise ValueError(
                    "`noise` must match conditioning token shape, "
                    f"got {tuple(noise.shape)} vs {tuple(tokens.shape)}."
                )
        ratio_view = ratio.to(dtype=tokens.dtype).view(batch_size, *([1] * (tokens.ndim - 1)))
        augmented = (1.0 - ratio_view) * tokens + ratio_view * noise
        if probability == 1.0:
            return augmented
        apply_mask = torch.rand(
            (batch_size,),
            device=tokens.device,
            dtype=torch.float32,
            generator=generator,
        ) < probability
        apply_view = apply_mask.view(batch_size, *([1] * (tokens.ndim - 1)))
        return torch.where(apply_view, augmented, tokens)

    def _augment_memory_clean_conditioning(self, tokens: torch.Tensor) -> torch.Tensor:
        training_cfg = self._edit_memory_training_cfg()
        augmentation_cfg = training_cfg.get("clean_conditioning_augmentation", {})
        return self._augment_clean_conditioning_tokens(
            tokens,
            enabled=bool(augmentation_cfg.get("enabled", False)),
            probability=float(augmentation_cfg.get("probability", 0.0)),
            ratio_min=float(augmentation_cfg.get("ratio_min", 0.0)),
            ratio_max=float(augmentation_cfg.get("ratio_max", 1.0)),
        )

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        if not getattr(self, "pack_proprio_after_text", False):
            proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
            return (
                torch.cat([context, proprio_token], dim=1),
                torch.cat([context_mask, proprio_mask], dim=1),
            )
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        if context.shape[:2] != context_mask.shape:
            raise ValueError(
                f"`context/context_mask` leading dims must match, got {tuple(context.shape[:2])} and {tuple(context_mask.shape)}"
            )
        if context.shape[0] != proprio_token.shape[0]:
            raise ValueError(
                f"`proprio` batch size must match context batch size ({context.shape[0]}), got {proprio_token.shape[0]}"
            )

        context_mask = context_mask.to(device=context.device, dtype=torch.bool)
        new_context = context.new_zeros(context.shape[0], context.shape[1] + 1, context.shape[2])

        # OmniGen2 treats context_mask.sum() as a contiguous prefix length, so the
        # proprio token must sit immediately after the valid text tokens.
        valid_counts = context_mask.sum(dim=1)
        valid_rank = context_mask.cumsum(dim=1) - 1
        invalid_mask = ~context_mask
        invalid_rank = invalid_mask.cumsum(dim=1) - 1
        target_indices = torch.where(
            context_mask,
            valid_rank,
            valid_counts[:, None] + 1 + invalid_rank,
        )
        new_context.scatter_(
            dim=1,
            index=target_indices[:, :, None].expand(-1, -1, context.shape[2]),
            src=context,
        )
        batch_indices = torch.arange(context.shape[0], device=context.device)
        new_context[batch_indices, valid_counts] = proprio_token[:, 0]
        positions = torch.arange(context.shape[1] + 1, device=context.device)
        new_context_mask = positions[None, :] <= valid_counts[:, None]

        return new_context, new_context_mask

    def _append_proprio_to_context_if_enabled(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
        source: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None:
            return context, context_mask
        if proprio is None:
            raise ValueError(f"`{source}` requires `proprio` when `proprio_dim` is enabled.")

        if proprio.ndim == 3:
            proprio = proprio[:, 0, :]
        elif proprio.ndim == 2:
            pass
        elif proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        else:
            raise ValueError(
                f"`{source}` `proprio` must be [B,T,D], [B,D], or [D], got shape {tuple(proprio.shape)}"
            )
        if proprio.shape[0] != context.shape[0]:
            raise ValueError(
                f"`{source}` `proprio` batch size must match context batch size "
                f"({context.shape[0]}), got {proprio.shape[0]}"
            )
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`{source}` `proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        return self._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
        )

    def _configure_action_attention_capture(
        self,
        *,
        condition_slice: slice | tuple[int, int] | list[int],
        condition_grid: tuple[int, int],
        prefix_len: int,
        action_len: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        capture = getattr(self.mot, "action_attention_capture", None)
        if capture is None or not hasattr(capture, "configure"):
            return
        capture.configure(
            condition_slice=condition_slice,
            condition_grid=condition_grid,
            prefix_len=prefix_len,
            action_len=action_len,
            metadata=metadata or {},
        )

    def _start_action_attention_capture_step(self, step_idx: int) -> None:
        capture = getattr(self.mot, "action_attention_capture", None)
        if capture is not None and hasattr(capture, "start_step"):
            capture.start_step(int(step_idx))

    def reset_edit_memory(self, batch_size: int | None = None, device: torch.device | str | None = None) -> None:
        bank = getattr(self, "edit_memory_bank", None)
        if bank is not None:
            bank.reset(batch_size=batch_size, device=device or self.device)
        self._edit_memory_infer_step_idx = 0
        self._edit_memory_last_span_decision = None
        self._edit_memory_last_gist_decision = None
        capture = getattr(self, "edit_memory_capture", None)
        if capture is not None:
            capture.clear()

    def set_edit_memory_training_step(self, step: int) -> None:
        """Set the zero-based optimizer step used by the sample-level memory curriculum."""
        self._edit_memory_training_step = max(int(step), 0)

    def _edit_memory_frame_summary(
        self,
        ref_tokens: torch.Tensor,
        text_hidden: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if ref_tokens.ndim != 3:
            raise ValueError(f"FLUX.2 reference tokens must be [B,N,D], got {tuple(ref_tokens.shape)}")
        summary = ref_tokens.mean(dim=1)
        projection = getattr(self, "edit_memory_task_projection", None)
        if projection is None or text_hidden is None:
            return summary
        if text_mask is None:
            task_summary = text_hidden.mean(dim=1)
        else:
            weights = text_mask.to(device=text_hidden.device, dtype=text_hidden.dtype)
            task_summary = (text_hidden * weights[:, :, None]).sum(dim=1) / weights.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
        return summary + projection(task_summary.to(dtype=projection.weight.dtype)).to(dtype=summary.dtype)

    def _edit_memory_span_decision(self, frame_summary: torch.Tensor) -> dict[str, Any] | None:
        controller = getattr(self, "edit_memory_span_controller", None)
        if controller is None:
            return None
        decision = controller(frame_summary)
        decision["gate_temperature"] = float(controller.gate_temperature)
        self._edit_memory_last_span_decision = decision
        return decision

    def _edit_memory_gist_decision(self, frame_summary: torch.Tensor) -> dict[str, torch.Tensor] | None:
        allocator = getattr(self, "edit_memory_gist_allocator", None)
        if allocator is None:
            return None
        decision = allocator(frame_summary)
        self._edit_memory_last_gist_decision = decision
        return decision

    def _edit_memory_prefix_summary(self, prefix_cache: dict[str, Any]) -> torch.Tensor:
        final_video = prefix_cache.get("final_video", {})
        image_tokens = final_video.get("img")
        projection = getattr(self, "edit_memory_prefix_projection", None)
        if not isinstance(image_tokens, torch.Tensor) or projection is None:
            raise RuntimeError("Memory heads require clean-prefix final image tokens and projection.")
        return projection(
            image_tokens.mean(dim=1).to(dtype=projection.weight.dtype)
        ).to(dtype=image_tokens.dtype)

    def _edit_memory_write_confidence(
        self,
        prefix_summary: torch.Tensor,
        gist_decision: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor] | None:
        if gist_decision is not None and gist_decision.get("write_decision") is not None:
            return gist_decision["write_decision"]
        head = getattr(self, "edit_memory_write_head", None)
        if head is None:
            return None
        logits = head(prefix_summary.to(dtype=next(head.parameters()).dtype)).float().squeeze(-1)
        return {
            "logits": logits,
            "probabilities": torch.sigmoid(logits),
            "hard_write": logits >= 0.0,
        }

    def _edit_memory_vlm_count_supervision(
        self,
        gist_decision: dict[str, torch.Tensor] | None,
        episode_ids: torch.Tensor | None,
        timesteps: torch.Tensor | None,
        valid: torch.Tensor,
        write_decision: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any] | None:
        cfg = getattr(self, "edit_memory_config", {}).get("vlm_count_supervision", {})
        if not bool(cfg.get("enabled", False)):
            return None
        write_only = bool(cfg.get("write_only", False))
        if gist_decision is None and not write_only:
            raise RuntimeError("VLM count supervision requires an Adaptive-k gist decision.")
        if episode_ids is None or timesteps is None:
            raise ValueError(
                "VLM count supervision requires dataset-provided episode_id and timestep for every frame."
            )
        cache = getattr(self, "edit_memory_vlm_count_cache", None)
        if cache is None or len(cache) == 0:
            if bool(cfg.get("require_complete", True)):
                raise RuntimeError(
                    "VLM count supervision is enabled but no semantic cache is loaded. "
                    "Set model.edit_memory_config.vlm_count_supervision.cache_path for training."
                )
            return None
        teacher_probabilities, original_count, found = cache.lookup(
            episode_ids, timesteps, device=self.device
        )
        valid = valid.to(device=self.device, dtype=torch.bool).reshape(-1)
        active = valid & found
        if bool(cfg.get("require_complete", True)) and bool((valid & ~found).any().item()):
            missing = torch.nonzero(valid & ~found, as_tuple=False).flatten()[0]
            raise KeyError(
                "VLM semantic cache coverage is incomplete for sampled frame "
                f"episode_id={int(episode_ids.reshape(-1)[missing].item())}, "
                f"timestep={int(timesteps.reshape(-1)[missing].item())}."
            )
        if write_only:
            if write_decision is None:
                raise RuntimeError("VLM write-only supervision requires a memory write head.")
            zero = write_decision["logits"].float().sum() * 0.0
            denom = active.float().sum().clamp_min(1.0)
            result = {
                "loss": zero,
                "count_loss": zero,
                "teacher_count": original_count,
                "teacher_probabilities": teacher_probabilities,
                "original_count": original_count,
                "agreement": active.float().sum() / denom,
                "expected_mae": zero.detach(),
                "zero_mapped_frac": (
                    (original_count == 0).float() * active.float()
                ).sum()
                / denom,
                "valid_count": active.float().sum(),
            }
        else:
            result = vlm_supervised_adaptive_k_loss(
                gist_decision,
                teacher_probabilities,
                original_count,
                active,
                count_weight=float(cfg.get("count_weight", 1.0)),
                class_weights=getattr(self, "edit_memory_vlm_class_weights", None),
            )
            if write_decision is None:
                write_decision = gist_decision.get("write_decision")
        write_target = result["original_count"] > 0
        result["write_target"] = write_target
        if write_decision is not None:
            result["write_agreement"] = (
                (write_decision["hard_write"] == write_target).float() * active.float()
            ).sum() / active.float().sum().clamp_min(1.0)
        if write_only and write_decision is not None:
            write_per_sample = F.binary_cross_entropy_with_logits(
                write_decision["logits"].float(),
                write_target.float(),
                reduction="none",
            )
            write_loss = (write_per_sample * active.float()).sum() / active.float().sum().clamp_min(1.0)
            result["write_loss"] = write_loss
            result["agreement"] = result["write_agreement"]
            result["loss"] = result["loss"] + float(cfg.get("write_weight", 1.0)) * write_loss
        result["coverage"] = active.float().sum() / valid.float().sum().clamp_min(1.0)
        result["active"] = active
        return result

    def _edit_memory_write_decision(
        self,
        gist_decision: dict[str, torch.Tensor] | None,
        vlm_metrics: dict[str, Any] | None,
        random_values: torch.Tensor | None = None,
        write_decision: dict[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any]]:
        training_cfg = getattr(self, "edit_memory_config", {}).get("training", {})
        curriculum = training_cfg.get("memory_curriculum", {})
        probability = 0.0
        # This helper is called only by the episode-window training path. The
        # trainer intentionally keeps the top-level module in eval mode while
        # enabling selected DiT submodules, so `self.training` is not a valid
        # train/infer discriminator here.
        if bool(curriculum.get("enabled", False)):
            if str(curriculum.get("granularity", "sample")).strip().lower() != "sample":
                raise ValueError("Memory curriculum currently requires sample granularity.")
            probability = teacher_memory_probability(
                getattr(self, "_edit_memory_training_step", 0),
                teacher_only_until_step=int(curriculum.get("teacher_only_until_step", 1066)),
                transition_start_step=int(curriculum.get("transition_start_step", 1067)),
                transition_end_step=int(curriculum.get("transition_end_step", 4267)),
            )
        if write_decision is None and gist_decision is not None:
            write_decision = gist_decision.get("write_decision")
        if vlm_metrics is None or probability <= 0.0:
            batch_size = (
                int(write_decision["hard_write"].numel())
                if write_decision is not None
                else 0
                if gist_decision is None
                else int(gist_decision["hard_count"].numel())
            )
            device = (
                write_decision["hard_write"].device
                if write_decision is not None
                else self.device
                if gist_decision is None
                else gist_decision["hard_count"].device
            )
            teacher_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
            return gist_decision, {
                "teacher_probability": float(probability),
                "teacher_mask": teacher_mask,
                "hard_write": None if write_decision is None else write_decision["hard_write"],
                "write_confidence": None
                if write_decision is None
                else write_decision["probabilities"],
            }
        if gist_decision is None:
            if write_decision is None:
                raise RuntimeError(
                    "Teacher memory curriculum requires either Adaptive-k or a write head."
                )
            active = vlm_metrics["active"].to(
                device=write_decision["hard_write"].device, dtype=torch.bool
            )
            if random_values is None:
                random_values = torch.rand(active.shape, device=active.device)
            else:
                random_values = random_values.to(
                    device=active.device, dtype=torch.float32
                ).reshape(-1)
            teacher_mask = active & (random_values < float(probability))
            hard_write = torch.where(
                teacher_mask,
                vlm_metrics["write_target"].to(device=active.device),
                write_decision["hard_write"],
            )
            write_confidence = torch.where(
                teacher_mask,
                hard_write.to(dtype=write_decision["probabilities"].dtype),
                write_decision["probabilities"],
            )
            return None, {
                "teacher_probability": float(probability),
                "teacher_mask": teacher_mask,
                "hard_write": hard_write,
                "write_confidence": write_confidence,
            }
        mixed, teacher_mask = mix_teacher_predicted_gist_decision(
            gist_decision,
            teacher_count=vlm_metrics["teacher_count"],
            teacher_active=vlm_metrics["active"],
            teacher_probability=probability,
            random_values=random_values,
        )
        hard_write = mixed["hard_count"] > 0
        write_confidence = None if write_decision is None else write_decision["probabilities"]
        if write_confidence is not None and hard_write is not None:
            write_confidence = torch.where(
                teacher_mask,
                hard_write.to(device=write_confidence.device, dtype=write_confidence.dtype),
                write_confidence,
            )
        return mixed, {
            "teacher_probability": float(probability),
            "teacher_mask": teacher_mask,
            "hard_write": hard_write,
            "write_confidence": write_confidence,
        }

    @staticmethod
    def _grid_hw_from_flux2_img_ids(img_ids: torch.Tensor | None) -> tuple[int, int] | None:
        if img_ids is None or img_ids.numel() == 0:
            return None
        if img_ids.ndim != 3 or img_ids.shape[-1] < 3:
            return None
        ids0 = img_ids[0].detach().float()
        height = int(ids0[:, 1].max().item()) + 1
        width = int(ids0[:, 2].max().item()) + 1
        if height <= 0 or width <= 0:
            return None
        return height, width

    def _build_flux2_gist_inputs(
        self,
        *,
        batch_size: int,
        ref_img_ids: torch.Tensor | None,
        device: torch.device,
        token_dtype: torch.dtype,
        step_idx: int | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        gist_param = getattr(self, "edit_memory_gist_tokens", None)
        if gist_param is None:
            return None, None
        num_gist = int(gist_param.shape[0])
        if num_gist <= 0:
            return None, None
        gist_tokens = gist_param.to(device=device, dtype=token_dtype).unsqueeze(0).expand(int(batch_size), -1, -1)
        type_embed = getattr(self, "edit_memory_gist_type_embed", None)
        if isinstance(type_embed, torch.nn.Parameter):
            gist_tokens = gist_tokens + type_embed.to(device=device, dtype=token_dtype).view(1, 1, -1)
        time_embed = getattr(self, "edit_memory_gist_time_embed", None)
        if isinstance(time_embed, torch.nn.Parameter):
            if step_idx is None:
                step_values = torch.zeros(int(batch_size), device=device, dtype=torch.long)
            elif isinstance(step_idx, torch.Tensor):
                step_values = step_idx.to(device=device, dtype=torch.long).flatten()
                if step_values.numel() == 1:
                    step_values = step_values.expand(int(batch_size))
                elif step_values.numel() != int(batch_size):
                    step_values = step_values[:1].expand(int(batch_size))
            else:
                step_values = torch.full((int(batch_size),), int(step_idx), device=device, dtype=torch.long)
            step_values = step_values.clamp_min(0).remainder(int(time_embed.shape[0]))
            gist_tokens = gist_tokens + time_embed.to(device=device, dtype=token_dtype).index_select(0, step_values).unsqueeze(1)
        id_dim = 4 if ref_img_ids is None else int(ref_img_ids.shape[-1])
        id_dtype = token_dtype if ref_img_ids is None else ref_img_ids.dtype
        gist_ids = torch.zeros(int(batch_size), num_gist, id_dim, device=device, dtype=id_dtype)
        if ref_img_ids is not None and ref_img_ids.numel() > 0:
            ref_ids = ref_img_ids.to(device=device, dtype=id_dtype)
            gist_ids[..., 0] = ref_ids[:, :1, 0].expand(-1, num_gist)
            if id_dim > 1:
                gist_ids[..., 1] = (ref_ids[..., 1].amax(dim=1, keepdim=True) + 1).expand(-1, num_gist)
            if id_dim > 2:
                gist_ids[..., 2] = (ref_ids[..., 2].amax(dim=1, keepdim=True) + 1).expand(-1, num_gist)
            if id_dim > 3:
                gist_ids[..., 3] = ref_ids[:, :1, 3].expand(-1, num_gist)
        return gist_tokens, gist_ids

    def _prefill_flux2_memory_prefix(
        self,
        *,
        ref_tokens: torch.Tensor,
        ref_img_ids: torch.Tensor,
        text_hidden: torch.Tensor,
        text_mask: torch.Tensor,
        gist_tokens: torch.Tensor | None,
        gist_ids: torch.Tensor | None,
        memory_kv_cache: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build the clean prefix used by both action conditioning and memory writes."""
        batch_size = int(ref_tokens.shape[0])
        empty_target = ref_tokens.new_zeros(batch_size, 0, ref_tokens.shape[-1])
        empty_target_ids = ref_img_ids.new_zeros(batch_size, 0, ref_img_ids.shape[-1])
        zero_timestep = torch.zeros(batch_size, dtype=ref_tokens.dtype, device=ref_tokens.device)
        prefix_pre = self.video_expert.pre_dit(
            x=empty_target,
            timestep=zero_timestep,
            context=text_hidden,
            context_mask=text_mask,
            ref_image_hidden_states=ref_tokens,
            gist_hidden_states=gist_tokens,
            target_img_ids=empty_target_ids,
            ref_img_ids=ref_img_ids,
            gist_img_ids=gist_ids,
        )
        prefix_mask = self._build_mot_attention_mask_flux2(
            batch_size=batch_size,
            txt_len=int(prefix_pre["txt_len"]),
            target_len=0,
            cond_len=int(prefix_pre["cond_len"]),
            action_len=0,
            device=ref_tokens.device,
            text_attention_mask=prefix_pre["text_mask"],
            gist_len=int(prefix_pre.get("gist_len", 0)),
        )
        prefix_cache = self.mot.prefill_flux2_video_cache(
            video_tokens=prefix_pre["tokens"],
            video_freqs=prefix_pre["freqs"],
            video_t_mod=prefix_pre["t_mod"],
            attention_mask=prefix_mask,
            memory_kv_cache=memory_kv_cache,
        )
        return prefix_pre, prefix_cache

    @staticmethod
    def _slice_flux2_kv_layers(
        video_kv_cache: dict[str, Any],
        *,
        start: int,
        length: int,
        layer_prefix: str,
        store_dtype: torch.dtype | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {"double": [], "single": []}
        if length <= 0:
            return out
        stop = int(start) + int(length)
        for layer_key in ("double", "single"):
            for layer_idx, cache_l in enumerate(video_kv_cache.get(layer_key, [])):
                k = cache_l["k"][:, start:stop]
                v = cache_l["v"][:, start:stop]
                # A batch-size-one slice can report as contiguous while still
                # retaining the complete clean-prefix K/V storage. Force a
                # compact differentiable copy for the memory bank.
                if store_dtype is None:
                    k = k.clone()
                    v = v.clone()
                else:
                    k = k.to(dtype=store_dtype, copy=True)
                    v = v.to(dtype=store_dtype, copy=True)
                idx = torch.arange(length, device=k.device, dtype=torch.long).view(1, length).expand(k.shape[0], -1)
                out[layer_key].append(
                    {
                        "k": k,
                        "v": v,
                        "topk_indices": idx,
                        "topk_scores": torch.ones(k.shape[0], length, device=k.device, dtype=torch.float32),
                        "spatial_xy": torch.zeros(k.shape[0], length, 2, device=k.device, dtype=torch.float32),
                        "layer_idx": int(layer_idx),
                        "token_source": str(layer_prefix),
                    }
                )
        return out

    def _edit_memory_losses_cfg(self) -> dict[str, float]:
        losses = getattr(self, "edit_memory_config", {}).get("losses", {})
        return {
            "lambda_attn": float(losses.get("lambda_attn", 0.1)),
            "lambda_memory_action": float(losses.get("lambda_memory_action", 1.0)),
            "lambda_distill": float(losses.get("lambda_distill", 0.5)),
            "lambda_diversity": float(losses.get("lambda_diversity", 0.0)),
        }

    def _edit_memory_training_cfg(self) -> dict[str, Any]:
        training = getattr(self, "edit_memory_config", {}).get("training", {})
        return {
            "memory_source": str(training.get("memory_source", "episode_window")).strip().lower(),
            "history_required": bool(training.get("history_required", True)),
            "history_teacher_no_grad": bool(training.get("history_teacher_no_grad", True)),
            "history_steps_per_update": int(training.get("history_steps_per_update", 0)),
            "history_credit_sampling": str(
                training.get("history_credit_sampling", "recent")
            ).strip().lower(),
            "old_history_memory_only": bool(training.get("old_history_memory_only", True)),
            "static_encode_chunk_size": max(
                int(training.get("static_encode_chunk_size", 8)), 1
            ),
            "two_phase_local_history_credit": bool(
                training.get("two_phase_local_history_credit", True)
            ),
            "replan_steps": max(int(training.get("replan_steps", 16)), 1),
            "tbptt_steps": max(int(training.get("tbptt_steps", 4)), 1),
            "memory_curriculum": dict(training.get("memory_curriculum", {})),
        }

    @staticmethod
    def _select_trainable_history_indices(
        history_indices: list[int],
        max_steps: int,
        strategy: str,
        *,
        device: torch.device,
    ) -> set[int]:
        if max_steps <= 0 or max_steps >= len(history_indices):
            return set(history_indices)
        if strategy == "recent":
            return set(history_indices[-max_steps:])
        if strategy != "uniform":
            raise ValueError(
                "edit_memory.training.history_credit_sampling must be recent or uniform; "
                f"got {strategy!r}."
            )
        order = torch.randperm(len(history_indices), device=device)[:max_steps].cpu().tolist()
        return {history_indices[position] for position in order}

    @staticmethod
    def _rank_synchronous_history_slot(
        global_num_history: int,
        training_step: int,
        *,
        seed: int = 3407,
    ) -> int | None:
        """Choose one shared slot without modulo-period bias across ZeRO-3 ranks."""
        count = int(global_num_history)
        if count <= 0:
            return None
        step = max(int(training_step), 0)
        cycle, position = divmod(step, count)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + cycle)
        return int(torch.randperm(count, generator=generator)[position].item())

    @staticmethod
    def _edit_memory_historical_recent_frames(bank_cfg: dict[str, Any]) -> int:
        # The current observation contributes the newest full-K/V replan
        # outside the historical bank, so retain N-1 full historical replans.
        return max(int(bank_cfg.get("recent_frames", 4)) - 1, 0)

    @staticmethod
    def _maintain_memorywam_entries(
        entries: list[dict[str, Any]],
        bank_cfg: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], int]:
        return evict_nonpersistent_frame_kv(
            entries,
            anchor_frames=int(bank_cfg.get("anchor_frames", 2)),
            recent_frames=ImageWAM._edit_memory_historical_recent_frames(bank_cfg),
            gist_tokens_per_frame=int(bank_cfg.get("gist_tokens_per_frame", 8)),
        )

    def _build_edit_memory_item(
        self,
        selected_memory: dict[str, list[dict[str, Any]]],
        *,
        step_idx: int,
        confidence: torch.Tensor,
    ) -> dict[str, Any]:
        item = {
            "double": selected_memory["double"],
            "single": selected_memory["single"],
            "gist_double": selected_memory.get("gist_double", []),
            "gist_single": selected_memory.get("gist_single", []),
            "step_start": int(step_idx),
            "step_end": int(step_idx),
            "confidence": confidence,
            "merge_count": 1,
        }
        for key in ("adaptive_gist_hard_count", "adaptive_gist_expected_count"):
            if key in selected_memory:
                item[key] = selected_memory[key]
        return item

    def _select_flux2_edit_memory_item(
        self,
        *,
        video_kv_cache: dict[str, Any],
        per_layer_teacher_importance: dict[str, Any] | None,
        txt_len: int,
        img_len: int,
        grid_hw: tuple[int, int] | None,
        step_idx: int,
        gist_len: int = 0,
        gist_decision: dict[str, torch.Tensor] | None = None,
        write_confidence: torch.Tensor | None = None,
        compact_adaptive_gist: bool = False,
    ) -> dict[str, Any]:
        if self.edit_memory_selector is None:
            raise RuntimeError("edit_memory selector is not initialized.")
        selected = self.edit_memory_selector(
            video_kv_cache=video_kv_cache,
            per_layer_teacher_importance=per_layer_teacher_importance,
            txt_len=int(txt_len),
            img_len=int(img_len),
            grid_hw=grid_hw,
        )
        if int(gist_len) > 0:
            store_dtype = None
            if self.edit_memory_selector is not None:
                store_dtype = getattr(self.edit_memory_selector, "store_dtype", None)
            gist_selected = self._slice_flux2_kv_layers(
                video_kv_cache,
                start=int(txt_len) + int(img_len),
                length=int(gist_len),
                layer_prefix="learned_gist",
                store_dtype=store_dtype,
            )
            selected["gist_double"] = gist_selected["double"]
            selected["gist_single"] = gist_selected["single"]
            selected = apply_gist_allocation(
                selected,
                gist_decision,
                compact=bool(compact_adaptive_gist),
            )
        if write_confidence is not None:
            confidence = write_confidence.float()
        elif gist_decision is not None:
            probabilities = gist_decision["probabilities"].float()
            selected_count = gist_decision["hard_count"].long()
            if probabilities.shape[1] == int(gist_decision["hard_keep"].shape[1]) + 1:
                confidence_index = selected_count.clamp(0, probabilities.shape[1] - 1)
            else:
                confidence_index = (selected_count - 1).clamp(0, probabilities.shape[1] - 1)
            confidence = probabilities.gather(1, confidence_index[:, None]).squeeze(1)
        else:
            confidence = memory_item_confidence(selected)
        return self._build_edit_memory_item(selected, step_idx=step_idx, confidence=confidence)

    @staticmethod
    def _attach_edit_memory_valid_mask(
        memory_item: dict[str, Any],
        valid: torch.Tensor,
        *,
        gist_valid: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        valid = valid.to(dtype=torch.bool)
        gist_valid = valid if gist_valid is None else gist_valid.to(
            device=valid.device,
            dtype=torch.bool,
        )
        out = dict(memory_item)
        for layer_key in ("double", "single", "gist_double", "gist_single"):
            layers = []
            for layer in memory_item.get(layer_key, []):
                layer_out = dict(layer)
                layer_sample_valid = (
                    gist_valid if layer_key.startswith("gist_") else valid
                )
                token_valid = layer_sample_valid.to(
                    device=layer_out["k"].device
                )[:, None].expand(-1, int(layer_out["k"].shape[1]))
                layer_out["memory_is_valid"] = token_valid
                layers.append(layer_out)
            out[layer_key] = layers
        confidence = out.get("confidence")
        if isinstance(confidence, torch.Tensor):
            out["confidence"] = confidence * gist_valid.to(
                device=confidence.device,
                dtype=confidence.dtype,
            )
        out["memory_is_valid"] = valid
        out["gist_memory_is_valid"] = gist_valid
        out["gist_write_selected"] = gist_valid
        return out

    def _build_flux2_history_memory_cache(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        training_cfg = self._edit_memory_training_cfg()
        required = bool(training_cfg["history_required"])
        needed = ("edit_memory_video", "edit_memory_action", "edit_memory_valid")
        if any(key not in sample for key in needed):
            if required:
                missing = [key for key in needed if key not in sample]
                raise ValueError(
                    "edit_memory.training.memory_source='history' requires dataset-provided "
                    f"history fields, missing {missing}. Set data.train.edit_memory_history_offsets."
                )
            return None

        history_video = sample["edit_memory_video"].to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        history_action = sample["edit_memory_action"].to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        history_valid = sample["edit_memory_valid"].to(device=self.device, dtype=torch.bool, non_blocking=True)
        if history_video.ndim != 6:
            raise ValueError(
                "`sample['edit_memory_video']` must be [B,M,C,T,H,W], "
                f"got {tuple(history_video.shape)}"
            )
        if history_action.ndim != 4:
            raise ValueError(
                "`sample['edit_memory_action']` must be [B,M,A,D], "
                f"got {tuple(history_action.shape)}"
            )
        if history_valid.ndim != 2:
            raise ValueError(
                "`sample['edit_memory_valid']` must be [B,M], "
                f"got {tuple(history_valid.shape)}"
            )
        batch_size, num_entries = int(history_video.shape[0]), int(history_video.shape[1])
        if history_action.shape[:2] != history_video.shape[:2] or history_valid.shape != history_video.shape[:2]:
            raise ValueError(
                "edit-memory history batch/entry dimensions are inconsistent: "
                f"video={tuple(history_video.shape)}, action={tuple(history_action.shape)}, "
                f"valid={tuple(history_valid.shape)}"
            )
        history_proprio = sample.get("edit_memory_proprio")
        if self.proprio_encoder is not None:
            if history_proprio is None:
                raise ValueError(
                    "`sample['edit_memory_proprio']` is required for history edit memory when proprio is enabled."
                )
            history_proprio = history_proprio.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            if history_proprio.ndim != 4 or history_proprio.shape[:2] != history_video.shape[:2]:
                raise ValueError(
                    "`sample['edit_memory_proprio']` must be [B,M,T,D], "
                    f"got {tuple(history_proprio.shape)}"
                )

        base_text_hidden, base_text_mask = self._encode_flux2_text(sample)
        bank_cfg = getattr(self, "edit_memory_config", {}).get("bank", {})
        entries: list[dict[str, Any]] = []
        grad_enabled = not bool(training_cfg["history_teacher_no_grad"])
        history_steps_per_update = int(training_cfg.get("history_steps_per_update", 0))
        if history_steps_per_update < 0:
            history_steps_per_update = 0
        if history_steps_per_update == 0:
            history_indices = [idx for idx in range(num_entries) if bool(history_valid[:, idx].any().item())]
        else:
            valid_history_indices = [idx for idx in range(num_entries) if bool(history_valid[:, idx].any().item())]
            history_indices = valid_history_indices[-history_steps_per_update:]
        with torch.set_grad_enabled(grad_enabled):
            for entry_idx in history_indices:
                ref_image = history_video[:, entry_idx, :, 0]
                ref_tokens, ref_img_ids = self._encode_flux2_image_tokens(ref_image, time_value=10.0)
                gist_tokens, gist_ids = self._build_flux2_gist_inputs(
                    batch_size=batch_size,
                    ref_img_ids=ref_img_ids,
                    device=ref_tokens.device,
                    token_dtype=ref_tokens.dtype,
                    step_idx=entry_idx,
                )
                text_hidden = base_text_hidden
                text_mask = base_text_mask
                if self.proprio_encoder is not None:
                    text_hidden, text_mask = self._append_proprio_to_context_if_enabled(
                        context=base_text_hidden,
                        context_mask=base_text_mask,
                        proprio=history_proprio[:, entry_idx],
                        source="FLUX.2 history edit-memory training sample",
                    )
                history_memory_cache = None
                if entries:
                    history_memory_cache = combine_memory_entries(
                        select_hybrid_memory_entries(
                            entries,
                            anchor_frames=int(bank_cfg.get("anchor_frames", 2)),
                            recent_frames=self._edit_memory_historical_recent_frames(bank_cfg),
                            gist_tokens_per_frame=int(bank_cfg.get("gist_tokens_per_frame", 8)),
                        )
                    )
                video_pre, video_kv_cache = self._prefill_flux2_memory_prefix(
                    ref_tokens=ref_tokens,
                    ref_img_ids=ref_img_ids,
                    text_hidden=text_hidden,
                    text_mask=text_mask,
                    gist_tokens=gist_tokens,
                    gist_ids=gist_ids,
                    memory_kv_cache=history_memory_cache,
                )
                hist_action = history_action[:, entry_idx]
                noise_action = torch.randn_like(hist_action)
                timestep_action = self.train_action_scheduler.sample_training_t(
                    batch_size=batch_size,
                    device=self.device,
                    dtype=hist_action.dtype,
                )
                noisy_action = self.train_action_scheduler.add_noise(hist_action, noise_action, timestep_action)
                action_pre = self.action_expert.pre_dit(
                    action_tokens=noisy_action,
                    timestep=self._scheduler_timestep_to_unit(timestep_action, self.train_action_scheduler),
                )
                action_len = int(action_pre["tokens"].shape[1])
                full_attention_mask = self._build_mot_attention_mask_flux2(
                    batch_size=batch_size,
                    txt_len=int(video_pre["txt_len"]),
                    target_len=0,
                    cond_len=int(video_pre["cond_len"]),
                    action_len=action_len,
                    device=ref_tokens.device,
                    text_attention_mask=video_pre["text_mask"],
                    gist_len=int(video_pre.get("gist_len", 0)),
                )
                prefix_len = int(video_pre["txt_len"]) + int(video_pre["cond_len"]) + int(video_pre.get("gist_len", 0))
                grid_hw = self._grid_hw_from_flux2_img_ids(ref_img_ids)
                self._configure_action_attention_capture(
                    condition_slice=(int(video_pre["txt_len"]), int(video_pre["txt_len"]) + int(video_pre["cond_len"])),
                    condition_grid=grid_hw,
                    prefix_len=prefix_len,
                    action_len=action_len,
                    metadata={
                        "stack": str(self.stack),
                        "source": "flux2_train_history_edit_memory_teacher",
                        "condition": "history_ref_image_tokens",
                        "history_entry_idx": int(entry_idx),
                        "txt_len": int(video_pre["txt_len"]),
                        "cond_len": int(video_pre["cond_len"]),
                    },
                )
                self._start_action_attention_capture_step(entry_idx)
                _ = self.mot.forward_action_with_video_cache(
                    action_tokens=action_pre["tokens"],
                    action_freqs=None,
                    action_t_mod=action_pre["t_mod"],
                    action_context_payload={"ids": action_pre["ids"]},
                    video_kv_cache=video_kv_cache,
                    attention_mask=full_attention_mask,
                    video_seq_len=prefix_len,
                )
                teacher_importance = self.edit_memory_capture.get_per_layer_teacher_importance()
                memory_item = self._select_flux2_edit_memory_item(
                    video_kv_cache=video_kv_cache,
                    per_layer_teacher_importance=teacher_importance,
                    txt_len=int(video_pre["txt_len"]),
                    img_len=int(video_pre["cond_len"]),
                    grid_hw=grid_hw,
                    step_idx=int(entry_idx),
                    gist_len=int(video_pre.get("gist_len", 0)),
                )
                self.edit_memory_capture.deactivate()
                memory_item = self._attach_edit_memory_valid_mask(memory_item, history_valid[:, entry_idx])
                entries.append(memory_item)
                entries, _ = self._maintain_memorywam_entries(entries, bank_cfg)
        hybrid_entries = select_hybrid_memory_entries(
            entries,
            anchor_frames=int(bank_cfg.get("anchor_frames", 2)),
            recent_frames=self._edit_memory_historical_recent_frames(bank_cfg),
            gist_tokens_per_frame=int(bank_cfg.get("gist_tokens_per_frame", 8)),
        )
        return combine_memory_entries(hybrid_entries)

    def _training_loss_flux2_episode_window(self, sample, tiled: bool = False):
        del tiled
        training_cfg = self._edit_memory_training_cfg()
        required = bool(training_cfg["history_required"])
        needed = ("edit_memory_video", "edit_memory_action", "edit_memory_valid")
        if any(key not in sample for key in needed):
            if required:
                missing = [key for key in needed if key not in sample]
                raise ValueError(
                    "edit_memory.training.memory_source='episode_window' requires dataset-provided "
                    f"history fields, missing {missing}. Set data.train.edit_memory_history_offsets."
                )
            raise ValueError("episode-window edit-memory training requires history fields.")

        # History stays on host; only the selected credit target/action is
        # transferred. Reference VAE tokens are encoded once in synchronized
        # chunks and reused by both memory-only and credit forwards.
        history_video = sample["edit_memory_video"]
        history_action = sample["edit_memory_action"]
        history_valid = sample["edit_memory_valid"].to(device=self.device, dtype=torch.bool, non_blocking=True)
        current_video = sample["video"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        current_action = sample["action"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if history_video.ndim != 6:
            raise ValueError(
                "`sample['edit_memory_video']` must be [B,M,C,T,H,W], "
                f"got {tuple(history_video.shape)}"
            )
        if current_video.ndim != 5:
            raise ValueError(f"`sample['video']` must be [B,C,T,H,W], got {tuple(current_video.shape)}")
        if history_action.ndim != 4:
            raise ValueError(
                "`sample['edit_memory_action']` must be [B,M,A,D], "
                f"got {tuple(history_action.shape)}"
            )
        batch_size, num_history = int(history_video.shape[0]), int(history_video.shape[1])
        global_num_history_tensor = torch.tensor(
            num_history, device=history_valid.device, dtype=torch.long
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_num_history_tensor, op=dist.ReduceOp.MAX)
        global_num_history = int(global_num_history_tensor.item())
        if current_video.shape[0] != batch_size or current_action.shape[0] != batch_size:
            raise ValueError("Current and history edit-memory batch sizes are inconsistent.")

        history_proprio = sample.get("edit_memory_proprio")
        current_proprio = sample.get("proprio")
        if current_proprio is not None:
            current_proprio = current_proprio.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if self.proprio_encoder is not None and history_proprio is None:
            raise ValueError("`edit_memory_proprio` is required for episode-window memory training when proprio is enabled.")

        history_action_is_pad = sample.get("edit_memory_action_is_pad")
        current_action_is_pad = sample.get("action_is_pad")
        if current_action_is_pad is not None:
            current_action_is_pad = current_action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        history_action_dim_is_pad = sample.get("edit_memory_action_dim_is_pad")
        current_action_dim_is_pad = sample.get("action_dim_is_pad")
        if current_action_dim_is_pad is not None:
            current_action_dim_is_pad = current_action_dim_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        vlm_cfg = getattr(self, "edit_memory_config", {}).get("vlm_count_supervision", {})
        vlm_enabled = bool(vlm_cfg.get("enabled", False))
        history_episode_ids = history_timesteps = current_episode_ids = current_timesteps = None
        if vlm_enabled:
            identity_keys = (
                "edit_memory_episode_id",
                "edit_memory_timestep",
                "edit_memory_history_episode_id",
                "edit_memory_history_timestep",
            )
            missing_identity = [key for key in identity_keys if key not in sample]
            if missing_identity:
                raise ValueError(
                    "VLM count supervision requires exact dataset identities, missing "
                    f"{missing_identity}."
                )
            current_episode_ids = sample["edit_memory_episode_id"].to(
                device=self.device, dtype=torch.long, non_blocking=True
            )
            current_timesteps = sample["edit_memory_timestep"].to(
                device=self.device, dtype=torch.long, non_blocking=True
            )
            history_episode_ids = sample["edit_memory_history_episode_id"].to(
                device=self.device, dtype=torch.long, non_blocking=True
            )
            history_timesteps = sample["edit_memory_history_timestep"].to(
                device=self.device, dtype=torch.long, non_blocking=True
            )
            if history_episode_ids.shape != history_valid.shape or history_timesteps.shape != history_valid.shape:
                raise ValueError(
                    "VLM history identities must match edit_memory_valid shape: "
                    f"ids={tuple(history_episode_ids.shape)}, timesteps={tuple(history_timesteps.shape)}, "
                    f"valid={tuple(history_valid.shape)}."
                )

        base_text_hidden, base_text_mask = self._encode_flux2_text(sample)
        with torch.no_grad():
            current_encoded_ref = self._encode_flux2_image_tokens(
                current_video[:, :, 0], time_value=10.0
            )
            encoded_history_refs: list[tuple[torch.Tensor, torch.Tensor]] = []
            encode_chunk = int(training_cfg.get("static_encode_chunk_size", 8))
            if global_num_history > 0:
                local_refs = history_video[:, :, :, 0]
                if num_history < global_num_history:
                    padding = current_video[:, None, :, 0].expand(
                        -1, global_num_history - num_history, -1, -1, -1
                    ).to(device=local_refs.device, dtype=local_refs.dtype)
                    local_refs = torch.cat([local_refs, padding], dim=1)
                flat_refs = local_refs.reshape(
                    batch_size * global_num_history,
                    local_refs.shape[2],
                    local_refs.shape[3],
                    local_refs.shape[4],
                )
                token_chunks, id_chunks = [], []
                for start in range(0, int(flat_refs.shape[0]), encode_chunk):
                    tokens_l, ids_l = self._encode_flux2_image_tokens(
                        flat_refs[start : start + encode_chunk], time_value=10.0
                    )
                    token_chunks.append(tokens_l)
                    id_chunks.append(ids_l)
                history_tokens = torch.cat(token_chunks, dim=0).reshape(
                    batch_size,
                    global_num_history,
                    token_chunks[0].shape[1],
                    token_chunks[0].shape[2],
                )
                history_ids = torch.cat(id_chunks, dim=0).reshape(
                    batch_size,
                    global_num_history,
                    id_chunks[0].shape[1],
                    id_chunks[0].shape[2],
                )
                encoded_history_refs = [
                    (history_tokens[:, idx], history_ids[:, idx])
                    for idx in range(global_num_history)
                ]
        bank_cfg = getattr(self, "edit_memory_config", {}).get("bank", {})
        entries: list[dict[str, Any]] = []
        total_video_loss = None
        total_action_loss = None
        loss_step_count = 0
        total_evicted_frame_kv = 0
        processed_history_steps = 0
        total_memory_regularizer = None
        total_span_value = 0.0
        total_gist_value = 0.0
        adaptive_metric_steps = 0
        memory_regularizer_steps = 0
        gist_hard_histogram = [0 for _ in range(9)]
        gist_hard_samples = 0
        total_vlm_loss = None
        total_vlm_soft_label_value = 0.0
        total_vlm_agreement = 0.0
        total_vlm_expected_mae = 0.0
        total_vlm_coverage = 0.0
        total_vlm_zero_mapped = 0.0
        vlm_loss_steps = 0
        vlm_teacher_histogram = [0 for _ in range(9)]
        vlm_teacher_samples = 0
        curriculum_teacher_probability_total = 0.0
        curriculum_teacher_samples = 0
        curriculum_valid_samples = 0
        memory_only_history_steps = 0
        history_steps_per_update = int(training_cfg.get("history_steps_per_update", 0))
        if history_steps_per_update < 0:
            history_steps_per_update = 0

        def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            weights = weights.to(device=values.device, dtype=values.dtype)
            denom = weights.sum().clamp_min(1.0)
            return (values * weights).sum() / denom

        def _memory_only_step(
            *,
            ref_image: torch.Tensor | None,
            encoded_ref: tuple[torch.Tensor, torch.Tensor] | None = None,
            proprio: torch.Tensor | None,
            valid: torch.Tensor,
            step_idx: int,
            episode_ids: torch.Tensor | None,
            timesteps: torch.Tensor | None,
        ) -> tuple[
            dict[str, Any],
            dict[str, Any] | None,
            dict[str, torch.Tensor],
            dict[str, Any],
            dict[str, Any] | None,
            dict[str, Any],
        ]:
            """Build the exact train/infer memory write without video or action targets."""
            if encoded_ref is None:
                if ref_image is None:
                    raise ValueError("ref_image is required when encoded_ref is absent.")
                ref_tokens, ref_img_ids = self._encode_flux2_image_tokens(ref_image, time_value=10.0)
            else:
                ref_tokens, ref_img_ids = encoded_ref
            text_hidden, text_mask = base_text_hidden, base_text_mask
            if self.proprio_encoder is not None:
                text_hidden, text_mask = self._append_proprio_to_context_if_enabled(
                    context=base_text_hidden,
                    context_mask=base_text_mask,
                    proprio=proprio,
                    source="FLUX.2 episode-window edit-memory training sample",
                )
            frame_summary = self._edit_memory_frame_summary(ref_tokens, text_hidden, text_mask)
            span_decision = self._edit_memory_span_decision(frame_summary)
            memory_cache = None
            if entries:
                memory_cache = combine_memory_entries(
                    select_hybrid_memory_entries(
                        entries,
                        anchor_frames=int(bank_cfg.get("anchor_frames", 2)),
                        recent_frames=self._edit_memory_historical_recent_frames(bank_cfg),
                        gist_tokens_per_frame=int(bank_cfg.get("gist_tokens_per_frame", 8)),
                        history_span=span_decision,
                        compact_dynamic=False,
                    )
                )
            gist_tokens, gist_ids = self._build_flux2_gist_inputs(
                batch_size=batch_size,
                ref_img_ids=ref_img_ids,
                device=ref_tokens.device,
                token_dtype=ref_tokens.dtype,
                step_idx=step_idx,
            )
            memory_prefix_pre, memory_prefix_kv_cache = self._prefill_flux2_memory_prefix(
                ref_tokens=ref_tokens,
                ref_img_ids=ref_img_ids,
                text_hidden=text_hidden,
                text_mask=text_mask,
                gist_tokens=gist_tokens,
                gist_ids=gist_ids,
                memory_kv_cache=memory_cache,
            )
            prefix_summary = self._edit_memory_prefix_summary(memory_prefix_kv_cache)
            predicted_gist_decision = self._edit_memory_gist_decision(prefix_summary)
            write_decision = self._edit_memory_write_confidence(
                prefix_summary, predicted_gist_decision
            )
            if predicted_gist_decision is not None:
                predicted_gist_decision["write_decision"] = write_decision
            vlm_metrics = self._edit_memory_vlm_count_supervision(
                predicted_gist_decision,
                episode_ids,
                timesteps,
                valid,
                write_decision=write_decision,
            )
            curriculum_random = None
            if episode_ids is not None and timesteps is not None:
                hashed = (
                    episode_ids.to(device=self.device, dtype=torch.long).reshape(-1) * 1_000_003
                    + timesteps.to(device=self.device, dtype=torch.long).reshape(-1) * 9_176
                    + int(getattr(self, "_edit_memory_training_step", 0)) * 37
                )
                curriculum_random = torch.remainder(hashed, 1_000_033).float() / 1_000_033.0
            write_gist_decision, curriculum_metrics = self._edit_memory_write_decision(
                predicted_gist_decision,
                vlm_metrics,
                random_values=curriculum_random,
                write_decision=write_decision,
            )
            memory_item = self._select_flux2_edit_memory_item(
                video_kv_cache=memory_prefix_kv_cache,
                per_layer_teacher_importance=None,
                txt_len=int(memory_prefix_pre["txt_len"]),
                img_len=int(memory_prefix_pre["cond_len"]),
                grid_hw=self._grid_hw_from_flux2_img_ids(ref_img_ids),
                step_idx=int(step_idx),
                gist_len=int(memory_prefix_pre.get("gist_len", 0)),
                gist_decision=write_gist_decision,
                write_confidence=curriculum_metrics["write_confidence"],
                compact_adaptive_gist=False,
            )
            hard_write = curriculum_metrics["hard_write"]
            gist_valid = valid if hard_write is None else valid & hard_write
            memory_item = self._attach_edit_memory_valid_mask(
                memory_item,
                valid,
                gist_valid=gist_valid,
            )
            zero_regularizer = ref_tokens.new_zeros((), dtype=torch.float32)
            regularizers = {
                "span": zero_regularizer
                if span_decision is None
                else span_decision["regularizer"].float(),
                "gist": zero_regularizer
                if predicted_gist_decision is None
                else predicted_gist_decision["regularizer"].float(),
            }
            metrics = {
                "span": None if span_decision is None else span_decision["soft_span"].detach(),
                "gist": None
                if predicted_gist_decision is None
                else predicted_gist_decision["expected_count"].detach(),
                "gist_hard": None
                if predicted_gist_decision is None
                else predicted_gist_decision["hard_count"].detach(),
                "write_gist_hard": None
                if write_gist_decision is None
                else write_gist_decision["hard_count"].detach(),
                "teacher_probability": float(curriculum_metrics["teacher_probability"]),
                "teacher_mask": curriculum_metrics["teacher_mask"].detach(),
                "write_confidence": None
                if curriculum_metrics["write_confidence"] is None
                else curriculum_metrics["write_confidence"].detach(),
                "hard_write": None
                if hard_write is None
                else hard_write.detach(),
            }
            shared = {
                "ref_tokens": ref_tokens,
                "ref_img_ids": ref_img_ids,
                "text_hidden": text_hidden,
                "text_mask": text_mask,
                "gist_tokens": gist_tokens,
                "gist_ids": gist_ids,
                "memory_prefix_pre": memory_prefix_pre,
                "memory_prefix_kv_cache": memory_prefix_kv_cache,
                "prefix_summary": prefix_summary,
            }
            return memory_item, memory_cache, regularizers, metrics, vlm_metrics, shared

        def _step(
            *,
            ref_image: torch.Tensor | None,
            encoded_ref: tuple[torch.Tensor, torch.Tensor] | None = None,
            target_image: torch.Tensor,
            action: torch.Tensor,
            action_is_pad: torch.Tensor | None,
            action_dim_is_pad: torch.Tensor | None,
            proprio: torch.Tensor | None,
            valid: torch.Tensor,
            step_idx: int,
            episode_ids: torch.Tensor | None,
            timesteps: torch.Tensor | None,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            dict[str, Any],
            dict[str, Any] | None,
            dict[str, torch.Tensor],
            dict[str, Any],
            dict[str, Any] | None,
        ]:
            memory_item, memory_cache, regularizers, metrics, vlm_metrics, shared = _memory_only_step(
                ref_image=ref_image,
                encoded_ref=encoded_ref,
                proprio=proprio,
                valid=valid,
                step_idx=step_idx,
                episode_ids=episode_ids,
                timesteps=timesteps,
            )
            ref_tokens = shared["ref_tokens"]
            ref_img_ids = shared["ref_img_ids"]
            text_hidden = shared["text_hidden"]
            text_mask = shared["text_mask"]
            gist_tokens = shared["gist_tokens"]
            gist_ids = shared["gist_ids"]
            memory_prefix_pre = shared["memory_prefix_pre"]
            memory_prefix_kv_cache = shared["memory_prefix_kv_cache"]
            target_tokens, target_img_ids = self._encode_flux2_image_tokens(target_image, time_value=0.0)
            noise_video = torch.randn_like(target_tokens)
            timestep_video = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=target_tokens.dtype,
            )
            noisy_latent = self.train_video_scheduler.add_noise(target_tokens, noise_video, timestep_video)
            target_video = self.train_video_scheduler.training_target(target_tokens, noise_video, timestep_video)
            noise_action = torch.randn_like(action)
            timestep_action = self.train_action_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=action.dtype,
            )
            noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
            target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)
            video_pre = self.video_expert.pre_dit(
                x=noisy_latent,
                timestep=self._scheduler_timestep_to_unit(timestep_video, self.train_video_scheduler),
                context=text_hidden,
                context_mask=text_mask,
                ref_image_hidden_states=ref_tokens,
                gist_hidden_states=gist_tokens,
                target_img_ids=target_img_ids,
                ref_img_ids=ref_img_ids,
                gist_img_ids=gist_ids,
            )
            prefix_attention_mask = self._build_mot_attention_mask_flux2(
                batch_size=batch_size,
                txt_len=int(video_pre["txt_len"]),
                target_len=int(video_pre["target_len"]),
                cond_len=int(video_pre["cond_len"]),
                action_len=0,
                device=target_tokens.device,
                text_attention_mask=video_pre["text_mask"],
                gist_len=int(video_pre.get("gist_len", 0)),
            )
            video_kv_cache = self.mot.prefill_flux2_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                attention_mask=prefix_attention_mask,
                memory_kv_cache=memory_cache,
            )
            pred_video = self.video_expert.post_dit(video_kv_cache["final_video"], video_pre)
            action_pre = self.action_expert.pre_dit(
                action_tokens=noisy_action,
                timestep=self._scheduler_timestep_to_unit(timestep_action, self.train_action_scheduler),
            )
            action_len = int(action_pre["tokens"].shape[1])
            full_attention_mask = self._build_mot_attention_mask_flux2(
                batch_size=batch_size,
                txt_len=int(memory_prefix_pre["txt_len"]),
                target_len=0,
                cond_len=int(memory_prefix_pre["cond_len"]),
                action_len=action_len,
                device=target_tokens.device,
                text_attention_mask=memory_prefix_pre["text_mask"],
                gist_len=int(memory_prefix_pre.get("gist_len", 0)),
            )
            prefix_len = (
                int(memory_prefix_pre["txt_len"])
                + int(memory_prefix_pre["cond_len"])
                + int(memory_prefix_pre.get("gist_len", 0))
            )
            action_tokens = self.mot.forward_action_with_video_cache(
                action_tokens=action_pre["tokens"],
                action_freqs=None,
                action_t_mod=action_pre["t_mod"],
                action_context_payload={"ids": action_pre["ids"]},
                video_kv_cache=memory_prefix_kv_cache,
                attention_mask=full_attention_mask,
                video_seq_len=prefix_len,
                memory_kv_cache=memory_cache,
            )
            pred_action = self.action_expert.post_dit(action_tokens, action_pre)
            video_loss_per_sample = F.mse_loss(
                pred_video.float(),
                target_video.float(),
                reduction="none",
            ).flatten(1).mean(dim=1)
            video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
                video_loss_per_sample.device,
                dtype=video_loss_per_sample.dtype,
            )
            loss_video = _weighted_mean(video_loss_per_sample * video_weight, valid)
            action_loss_per_sample = self._compute_action_loss_per_sample(
                pred_action=pred_action,
                target_action=target_action,
                action_is_pad=action_is_pad,
                action_dim_is_pad=action_dim_is_pad,
            )
            action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
                action_loss_per_sample.device,
                dtype=action_loss_per_sample.dtype,
            )
            loss_action = _weighted_mean(action_loss_per_sample * action_weight, valid)
            return loss_video, loss_action, memory_item, memory_cache, regularizers, metrics, vlm_metrics

        def _accumulate_vlm(metrics: dict[str, Any] | None) -> None:
            nonlocal total_vlm_loss, total_vlm_soft_label_value
            nonlocal total_vlm_agreement, total_vlm_expected_mae, total_vlm_coverage
            nonlocal total_vlm_zero_mapped, vlm_loss_steps, vlm_teacher_samples
            if metrics is None:
                return
            total_vlm_loss = metrics["loss"] if total_vlm_loss is None else total_vlm_loss + metrics["loss"]
            total_vlm_soft_label_value += float(metrics["count_loss"].detach().item())
            total_vlm_agreement += float(metrics["agreement"].detach().item())
            total_vlm_expected_mae += float(metrics["expected_mae"].detach().item())
            total_vlm_coverage += float(metrics["coverage"].detach().item())
            total_vlm_zero_mapped += float(metrics["zero_mapped_frac"].detach().item())
            active = metrics["active"]
            for count in metrics["teacher_count"][active].tolist():
                vlm_teacher_histogram[int(count)] += 1
                vlm_teacher_samples += 1
            vlm_loss_steps += 1

        def _accumulate_curriculum(metrics: dict[str, Any]) -> None:
            nonlocal curriculum_teacher_probability_total
            nonlocal curriculum_teacher_samples, curriculum_valid_samples
            curriculum_teacher_probability_total += float(metrics["teacher_probability"])
            teacher_mask = metrics["teacher_mask"]
            curriculum_teacher_samples += int(teacher_mask.sum().item())
            curriculum_valid_samples += int(teacher_mask.numel())

        history_step_indices = [
            idx for idx in range(num_history) if bool(history_valid[:, idx].any().item())
        ]
        credit_strategy = str(training_cfg.get("history_credit_sampling", "recent"))
        selected_history_slot = None
        if history_steps_per_update > 0 and history_step_indices:
            if history_steps_per_update != 1:
                raise ValueError(
                    "Rank-synchronous ZeRO-3 episode training currently requires "
                    "history_steps_per_update=1."
                )
            if credit_strategy == "recent":
                selected_history_slot = history_step_indices[-1]
            elif credit_strategy == "uniform":
                rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
                local_position = self._rank_synchronous_history_slot(
                    len(history_step_indices),
                    int(getattr(self, "_edit_memory_training_step", 0)),
                    seed=3407 + int(rank),
                )
                selected_history_slot = history_step_indices[int(local_position)]
            else:
                raise ValueError(
                    "edit_memory.training.history_credit_sampling must be recent or uniform; "
                    f"got {credit_strategy!r}."
                )
        trainable_history_indices = (
            set() if selected_history_slot is None else {selected_history_slot}
        )
        credit_distances = [num_history - index for index in trainable_history_indices]
        anchor_count = min(max(int(bank_cfg.get("anchor_frames", 2)), 0), num_history)
        recent_start = max(0, num_history - self._edit_memory_historical_recent_frames(bank_cfg))
        credit_gist_count = sum(
            anchor_count <= index < recent_start for index in trainable_history_indices
        )

        selected_entries_before: list[dict[str, Any]] | None = None
        for hist_idx in range(global_num_history):
            local_history_exists = hist_idx < num_history
            valid = (
                history_valid[:, hist_idx]
                if local_history_exists
                else torch.zeros(batch_size, device=self.device, dtype=torch.bool)
            )
            if local_history_exists and bool(valid.any().item()):
                processed_history_steps += 1
            if hist_idx == selected_history_slot:
                selected_entries_before = list(entries)
            encoded_ref = (
                encoded_history_refs[hist_idx]
                if local_history_exists
                else current_encoded_ref
            )
            proprio_l = (
                current_proprio
                if not local_history_exists
                else None
                if history_proprio is None
                else history_proprio[:, hist_idx].to(
                    device=self.device, dtype=self.torch_dtype, non_blocking=True
                )
            )
            with torch.no_grad():
                (
                    memory_item,
                    unused_memory_cache,
                    regularizers,
                    adaptive_metrics,
                    vlm_metrics,
                    unused_shared,
                ) = _memory_only_step(
                    ref_image=None,
                    encoded_ref=encoded_ref,
                    proprio=proprio_l,
                    valid=valid,
                    step_idx=hist_idx,
                    episode_ids=(
                        current_episode_ids
                        if not local_history_exists
                        else None if history_episode_ids is None else history_episode_ids[:, hist_idx]
                    ),
                    timesteps=(
                        current_timesteps
                        if not local_history_exists
                        else None if history_timesteps is None else history_timesteps[:, hist_idx]
                    ),
                )
                if local_history_exists and bool(valid.any().item()):
                    memory_only_history_steps += 1
            # The selected MemoryBank item owns every tensor needed by later
            # history. Drop the full clean-prefix intermediates immediately so
            # the next memory-only forward cannot overlap their GPU lifetime.
            del unused_memory_cache, unused_shared, regularizers, vlm_metrics
            if local_history_exists and bool(valid.any().item()):
                _accumulate_curriculum(adaptive_metrics)
                if bool(memory_item["memory_is_valid"].any().item()):
                    entries.append(memory_item)
                    entries, num_evicted = self._maintain_memorywam_entries(entries, bank_cfg)
                    total_evicted_frame_kv += int(num_evicted)
            del memory_item, adaptive_metrics
        # Phase 2: every rank executes exactly one full historical credit
        # forward. The chosen slot is local, so every non-empty episode gets a
        # real history gradient without changing ZeRO-3 module order.
        chronological_entries = entries
        if history_steps_per_update > 0 and global_num_history > 0:
            has_valid_credit = selected_history_slot is not None
            credit_valid = (
                history_valid[:, selected_history_slot]
                if has_valid_credit
                else torch.zeros(batch_size, device=self.device, dtype=torch.bool)
            )
            entries = (
                list(selected_entries_before)
                if selected_entries_before is not None
                else list(chronological_entries)
            )
            selected_idx = int(selected_history_slot) if has_valid_credit else -1
            loss_video, loss_action, credit_item, _, regularizers, adaptive_metrics, vlm_metrics = _step(
                ref_image=None,
                encoded_ref=encoded_history_refs[selected_idx] if has_valid_credit else current_encoded_ref,
                target_image=(
                    history_video[:, selected_idx, :, -1]
                    if has_valid_credit
                    else current_video[:, :, -1]
                ),
                action=(
                    history_action[:, selected_idx].to(
                        device=self.device, dtype=self.torch_dtype, non_blocking=True
                    )
                    if has_valid_credit
                    else current_action
                ),
                action_is_pad=(
                    None
                    if history_action_is_pad is None or not has_valid_credit
                    else history_action_is_pad[:, selected_idx].to(
                        device=self.device, dtype=torch.bool, non_blocking=True
                    )
                ),
                action_dim_is_pad=(
                    None
                    if history_action_dim_is_pad is None or not has_valid_credit
                    else history_action_dim_is_pad[:, selected_idx].to(
                        device=self.device, dtype=torch.bool, non_blocking=True
                    )
                ),
                proprio=(
                    current_proprio
                    if not has_valid_credit
                    else None
                    if history_proprio is None
                    else history_proprio[:, selected_idx].to(
                        device=self.device, dtype=self.torch_dtype, non_blocking=True
                    )
                ),
                valid=credit_valid,
                step_idx=max(selected_idx, 0),
                episode_ids=(
                    current_episode_ids
                    if not has_valid_credit
                    else None if history_episode_ids is None else history_episode_ids[:, selected_idx]
                ),
                timesteps=(
                    current_timesteps
                    if not has_valid_credit
                    else None if history_timesteps is None else history_timesteps[:, selected_idx]
                ),
            )
            entries = chronological_entries
            if has_valid_credit:
                for entry_idx, entry in enumerate(entries):
                    if int(entry.get("step_start", -1)) == selected_idx:
                        entries[entry_idx] = credit_item
                        break
            total_video_loss = loss_video if total_video_loss is None else total_video_loss + loss_video
            total_action_loss = loss_action if total_action_loss is None else total_action_loss + loss_action
            if has_valid_credit:
                loss_step_count += 1
            credit_scale = credit_valid.float().mean()
            step_regularizer = (regularizers["span"] + regularizers["gist"]) * credit_scale
            total_memory_regularizer = (
                step_regularizer
                if total_memory_regularizer is None
                else total_memory_regularizer + step_regularizer
            )
            if has_valid_credit:
                memory_regularizer_steps += 1
                _accumulate_vlm(vlm_metrics)
                if adaptive_metrics["span"] is not None:
                    total_span_value += float(adaptive_metrics["span"].float().mean().item())
                if adaptive_metrics["gist"] is not None:
                    total_gist_value += float(adaptive_metrics["gist"].float().mean().item())
                if adaptive_metrics["gist_hard"] is not None:
                    for count in adaptive_metrics["gist_hard"].flatten().tolist():
                        gist_hard_histogram[int(count)] += 1
                        gist_hard_samples += 1
                adaptive_metric_steps += 1
            elif vlm_metrics is not None:
                # Keep this phase-2 VLM/write-head invocation in the backward
                # graph without counting a dummy target in the loss average.
                # The inactive VLM loss is differentiable but exactly zero.
                total_vlm_loss = (
                    vlm_metrics["loss"]
                    if total_vlm_loss is None
                    else total_vlm_loss + vlm_metrics["loss"]
                )

        current_valid = torch.ones(batch_size, device=self.device, dtype=torch.bool)
        loss_video, loss_action, memory_item, _, regularizers, adaptive_metrics, vlm_metrics = _step(
            ref_image=None,
            encoded_ref=current_encoded_ref,
            target_image=current_video[:, :, -1],
            action=current_action,
            action_is_pad=current_action_is_pad,
            action_dim_is_pad=current_action_dim_is_pad,
            proprio=current_proprio,
            valid=current_valid,
            step_idx=num_history,
            episode_ids=current_episode_ids,
            timesteps=current_timesteps,
        )
        _accumulate_curriculum(adaptive_metrics)
        if bool(memory_item["memory_is_valid"].any().item()):
            entries.append(memory_item)
            entries, num_evicted = self._maintain_memorywam_entries(entries, bank_cfg)
            total_evicted_frame_kv += int(num_evicted)
        total_video_loss = loss_video if total_video_loss is None else total_video_loss + loss_video
        total_action_loss = loss_action if total_action_loss is None else total_action_loss + loss_action
        loss_step_count += 1
        # The current span affects the current prediction. Its new gist is not read
        # until a later replan, so charging its token cost here would be one-sided.
        current_regularizer = regularizers["span"]
        total_memory_regularizer = (
            current_regularizer
            if total_memory_regularizer is None
            else total_memory_regularizer + current_regularizer
        )
        memory_regularizer_steps += 1
        _accumulate_vlm(vlm_metrics)
        if adaptive_metrics["span"] is not None:
            total_span_value += float(adaptive_metrics["span"].float().mean().item())
        if adaptive_metrics["gist"] is not None:
            total_gist_value += float(adaptive_metrics["gist"].float().mean().item())
        if adaptive_metrics["gist_hard"] is not None:
            for count in adaptive_metrics["gist_hard"].flatten().tolist():
                gist_hard_histogram[int(count)] += 1
                gist_hard_samples += 1
        adaptive_metric_steps += 1
        num_loss_steps = max(loss_step_count, 1)
        loss_video = total_video_loss / float(num_loss_steps)
        loss_action = total_action_loss / float(num_loss_steps)
        loss_memory_regularizer = (
            loss_video.new_zeros(())
            if total_memory_regularizer is None
            else total_memory_regularizer / float(max(memory_regularizer_steps, 1))
        )
        loss_vlm_count = (
            loss_video.new_zeros(())
            if total_vlm_loss is None
            else total_vlm_loss / float(max(vlm_loss_steps, 1))
        )
        loss_total = (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_action * loss_action
            + loss_memory_regularizer
            + loss_vlm_count
        )

        final_memory_cache = combine_memory_entries(
            select_hybrid_memory_entries(
                entries,
                anchor_frames=int(bank_cfg.get("anchor_frames", 2)),
                recent_frames=self._edit_memory_historical_recent_frames(bank_cfg),
                gist_tokens_per_frame=int(bank_cfg.get("gist_tokens_per_frame", 8)),
            )
        )
        confidence_value = 0.0 if final_memory_cache is None else float(
            memory_item_confidence(final_memory_cache).detach().float().mean().item()
        )
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            "loss_edit_memory_action": self.loss_lambda_action * float(loss_action.detach().item()),
            "loss_edit_memory_distill": 0.0,
            "loss_edit_memory_vlm_count": float(loss_vlm_count.detach().item()),
            "loss_edit_memory_vlm_soft_label": total_vlm_soft_label_value
            / float(max(vlm_loss_steps, 1)),
            "edit_memory_vlm_agreement": total_vlm_agreement / float(max(vlm_loss_steps, 1)),
            "edit_memory_vlm_expected_count_mae": total_vlm_expected_mae / float(max(vlm_loss_steps, 1)),
            "edit_memory_vlm_coverage": total_vlm_coverage / float(max(vlm_loss_steps, 1)),
            "edit_memory_vlm_zero_mapped_frac": total_vlm_zero_mapped / float(max(vlm_loss_steps, 1)),
            "edit_memory_training_source_history": 1.0,
            "edit_memory_train_history_valid_frac": (
                0.0 if history_valid.numel() == 0 else float(history_valid.float().mean().item())
            ),
            "edit_memory_episode_window_steps": float(processed_history_steps + 1),
            "edit_memory_global_history_slots": float(global_num_history),
            "edit_memory_selected_history_slot": float(
                -1 if selected_history_slot is None else selected_history_slot
            ),
            "edit_memory_memory_only_history_steps": float(memory_only_history_steps),
            "edit_memory_curriculum_teacher_probability": (
                curriculum_teacher_probability_total / float(max(processed_history_steps + 1, 1))
            ),
            "edit_memory_curriculum_teacher_frac": (
                float(curriculum_teacher_samples) / float(max(curriculum_valid_samples, 1))
            ),
            "edit_memory_training_step": float(getattr(self, "_edit_memory_training_step", 0)),
            "edit_memory_credit_distance": (
                0.0 if not credit_distances else float(sum(credit_distances)) / float(len(credit_distances))
            ),
            "edit_memory_credit_gist_frac": (
                0.0
                if not trainable_history_indices
                else float(credit_gist_count) / float(len(trainable_history_indices))
            ),
            "edit_memory_frame_kv_evicted": float(total_evicted_frame_kv),
            "loss_edit_memory_regularizer": float(loss_memory_regularizer.detach().item()),
            "edit_memory_adaptive_span_frames": total_span_value / float(max(adaptive_metric_steps, 1)),
            "edit_memory_adaptive_gist_tokens": total_gist_value / float(max(adaptive_metric_steps, 1)),
        }
        for count in range(1, 9):
            loss_dict[f"edit_memory_hard_p_n{count}"] = (
                float(gist_hard_histogram[count]) / float(max(gist_hard_samples, 1))
            )
            loss_dict[f"edit_memory_vlm_teacher_p_n{count}"] = (
                float(vlm_teacher_histogram[count]) / float(max(vlm_teacher_samples, 1))
            )
        loss_dict.update(
            self._edit_memory_log_payload(
                final_memory_cache,
                num_entries=0 if final_memory_cache is None else int(final_memory_cache.get("num_entries", 0)),
                write_confidence=confidence_value,
                write_happened=final_memory_cache is not None,
                num_merged=total_evicted_frame_kv,
            )
        )
        return loss_total, loss_dict

    @staticmethod
    def _edit_memory_log_payload(
        memory_item: dict[str, Any] | None,
        *,
        num_entries: int,
        write_confidence: float,
        write_happened: bool,
        num_merged: int,
    ) -> dict[str, float]:
        return {
            "edit_memory_num_entries": float(num_entries),
            "edit_memory_num_tokens_per_layer": float(memory_tokens_per_layer(memory_item)),
            "edit_memory_write_confidence": float(write_confidence),
            "edit_memory_write_happened": 1.0 if write_happened else 0.0,
            "edit_memory_num_merged": float(num_merged),
            "edit_memory_num_evicted_frame_kv": float(num_merged),
            "edit_memory_kv_tokens_total": float(memory_kv_tokens_total(memory_item)),
        }

    @staticmethod
    def _image_tensor_to_pil_list(image: torch.Tensor) -> list[Image.Image]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"`image` must be [B,3,H,W], got {tuple(image.shape)}")
        image = image.detach().float().cpu()
        if image.min() < 0:
            image = (image + 1.0) * 0.5
        image = image.clamp(0, 1)
        arr = (image.permute(0, 2, 3, 1).numpy() * 255.0).round().astype("uint8")
        return [Image.fromarray(item) for item in arr]

    def _build_dim_user_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        tokenizer = self.dim_processor.tokenizer
        im_start_id = tokenizer("<|im_start|>").input_ids[0]
        im_end_id = tokenizer("<|im_end|>").input_ids[0]
        user_id = tokenizer("user").input_ids[0]
        newline_id = tokenizer("\n").input_ids[0]
        labels = torch.full_like(input_ids, -100, dtype=torch.long)
        for b in range(input_ids.shape[0]):
            user_begin, user_end = [], []
            for idx in range(3, input_ids.shape[1]):
                if (
                    input_ids[b, idx - 3] == im_start_id
                    and input_ids[b, idx - 2] == user_id
                    and input_ids[b, idx - 1] == newline_id
                ):
                    user_begin.append(idx)
                if input_ids[b, idx] == im_end_id and len(user_begin) == len(user_end) + 1:
                    user_end.append(idx)
            if len(user_begin) != len(user_end):
                raise ValueError("DIM Qwen user span parse failed: begin/end counts mismatch.")
            for start, end in zip(user_begin, user_end):
                labels[b, start:end] = input_ids[b, start:end]
        return labels

    def _build_dim_messages(self, instruction: str | Sequence[str], image: torch.Tensor):
        pil_images = self._image_tensor_to_pil_list(image)
        if isinstance(instruction, str):
            instructions = [instruction] * len(pil_images)
        else:
            instructions = list(instruction)
        if len(instructions) != len(pil_images):
            raise ValueError(f"Expected {len(pil_images)} DIM instructions, got {len(instructions)}.")
        messages = []
        for img, prompt in zip(pil_images, instructions):
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": str(prompt)},
                        ],
                    }
                ]
            )
        return messages

    def _prepare_dim_condition_from_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_encoder is None or not hasattr(self, "dim_projector"):
            raise ValueError("DIM online conditioning requires loaded Qwen MLLM and projector.")
        batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        with torch.no_grad():
            outputs = self.text_encoder(
                **batch,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]
        projected = self.dim_projector(hidden.to(dtype=self.torch_dtype))
        user_labels = self._build_dim_user_labels(batch["input_ids"]).to(projected.device)
        valid = (user_labels != -100) & batch["attention_mask"].to(device=projected.device, dtype=torch.bool)
        max_len = int(getattr(self, "dim_max_condition_length", projected.shape[1]))
        out = projected.new_zeros(projected.shape[0], max_len, projected.shape[-1])
        mask = torch.zeros(projected.shape[0], max_len, device=projected.device, dtype=torch.bool)
        for b in range(projected.shape[0]):
            item = projected[b, valid[b]]
            if item.shape[0] > max_len:
                item = item[-max_len:]
            out[b, : item.shape[0]] = item
            mask[b, : item.shape[0]] = True
        return out, mask

    def _encode_dim_condition_online(
        self,
        instruction: str | Sequence[str],
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        messages = self._build_dim_messages(instruction, image)
        texts = [
            self.dim_processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in messages
        ]
        pil_images = [msg[0]["content"][0]["image"] for msg in messages]
        batch = self.dim_processor(
            text=texts,
            images=pil_images,
            padding=True,
            return_tensors="pt",
        )
        return self._prepare_dim_condition_from_batch(batch)

    @torch.no_grad()
    def _encode_dim_image_latents(self, image: torch.Tensor) -> torch.Tensor:
        return self.video_expert.encode_image_latents(image.to(device=self.device, dtype=self.torch_dtype))

    @torch.no_grad()
    def _decode_dim_image_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.video_expert.decode_image_latents(latents)

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "ImageWAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for ImageWAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        action_dim_is_pad = sample.get("action_dim_is_pad", None)
        if action_dim_is_pad is not None:
            if action_dim_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_dim_is_pad']` must be 2D [B, D], got shape {tuple(action_dim_is_pad.shape)}"
                )
            if action_dim_is_pad.shape[0] != batch_size or action_dim_is_pad.shape[1] != action.shape[2]:
                raise ValueError(
                    "`sample['action_dim_is_pad']` shape mismatch: "
                    f"got {tuple(action_dim_is_pad.shape)} vs expected ({batch_size}, {action.shape[2]})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
            "image_is_pad": image_is_pad,
        }

    @torch.no_grad()
    def _encode_omnigen2_image_latents(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(device=self.device, dtype=getattr(self.vae, "dtype", self.torch_dtype), non_blocking=True)
        latent = self.vae.encode(image).latent_dist.sample()
        if getattr(self.vae.config, "shift_factor", None) is not None:
            latent = latent - self.vae.config.shift_factor
        if getattr(self.vae.config, "scaling_factor", None) is not None:
            latent = latent * self.vae.config.scaling_factor
        return latent.to(dtype=self.torch_dtype)

    @torch.no_grad()
    def _decode_omnigen2_image_latents(self, latents: torch.Tensor) -> torch.Tensor:
        z = latents.to(device=self.device, dtype=getattr(self.vae, "dtype", self.torch_dtype))
        if getattr(self.vae.config, "scaling_factor", None) is not None:
            z = z / self.vae.config.scaling_factor
        if getattr(self.vae.config, "shift_factor", None) is not None:
            z = z + self.vae.config.shift_factor
        decoded = self.vae.decode(z)
        image = decoded.sample if hasattr(decoded, "sample") else decoded[0]
        return image.detach().float().clamp(-1, 1)

    @torch.no_grad()
    def _encode_omnigen2_text(self, sample) -> tuple[torch.Tensor, torch.Tensor]:
        if "text_hidden_states" in sample and "text_attention_mask" in sample:
            return (
                sample["text_hidden_states"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
                sample["text_attention_mask"].to(device=self.device, dtype=torch.bool, non_blocking=True),
            )
        if "text_ids" in sample and "text_mask" in sample:
            text_ids = sample["text_ids"].to(device=self.device, non_blocking=True)
            text_mask = sample["text_mask"].to(device=self.device, dtype=torch.bool, non_blocking=True)
        else:
            if self.text_encoder is None or self.tokenizer is None:
                raise ValueError(
                    "OmniGen2 stack needs precomputed text_hidden_states/text_attention_mask "
                    "or a loaded Qwen tokenizer/text_encoder."
                )
            instruction = sample.get("instruction")
            if instruction is None:
                raise ValueError("OmniGen2 stack sample must include `instruction` when text is not precomputed.")
            if self.omnigen2_online_text_cache_compatible:
                encoded = self.tokenizer(
                    instruction,
                    padding="max_length",
                    truncation=True,
                    max_length=self.qwen_context_len,
                    return_tensors="pt",
                )
            else:
                encoded = self.tokenizer(
                    instruction,
                    padding=True,
                    return_tensors="pt",
                )
            text_ids = encoded.input_ids.to(self.device)
            text_mask = encoded.attention_mask.to(self.device, dtype=torch.bool)
        text_hidden_states = self.text_encoder(
            input_ids=text_ids,
            attention_mask=text_mask,
            output_hidden_states=False,
        ).last_hidden_state
        return text_hidden_states.to(dtype=self.torch_dtype), text_mask

    @torch.no_grad()
    def _prepare_omnigen2_infer_text(
        self,
        prompt: Optional[str],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            return self._encode_omnigen2_text({"instruction": [prompt]})
        if context is None or context_mask is None:
            raise ValueError("`context` and `context_mask` must be both provided together.")
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        return (
            context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
        )

    def build_inputs_omnigen2(self, sample, tiled: bool = False):
        debug_forward = (
            int(os.environ.get("IMAGEWAM_DEBUG_OMNIGEN2_FORWARD_EVERY", "0") or "0") > 0
            or float(os.environ.get("IMAGEWAM_DEBUG_OMNIGEN2_FORWARD_THRESHOLD", "0") or "0") > 0
        )
        build_profile: list[tuple[str, float]] = []

        def _sync_and_mark(name: str, start_time: float) -> float:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            now = time.perf_counter()
            build_profile.append((name, now - start_time))
            return now

        t0 = time.perf_counter()
        video = sample.get("video")
        if "target_latent" in sample:
            target_latent = sample["target_latent"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        else:
            next_frame = sample.get("next_frame", sample.get("target_image"))
            if next_frame is None and video is not None:
                if video.ndim != 5:
                    raise ValueError(f"`sample['video']` must be [B,C,T,H,W], got {tuple(video.shape)}")
                next_frame = video[:, :, -1]
            if next_frame is None:
                raise ValueError("OmniGen2 stack sample requires `next_frame`, `target_image`, `video`, or `target_latent`.")
            target_latent = self._encode_omnigen2_image_latents(next_frame)
        if debug_forward:
            t0 = _sync_and_mark("build.target_latent", t0)

        if "ref_image_latents" in sample:
            ref_image_latents = sample["ref_image_latents"]
            if isinstance(ref_image_latents, torch.Tensor):
                ref_image_latents = ref_image_latents.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        else:
            current_frame = sample.get("current_frame", sample.get("input_image"))
            if current_frame is None and video is not None:
                if video.ndim != 5:
                    raise ValueError(f"`sample['video']` must be [B,C,T,H,W], got {tuple(video.shape)}")
                current_frame = video[:, :, 0]
            if current_frame is None:
                raise ValueError("OmniGen2 stack sample requires `current_frame`, `input_image`, `video`, or `ref_image_latents`.")
            ref_image_latents = self._encode_omnigen2_image_latents(current_frame)
        if debug_forward:
            t0 = _sync_and_mark("build.ref_latent", t0)

        text_hidden_states, text_attention_mask = self._encode_omnigen2_text(sample)
        if self.proprio_encoder is not None:
            text_hidden_states, text_attention_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden_states,
                context_mask=text_attention_mask,
                proprio=sample.get("proprio"),
                source="OmniGen2 training sample",
            )
        if debug_forward:
            t0 = _sync_and_mark("build.text_to_device", t0)
        action = sample["action"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        action_dim_is_pad = sample.get("action_dim_is_pad")
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if debug_forward:
            _sync_and_mark("build.action_to_device", t0)
            self._last_omnigen2_build_profile = build_profile
        return {
            "target_latent": target_latent,
            "ref_image_latents": ref_image_latents,
            "text_hidden_states": text_hidden_states,
            "text_attention_mask": text_attention_mask,
            "action": action,
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
        }

    @torch.no_grad()
    def _encode_ovis_u1_image_tokens(
        self,
        image: torch.Tensor,
        *,
        time_value: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .ovis_u1_video_expert import OvisU1VideoExpert

        image = image.to(device=self.device, dtype=getattr(self.vae, "dtype", self.torch_dtype), non_blocking=True)
        latent = self.vae.encode(image).latent_dist.sample()
        if getattr(self.vae.config, "shift_factor", None) is not None:
            latent = latent - self.vae.config.shift_factor
        if getattr(self.vae.config, "scaling_factor", None) is not None:
            latent = latent * self.vae.config.scaling_factor
        latent = latent.to(dtype=self.torch_dtype)
        tokens = OvisU1VideoExpert.pack_latents(latent)
        _, _, latent_h, latent_w = latent.shape
        ids = OvisU1VideoExpert.build_img_ids(
            batch_size=int(latent.shape[0]),
            token_height=latent_h // 2,
            token_width=latent_w // 2,
            time_value=float(time_value),
            device=latent.device,
            dtype=latent.dtype,
        )
        return tokens, ids

    @torch.no_grad()
    def _decode_ovis_u1_image_tokens(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        latent_h = 2 * ((int(height) + 15) // 16)
        latent_w = 2 * ((int(width) + 15) // 16)
        latents = rearrange(
            tokens,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=latent_h // 2,
            w=latent_w // 2,
            ph=2,
            pw=2,
        )
        z = latents.to(device=self.device, dtype=getattr(self.vae, "dtype", self.torch_dtype))
        if getattr(self.vae.config, "scaling_factor", None) is not None:
            z = z / self.vae.config.scaling_factor
        if getattr(self.vae.config, "shift_factor", None) is not None:
            z = z + self.vae.config.shift_factor
        image = self.vae.decode(z, return_dict=False)[0]
        return image.detach().float().clamp(-1, 1)

    @torch.no_grad()
    def _prepare_ovis_u1_vlm_image_tensors(
        self,
        image: torch.Tensor,
        visual_tokenizer,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"`image` must be [B,3,H,W] or [3,H,W], got {tuple(image.shape)}")
        image = image.to(device=visual_tokenizer.device, dtype=torch.float32, non_blocking=True)
        if image.min() < 0:
            image = (image + 1.0) * 0.5
        image = image.clamp(0, 1)

        patch_size = int(visual_tokenizer.image_processor.patch_size)
        temporal_patch_size = int(visual_tokenizer.image_processor.temporal_patch_size)
        if temporal_patch_size != 1:
            raise ValueError("Ovis-U1 tensor online condition path currently expects temporal_patch_size=1.")
        hidden_stride = int(visual_tokenizer.image_processor.hidden_stride)
        min_pixels = int(visual_tokenizer.image_processor.min_pixels)
        max_pixels = int(visual_tokenizer.image_processor.max_pixels)
        _, _, height, width = image.shape
        resized_height, resized_width = visual_tokenizer.smart_resize(
            int(height),
            int(width),
            factor=patch_size * hidden_stride,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        image = F.interpolate(
            image,
            size=(int(resized_height), int(resized_width)),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0, 1)

        mean = torch.tensor(
            visual_tokenizer.image_processor.image_mean,
            device=image.device,
            dtype=image.dtype,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            visual_tokenizer.image_processor.image_std,
            device=image.device,
            dtype=image.dtype,
        ).view(1, 3, 1, 1)
        image = (image - mean) / std

        batch_size, channel, resized_height, resized_width = image.shape
        grid_h = resized_height // patch_size
        grid_w = resized_width // patch_size
        if grid_h % hidden_stride != 0 or grid_w % hidden_stride != 0:
            raise ValueError(
                "Ovis-U1 VLM resized grid must be divisible by hidden_stride, "
                f"got grid=({grid_h},{grid_w}) hidden_stride={hidden_stride}."
            )
        patches = image.reshape(
            batch_size,
            channel,
            grid_h // hidden_stride,
            hidden_stride,
            patch_size,
            grid_w // hidden_stride,
            hidden_stride,
            patch_size,
        )
        patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7).contiguous()
        flatten_patches = patches.reshape(batch_size * grid_h * grid_w, channel * patch_size * patch_size)
        grid_thws = torch.tensor(
            [[1, grid_h, grid_w]],
            device=image.device,
            dtype=torch.long,
        ).repeat(batch_size, 1)
        return flatten_patches, grid_thws, int(resized_height), int(resized_width)

    @torch.no_grad()
    def _encode_ovis_u1_condition_online(
        self,
        prompt: str | Sequence[str],
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        condition_encoder = getattr(self, "ovis_condition_encoder", None)
        if condition_encoder is None:
            raise ValueError(
                "Ovis-U1 online condition encoding requires `load_condition_encoder=true`. "
                "Otherwise provide precomputed `text_hidden_states`/`text_attention_mask`."
            )
        if image.ndim == 3:
            batch_size = 1
        elif image.ndim == 4:
            batch_size = int(image.shape[0])
        else:
            raise ValueError(f"`image` must be [B,3,H,W] or [3,H,W], got {tuple(image.shape)}")
        if isinstance(prompt, str):
            prompts = [prompt] * batch_size
        elif isinstance(prompt, Sequence):
            prompts = list(prompt)
            if len(prompts) != batch_size:
                raise ValueError(f"Expected {batch_size} prompts, got {len(prompts)}.")
        else:
            raise ValueError(f"`prompt` must be str or sequence of str, got {type(prompt)}")
        text_tokenizer = condition_encoder.get_text_tokenizer()
        visual_tokenizer = condition_encoder.get_visual_tokenizer()
        pixel_values, grid_thws, _, _ = self._prepare_ovis_u1_vlm_image_tensors(
            image,
            visual_tokenizer,
        )
        prefix_ids = text_tokenizer(
            "<|im_start|>user\n",
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(device=self.device)
        suffix_texts = ["\n" + str(item).strip() + "<|im_end|>\n" for item in prompts]
        suffix = text_tokenizer(
            suffix_texts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        model_device = getattr(condition_encoder, "device", next(condition_encoder.parameters()).device)
        prefix_ids = prefix_ids.to(device=model_device).expand(batch_size, -1)
        image_placeholders = torch.tensor(
            visual_tokenizer.construct_image_placeholders((1, 1)),
            device=model_device,
            dtype=torch.long,
        ).unsqueeze(0).expand(batch_size, -1)
        suffix_ids = suffix.input_ids.to(device=model_device)
        suffix_mask = suffix.attention_mask.to(device=model_device, dtype=torch.bool)
        input_ids = torch.cat([prefix_ids, image_placeholders, suffix_ids], dim=1)
        attention_mask = torch.cat(
            [
                torch.ones_like(prefix_ids, dtype=torch.bool),
                torch.ones_like(image_placeholders, dtype=torch.bool),
                suffix_mask,
            ],
            dim=1,
        )
        pixel_values = pixel_values.to(
            device=visual_tokenizer.device,
            dtype=getattr(visual_tokenizer, "dtype", self.torch_dtype),
        )
        grid_thws = grid_thws.to(device=visual_tokenizer.device)
        _, inputs_embeds, labels, merged_attention_mask, _ = condition_encoder.merge_multimodal(
            text_input_ids=input_ids,
            text_attention_masks=attention_mask,
            text_labels=None,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            left_padding=True,
        )
        inputs_embeds = inputs_embeds.detach()
        torch.cuda.empty_cache()
        llm = condition_encoder.get_llm()
        llm_device = llm.device
        outputs = llm(
            inputs_embeds=inputs_embeds.to(llm_device),
            labels=labels.to(llm_device),
            attention_mask=merged_attention_mask.to(llm_device),
            output_hidden_states=True,
        )
        text_hidden_states = torch.cat([outputs.hidden_states[-1], outputs.hidden_states[-2]], dim=-1).to(
            device=self.device,
            dtype=self.torch_dtype,
        )
        text_attention_mask = merged_attention_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        return text_hidden_states, text_attention_mask

    @torch.no_grad()
    def _encode_ovis_u1_text(self, sample, condition_image: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        cached_text_hidden_states = sample.get("text_hidden_states")
        if cached_text_hidden_states is not None:
            text_hidden_states = cached_text_hidden_states.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
            text_attention_mask = sample.get("text_attention_mask")
            if text_attention_mask is None:
                text_attention_mask = torch.ones(
                    text_hidden_states.shape[:2],
                    device=self.device,
                    dtype=torch.bool,
                )
            else:
                text_attention_mask = text_attention_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
            return text_hidden_states, text_attention_mask
        instruction = sample.get("instruction")
        if instruction is None:
            instruction = sample.get("prompt")
        if instruction is None:
            instruction = sample.get("task")
        if instruction is not None:
            if condition_image is None:
                raise ValueError("Ovis-U1 online condition encoding requires the condition/input image.")
            return self._encode_ovis_u1_condition_online(instruction, condition_image)
        raise ValueError(
            "Ovis-U1 training currently requires precomputed `text_hidden_states` "
            "from OvisU1.generate_condition(), or `instruction`/`prompt` with `load_condition_encoder=true`. "
            f"Available sample keys: {sorted(sample.keys())}"
        )

    @torch.no_grad()
    def _prepare_ovis_u1_infer_text(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            return self._encode_ovis_u1_condition_online(prompt, input_image)
        if context is None or context_mask is None:
            raise ValueError("`context` and `context_mask` must be both provided together.")
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        return (
            context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
        )

    def build_inputs_ovis_u1(self, sample, tiled: bool = False):
        video = sample.get("video")
        condition_image = None
        if "target_latent" in sample and "target_img_ids" in sample:
            target_tokens = sample["target_latent"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            target_img_ids = sample["target_img_ids"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        else:
            next_frame = sample.get("next_frame", sample.get("target_image"))
            if next_frame is None and video is not None:
                if video.ndim != 5:
                    raise ValueError(f"`sample['video']` must be [B,C,T,H,W], got {tuple(video.shape)}")
                next_frame = video[:, :, -1]
            if next_frame is None:
                raise ValueError("Ovis-U1 stack sample requires `next_frame`, `target_image`, or target token fields.")
            target_tokens, target_img_ids = self._encode_ovis_u1_image_tokens(next_frame, time_value=0.0)

        if "ref_image_latents" in sample and "ref_img_ids" in sample:
            ref_tokens = sample["ref_image_latents"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            ref_img_ids = sample["ref_img_ids"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        else:
            current_frame = sample.get("current_frame", sample.get("input_image"))
            if current_frame is None and video is not None:
                if video.ndim != 5:
                    raise ValueError(f"`sample['video']` must be [B,C,T,H,W], got {tuple(video.shape)}")
                current_frame = video[:, :, 0]
            if current_frame is None:
                raise ValueError("Ovis-U1 stack sample requires `current_frame`, `input_image`, or ref token fields.")
            condition_image = current_frame
            ref_tokens, ref_img_ids = self._encode_ovis_u1_image_tokens(current_frame, time_value=1.0)

        if condition_image is None:
            condition_image = sample.get("current_frame", sample.get("input_image"))
            if condition_image is None and video is not None:
                condition_image = video[:, :, 0]
        text_hidden_states, text_attention_mask = self._encode_ovis_u1_text(sample, condition_image=condition_image)
        if self.proprio_encoder is not None:
            text_hidden_states, text_attention_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden_states,
                context_mask=text_attention_mask,
                proprio=sample.get("proprio"),
                source="Ovis-U1 training sample",
            )
        action = sample["action"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        action_dim_is_pad = sample.get("action_dim_is_pad")
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        return {
            "target_latent": target_tokens,
            "target_img_ids": target_img_ids,
            "ref_image_latents": ref_tokens,
            "ref_img_ids": ref_img_ids,
            "text_hidden_states": text_hidden_states,
            "text_attention_mask": text_attention_mask,
            "action": action,
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
        }

    @torch.no_grad()
    def _encode_flux2_image_tokens(
        self,
        image: torch.Tensor,
        *,
        time_value: float,
        temporal_value: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .flux2_video_expert import Flux2VideoExpert

        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"`image` must be [B,3,H,W] or [3,H,W], got {tuple(image.shape)}")
        if image.shape[-2] % 16 != 0 or image.shape[-1] % 16 != 0:
            raise ValueError(f"FLUX.2 image spatial dims must be multiples of 16, got {tuple(image.shape[-2:])}")
        image = image.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        latents = self.vae.encode(image).to(dtype=self.torch_dtype)
        tokens = Flux2VideoExpert.pack_latents(latents)
        _, _, latent_h, latent_w = latents.shape
        ids = Flux2VideoExpert.build_img_ids(
            batch_size=int(latents.shape[0]),
            token_height=int(latent_h),
            token_width=int(latent_w),
            time_value=float(time_value),
            temporal_value=float(temporal_value),
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return tokens, ids

    @torch.no_grad()
    def _decode_flux2_image_tokens(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        from .flux2_video_expert import Flux2VideoExpert

        latent_h = int(height) // 16
        latent_w = int(width) // 16
        latents = Flux2VideoExpert.unpack_latents(tokens, latent_h, latent_w)
        image = self.vae.decode(latents.to(device=self.device, dtype=self.torch_dtype))
        return image.detach().float().clamp(-1, 1)

    @torch.no_grad()
    def _encode_flux2_text(self, sample) -> tuple[torch.Tensor, torch.Tensor]:
        cached_text_hidden_states = sample.get("text_hidden_states")
        if cached_text_hidden_states is not None:
            text_hidden_states = cached_text_hidden_states.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
            text_attention_mask = sample.get("text_attention_mask")
            if text_attention_mask is None:
                raise ValueError("FLUX.2 cached `text_hidden_states` must be paired with `text_attention_mask`.")
            else:
                text_attention_mask = text_attention_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
            return text_hidden_states, text_attention_mask

        prompt = sample.get("instruction", sample.get("prompt", sample.get("task")))
        if prompt is None:
            raise ValueError(
                "FLUX.2 stack requires precomputed `text_hidden_states` or an `instruction`/`prompt`/`task`."
            )
        if self.text_encoder is None:
            raise ValueError("FLUX.2 online text encoding requires `load_text_encoder=true`.")
        video = sample.get("video")
        batch_size = int(video.shape[0]) if isinstance(video, torch.Tensor) and video.ndim == 5 else 1
        prompts = [prompt] * batch_size if isinstance(prompt, str) else list(prompt)
        return self._encode_flux2_prompts(prompts)

    @torch.no_grad()
    def _encode_flux2_prompts(self, prompts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_encoder is None:
            self._load_flux2_text_encoder_for_inference()
        if not hasattr(self.text_encoder, "tokenizer") or not hasattr(self.text_encoder, "model"):
            text_hidden = self.text_encoder(list(prompts)).to(device=self.device, dtype=self.torch_dtype)
            text_mask = torch.ones(text_hidden.shape[:2], device=self.device, dtype=torch.bool)
            return text_hidden, text_mask

        ensure_flux2_importable_path = getattr(self, "model_paths", {}).get("flux2_src")
        from .flux2_imports import ensure_flux2_importable

        ensure_flux2_importable(ensure_flux2_importable_path)
        from flux2.text_encoder import OUTPUT_LAYERS_QWEN3

        tokenizer = self.text_encoder.tokenizer
        model = self.text_encoder.model
        max_length = int(getattr(self.text_encoder, "max_length", 512))
        all_input_ids = []
        all_attention_masks = []
        for prompt in prompts:
            messages = [{"role": "user", "content": str(prompt)}]
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            model_inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            all_input_ids.append(model_inputs["input_ids"])
            all_attention_masks.append(model_inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(model.device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(model.device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = torch.stack([outputs.hidden_states[k] for k in OUTPUT_LAYERS_QWEN3], dim=1)
        hidden = rearrange(hidden, "b c l d -> b l (c d)")
        return (
            hidden.to(device=self.device, dtype=self.torch_dtype),
            attention_mask.to(device=self.device, dtype=torch.bool),
        )

    def _load_flux2_text_encoder_for_inference(self) -> None:
        from types import SimpleNamespace
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_spec = getattr(self, "flux2_qwen3_model_spec", None)
        if model_spec is None:
            model_spec = getattr(self, "model_paths", {}).get("qwen3")
        if model_spec is None:
            if int(self.text_dim) == 7680:
                model_spec = "Qwen/Qwen3-4B"
            elif int(self.text_dim) == 12288:
                model_spec = "Qwen/Qwen3-8B"
            else:
                raise ValueError(f"Cannot infer FLUX.2 Qwen3 model spec from text_dim={self.text_dim}")
        logger.info("Lazy-loading FLUX.2 Qwen3 text encoder for prompt inference: %s", model_spec)
        qwen3_model = AutoModelForCausalLM.from_pretrained(
            model_spec,
            torch_dtype=self.torch_dtype,
        ).to(self.device).eval()
        self.text_encoder = SimpleNamespace(
            model=qwen3_model,
            tokenizer=AutoTokenizer.from_pretrained(model_spec),
            max_length=int(getattr(self, "qwen_context_len", 512)),
        )

    @torch.no_grad()
    def _prepare_flux2_infer_text(
        self,
        prompt: Optional[str],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            return self._encode_flux2_prompts([prompt])
        if context is None or context_mask is None:
            raise ValueError("`context` and `context_mask` must be both provided together.")
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        return (
            context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
        )

    def build_inputs_flux2(self, sample, tiled: bool = False):
        del tiled
        video = sample.get("video")
        if "target_latent" in sample and "target_img_ids" in sample:
            target_tokens = sample["target_latent"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            target_img_ids = sample["target_img_ids"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        else:
            next_frame = sample.get("next_frame", sample.get("target_image"))
            if next_frame is None and video is not None:
                if video.ndim != 5:
                    raise ValueError(f"`sample['video']` must be [B,C,T,H,W], got {tuple(video.shape)}")
                next_frame = video[:, :, -1]
            if next_frame is None:
                raise ValueError("FLUX.2 stack sample requires `next_frame`, `target_image`, or target token fields.")
            target_tokens, target_img_ids = self._encode_flux2_image_tokens(next_frame, time_value=0.0)

        if "ref_image_latents" in sample and "ref_img_ids" in sample:
            ref_tokens = sample["ref_image_latents"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            ref_img_ids = sample["ref_img_ids"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        else:
            current_frame = sample.get("current_frame", sample.get("input_image"))
            if current_frame is None and video is not None:
                if video.ndim != 5:
                    raise ValueError(f"`sample['video']` must be [B,C,T,H,W], got {tuple(video.shape)}")
                current_frame = video[:, :, 0]
            if current_frame is None:
                raise ValueError("FLUX.2 stack sample requires `current_frame`, `input_image`, or ref token fields.")
            ref_tokens, ref_img_ids = self._encode_flux2_image_tokens(current_frame, time_value=10.0)
        if getattr(self, "edit_memory_enabled", False) and self.training:
            ref_tokens = self._augment_memory_clean_conditioning(ref_tokens)

        text_hidden_states, text_attention_mask = self._encode_flux2_text(sample)
        if self.proprio_encoder is not None:
            text_hidden_states, text_attention_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden_states,
                context_mask=text_attention_mask,
                proprio=sample.get("proprio"),
                source="FLUX.2 training sample",
            )
        action = sample["action"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        action_dim_is_pad = sample.get("action_dim_is_pad")
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        return {
            "target_latent": target_tokens,
            "target_img_ids": target_img_ids,
            "ref_image_latents": ref_tokens,
            "ref_img_ids": ref_img_ids,
            "text_hidden_states": text_hidden_states,
            "text_attention_mask": text_attention_mask,
            "action": action,
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
        }

    def build_inputs_dim(self, sample, tiled: bool = False):
        del tiled
        video = sample.get("video")
        if video is None or video.ndim != 5:
            raise ValueError(f"DIM stack sample requires `video` [B,3,T,H,W], got {None if video is None else tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"DIM stack expects RGB video, got {tuple(video.shape)}")
        source_image = sample.get("current_frame", sample.get("input_image"))
        target_image = sample.get("next_frame", sample.get("target_image"))
        if source_image is None:
            source_image = video[:, :, 0]
        if target_image is None:
            target_image = video[:, :, -1]
        if source_image.shape[-2] % int(self.video_expert.vae_downsample_rate) != 0 or source_image.shape[-1] % int(self.video_expert.vae_downsample_rate) != 0:
            raise ValueError(
                "DIM image spatial dims must be multiples of "
                f"{self.video_expert.vae_downsample_rate}, got {tuple(source_image.shape[-2:])}."
            )
        source_image = source_image.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        target_image = target_image.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        source_latent = self._encode_dim_image_latents(source_image)
        target_latent = self._encode_dim_image_latents(target_image)

        cached_dim_condition = sample.get("dim_condition_hidden_states")
        cached_dim_mask = sample.get("dim_condition_attention_mask")
        cached_text = sample.get("text_hidden_states")
        cached_text_mask = sample.get("text_attention_mask")
        if cached_dim_condition is not None and cached_dim_mask is not None:
            condition, condition_mask = (
                cached_dim_condition.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
                cached_dim_mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
            )
        elif cached_text is not None and cached_text_mask is not None:
            condition, condition_mask = (
                cached_text.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
                cached_text_mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
            )
        else:
            instruction = sample.get("instruction", sample.get("prompt"))
            if instruction is None:
                raise ValueError("DIM stack requires `instruction`/`prompt` for online Qwen image+text conditioning.")
            condition, condition_mask = self._encode_dim_condition_online(instruction, source_image)
        if self.proprio_encoder is not None:
            condition, condition_mask = self._append_proprio_to_context_if_enabled(
                context=condition,
                context_mask=condition_mask,
                proprio=sample.get("proprio"),
                source="DIM training sample",
            )

        action = sample["action"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        action_dim_is_pad = sample.get("action_dim_is_pad")
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        return {
            "target_latent": target_latent,
            "source_latent": source_latent,
            "condition": condition,
            "condition_mask": condition_mask,
            "action": action,
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> first-frame video only
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    @torch.no_grad()
    def _build_mot_attention_mask_omnigen2(
        self,
        encoder_seq_lengths: list[int],
        seq_lengths: list[int],
        max_video_seq_len: int,
        action_seq_len: int,
        device: torch.device,
        l_effective_ref_img_len: list[list[int]],
    ) -> torch.Tensor:
        batch_size = len(seq_lengths)
        total_seq_len = int(max_video_seq_len) + int(action_seq_len)
        action_start = int(max_video_seq_len)
        mask = torch.zeros(batch_size, total_seq_len, total_seq_len, dtype=torch.bool, device=device)
        for i, (cap_len, video_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
            prefix_len = cap_len + sum(l_effective_ref_img_len[i])

            # video -> video: text/ref prefix cannot attend noisy target, matching
            # wan22 first_frame_causal where first-frame tokens cannot see future tokens.
            mask[i, :prefix_len, :prefix_len] = True
            mask[i, prefix_len:video_len, :video_len] = True
            # action -> action
            mask[i, action_start:, action_start:] = True
            # action -> text/ref prefix only. Text is included here because
            # OmniGen2-style ActionDiT has no separate text cross-attention.
            mask[i, action_start:, :prefix_len] = True
        return mask

    @torch.no_grad()
    def _build_mot_attention_mask_ovis_u1(
        self,
        batch_size: int,
        txt_len: int,
        target_len: int,
        cond_len: int,
        action_len: int,
        device: torch.device,
        text_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        t0 = 0
        x0 = txt_len
        c0 = txt_len + target_len
        a0 = txt_len + target_len + cond_len
        total = a0 + action_len
        joint = torch.zeros(batch_size, total, total, dtype=torch.bool, device=device)
        # Text/condition are stable prefix states and do not attend noisy target.
        joint[:, t0:x0, t0:x0] = True
        joint[:, t0:x0, c0:a0] = True
        joint[:, c0:a0, t0:x0] = True
        joint[:, c0:a0, c0:a0] = True
        # Target/noisy image can use text, condition, and target image tokens.
        joint[:, x0:c0, t0:a0] = True
        # Action uses text, condition, and action self context, but not target/noisy image.
        joint[:, a0:total, t0:x0] = True
        joint[:, a0:total, c0:total] = True
        if text_attention_mask is not None:
            if text_attention_mask.ndim != 2 or tuple(text_attention_mask.shape) != (batch_size, txt_len):
                raise ValueError(
                    "`text_attention_mask` must be [B,txt_len], "
                    f"got {tuple(text_attention_mask.shape)} for B={batch_size}, txt_len={txt_len}"
                )
            text_valid = text_attention_mask.to(device=device, dtype=torch.bool)
            joint[:, :, t0:x0] &= text_valid[:, None, :]

        image_total = target_len + cond_len + action_len
        image_action_start = target_len + cond_len
        image = torch.zeros(batch_size, image_total, image_total, dtype=torch.bool, device=device)
        image[:, :target_len, : target_len + cond_len] = True
        image[:, target_len:image_action_start, target_len:image_action_start] = True
        image[:, image_action_start:, target_len:image_total] = True
        return {"joint_double": joint, "image_double": image, "single": joint.clone()}

    @torch.no_grad()
    def _build_mot_attention_mask_flux2(
        self,
        batch_size: int,
        txt_len: int,
        target_len: int,
        cond_len: int,
        action_len: int,
        device: torch.device,
        text_attention_mask: torch.Tensor | None = None,
        gist_len: int = 0,
    ) -> dict[str, torch.Tensor]:
        t0 = 0
        r0 = txt_len
        g0 = txt_len + cond_len
        x0 = g0 + int(gist_len)
        a0 = x0 + target_len
        total = a0 + action_len
        mask = torch.zeros(batch_size, total, total, dtype=torch.bool, device=device)
        # Stable text/ref/gist prefix cannot attend to noisy target/action tokens.
        mask[:, t0:r0, t0:x0] = True
        mask[:, r0:g0, t0:x0] = True
        mask[:, g0:x0, t0:x0] = True
        # Target/noisy image uses stable prefix and target self-context.
        mask[:, x0:a0, t0:a0] = True
        # Action uses stable prefix and action self-context, not target/noisy image.
        mask[:, a0:total, t0:x0] = True
        mask[:, a0:total, a0:total] = True
        if text_attention_mask is not None:
            if text_attention_mask.ndim != 2 or tuple(text_attention_mask.shape) != (batch_size, txt_len):
                raise ValueError(
                    "`text_attention_mask` must be [B,txt_len], "
                    f"got {tuple(text_attention_mask.shape)} for B={batch_size}, txt_len={txt_len}"
                )
            text_valid = text_attention_mask.to(device=device, dtype=torch.bool)
            mask[:, :, t0:r0] &= text_valid[:, None, :]
        return {"double_joint": mask, "single": mask.clone()}

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def _compute_action_loss_per_sample(
        self,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        action_dim_is_pad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        action_loss_dim = F.mse_loss(pred_action.float(), target_action.float(), reduction="none")
        if action_dim_is_pad is not None:
            dim_valid = (~action_dim_is_pad).to(device=action_loss_dim.device, dtype=action_loss_dim.dtype)
            dim_valid_sum = dim_valid.sum(dim=1).clamp(min=1.0).unsqueeze(1)
            action_loss_token = (action_loss_dim * dim_valid.unsqueeze(1)).sum(dim=2) / dim_valid_sum
        else:
            action_loss_token = action_loss_dim.mean(dim=2)

        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            return (action_loss_token * valid).sum(dim=1) / valid_sum
        return action_loss_token.mean(dim=1)

    def _training_loss_omnigen2(self, sample, tiled: bool = False):
        debug_every = int(os.environ.get("IMAGEWAM_DEBUG_OMNIGEN2_FORWARD_EVERY", "0") or "0")
        debug_threshold = float(os.environ.get("IMAGEWAM_DEBUG_OMNIGEN2_FORWARD_THRESHOLD", "0") or "0")
        debug_forward = debug_every > 0 or debug_threshold > 0
        debug_step = int(getattr(self, "_omnigen2_forward_debug_step", 0)) + 1
        self._omnigen2_forward_debug_step = debug_step
        debug_segments: list[tuple[str, float]] = []
        debug_meta: dict[str, Any] = {}

        def _sync_and_mark(name: str, start_time: float) -> float:
            if debug_forward and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            now = time.perf_counter()
            if debug_forward:
                debug_segments.append((name, now - start_time))
            return now

        if debug_forward and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        inputs = self.build_inputs_omnigen2(sample, tiled=tiled)
        t0 = _sync_and_mark("build_inputs", t0)
        target_latent = inputs["target_latent"]
        action = inputs["action"]
        batch_size = target_latent.shape[0]

        noise_video = torch.randn_like(target_latent)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=target_latent.dtype,
        )
        noisy_latent = self.train_video_scheduler.add_noise(target_latent, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(target_latent, noise_video, timestep_video)

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)
        t0 = _sync_and_mark("noise_sched", t0)

        video_pre = self.video_expert.pre_dit(
            x=noisy_latent,
            timestep=timestep_video,
            context=inputs["text_hidden_states"],
            context_mask=inputs["text_attention_mask"],
            ref_image_hidden_states=inputs["ref_image_latents"],
        )
        t0 = _sync_and_mark("video_pre", t0)
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
        )
        t0 = _sync_and_mark("action_pre", t0)

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]
        attention_mask = self._build_mot_attention_mask_omnigen2(
            encoder_seq_lengths=video_pre["encoder_seq_lengths"],
            seq_lengths=video_pre["seq_lengths"],
            max_video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            l_effective_ref_img_len=video_pre.get("l_effective_ref_img_len"),
            device=video_tokens.device,
        )
        if debug_forward:
            debug_meta = {
                "B": int(batch_size),
                "video_seq": int(video_tokens.shape[1]),
                "action_seq": int(action_tokens.shape[1]),
                "cap_min": int(min(video_pre["encoder_seq_lengths"])),
                "cap_max": int(max(video_pre["encoder_seq_lengths"])),
                "seq_min": int(min(video_pre["seq_lengths"])),
                "seq_max": int(max(video_pre["seq_lengths"])),
                "target_shape": tuple(target_latent.shape),
                "action_shape": tuple(action.shape),
            }
        t0 = _sync_and_mark("mask", t0)
        tokens_out = self.mot(
            embeds_all={"video": video_tokens, "action": action_tokens},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={"video": None, "action": None},
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        t0 = _sync_and_mark("mot", t0)

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        video_loss_per_sample = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 2, 3))
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            video_loss_per_sample.device, dtype=video_loss_per_sample.dtype
        )
        loss_video = (video_loss_per_sample * video_weight).mean()

        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=inputs["action_is_pad"],
            action_dim_is_pad=inputs.get("action_dim_is_pad"),
        )
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        t0 = _sync_and_mark("post_loss", t0)
        if debug_forward:
            total_debug = sum(value for _, value in debug_segments)
            should_log = (debug_every > 0 and debug_step % debug_every == 0) or (
                debug_threshold > 0 and total_debug >= debug_threshold
            )
            phase_names = ["build_inputs", "noise_sched", "video_pre", "action_pre", "mask", "mot", "post_loss"]
            segment_dict = dict(debug_segments)
            durations = [float(segment_dict.get(name, 0.0)) for name in phase_names]
            arrivals = []
            cumulative = 0.0
            for value in durations:
                cumulative += value
                arrivals.append(cumulative)
            peak_gib = (
                torch.cuda.max_memory_allocated(self.device) / (1024**3)
                if self.device.type == "cuda"
                else 0.0
            )
            meta_values = [
                float(debug_step),
                1.0 if should_log else 0.0,
                float(debug_meta.get("B", 0)),
                float(debug_meta.get("video_seq", 0)),
                float(debug_meta.get("action_seq", 0)),
                float(debug_meta.get("cap_min", 0)),
                float(debug_meta.get("cap_max", 0)),
                float(debug_meta.get("seq_min", 0)),
                float(debug_meta.get("seq_max", 0)),
                float(peak_gib),
            ]
            profile_values = durations + arrivals + meta_values
            self._last_omnigen2_forward_profile = {
                "phase_names": phase_names,
                "num_phases": len(phase_names),
            }
            self._last_omnigen2_forward_profile_tensor = torch.tensor(
                profile_values,
                device=loss_total.device,
                dtype=torch.float32,
            )
        return loss_total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }

    def _training_loss_ovis_u1(self, sample, tiled: bool = False):
        inputs = self.build_inputs_ovis_u1(sample, tiled=tiled)
        target_latent = inputs["target_latent"]
        action = inputs["action"]
        batch_size = int(target_latent.shape[0])

        noise_video = torch.randn_like(target_latent)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=target_latent.dtype,
        )
        noisy_latent = self.train_video_scheduler.add_noise(target_latent, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(target_latent, noise_video, timestep_video)

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=noisy_latent,
            timestep=self._scheduler_timestep_to_unit(timestep_video, self.train_video_scheduler),
            context=inputs["text_hidden_states"],
            context_mask=inputs["text_attention_mask"],
            ref_image_hidden_states=inputs["ref_image_latents"],
            target_img_ids=inputs["target_img_ids"],
            ref_img_ids=inputs["ref_img_ids"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=self._scheduler_timestep_to_unit(timestep_action, self.train_action_scheduler),
        )
        attention_mask = self._build_mot_attention_mask_ovis_u1(
            batch_size=batch_size,
            txt_len=int(video_pre["txt_len"]),
            target_len=int(video_pre["target_len"]),
            cond_len=int(video_pre["cond_len"]),
            action_len=int(action_pre["tokens"].shape[1]),
            device=noisy_latent.device,
            text_attention_mask=video_pre["text_mask"],
        )
        tokens_out = self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"]},
            context_all={"video": None, "action": {"ids": action_pre["ids"]}},
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        video_loss_per_sample = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").flatten(1).mean(dim=1)
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            video_loss_per_sample.device,
            dtype=video_loss_per_sample.dtype,
        )
        loss_video = (video_loss_per_sample * video_weight).mean()
        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=inputs["action_is_pad"],
            action_dim_is_pad=inputs.get("action_dim_is_pad"),
        )
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device,
            dtype=action_loss_per_sample.dtype,
        )
        loss_action = (action_loss_per_sample * action_weight).mean()
        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        return loss_total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }

    def _edit_memory_episode_tbptt_step(
        self,
        sample: dict[str, Any],
        *,
        bank: EditKVMemoryBank,
        valid: torch.Tensor,
        step_idx: int,
        total_valid_steps: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Run one chronological replan with the same bank read/write path as inference."""
        video = sample["video"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action = sample["action"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        valid = valid.to(device=self.device, dtype=torch.bool)
        batch_size = int(video.shape[0])
        if batch_size != 1:
            raise ValueError("KV8 episode-TBPTT currently requires per-rank batch_size=1.")

        proprio = sample.get("proprio")
        if proprio is not None:
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        action_dim_is_pad = sample.get("action_dim_is_pad")
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )

        text_hidden, text_mask = self._encode_flux2_text(sample)
        if self.proprio_encoder is not None or proprio is not None:
            text_hidden, text_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden,
                context_mask=text_mask,
                proprio=proprio,
                source="FLUX.2 KV8 episode-TBPTT training sample",
            )
        with torch.no_grad():
            ref_tokens, ref_img_ids = self._encode_flux2_image_tokens(
                video[:, :, 0],
                time_value=10.0,
                temporal_value=float(step_idx),
            )
        ref_tokens = self._augment_memory_clean_conditioning(ref_tokens)

        frame_summary = self._edit_memory_frame_summary(ref_tokens, text_hidden, text_mask)
        span_decision = self._edit_memory_span_decision(frame_summary)
        memory_cache = bank.read(history_span=span_decision, compact_dynamic=True)
        gist_tokens, gist_ids = self._build_flux2_gist_inputs(
            batch_size=batch_size,
            ref_img_ids=ref_img_ids,
            device=ref_tokens.device,
            token_dtype=ref_tokens.dtype,
            step_idx=int(step_idx),
        )
        clean_prefix, clean_prefix_cache = self._prefill_flux2_memory_prefix(
            ref_tokens=ref_tokens,
            ref_img_ids=ref_img_ids,
            text_hidden=text_hidden,
            text_mask=text_mask,
            gist_tokens=gist_tokens,
            gist_ids=gist_ids,
            memory_kv_cache=memory_cache,
        )
        predicted_gist_decision = None
        write_decision = None
        if (
            self.edit_memory_gist_allocator is not None
            or self.edit_memory_write_head is not None
        ):
            prefix_summary = self._edit_memory_prefix_summary(clean_prefix_cache)
            predicted_gist_decision = self._edit_memory_gist_decision(prefix_summary)
            write_decision = self._edit_memory_write_confidence(
                prefix_summary, predicted_gist_decision
            )
            if predicted_gist_decision is not None:
                predicted_gist_decision["write_decision"] = write_decision

        episode_ids = sample.get("edit_memory_episode_id")
        timesteps = sample.get("edit_memory_timestep")
        if episode_ids is not None:
            episode_ids = episode_ids.to(device=self.device, dtype=torch.long, non_blocking=True)
        if timesteps is not None:
            timesteps = timesteps.to(device=self.device, dtype=torch.long, non_blocking=True)
        vlm_metrics = self._edit_memory_vlm_count_supervision(
            predicted_gist_decision,
            episode_ids,
            timesteps,
            valid,
            write_decision=write_decision,
        )
        curriculum_random = None
        if episode_ids is not None and timesteps is not None:
            hashed = (
                episode_ids.reshape(-1) * 1_000_003
                + timesteps.reshape(-1) * 9_176
                + int(getattr(self, "_edit_memory_training_step", 0)) * 37
            )
            curriculum_random = torch.remainder(hashed, 1_000_033).float() / 1_000_033.0
        write_gist_decision, curriculum_metrics = self._edit_memory_write_decision(
            predicted_gist_decision,
            vlm_metrics,
            random_values=curriculum_random,
            write_decision=write_decision,
        )

        target_tokens, target_img_ids = self._encode_flux2_image_tokens(
            video[:, :, -1],
            time_value=0.0,
            temporal_value=float(step_idx + 1),
        )
        noise_video = torch.randn_like(target_tokens)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=target_tokens.dtype,
        )
        noisy_latent = self.train_video_scheduler.add_noise(
            target_tokens, noise_video, timestep_video
        )
        target_video = self.train_video_scheduler.training_target(
            target_tokens, noise_video, timestep_video
        )
        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(
            action, noise_action, timestep_action
        )
        target_action = self.train_action_scheduler.training_target(
            action, noise_action, timestep_action
        )

        video_pre = self.video_expert.pre_dit(
            x=noisy_latent,
            timestep=self._scheduler_timestep_to_unit(
                timestep_video, self.train_video_scheduler
            ),
            context=text_hidden,
            context_mask=text_mask,
            ref_image_hidden_states=ref_tokens,
            gist_hidden_states=gist_tokens,
            target_img_ids=target_img_ids,
            ref_img_ids=ref_img_ids,
            gist_img_ids=gist_ids,
        )
        prefix_attention_mask = self._build_mot_attention_mask_flux2(
            batch_size=batch_size,
            txt_len=int(video_pre["txt_len"]),
            target_len=int(video_pre["target_len"]),
            cond_len=int(video_pre["cond_len"]),
            action_len=0,
            device=target_tokens.device,
            text_attention_mask=video_pre["text_mask"],
            gist_len=int(video_pre.get("gist_len", 0)),
        )
        video_kv_cache = self.mot.prefill_flux2_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            attention_mask=prefix_attention_mask,
            memory_kv_cache=memory_cache,
        )
        pred_video = self.video_expert.post_dit(
            video_kv_cache["final_video"], video_pre
        )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=self._scheduler_timestep_to_unit(
                timestep_action, self.train_action_scheduler
            ),
            temporal_value=float(step_idx),
        )
        action_len = int(action_pre["tokens"].shape[1])
        action_attention_mask = self._build_mot_attention_mask_flux2(
            batch_size=batch_size,
            txt_len=int(clean_prefix["txt_len"]),
            target_len=0,
            cond_len=int(clean_prefix["cond_len"]),
            action_len=action_len,
            device=target_tokens.device,
            text_attention_mask=clean_prefix["text_mask"],
            gist_len=int(clean_prefix.get("gist_len", 0)),
        )
        clean_prefix_len = (
            int(clean_prefix["txt_len"])
            + int(clean_prefix["cond_len"])
            + int(clean_prefix.get("gist_len", 0))
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=None,
            action_t_mod=action_pre["t_mod"],
            action_context_payload={"ids": action_pre["ids"]},
            video_kv_cache=clean_prefix_cache,
            attention_mask=action_attention_mask,
            video_seq_len=clean_prefix_len,
            memory_kv_cache=memory_cache,
        )
        pred_action = self.action_expert.post_dit(action_tokens, action_pre)

        memory_item = self._select_flux2_edit_memory_item(
            video_kv_cache=clean_prefix_cache,
            per_layer_teacher_importance=None,
            txt_len=int(clean_prefix["txt_len"]),
            img_len=int(clean_prefix["cond_len"]),
            grid_hw=self._grid_hw_from_flux2_img_ids(ref_img_ids),
            step_idx=int(step_idx),
            gist_len=int(clean_prefix.get("gist_len", 0)),
            gist_decision=write_gist_decision,
            write_confidence=curriculum_metrics["write_confidence"],
            compact_adaptive_gist=True,
        )
        hard_write = curriculum_metrics["hard_write"]
        gist_valid = valid if hard_write is None else valid & hard_write
        memory_item = self._attach_edit_memory_valid_mask(
            memory_item,
            valid,
            gist_valid=gist_valid,
        )
        write_confidence = curriculum_metrics["write_confidence"]
        if write_confidence is None:
            write_confidence = memory_item["confidence"]
        write_happened = False
        if bool(valid.any().item()):
            write_happened = bank.maybe_write(
                memory_item,
                step_idx=int(step_idx),
                confidence=write_confidence,
                gist_write_selected=gist_valid,
            )

        video_loss_per_sample = F.mse_loss(
            pred_video.float(), target_video.float(), reduction="none"
        ).flatten(1).mean(dim=1)
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            video_loss_per_sample.device,
            dtype=video_loss_per_sample.dtype,
        )
        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=action_is_pad,
            action_dim_is_pad=action_dim_is_pad,
        )
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device,
            dtype=action_loss_per_sample.dtype,
        )
        valid_float = valid.to(dtype=video_loss_per_sample.dtype)
        valid_denom = valid_float.sum().clamp_min(1.0)
        loss_video = (video_loss_per_sample * video_weight * valid_float).sum() / valid_denom
        loss_action = (action_loss_per_sample * action_weight * valid_float).sum() / valid_denom
        zero = loss_video.new_zeros(())
        loss_regularizer = zero
        if span_decision is not None:
            loss_regularizer = loss_regularizer + span_decision["regularizer"].float()
        if predicted_gist_decision is not None:
            loss_regularizer = loss_regularizer + predicted_gist_decision["regularizer"].float()
        loss_vlm = zero if vlm_metrics is None else vlm_metrics["loss"]
        valid_fraction = valid.float().sum() / total_valid_steps.clamp_min(1.0)
        loss_total = valid_fraction * (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_action * loss_action
            + loss_regularizer
            + loss_vlm
        )

        weight = float(valid_fraction.detach().item())
        metrics = {
            "loss_video": weight * self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": weight * self.loss_lambda_action * float(loss_action.detach().item()),
            "loss_edit_memory_action": weight
            * self.loss_lambda_action
            * float(loss_action.detach().item()),
            "loss_edit_memory_regularizer": weight * float(loss_regularizer.detach().item()),
            "loss_edit_memory_vlm_count": weight * float(loss_vlm.detach().item()),
            "edit_memory_curriculum_teacher_probability": weight
            * float(curriculum_metrics["teacher_probability"]),
            "edit_memory_curriculum_teacher_frac": weight
            * float(curriculum_metrics["teacher_mask"].float().mean().item()),
            "edit_memory_write_confidence": weight
            * float(bank.last_write_confidence),
            "edit_memory_write_happened": weight * float(write_happened),
            "edit_memory_adaptive_gist_tokens": weight
            * (
                float(
                    getattr(self, "edit_memory_config", {})
                    .get("bank", {})
                    .get("gist_tokens_per_frame", 8)
                )
                if predicted_gist_decision is None
                else float(
                    predicted_gist_decision["expected_count"]
                    .detach()
                    .float()
                    .mean()
                    .item()
                )
            ),
        }
        if vlm_metrics is not None:
            metrics.update(
                {
                    "loss_edit_memory_vlm_soft_label": weight
                    * float(vlm_metrics["count_loss"].detach().item()),
                    "edit_memory_vlm_agreement": weight
                    * float(vlm_metrics["agreement"].detach().item()),
                    "edit_memory_vlm_expected_count_mae": weight
                    * float(vlm_metrics["expected_mae"].detach().item()),
                    "edit_memory_vlm_coverage": weight
                    * float(vlm_metrics["coverage"].detach().item()),
                    "edit_memory_vlm_zero_mapped_frac": weight
                    * float(vlm_metrics["zero_mapped_frac"].detach().item()),
                }
            )
        return loss_total, metrics

    def _edit_memory_new_training_bank(self) -> EditKVMemoryBank:
        bank_cfg = getattr(self, "edit_memory_config", {}).get("bank", {})
        bank = EditKVMemoryBank(
            max_entries=int(bank_cfg.get("max_entries", 0)),
            write_threshold=float(bank_cfg.get("write_threshold", 0.5)),
            cooldown_steps=int(bank_cfg.get("cooldown_steps", 0)),
            consolidation=str(bank_cfg.get("consolidation", "none")),
            merge_until_under_limit=bool(bank_cfg.get("merge_until_under_limit", True)),
            anchor_frames=int(bank_cfg.get("anchor_frames", 2)),
            recent_frames=self._edit_memory_historical_recent_frames(bank_cfg),
            gist_tokens_per_frame=int(bank_cfg.get("gist_tokens_per_frame", 8)),
            detach_on_write=False,
        )
        bank.reset(batch_size=1, device=self.device)
        return bank

    def _edit_memory_eatb_stores(self) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
        host_bank = getattr(self, "_edit_memory_eatb_host_bank", None)
        if host_bank is None:
            host_bank = {}
            self._edit_memory_eatb_host_bank = host_bank
        pending = getattr(self, "_edit_memory_eatb_pending", None)
        if pending is None:
            pending = []
            self._edit_memory_eatb_pending = pending
        return host_bank, pending

    def _edit_memory_eatb_load_bank(self, episode_id: int, start_chunk_id: int) -> EditKVMemoryBank:
        host_bank, _ = self._edit_memory_eatb_stores()
        bank = self._edit_memory_new_training_bank()
        if episode_id < 0:
            return bank
        if start_chunk_id == 0:
            host_bank.pop(int(episode_id), None)
            return bank
        state = host_bank.get(int(episode_id))
        if state is None:
            raise KeyError(
                f"Missing EATB bank for episode_id={episode_id}, start_chunk_id={start_chunk_id}; "
                "temporal windows must be consumed in order."
            )
        expected_next = int(state["next_chunk_id"])
        if expected_next != int(start_chunk_id):
            raise ValueError(
                f"EATB bank cursor mismatch for episode_id={episode_id}: "
                f"stored next_chunk_id={expected_next}, requested={start_chunk_id}."
            )
        host_bank.pop(int(episode_id))
        bank.load_state_dict(state["bank"], device=self.device)
        return bank

    def commit_edit_memory_eatb_state(self) -> None:
        """Write detached final banks after the corresponding optimizer update."""
        host_bank, pending = self._edit_memory_eatb_stores()
        for item in pending:
            episode_id = int(item["episode_id"])
            if episode_id < 0:
                continue
            if bool(item["is_last_window"]):
                host_bank.pop(episode_id, None)
                continue
            bank = item["bank"]
            bank.detach_graph()
            host_bank[episode_id] = {
                "next_chunk_id": int(item["next_chunk_id"]),
                "bank": bank.state_dict(device="cpu"),
            }
        pending.clear()

    def reset_edit_memory_eatb_state(self) -> None:
        host_bank, pending = self._edit_memory_eatb_stores()
        host_bank.clear()
        pending.clear()

    def edit_memory_eatb_state_dict(self) -> dict[str, Any]:
        host_bank, pending = self._edit_memory_eatb_stores()
        if pending:
            raise RuntimeError("Cannot checkpoint EATB state before the optimizer-step commit.")
        return {
            "version": 1,
            "episodes": {
                int(episode_id): {
                    "next_chunk_id": int(state["next_chunk_id"]),
                    "bank": state["bank"],
                }
                for episode_id, state in host_bank.items()
            },
        }

    def load_edit_memory_eatb_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("version", 0)) != 1:
            raise ValueError(f"Unsupported EATB state version: {state.get('version')!r}.")
        host_bank, pending = self._edit_memory_eatb_stores()
        pending.clear()
        host_bank.clear()
        for episode_id, item in state.get("episodes", {}).items():
            host_bank[int(episode_id)] = {
                "next_chunk_id": int(item["next_chunk_id"]),
                "bank": item["bank"],
            }

    @staticmethod
    def _edit_memory_slice_eatb_step(
        sample: dict[str, Any],
        *,
        sample_idx: int,
        chunk_idx: int,
        batch_size: int,
        num_chunks: int,
    ) -> dict[str, Any]:
        one = ImageWAM._edit_memory_slice_episode_batch(sample, sample_idx, batch_size)
        temporal_keys = {
            "video",
            "action",
            "proprio",
            "image_is_pad",
            "action_is_pad",
            "proprio_is_pad",
            "action_dim_is_pad",
            "proprio_dim_is_pad",
            "edit_memory_episode_id",
            "edit_memory_timestep",
        }
        for key in temporal_keys:
            value = one.get(key)
            if (
                isinstance(value, torch.Tensor)
                and value.ndim >= 2
                and int(value.shape[1]) == int(num_chunks)
            ):
                one[key] = value[:, chunk_idx]
        return one

    def _iter_training_losses_flux2_eatb(self, sample: dict[str, Any]):
        required = (
            "eatb_episode_id",
            "eatb_start_chunk_id",
            "eatb_chunk_ids",
            "eatb_chunk_mask",
            "eatb_is_last_window",
        )
        missing = [key for key in required if key not in sample]
        if missing:
            raise ValueError(f"EATB batch is missing fields: {missing}.")
        _, pending = self._edit_memory_eatb_stores()
        if pending:
            raise RuntimeError("Previous EATB banks were not committed after the optimizer update.")

        chunk_mask = sample["eatb_chunk_mask"].to(
            device=self.device, dtype=torch.bool, non_blocking=True
        )
        chunk_ids = sample["eatb_chunk_ids"].to(
            device=self.device, dtype=torch.long, non_blocking=True
        )
        if chunk_mask.ndim != 2 or chunk_ids.shape != chunk_mask.shape:
            raise ValueError(
                f"EATB chunk metadata must be [B,K], got mask={tuple(chunk_mask.shape)} "
                f"and ids={tuple(chunk_ids.shape)}."
            )
        batch_size, num_chunks = chunk_mask.shape
        configured_chunks = max(int(self._edit_memory_training_cfg().get("tbptt_steps", 8)), 1)
        if int(num_chunks) != configured_chunks:
            raise ValueError(
                f"EATB batch has K={num_chunks}, expected tbptt_steps={configured_chunks}."
            )

        local_valid = chunk_mask.long().sum()
        global_valid = local_valid.clone()
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_valid, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()
        if int(global_valid.item()) <= 0:
            raise ValueError("EATB optimizer step contains no valid chunks on any rank.")
        total_valid_steps = global_valid.to(dtype=torch.float32)

        episode_ids = sample["eatb_episode_id"].reshape(batch_size, -1)[:, 0].tolist()
        start_chunk_ids = sample["eatb_start_chunk_id"].reshape(batch_size, -1)[:, 0].tolist()
        last_windows = sample["eatb_is_last_window"].reshape(batch_size, -1)[:, 0].tolist()
        active_ids = [int(episode_id) for episode_id in episode_ids if int(episode_id) >= 0]
        if len(active_ids) != len(set(active_ids)):
            raise ValueError(f"EATB batch contains duplicate episode ids: {active_ids}.")

        for sample_idx in range(int(batch_size)):
            episode_id = int(episode_ids[sample_idx])
            start_chunk_id = int(start_chunk_ids[sample_idx])
            valid_ids = chunk_ids[sample_idx][chunk_mask[sample_idx]]
            expected_ids = torch.arange(
                start_chunk_id,
                start_chunk_id + int(valid_ids.numel()),
                device=valid_ids.device,
                dtype=valid_ids.dtype,
            )
            if not torch.equal(valid_ids, expected_ids):
                raise ValueError(
                    f"EATB episode_id={episode_id} is not consecutive: "
                    f"got {valid_ids.tolist()}, expected {expected_ids.tolist()}."
                )

            bank = self._edit_memory_eatb_load_bank(episode_id, start_chunk_id)
            lane_loss = None
            lane_metrics: dict[str, float] = {}
            for chunk_idx in range(int(num_chunks)):
                step_sample = self._edit_memory_slice_eatb_step(
                    sample,
                    sample_idx=sample_idx,
                    chunk_idx=chunk_idx,
                    batch_size=int(batch_size),
                    num_chunks=int(num_chunks),
                )
                valid = chunk_mask[sample_idx : sample_idx + 1, chunk_idx]
                step_id = (
                    int(chunk_ids[sample_idx, chunk_idx].item())
                    if bool(valid.item())
                    else start_chunk_id + chunk_idx
                )
                step_loss, step_metrics = self._edit_memory_episode_tbptt_step(
                    step_sample,
                    bank=bank,
                    valid=valid,
                    step_idx=step_id,
                    total_valid_steps=total_valid_steps,
                )
                lane_loss = step_loss if lane_loss is None else lane_loss + step_loss
                for key, value in step_metrics.items():
                    lane_metrics[key] = lane_metrics.get(key, 0.0) + float(value)

            pending.append(
                {
                    "episode_id": episode_id,
                    "next_chunk_id": start_chunk_id + int(valid_ids.numel()),
                    "is_last_window": bool(last_windows[sample_idx]),
                    "bank": bank,
                }
            )
            if lane_loss is None:
                raise RuntimeError("EATB lane produced no objective.")
            if sample_idx == int(batch_size) - 1:
                lane_metrics.update(
                    {
                        "edit_memory_eatb_k": float(num_chunks),
                        "edit_memory_eatb_local_valid_chunks": float(local_valid.item()),
                        "edit_memory_eatb_global_valid_chunks": float(global_valid.item()),
                        "edit_memory_eatb_active_episodes": float(len(active_ids)),
                        "edit_memory_episode_full_bptt": 0.0,
                    }
                )
            scale = float(world_size)
            yield lane_loss * scale, {
                key: float(value) * scale for key, value in lane_metrics.items()
            }

    @staticmethod
    def _edit_memory_slice_episode_batch(
        sample: dict[str, Any], sample_idx: int, batch_size: int
    ) -> dict[str, Any]:
        sliced = {}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.shape[0]) == batch_size:
                sliced[key] = value[sample_idx : sample_idx + 1]
            elif isinstance(value, list) and len(value) == batch_size:
                sliced[key] = value[sample_idx : sample_idx + 1]
            elif isinstance(value, tuple) and len(value) == batch_size:
                sliced[key] = value[sample_idx : sample_idx + 1]
            else:
                sliced[key] = value
        return sliced

    def _iter_training_losses_flux2_episode_tbptt(
        self,
        sample: dict[str, Any],
    ):
        local_batch_size = int(sample["video"].shape[0])
        if local_batch_size <= 0:
            raise ValueError("KV8 episode-TBPTT requires a non-empty local batch.")
        global_batch_size_tensor = torch.tensor(
            local_batch_size, device=self.device, dtype=torch.long
        )
        global_real_episodes_tensor = global_batch_size_tensor.clone()
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_batch_size_tensor, op=dist.ReduceOp.MAX)
            dist.all_reduce(global_real_episodes_tensor, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()
        global_batch_size = int(global_batch_size_tensor.item())
        batch_scale = float(world_size) / float(global_real_episodes_tensor.item())
        for sample_idx in range(global_batch_size):
            episode_is_valid = sample_idx < local_batch_size
            episode_sample = self._edit_memory_slice_episode_batch(
                sample,
                sample_idx if episode_is_valid else 0,
                local_batch_size,
            )
            for loss, metrics in self._iter_training_losses_flux2_episode_tbptt_single(
                episode_sample,
                episode_is_valid=episode_is_valid,
            ):
                yield loss * batch_scale, {
                    key: float(value) * batch_scale for key, value in metrics.items()
                }

    def _iter_training_losses_flux2_episode_tbptt_single(
        self,
        sample: dict[str, Any],
        *,
        episode_is_valid: bool = True,
    ):
        memory_source = str(
            self._edit_memory_training_cfg().get("memory_source", "episode_tbptt")
        ).strip().lower()
        if memory_source not in {"episode_tbptt", "episode_autoregressive"}:
            raise ValueError(
                "Episode replay requires memory_source=episode_tbptt or "
                f"episode_autoregressive, got {memory_source!r}."
            )
        full_episode_bptt = memory_source == "episode_autoregressive"
        required = (
            "edit_memory_video",
            "edit_memory_action",
            "edit_memory_action_is_pad",
            "edit_memory_valid",
        )
        missing = [key for key in required if key not in sample]
        if missing:
            raise ValueError(f"KV8 episode-TBPTT is missing dataset fields: {missing}.")
        history_valid = sample["edit_memory_valid"].to(
            device=self.device, dtype=torch.bool, non_blocking=True
        )
        batch_size, _ = history_valid.shape
        if int(batch_size) != 1:
            raise ValueError(
                "Each independent KV8 episode-TBPTT stream must have batch_size=1."
            )
        valid_history_indices = torch.nonzero(
            history_valid[0], as_tuple=False
        ).flatten()
        if not episode_is_valid:
            valid_history_indices = valid_history_indices[:0]
        local_history_steps = int(valid_history_indices.numel())
        configured_replan_steps = int(
            self._edit_memory_training_cfg().get("replan_steps", 16)
        )
        sample_stride = sample.get("edit_memory_history_stride")
        if sample_stride is None or not bool(
            (sample_stride.to(device=self.device).long() == configured_replan_steps).all()
        ):
            raise ValueError(
                "KV8 episode-TBPTT history stride must equal replan_steps="
                f"{configured_replan_steps}."
            )

        global_sequence_steps_tensor = torch.tensor(
            local_history_steps + (1 if episode_is_valid else 0),
            device=self.device,
            dtype=torch.long,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_sequence_steps_tensor, op=dist.ReduceOp.MAX)
        global_sequence_steps = int(global_sequence_steps_tensor.item())
        total_valid_steps = history_valid.new_tensor(
            float(local_history_steps + (1 if episode_is_valid else 0)),
            dtype=torch.float32,
        )
        configured_tbptt_steps = max(
            int(self._edit_memory_training_cfg().get("tbptt_steps", 4)), 1
        )
        offload_tbptt_activations = (
            not full_episode_bptt
            and os.environ.get("IMAGEWAM_TBPTT_SAVE_ON_CPU", "0") == "1"
        )
        pin_offloaded_activations = (
            os.environ.get("IMAGEWAM_TBPTT_SAVE_ON_CPU_PIN_MEMORY", "1") == "1"
        )
        # Paper-style autoregressive episode training keeps one graph across
        # every chronological replan. The legacy TBPTT mode remains available
        # as an explicit memory-saving fallback.
        backward_span = (
            max(global_sequence_steps, 1)
            if full_episode_bptt
            else configured_tbptt_steps
        )
        bank_cfg = getattr(self, "edit_memory_config", {}).get("bank", {})
        bank = EditKVMemoryBank(
            max_entries=int(bank_cfg.get("max_entries", 0)),
            write_threshold=float(bank_cfg.get("write_threshold", 0.5)),
            cooldown_steps=int(bank_cfg.get("cooldown_steps", 0)),
            consolidation=str(bank_cfg.get("consolidation", "none")),
            merge_until_under_limit=bool(bank_cfg.get("merge_until_under_limit", True)),
            anchor_frames=int(bank_cfg.get("anchor_frames", 2)),
            recent_frames=self._edit_memory_historical_recent_frames(bank_cfg),
            gist_tokens_per_frame=int(bank_cfg.get("gist_tokens_per_frame", 8)),
            detach_on_write=False,
        )
        bank.reset(batch_size=int(batch_size), device=self.device)

        chunk_loss = None
        chunk_steps = 0
        processed_replans = 0
        episode_metrics: dict[str, float] = {}
        for sequence_idx in range(global_sequence_steps):
            is_history = sequence_idx < local_history_steps
            is_current = episode_is_valid and sequence_idx == local_history_steps
            if is_history:
                history_idx = int(valid_history_indices[sequence_idx].item())
                step_sample = dict(sample)
                step_sample["video"] = sample["edit_memory_video"][:, history_idx]
                step_sample["action"] = sample["edit_memory_action"][:, history_idx]
                step_sample["action_is_pad"] = sample[
                    "edit_memory_action_is_pad"
                ][:, history_idx]
                step_sample["proprio"] = sample["edit_memory_proprio"][:, history_idx]
                if "edit_memory_action_dim_is_pad" in sample:
                    step_sample["action_dim_is_pad"] = sample[
                        "edit_memory_action_dim_is_pad"
                    ][:, history_idx]
                step_sample["edit_memory_episode_id"] = sample[
                    "edit_memory_history_episode_id"
                ][:, history_idx]
                step_sample["edit_memory_timestep"] = sample[
                    "edit_memory_history_timestep"
                ][:, history_idx]
                valid = history_valid[:, history_idx]
            elif is_current:
                step_sample = sample
                valid = torch.ones(
                    int(batch_size), device=self.device, dtype=torch.bool
                )
            else:
                # Every rank executes the same model-forward sequence. Reuse a
                # shape-correct terminal sample for padding, keep the bank
                # unchanged, and remove its objective with `valid=false`.
                step_sample = sample
                valid = torch.zeros(
                    int(batch_size), device=self.device, dtype=torch.bool
                )

            activation_context = (
                torch.autograd.graph.save_on_cpu(
                    pin_memory=pin_offloaded_activations,
                )
                if offload_tbptt_activations
                else nullcontext()
            )
            with activation_context:
                step_loss, step_metrics = self._edit_memory_episode_tbptt_step(
                    step_sample,
                    bank=bank,
                    valid=valid,
                    step_idx=int(processed_replans),
                    total_valid_steps=total_valid_steps,
                )
            processed_replans += int(valid.long().sum().item())
            for key, value in step_metrics.items():
                episode_metrics[key] = episode_metrics.get(key, 0.0) + float(value)

            chunk_loss = step_loss if chunk_loss is None else chunk_loss + step_loss
            chunk_steps += 1
            is_chunk_boundary = (
                not full_episode_bptt and chunk_steps >= configured_tbptt_steps
            )
            is_final_sequence_step = sequence_idx + 1 == global_sequence_steps
            if is_chunk_boundary or is_final_sequence_step:
                if not full_episode_bptt:
                    bank.detach_graph()
                output_metrics = {}
                if is_final_sequence_step:
                    output_metrics = dict(episode_metrics)
                    final_cache = bank.read()
                    output_metrics.update(
                        self._edit_memory_log_payload(
                            final_cache,
                            num_entries=len(bank.entries),
                            write_confidence=float(bank.last_write_confidence),
                            write_happened=bool(bank.last_write_happened),
                            num_merged=int(bank.last_num_evicted),
                        )
                    )
                    output_metrics.update(
                        {
                            "edit_memory_episode_tbptt_replans": float(
                                total_valid_steps.detach().item()
                            ),
                            "edit_memory_episode_tbptt_steps": float(backward_span),
                            "edit_memory_episode_full_bptt": float(
                                full_episode_bptt
                            ),
                            "edit_memory_global_history_slots": float(
                                global_sequence_steps - 1
                            ),
                            "edit_memory_training_step": float(
                                getattr(self, "_edit_memory_training_step", 0)
                            ),
                        }
                    )
                yield chunk_loss, output_metrics
                chunk_loss = None
                step_loss = None
                chunk_steps = 0
                if (
                    os.environ.get("IMAGEWAM_TBPTT_EMPTY_CACHE", "0") == "1"
                    and torch.cuda.is_available()
                ):
                    torch.cuda.empty_cache()

    def _training_loss_flux2(self, sample, tiled: bool = False):
        if getattr(self, "edit_memory_enabled", False):
            memory_source = str(self._edit_memory_training_cfg()["memory_source"])
            if memory_source in {"episode_window", "sequence", "persistent", "persistent_episode"}:
                return self._training_loss_flux2_episode_window(sample, tiled=tiled)

        inputs = self.build_inputs_flux2(sample, tiled=tiled)
        target_latent = inputs["target_latent"]
        action = inputs["action"]
        batch_size = int(target_latent.shape[0])

        noise_video = torch.randn_like(target_latent)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=target_latent.dtype,
        )
        noisy_latent = self.train_video_scheduler.add_noise(target_latent, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(target_latent, noise_video, timestep_video)

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        gist_tokens, gist_ids = (None, None)
        if getattr(self, "edit_memory_enabled", False):
            gist_tokens, gist_ids = self._build_flux2_gist_inputs(
                batch_size=batch_size,
                ref_img_ids=inputs["ref_img_ids"],
                device=noisy_latent.device,
                token_dtype=inputs["ref_image_latents"].dtype,
                step_idx=0,
            )
        video_pre = self.video_expert.pre_dit(
            x=noisy_latent,
            timestep=self._scheduler_timestep_to_unit(timestep_video, self.train_video_scheduler),
            context=inputs["text_hidden_states"],
            context_mask=inputs["text_attention_mask"],
            ref_image_hidden_states=inputs["ref_image_latents"],
            gist_hidden_states=gist_tokens,
            target_img_ids=inputs["target_img_ids"],
            ref_img_ids=inputs["ref_img_ids"],
            gist_img_ids=gist_ids,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=self._scheduler_timestep_to_unit(timestep_action, self.train_action_scheduler),
        )
        if getattr(self, "edit_memory_enabled", False):
            memory_training_cfg = self._edit_memory_training_cfg()
            memory_source = str(memory_training_cfg["memory_source"])
            memory_kv_cache = None
            if memory_source == "history":
                memory_kv_cache = self._build_flux2_history_memory_cache(sample)
            prefix_attention_mask = self._build_mot_attention_mask_flux2(
                batch_size=batch_size,
                txt_len=int(video_pre["txt_len"]),
                target_len=int(video_pre["target_len"]),
                cond_len=int(video_pre["cond_len"]),
                action_len=0,
                device=noisy_latent.device,
                text_attention_mask=video_pre["text_mask"],
                gist_len=int(video_pre.get("gist_len", 0)),
            )
            video_kv_cache = self.mot.prefill_flux2_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                attention_mask=prefix_attention_mask,
                memory_kv_cache=memory_kv_cache,
            )
            pred_video = self.video_expert.post_dit(video_kv_cache["final_video"], video_pre)

            action_len = int(action_pre["tokens"].shape[1])
            full_attention_mask = self._build_mot_attention_mask_flux2(
                batch_size=batch_size,
                txt_len=int(video_pre["txt_len"]),
                target_len=int(video_pre["target_len"]),
                cond_len=int(video_pre["cond_len"]),
                action_len=action_len,
                device=noisy_latent.device,
                text_attention_mask=video_pre["text_mask"],
                gist_len=int(video_pre.get("gist_len", 0)),
            )
            prefix_len = (
                int(video_pre["txt_len"])
                + int(video_pre["cond_len"])
                + int(video_pre.get("gist_len", 0))
                + int(video_pre["target_len"])
            )
            grid_hw = self._grid_hw_from_flux2_img_ids(inputs.get("ref_img_ids"))
            if memory_source == "current":
                self._configure_action_attention_capture(
                    condition_slice=(int(video_pre["txt_len"]), int(video_pre["txt_len"]) + int(video_pre["cond_len"])),
                    condition_grid=grid_hw,
                    prefix_len=prefix_len,
                    action_len=action_len,
                    metadata={
                        "stack": str(self.stack),
                        "source": "flux2_train_current_edit_memory_teacher",
                        "condition": "ref_image_tokens",
                        "txt_len": int(video_pre["txt_len"]),
                        "cond_len": int(video_pre["cond_len"]),
                        "target_len": int(video_pre["target_len"]),
                    },
                )
                self._start_action_attention_capture_step(0)
            action_tokens = self.mot.forward_action_with_video_cache(
                action_tokens=action_pre["tokens"],
                action_freqs=None,
                action_t_mod=action_pre["t_mod"],
                action_context_payload={"ids": action_pre["ids"]},
                video_kv_cache=video_kv_cache,
                attention_mask=full_attention_mask,
                video_seq_len=prefix_len,
                memory_kv_cache=memory_kv_cache,
            )
            pred_action = self.action_expert.post_dit(action_tokens, action_pre)
            if memory_source == "current":
                teacher_importance = self.edit_memory_capture.get_per_layer_teacher_importance()
                memory_kv_cache = self._select_flux2_edit_memory_item(
                    video_kv_cache=video_kv_cache,
                    per_layer_teacher_importance=teacher_importance,
                    txt_len=int(video_pre["txt_len"]),
                    img_len=int(video_pre["cond_len"]),
                    grid_hw=grid_hw,
                    step_idx=0,
                    gist_len=int(video_pre.get("gist_len", 0)),
                )
                self.edit_memory_capture.deactivate()
            elif memory_source == "history":
                pass
            else:
                raise ValueError(
                    "Unsupported edit_memory.training.memory_source="
                    f"{memory_source!r}; expected 'history' or 'current'."
                )

            video_loss_per_sample = F.mse_loss(
                pred_video.float(),
                target_video.float(),
                reduction="none",
            ).flatten(1).mean(dim=1)
            video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
                video_loss_per_sample.device,
                dtype=video_loss_per_sample.dtype,
            )
            loss_video = (video_loss_per_sample * video_weight).mean()
            action_loss_per_sample = self._compute_action_loss_per_sample(
                pred_action=pred_action,
                target_action=target_action,
                action_is_pad=inputs["action_is_pad"],
                action_dim_is_pad=inputs.get("action_dim_is_pad"),
            )
            action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
                action_loss_per_sample.device,
                dtype=action_loss_per_sample.dtype,
            )
            loss_action = (action_loss_per_sample * action_weight).mean()
            loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
            if memory_kv_cache is None:
                confidence_value = 0.0
                num_memory_entries = 0
            else:
                confidence_value = float(memory_item_confidence(memory_kv_cache).detach().float().mean().item())
                num_memory_entries = int(memory_kv_cache.get("num_entries", 1))
            loss_dict = {
                "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
                "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
                "loss_edit_memory_action": self.loss_lambda_action * float(loss_action.detach().item()),
                "loss_edit_memory_distill": 0.0,
                "edit_memory_training_source_history": 1.0 if memory_source == "history" else 0.0,
            }
            history_valid = sample.get("edit_memory_valid")
            if isinstance(history_valid, torch.Tensor):
                loss_dict["edit_memory_train_history_valid_frac"] = (
                    0.0 if history_valid.numel() == 0 else float(history_valid.float().mean().item())
                )
            loss_dict.update(
                self._edit_memory_log_payload(
                    memory_kv_cache,
                    num_entries=num_memory_entries,
                    write_confidence=confidence_value,
                    write_happened=memory_kv_cache is not None,
                    num_merged=0,
                )
            )
            return loss_total, loss_dict

        attention_mask = self._build_mot_attention_mask_flux2(
            batch_size=batch_size,
            txt_len=int(video_pre["txt_len"]),
            target_len=int(video_pre["target_len"]),
            cond_len=int(video_pre["cond_len"]),
            action_len=int(action_pre["tokens"].shape[1]),
            device=noisy_latent.device,
            text_attention_mask=video_pre["text_mask"],
        )
        tokens_out = self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"]},
            context_all={"video": None, "action": {"ids": action_pre["ids"]}},
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        video_loss_per_sample = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").flatten(1).mean(dim=1)
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            video_loss_per_sample.device,
            dtype=video_loss_per_sample.dtype,
        )
        loss_video = (video_loss_per_sample * video_weight).mean()
        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=inputs["action_is_pad"],
            action_dim_is_pad=inputs.get("action_dim_is_pad"),
        )
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device,
            dtype=action_loss_per_sample.dtype,
        )
        loss_action = (action_loss_per_sample * action_weight).mean()
        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        return loss_total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }

    def _training_loss_dim(self, sample, tiled: bool = False):
        inputs = self.build_inputs_dim(sample, tiled=tiled)
        target_latent = inputs["target_latent"]
        source_latent = inputs["source_latent"]
        condition = inputs["condition"]
        condition_mask = inputs["condition_mask"]
        action = inputs["action"]
        batch_size = int(target_latent.shape[0])

        noise_video = torch.randn_like(target_latent)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=target_latent.dtype,
        )
        noisy_latent = self.train_video_scheduler.add_noise(target_latent, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(target_latent, noise_video, timestep_video)

        cond_latent = torch.randn_like(target_latent)
        timestep_cond = torch.full(
            (batch_size,),
            float(self.train_video_scheduler.num_train_timesteps),
            device=self.device,
            dtype=target_latent.dtype,
        )

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=noisy_latent,
            timestep=timestep_video,
            context=condition,
            context_mask=condition_mask,
            latents_condition=source_latent if self.dim_with_latents_condition else None,
        )
        cond_pre = self.video_expert.pre_dit(
            x=cond_latent,
            timestep=timestep_cond,
            context=condition,
            context_mask=condition_mask,
            latents_condition=source_latent if self.dim_with_latents_condition else None,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "cond": cond_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=None,
            freqs_all={},
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                    "hw": (video_pre["meta"]["token_height"], video_pre["meta"]["token_width"]),
                },
                "cond": {
                    "context": cond_pre["context"],
                    "mask": cond_pre["context_mask"],
                    "hw": (cond_pre["meta"]["token_height"], cond_pre["meta"]["token_width"]),
                },
                "action": {
                    "context_tokens": cond_pre["context_tokens"],
                    "context_mask_bool": cond_pre["context_mask_bool"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "cond": cond_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        video_loss_per_sample = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").flatten(1).mean(dim=1)
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            video_loss_per_sample.device,
            dtype=video_loss_per_sample.dtype,
        )
        loss_video = (video_loss_per_sample * video_weight).mean()
        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=inputs["action_is_pad"],
            action_dim_is_pad=inputs.get("action_dim_is_pad"),
        )
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device,
            dtype=action_loss_per_sample.dtype,
        )
        loss_action = (action_loss_per_sample * action_weight).mean()
        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        return loss_total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }

    def iter_training_losses(self, sample, tiled: bool = False):
        if (
            self.stack == "flux2"
            and getattr(self, "edit_memory_enabled", False)
            and str(self._edit_memory_training_cfg()["memory_source"]) == "eatb"
        ):
            yield from self._iter_training_losses_flux2_eatb(sample)
            return
        if (
            self.stack == "flux2"
            and getattr(self, "edit_memory_enabled", False)
            and str(self._edit_memory_training_cfg()["memory_source"])
            in {"episode_tbptt", "episode_autoregressive"}
        ):
            yield from self._iter_training_losses_flux2_episode_tbptt(sample)
            return
        yield self.training_loss(sample, tiled=tiled)

    def training_loss(self, sample, tiled: bool = False):
        if self.stack == "dim":
            return self._training_loss_dim(sample, tiled=tiled)
        if self.stack == "flux2":
            return self._training_loss_flux2(sample, tiled=tiled)
        if self.stack == "ovis_u1":
            return self._training_loss_ovis_u1(sample, tiled=tiled)
        if self.stack == "omnigen2":
            return self._training_loss_omnigen2(sample, tiled=tiled)

        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)

        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=action_is_pad,
            action_dim_is_pad=inputs.get("action_dim_is_pad"),
        )

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )

        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        memory_kv_cache: dict[str, object] | None = None,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            memory_kv_cache=memory_kv_cache,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
            )["action"]
        
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        profile_infer_timing: bool = False,
    ) -> dict[str, Any]:
        if self.stack == "dim":
            return self.infer_action_dim(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
        if self.stack == "flux2":
            return self.infer_action_flux2(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
        if self.stack == "ovis_u1":
            return self.infer_action_ovis_u1(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
        if self.stack == "omnigen2":
            return self.infer_action_omnigen2(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                profile_infer_timing=profile_infer_timing,
            )

        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        patch_size = tuple(getattr(self.video_expert, "patch_size", (1, 1, 1)))
        condition_grid = (
            int(first_frame_latents.shape[3]) // int(patch_size[1]),
            int(first_frame_latents.shape[4]) // int(patch_size[2]),
        )
        self._configure_action_attention_capture(
            condition_slice=(0, video_seq_len),
            condition_grid=condition_grid,
            prefix_len=video_seq_len,
            action_len=int(latents_action.shape[1]),
            metadata={
                "stack": str(self.stack),
                "source": "imagewam_infer_action",
                "condition": "first_frame_latents",
                "input_size": [int(height), int(width)],
            },
        )
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for action_step_idx, (step_t_action, step_delta_action) in enumerate(zip(infer_timesteps_action, infer_deltas_action)):
            self._start_action_attention_capture_step(action_step_idx)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def infer_action_dim(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        downsample = int(self.video_expert.vae_downsample_rate)
        if input_image.shape[-2] % downsample != 0 or input_image.shape[-1] % downsample != 0:
            raise ValueError(
                f"DIM input spatial dims must be multiples of {downsample}, got {tuple(input_image.shape[-2:])}."
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        batch_size = int(input_image.shape[0])

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            condition = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            condition_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        else:
            if prompt is None:
                raise ValueError("DIM action inference requires `prompt` unless precomputed context is provided.")
            condition, condition_mask = self._encode_dim_condition_online(prompt, input_image)
        if self.proprio_encoder is not None or proprio is not None:
            condition, condition_mask = self._append_proprio_to_context_if_enabled(
                context=condition,
                context_mask=condition_mask,
                proprio=proprio,
                source="DIM action inference",
            )

        source_latent = self._encode_dim_image_latents(input_image)
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        cond_latent = torch.randn(
            source_latent.shape,
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (batch_size, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        timestep_cond = torch.full(
            (batch_size,),
            float(self.infer_video_scheduler.num_train_timesteps),
            device=self.device,
            dtype=source_latent.dtype,
        )
        cond_pre = self.video_expert.pre_dit(
            x=cond_latent,
            timestep=timestep_cond,
            context=condition,
            context_mask=condition_mask,
            latents_condition=source_latent if self.dim_with_latents_condition else None,
        )
        cond_cache = self.mot.prefill_sana_condition_cache(
            cond_tokens=cond_pre["tokens"],
            cond_t_mod=cond_pre["t_mod"],
            cond_context_payload={
                "context": cond_pre["context"],
                "mask": cond_pre["context_mask"],
                "hw": (cond_pre["meta"]["token_height"], cond_pre["meta"]["token_width"]),
            },
        )
        infer_timesteps, infer_deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        action_context_payload = {
            "context_tokens": cond_pre["context_tokens"],
            "context_mask_bool": cond_pre["context_mask_bool"],
        }
        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep_action = step_t.expand(batch_size).to(dtype=latents_action.dtype, device=self.device)
            action_pre = self.action_expert.pre_dit(
                action_tokens=latents_action,
                timestep=timestep_action,
            )
            action_tokens = self.mot.forward_sana_action_with_condition_cache(
                action_tokens=action_pre["tokens"],
                action_t_mod=action_pre["t_mod"],
                action_context_payload=action_context_payload,
                condition_cache=cond_cache,
            )
            pred_action = self.action_expert.post_dit(action_tokens, action_pre)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta, latents_action)
        result = {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}
        if getattr(self, "edit_memory_enabled", False):
            memory_debug: dict[str, Any] = {
                "replan_index": int(self._edit_memory_infer_step_idx - 1),
                "stored_frames": 0 if self.edit_memory_bank is None else int(len(self.edit_memory_bank.frames)),
            }
            if span_decision is not None:
                memory_debug["history_span_hard"] = int(span_decision["hard_span"][0].item())
                memory_debug["history_span_soft"] = float(span_decision["soft_span"][0].item())
            if gist_decision is not None:
                memory_debug["gist_count_hard"] = int(gist_decision["hard_count"][0].item())
                memory_debug["gist_count_expected"] = float(gist_decision["expected_count"][0].item())
            result["memory_debug"] = memory_debug
        return result

    @torch.no_grad()
    def infer_video_dim(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        downsample = int(self.video_expert.vae_downsample_rate)
        if input_image.shape[-2] % downsample != 0 or input_image.shape[-1] % downsample != 0:
            raise ValueError(
                f"DIM input spatial dims must be multiples of {downsample}, got {tuple(input_image.shape[-2:])}."
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        batch_size = int(input_image.shape[0])

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            condition = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            condition_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        else:
            if prompt is None:
                raise ValueError("DIM video inference requires `prompt` unless precomputed context is provided.")
            condition, condition_mask = self._encode_dim_condition_online(prompt, input_image)
        if self.proprio_encoder is not None or proprio is not None:
            condition, condition_mask = self._append_proprio_to_context_if_enabled(
                context=condition,
                context_mask=condition_mask,
                proprio=proprio,
                source="DIM video inference",
            )

        source_latent = self._encode_dim_image_latents(input_image)
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            source_latent.shape,
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        infer_timesteps, infer_deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep_video = step_t.expand(batch_size).to(dtype=latents_video.dtype, device=self.device)
            video_pre = self.video_expert.pre_dit(
                x=latents_video,
                timestep=timestep_video,
                context=condition,
                context_mask=condition_mask,
                latents_condition=source_latent if self.dim_with_latents_condition else None,
            )
            x = video_pre["tokens"]
            context_payload = {
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
                "hw": (video_pre["meta"]["token_height"], video_pre["meta"]["token_width"]),
            }
            hw = tuple(context_payload["hw"])
            for layer_idx, block in enumerate(self.video_expert.blocks):
                state = self.mot._sana_video_io(block, x, video_pre["t_mod"], hw)
                attn = self.mot._sana_linear_attention(block.attn, state["q"], state["k"], state["v"])
                x = self.mot._sana_video_post(block, attn, state, context_payload)
            pred_video = self.video_expert.post_dit(x, video_pre)
            latents_video = self.infer_video_scheduler.step(pred_video, step_delta, latents_video)

        image = self._decode_dim_image_latents(latents_video)
        return {"image": image[0].detach().cpu()}

    @torch.no_grad()
    def infer_dim_separate(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        action_out = self.infer_action_dim(
            prompt=prompt,
            input_image=input_image,
            action_horizon=action_horizon,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
        )
        video_out = self.infer_video_dim(
            prompt=prompt,
            input_image=input_image,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
        )
        image = video_out["image"].detach().float().clamp(-1, 1)
        image = ((image + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        return {"action": action_out["action"], "video": [Image.fromarray(image)]}

    @torch.no_grad()
    def infer_action_flux2(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")

        text_hidden, text_mask = self._prepare_flux2_infer_text(prompt, context, context_mask)
        if self.proprio_encoder is not None or proprio is not None:
            text_hidden, text_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden,
                context_mask=text_mask,
                proprio=proprio,
                source="FLUX.2 action inference",
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        ref_tokens, ref_img_ids = self._encode_flux2_image_tokens(
            input_image,
            time_value=10.0,
            temporal_value=float(self._edit_memory_infer_step_idx),
        )
        frame_summary = self._edit_memory_frame_summary(ref_tokens, text_hidden, text_mask)
        span_decision = self._edit_memory_span_decision(frame_summary)
        gist_decision = None
        write_decision = None
        write_happened = False
        entries_before_read = 0
        entries_before_write = 0
        memory_item = None
        batch_size = int(ref_tokens.shape[0])
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (batch_size, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        gist_tokens, gist_ids = (None, None)
        if getattr(self, "edit_memory_enabled", False):
            gist_tokens, gist_ids = self._build_flux2_gist_inputs(
                batch_size=batch_size,
                ref_img_ids=ref_img_ids,
                device=ref_tokens.device,
                token_dtype=ref_tokens.dtype,
                step_idx=int(self._edit_memory_infer_step_idx),
            )
        memory_kv_cache = None
        edit_memory_debug = os.environ.get("IMAGEWAM_EDIT_MEMORY_DEBUG", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if getattr(self, "edit_memory_enabled", False) and self.edit_memory_bank is not None:
            entries_before_read = int(len(self.edit_memory_bank.entries))
            memory_kv_cache = self.edit_memory_bank.read(
                history_span=span_decision,
                compact_dynamic=True,
            )
            if edit_memory_debug:
                logger.info(
                    "[edit_memory] read infer_step=%d entries=%d tokens_per_layer=%d kv_tokens_total=%d",
                    int(self._edit_memory_infer_step_idx),
                    entries_before_read,
                    memory_tokens_per_layer(memory_kv_cache),
                    memory_kv_tokens_total(memory_kv_cache),
                )
        video_pre, video_kv_cache = self._prefill_flux2_memory_prefix(
            ref_tokens=ref_tokens,
            ref_img_ids=ref_img_ids,
            text_hidden=text_hidden,
            text_mask=text_mask,
            gist_tokens=gist_tokens,
            gist_ids=gist_ids,
            memory_kv_cache=memory_kv_cache,
        )
        if (
            self.edit_memory_gist_allocator is not None
            or self.edit_memory_write_head is not None
        ):
            prefix_summary = self._edit_memory_prefix_summary(video_kv_cache)
            gist_decision = self._edit_memory_gist_decision(prefix_summary)
            write_decision = self._edit_memory_write_confidence(prefix_summary, gist_decision)
            if gist_decision is not None:
                gist_decision["write_decision"] = write_decision
        full_attention_mask = self._build_mot_attention_mask_flux2(
            batch_size=batch_size,
            txt_len=int(video_pre["txt_len"]),
            target_len=0,
            cond_len=int(video_pre["cond_len"]),
            action_len=int(latents_action.shape[1]),
            device=latents_action.device,
            text_attention_mask=video_pre["text_mask"],
            gist_len=int(video_pre.get("gist_len", 0)),
        )
        prefix_len = int(video_pre["txt_len"]) + int(video_pre["cond_len"]) + int(video_pre.get("gist_len", 0))
        grid_hw = (int(input_image.shape[-2]) // 16, int(input_image.shape[-1]) // 16)
        for action_step_idx, (step_t_action, step_delta_action) in enumerate(zip(infer_timesteps_action, infer_deltas_action)):
            timestep_action = step_t_action.expand(batch_size).to(dtype=latents_action.dtype, device=self.device)
            action_pre = self.action_expert.pre_dit(
                action_tokens=latents_action,
                timestep=self._scheduler_timestep_to_unit(timestep_action, self.infer_action_scheduler),
                temporal_value=float(self._edit_memory_infer_step_idx),
            )
            action_tokens = self.mot.forward_action_with_video_cache(
                action_tokens=action_pre["tokens"],
                action_freqs=None,
                action_t_mod=action_pre["t_mod"],
                action_context_payload={"ids": action_pre["ids"]},
                video_kv_cache=video_kv_cache,
                attention_mask=full_attention_mask,
                video_seq_len=prefix_len,
                memory_kv_cache=memory_kv_cache,
            )
            pred_action = self.action_expert.post_dit(action_tokens, action_pre)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        if getattr(self, "edit_memory_enabled", False) and self.edit_memory_bank is not None:
            memory_item = self._select_flux2_edit_memory_item(
                video_kv_cache=video_kv_cache,
                per_layer_teacher_importance=None,
                txt_len=int(video_pre["txt_len"]),
                img_len=int(video_pre["cond_len"]),
                grid_hw=grid_hw,
                step_idx=int(self._edit_memory_infer_step_idx),
                gist_len=int(video_pre.get("gist_len", 0)),
                gist_decision=gist_decision,
                write_confidence=None
                if write_decision is None
                else write_decision["probabilities"],
                compact_adaptive_gist=True,
            )
            confidence = memory_item["confidence"]
            entries_before_write = int(len(self.edit_memory_bank.entries))
            gist_write_selected = (
                True
                if write_decision is None
                else write_decision["hard_write"]
            )
            write_happened = self.edit_memory_bank.maybe_write(
                memory_item,
                step_idx=int(self._edit_memory_infer_step_idx),
                confidence=confidence,
                gist_write_selected=gist_write_selected,
            )
            if edit_memory_debug:
                logger.info(
                    "[edit_memory] store infer_step=%d confidence=%.6f full_frame=%s gist_write=%s entries_before=%d "
                    "entries_after=%d evicted=%d threshold=%.6f cooldown=%d tokens_per_layer=%d kv_tokens_total=%d",
                    int(self._edit_memory_infer_step_idx),
                    float(self.edit_memory_bank.last_write_confidence),
                    bool(self.edit_memory_bank.last_full_frame_stored),
                    bool(write_happened),
                    entries_before_write,
                    int(len(self.edit_memory_bank.entries)),
                    int(getattr(self.edit_memory_bank, "last_num_evicted", self.edit_memory_bank.last_num_merged)),
                    float(self.edit_memory_bank.write_threshold),
                    int(self.edit_memory_bank.cooldown_steps),
                    memory_tokens_per_layer(memory_item),
                    memory_kv_tokens_total(memory_item),
            )
            self._edit_memory_infer_step_idx += 1

        memory_debug = None
        if getattr(self, "edit_memory_enabled", False) and self.edit_memory_bank is not None:
            hard_count = None
            if gist_decision is not None and gist_decision.get("hard_count") is not None:
                hard_count = int(gist_decision["hard_count"][0].detach().cpu().item())
            memory_debug = {
                "infer_step": int(self._edit_memory_infer_step_idx - 1),
                "entries_before_read": int(entries_before_read),
                "entries_before_write": int(entries_before_write),
                "entries_after_write": int(len(self.edit_memory_bank.entries)),
                "read_tokens_per_layer": int(memory_tokens_per_layer(memory_kv_cache)),
                "read_kv_tokens_total": int(memory_kv_tokens_total(memory_kv_cache)),
                "item_tokens_per_layer": int(memory_tokens_per_layer(memory_item)),
                "item_kv_tokens_total": int(memory_kv_tokens_total(memory_item)),
                "adaptive_gist_hard_count": hard_count,
                "write_confidence": float(self.edit_memory_bank.last_write_confidence),
                "write_happened": bool(write_happened),
                "gist_write_happened": bool(write_happened),
                "full_frame_stored": bool(
                    self.edit_memory_bank.last_full_frame_stored
                ),
                "evicted_entries": int(
                    getattr(
                        self.edit_memory_bank,
                        "last_num_evicted",
                        self.edit_memory_bank.last_num_merged,
                    )
                ),
            }
        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
            "memory_debug": memory_debug,
        }

    def _forward_flux2_video_only(
        self,
        video_pre: dict[str, Any],
        attention_mask: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        video_state = video_pre["tokens"]
        if not isinstance(video_state, dict):
            raise ValueError("FLUX.2 video tokens must be a dict with `txt` and `img` tensors.")

        txt = video_state["txt"]
        img = video_state["img"]
        video_freqs = video_pre["freqs"]
        txt_pe = video_freqs["txt"]
        img_pe = video_freqs["img"]
        video_t_mod = video_pre["t_mod"]
        video_expert = self.video_expert

        def _flatten_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], tensor.shape[1] * tensor.shape[3])

        from flux2.model import apply_rope

        for layer_idx in range(int(getattr(video_expert, "double_layers"))):
            block = video_expert.double_blocks[layer_idx]
            q, k, v, pe_full, num_txt_tokens, mods = block._prepare_qkv(
                img,
                txt,
                img_pe,
                txt_pe,
                video_t_mod["double_img"],
                video_t_mod["double_txt"],
            )
            q, k = apply_rope(q, k, pe_full)
            mixed = self.mot._mixed_attention(
                _flatten_heads(q),
                _flatten_heads(k),
                _flatten_heads(v),
                attention_mask["double_joint"],
            )
            txt_attn, img_attn = torch.split(mixed, [num_txt_tokens, img.shape[1]], dim=1)
            img, txt = block._apply_residuals(img, txt, img_attn, txt_attn, mods)

        video_stream = torch.cat([txt, img], dim=1)
        stream_pe = torch.cat([txt_pe, img_pe], dim=2)
        for layer_idx in range(int(getattr(video_expert, "single_layers"))):
            block = video_expert.single_blocks[layer_idx]
            q, k, v, mlp, gate = block._qkv(video_stream, video_t_mod["single"])
            q, k = apply_rope(q, k, stream_pe)
            mixed = self.mot._mixed_attention(
                _flatten_heads(q),
                _flatten_heads(k),
                _flatten_heads(v),
                attention_mask["single"],
            )
            video_stream = block._out(video_stream, mixed, mlp, gate)

        txt_len = int(txt.shape[1])
        txt, img = video_stream[:, :txt_len], video_stream[:, txt_len:]
        return {"txt": txt, "img": img}

    @torch.no_grad()
    def infer_video_flux2(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        from .flux2_video_expert import Flux2VideoExpert

        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"FLUX.2 image spatial dims must be multiples of 16, got HxW=({height},{width})")

        text_hidden, text_mask = self._prepare_flux2_infer_text(prompt, context, context_mask)
        if self.proprio_encoder is not None or proprio is not None:
            text_hidden, text_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden,
                context_mask=text_mask,
                proprio=proprio,
                source="FLUX.2 video inference",
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        ref_tokens, ref_img_ids = self._encode_flux2_image_tokens(
            input_image,
            time_value=10.0,
            temporal_value=float(self._edit_memory_infer_step_idx),
        )
        batch_size = int(ref_tokens.shape[0])
        latent_h = int(height) // 16
        latent_w = int(width) // 16
        memory_kv_cache = None
        gist_tokens, gist_ids = (None, None)
        if getattr(self, "edit_memory_enabled", False):
            frame_summary = self._edit_memory_frame_summary(
                ref_tokens, text_hidden, text_mask
            )
            span_decision = self._edit_memory_span_decision(frame_summary)
            gist_tokens, gist_ids = self._build_flux2_gist_inputs(
                batch_size=batch_size,
                ref_img_ids=ref_img_ids,
                device=ref_tokens.device,
                token_dtype=ref_tokens.dtype,
                step_idx=int(self._edit_memory_infer_step_idx),
            )
            if self.edit_memory_bank is not None:
                memory_kv_cache = self.edit_memory_bank.read(
                    history_span=span_decision,
                    compact_dynamic=True,
                )

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            ref_tokens.shape,
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        target_img_ids = Flux2VideoExpert.build_img_ids(
            batch_size=batch_size,
            token_height=latent_h,
            token_width=latent_w,
            time_value=0.0,
            temporal_value=float(self._edit_memory_infer_step_idx + 1),
            device=self.device,
            dtype=self.torch_dtype,
        )

        infer_timesteps, infer_deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep_video = step_t.expand(batch_size).to(dtype=latents_video.dtype, device=self.device)
            video_pre = self.video_expert.pre_dit(
                x=latents_video,
                timestep=self._scheduler_timestep_to_unit(timestep_video, self.infer_video_scheduler),
                context=text_hidden,
                context_mask=text_mask,
                ref_image_hidden_states=ref_tokens,
                gist_hidden_states=gist_tokens,
                target_img_ids=target_img_ids,
                ref_img_ids=ref_img_ids,
                gist_img_ids=gist_ids,
            )
            attention_mask = self._build_mot_attention_mask_flux2(
                batch_size=batch_size,
                txt_len=int(video_pre["txt_len"]),
                target_len=int(video_pre["target_len"]),
                cond_len=int(video_pre["cond_len"]),
                action_len=0,
                device=latents_video.device,
                text_attention_mask=video_pre["text_mask"],
                gist_len=int(video_pre.get("gist_len", 0)),
            )
            video_kv_cache = self.mot.prefill_flux2_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                attention_mask=attention_mask,
                memory_kv_cache=memory_kv_cache,
            )
            pred_video = self.video_expert.post_dit(
                video_kv_cache["final_video"], video_pre
            )
            latents_video = self.infer_video_scheduler.step(pred_video, step_delta, latents_video)

        image = self._decode_flux2_image_tokens(latents_video, height=height, width=width)
        return {"image": image[0].detach().cpu()}

    @torch.no_grad()
    def infer_action_omnigen2(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        profile_infer_timing: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`input_image` spatial dims must be multiples of 16, got HxW=({height},{width})")

        profile_segments: list[tuple[str, float]] = []
        profile_enabled = bool(profile_infer_timing)
        profile_last = 0.0

        def _sync_profile_device() -> None:
            if not profile_enabled:
                return
            device = torch.device(self.device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        def _mark_profile(name: str) -> None:
            nonlocal profile_last
            if not profile_enabled:
                return
            _sync_profile_device()
            now = time.perf_counter()
            profile_segments.append((name, now - profile_last))
            profile_last = now

        if profile_enabled:
            _sync_profile_device()
            profile_last = time.perf_counter()

        text_hidden, text_mask = self._prepare_omnigen2_infer_text(prompt, context, context_mask)
        _mark_profile("prepare_text_s")
        if self.proprio_encoder is not None or proprio is not None:
            text_hidden, text_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden,
                context_mask=text_mask,
                proprio=proprio,
                source="OmniGen2 action inference",
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        ref_latent = self._encode_omnigen2_image_latents(input_image)
        _mark_profile("encode_image_latents_s")
        batch_size = int(ref_latent.shape[0])

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (batch_size, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        _mark_profile("sample_latents_s")

        timestep_video = torch.zeros((batch_size,), dtype=ref_latent.dtype, device=self.device)
        _mark_profile("prepare_video_timestep_s")
        video_pre = self.video_expert.pre_dit(
            x=None,
            timestep=timestep_video,
            context=text_hidden,
            context_mask=text_mask,
            ref_image_hidden_states=ref_latent,
        )
        _mark_profile("video_pre_dit_s")
        video_seq_len = int(video_pre["tokens"].shape[1])
        text_len = int(video_pre["encoder_seq_lengths"][0])
        ref_len = int(sum(video_pre["l_effective_ref_img_len"][0]))
        self._configure_action_attention_capture(
            condition_slice=(text_len, text_len + ref_len),
            condition_grid=(int(input_image.shape[-2]) // 16, int(input_image.shape[-1]) // 16),
            prefix_len=video_seq_len,
            action_len=int(latents_action.shape[1]),
            metadata={
                "stack": str(self.stack),
                "source": "omnigen2_infer_action",
                "condition": "ref_image_latents",
                "text_len": text_len,
                "ref_len": ref_len,
                "input_size": [int(input_image.shape[-2]), int(input_image.shape[-1])],
            },
        )
        attention_mask = self._build_mot_attention_mask_omnigen2(
            encoder_seq_lengths=video_pre["encoder_seq_lengths"],
            seq_lengths=video_pre["seq_lengths"],
            max_video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            l_effective_ref_img_len=video_pre["l_effective_ref_img_len"],
            device=video_pre["tokens"].device,
        )
        _mark_profile("build_attention_mask_s")
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload=None,
            video_attention_mask=attention_mask[:, :video_seq_len, :video_seq_len],
        )
        _mark_profile("prefill_video_cache_s")

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        _mark_profile("build_action_schedule_s")
        action_predict_times: list[float] = []
        action_loop_start = 0.0
        timing_payload = None
        if profile_enabled:
            _sync_profile_device()
            action_loop_start = time.perf_counter()
        for action_step_idx, (step_t_action, step_delta_action) in enumerate(zip(infer_timesteps_action, infer_deltas_action)):
            self._start_action_attention_capture_step(action_step_idx)
            timestep_action = step_t_action.expand(batch_size).to(dtype=latents_action.dtype, device=self.device)
            if profile_enabled:
                _sync_profile_device()
                action_predict_start = time.perf_counter()
            pred_action = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=text_hidden,
                context_mask=text_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            if profile_enabled:
                _sync_profile_device()
                action_predict_times.append(time.perf_counter() - action_predict_start)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
        if profile_enabled:
            _sync_profile_device()
            action_denoise_loop_s = time.perf_counter() - action_loop_start
            profile_segments.append(("action_denoise_loop_s", action_denoise_loop_s))
            action_predict_total_s = sum(action_predict_times)
            action_predict_count = len(action_predict_times)
            action_predict_avg_s = action_predict_total_s / action_predict_count if action_predict_count > 0 else 0.0
            action_predict_min_s = min(action_predict_times) if action_predict_times else 0.0
            action_predict_max_s = max(action_predict_times) if action_predict_times else 0.0
            total_profiled_s = sum(value for _, value in profile_segments)
            timing_payload = {
                "segments": {name: float(value) for name, value in profile_segments},
                "action_horizon": int(action_horizon),
                "num_inference_steps": int(action_predict_count),
                "action_predict_total_s": float(action_predict_total_s),
                "action_predict_avg_s": float(action_predict_avg_s),
                "action_predict_min_s": float(action_predict_min_s),
                "action_predict_max_s": float(action_predict_max_s),
                "total_profiled_s": float(total_profiled_s),
            }

        result = {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}
        if timing_payload is not None:
            result["timing"] = timing_payload
        return result

    @torch.no_grad()
    def infer_action_ovis_u1(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`input_image` spatial dims must be multiples of 16, got HxW=({height},{width})")

        text_hidden, text_mask = self._prepare_ovis_u1_infer_text(prompt, input_image, context, context_mask)
        if self.proprio_encoder is not None or proprio is not None:
            text_hidden, text_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden,
                context_mask=text_mask,
                proprio=proprio,
                source="Ovis-U1 action inference",
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        ref_tokens, ref_img_ids = self._encode_ovis_u1_image_tokens(input_image, time_value=1.0)
        batch_size = int(ref_tokens.shape[0])
        empty_target = torch.empty(
            batch_size,
            0,
            int(self.video_expert.transformer.in_channels),
            device=self.device,
            dtype=self.torch_dtype,
        )
        empty_target_ids = torch.empty(batch_size, 0, 3, device=self.device, dtype=self.torch_dtype)
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (batch_size, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        timestep_video = torch.zeros((batch_size,), dtype=self.torch_dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=empty_target,
            timestep=timestep_video,
            context=text_hidden,
            context_mask=text_mask,
            ref_image_hidden_states=ref_tokens,
            target_img_ids=empty_target_ids,
            ref_img_ids=ref_img_ids,
        )
        prefix_attention_mask = self._build_mot_attention_mask_ovis_u1(
            batch_size=batch_size,
            txt_len=int(video_pre["txt_len"]),
            target_len=int(video_pre["target_len"]),
            cond_len=int(video_pre["cond_len"]),
            action_len=0,
            device=self.device,
            text_attention_mask=video_pre["text_mask"],
        )
        video_kv_cache = self.mot.prefill_yak_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            attention_mask=prefix_attention_mask,
        )
        full_attention_mask = self._build_mot_attention_mask_ovis_u1(
            batch_size=batch_size,
            txt_len=int(video_pre["txt_len"]),
            target_len=int(video_pre["target_len"]),
            cond_len=int(video_pre["cond_len"]),
            action_len=int(latents_action.shape[1]),
            device=self.device,
            text_attention_mask=video_pre["text_mask"],
        )
        prefix_len = int(video_pre["txt_len"]) + int(video_pre["cond_len"])
        cond_len = int(video_pre["cond_len"])
        action_len = int(latents_action.shape[1])
        action_attention_mask = {
            "joint_double": full_attention_mask["joint_double"][:, prefix_len : prefix_len + action_len, : prefix_len + action_len],
            "image_double": full_attention_mask["image_double"][:, cond_len : cond_len + action_len, : cond_len + action_len],
            "single": full_attention_mask["single"][:, prefix_len : prefix_len + action_len, : prefix_len + action_len],
        }
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.expand(batch_size).to(dtype=latents_action.dtype, device=self.device)
            action_pre = self.action_expert.pre_dit(
                action_tokens=latents_action,
                timestep=self._scheduler_timestep_to_unit(timestep_action, self.infer_action_scheduler),
            )
            action_tokens = self.mot.forward_action_with_video_cache(
                action_tokens=action_pre["tokens"],
                action_freqs=None,
                action_t_mod=action_pre["t_mod"],
                action_context_payload={"ids": action_pre["ids"]},
                video_kv_cache=video_kv_cache,
                attention_mask=action_attention_mask,
                video_seq_len=prefix_len,
            )
            pred_action = self.action_expert.post_dit(action_tokens, action_pre)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}

    @torch.no_grad()
    def infer_video_ovis_u1(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        from .ovis_u1_video_expert import OvisU1VideoExpert

        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`input_image` spatial dims must be multiples of 16, got HxW=({height},{width})")
        text_hidden, text_mask = self._prepare_ovis_u1_infer_text(prompt, input_image, context, context_mask)
        if self.proprio_encoder is not None or proprio is not None:
            text_hidden, text_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden,
                context_mask=text_mask,
                proprio=proprio,
                source="Ovis-U1 video inference",
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        ref_tokens, ref_img_ids = self._encode_ovis_u1_image_tokens(input_image, time_value=1.0)
        batch_size = int(ref_tokens.shape[0])
        latent_h = 2 * ((int(height) + 15) // 16)
        latent_w = 2 * ((int(width) + 15) // 16)
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (batch_size, (latent_h // 2) * (latent_w // 2), int(self.video_expert.transformer.in_channels)),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        target_img_ids = OvisU1VideoExpert.build_img_ids(
            batch_size=batch_size,
            token_height=latent_h // 2,
            token_width=latent_w // 2,
            time_value=0.0,
            device=self.device,
            dtype=self.torch_dtype,
        )
        infer_timesteps, infer_deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep_video = step_t.expand(batch_size).to(dtype=latents_video.dtype, device=self.device)
            video_pre = self.video_expert.pre_dit(
                x=latents_video,
                timestep=self._scheduler_timestep_to_unit(timestep_video, self.infer_video_scheduler),
                context=text_hidden,
                context_mask=text_mask,
                ref_image_hidden_states=ref_tokens,
                target_img_ids=target_img_ids,
                ref_img_ids=ref_img_ids,
            )
            pred_video = self.video_expert.native_forward_from_pre_state(video_pre)
            latents_video = self.infer_video_scheduler.step(pred_video, step_delta, latents_video)

        image = self._decode_ovis_u1_image_tokens(latents_video, height=height, width=width)
        return {"image": image[0].detach().cpu()}

    @torch.no_grad()
    def infer_video_omnigen2(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")

        text_hidden, text_mask = self._prepare_omnigen2_infer_text(prompt, context, context_mask)
        if self.proprio_encoder is not None or proprio is not None:
            text_hidden, text_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden,
                context_mask=text_mask,
                proprio=proprio,
                source="OmniGen2 video inference",
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        ref_latent = self._encode_omnigen2_image_latents(input_image)
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            ref_latent.shape,
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        video_only_mot = MoT({"video": self.video_expert}, mot_checkpoint_mixed_attn=False).to(self.device)
        infer_timesteps, infer_deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep_video = step_t.expand(ref_latent.shape[0]).to(dtype=latents_video.dtype, device=self.device)
            video_pre = self.video_expert.pre_dit(
                x=latents_video,
                timestep=timestep_video,
                context=text_hidden,
                context_mask=text_mask,
                ref_image_hidden_states=ref_latent,
            )
            attention_mask = self._build_mot_attention_mask_omnigen2(
                encoder_seq_lengths=video_pre["encoder_seq_lengths"],
                seq_lengths=video_pre["seq_lengths"],
                max_video_seq_len=video_pre["tokens"].shape[1],
                action_seq_len=0,
                l_effective_ref_img_len=video_pre["l_effective_ref_img_len"],
                device=video_pre["tokens"].device,
            )
            tokens_out = video_only_mot(
                embeds_all={"video": video_pre["tokens"]},
                attention_mask=attention_mask,
                freqs_all={"video": video_pre["freqs"]},
                context_all={"video": None},
                t_mod_all={"video": video_pre["t_mod"]},
            )
            pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
            latents_video = self.infer_video_scheduler.step(pred_video, step_delta, latents_video)

        image = self._decode_omnigen2_image_latents(latents_video)
        return {"image": image[0].detach().cpu()}

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        if self.stack == "ovis_u1":
            if action_horizon is None:
                raise ValueError("`action_horizon` is required for Ovis-U1 action inference.")
            action_out = self.infer_action_ovis_u1(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
            video_out = self.infer_video_ovis_u1(
                prompt=prompt,
                input_image=input_image,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
            image = video_out["image"].detach().float().clamp(-1, 1)
            image = ((image + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            return {
                "action": action_out["action"],
                "video": [Image.fromarray(image)],
            }
        if self.stack == "omnigen2":
            if action_horizon is None:
                raise ValueError("`action_horizon` is required for OmniGen2 action inference.")
            action_out = self.infer_action_omnigen2(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
            video_out = self.infer_video_omnigen2(
                prompt=prompt,
                input_image=input_image,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
            image = video_out["image"].detach().float().clamp(-1, 1)
            image = ((image + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            return {
                "action": action_out["action"],
                "video": [Image.fromarray(image)],
            }
        if self.stack == "dim":
            if action_horizon is None:
                raise ValueError("`action_horizon` is required for DIM action inference.")
            return self.infer_dim_separate(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
        if self.stack == "flux2":
            if action_horizon is None:
                raise ValueError("`action_horizon` is required for FLUX.2 action inference.")
            # Both heads must consume the same pre-write MemoryBank snapshot.
            # Video inference is read-only; action inference writes the current
            # observation exactly once after producing its action chunk.
            video_out = self.infer_video_flux2(
                prompt=prompt,
                input_image=input_image,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
            action_out = self.infer_action_flux2(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
            image = video_out["image"].detach().float().clamp(-1, 1)
            image = ((image + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            return {"action": action_out["action"], "video": [Image.fromarray(image)]}

        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        if bool(getattr(self, "save_lora_merged", False)):
            from .lora import lora_merged_state_dict

            mot_state = lora_merged_state_dict(self.mot)
            checkpoint_format = "lora_merged"
        elif bool(getattr(self, "save_trainable_only", False)):
            trainable_names = {name for name, param in self.mot.named_parameters() if param.requires_grad}
            mot_state = {
                key: value
                for key, value in self.mot.state_dict().items()
                if key in trainable_names
            }
            checkpoint_format = "trainable_only"
        else:
            mot_state = self.mot.state_dict()
            checkpoint_format = "full"
        payload = {
            "mot": mot_state,
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        edit_memory_state = {
            name: value
            for name, value in self.state_dict().items()
            if name.startswith("edit_memory_")
        }
        if edit_memory_state:
            payload["edit_memory"] = edit_memory_state
            payload["edit_memory_contract"] = self._edit_memory_checkpoint_contract()
            cache = getattr(self, "edit_memory_vlm_count_cache", None)
            if cache is not None and cache.fingerprint is not None:
                payload["edit_memory_vlm_count_supervision"] = {
                    "cache_fingerprint": cache.fingerprint,
                    "records": len(cache),
                    "metadata": dict(cache.metadata),
                }
        if checkpoint_format != "full":
            payload["checkpoint_format"] = checkpoint_format
            payload["save_trainable_only"] = bool(getattr(self, "save_trainable_only", False))
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None, *, strict_model_weights: bool = False):
        payload = torch.load(path, map_location="cpu")
        logger.info("Loading ImageWAM checkpoint from %s with payload keys=%s step=%s", path, sorted(payload.keys()), payload.get("step"))
        if "mot" in payload:
            mot_state = payload["mot"]
            if self.stack == "flux2":
                from .lora import merge_lora_state_dict_to_plain, remap_plain_linear_keys_to_lora_base

                mot_state = merge_lora_state_dict_to_plain(mot_state)
                mot_state = remap_plain_linear_keys_to_lora_base(self.mot, mot_state)
            load_result = self.mot.load_state_dict(mot_state, strict=False)
            missing_keys = list(load_result.missing_keys)
            unexpected_keys = list(load_result.unexpected_keys)
            logger.info(
                "Loaded MoT weights from checkpoint: missing_keys=%d unexpected_keys=%d",
                len(missing_keys),
                len(unexpected_keys),
            )
            if missing_keys:
                logger.warning("First missing MoT keys: %s", missing_keys[:20])
            if unexpected_keys:
                logger.warning("First unexpected MoT keys: %s", unexpected_keys[:20])
            if strict_model_weights and (missing_keys or unexpected_keys):
                raise RuntimeError(
                    "Strict MoT checkpoint loading failed: "
                    f"missing={missing_keys[:20]} unexpected={unexpected_keys[:20]}"
                )
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            load_result = self.video_expert.load_state_dict(payload["dit"], strict=False)
            logger.info(
                "Loaded legacy video expert weights: missing_keys=%d unexpected_keys=%d",
                len(load_result.missing_keys),
                len(load_result.unexpected_keys),
            )
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
                logger.info("Loaded proprio_encoder weights from checkpoint.")
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        if "edit_memory" in payload:
            saved_contract = payload.get("edit_memory_contract")
            expected_contract = self._edit_memory_checkpoint_contract()
            if not isinstance(saved_contract, dict):
                raise RuntimeError("Stage2 checkpoint is missing the required `edit_memory_contract` mapping.")
            if saved_contract != expected_contract:
                all_keys = sorted(set(saved_contract) | set(expected_contract))
                mismatch = {
                    key: {"checkpoint": saved_contract.get(key), "active": expected_contract.get(key)}
                    for key in all_keys
                    if saved_contract.get(key) != expected_contract.get(key)
                }
                raise RuntimeError(f"Edit-memory train/infer contract mismatch: {mismatch}")
            edit_memory_state = payload["edit_memory"]
            if not isinstance(edit_memory_state, dict):
                raise TypeError("Checkpoint `edit_memory` payload must be a state-dict mapping.")
            current_state = self.state_dict()
            expected_keys = {name for name in current_state if name.startswith("edit_memory_")}
            provided_keys = set(edit_memory_state)
            missing = sorted(expected_keys - provided_keys)
            unexpected = sorted(provided_keys - expected_keys)
            shape_mismatch = sorted(
                name
                for name in expected_keys & provided_keys
                if tuple(current_state[name].shape) != tuple(edit_memory_state[name].shape)
            )
            if missing or unexpected or shape_mismatch:
                raise RuntimeError(
                    "Edit-memory checkpoint is incompatible with the active memory config: "
                    f"missing={missing[:20]} unexpected={unexpected[:20]} "
                    f"shape_mismatch={shape_mismatch[:20]}"
                )
            self.load_state_dict(edit_memory_state, strict=False)
            logger.info("Loaded and exactly validated %d edit-memory tensors.", len(edit_memory_state))
            saved_supervision = payload.get("edit_memory_vlm_count_supervision")
            cache = getattr(self, "edit_memory_vlm_count_cache", None)
            if cache is not None and cache.fingerprint is not None and isinstance(saved_supervision, dict):
                saved_fingerprint = saved_supervision.get("cache_fingerprint")
                if saved_fingerprint != cache.fingerprint:
                    raise RuntimeError(
                        "VLM semantic cache fingerprint mismatch while resuming training: "
                        f"checkpoint={saved_fingerprint}, active={cache.fingerprint}."
                    )
        elif getattr(self, "edit_memory_enabled", False):
            logger.warning("Checkpoint has no edit-memory weights; memory parameters remain initialized.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def _edit_memory_checkpoint_contract(self) -> dict[str, Any]:
        config = getattr(self, "edit_memory_config", {}) or {}
        bank = config.get("bank", {})
        span = config.get("adaptive_span", {})
        gist = config.get("adaptive_gist", {})
        replan_steps = os.environ.get("IMAGEWAM_REPLAN_STEPS")
        return {
            "version": str(config.get("version", "unknown")),
            "replan_steps": None if replan_steps in {None, ""} else int(replan_steps),
            "anchor_frames": int(bank.get("anchor_frames", 2)),
            "recent_frames": int(bank.get("recent_frames", 4)),
            "gist_tokens_per_frame": int(bank.get("gist_tokens_per_frame", 8)),
            "read_all_history": int(bank.get("max_entries", 0)) == 0,
            "adaptive_span_enabled": bool(span.get("enabled", False)),
            "span_min_frames": int(span.get("min_frames", 1)),
            "span_max_frames": int(span.get("max_frames", 64)),
            "span_gate_temperature": float(span.get("gate_temperature", 1.0)),
            "adaptive_gist_enabled": bool(gist.get("enabled", False)),
            "gist_min_tokens": int(gist.get("min_tokens", 1)),
            "gist_max_tokens": int(gist.get("max_tokens", 8)),
            "gist_gap_temperature": float(gist.get("gap_temperature", 0.5)),
            "gist_no_cut_gap": float(gist.get("no_cut_gap", 0.05)),
            "gist_target_tokens": float(gist.get("target_tokens", 4.0)),
            "gist_token_cost": float(gist.get("token_cost", 0.001)),
            "memory_write_head": getattr(self, "edit_memory_write_head", None) is not None,
            "memory_write_threshold": float(bank.get("write_threshold", 0.5)),
            "memory_head_uses_clean_prefix": True,
        }

    def apply_trainable_policy(self) -> None:
        """Refine trainer's default DiT-only policy for parameter-efficient modes."""
        if self.stack != "flux2":
            return
        video_expert = self.mot.mixtures["video"] if "video" in self.mot.mixtures else None
        action_expert = self.mot.mixtures["action"] if "action" in self.mot.mixtures else None
        if action_expert is not None:
            action_expert.train()
            action_expert.requires_grad_(True)
        if getattr(self, "edit_memory_enabled", False):
            for name, parameter in self.named_parameters():
                if name.startswith("edit_memory_"):
                    parameter.requires_grad = True
        if video_expert is None or not bool(getattr(video_expert, "flux2_lora_enabled", False)):
            return
        video_expert.train()
        for param in video_expert.parameters():
            param.requires_grad = False
        for name, param in video_expert.named_parameters():
            if ".lora_A" in name or ".lora_B" in name:
                param.requires_grad = True

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
