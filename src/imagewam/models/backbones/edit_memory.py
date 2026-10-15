from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class VLMSemanticCountCache:
    """Qwen3-VL count labels keyed by (episode_id, timestep)."""

    FORMAT = "samta_semantic_change_curve_v1"

    def __init__(self, path: str | Path | None, max_tokens: int = 8) -> None:
        self.max_tokens = int(max_tokens)
        path_text = "" if path is None else str(path).strip()
        self.path = None if path_text.lower() in {"", "null", "none"} else Path(path_text)
        self.targets: dict[tuple[int, int], tuple[torch.Tensor, int]] = {}
        self.metadata: dict[str, Any] = {}
        self._fingerprint: str | None = None
        if self.path is None:
            return
        if not self.path.is_file():
            raise FileNotFoundError(f"VLM semantic count cache does not exist: {self.path}")
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("format") != self.FORMAT:
            raise ValueError(
                f"VLM count cache must use format={self.FORMAT!r}; got "
                f"{None if not isinstance(payload, dict) else payload.get('format')!r}."
            )
        if int(payload.get("k_max", -1)) != self.max_tokens:
            raise ValueError(
                f"VLM count cache k_max={payload.get('k_max')} does not match max_tokens={self.max_tokens}."
            )
        required_metadata = (
            "teacher_model",
            "teacher_prompt_hash",
            "source_checkpoint_sha256",
            "preprocess_hash",
            "semantic_scorer_hash",
        )
        missing = [key for key in required_metadata if key not in payload]
        if missing:
            raise ValueError(f"VLM count cache is missing metadata: {missing}")
        self.metadata = {key: payload[key] for key in required_metadata}
        digest = hashlib.sha256()
        digest.update(json.dumps(self.metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        for record in payload.get("records", []):
            key = (int(record["episode_id"]), int(record["timestep"]))
            if key in self.targets:
                raise ValueError(f"VLM count cache contains duplicate key {key}.")
            if "teacher_probabilities" not in record or "teacher_count" not in record:
                raise ValueError(
                    f"VLM count cache record {key} must contain teacher_probabilities and teacher_count."
                )
            probabilities = torch.as_tensor(
                record["teacher_probabilities"], dtype=torch.float32
            ).reshape(-1)
            teacher_count = int(record["teacher_count"])
            if (
                probabilities.numel() != self.max_tokens + 1
                or not torch.isfinite(probabilities).all()
                or bool((probabilities < 0).any().item())
            ):
                raise ValueError(
                    f"Invalid VLM teacher probabilities for {key}: shape={tuple(probabilities.shape)}"
                )
            if not torch.isclose(
                probabilities.sum(), probabilities.new_tensor(1.0), atol=1.0e-5, rtol=1.0e-5
            ):
                raise ValueError(f"VLM teacher probabilities must sum to one for {key}.")
            if not 0 <= teacher_count <= self.max_tokens:
                raise ValueError(f"Invalid VLM teacher_count={teacher_count} for {key}.")
            if teacher_count != int(probabilities.argmax().item()):
                raise ValueError(
                    f"VLM teacher_count does not match teacher_probabilities argmax for {key}."
                )
            self.targets[key] = (probabilities.contiguous().clone(), teacher_count)
            digest.update(f"{key[0]}:{key[1]}:".encode("ascii"))
            digest.update(probabilities.contiguous().numpy().tobytes())
            digest.update(f":{teacher_count}".encode("ascii"))
        if not self.targets:
            raise ValueError("VLM count cache contains no records.")
        self._fingerprint = digest.hexdigest()

    def __len__(self) -> int:
        return len(self.targets)

    @property
    def fingerprint(self) -> str | None:
        return self._fingerprint

    def lookup(
        self,
        episode_ids: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = episode_ids.detach().reshape(-1).cpu().tolist()
        steps = timesteps.detach().reshape(-1).cpu().tolist()
        if len(ids) != len(steps):
            raise ValueError("episode_ids and timesteps must contain the same number of values.")
        probabilities = torch.zeros(
            len(ids), self.max_tokens + 1, device=device, dtype=torch.float32
        )
        probabilities[:, 0] = 1.0
        teacher_counts = torch.zeros(len(ids), device=device, dtype=torch.long)
        found = torch.zeros(len(ids), device=device, dtype=torch.bool)
        for index, (episode_id, timestep) in enumerate(zip(ids, steps)):
            target = self.targets.get((int(episode_id), int(timestep)))
            if target is not None:
                value, teacher_count = target
                probabilities[index] = value.to(device=device)
                teacher_counts[index] = teacher_count
                found[index] = True
        return probabilities, teacher_counts, found

    def class_weights(
        self,
        *,
        max_weight: float,
    ) -> torch.Tensor:
        counts = torch.tensor(
            [teacher_count for _, teacher_count in self.targets.values()], dtype=torch.long
        )
        histogram = torch.bincount(counts, minlength=self.max_tokens + 1).float()
        weights = (histogram.sum() / ((self.max_tokens + 1) * histogram.clamp_min(1.0))).sqrt()
        weights = weights / weights.mean().clamp_min(1.0e-8)
        return weights.clamp(min=1.0 / float(max_weight), max=float(max_weight))


def vlm_supervised_adaptive_k_loss(
    decision: dict[str, torch.Tensor],
    teacher_probabilities: torch.Tensor,
    teacher_counts: torch.Tensor,
    valid: torch.Tensor,
    *,
    count_weight: float,
    class_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Fit the unified K=0..K allocator directly to the cached VLM distribution."""
    probabilities = decision["probabilities"].float()
    targets = teacher_probabilities.detach().float()
    teacher_count = teacher_counts.detach().long().reshape(-1)
    if targets.shape != probabilities.shape:
        raise ValueError(
            "VLM distribution does not match unified Adaptive-k classes: "
            f"{tuple(targets.shape)} != {tuple(probabilities.shape)}"
        )
    if teacher_count.shape[0] != probabilities.shape[0]:
        raise ValueError("VLM probabilities and counts must have the same batch size.")
    if not torch.isfinite(targets).all() or bool((targets < 0).any().item()):
        raise ValueError("teacher_probabilities must be finite and non-negative.")
    if not torch.allclose(
        targets.sum(dim=1),
        torch.ones(targets.shape[0], device=targets.device),
        atol=1.0e-5,
        rtol=1.0e-5,
    ):
        raise ValueError("Each VLM teacher distribution must sum to one.")
    valid = valid.to(device=probabilities.device, dtype=torch.bool).reshape(-1)
    denom = valid.float().sum().clamp_min(1.0)
    log_probabilities = probabilities.clamp_min(1.0e-8).log()
    per_sample = -(targets * log_probabilities).sum(dim=1)
    sample_weights = valid.float()
    if class_weights is not None:
        weights = class_weights.to(device=probabilities.device, dtype=probabilities.dtype)
        sample_weights = sample_weights * weights[teacher_count]
    loss = (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)
    total = float(count_weight) * loss
    agreement = (
        (decision["hard_count"].long() == teacher_count).float() * valid.float()
    ).sum() / denom
    expected_mae = (
        (decision["expected_count"].float() - teacher_count.float()).abs() * valid.float()
    ).sum() / denom
    zero_fraction = ((teacher_count == 0).float() * valid.float()).sum() / denom
    return {
        "loss": total,
        "count_loss": loss,
        "teacher_count": teacher_count,
        "teacher_probabilities": targets,
        "original_count": teacher_count,
        "agreement": agreement,
        "expected_mae": expected_mae,
        "zero_mapped_frac": zero_fraction,
        "valid_count": valid.float().sum(),
    }


def teacher_memory_probability(
    step: int,
    *,
    teacher_only_until_step: int,
    transition_start_step: int,
    transition_end_step: int,
) -> float:
    """EventVLA-style teacher-to-predict probability for a zero-based optimizer step."""
    step = int(step)
    teacher_only_until_step = int(teacher_only_until_step)
    transition_start_step = int(transition_start_step)
    transition_end_step = int(transition_end_step)
    if transition_start_step != teacher_only_until_step + 1:
        raise ValueError("Memory curriculum transition must start immediately after teacher-only steps.")
    if transition_end_step < transition_start_step:
        raise ValueError("Memory curriculum transition_end_step must be >= transition_start_step.")
    if step <= teacher_only_until_step:
        return 1.0
    if step > transition_end_step:
        return 0.0
    width = max(transition_end_step - transition_start_step, 1)
    return float(transition_end_step - step) / float(width)


def mix_teacher_predicted_gist_decision(
    decision: dict[str, torch.Tensor],
    *,
    teacher_count: torch.Tensor,
    teacher_active: torch.Tensor,
    teacher_probability: float,
    random_values: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Choose teacher or predicted hard N per sample while keeping one shared soft-k head."""
    predicted_count = decision["hard_count"].long()
    teacher_count = teacher_count.to(device=predicted_count.device, dtype=torch.long).reshape(-1)
    teacher_active = teacher_active.to(device=predicted_count.device, dtype=torch.bool).reshape(-1)
    if teacher_count.shape != predicted_count.shape or teacher_active.shape != predicted_count.shape:
        raise ValueError("Teacher counts, active mask, and predicted counts must have identical shape.")
    probability = min(max(float(teacher_probability), 0.0), 1.0)
    if random_values is None:
        random_values = torch.rand(predicted_count.shape, device=predicted_count.device)
    else:
        random_values = random_values.to(device=predicted_count.device, dtype=torch.float32).reshape(-1)
        if random_values.shape != predicted_count.shape:
            raise ValueError("random_values must have one value per sample.")
    use_teacher = teacher_active & (random_values < probability)
    write_count = torch.where(use_teacher, teacher_count, predicted_count)

    sorted_indices = decision["sorted_indices"]
    rank = torch.empty_like(sorted_indices)
    rank.scatter_(
        1,
        sorted_indices,
        torch.arange(sorted_indices.shape[1], device=sorted_indices.device).expand_as(sorted_indices),
    )
    hard_keep = rank < write_count[:, None]
    predicted_straight_through = decision["straight_through_keep"]
    write_keep = torch.where(
        use_teacher[:, None],
        hard_keep.to(dtype=predicted_straight_through.dtype),
        predicted_straight_through,
    )
    mixed = dict(decision)
    mixed["hard_count"] = write_count
    mixed["hard_keep"] = hard_keep
    mixed["straight_through_keep"] = write_keep
    mixed["teacher_memory_mask"] = use_teacher
    mixed["predicted_hard_count"] = predicted_count
    return mixed, use_teacher


def _store_dtype_from_name(name: str | torch.dtype | None) -> torch.dtype | None:
    if isinstance(name, torch.dtype):
        return name
    if name is None:
        return None
    key = str(name).strip().lower()
    if key in {"", "none", "null", "auto"}:
        return None
    if key in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half", "torch.float16"}:
        return torch.float16
    if key in {"fp32", "float32", "float", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported edit-memory store_dtype={name!r}.")


def gather_by_index(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"`tokens` must be [B,N,D], got {tuple(tokens.shape)}")
    if indices.ndim != 2:
        raise ValueError(f"`indices` must be [B,K], got {tuple(indices.shape)}")
    if tokens.shape[0] != indices.shape[0]:
        raise ValueError(
            f"`tokens` and `indices` batch sizes must match, got {tokens.shape[0]} and {indices.shape[0]}"
        )
    gather_index = indices.to(device=tokens.device, dtype=torch.long).unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    return torch.gather(tokens, dim=1, index=gather_index)


class AdaptiveGistAllocator(nn.Module):
    """Rank learned gist candidates and predict a unified K=0..max_tokens distribution."""

    def __init__(
        self,
        input_dim: int,
        min_tokens: int = 1,
        max_tokens: int = 8,
        hidden_dim: int = 256,
        gap_temperature: float = 0.5,
        token_cost: float = 0.001,
        target_tokens: float = 4.0,
        no_cut_gap: float = 0.05,
    ) -> None:
        super().__init__()
        self.min_tokens = int(min_tokens)
        self.max_tokens = int(max_tokens)
        if self.min_tokens not in {0, 1} or self.max_tokens < max(self.min_tokens, 1):
            raise ValueError("Adaptive gist allocation requires min_tokens in {0,1} and max_tokens >= 1.")
        self.gap_temperature = float(gap_temperature)
        self.token_cost = float(token_cost)
        self.target_tokens = float(target_tokens)
        self.query = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(input_dim)),
        )
        self.candidates = nn.Parameter(torch.empty(self.max_tokens, int(input_dim)))
        self.count_head = None
        if self.min_tokens == 0:
            self.count_head = nn.Sequential(
                nn.LayerNorm(int(input_dim)),
                nn.Linear(int(input_dim), int(hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(hidden_dim), self.max_tokens + 1),
            )
        self.register_buffer("no_cut_gap", torch.tensor(float(no_cut_gap)), persistent=True)
        nn.init.normal_(self.candidates, mean=0.0, std=0.02)

    def forward(self, frame_summary: torch.Tensor) -> dict[str, torch.Tensor]:
        if frame_summary.ndim != 2:
            raise ValueError(f"`frame_summary` must be [B,D], got {tuple(frame_summary.shape)}")
        query_dtype = self.candidates.dtype
        query = F.normalize(self.query(frame_summary.to(dtype=query_dtype)).float(), dim=-1)
        candidates = F.normalize(self.candidates.float(), dim=-1)
        scores = query @ candidates.transpose(0, 1)
        sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True)
        if self.count_head is not None:
            count_logits = self.count_head(frame_summary.to(dtype=query_dtype)).float()
            probabilities = torch.softmax(
                count_logits / max(self.gap_temperature, 1e-4), dim=-1
            )
            count_values = torch.arange(
                0, self.max_tokens + 1, device=probabilities.device, dtype=probabilities.dtype
            )
            hard_count = count_logits.argmax(dim=-1)
            keep_by_rank = torch.flip(
                torch.cumsum(torch.flip(probabilities[:, 1:], dims=[1]), dim=1), dims=[1]
            )
        else:
            if self.max_tokens == 1:
                count_logits = self.no_cut_gap.expand(frame_summary.shape[0], 1)
            else:
                adjacent_gaps = sorted_scores[:, :-1] - sorted_scores[:, 1:]
                count_logits = torch.cat(
                    [adjacent_gaps, self.no_cut_gap.expand(frame_summary.shape[0], 1)], dim=-1
                )
            probabilities = torch.softmax(
                count_logits / max(self.gap_temperature, 1e-4), dim=-1
            )
            count_values = torch.arange(
                1, self.max_tokens + 1, device=probabilities.device, dtype=probabilities.dtype
            )
            hard_count = count_logits.argmax(dim=-1) + 1
            keep_by_rank = torch.flip(
                torch.cumsum(torch.flip(probabilities, dims=[1]), dim=1), dims=[1]
            )
        expected_count = (probabilities * count_values).sum(dim=-1)
        rank = torch.empty_like(sorted_indices)
        rank.scatter_(
            1,
            sorted_indices,
            torch.arange(self.max_tokens, device=scores.device).expand_as(sorted_indices),
        )
        hard_keep = rank < hard_count[:, None]
        soft_keep = torch.gather(keep_by_rank, 1, rank)
        straight_through_keep = hard_keep.to(soft_keep.dtype) + soft_keep - soft_keep.detach()
        result = {
            "scores": scores,
            "sorted_indices": sorted_indices,
            "gap_logits": count_logits,
            "count_logits": count_logits,
            "probabilities": probabilities,
            "expected_count": expected_count,
            "hard_count": hard_count,
            "hard_keep": hard_keep,
            "soft_keep": soft_keep,
            "straight_through_keep": straight_through_keep,
            "regularizer": self.token_cost * F.relu(expected_count.mean() - self.target_tokens).square(),
        }
        if self.min_tokens == 0:
            write_probability = (1.0 - probabilities[:, 0]).clamp(1.0e-6, 1.0 - 1.0e-6)
            result["write_decision"] = {
                "logits": torch.logit(write_probability),
                "probabilities": write_probability,
                "hard_write": hard_count > 0,
            }
        return result


class DynamicHistorySpanController(nn.Module):
    """Input-conditioned span using the Adaptive Attention Span parameterization."""

    def __init__(
        self,
        input_dim: int,
        min_frames: int = 1,
        max_frames: int = 64,
        hidden_dim: int = 256,
        gate_temperature: float = 1.0,
        span_cost: float = 0.0001,
    ) -> None:
        super().__init__()
        self.min_frames = int(min_frames)
        self.max_frames = int(max_frames)
        if self.min_frames < 0 or self.max_frames < self.min_frames:
            raise ValueError("Invalid adaptive history span bounds.")
        self.gate_temperature = float(gate_temperature)
        self.span_cost = float(span_cost)
        self.predictor = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.constant_(self.predictor[-1].bias, 4.0)

    def forward(self, frame_summary: torch.Tensor) -> dict[str, torch.Tensor]:
        predictor_dtype = next(self.predictor.parameters()).dtype
        logits = self.predictor(frame_summary.to(dtype=predictor_dtype)).float().squeeze(-1)
        soft_span = self.min_frames + (self.max_frames - self.min_frames) * torch.sigmoid(logits)
        hard_span = soft_span.round().to(torch.long).clamp(self.min_frames, self.max_frames)
        return {
            "logits": logits,
            "soft_span": soft_span,
            "hard_span": hard_span,
            "regularizer": self.span_cost * (soft_span / max(self.max_frames, 1)).mean(),
        }


def indices_to_xy(indices: torch.Tensor, grid_hw: tuple[int, int] | None) -> torch.Tensor:
    if indices.ndim != 2:
        raise ValueError(f"`indices` must be [B,K], got {tuple(indices.shape)}")
    if grid_hw is None:
        return torch.zeros(*indices.shape, 2, device=indices.device, dtype=torch.float32)
    height, width = int(grid_hw[0]), int(grid_hw[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"`grid_hw` must be positive, got {grid_hw}.")
    idx = indices.to(dtype=torch.long)
    y = torch.div(idx, width, rounding_mode="floor")
    x = idx.remainder(width)
    return torch.stack([x, y], dim=-1).to(dtype=torch.float32)


class PerLayerTopKEditKVSelector(nn.Module):
    """Select native per-layer FLUX.2 edit K/V tokens for MemoryWAM-style cache memory."""

    def __init__(
        self,
        topk_tokens_per_layer: int | str | None,
        select_image_tokens_only: bool = True,
        detach_stored_memory: bool = True,
        store_dtype: str | torch.dtype | None = "bfloat16",
    ) -> None:
        super().__init__()
        self.topk_tokens_per_layer = self._parse_topk(topk_tokens_per_layer)
        self.select_image_tokens_only = bool(select_image_tokens_only)
        self.detach_stored_memory = bool(detach_stored_memory)
        self.store_dtype = _store_dtype_from_name(store_dtype)

    @staticmethod
    def _parse_topk(value: int | str | None) -> int | None:
        if value is None:
            return None
        key = str(value).strip().lower()
        if key in {"", "none", "null", "all", "full", "-1"}:
            return None
        return int(value)

    @staticmethod
    def _teacher_for_layer(
        per_layer_teacher_importance: dict[str, Any] | None,
        layer_key: str,
        layer_idx: int,
    ) -> torch.Tensor | None:
        if per_layer_teacher_importance is None:
            return None
        by_type = per_layer_teacher_importance.get(layer_key)
        if by_type is None:
            return None
        if isinstance(by_type, dict):
            return by_type.get(layer_idx)
        if isinstance(by_type, (list, tuple)) and layer_idx < len(by_type):
            return by_type[layer_idx]
        return None

    @staticmethod
    def _fit_teacher_shape(importance: torch.Tensor, batch_size: int, img_len: int, device: torch.device) -> torch.Tensor:
        if importance.ndim == 1:
            importance = importance.unsqueeze(0)
        if importance.ndim != 2:
            raise ValueError(f"Teacher importance must be [B,N] or [N], got {tuple(importance.shape)}")
        if importance.shape[0] == 1 and batch_size > 1:
            importance = importance.expand(batch_size, -1)
        if importance.shape[0] != batch_size:
            raise ValueError(
                f"Teacher importance batch size must be {batch_size}, got {importance.shape[0]}"
            )
        importance = importance.to(device=device, dtype=torch.float32)
        if importance.shape[1] > img_len:
            importance = importance[:, :img_len]
        elif importance.shape[1] < img_len:
            pad = importance.new_zeros(batch_size, img_len - importance.shape[1])
            importance = torch.cat([importance, pad], dim=1)
        return torch.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    def _select_one_layer(
        self,
        cache_l: dict[str, torch.Tensor],
        teacher_importance_l: torch.Tensor | None,
        *,
        txt_len: int,
        img_len: int,
        grid_hw: tuple[int, int] | None,
        layer_idx: int,
    ) -> dict[str, Any]:
        k_l = cache_l["k"]
        v_l = cache_l["v"]
        if k_l.ndim != 3 or v_l.ndim != 3:
            raise ValueError(f"Layer cache K/V must be [B,N,D], got {tuple(k_l.shape)} and {tuple(v_l.shape)}")
        if k_l.shape[:2] != v_l.shape[:2] or k_l.shape[0] != v_l.shape[0]:
            raise ValueError(f"Layer cache K/V shapes are incompatible: {tuple(k_l.shape)} vs {tuple(v_l.shape)}")

        if self.select_image_tokens_only:
            start = int(txt_len)
            stop = start + int(img_len)
        else:
            start = 0
            stop = int(k_l.shape[1])
        if start < 0 or stop < start or stop > int(k_l.shape[1]):
            raise ValueError(
                f"Invalid edit-memory token slice [{start}, {stop}) for layer length {k_l.shape[1]}."
            )

        img_k = k_l[:, start:stop]
        img_v = v_l[:, start:stop]
        select_len = int(img_k.shape[1])
        keep_full_order = self.topk_tokens_per_layer is None
        topk = select_len if keep_full_order else min(max(int(self.topk_tokens_per_layer), 0), select_len)
        batch_size = int(img_k.shape[0])
        if topk == 0:
            empty_idx = torch.empty(batch_size, 0, device=img_k.device, dtype=torch.long)
            return {
                "k": img_k[:, :0],
                "v": img_v[:, :0],
                "topk_indices": empty_idx,
                "topk_scores": img_k.new_empty(batch_size, 0, dtype=torch.float32),
                "spatial_xy": img_k.new_empty(batch_size, 0, 2, dtype=torch.float32),
                "layer_idx": int(layer_idx),
            }

        if teacher_importance_l is not None:
            score = self._fit_teacher_shape(teacher_importance_l, batch_size, select_len, img_k.device)
        else:
            score = img_v.float().norm(dim=-1)
            score = torch.softmax(score, dim=-1)

        if keep_full_order:
            topk_indices = torch.arange(select_len, device=img_k.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
            topk_scores = score
        else:
            topk_scores, topk_indices = torch.topk(score, k=topk, dim=-1)
        selected_k = gather_by_index(img_k, topk_indices)
        selected_v = gather_by_index(img_v, topk_indices)
        if self.detach_stored_memory:
            selected_k = selected_k.detach()
            selected_v = selected_v.detach()
            topk_indices = topk_indices.detach()
            topk_scores = topk_scores.detach()
        if self.store_dtype is not None:
            selected_k = selected_k.to(dtype=self.store_dtype)
            selected_v = selected_v.to(dtype=self.store_dtype)
        return {
            "k": selected_k,
            "v": selected_v,
            "topk_indices": topk_indices,
            "topk_scores": topk_scores.to(dtype=torch.float32),
            "spatial_xy": indices_to_xy(topk_indices, grid_hw),
            "layer_idx": int(layer_idx),
        }

    def forward(
        self,
        video_kv_cache: dict[str, Any],
        per_layer_teacher_importance: dict[str, Any] | None,
        txt_len: int,
        img_len: int,
        grid_hw: tuple[int, int] | None,
    ) -> dict[str, list[dict[str, Any]]]:
        selected: dict[str, list[dict[str, Any]]] = {"double": [], "single": []}
        for layer_key in ("double", "single"):
            for layer_idx, cache_l in enumerate(video_kv_cache[layer_key]):
                teacher_l = self._teacher_for_layer(per_layer_teacher_importance, layer_key, layer_idx)
                selected[layer_key].append(
                    self._select_one_layer(
                        cache_l,
                        teacher_l,
                        txt_len=int(txt_len),
                        img_len=int(img_len),
                        grid_hw=grid_hw,
                        layer_idx=int(layer_idx),
                    )
                )
        return selected


class ActionAttentionCapture:
    """Aggregate action-to-image attention by layer without keeping full attention tensors."""

    def __init__(self) -> None:
        self.active = False
        self.condition_slice = (0, 0)
        self.condition_grid: tuple[int, int] | None = None
        self.prefix_len = 0
        self.action_len = 0
        self.metadata: dict[str, Any] = {}
        self._per_layer: dict[str, dict[int, torch.Tensor]] = {"double": {}, "single": {}}

    def configure(
        self,
        *,
        condition_slice: slice | tuple[int, int] | list[int],
        condition_grid: tuple[int, int] | None,
        prefix_len: int,
        action_len: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(condition_slice, slice):
            if condition_slice.start is None or condition_slice.stop is None:
                raise ValueError("`condition_slice` must have explicit start/stop.")
            start, stop = int(condition_slice.start), int(condition_slice.stop)
        else:
            start, stop = int(condition_slice[0]), int(condition_slice[1])
        if stop < start:
            raise ValueError(f"Invalid condition_slice=({start}, {stop}).")
        self.condition_slice = (start, stop)
        self.condition_grid = None if condition_grid is None else (int(condition_grid[0]), int(condition_grid[1]))
        self.prefix_len = int(prefix_len)
        self.action_len = int(action_len)
        self.metadata = dict(metadata or {})
        self._per_layer = {"double": {}, "single": {}}
        self.active = True

    def clear(self) -> None:
        self._per_layer = {"double": {}, "single": {}}
        self.active = False

    def deactivate(self) -> None:
        self.active = False

    def start_step(self, step_idx: int) -> None:
        del step_idx
        if self.active:
            self._per_layer = {"double": {}, "single": {}}

    def should_capture(self, layer_idx: int) -> bool:
        del layer_idx
        return bool(self.active)

    @staticmethod
    def _layer_key(block_type: str) -> str:
        if str(block_type).endswith("double") or "double" in str(block_type):
            return "double"
        if str(block_type).endswith("single") or "single" in str(block_type):
            return "single"
        return str(block_type)

    def update(
        self,
        attn_probs: torch.Tensor,
        *,
        layer_idx: int,
        block_type: str,
        prefix_len: int,
        action_len: int,
    ) -> None:
        if not self.active:
            return
        if attn_probs.ndim != 4:
            raise ValueError(f"`attn_probs` must be [B,H,Q,K], got {tuple(attn_probs.shape)}")
        key_len = int(attn_probs.shape[-1])
        prefix_len = int(prefix_len)
        action_len = int(action_len)
        memory_len = max(0, key_len - prefix_len - action_len)
        cond_start, cond_stop = self.condition_slice
        key_start = memory_len + cond_start
        key_stop = memory_len + cond_stop
        if key_start < 0 or key_stop > key_len or key_stop <= key_start:
            return

        # Average over heads and action queries immediately, storing only [B,img_len].
        importance = attn_probs[:, :, :, key_start:key_stop].float().mean(dim=(1, 2))
        denom = importance.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        importance = (importance / denom).detach()
        self._per_layer[self._layer_key(block_type)][int(layer_idx)] = importance

    def get_per_layer_teacher_importance(self) -> dict[str, dict[int, torch.Tensor]]:
        return {
            "double": dict(self._per_layer.get("double", {})),
            "single": dict(self._per_layer.get("single", {})),
        }


def memory_item_confidence(memory_item: dict[str, Any]) -> torch.Tensor:
    scores = []
    for layer_key in ("double", "single"):
        for layer in memory_item.get(layer_key, []):
            topk_scores = layer.get("topk_scores")
            if isinstance(topk_scores, torch.Tensor) and topk_scores.numel() > 0:
                scores.append(topk_scores.float().sum(dim=-1))
    if not scores:
        first_layer = (memory_item.get("double") or memory_item.get("single") or [None])[0]
        if first_layer is not None:
            ref = first_layer["k"]
            return torch.zeros(ref.shape[0], device=ref.device, dtype=torch.float32)
        return torch.tensor(0.0)
    return torch.stack(scores, dim=0).mean(dim=0)


def memory_tokens_per_layer(memory_item_or_cache: dict[str, Any] | None) -> int:
    if not memory_item_or_cache:
        return 0
    for layer_key in ("double", "single"):
        layers = memory_item_or_cache.get(layer_key) or []
        if layers:
            return int(layers[0]["k"].shape[1])
    return 0


def memory_kv_tokens_total(memory_item_or_cache: dict[str, Any] | None) -> int:
    if not memory_item_or_cache:
        return 0
    total = 0
    for layer_key in ("double", "single"):
        for layer in memory_item_or_cache.get(layer_key, []):
            total += int(layer["k"].shape[1])
    return total


def combine_memory_entries(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Concatenate memory entries in temporal order for native per-layer injection."""
    if not entries:
        return None
    first = entries[0]
    combined: dict[str, Any] = {"double": [], "single": []}
    for layer_key in ("double", "single"):
        num_layers = len(first.get(layer_key, []))
        for layer_idx in range(num_layers):
            layers = [entry[layer_key][layer_idx] for entry in entries]
            layer_combined = {
                "k": torch.cat([layer["k"] for layer in layers], dim=1),
                "v": torch.cat([layer["v"] for layer in layers], dim=1),
                "topk_indices": torch.cat([layer["topk_indices"] for layer in layers], dim=1),
                "topk_scores": torch.cat([layer["topk_scores"] for layer in layers], dim=1),
                "spatial_xy": torch.cat([layer["spatial_xy"] for layer in layers], dim=1),
                "layer_idx": int(layers[0].get("layer_idx", layer_idx)),
            }
            valid_chunks = []
            for layer in layers:
                valid = layer.get("memory_is_valid")
                if isinstance(valid, torch.Tensor):
                    valid_chunks.append(valid.to(device=layer["k"].device, dtype=torch.bool))
                else:
                    valid_chunks.append(
                        torch.ones(
                            layer["k"].shape[0],
                            layer["k"].shape[1],
                            device=layer["k"].device,
                            dtype=torch.bool,
                        )
                    )
            if valid_chunks:
                layer_combined["memory_is_valid"] = torch.cat(valid_chunks, dim=1)
            gate_chunks = []
            for layer in layers:
                gate = layer.get("memory_attention_gate")
                if isinstance(gate, torch.Tensor):
                    gate_chunks.append(gate.to(device=layer["k"].device, dtype=torch.float32))
                else:
                    gate_chunks.append(
                        torch.ones(
                            layer["k"].shape[0],
                            layer["k"].shape[1],
                            device=layer["k"].device,
                            dtype=torch.float32,
                        )
                    )
            layer_combined["memory_attention_gate"] = torch.cat(gate_chunks, dim=1)
            combined[layer_key].append(layer_combined)
    combined["num_entries"] = len(entries)
    combined["step_start"] = min(int(entry.get("step_start", 0)) for entry in entries)
    combined["step_end"] = max(int(entry.get("step_end", 0)) for entry in entries)
    combined["memory_kinds"] = [str(entry.get("memory_kind", "frame")) for entry in entries]
    return combined


def _concat_layer_parts(parts: list[dict[str, Any]]) -> dict[str, Any]:
    if not parts:
        raise ValueError("Cannot concatenate an empty layer list.")
    out = {
        "k": torch.cat([part["k"] for part in parts], dim=1),
        "v": torch.cat([part["v"] for part in parts], dim=1),
        "layer_idx": int(parts[0].get("layer_idx", 0)),
    }
    batch_size = int(out["k"].shape[0])
    token_counts = [int(part["k"].shape[1]) for part in parts]
    device = out["k"].device
    if all(isinstance(part.get("topk_indices"), torch.Tensor) for part in parts):
        out["topk_indices"] = torch.cat([part["topk_indices"].to(device=device) for part in parts], dim=1)
    else:
        chunks = []
        offset = 0
        for count in token_counts:
            chunks.append(torch.arange(offset, offset + count, device=device, dtype=torch.long).view(1, count).expand(batch_size, -1))
            offset += count
        out["topk_indices"] = torch.cat(chunks, dim=1)
    if all(isinstance(part.get("topk_scores"), torch.Tensor) for part in parts):
        out["topk_scores"] = torch.cat([part["topk_scores"].to(device=device, dtype=torch.float32) for part in parts], dim=1)
    else:
        out["topk_scores"] = torch.ones(batch_size, sum(token_counts), device=device, dtype=torch.float32)
    if all(isinstance(part.get("spatial_xy"), torch.Tensor) for part in parts):
        out["spatial_xy"] = torch.cat([part["spatial_xy"].to(device=device, dtype=torch.float32) for part in parts], dim=1)
    else:
        out["spatial_xy"] = torch.zeros(batch_size, sum(token_counts), 2, device=device, dtype=torch.float32)
    valid_parts = []
    for part, count in zip(parts, token_counts):
        valid = part.get("memory_is_valid")
        if isinstance(valid, torch.Tensor):
            valid_parts.append(valid.to(device=device, dtype=torch.bool))
        else:
            valid_parts.append(torch.ones(batch_size, count, device=device, dtype=torch.bool))
    out["memory_is_valid"] = torch.cat(valid_parts, dim=1)
    gate_parts = []
    for part, count in zip(parts, token_counts):
        gate = part.get("memory_attention_gate")
        if isinstance(gate, torch.Tensor):
            gate_parts.append(gate.to(device=device, dtype=torch.float32))
        else:
            gate_parts.append(torch.ones(batch_size, count, device=device, dtype=torch.float32))
    out["memory_attention_gate"] = torch.cat(gate_parts, dim=1)
    return out


def _materialize_hybrid_memory_entry(
    entry: dict[str, Any],
    *,
    include_frame: bool,
    include_gist: bool,
    memory_kind: str,
) -> dict[str, Any]:
    """Expose the K/V parts used by action attention as `double`/`single` layers."""
    out = dict(entry)
    out["memory_kind"] = str(memory_kind)
    for layer_key, gist_key in (("double", "gist_double"), ("single", "gist_single")):
        frame_layers = entry.get(layer_key, []) if include_frame else []
        gist_layers = entry.get(gist_key, []) if include_gist else []
        if gist_layers:
            layers = []
            num_layers = max(len(frame_layers), len(gist_layers))
            for layer_idx in range(num_layers):
                parts = []
                if layer_idx < len(frame_layers):
                    parts.append(frame_layers[layer_idx])
                if layer_idx < len(gist_layers):
                    parts.append(gist_layers[layer_idx])
                layers.append(_concat_layer_parts(parts))
            out[layer_key] = layers
        elif include_frame:
            out[layer_key] = list(frame_layers)
        else:
            out[layer_key] = []
    return out


def _memory_layers_have_valid_tokens(layers: list[dict[str, Any]]) -> bool:
    for layer in layers:
        valid = layer.get("memory_is_valid")
        if not isinstance(valid, torch.Tensor) or bool(valid.any().item()):
            return True
    return False


def _evict_frame_kv_from_entry(entry: dict[str, Any], *, gist_tokens_per_frame: int) -> dict[str, Any]:
    """Drop full visual frame K/V while keeping the frame's persistent gist K/V."""
    if entry.get("frame_kv_evicted", False):
        return entry
    has_learned_gist = _memory_layers_have_valid_tokens(
        list(entry.get("gist_double", [])) + list(entry.get("gist_single", []))
    )
    out = dict(entry)
    out["double"] = []
    out["single"] = []
    if not has_learned_gist:
        out["gist_double"] = []
        out["gist_single"] = []
    out["frame_kv_evicted"] = True
    out["memory_kind"] = "gist" if has_learned_gist else "evicted_no_gist"
    out["has_persistent_gist"] = has_learned_gist
    return out


def evict_nonpersistent_frame_kv(
    entries: list[dict[str, Any]],
    *,
    anchor_frames: int = 2,
    recent_frames: int = 4,
    gist_tokens_per_frame: int = 8,
) -> tuple[list[dict[str, Any]], int]:
    """Maintain MemoryWAM's persistent state: full K/V only for initial and recent frames.

    Every frame can keep its learned gist K/V forever. Full visual-token K/V is evicted
    as soon as the frame is neither an initial anchor nor inside the recent window.
    """
    if not entries:
        return [], 0
    n = len(entries)
    anchor_n = max(int(anchor_frames), 0)
    recent_n = max(int(recent_frames), 0)
    anchor_indices = set(range(min(anchor_n, n)))
    recent_start = max(0, n - recent_n)
    recent_indices = set(range(recent_start, n))

    evicted = 0
    maintained: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        if idx in anchor_indices or idx in recent_indices:
            maintained.append(entry)
            continue
        next_entry = _evict_frame_kv_from_entry(
            entry,
            gist_tokens_per_frame=max(int(gist_tokens_per_frame), 0),
        )
        if next_entry is not entry:
            evicted += 1
        maintained.append(next_entry)
    return maintained, evicted


def _gather_2d(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2 or indices.ndim != 2:
        raise ValueError(f"Expected [B,N] tensors, got {tuple(value.shape)} and {tuple(indices.shape)}")
    return torch.gather(value, dim=1, index=indices.to(device=value.device, dtype=torch.long))


def apply_gist_allocation(
    memory_item: dict[str, Any],
    decision: dict[str, torch.Tensor] | None,
    *,
    compact: bool,
) -> dict[str, Any]:
    """Attach differentiable gates in training or physically compact gist K/V in inference."""
    if decision is None:
        return memory_item
    out = dict(memory_item)
    keep = decision["straight_through_keep"]
    hard_count = decision["hard_count"]
    sorted_indices = decision["sorted_indices"]
    batch_size, max_tokens = keep.shape
    if compact:
        compact_count = int(hard_count.max().item())
        selected = sorted_indices[:, :compact_count]
        rank_valid = torch.arange(compact_count, device=hard_count.device)[None, :] < hard_count[:, None]
    for layer_key in ("gist_double", "gist_single"):
        next_layers = []
        for layer in memory_item.get(layer_key, []):
            if int(layer["k"].shape[1]) != max_tokens:
                raise ValueError(
                    f"Adaptive gist decision has {max_tokens} candidates but layer has {layer['k'].shape[1]}."
                )
            layer_out = dict(layer)
            if compact:
                layer_out["k"] = gather_by_index(layer["k"], selected)
                layer_out["v"] = gather_by_index(layer["v"], selected)
                layer_out["topk_indices"] = _gather_2d(layer["topk_indices"], selected)
                layer_out["topk_scores"] = _gather_2d(layer["topk_scores"], selected)
                layer_out["spatial_xy"] = gather_by_index(layer["spatial_xy"], selected)
                selected_valid = rank_valid
                base_valid = layer.get("memory_is_valid")
                if isinstance(base_valid, torch.Tensor):
                    selected_valid = selected_valid & _gather_2d(base_valid.to(torch.long), selected).bool()
                layer_out["memory_is_valid"] = selected_valid.to(device=layer_out["k"].device)
                layer_out["memory_attention_gate"] = selected_valid.to(
                    device=layer_out["k"].device, dtype=torch.float32
                )
            else:
                layer_out["memory_attention_gate"] = keep.to(
                    device=layer_out["k"].device, dtype=torch.float32
                )
            next_layers.append(layer_out)
        out[layer_key] = next_layers
    out["adaptive_gist_hard_count"] = hard_count
    out["adaptive_gist_expected_count"] = decision["expected_count"]
    return out


def _compress_layer_tokens(layer: dict[str, Any], tokens_per_layer: int) -> dict[str, Any]:
    k = layer["k"]
    token_count = int(k.shape[1])
    keep = min(max(int(tokens_per_layer), 0), token_count)
    if keep >= token_count:
        return dict(layer)
    if keep == 0:
        empty = torch.empty(k.shape[0], 0, device=k.device, dtype=torch.long)
        out = dict(layer)
        out["k"] = layer["k"][:, :0]
        out["v"] = layer["v"][:, :0]
        out["topk_indices"] = empty
        out["topk_scores"] = layer["topk_scores"][:, :0]
        out["spatial_xy"] = layer["spatial_xy"][:, :0]
        valid = layer.get("memory_is_valid")
        if isinstance(valid, torch.Tensor):
            out["memory_is_valid"] = valid[:, :0]
        return out

    scores = layer.get("topk_scores")
    if not isinstance(scores, torch.Tensor) or scores.numel() == 0:
        scores = k.float().norm(dim=-1)
    scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
    _, local_idx = torch.topk(scores, k=keep, dim=-1)
    out = dict(layer)
    out["k"] = gather_by_index(layer["k"], local_idx)
    out["v"] = gather_by_index(layer["v"], local_idx)
    out["topk_indices"] = _gather_2d(layer["topk_indices"], local_idx)
    out["topk_scores"] = _gather_2d(layer["topk_scores"], local_idx)
    out["spatial_xy"] = gather_by_index(layer["spatial_xy"], local_idx)
    valid = layer.get("memory_is_valid")
    if isinstance(valid, torch.Tensor):
        out["memory_is_valid"] = _gather_2d(valid.to(dtype=torch.long), local_idx).to(dtype=torch.bool)
    return out


def compress_memory_item(memory_item: dict[str, Any], tokens_per_layer: int, *, memory_kind: str = "gist") -> dict[str, Any]:
    """Legacy top-k compaction helper.

    MemoryWAM v0.2 uses learned gist-token K/V instead of top-k pooled frame K/V.
    Keep this helper available for old checkpoints/configs, but the hybrid policy
    below no longer uses it as a substitute for missing gist tokens.
    """
    compressed = dict(memory_item)
    compressed["memory_kind"] = str(memory_kind)
    compressed["compressed_tokens_per_layer"] = int(tokens_per_layer)
    for layer_key in ("double", "single"):
        compressed[layer_key] = [
            _compress_layer_tokens(layer, int(tokens_per_layer))
            for layer in memory_item.get(layer_key, [])
        ]
    return compressed


def select_hybrid_memory_entries(
    entries: list[dict[str, Any]],
    *,
    anchor_frames: int = 2,
    recent_frames: int = 4,
    gist_tokens_per_frame: int = 8,
    history_span: dict[str, Any] | None = None,
    compact_dynamic: bool = False,
) -> list[dict[str, Any]]:
    """Materialize MemoryWAM's fixed hybrid cache order for attention.

    The visible cache order is:
        [anchor full K/V] + [all learned gist K/V in temporal order] + [recent full K/V]

    Anchor frames that also fall inside the recent window keep only one copy of
    their full K/V. Every learned gist K/V is kept exactly once.
    """
    if not entries:
        return []
    n = len(entries)
    anchor_n = max(int(anchor_frames), 0)
    recent_n = max(int(recent_frames), 0)
    gist_tokens = max(int(gist_tokens_per_frame), 0)
    anchor_indices = set(range(min(anchor_n, n)))
    recent_start = max(0, n - recent_n)
    recent_indices = set(range(recent_start, n))

    selected: list[dict[str, Any]] = []
    for idx in sorted(anchor_indices):
        entry = entries[idx]
        selected.append(
            _materialize_hybrid_memory_entry(
                entry,
                include_frame=True,
                include_gist=False,
                memory_kind="anchor",
            )
        )

    for idx, entry in enumerate(entries):
        has_learned_gist = _memory_layers_have_valid_tokens(
            list(entry.get("gist_double", [])) + list(entry.get("gist_single", []))
        )
        if not has_learned_gist or gist_tokens <= 0:
            continue
        gist_entry = _materialize_hybrid_memory_entry(
            entry,
            include_frame=False,
            include_gist=True,
            memory_kind="gist",
        )
        if history_span is not None:
            age = n - idx
            hard_span = history_span["hard_span"]
            soft_span = history_span["soft_span"]
            hard_keep = age <= hard_span
            if compact_dynamic and not bool(hard_keep.any().item()):
                continue
            temperature = max(float(history_span.get("gate_temperature", 1.0)), 1e-4)
            soft_keep = torch.sigmoid((soft_span - float(age) + 0.5) / temperature)
            keep = hard_keep.to(soft_keep.dtype) + soft_keep - soft_keep.detach()
            for layer_key in ("double", "single"):
                gated_layers = []
                for layer in gist_entry.get(layer_key, []):
                    layer_out = dict(layer)
                    gate = keep.to(device=layer_out["k"].device, dtype=torch.float32)[:, None]
                    gate = gate.expand(-1, int(layer_out["k"].shape[1]))
                    layer_out["memory_attention_gate"] = gate
                    if compact_dynamic:
                        valid = layer_out.get("memory_is_valid")
                        hard_valid = hard_keep.to(device=layer_out["k"].device)[:, None].expand_as(gate)
                        layer_out["memory_is_valid"] = (
                            hard_valid if valid is None else valid.to(torch.bool) & hard_valid
                        )
                    gated_layers.append(layer_out)
                gist_entry[layer_key] = gated_layers
        selected.append(gist_entry)

    for idx in sorted(recent_indices - anchor_indices):
        entry = entries[idx]
        selected.append(
            _materialize_hybrid_memory_entry(
                entry,
                include_frame=True,
                include_gist=False,
                memory_kind="recent",
            )
        )
    return selected


class EditKVMemoryBank:
    def __init__(
        self,
        *,
        max_entries: int = 0,
        write_threshold: float = 0.55,
        cooldown_steps: int = 10,
        consolidation: str = "adjacent_merge",
        merge_until_under_limit: bool = True,
        anchor_frames: int = 2,
        recent_frames: int = 4,
        gist_tokens_per_frame: int = 8,
        detach_on_write: bool | str = "auto",
    ) -> None:
        self.max_entries = int(max_entries)
        self.write_threshold = float(write_threshold)
        self.cooldown_steps = int(cooldown_steps)
        self.consolidation = str(consolidation)
        self.merge_until_under_limit = bool(merge_until_under_limit)
        self.anchor_frames = max(int(anchor_frames), 0)
        self.recent_frames = max(int(recent_frames), 0)
        self.gist_tokens_per_frame = max(int(gist_tokens_per_frame), 0)
        self.detach_on_write = detach_on_write
        self.frames: list[dict[str, Any]] = []
        self.entries: list[dict[str, Any]] = []
        self.last_write_step: int | None = None
        self.last_write_happened = False
        self.last_write_confidence = 0.0
        self.last_full_frame_stored = False
        self.last_num_merged = 0
        self.last_num_evicted = 0
        self.batch_size: int | None = None
        self.device: torch.device | None = None

    def reset(self, batch_size: int | None = None, device: torch.device | str | None = None) -> None:
        self.frames = []
        self.entries = []
        self.last_write_step = None
        self.last_write_happened = False
        self.last_write_confidence = 0.0
        self.last_full_frame_stored = False
        self.last_num_merged = 0
        self.last_num_evicted = 0
        self.batch_size = None if batch_size is None else int(batch_size)
        self.device = None if device is None else torch.device(device)

    @staticmethod
    def _confidence_scalar(confidence: torch.Tensor | float | int) -> float:
        if isinstance(confidence, torch.Tensor):
            if confidence.numel() == 0:
                return 0.0
            return float(confidence.detach().float().mean().item())
        return float(confidence)

    @staticmethod
    def _detach_memory_item(memory_item: dict[str, Any]) -> dict[str, Any]:
        def _detach(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.detach()
            if isinstance(value, dict):
                return {key: _detach(item) for key, item in value.items()}
            if isinstance(value, list):
                return [_detach(item) for item in value]
            if isinstance(value, tuple):
                return tuple(_detach(item) for item in value)
            return value

        return _detach(memory_item)

    def _should_detach_on_write(self) -> bool:
        mode = self.detach_on_write
        if isinstance(mode, str):
            key = mode.strip().lower()
            if key in {"", "auto"}:
                return not torch.is_grad_enabled()
            if key in {"1", "true", "yes", "on"}:
                return True
            if key in {"0", "false", "no", "off"}:
                return False
            raise ValueError(f"Unsupported edit-memory detach_on_write={mode!r}.")
        return bool(mode)

    def detach_graph(self) -> None:
        """Cut TBPTT autograd history without changing the stored memory values."""
        self.frames = [self._detach_memory_item(item) for item in self.frames]
        self.entries = [self._detach_memory_item(item) for item in self.entries]

    @staticmethod
    def _move_memory_item(memory_item: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
        target = torch.device(device)

        def _move(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.detach().to(device=target).clone()
            if isinstance(value, dict):
                return {key: _move(item) for key, item in value.items()}
            if isinstance(value, list):
                return [_move(item) for item in value]
            if isinstance(value, tuple):
                return tuple(_move(item) for item in value)
            return value

        return _move(memory_item)

    def state_dict(self, *, device: torch.device | str = "cpu") -> dict[str, Any]:
        """Return a detached, portable state for EATB carry and checkpointing."""
        return {
            "version": 1,
            "frames": [self._move_memory_item(item, device) for item in self.frames],
            "last_write_step": self.last_write_step,
            "last_write_happened": bool(self.last_write_happened),
            "last_write_confidence": float(self.last_write_confidence),
            "last_full_frame_stored": bool(self.last_full_frame_stored),
            "last_num_merged": int(self.last_num_merged),
            "last_num_evicted": int(self.last_num_evicted),
            "batch_size": self.batch_size,
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
        *,
        device: torch.device | str,
    ) -> None:
        if int(state.get("version", 0)) != 1:
            raise ValueError(f"Unsupported EditKVMemoryBank state version: {state.get('version')!r}.")
        self.frames = [self._move_memory_item(item, device) for item in state.get("frames", [])]
        self.entries = select_hybrid_memory_entries(
            self.frames,
            anchor_frames=self.anchor_frames,
            recent_frames=self.recent_frames,
            gist_tokens_per_frame=self.gist_tokens_per_frame,
        )
        self.last_write_step = state.get("last_write_step")
        self.last_write_happened = bool(state.get("last_write_happened", False))
        self.last_write_confidence = float(state.get("last_write_confidence", 0.0))
        self.last_full_frame_stored = bool(state.get("last_full_frame_stored", False))
        self.last_num_merged = int(state.get("last_num_merged", 0))
        self.last_num_evicted = int(state.get("last_num_evicted", 0))
        batch_size = state.get("batch_size")
        self.batch_size = None if batch_size is None else int(batch_size)
        self.device = torch.device(device)

    @staticmethod
    def _apply_gist_write_mask(
        memory_item: dict[str, Any],
        gist_write_selected: torch.Tensor | bool,
    ) -> dict[str, Any]:
        out = dict(memory_item)
        if isinstance(gist_write_selected, torch.Tensor):
            selected = gist_write_selected.detach().to(dtype=torch.bool).reshape(-1)
        else:
            batch_size = 1
            for layer_key in ("gist_double", "gist_single", "double", "single"):
                layers = memory_item.get(layer_key, [])
                if layers:
                    batch_size = int(layers[0]["k"].shape[0])
                    break
            selected = torch.full((batch_size,), bool(gist_write_selected), dtype=torch.bool)
        out["gist_write_selected"] = selected
        for layer_key in ("gist_double", "gist_single"):
            next_layers = []
            for layer in memory_item.get(layer_key, []):
                layer_out = dict(layer)
                selected_on_device = selected.to(device=layer_out["k"].device)
                token_valid = selected_on_device[:, None].expand(
                    -1, int(layer_out["k"].shape[1])
                )
                existing_valid = layer_out.get("memory_is_valid")
                if isinstance(existing_valid, torch.Tensor):
                    token_valid = existing_valid.to(
                        device=layer_out["k"].device,
                        dtype=torch.bool,
                    ) & token_valid
                layer_out["memory_is_valid"] = token_valid
                next_layers.append(layer_out)
            out[layer_key] = next_layers
        return out

    def maybe_write(
        self,
        memory_item: dict[str, Any],
        step_idx: int,
        confidence: torch.Tensor | float | int,
        *,
        gist_write_selected: torch.Tensor | bool | None = None,
    ) -> bool:
        """Always store the chronological frame; threshold only gates its persistent gist.

        Full frame K/V is retained for the first ``anchor_frames`` raw replans and
        the latest ``recent_frames`` raw replans. A frame outside those windows
        keeps gist K/V only when ``gist_write_selected`` is true.
        """
        conf_value = self._confidence_scalar(confidence)
        self.last_write_happened = False
        self.last_write_confidence = conf_value
        self.last_full_frame_stored = False
        self.last_num_merged = 0
        self.last_num_evicted = 0
        if gist_write_selected is None:
            gist_write_selected = conf_value >= self.write_threshold
        if (
            self.last_write_step is not None
            and int(step_idx) - int(self.last_write_step) < self.cooldown_steps
        ):
            if isinstance(gist_write_selected, torch.Tensor):
                gist_write_selected = torch.zeros_like(
                    gist_write_selected,
                    dtype=torch.bool,
                )
            else:
                gist_write_selected = False

        item = self._detach_memory_item(memory_item) if self._should_detach_on_write() else dict(memory_item)
        item.setdefault("step_start", int(step_idx))
        item.setdefault("step_end", int(step_idx))
        if isinstance(confidence, torch.Tensor):
            item["confidence"] = confidence.detach() if self._should_detach_on_write() else confidence
        else:
            item["confidence"] = float(confidence)
        item.setdefault("merge_count", 1)
        item = self._apply_gist_write_mask(item, gist_write_selected)
        self.frames.append(item)
        selected = item["gist_write_selected"]
        gist_write_happened = bool(selected.any().item())
        if gist_write_happened:
            self.last_write_step = int(step_idx)
        self.last_write_happened = gist_write_happened
        self.last_full_frame_stored = True
        self.last_num_evicted = self._refresh_hybrid_entries()
        self.last_num_merged = self.last_num_evicted
        return gist_write_happened

    def read(
        self,
        history_span: dict[str, Any] | None = None,
        *,
        compact_dynamic: bool = False,
    ) -> dict[str, Any] | None:
        if history_span is None:
            return combine_memory_entries(self.entries)
        return combine_memory_entries(
            select_hybrid_memory_entries(
                self.frames,
                anchor_frames=self.anchor_frames,
                recent_frames=self.recent_frames,
                gist_tokens_per_frame=self.gist_tokens_per_frame,
                history_span=history_span,
                compact_dynamic=compact_dynamic,
            )
        )

    def _refresh_hybrid_entries(self) -> int:
        self.frames, num_evicted = evict_nonpersistent_frame_kv(
            self.frames,
            anchor_frames=self.anchor_frames,
            recent_frames=self.recent_frames,
            gist_tokens_per_frame=self.gist_tokens_per_frame,
        )
        self.entries = select_hybrid_memory_entries(
            self.frames,
            anchor_frames=self.anchor_frames,
            recent_frames=self.recent_frames,
            gist_tokens_per_frame=self.gist_tokens_per_frame,
        )
        return int(num_evicted)

    def consolidate_if_needed(self) -> int:
        if self.max_entries <= 0:
            return 0
        if len(self.entries) <= self.max_entries:
            return 0
        if self.consolidation != "adjacent_merge":
            dropped = len(self.entries) - self.max_entries
            self.entries = self.entries[-self.max_entries :]
            return dropped

        num_merged = 0
        while len(self.entries) > self.max_entries:
            sims = [
                self.compute_merge_similarity(self.entries[i], self.entries[i + 1])
                for i in range(len(self.entries) - 1)
            ]
            merge_idx = max(range(len(sims)), key=lambda idx: sims[idx])
            merged = self.merge_entries(self.entries[merge_idx], self.entries[merge_idx + 1])
            self.entries = self.entries[:merge_idx] + [merged] + self.entries[merge_idx + 2 :]
            num_merged += 1
            if not self.merge_until_under_limit:
                break
        return num_merged

    @staticmethod
    def _entry_summary(entry: dict[str, Any]) -> torch.Tensor:
        parts = []
        for layer_key in ("double", "single"):
            for layer in entry.get(layer_key, []):
                for tensor_key in ("k", "v"):
                    value = layer.get(tensor_key)
                    if isinstance(value, torch.Tensor) and value.numel() > 0:
                        parts.append(value.detach().float().reshape(-1, value.shape[-1]).mean(dim=0))
        if not parts:
            return torch.zeros(1)
        return torch.stack(parts, dim=0).mean(dim=0)

    @classmethod
    def compute_merge_similarity(cls, a: dict[str, Any], b: dict[str, Any]) -> float:
        summary_a = cls._entry_summary(a)
        summary_b = cls._entry_summary(b).to(device=summary_a.device)
        if summary_a.numel() != summary_b.numel():
            dim = min(summary_a.numel(), summary_b.numel())
            summary_a = summary_a.flatten()[:dim]
            summary_b = summary_b.flatten()[:dim]
        if summary_a.float().norm().item() == 0.0 or summary_b.float().norm().item() == 0.0:
            return 0.0
        return float(F.cosine_similarity(summary_a.flatten(), summary_b.flatten(), dim=0).item())

    @classmethod
    def merge_entries(cls, a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        wa = cls._confidence_scalar(a.get("confidence", 1.0))
        wb = cls._confidence_scalar(b.get("confidence", 1.0))
        denom = wa + wb + 1e-8
        keep_a = wa >= wb
        merged: dict[str, Any] = {"double": [], "single": []}
        for layer_key in ("double", "single"):
            for layer_a, layer_b in zip(a.get(layer_key, []), b.get(layer_key, [])):
                base = layer_a if keep_a else layer_b
                k_new = (wa * layer_a["k"].float() + wb * layer_b["k"].float()) / denom
                v_new = (wa * layer_a["v"].float() + wb * layer_b["v"].float()) / denom
                merged[layer_key].append(
                    {
                        "k": k_new.to(dtype=layer_a["k"].dtype),
                        "v": v_new.to(dtype=layer_a["v"].dtype),
                        "topk_indices": base["topk_indices"],
                        "topk_scores": (wa * layer_a["topk_scores"].float() + wb * layer_b["topk_scores"].float()) / denom,
                        "spatial_xy": base["spatial_xy"],
                        "layer_idx": int(base.get("layer_idx", len(merged[layer_key]))),
                    }
                )
        confidence_a = cls._confidence_scalar(a.get("confidence", 0.0))
        confidence_b = cls._confidence_scalar(b.get("confidence", 0.0))
        merged["confidence"] = max(confidence_a, confidence_b)
        merged["step_start"] = min(int(a.get("step_start", 0)), int(b.get("step_start", 0)))
        merged["step_end"] = max(int(a.get("step_end", 0)), int(b.get("step_end", 0)))
        merged["merge_count"] = int(a.get("merge_count", 1)) + int(b.get("merge_count", 1))
        return merged
