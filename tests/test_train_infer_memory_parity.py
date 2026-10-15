import inspect

import torch
import torch.nn as nn

from imagewam.models.backbones.edit_memory import (
    AdaptiveGistAllocator,
    EditKVMemoryBank,
    apply_gist_allocation,
    combine_memory_entries,
    evict_nonpersistent_frame_kv,
    select_hybrid_memory_entries,
)
from imagewam.models.backbones.imagewam import ImageWAM
from imagewam.models.backbones.action_dit_flux2 import ActionDiTFlux2
from imagewam.models.backbones.flux2_video_expert import Flux2VideoExpert


def _layer(step: int, count: int) -> dict:
    value = torch.full((1, count, 2), float(step))
    return {
        "k": value,
        "v": value + 0.25,
        "topk_indices": torch.arange(count).reshape(1, count),
        "topk_scores": torch.ones(1, count),
        "spatial_xy": torch.zeros(1, count, 2),
        "layer_idx": 0,
    }


def _entry(step: int) -> dict:
    return {
        "double": [_layer(step, 12)],
        "single": [_layer(step, 12)],
        "gist_double": [_layer(step, 8)],
        "gist_single": [_layer(step, 8)],
        "step_start": step,
        "step_end": step,
        "confidence": torch.ones(1),
    }


def test_fixed_train_infer_cache_is_exact() -> None:
    source = [_entry(step) for step in range(10)]
    maintained, _ = evict_nonpersistent_frame_kv(
        source, anchor_frames=2, recent_frames=4, gist_tokens_per_frame=8
    )
    train_cache = combine_memory_entries(
        select_hybrid_memory_entries(
            maintained, anchor_frames=2, recent_frames=4, gist_tokens_per_frame=8
        )
    )
    bank = EditKVMemoryBank(
        anchor_frames=2,
        recent_frames=4,
        gist_tokens_per_frame=8,
        write_threshold=0.0,
        cooldown_steps=0,
        detach_on_write=False,
    )
    for step, entry in enumerate(source):
        bank.maybe_write(entry, step_idx=step, confidence=1.0)
    infer_cache = bank.read()
    assert train_cache is not None and infer_cache is not None
    for layer_key in ("double", "single"):
        assert torch.equal(train_cache[layer_key][0]["k"], infer_cache[layer_key][0]["k"])
        assert torch.equal(train_cache[layer_key][0]["v"], infer_cache[layer_key][0]["v"])
        assert torch.equal(
            train_cache[layer_key][0]["memory_is_valid"],
            infer_cache[layer_key][0]["memory_is_valid"],
        )


def test_training_and_inference_use_three_history_plus_current_replan() -> None:
    bank_cfg = {"recent_frames": 4}
    assert ImageWAM._edit_memory_historical_recent_frames(bank_cfg) == 3


def test_action_visible_full_frames_match_memorywam_schedule() -> None:
    history = [_entry(step) for step in range(8)]
    maintained, _ = evict_nonpersistent_frame_kv(
        history,
        anchor_frames=2,
        recent_frames=3,
        gist_tokens_per_frame=8,
    )
    visible = select_hybrid_memory_entries(
        maintained,
        anchor_frames=2,
        recent_frames=3,
        gist_tokens_per_frame=8,
    )
    full_history_steps = [
        entry["step_start"]
        for entry in visible
        if entry["memory_kind"] in {"anchor", "recent"}
    ]
    assert full_history_steps + [8] == [0, 1, 5, 6, 7, 8]

    history.append(_entry(8))
    maintained, _ = evict_nonpersistent_frame_kv(
        history,
        anchor_frames=2,
        recent_frames=3,
        gist_tokens_per_frame=8,
    )
    visible = select_hybrid_memory_entries(
        maintained,
        anchor_frames=2,
        recent_frames=3,
        gist_tokens_per_frame=8,
    )
    full_history_steps = [
        entry["step_start"]
        for entry in visible
        if entry["memory_kind"] in {"anchor", "recent"}
    ]
    assert full_history_steps + [9] == [0, 1, 6, 7, 8, 9]


def test_flux2_temporal_ids_match_across_video_gist_and_action() -> None:
    ref_ids = Flux2VideoExpert.build_img_ids(
        batch_size=1,
        token_height=2,
        token_width=3,
        time_value=10.0,
        temporal_value=7.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    target_ids = Flux2VideoExpert.build_img_ids(
        batch_size=1,
        token_height=2,
        token_width=3,
        time_value=0.0,
        temporal_value=8.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    action_ids = ActionDiTFlux2.build_action_ids(
        batch_size=1,
        seq_len=4,
        temporal_value=7.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    model = ImageWAM.__new__(ImageWAM)
    nn.Module.__init__(model)
    model.edit_memory_gist_tokens = nn.Parameter(torch.randn(8, 5))
    _, gist_ids = model._build_flux2_gist_inputs(
        batch_size=1,
        ref_img_ids=ref_ids,
        device=torch.device("cpu"),
        token_dtype=torch.float32,
        step_idx=7,
    )
    assert gist_ids is not None
    assert torch.equal(ref_ids[..., 3], torch.full_like(ref_ids[..., 3], 7.0))
    assert torch.equal(target_ids[..., 3], torch.full_like(target_ids[..., 3], 8.0))
    assert torch.equal(action_ids[..., 3], torch.full_like(action_ids[..., 3], 7.0))
    assert torch.equal(gist_ids[..., 0], torch.full_like(gist_ids[..., 0], 10.0))
    assert torch.equal(gist_ids[..., 1], torch.full_like(gist_ids[..., 1], 2.0))
    assert torch.equal(gist_ids[..., 2], torch.full_like(gist_ids[..., 2], 3.0))
    assert torch.equal(gist_ids[..., 3], torch.full_like(gist_ids[..., 3], 7.0))


def test_adaptive_gist_train_hard_ids_equal_infer_compaction() -> None:
    torch.manual_seed(13)
    allocator = AdaptiveGistAllocator(input_dim=4, max_tokens=8, hidden_dim=8)
    decision = allocator(torch.randn(1, 4))
    item = _entry(0)
    train_item = apply_gist_allocation(item, decision, compact=False)
    infer_item = apply_gist_allocation(item, decision, compact=True)
    train_ids = torch.nonzero(
        train_item["gist_double"][0]["memory_attention_gate"][0] == 1,
        as_tuple=False,
    ).flatten()
    infer_ids = infer_item["gist_double"][0]["topk_indices"][0]
    assert torch.equal(torch.sort(train_ids).values, torch.sort(infer_ids).values)
    assert infer_ids.numel() == int(decision["hard_count"].item())


def test_adaptive_span_train_hard_frames_equal_infer_frames() -> None:
    entries = [_entry(step) for step in range(10)]
    maintained, _ = evict_nonpersistent_frame_kv(
        entries, anchor_frames=2, recent_frames=4, gist_tokens_per_frame=8
    )
    decision = {
        "soft_span": torch.tensor([3.2]),
        "hard_span": torch.tensor([3]),
        "gate_temperature": 1.0,
    }
    train_visible = select_hybrid_memory_entries(
        maintained,
        anchor_frames=2,
        recent_frames=4,
        gist_tokens_per_frame=8,
        history_span=decision,
        compact_dynamic=False,
    )
    infer_visible = select_hybrid_memory_entries(
        maintained,
        anchor_frames=2,
        recent_frames=4,
        gist_tokens_per_frame=8,
        history_span=decision,
        compact_dynamic=True,
    )
    train_hard_frames = [
        entry["step_start"]
        for entry in train_visible
        if entry["memory_kind"] == "gist"
        and float(entry["double"][0]["memory_attention_gate"][0, 0].item()) == 1.0
    ]
    infer_frames = [
        entry["step_start"] for entry in infer_visible if entry["memory_kind"] == "gist"
    ]
    assert train_hard_frames == infer_frames


def test_adaptive_gist_attention_is_within_bf16_parity_tolerance() -> None:
    torch.manual_seed(29)
    allocator = AdaptiveGistAllocator(input_dim=4, max_tokens=8, hidden_dim=8)
    decision = allocator(torch.randn(1, 4))
    item = _entry(0)
    keys = torch.randn(1, 8, 2) * 0.2
    values = torch.randn(1, 8, 2) * 0.2
    for layer_key in ("gist_double", "gist_single"):
        item[layer_key][0]["k"] = keys
        item[layer_key][0]["v"] = values
    train_layer = apply_gist_allocation(item, decision, compact=False)["gist_double"][0]
    infer_layer = apply_gist_allocation(item, decision, compact=True)["gist_double"][0]
    query = torch.randn(1, 1, 2) * 0.2

    train_logits = query @ train_layer["k"].transpose(1, 2)
    train_logits = train_logits - 12.0 * (1.0 - train_layer["memory_attention_gate"][:, None, :])
    train_output = torch.softmax(train_logits, dim=-1) @ train_layer["v"]
    infer_logits = query @ infer_layer["k"].transpose(1, 2)
    infer_output = torch.softmax(infer_logits, dim=-1) @ infer_layer["v"]

    diff = (train_output - infer_output).abs()
    print(
        "METRIC adaptive_attention max_abs=%.8f mean_abs=%.8f atol=0.001 rtol=0.001"
        % (float(diff.max().item()), float(diff.mean().item()))
    )
    assert float(diff.max().item()) <= 1e-3
    assert torch.allclose(train_output, infer_output, atol=1e-3, rtol=1e-3)


def test_clean_prefix_memory_heads_are_shared_exactly() -> None:
    torch.manual_seed(41)
    model = ImageWAM.__new__(ImageWAM)
    nn.Module.__init__(model)
    model.edit_memory_prefix_projection = nn.Linear(6, 4, bias=False)
    model.edit_memory_gist_allocator = AdaptiveGistAllocator(
        input_dim=4, max_tokens=8, hidden_dim=8
    )
    model.edit_memory_write_head = nn.Sequential(
        nn.LayerNorm(4), nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 1)
    )
    prefix_cache = {"final_video": {"img": torch.randn(2, 5, 6)}}
    train_summary = model._edit_memory_prefix_summary(prefix_cache)
    infer_summary = model._edit_memory_prefix_summary(prefix_cache)
    train_k = model._edit_memory_gist_decision(train_summary)
    infer_k = model._edit_memory_gist_decision(infer_summary)
    train_write = model._edit_memory_write_confidence(train_summary)
    infer_write = model._edit_memory_write_confidence(infer_summary)
    assert torch.equal(train_summary, infer_summary)
    assert torch.equal(train_k["hard_count"], infer_k["hard_count"])
    assert torch.equal(train_write["probabilities"], infer_write["probabilities"])


def test_flux2_video_inference_uses_the_training_memory_kv_path() -> None:
    video_source = inspect.getsource(ImageWAM.infer_video_flux2)
    assert "self.edit_memory_bank.read(" in video_source
    assert "gist_hidden_states=gist_tokens" in video_source
    assert "self.mot.prefill_flux2_video_cache(" in video_source
    assert "memory_kv_cache=memory_kv_cache" in video_source

    joint_source = inspect.getsource(ImageWAM.infer)
    flux2_source = joint_source[joint_source.index('if self.stack == "flux2":') :]
    assert flux2_source.index("video_out = self.infer_video_flux2(") < flux2_source.index(
        "action_out = self.infer_action_flux2("
    )


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
