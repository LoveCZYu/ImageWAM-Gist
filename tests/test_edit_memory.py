import torch
import tempfile
from pathlib import Path

from imagewam.models.backbones.edit_memory import (
    AdaptiveGistAllocator,
    DynamicHistorySpanController,
    EditKVMemoryBank,
    apply_gist_allocation,
    evict_nonpersistent_frame_kv,
    memory_tokens_per_layer,
    select_hybrid_memory_entries,
    VLMSemanticCountCache,
    vlm_supervised_adaptive_k_loss,
)


def _layer(value: torch.Tensor, count: int) -> dict:
    tokens = value.reshape(1, 1, 1).expand(1, count, 1)
    return {
        "k": tokens,
        "v": tokens,
        "topk_indices": torch.arange(count).reshape(1, count),
        "topk_scores": torch.ones(1, count),
        "spatial_xy": torch.zeros(1, count, 2),
        "layer_idx": 0,
    }


def _entry(step: int, value: torch.Tensor, frame_tokens: int = 12, gist_tokens: int = 8) -> dict:
    return {
        "double": [_layer(value, frame_tokens)],
        "single": [_layer(value, frame_tokens)],
        "gist_double": [_layer(value, gist_tokens)],
        "gist_single": [_layer(value, gist_tokens)],
        "step_start": step,
        "step_end": step,
        "confidence": torch.ones(1),
    }


def test_memorywam_fixed_hybrid_order_and_counts() -> None:
    entries = [_entry(step, torch.tensor(float(step))) for step in range(8)]
    maintained, evicted = evict_nonpersistent_frame_kv(
        entries, anchor_frames=2, recent_frames=4, gist_tokens_per_frame=8
    )
    assert evicted == 2
    assert [bool(entry["double"]) for entry in maintained] == [True, True, False, False, True, True, True, True]

    visible = select_hybrid_memory_entries(
        maintained, anchor_frames=2, recent_frames=4, gist_tokens_per_frame=8
    )
    assert [entry["memory_kind"] for entry in visible] == [
        "anchor",
        "anchor",
        "gist",
        "gist",
        "gist",
        "gist",
        "gist",
        "gist",
        "gist",
        "gist",
        "recent",
        "recent",
        "recent",
        "recent",
    ]
    assert [entry["step_start"] for entry in visible] == [0, 1, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7]


def test_bank_reads_all_gists_and_resets_between_episodes() -> None:
    bank = EditKVMemoryBank(
        anchor_frames=2,
        recent_frames=4,
        gist_tokens_per_frame=8,
        max_entries=0,
        write_threshold=0.0,
        cooldown_steps=0,
        consolidation="none",
    )
    for step in range(8):
        bank.maybe_write(_entry(step, torch.tensor(float(step))), step_idx=step, confidence=1.0)
    assert len(bank.frames) == 8
    assert memory_tokens_per_layer(bank.read()) == 2 * 12 + 8 * 8 + 4 * 12
    bank.reset(batch_size=1, device="cpu")
    assert bank.frames == []
    assert bank.entries == []
    assert bank.read() is None


def test_raw_anchor_and_recent_frames_ignore_write_threshold() -> None:
    bank = EditKVMemoryBank(
        anchor_frames=2,
        recent_frames=4,
        gist_tokens_per_frame=8,
        max_entries=0,
        write_threshold=0.5,
        cooldown_steps=0,
        consolidation="none",
    )
    for step in range(9):
        wrote_gist = bank.maybe_write(
            _entry(step, torch.tensor(float(step))),
            step_idx=step,
            confidence=0.0,
            gist_write_selected=False,
        )
        assert not wrote_gist
        assert bank.last_full_frame_stored
    assert len(bank.frames) == 9
    assert [bool(entry["double"]) for entry in bank.frames] == [
        True,
        True,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
    ]
    assert [
        (entry["memory_kind"], entry["step_start"]) for entry in bank.entries
    ] == [
        ("anchor", 0),
        ("anchor", 1),
        ("recent", 5),
        ("recent", 6),
        ("recent", 7),
        ("recent", 8),
    ]

    bank.maybe_write(
        _entry(9, torch.tensor(9.0)),
        step_idx=9,
        confidence=0.0,
        gist_write_selected=False,
    )
    assert [
        (entry["memory_kind"], entry["step_start"]) for entry in bank.entries
    ] == [
        ("anchor", 0),
        ("anchor", 1),
        ("recent", 6),
        ("recent", 7),
        ("recent", 8),
        ("recent", 9),
    ]


def test_fixed_eight_keeps_gist_for_every_nonfull_frame() -> None:
    bank = EditKVMemoryBank(
        anchor_frames=2,
        recent_frames=4,
        gist_tokens_per_frame=8,
        max_entries=0,
        write_threshold=1.0,
        cooldown_steps=0,
        consolidation="none",
    )
    for step in range(10):
        bank.maybe_write(
            _entry(step, torch.tensor(float(step))),
            step_idx=step,
            confidence=0.0,
            gist_write_selected=True,
        )
    full_steps = [
        entry["step_start"]
        for entry in bank.entries
        if entry["memory_kind"] in {"anchor", "recent"}
    ]
    gist_steps = [
        entry["step_start"]
        for entry in bank.entries
        if entry["memory_kind"] == "gist"
    ]
    assert full_steps == [0, 1, 6, 7, 8, 9]
    assert gist_steps == list(range(10))
    assert all(
        entry["gist_double"][0]["k"].shape[1] == 8
        for entry in bank.frames
        if entry["gist_double"]
    )


def test_future_loss_reaches_written_gist() -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    bank = EditKVMemoryBank(
        anchor_frames=0,
        recent_frames=0,
        gist_tokens_per_frame=8,
        write_threshold=0.0,
        cooldown_steps=0,
        detach_on_write="auto",
    )
    bank.maybe_write(_entry(0, parameter, frame_tokens=1), step_idx=0, confidence=1.0)
    memory = bank.read()
    assert memory is not None
    loss = memory["double"][0]["v"].sum()
    loss.backward()
    assert parameter.grad is not None
    assert parameter.grad.item() == 8.0


def test_adaptive_k_count_and_compaction_are_the_same_decision() -> None:
    torch.manual_seed(7)
    allocator = AdaptiveGistAllocator(
        input_dim=4, min_tokens=0, max_tokens=8, hidden_dim=8, token_cost=0.0
    )
    summary = torch.randn(1, 4, requires_grad=True)
    decision = allocator(summary)
    assert decision["probabilities"].shape == (1, 9)
    assert torch.equal(decision["hard_count"], decision["count_logits"].argmax(dim=-1))
    assert 0 <= int(decision["hard_count"].item()) <= 8
    assert torch.equal(decision["write_decision"]["hard_write"], decision["hard_count"] > 0)

    item = _entry(0, torch.tensor(1.0))
    compact = apply_gist_allocation(item, decision, compact=True)
    assert compact["gist_double"][0]["k"].shape[1] == int(decision["hard_count"].item())
    synthetic_future_loss = (
        decision["straight_through_keep"] * torch.arange(1, 9, dtype=torch.float32)
    ).sum()
    synthetic_future_loss.backward()
    assert summary.grad is not None
    assert torch.isfinite(summary.grad).all()


def test_direct_vlm_distribution_trains_unified_k0_to_k8() -> None:
    teacher_probabilities = torch.tensor(
        [
            [0.60, 0.20, 0.05, 0.05, 0.025, 0.025, 0.02, 0.02, 0.01],
            [0.02, 0.03, 0.05, 0.60, 0.10, 0.05, 0.05, 0.05, 0.05],
        ]
    )
    teacher_counts = torch.tensor([0, 3])
    allocator = AdaptiveGistAllocator(
        input_dim=4, min_tokens=0, max_tokens=8, hidden_dim=8, token_cost=0.0
    )
    summary = torch.randn(2, 4, requires_grad=True)
    decision = allocator(summary)
    result = vlm_supervised_adaptive_k_loss(
        decision,
        teacher_probabilities,
        teacher_counts,
        torch.ones(2, dtype=torch.bool),
        count_weight=1.0,
    )
    expected = -(
        teacher_probabilities * decision["probabilities"].float().log()
    ).sum(dim=1).mean()
    assert torch.allclose(result["loss"], expected)
    loss = result["loss"]
    loss.backward()
    assert summary.grad is not None and torch.isfinite(summary.grad).all()


def test_k0_compacts_to_no_gist_and_does_not_write() -> None:
    allocator = AdaptiveGistAllocator(input_dim=4, min_tokens=0, max_tokens=8, hidden_dim=8)
    with torch.no_grad():
        allocator.count_head[-1].weight.zero_()
        allocator.count_head[-1].bias.fill_(-10.0)
        allocator.count_head[-1].bias[0] = 10.0
    decision = allocator(torch.zeros(1, 4))
    assert decision["hard_count"].item() == 0
    compact = apply_gist_allocation(_entry(0, torch.tensor(1.0)), decision, compact=True)
    assert compact["gist_double"][0]["k"].shape[1] == 0
    bank = EditKVMemoryBank(anchor_frames=0, recent_frames=0, gist_tokens_per_frame=8)
    wrote = bank.maybe_write(
        compact,
        step_idx=0,
        confidence=decision["write_decision"]["probabilities"],
        gist_write_selected=decision["write_decision"]["hard_write"],
    )
    assert not wrote
    assert bank.read() is None


def test_vlm_semantic_cache_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "semantic.pt"
        payload = {
            "format": "samta_semantic_change_curve_v1",
            "k_max": 8,
            "teacher_model": "test",
            "teacher_prompt_hash": "a",
            "source_checkpoint_sha256": "b" * 64,
            "preprocess_hash": "c",
            "semantic_scorer_hash": "d",
            "records": [
                {
                    "episode_id": 4,
                    "timestep": 12,
                    "semantic_loss": torch.arange(9).float(),
                    "teacher_probabilities": torch.tensor(
                        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.64]
                    ),
                    "teacher_count": 8,
                }
            ],
        }
        torch.save(payload, path)
        cache = VLMSemanticCountCache(path)
        probabilities, counts, found = cache.lookup(
            torch.tensor([4, 5]), torch.tensor([12, 0]), device=torch.device("cpu")
        )
        assert found.tolist() == [True, False]
        assert counts.tolist() == [8, 0]
        assert int(probabilities[0].argmax()) == 8
        assert len(cache.fingerprint or "") == 64


def test_dynamic_span_keeps_anchors_recent_and_selected_gist_history() -> None:
    entries = [_entry(step, torch.tensor(float(step))) for step in range(8)]
    maintained, _ = evict_nonpersistent_frame_kv(
        entries, anchor_frames=2, recent_frames=4, gist_tokens_per_frame=8
    )
    decision = {
        "soft_span": torch.tensor([2.0], requires_grad=True),
        "hard_span": torch.tensor([2]),
        "gate_temperature": 1.0,
    }
    visible = select_hybrid_memory_entries(
        maintained,
        anchor_frames=2,
        recent_frames=4,
        gist_tokens_per_frame=8,
        history_span=decision,
        compact_dynamic=True,
    )
    assert [entry["step_start"] for entry in visible if entry["memory_kind"] == "anchor"] == [0, 1]
    assert [entry["step_start"] for entry in visible if entry["memory_kind"] == "gist"] == [6, 7]
    assert [entry["step_start"] for entry in visible if entry["memory_kind"] == "recent"] == [4, 5, 6, 7]


def test_dynamic_span_controller_receives_gate_gradient() -> None:
    torch.manual_seed(11)
    controller = DynamicHistorySpanController(
        input_dim=4, min_frames=1, max_frames=8, hidden_dim=8, span_cost=0.0
    )
    summary = torch.randn(1, 4, requires_grad=True)
    decision = controller(summary)
    decision["gate_temperature"] = controller.gate_temperature
    entries = [_entry(step, torch.tensor(float(step))) for step in range(8)]
    visible = select_hybrid_memory_entries(
        entries,
        anchor_frames=0,
        recent_frames=0,
        gist_tokens_per_frame=8,
        history_span=decision,
        compact_dynamic=False,
    )
    loss = sum(entry["double"][0]["memory_attention_gate"].sum() for entry in visible)
    loss.backward()
    assert summary.grad is not None
    assert torch.isfinite(summary.grad).all()


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
