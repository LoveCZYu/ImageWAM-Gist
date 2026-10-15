import torch

from imagewam.models.backbones.edit_memory import (
    AdaptiveGistAllocator,
    mix_teacher_predicted_gist_decision,
    teacher_memory_probability,
)
from imagewam.models.backbones.imagewam import ImageWAM


class _DummyVLMCache:
    def __len__(self) -> int:
        return 1

    def lookup(self, episode_ids, timesteps, *, device):
        batch_size = int(episode_ids.numel())
        probabilities = torch.zeros(batch_size, 9, device=device)
        probabilities[:, 0] = 1.0
        teacher_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        found = torch.ones(batch_size, dtype=torch.bool, device=device)
        return probabilities, teacher_counts, found


def test_teacher_to_predict_schedule_boundaries() -> None:
    kwargs = {
        "teacher_only_until_step": 1066,
        "transition_start_step": 1067,
        "transition_end_step": 4267,
    }
    assert teacher_memory_probability(0, **kwargs) == 1.0
    assert teacher_memory_probability(1066, **kwargs) == 1.0
    assert teacher_memory_probability(1067, **kwargs) == 1.0
    assert 0.49 < teacher_memory_probability(2667, **kwargs) < 0.51
    assert teacher_memory_probability(4267, **kwargs) == 0.0
    assert teacher_memory_probability(4268, **kwargs) == 0.0
    assert teacher_memory_probability(10670, **kwargs) == 0.0


def test_curriculum_mixes_teacher_count_per_sample_without_replacing_head() -> None:
    torch.manual_seed(7)
    allocator = AdaptiveGistAllocator(input_dim=4, min_tokens=0, max_tokens=8, hidden_dim=8)
    decision = allocator(torch.randn(3, 4))
    teacher_count = torch.tensor([0, 8, 4])
    mixed, teacher_mask = mix_teacher_predicted_gist_decision(
        decision,
        teacher_count=teacher_count,
        teacher_active=torch.tensor([True, True, False]),
        teacher_probability=0.5,
        random_values=torch.tensor([0.1, 0.9, 0.1]),
    )
    assert teacher_mask.tolist() == [True, False, False]
    assert int(mixed["hard_count"][0]) == 0
    assert int(mixed["hard_count"][1]) == int(decision["hard_count"][1])
    assert int(mixed["hard_count"][2]) == int(decision["hard_count"][2])
    assert mixed["probabilities"].data_ptr() == decision["probabilities"].data_ptr()
    assert torch.equal(
        mixed["straight_through_keep"][0],
        mixed["hard_keep"][0].to(mixed["straight_through_keep"].dtype),
    )


def test_predicted_memory_path_is_the_inference_decision() -> None:
    torch.manual_seed(17)
    allocator = AdaptiveGistAllocator(input_dim=4, min_tokens=0, max_tokens=8, hidden_dim=8)
    decision = allocator(torch.randn(2, 4))
    mixed, teacher_mask = mix_teacher_predicted_gist_decision(
        decision,
        teacher_count=torch.tensor([1, 8]),
        teacher_active=torch.ones(2, dtype=torch.bool),
        teacher_probability=0.0,
        random_values=torch.zeros(2),
    )
    assert not bool(teacher_mask.any())
    assert torch.equal(mixed["hard_count"], decision["hard_count"])
    assert torch.equal(mixed["straight_through_keep"], decision["straight_through_keep"])


def test_teacher_curriculum_does_not_depend_on_top_level_train_flag() -> None:
    model = ImageWAM.__new__(ImageWAM)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.edit_memory_config = {
        "training": {
            "memory_curriculum": {
                "enabled": True,
                "teacher_only_until_step": 1066,
                "transition_start_step": 1067,
                "transition_end_step": 4267,
                "granularity": "sample",
            }
        }
    }
    model._edit_memory_training_step = 0
    model.eval()
    allocator = AdaptiveGistAllocator(input_dim=4, min_tokens=0, max_tokens=8, hidden_dim=8)
    decision = allocator(torch.randn(2, 4))
    mixed, metrics = model._edit_memory_write_decision(
        decision,
        {
            "teacher_count": torch.tensor([2, 7]),
            "active": torch.ones(2, dtype=torch.bool),
        },
    )
    assert metrics["teacher_probability"] == 1.0
    assert metrics["teacher_mask"].tolist() == [True, True]
    assert mixed is not None
    assert mixed["hard_count"].tolist() == [2, 7]


def test_vlm_n0_teaches_no_write_without_removing_softk_head() -> None:
    model = ImageWAM.__new__(ImageWAM)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.edit_memory_config = {
        "training": {
            "memory_curriculum": {
                "enabled": True,
                "teacher_only_until_step": 1066,
                "transition_start_step": 1067,
                "transition_end_step": 4267,
                "granularity": "sample",
            }
        }
    }
    model._edit_memory_training_step = 0
    allocator = AdaptiveGistAllocator(input_dim=4, min_tokens=0, max_tokens=8, hidden_dim=8)
    decision = allocator(torch.randn(2, 4))
    mixed, metrics = model._edit_memory_write_decision(
        decision,
        {
            "teacher_count": torch.tensor([0, 4]),
            "write_target": torch.tensor([False, True]),
            "active": torch.ones(2, dtype=torch.bool),
        },
        random_values=torch.zeros(2),
    )
    assert mixed is not None
    assert mixed["hard_count"].tolist() == [0, 4]
    assert metrics["hard_write"].tolist() == [False, True]
    assert mixed["probabilities"].data_ptr() == decision["probabilities"].data_ptr()


def test_fixed8_curriculum_teaches_write_without_changing_gist_count() -> None:
    model = ImageWAM.__new__(ImageWAM)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.edit_memory_config = {
        "training": {
            "memory_curriculum": {
                "enabled": True,
                "teacher_only_until_step": 14,
                "transition_start_step": 15,
                "transition_end_step": 57,
                "granularity": "sample",
            }
        }
    }
    model._edit_memory_training_step = 0
    write_decision = {
        "logits": torch.tensor([2.0, -2.0]),
        "probabilities": torch.tensor([0.88, 0.12]),
        "hard_write": torch.tensor([True, False]),
    }
    mixed, metrics = model._edit_memory_write_decision(
        None,
        {
            "teacher_count": torch.tensor([1, 8]),
            "write_target": torch.tensor([False, True]),
            "active": torch.ones(2, dtype=torch.bool),
        },
        random_values=torch.zeros(2),
        write_decision=write_decision,
    )
    assert mixed is None
    assert metrics["hard_write"].tolist() == [False, True]
    assert metrics["teacher_mask"].tolist() == [True, True]


def test_fixed8_vlm_write_only_loss_trains_the_write_head() -> None:
    model = ImageWAM.__new__(ImageWAM)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.edit_memory_config = {
        "vlm_count_supervision": {
            "enabled": True,
            "write_only": True,
            "require_complete": True,
            "count_weight": 1.0,
        }
    }
    model.edit_memory_vlm_count_cache = _DummyVLMCache()
    logits = torch.tensor([2.0, -2.0], requires_grad=True)
    write_decision = {
        "logits": logits,
        "probabilities": torch.sigmoid(logits),
        "hard_write": logits >= 0,
    }
    metrics = model._edit_memory_vlm_count_supervision(
        None,
        episode_ids=torch.tensor([1, 2]),
        timesteps=torch.tensor([0, 0]),
        valid=torch.ones(2, dtype=torch.bool),
        write_decision=write_decision,
    )
    assert metrics is not None
    assert metrics["write_target"].tolist() == [False, False]
    metrics["loss"].backward()
    assert logits.grad is not None
    assert bool((logits.grad > 0).all())


def test_vlm_inactive_dummy_keeps_zero_loss_graph_for_zero3() -> None:
    model = ImageWAM.__new__(ImageWAM)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.edit_memory_config = {
        "vlm_count_supervision": {
            "enabled": True,
            "require_complete": True,
            "count_weight": 1.0,
        }
    }
    model.edit_memory_vlm_count_cache = _DummyVLMCache()
    model.edit_memory_vlm_class_weights = None
    allocator = AdaptiveGistAllocator(input_dim=4, min_tokens=0, max_tokens=8, hidden_dim=8)
    summary = torch.randn(2, 4)
    decision = allocator(summary)
    metrics = model._edit_memory_vlm_count_supervision(
        decision,
        episode_ids=torch.tensor([1, 2]),
        timesteps=torch.tensor([0, 0]),
        valid=torch.zeros(2, dtype=torch.bool),
    )
    assert metrics is not None
    assert metrics["loss"].item() == 0.0
    metrics["loss"].backward()
    count_weight = allocator.count_head[-1].weight
    assert count_weight.grad is not None
    assert torch.count_nonzero(count_weight.grad) == 0


def test_unified_student_has_identical_train_and_infer_decisions() -> None:
    model = ImageWAM.__new__(ImageWAM)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.edit_memory_config = {"training": {"memory_curriculum": {"enabled": False}}}
    model.edit_memory_write_head = None
    allocator = AdaptiveGistAllocator(input_dim=4, min_tokens=0, max_tokens=8, hidden_dim=8)
    summary = torch.randn(3, 4)
    inference_decision = allocator(summary)
    write_decision = model._edit_memory_write_confidence(summary, inference_decision)
    training_decision, metrics = model._edit_memory_write_decision(
        inference_decision, vlm_metrics=None, write_decision=write_decision
    )
    assert training_decision is not None
    assert torch.equal(training_decision["hard_count"], inference_decision["hard_count"])
    assert torch.equal(training_decision["sorted_indices"], inference_decision["sorted_indices"])
    assert torch.equal(metrics["hard_write"], inference_decision["hard_count"] > 0)


def test_rank_synchronous_history_slot_has_no_short_modulo_cycle() -> None:
    for count in (30, 60):
        slots = [
            ImageWAM._rank_synchronous_history_slot(count, step)
            for step in range(count)
        ]
        assert sorted(slots) == list(range(count))


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_"):
            value()
            print(f"PASS {name}")
