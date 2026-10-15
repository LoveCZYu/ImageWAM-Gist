import types
from unittest import mock

import torch

from imagewam.datasets.lerobot.robot_video_dataset import (
    episode_replan_sample_indices,
)
from imagewam.models.backbones.edit_memory import EditKVMemoryBank
from imagewam.models.backbones.imagewam import ImageWAM
from imagewam.trainer import collate_variable_edit_memory_history


def test_terminal_sample_is_last_complete_stride_aligned_replan() -> None:
    assert episode_replan_sample_indices(
        0,
        100,
        16,
        current_horizon=16,
        terminal_only=False,
    ) == [0, 16, 32, 48, 64, 80]
    assert episode_replan_sample_indices(
        0,
        100,
        16,
        current_horizon=16,
        terminal_only=True,
    ) == [80]


def test_tbptt_detach_preserves_memory_values_and_cuts_graph() -> None:
    source = torch.randn(1, 3, 4, requires_grad=True)
    layer = {
        "k": source * 2.0,
        "v": source * 3.0,
        "topk_indices": torch.arange(3).view(1, 3),
        "topk_scores": source.float().norm(dim=-1),
        "spatial_xy": torch.zeros(1, 3, 2),
    }
    item = {
        "double": [layer],
        "single": [],
        "gist_double": [],
        "gist_single": [],
        "confidence": torch.sigmoid(source.mean()),
    }
    bank = EditKVMemoryBank(detach_on_write=False)
    bank.frames = [item]
    bank.entries = [item]
    before_k = bank.entries[0]["double"][0]["k"].detach().clone()
    before_v = bank.entries[0]["double"][0]["v"].detach().clone()

    bank.detach_graph()

    after = bank.entries[0]["double"][0]
    assert torch.equal(after["k"], before_k)
    assert torch.equal(after["v"], before_v)
    assert not after["k"].requires_grad
    assert not after["v"].requires_grad
    assert not bank.entries[0]["confidence"].requires_grad


def test_variable_episode_histories_are_right_padded_with_invalid_slots() -> None:
    batch = collate_variable_edit_memory_history(
        [
            {
                "video": torch.zeros(3, 2, 4, 4),
                "edit_memory_video": torch.ones(2, 3, 2, 4, 4),
                "edit_memory_valid": torch.ones(2, dtype=torch.bool),
            },
            {
                "video": torch.zeros(3, 2, 4, 4),
                "edit_memory_video": torch.ones(4, 3, 2, 4, 4),
                "edit_memory_valid": torch.ones(4, dtype=torch.bool),
            },
        ]
    )
    assert batch["edit_memory_video"].shape == (2, 4, 3, 2, 4, 4)
    assert batch["edit_memory_valid"].tolist() == [
        [True, True, False, False],
        [True, True, True, True],
    ]


def _episode_sample(history_steps: int) -> dict[str, torch.Tensor]:
    return {
        "video": torch.zeros(1, 3, 2, 1, 1),
        "action": torch.zeros(1, 1, 1),
        "edit_memory_video": torch.zeros(1, history_steps, 3, 2, 1, 1),
        "edit_memory_action": torch.zeros(1, history_steps, 1, 1),
        "edit_memory_proprio": torch.zeros(1, history_steps, 1),
        "edit_memory_action_is_pad": torch.zeros(
            1, history_steps, 1, dtype=torch.bool
        ),
        "edit_memory_valid": torch.ones(
            1, history_steps, dtype=torch.bool
        ),
        "edit_memory_history_episode_id": torch.zeros(
            1, history_steps, dtype=torch.long
        ),
        "edit_memory_history_timestep": torch.arange(
            history_steps, dtype=torch.long
        ).view(1, history_steps),
        "edit_memory_history_stride": torch.tensor([16]),
    }


def _episode_replay_model(memory_source: str) -> ImageWAM:
    model = ImageWAM.__new__(ImageWAM)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.edit_memory_config = {
        "training": {
            "memory_source": memory_source,
            "replan_steps": 16,
            "tbptt_steps": 4,
        },
        "bank": {
            "anchor_frames": 2,
            "recent_frames": 4,
            "gist_tokens_per_frame": 8,
            "max_entries": 0,
            "write_threshold": 0.5,
            "cooldown_steps": 0,
            "consolidation": "none",
            "detach_on_write": False,
        },
    }
    return model


def test_padding_slots_execute_the_full_forward_with_zero_valid_loss() -> None:
    model = _episode_replay_model("episode_autoregressive")
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    model.register_parameter("test_parameter", parameter)
    valid_calls: list[bool] = []

    def fake_step(
        self,
        sample,
        *,
        bank,
        valid,
        step_idx,
        total_valid_steps,
    ):
        del self, sample, bank, step_idx, total_valid_steps
        valid_calls.append(bool(valid.item()))
        return parameter * valid.float().sum(), {}

    model._edit_memory_episode_tbptt_step = types.MethodType(fake_step, model)

    def pad_global_sequence(tensor, op=None):
        del op
        tensor.fill_(4)

    with (
        mock.patch.object(torch.distributed, "is_available", return_value=True),
        mock.patch.object(torch.distributed, "is_initialized", return_value=True),
        mock.patch.object(torch.distributed, "all_reduce", side_effect=pad_global_sequence),
    ):
        objectives = list(
            model._iter_training_losses_flux2_episode_tbptt_single(
                _episode_sample(history_steps=1)
            )
        )

    assert len(objectives) == 1
    assert valid_calls == [True, True, False, False]
    loss, metrics = objectives[0]
    assert metrics["edit_memory_episode_full_bptt"] == 1.0
    loss.backward()
    assert torch.equal(parameter.grad, torch.tensor(2.0))


def test_autoregressive_episode_keeps_gradient_from_last_loss_to_first_gist() -> None:
    history_steps = 5
    model = _episode_replay_model("episode_autoregressive")
    sources = torch.nn.ParameterList(
        [torch.nn.Parameter(torch.tensor(float(idx + 1))) for idx in range(6)]
    )
    model.sources = sources

    def fake_step(
        self,
        sample,
        *,
        bank,
        valid,
        step_idx,
        total_valid_steps,
    ):
        del sample, total_valid_steps
        source = self.sources[step_idx]
        valid_tokens = valid[:, None]
        layer = {
            "k": source.view(1, 1, 1),
            "v": (source * 2.0).view(1, 1, 1),
            "topk_indices": torch.zeros(1, 1, dtype=torch.long),
            "topk_scores": source.view(1, 1),
            "spatial_xy": torch.zeros(1, 1, 2),
            "memory_is_valid": valid_tokens,
        }
        item = {
            "double": [layer],
            "single": [],
            "gist_double": [dict(layer)],
            "gist_single": [],
            "confidence": source.sigmoid().view(1),
        }
        if bool(valid.item()):
            bank.maybe_write(
                item,
                step_idx=step_idx,
                confidence=1.0,
                gist_write_selected=True,
            )
        if step_idx == history_steps:
            first_gist = bank.frames[0]["gist_double"][0]["k"]
            loss = first_gist.sum() + source * 0.0
        else:
            loss = source * 0.0
        return loss, {}

    model._edit_memory_episode_tbptt_step = types.MethodType(fake_step, model)
    objectives = list(
        model._iter_training_losses_flux2_episode_tbptt_single(
            _episode_sample(history_steps=history_steps)
        )
    )
    assert len(objectives) == 1
    loss, metrics = objectives[0]
    assert metrics["edit_memory_episode_tbptt_steps"] == 6.0
    assert metrics["edit_memory_episode_full_bptt"] == 1.0
    loss.backward()
    assert sources[0].grad is not None
    assert float(sources[0].grad.item()) == 1.0


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
