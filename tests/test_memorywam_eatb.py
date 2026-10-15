from __future__ import annotations

import pytest
import torch

from imagewam.datasets.eatb import EATBDistributedBatchSampler, EATBWindowDataset
from imagewam.models.backbones.imagewam import ImageWAM
from imagewam.trainer import Wan22Trainer


class _EpisodeDataset:
    def __init__(self, lengths: list[int]) -> None:
        starts = []
        ends = []
        cursor = 0
        for length in lengths:
            starts.append(cursor)
            cursor += int(length)
            ends.append(cursor)
        self._edit_memory_episode_from = torch.tensor(starts, dtype=torch.long)
        self._edit_memory_episode_to = torch.tensor(ends, dtype=torch.long)
        self.edit_memory_history_stride = 1
        self.edit_memory_drop_incomplete_current = False
        self.edit_memory_history_mode = "episode"
        self.edit_memory_replan_aligned_samples = True
        self.edit_memory_episode_terminal_samples_only = True
        self.edit_memory_history_max_steps = 0
        self.num_frames = 2

    def _get(self, index: int) -> dict:
        raw_index = int(self._edit_memory_sample_raw_indices[int(index)].item())
        value = float(raw_index)
        return {
            "video": torch.full((3, 2, 1, 1), value),
            "action": torch.full((1, 1), value),
            "proprio": torch.full((1,), value),
            "prompt": "task",
        }


def _window_episode(dataset: EATBWindowDataset, index: int) -> int:
    return index if index < 0 else dataset._windows[index].episode_id


def test_eatb_fixed_cohort_has_no_lane_backfill_and_fixed_rank_ownership() -> None:
    dataset = EATBWindowDataset(_EpisodeDataset([5, 1, 3]), num_chunks=2)
    rank0 = EATBDistributedBatchSampler(
        dataset, local_batch_size=1, num_replicas=2, rank=0
    )
    rank1 = EATBDistributedBatchSampler(
        dataset, local_batch_size=1, num_replicas=2, rank=1
    )

    rank0_episodes = [_window_episode(dataset, batch[0]) for batch in rank0]
    rank1_episodes = [_window_episode(dataset, batch[0]) for batch in rank1]
    assert rank0_episodes == [0, 0, 0, 2, 2]
    assert rank1_episodes == [1, -2, -2, -4, -4]

    rank0.set_resume_batch_offset(2)
    assert [_window_episode(dataset, batch[0]) for batch in rank0] == [0, 2, 2]


def test_eatb_tail_and_inactive_lane_padding_match_official_contract() -> None:
    dataset = EATBWindowDataset(_EpisodeDataset([5]), num_chunks=2)
    tail = dataset[2]
    assert tail["eatb_chunk_ids"].tolist() == [4, -1]
    assert tail["eatb_chunk_mask"].tolist() == [True, False]
    assert tail["eatb_num_chunks"].item() == 1
    assert tail["video"][:, 0, 0, 0, 0].tolist() == [4.0, 0.0]

    inactive = dataset[-7]
    assert inactive["eatb_episode_id"].item() == -7
    assert inactive["eatb_chunk_ids"].tolist() == [-1, -1]
    assert inactive["eatb_chunk_mask"].tolist() == [False, False]
    assert torch.count_nonzero(inactive["video"]) == 0


def _eatb_model() -> ImageWAM:
    model = ImageWAM.__new__(ImageWAM)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.edit_memory_config = {
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
        "training": {"memory_source": "eatb", "tbptt_steps": 2},
    }
    return model


def _memory_item(source: torch.Tensor, step: int = 0) -> dict:
    layer = {
        "k": (source * 2).reshape(1, 1, 1),
        "v": (source * 3).reshape(1, 1, 1),
        "topk_indices": torch.zeros(1, 1, dtype=torch.long),
        "topk_scores": source.reshape(1, 1),
        "spatial_xy": torch.zeros(1, 1, 2),
    }
    return {
        "double": [layer],
        "single": [],
        "gist_double": [dict(layer)],
        "gist_single": [],
        "step_start": step,
        "step_end": step,
        "confidence": source.sigmoid().reshape(1),
    }


def _queue_bank(
    model: ImageWAM,
    *,
    episode_id: int,
    next_chunk_id: int,
    source: torch.Tensor,
    is_last_window: bool = False,
) -> None:
    bank = model._edit_memory_new_training_bank()
    item = _memory_item(source)
    bank.frames = [item]
    bank.entries = [item]
    _, pending = model._edit_memory_eatb_stores()
    pending.append(
        {
            "episode_id": episode_id,
            "next_chunk_id": next_chunk_id,
            "is_last_window": is_last_window,
            "bank": bank,
        }
    )


def test_eatb_state_appears_only_after_commit_and_detaches_across_segments() -> None:
    model = _eatb_model()
    source = torch.tensor(2.0, requires_grad=True)
    _queue_bank(model, episode_id=7, next_chunk_id=2, source=source)
    host_bank, pending = model._edit_memory_eatb_stores()
    assert host_bank == {}
    assert len(pending) == 1

    model.commit_edit_memory_eatb_state()
    saved = model.edit_memory_eatb_state_dict()["episodes"][7]
    assert torch.equal(saved["bank"]["frames"][0]["double"][0]["k"], torch.tensor([[[4.0]]]))
    assert not saved["bank"]["frames"][0]["double"][0]["k"].requires_grad

    carried = model._edit_memory_eatb_load_bank(7, 2)
    next_source = torch.tensor(3.0, requires_grad=True)
    loss = carried.frames[0]["double"][0]["k"].sum() + next_source
    loss.backward()
    assert source.grad is None
    assert next_source.grad.item() == 1.0


def test_eatb_missing_or_out_of_order_bank_fails_without_consuming_state() -> None:
    model = _eatb_model()
    with pytest.raises(KeyError, match="Missing EATB bank"):
        model._edit_memory_eatb_load_bank(3, 2)

    _queue_bank(
        model,
        episode_id=3,
        next_chunk_id=2,
        source=torch.tensor(1.0),
    )
    model.commit_edit_memory_eatb_state()
    with pytest.raises(ValueError, match="cursor mismatch"):
        model._edit_memory_eatb_load_bank(3, 4)
    model._edit_memory_eatb_load_bank(3, 2)


def test_eatb_final_window_releases_episode_state() -> None:
    model = _eatb_model()
    _queue_bank(model, episode_id=5, next_chunk_id=2, source=torch.tensor(1.0))
    model.commit_edit_memory_eatb_state()
    _queue_bank(
        model,
        episode_id=5,
        next_chunk_id=4,
        source=torch.tensor(2.0),
        is_last_window=True,
    )
    model.commit_edit_memory_eatb_state()
    assert model.edit_memory_eatb_state_dict()["episodes"] == {}


class _FakeAccelerator:
    def __init__(self, model: ImageWAM, rank: int = 1, world_size: int = 2) -> None:
        self._model = model
        self.process_index = rank
        self.num_processes = world_size

    def unwrap_model(self, model: ImageWAM) -> ImageWAM:
        assert model is self._model
        return model


def _checkpoint_trainer(model: ImageWAM, *, world_size: int = 2) -> Wan22Trainer:
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = model
    trainer.accelerator = _FakeAccelerator(model, world_size=world_size)
    trainer.eatb_enabled = True
    trainer.global_step = 11
    trainer.epoch = 2
    trainer.batch_in_epoch = 3
    return trainer


def test_eatb_rank_checkpoint_roundtrip_and_world_size_validation(tmp_path) -> None:
    source_model = _eatb_model()
    _queue_bank(
        source_model,
        episode_id=9,
        next_chunk_id=6,
        source=torch.tensor(4.0),
    )
    source_model.commit_edit_memory_eatb_state()
    _checkpoint_trainer(source_model)._save_eatb_rank_state(str(tmp_path))

    restored_model = _eatb_model()
    restored = _checkpoint_trainer(restored_model)
    restored._load_eatb_rank_state(str(tmp_path))
    bank = restored_model._edit_memory_eatb_load_bank(9, 6)
    assert torch.equal(bank.frames[0]["double"][0]["k"], torch.tensor([[[8.0]]]))

    wrong_world = _checkpoint_trainer(_eatb_model(), world_size=4)
    with pytest.raises(ValueError, match="metadata mismatch"):
        wrong_world._load_eatb_rank_state(str(tmp_path))
