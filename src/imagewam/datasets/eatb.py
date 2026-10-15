from __future__ import annotations

import copy
import dataclasses
from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import Dataset, Sampler

from .lerobot.robot_video_dataset import episode_replan_sample_indices


@dataclasses.dataclass(frozen=True)
class EATBWindowInfo:
    episode_id: int
    start_chunk_id: int
    step_dataset_indices: tuple[int, ...]
    is_last_window: bool


class EATBWindowDataset(Dataset[dict]):
    """Expose fixed-K, consecutive replan windows from complete episodes."""

    _TEMPORAL_SAMPLE_KEYS = (
        "video",
        "action",
        "proprio",
        "image_is_pad",
        "action_is_pad",
        "proprio_is_pad",
        "action_dim_is_pad",
        "proprio_dim_is_pad",
    )

    def __init__(self, episode_dataset: Dataset, num_chunks: int) -> None:
        self.episode_dataset = episode_dataset
        self.num_chunks = int(num_chunks)
        if self.num_chunks <= 0:
            raise ValueError(f"EATB num_chunks must be positive, got {num_chunks}.")

        required = (
            "_edit_memory_episode_from",
            "_edit_memory_episode_to",
            "edit_memory_history_stride",
            "edit_memory_drop_incomplete_current",
            "num_frames",
            "_get",
        )
        missing = [name for name in required if not hasattr(episode_dataset, name)]
        if missing:
            raise TypeError(f"EATB requires a RobotVideoDataset-style episode dataset; missing {missing}.")
        if str(getattr(episode_dataset, "edit_memory_history_mode", "none")) != "episode":
            raise ValueError("EATB requires edit_memory_history_mode='episode'.")
        if not bool(getattr(episode_dataset, "edit_memory_replan_aligned_samples", False)):
            raise ValueError("EATB requires edit_memory_replan_aligned_samples=true.")
        if not bool(getattr(episode_dataset, "edit_memory_episode_terminal_samples_only", False)):
            raise ValueError(
                "EATB expects one complete source item per episode; set "
                "edit_memory_episode_terminal_samples_only=true."
            )
        if int(getattr(episode_dataset, "edit_memory_history_max_steps", 0)) != 0:
            raise ValueError("EATB requires edit_memory_history_max_steps=0 so no replan is omitted.")

        episode_from = episode_dataset._edit_memory_episode_from.tolist()
        episode_to = episode_dataset._edit_memory_episode_to.tolist()
        stride = int(episode_dataset.edit_memory_history_stride)
        current_horizon = (
            int(episode_dataset.num_frames) - 1
            if bool(episode_dataset.edit_memory_drop_incomplete_current)
            else 0
        )

        step_raw_indices: list[int] = []
        windows: list[EATBWindowInfo] = []
        episode_windows: list[tuple[int, tuple[int, ...]]] = []
        for episode_id, (episode_start, episode_end) in enumerate(zip(episode_from, episode_to, strict=True)):
            raw_indices = episode_replan_sample_indices(
                episode_start,
                episode_end,
                stride,
                current_horizon=current_horizon,
                terminal_only=False,
            )
            if not raw_indices:
                continue
            first_step_dataset_index = len(step_raw_indices)
            step_raw_indices.extend(raw_indices)
            window_indices: list[int] = []
            for start_chunk_id in range(0, len(raw_indices), self.num_chunks):
                stop_chunk_id = min(start_chunk_id + self.num_chunks, len(raw_indices))
                step_indices = tuple(
                    range(
                        first_step_dataset_index + start_chunk_id,
                        first_step_dataset_index + stop_chunk_id,
                    )
                )
                window_indices.append(len(windows))
                windows.append(
                    EATBWindowInfo(
                        episode_id=int(episode_id),
                        start_chunk_id=int(start_chunk_id),
                        step_dataset_indices=step_indices,
                        is_last_window=stop_chunk_id == len(raw_indices),
                    )
                )
            episode_windows.append((int(episode_id), tuple(window_indices)))

        if not windows:
            raise ValueError("EATB created no temporal windows from the training dataset.")

        # The source dataset is configured for terminal full-episode replay. A
        # shallow view disables history materialization and indexes individual
        # aligned replans instead.
        self._step_dataset = copy.copy(episode_dataset)
        self._step_dataset.edit_memory_history_mode = "none"
        self._step_dataset.edit_memory_episode_terminal_samples_only = False
        self._step_dataset.skip_padding_as_possible = False
        self._step_dataset._edit_memory_sample_raw_indices = torch.as_tensor(
            step_raw_indices, dtype=torch.long
        )
        self._step_raw_indices = tuple(int(index) for index in step_raw_indices)
        self._windows = tuple(windows)
        self._episode_windows = tuple(episode_windows)

    @property
    def episode_windows(self) -> tuple[tuple[int, tuple[int, ...]], ...]:
        return self._episode_windows

    def __len__(self) -> int:
        return len(self._windows)

    def _load_step(self, step_dataset_index: int) -> dict:
        # RobotVideoDataset.__getitem__ falls back to a random sample after an
        # exception. EATB must fail instead because substitution breaks the
        # episode state sequence.
        return self._step_dataset._get(int(step_dataset_index))

    def _window_sample(self, window_index: int) -> dict:
        window = self._windows[int(window_index)]
        steps = [self._load_step(index) for index in window.step_dataset_indices]
        if not steps:
            raise RuntimeError(f"EATB window {window_index} contains no valid replans.")

        sample = dict(steps[0])
        sample.pop("_profile", None)
        for key in self._TEMPORAL_SAMPLE_KEYS:
            if key not in steps[0]:
                sample.pop(key, None)
                continue
            values = [step[key] for step in steps]
            values.extend(torch.zeros_like(values[0]) for _ in range(self.num_chunks - len(values)))
            sample[key] = torch.stack(values, dim=0)

        valid_count = len(steps)
        chunk_ids = torch.full((self.num_chunks,), -1, dtype=torch.long)
        chunk_ids[:valid_count] = torch.arange(
            window.start_chunk_id,
            window.start_chunk_id + valid_count,
            dtype=torch.long,
        )
        chunk_mask = torch.zeros(self.num_chunks, dtype=torch.bool)
        chunk_mask[:valid_count] = True

        raw_indices = [self._step_raw_indices[index] for index in window.step_dataset_indices]
        episode_start = int(self.episode_dataset._edit_memory_episode_from[window.episode_id].item())
        timesteps = torch.zeros(self.num_chunks, dtype=torch.long)
        timesteps[:valid_count] = torch.as_tensor(
            [raw_index - episode_start for raw_index in raw_indices], dtype=torch.long
        )
        episode_ids = torch.full((self.num_chunks,), int(window.episode_id), dtype=torch.long)

        sample.update(
            {
                "edit_memory_episode_id": episode_ids,
                "edit_memory_timestep": timesteps,
                "edit_memory_history_stride": torch.tensor(
                    int(self.episode_dataset.edit_memory_history_stride), dtype=torch.long
                ),
                "eatb_episode_id": torch.tensor(window.episode_id, dtype=torch.long),
                "eatb_start_chunk_id": torch.tensor(window.start_chunk_id, dtype=torch.long),
                "eatb_num_chunks": torch.tensor(valid_count, dtype=torch.long),
                "eatb_chunk_ids": chunk_ids,
                "eatb_chunk_mask": chunk_mask,
                "eatb_is_last_window": torch.tensor(window.is_last_window, dtype=torch.bool),
            }
        )
        return sample

    def __getitem__(self, index: int) -> dict:
        index = int(index)
        if index >= 0:
            return self._window_sample(index)

        # Official EATB pads inactive batch rows with a negative episode id and
        # an all-false chunk mask while preserving static tensor shapes.
        sample = self._window_sample(0)
        for key in self._TEMPORAL_SAMPLE_KEYS:
            value = sample.get(key)
            if isinstance(value, torch.Tensor):
                sample[key] = torch.zeros_like(value)
        sample["edit_memory_episode_id"] = torch.full(
            (self.num_chunks,), index, dtype=torch.long
        )
        sample["edit_memory_timestep"] = torch.zeros(self.num_chunks, dtype=torch.long)
        sample["eatb_episode_id"] = torch.tensor(index, dtype=torch.long)
        sample["eatb_start_chunk_id"] = torch.tensor(0, dtype=torch.long)
        sample["eatb_num_chunks"] = torch.tensor(0, dtype=torch.long)
        sample["eatb_chunk_ids"] = torch.full((self.num_chunks,), -1, dtype=torch.long)
        sample["eatb_chunk_mask"] = torch.zeros(self.num_chunks, dtype=torch.bool)
        sample["eatb_is_last_window"] = torch.tensor(True, dtype=torch.bool)
        for key in ("context", "context_mask", "text_hidden_states", "text_attention_mask"):
            value = sample.get(key)
            if isinstance(value, torch.Tensor):
                sample[key] = torch.zeros_like(value)
        return sample


class EATBDistributedBatchSampler(Sampler[list[int]]):
    """Shard fixed episode lanes across ranks without migrating episode state."""

    def __init__(
        self,
        dataset: EATBWindowDataset,
        *,
        local_batch_size: int,
        num_replicas: int,
        rank: int,
    ) -> None:
        self.dataset = dataset
        self.local_batch_size = int(local_batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        if self.local_batch_size <= 0:
            raise ValueError("EATB local_batch_size must be positive.")
        if self.num_replicas <= 0 or not 0 <= self.rank < self.num_replicas:
            raise ValueError(
                f"Invalid EATB distributed layout: rank={rank}, num_replicas={num_replicas}."
            )
        self.global_batch_size = self.local_batch_size * self.num_replicas
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int) -> None:
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int) -> None:
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self) -> None:
        self.resume_batch_offset = 0

    def _global_batches(self) -> Iterator[list[int]]:
        episodes = self.dataset.episode_windows
        for cohort_start in range(0, len(episodes), self.global_batch_size):
            cohort = list(episodes[cohort_start : cohort_start + self.global_batch_size])
            cohort.extend(
                (-(cohort_start + lane + 1), tuple())
                for lane in range(len(cohort), self.global_batch_size)
            )
            rounds = max((len(windows) for _, windows in cohort), default=0)
            for round_index in range(rounds):
                batch = []
                for lane_index, (_, windows) in enumerate(cohort):
                    if round_index < len(windows):
                        batch.append(int(windows[round_index]))
                    else:
                        batch.append(-(cohort_start + lane_index + 1))
                yield batch

    def __iter__(self) -> Iterator[list[int]]:
        start = self.rank * self.local_batch_size
        stop = start + self.local_batch_size
        for batch_index, global_batch in enumerate(self._global_batches()):
            if batch_index < self.resume_batch_offset:
                continue
            yield global_batch[start:stop]

    def __len__(self) -> int:
        return sum(1 for _ in self._global_batches())
