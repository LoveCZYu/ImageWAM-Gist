from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


class ResumableEpochSampler(Sampler[int]):
    def __init__(
        self,
        dataset: Sized,
        seed: int,
        batch_size: int,
        num_processes: int,
        *,
        history_length_bucketing: bool = False,
        history_bucket_multiplier: int = 50,
    ):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.history_length_bucketing = bool(history_length_bucketing)
        self.history_bucket_multiplier = max(int(history_bucket_multiplier), 1)
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        length_fn = getattr(self.dataset, "edit_memory_history_length", None)
        if self.history_length_bucketing and callable(length_fn):
            global_batch = max(self.batch_size * self.num_processes, 1)
            mega_batch = global_batch * self.history_bucket_multiplier
            grouped = []
            for start in range(0, len(indices), mega_batch):
                chunk = indices[start : start + mega_batch]
                chunk.sort(key=length_fn, reverse=True)
                grouped.extend(chunk)
            indices = grouped
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)
