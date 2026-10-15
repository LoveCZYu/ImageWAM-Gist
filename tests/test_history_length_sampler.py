from imagewam.utils.samplers import ResumableEpochSampler


class _HistoryDataset:
    def __len__(self):
        return 160

    @staticmethod
    def edit_memory_history_length(index: int) -> int:
        return (int(index) * 37) % 160


def test_global_batches_are_history_length_grouped() -> None:
    dataset = _HistoryDataset()
    sampler = ResumableEpochSampler(
        dataset,
        seed=7,
        batch_size=1,
        num_processes=16,
        history_length_bucketing=True,
        history_bucket_multiplier=10,
    )
    indices = list(iter(sampler))
    lengths = [dataset.edit_memory_history_length(index) for index in indices]
    for start in range(0, len(lengths), 16):
        batch = lengths[start : start + 16]
        assert batch == sorted(batch, reverse=True)


if __name__ == "__main__":
    test_global_batches_are_history_length_grouped()
    print("PASS test_global_batches_are_history_length_grouped")
