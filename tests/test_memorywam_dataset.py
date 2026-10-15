import torch

from imagewam.datasets.lerobot.robot_video_dataset import (
    RobotVideoDataset,
    compose_robotwin_camera_video,
)


def _dataset(mode: str, max_steps: int) -> RobotVideoDataset:
    dataset = RobotVideoDataset.__new__(RobotVideoDataset)
    dataset._edit_memory_episode_from = torch.tensor([0, 100])
    dataset._edit_memory_episode_to = torch.tensor([100, 200])
    dataset.edit_memory_history_mode = mode
    dataset.edit_memory_history_stride = 16
    dataset.edit_memory_history_max_steps = max_steps
    dataset.edit_memory_history_anchor_frames = 2
    dataset.edit_memory_history_offsets = [48, 32, 16]
    dataset.sample_index_stride = 1
    dataset._edit_memory_sample_raw_indices = None
    return dataset


def test_full_episode_history_uses_every_prior_replan() -> None:
    dataset = _dataset("episode", max_steps=0)
    assert dataset._edit_memory_history_raw_indices(170) == [100, 116, 132, 148, 164]


def test_bounded_history_preserves_anchors_and_latest_replan() -> None:
    dataset = _dataset("episode", max_steps=3)
    assert dataset._edit_memory_history_raw_indices(170) == [100, 116, 164]


def test_history_never_crosses_episode_boundary() -> None:
    dataset = _dataset("offsets", max_steps=0)
    assert dataset._edit_memory_history_raw_indices(110) == [100]


def test_episode_aligned_sample_indices_match_replans() -> None:
    dataset = _dataset("episode", max_steps=0)
    dataset._edit_memory_sample_raw_indices = torch.tensor([0, 16, 32, 100, 116])
    assert len(dataset) == 5
    assert dataset._edit_memory_resolve_sample_index(0) == 0
    assert dataset._edit_memory_resolve_sample_index(3) == 100
    assert dataset._edit_memory_history_raw_indices(116) == [100]


def test_shared_robotwin_camera_composition_has_expected_layout() -> None:
    cameras = torch.stack(
        [
            torch.full((1, 3, 12, 16), 0.25),
            torch.full((1, 3, 12, 16), 0.50),
            torch.full((1, 3, 12, 16), 0.75),
        ],
        dim=0,
    )
    composed = compose_robotwin_camera_video(cameras, "compact_288x256")
    assert tuple(composed.shape) == (1, 3, 288, 256)
    assert torch.allclose(composed[..., :192, :], torch.full((1, 3, 192, 256), 0.25))
    assert torch.allclose(composed[..., 192:, :128], torch.full((1, 3, 96, 128), 0.50))
    assert torch.allclose(composed[..., 192:, 128:], torch.full((1, 3, 96, 128), 0.75))


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
