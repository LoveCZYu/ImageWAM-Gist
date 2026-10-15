import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robotwin.imagewam_policy.deploy_policy import WorldActionRobotWinPolicy
from imagewam.datasets.lerobot.robot_video_dataset import compose_robotwin_camera_video
from imagewam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json


def test_robotwin_training_and_deployment_image_preprocessing_are_exact() -> None:
    rng = np.random.default_rng(17)
    raw = [rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8) for _ in range(3)]
    shape_meta = {
        "images": [
            {"key": "cam_high", "shape": [3, 240, 320]},
            {"key": "cam_left_wrist", "shape": [3, 240, 320]},
            {"key": "cam_right_wrist", "shape": [3, 240, 320]},
        ]
    }
    policy = WorldActionRobotWinPolicy.__new__(WorldActionRobotWinPolicy)
    policy.processor = SimpleNamespace(shape_meta=shape_meta)
    policy.model = SimpleNamespace(device=torch.device("cpu"), torch_dtype=torch.float32)
    policy.robotwin_camera_layout = "compact_288x256"
    observation = {
        "observation": {
            "head_camera": {"rgb": raw[0]},
            "left_camera": {"rgb": raw[1]},
            "right_camera": {"rgb": raw[2]},
        }
    }

    deploy = policy._build_robotwin_image_tensor(observation)
    training_cameras = []
    for image, meta in zip(raw, shape_meta["images"]):
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        training_cameras.append(transforms_F.resize(tensor, meta["shape"][-2:], antialias=True))
    training = compose_robotwin_camera_video(
        torch.stack(training_cameras, dim=0).unsqueeze(1),
        "compact_288x256",
    )[0].unsqueeze(0)
    training = training * 2.0 - 1.0

    diff = (deploy - training).abs()
    print(
        "METRIC robotwin_image exact=%s max_abs=%.8f mean_abs=%.8f mismatch_frac=%.8f"
        % (
            torch.equal(deploy, training),
            float(diff.max().item()),
            float(diff.mean().item()),
            float((diff != 0).float().mean().item()),
        )
    )
    assert torch.equal(deploy, training)


def test_action_normalization_and_deployment_decode_roundtrip() -> None:
    stats_path = os.environ.get("MEMORYWAM_DATASET_STATS")
    if not stats_path:
        print("SKIP action_decode MEMORYWAM_DATASET_STATS is not set")
        return
    os.environ.setdefault("IMAGEWAM_QWEN_CACHE", "/tmp/memorywam-qwen-cache")
    with initialize_config_dir(config_dir=str((PROJECT_ROOT / "configs").resolve()), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["task=robotwin_flux2_klein_4b_base_memorywam", "memory=fixed"],
        )
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(stats_path))
    policy = WorldActionRobotWinPolicy.__new__(WorldActionRobotWinPolicy)
    policy.processor = processor
    field = processor.normalizer.normalizers["action"]["default"]
    mean = field.stats["mean"].float().reshape(1, 1, -1)
    std = field.stats["std"].float().reshape(1, 1, -1)
    offsets = torch.linspace(-0.1, 0.1, 16).reshape(1, 16, 1)
    raw_action = mean + offsets * std
    normalized = field.forward(raw_action.clone())
    decoded = torch.from_numpy(policy._denormalize_action(normalized))
    diff = (decoded - raw_action).abs()
    print(
        "METRIC action_decode max_abs=%.8f mean_abs=%.8f"
        % (float(diff.max().item()), float(diff.mean().item()))
    )
    assert torch.allclose(decoded, raw_action, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
