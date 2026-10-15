import os
from pathlib import Path

from hydra import compose, initialize_config_dir


def main() -> None:
    os.environ.setdefault("IMAGEWAM_QWEN_CACHE", "/tmp/memorywam-qwen-cache")
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    expected = {
        "fixed": (False, False),
        "adaptive_span": (True, False),
        "adaptive_gist": (False, True),
        "adaptive_both": (True, True),
    }
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        for mode, switches in expected.items():
            config = compose(
                config_name="train",
                overrides=["task=robotwin_flux2_klein_4b_base_memorywam", f"memory={mode}"],
            )
            memory = config.model.edit_memory_config
            assert (bool(memory.adaptive_span.enabled), bool(memory.adaptive_gist.enabled)) == switches
            assert (int(memory.bank.anchor_frames), int(memory.bank.recent_frames)) == (2, 4)
            assert int(memory.bank.gist_tokens_per_frame) == 8
            assert str(memory.training.memory_source) == "episode_window"
            assert bool(config.data.train.edit_memory_replan_aligned_samples)
            assert int(config.data.train.edit_memory_history_stride) == 16
            assert int(config.data.val.edit_memory_history_stride) == 16
            print(f"PASS config={mode} span={switches[0]} gist={switches[1]}")


if __name__ == "__main__":
    main()
