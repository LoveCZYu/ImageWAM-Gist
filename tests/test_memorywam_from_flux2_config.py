import os
import inspect
from pathlib import Path

from hydra import compose, initialize_config_dir

from imagewam.datasets.lerobot.robot_video_dataset import RobotVideoDataset


def main() -> None:
    os.environ["IMAGEWAM_RMBENCH_QWEN_CACHE"] = "/tmp/rmbench-qwen-cache"
    os.environ["IMAGEWAM_REPLAN_STEPS"] = "16"
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(
            config_name="train",
            overrides=["task=rmbench_flux2_klein_4b_base_memorywam_from_flux2"],
        )
    memory = config.model.edit_memory_config
    assert not bool(config.model.flux2_lora_config.enabled)
    assert not bool(config.model.flux2_lora_config.save_trainable_only)
    assert not bool(memory.adaptive_span.enabled)
    assert not bool(memory.adaptive_gist.enabled)
    assert int(memory.bank.anchor_frames) == 2
    assert int(memory.bank.recent_frames) == 4
    assert int(memory.bank.gist_tokens_per_frame) == 8
    assert int(memory.bank.max_entries) == 0
    assert str(memory.gist.temporal_embedding).lower() == "rope"
    assert str(memory.training.memory_source) == "eatb"
    assert int(memory.training.replan_steps) == 16
    assert int(memory.training.tbptt_steps) == 8
    assert not bool(memory.training.memory_curriculum.enabled)
    assert float(config.model.loss.lambda_video) == 1.0
    assert float(config.model.loss.lambda_action) == 1.0
    assert int(memory.training.memory_curriculum.teacher_only_until_step) == 28
    assert int(memory.training.memory_curriculum.transition_start_step) == 29
    assert int(memory.training.memory_curriculum.transition_end_step) == 114
    assert bool(config.data.train.edit_memory_replan_aligned_samples)
    assert bool(config.data.train.edit_memory_episode_terminal_samples_only)
    assert int(config.data.train.edit_memory_history_stride) == 16
    assert int(config.data.train.qwen_context_len) == 512
    assert int(config.model.qwen_context_len) == 512
    assert not bool(config.data.train.require_text_cache)
    assert not bool(config.data.val.require_text_cache)
    assert config.data.train.lerobot_video_backend == "pyav"
    assert config.data.val.lerobot_video_backend == "pyav"
    assert int(config.batch_size) == 1
    assert int(config.gradient_accumulation_steps) == 1
    assert str(config.model.video_scheduler.sampling_distribution) == "logit_normal"
    assert float(config.model.video_scheduler.logit_mean) == 0.0
    assert float(config.model.video_scheduler.logit_std) == 1.0
    assert str(config.model.video_scheduler.loss_weighting) == "none"
    assert float(config.model.video_scheduler.train_shift) == 5.0
    assert str(config.model.action_scheduler.sampling_distribution) == "logit_normal"
    assert float(config.model.action_scheduler.logit_mean) == 0.0
    assert float(config.model.action_scheduler.logit_std) == 1.0
    assert str(config.model.action_scheduler.loss_weighting) == "none"
    assert float(config.model.action_scheduler.train_shift) == 1.0
    assert float(config.model.action_scheduler.infer_shift) == 5.0
    augmentation = memory.training.clean_conditioning_augmentation
    assert bool(augmentation.enabled)
    assert float(augmentation.probability) == 1.0
    assert float(augmentation.ratio_min) == 0.0
    assert float(augmentation.ratio_max) == 1.0
    assert int(config.num_epochs) == 5
    assert int(config.eval_every) == 0
    assert config.resume is None
    dataset_parameters = set(inspect.signature(RobotVideoDataset.__init__).parameters)
    for split in (config.data.train, config.data.val):
        unknown = set(split.keys()) - {"_target_"} - dataset_parameters
        assert not unknown, f"Dataset config has unsupported constructor fields: {sorted(unknown)}"
    print("PASS from_flux2 full-FLUX KV8 autoregressive-episode config")


if __name__ == "__main__":
    main()
