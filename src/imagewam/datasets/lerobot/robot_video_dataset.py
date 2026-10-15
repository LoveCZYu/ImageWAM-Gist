import hashlib
import os
import time
from typing import Optional
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from imagewam.utils.logging_config import get_logger
from imagewam.utils import misc, pytorch_utils
from imagewam.utils.mem_tools import PeriodicTrim
from accelerate import PartialState
logger = get_logger(__name__)

# export IMAGEWAM_MEM_TRIM_EVERY=50          
# export IMAGEWAM_M_TRIM_THRESHOLD=65536    
# export IMAGEWAM_M_MMAP_THRESHOLD=65536   
# export IMAGEWAM_M_TOP_PAD=0
# # export MALLOC_ARENA_MAX=2


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def episode_replan_sample_indices(
    episode_start: int,
    episode_end: int,
    stride: int,
    *,
    current_horizon: int,
    terminal_only: bool,
) -> list[int]:
    aligned_end = int(episode_end) - max(int(current_horizon), 0)
    indices = list(
        range(
            int(episode_start),
            max(int(episode_start), aligned_end),
            int(stride),
        )
    )
    return indices[-1:] if terminal_only and indices else indices


def robotwin_camera_sizes(layout: str) -> tuple[list[int], list[int], list[int]]:
    layout = str(layout).strip().lower()
    if layout in {"compact", "compact_288x256", "288x256"}:
        return [192, 256], [96, 128], [96, 128]
    if layout in {"legacy", "legacy_384x320", "384x320"}:
        return [256, 320], [128, 160], [128, 160]
    raise ValueError(
        f"Unsupported robotwin_camera_layout={layout!r}. "
        "Expected one of: compact_288x256, legacy_384x320."
    )


def compose_robotwin_camera_video(video: torch.Tensor, layout: str) -> torch.Tensor:
    """Compose three preprocessed cameras identically in training and deployment."""
    if video.ndim != 5 or int(video.shape[0]) != 3:
        raise ValueError(f"RobotWin layout requires [3,T,C,H,W], got {tuple(video.shape)}")
    top_size, left_size, right_size = robotwin_camera_sizes(layout)
    top = transforms_F.resize(video[0], top_size, antialias=True)
    left = transforms_F.resize(video[1], left_size, antialias=True)
    right = transforms_F.resize(video[2], right_size, antialias=True)
    return torch.cat([top, torch.cat([left, right], dim=-1)], dim=-2)


class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        val_split_level: str = "episode",
        global_sample_stride=1,
        sample_index_stride: int = 1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        robotwin_camera_layout: str = "compact_288x256",
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        require_text_cache: bool = True,
        qwen_text_cache_dir: Optional[str] = None,
        qwen_context_len: int = 128,
        qwen_text_cache_format: str = "qwen2_5_vl",
        endpoint_frames_only: bool = False,
        nonidle_filter_path: Optional[str] = None,
        profile_getitem: bool = False,
        condition_frame_augmentation: Optional[dict] = None,
        video_augmentation: Optional[dict] = None,
        hetero_bridge: Optional[dict] = None,
        lerobot_meta_cache: Optional[str] = None,
        arrow_cache_dir: Optional[str] = None,
        lerobot_backend: str = "v2",
        lerobot_video_backend: Optional[str] = None,
        lerobot_v3_init_num_workers: int = 1,
        lerobot_v3_index_cache: Optional[str] = None,
        lerobot_v3_video_backend: Optional[str] = None,
        lerobot_tolerance_s: Optional[float] = None,
        episode_index_filter: Optional[dict] = None,
        slow_getitem_log_sec: float = 0.0,
        edit_memory_history_mode: str = "none",
        edit_memory_history_offsets: Optional[list[int]] = None,
        edit_memory_history_stride: int = 16,
        edit_memory_history_max_steps: int = 0,
        edit_memory_history_anchor_frames: int = 2,
        edit_memory_history_use_aug: bool = False,
        edit_memory_replan_aligned_samples: bool = False,
        edit_memory_drop_incomplete_current: bool = False,
        edit_memory_episode_terminal_samples_only: bool = False,
    ):
        image_obs_indices = [0, num_frames - 1] if endpoint_frames_only else None
        self.slow_getitem_log_sec = float(
            os.environ.get("IMAGEWAM_SLOW_GETITEM_LOG_SEC", slow_getitem_log_sec)
        )
        effective_profile_getitem = bool(profile_getitem) or self.slow_getitem_log_sec > 0.0
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            val_split_level=val_split_level,
            global_sample_stride=global_sample_stride,
            sample_index_stride=sample_index_stride,
            image_obs_indices=image_obs_indices,
            nonidle_filter_path=nonidle_filter_path,
            profile_getitem=effective_profile_getitem,
            hetero_bridge=OmegaConf.to_container(hetero_bridge, resolve=True) if isinstance(hetero_bridge, DictConfig) else hetero_bridge,
            lerobot_meta_cache=lerobot_meta_cache,
            arrow_cache_dir=arrow_cache_dir,
            lerobot_backend=lerobot_backend,
            lerobot_video_backend=lerobot_video_backend,
            lerobot_v3_init_num_workers=lerobot_v3_init_num_workers,
            lerobot_v3_index_cache=lerobot_v3_index_cache,
            lerobot_v3_video_backend=lerobot_v3_video_backend,
            lerobot_tolerance_s=lerobot_tolerance_s,
            episode_index_filter=OmegaConf.to_container(episode_index_filter, resolve=True) if isinstance(episode_index_filter, DictConfig) else episode_index_filter,
        )
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.robotwin_camera_layout = str(robotwin_camera_layout)
        self.override_instruction = override_instruction
        self.require_text_cache = bool(require_text_cache)
        self.qwen_text_cache_dir = qwen_text_cache_dir
        self.qwen_context_len = int(qwen_context_len)
        self.qwen_text_cache_format = str(qwen_text_cache_format)
        self.endpoint_frames_only = bool(endpoint_frames_only)
        self.sample_index_stride = int(sample_index_stride)
        self.edit_memory_history_mode = str(edit_memory_history_mode).strip().lower()
        if self.edit_memory_history_mode not in {"none", "offsets", "episode"}:
            raise ValueError(
                "`edit_memory_history_mode` must be one of: none, offsets, episode; "
                f"got {edit_memory_history_mode!r}."
            )
        self.edit_memory_history_offsets = sorted(
            {int(offset) for offset in (edit_memory_history_offsets or []) if int(offset) > 0},
            reverse=True,
        )
        self.edit_memory_history_stride = int(edit_memory_history_stride)
        if self.edit_memory_history_stride <= 0:
            raise ValueError("`edit_memory_history_stride` must be positive.")
        self.edit_memory_history_max_steps = int(edit_memory_history_max_steps)
        if self.edit_memory_history_max_steps < 0:
            raise ValueError("`edit_memory_history_max_steps` must be >= 0; 0 means full episode history.")
        self.edit_memory_history_anchor_frames = max(int(edit_memory_history_anchor_frames), 0)
        self.edit_memory_history_use_aug = bool(edit_memory_history_use_aug)
        self.edit_memory_replan_aligned_samples = bool(edit_memory_replan_aligned_samples)
        self.edit_memory_drop_incomplete_current = bool(edit_memory_drop_incomplete_current)
        self.edit_memory_episode_terminal_samples_only = bool(
            edit_memory_episode_terminal_samples_only
        )
        self._edit_memory_sample_raw_indices: torch.Tensor | None = None
        if self.edit_memory_history_mode != "none":
            episode_index = self.lerobot_dataset.episode_data_index
            self._edit_memory_episode_from = torch.as_tensor(
                episode_index["from"], dtype=torch.long
            ).cpu()
            self._edit_memory_episode_to = torch.as_tensor(
                episode_index["to"], dtype=torch.long
            ).cpu()
            if self.edit_memory_replan_aligned_samples:
                if self.sample_index_stride != 1:
                    raise ValueError(
                        "edit_memory_replan_aligned_samples requires sample_index_stride=1 "
                        "so episode-relative replan indices map exactly to dataset rows."
                    )
                aligned = []
                for episode_start, episode_end in zip(
                    self._edit_memory_episode_from.tolist(),
                    self._edit_memory_episode_to.tolist(),
                ):
                    aligned.extend(
                        episode_replan_sample_indices(
                            episode_start,
                            episode_end,
                            self.edit_memory_history_stride,
                            current_horizon=(
                                self.num_frames - 1
                                if self.edit_memory_drop_incomplete_current
                                else 0
                            ),
                            terminal_only=self.edit_memory_episode_terminal_samples_only,
                        )
                    )
                self._edit_memory_sample_raw_indices = torch.as_tensor(aligned, dtype=torch.long)
        else:
            self._edit_memory_episode_from = None
            self._edit_memory_episode_to = None
        self.profile_getitem = effective_profile_getitem
        augmentation_cfg = video_augmentation if video_augmentation is not None else condition_frame_augmentation
        if augmentation_cfg is not None and is_training_set:
            # Hydra's instantiate(..., recursive=True) may have already built nested
            # _target_ configs; do not call instantiate() on an nn.Module twice.
            if isinstance(augmentation_cfg, torch.nn.Module):
                self.video_augmentation = augmentation_cfg
            else:
                self.video_augmentation = instantiate(augmentation_cfg)
        else:
            self.video_augmentation = None

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)

        # Per-worker periodic memory reclaim. Each __getitem__ leaves behind
        # small Python allocations (HF row dicts, list comprehensions,
        # pyav-decoded buffers, ...) that pile up as free chunks in the
        # worker's glibc arena. glibc only auto-trims when the free
        # top-of-heap exceeds M_TRIM_THRESHOLD (128 KB by default), so RSS
        # tends to ratchet up between auto-trims. Setting
        # IMAGEWAM_MEM_TRIM_EVERY=N forces gc.collect() + malloc_trim(0) every
        # N samples per worker (50-200 is a reasonable starting point).
        self._mem_trim = PeriodicTrim(
            every=int(os.environ.get("IMAGEWAM_MEM_TRIM_EVERY", "0")),
            do_gc=os.environ.get("IMAGEWAM_MEM_TRIM_GC", "1") != "0",
        )

    def __len__(self):
        if self._edit_memory_sample_raw_indices is not None:
            return int(self._edit_memory_sample_raw_indices.numel())
        return len(self.lerobot_dataset)

    def _edit_memory_resolve_sample_index(self, idx: int) -> int:
        if self._edit_memory_sample_raw_indices is None:
            return int(idx)
        if idx < 0 or idx >= int(self._edit_memory_sample_raw_indices.numel()):
            raise IndexError(f"MemoryWAM sample index {idx} is out of bounds for {len(self)} aligned replans.")
        return self._edit_memory_dataset_index(int(self._edit_memory_sample_raw_indices[idx].item()))

    @staticmethod
    def _robotwin_camera_sizes(layout: str) -> tuple[list[int], list[int], list[int]]:
        return robotwin_camera_sizes(layout)

    def _edit_memory_episode_bounds(self, raw_idx: int) -> tuple[int, int]:
        if self._edit_memory_episode_from is None or self._edit_memory_episode_to is None:
            raise RuntimeError("Episode indices are unavailable while edit-memory history is enabled.")
        raw = torch.tensor(int(raw_idx), dtype=torch.long)
        episode_pos = int(torch.searchsorted(self._edit_memory_episode_to, raw, right=True).item())
        episode_pos = min(max(episode_pos, 0), int(self._edit_memory_episode_to.numel()) - 1)
        return (
            int(self._edit_memory_episode_from[episode_pos].item()),
            int(self._edit_memory_episode_to[episode_pos].item()),
        )

    def _edit_memory_identity(self, raw_idx: int) -> tuple[int, int]:
        if self._edit_memory_episode_from is None or self._edit_memory_episode_to is None:
            raise RuntimeError("Episode indices are unavailable while edit-memory history is enabled.")
        raw = torch.tensor(int(raw_idx), dtype=torch.long)
        episode_pos = int(torch.searchsorted(self._edit_memory_episode_to, raw, right=True).item())
        episode_pos = min(max(episode_pos, 0), int(self._edit_memory_episode_to.numel()) - 1)
        episode_start = int(self._edit_memory_episode_from[episode_pos].item())
        episode_end = int(self._edit_memory_episode_to[episode_pos].item())
        if not episode_start <= int(raw_idx) < episode_end:
            raise IndexError(
                f"Raw frame {raw_idx} is outside episode {episode_pos} bounds "
                f"[{episode_start}, {episode_end})."
            )
        return episode_pos, int(raw_idx) - episode_start

    def _edit_memory_dataset_index(self, raw_idx: int) -> int:
        return max(0, int(raw_idx) // max(self.sample_index_stride, 1))

    def _edit_memory_history_raw_indices(self, raw_idx: int) -> list[int]:
        episode_start, episode_end = self._edit_memory_episode_bounds(raw_idx)
        current = max(episode_start, min(int(raw_idx), episode_end - 1))
        if self.edit_memory_history_mode == "offsets":
            indices = [max(episode_start, current - offset) for offset in self.edit_memory_history_offsets]
        elif self.edit_memory_history_mode == "episode":
            indices = list(range(episode_start, current, self.edit_memory_history_stride))
        else:
            return []
        indices = sorted(set(index for index in indices if index < current))
        limit = self.edit_memory_history_max_steps
        if limit > 0 and len(indices) > limit:
            anchors = indices[: min(self.edit_memory_history_anchor_frames, limit)]
            recent = indices[-max(limit - len(anchors), 0) :]
            indices = sorted(set(anchors + recent))
        return indices

    def edit_memory_history_length(self, idx: int) -> int:
        """Cheap exact history length for distributed length bucketing."""
        if self.edit_memory_history_mode == "none":
            return 0
        if self._edit_memory_sample_raw_indices is not None:
            raw_idx = int(self._edit_memory_sample_raw_indices[int(idx)].item())
        else:
            raw_idx = int(idx) * max(self.sample_index_stride, 1)
        episode_start, episode_end = self._edit_memory_episode_bounds(raw_idx)
        current = max(episode_start, min(raw_idx, episode_end - 1))
        if self.edit_memory_history_mode == "episode":
            count = (max(current - episode_start, 0) + self.edit_memory_history_stride - 1) // self.edit_memory_history_stride
        else:
            count = len(
                {
                    max(episode_start, current - offset)
                    for offset in self.edit_memory_history_offsets
                    if max(episode_start, current - offset) < current
                }
            )
        if self.edit_memory_history_max_steps > 0:
            count = min(count, self.edit_memory_history_max_steps)
        return int(count)

    def _edit_memory_sample_to_video(self, sample: dict, *, apply_augmentation: bool) -> torch.Tensor:
        video = sample["pixel_values"]
        num_cameras = 1
        if video.ndim == 5:
            if not self.endpoint_frames_only:
                video = video[:, self.video_sample_indices]
            num_cameras, time_steps, channels, height, width = video.shape
        else:
            if video.ndim != 4:
                raise ValueError(f"Expected [T,C,H,W] or [V,T,C,H,W], got {tuple(video.shape)}")
            if not self.endpoint_frames_only:
                video = video[self.video_sample_indices]
            time_steps, channels, height, width = video.shape
        video = video.view(num_cameras, time_steps, channels, height, width)
        if apply_augmentation and self.video_augmentation is not None:
            video = self.video_augmentation(video)
        if self.concat_multi_camera == "robotwin":
            video = compose_robotwin_camera_video(video, self.robotwin_camera_layout)
        elif num_cameras > 1:
            concat_dim = -1 if self.concat_multi_camera == "horizontal" else -2
            if self.concat_multi_camera not in {"horizontal", "vertical"}:
                raise ValueError(f"Unsupported concat_multi_camera={self.concat_multi_camera!r}.")
            video = torch.cat([video[index] for index in range(num_cameras)], dim=concat_dim)
        else:
            video = video.squeeze(0)
        video = self.normalize_transform(self.crop_transform(self.resize_transform(video)))
        return video.permute(1, 0, 2, 3)

    def _get(self, idx):
        sample_idx = self._edit_memory_resolve_sample_index(int(idx))
        sample = None
        profile = {} if self.profile_getitem else None
        t0 = time.perf_counter()

        def _mark(name: str):
            nonlocal t0
            if profile is None:
                return
            now = time.perf_counter()
            profile[f"robot.{name}"] = now - t0
            t0 = now

        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]
            if profile is not None and "_profile" in sample:
                profile.update(sample["_profile"])
            _mark("lerobot_dataset_get")

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            retry_idx = int(np.random.randint(len(self)))
            sample_idx = self._edit_memory_resolve_sample_index(retry_idx)
        _mark("padding_retry")

        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            if not self.endpoint_frames_only:
                video = video[:, self.video_sample_indices, :, :, :] # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            if not self.endpoint_frames_only:
                video = video[self.video_sample_indices, :, :, :] # [T_video, C, H, W]
            T_video, C, H, W = video.shape
        if not self.endpoint_frames_only:
            image_is_pad = image_is_pad[self.video_sample_indices]
        _mark("video_select")

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        if self.video_augmentation is not None:
            video = self.video_augmentation(video)

        if self.concat_multi_camera == "robotwin":
            video = compose_robotwin_camera_video(video, self.robotwin_camera_layout)
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        # final resize and normalization
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)  # [T_video, C, H, W]
        video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]
        _mark("image_postprocess")

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        expected_video_transitions = (self.num_frames - 1) // self.action_video_freq_ratio
        if self.endpoint_frames_only:
            expected_video_transitions = max(expected_video_transitions, 1)
        else:
            expected_video_transitions = video.shape[1] - 1
        if action.shape[0] % expected_video_transitions != 0:
            raise ValueError(
                f"`action` horizon must be divisible by video transitions, got {action.shape[0]} and {expected_video_transitions}"
            )

        task = sample["instruction"]
        
        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "instruction": instruction,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        if "action_dim_is_pad" in sample:
            data["action_dim_is_pad"] = sample["action_dim_is_pad"]
        if "proprio_dim_is_pad" in sample:
            data["proprio_dim_is_pad"] = sample["proprio_dim_is_pad"]
        if "embodiment" in sample:
            data["embodiment"] = sample["embodiment"]
        if self.edit_memory_history_mode != "none":
            raw_idx = int(sample.get("idx", sample_idx * max(self.sample_index_stride, 1)))
            history_raw_indices = self._edit_memory_history_raw_indices(raw_idx)
            current_episode_id, current_timestep = self._edit_memory_identity(raw_idx)
            history_identities = [
                self._edit_memory_identity(history_raw_idx) for history_raw_idx in history_raw_indices
            ]
            history_samples = [
                self.lerobot_dataset[self._edit_memory_dataset_index(history_raw_idx)]
                for history_raw_idx in history_raw_indices
            ]
            if history_samples:
                data["edit_memory_video"] = torch.stack(
                    [
                        self._edit_memory_sample_to_video(
                            history_sample,
                            apply_augmentation=self.edit_memory_history_use_aug,
                        )
                        for history_sample in history_samples
                    ],
                    dim=0,
                )
                data["edit_memory_action"] = torch.stack(
                    [history_sample["action"] for history_sample in history_samples], dim=0
                )
                data["edit_memory_proprio"] = torch.stack(
                    [history_sample["proprio"][:-1] for history_sample in history_samples], dim=0
                )
                data["edit_memory_action_is_pad"] = torch.stack(
                    [history_sample["action_is_pad"] for history_sample in history_samples], dim=0
                )
                if "action_dim_is_pad" in sample:
                    data["edit_memory_action_dim_is_pad"] = torch.stack(
                        [
                            history_sample.get("action_dim_is_pad", sample["action_dim_is_pad"])
                            for history_sample in history_samples
                        ],
                        dim=0,
                    )
            else:
                data["edit_memory_video"] = video.new_empty((0, *video.shape))
                data["edit_memory_action"] = action.new_empty((0, *action.shape))
                data["edit_memory_proprio"] = proprio.new_empty((0, *proprio.shape))
                data["edit_memory_action_is_pad"] = sample["action_is_pad"].new_empty(
                    (0, *sample["action_is_pad"].shape)
                )
                if "action_dim_is_pad" in sample:
                    data["edit_memory_action_dim_is_pad"] = sample["action_dim_is_pad"].new_empty(
                        (0, *sample["action_dim_is_pad"].shape)
                    )
            data["edit_memory_valid"] = torch.ones(len(history_samples), dtype=torch.bool)
            data["edit_memory_raw_idx"] = torch.as_tensor(history_raw_indices, dtype=torch.long)
            data["edit_memory_history_stride"] = torch.tensor(
                self.edit_memory_history_stride, dtype=torch.long
            )
            data["edit_memory_episode_id"] = torch.tensor(current_episode_id, dtype=torch.long)
            data["edit_memory_timestep"] = torch.tensor(current_timestep, dtype=torch.long)
            data["edit_memory_history_episode_id"] = torch.as_tensor(
                [identity[0] for identity in history_identities], dtype=torch.long
            )
            data["edit_memory_history_timestep"] = torch.as_tensor(
                [identity[1] for identity in history_identities], dtype=torch.long
            )
        if self.require_text_cache:
            context, context_mask = self._get_cached_text_context(instruction)
            # NOTE: to keep consistent with wan2.2's behavior
            context[~context_mask] = 0.0
            context_mask = torch.ones_like(context_mask)
            data["context"] = context
            data["context_mask"] = context_mask
        if self.qwen_text_cache_dir is not None:
            text_hidden_states, text_attention_mask = self._get_cached_qwen_context(instruction)
            data["text_hidden_states"] = text_hidden_states
            data["text_attention_mask"] = text_attention_mask
        _mark("qwen_cache")
        if profile is not None:
            data["_profile"] = profile
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/omnigen2/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def _get_cached_qwen_context(self, prompt: str):
        if self.qwen_text_cache_dir is None:
            raise ValueError("qwen_text_cache_dir is not set.")
        cache_dir = self.qwen_text_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        suffix_by_format = {
            "qwen2_5_vl": "qwen2_5_vl",
            "qwen3_flux2": "qwen3_flux2",
        }
        if self.qwen_text_cache_format not in suffix_by_format:
            raise ValueError(
                f"Unsupported qwen_text_cache_format={self.qwen_text_cache_format!r}; "
                "expected 'qwen2_5_vl' or 'qwen3_flux2'."
            )
        suffix = suffix_by_format[self.qwen_text_cache_format]
        cache_path = os.path.join(cache_dir, f"{hashed}.{suffix}_len{self.qwen_context_len}.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing Qwen text embedding cache: {cache_path}. "
                "Use qwen_text_cache_format='qwen2_5_vl' with scripts/omnigen2/precompute_qwen_embeds.py, "
                "or qwen_text_cache_format='qwen3_flux2' with scripts/flux2/precompute_flux2_qwen3_embeds.py."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["text_hidden_states"]
        context_mask = payload["text_attention_mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `text_hidden_states` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `text_attention_mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.qwen_context_len:
            raise ValueError(
                f"Cached qwen_context_len mismatch: expected {self.qwen_context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.qwen_context_len:
            raise ValueError(
                f"Cached qwen mask length mismatch: expected {self.qwen_context_len}, got {context_mask.shape[0]} in {cache_path}"
            )
        return context, context_mask

    def __getitem__(self, idx):
        t0 = time.perf_counter()
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        elapsed = time.perf_counter() - t0
        if self.slow_getitem_log_sec > 0.0 and elapsed >= self.slow_getitem_log_sec:
            profile = data.get("_profile", {})
            slow_parts = []
            if isinstance(profile, dict):
                time_profile = []
                counter_profile = []
                for key, value in profile.items():
                    if not isinstance(value, (int, float)):
                        continue
                    row = (key, float(value))
                    if (
                        key.endswith(".calls")
                        or key.endswith(".requested_frames")
                        or key.endswith(".frame_span")
                        or key.endswith(".max_frame_index")
                        or key.endswith(".frame_span_per_call_max")
                        or key.endswith(".max_frame_index_per_call_max")
                        or key.endswith(".frames_decoded")
                        or key.endswith(".pyav_eof_fallbacks")
                    ):
                        counter_profile.append(row)
                    else:
                        time_profile.append(row)
                for key, value in sorted(time_profile, key=lambda item: item[1], reverse=True)[:12]:
                    slow_parts.append(f"{key}={value:.3f}s")
                remaining = max(0, 12 - len(slow_parts))
                for key, value in sorted(counter_profile, key=lambda item: item[1], reverse=True)[:remaining]:
                    slow_parts.append(f"{key}={value:.1f}")
            logger.warning(
                "[slow-getitem] idx=%s elapsed=%.3fs %s",
                idx,
                elapsed,
                " | ".join(slow_parts),
            )
        # Force glibc to return free pages periodically (no-op when disabled).
        self._mem_trim.tick()
        return data
