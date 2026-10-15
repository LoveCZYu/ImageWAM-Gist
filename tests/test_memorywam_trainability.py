import types

import torch
import torch.nn as nn

from imagewam.models.backbones.imagewam import ImageWAM
from imagewam.trainer import Wan22Trainer


class _VideoExpert(nn.Module):
    def __init__(self, *, lora_enabled: bool = True) -> None:
        super().__init__()
        self.flux2_lora_enabled = bool(lora_enabled)
        self.block = nn.Module()
        self.block.register_parameter("weight", nn.Parameter(torch.ones(2, 2)))
        self.block.register_parameter("lora_A", nn.Parameter(torch.ones(1, 2)))
        self.block.register_parameter("lora_B", nn.Parameter(torch.ones(2, 1)))


class _TrainableModel(nn.Module):
    def __init__(self, *, lora_enabled: bool = True) -> None:
        super().__init__()
        self.stack = "flux2"
        self.mot = nn.Module()
        self.mot.mixtures = nn.ModuleDict(
            {
                "video": _VideoExpert(lora_enabled=lora_enabled),
                "action": nn.Linear(2, 2),
            }
        )
        self.dit = self.mot
        self.edit_memory_gist_tokens = nn.Parameter(torch.ones(8, 2))
        self.proprio_encoder = nn.Linear(2, 2)
        self.edit_memory_enabled = True
        self.apply_trainable_policy = types.MethodType(ImageWAM.apply_trainable_policy, self)


def test_optimizer_policy_includes_memory_action_lora_and_proprio() -> None:
    model = _TrainableModel()
    Wan22Trainer._apply_dit_only_train_mode(model)
    video = model.mot.mixtures["video"]
    action = model.mot.mixtures["action"]
    assert not video.block.weight.requires_grad
    assert video.block.lora_A.requires_grad and video.block.lora_B.requires_grad
    assert all(parameter.requires_grad for parameter in action.parameters())
    assert model.edit_memory_gist_tokens.requires_grad
    assert all(parameter.requires_grad for parameter in model.proprio_encoder.parameters())
    optimizer_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert "edit_memory_gist_tokens" in optimizer_names
    assert any("mixtures.action" in name for name in optimizer_names)
    assert any("lora_A" in name for name in optimizer_names)


def test_from_flux2_policy_trains_full_video_action_memory_and_proprio() -> None:
    model = _TrainableModel(lora_enabled=False)
    Wan22Trainer._apply_dit_only_train_mode(model)
    video = model.mot.mixtures["video"]
    action = model.mot.mixtures["action"]
    assert all(parameter.requires_grad for parameter in video.parameters())
    assert all(parameter.requires_grad for parameter in action.parameters())
    assert model.edit_memory_gist_tokens.requires_grad
    assert all(parameter.requires_grad for parameter in model.proprio_encoder.parameters())


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
