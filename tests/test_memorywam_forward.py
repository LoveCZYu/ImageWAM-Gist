import os
import sys
import tempfile
import types

import torch
import torch.nn as nn

from imagewam.models.backbones.flux2_video_expert import Flux2VideoExpert
from imagewam.models.backbones.edit_memory import (
    combine_memory_entries,
    evict_nonpersistent_frame_kv,
    select_hybrid_memory_entries,
)
from imagewam.models.backbones.imagewam import ImageWAM
from imagewam.models.backbones.mot import MoT


def test_uniform_history_credit_samples_the_full_window() -> None:
    torch.manual_seed(7)
    seen = set()
    for _ in range(64):
        selected = ImageWAM._select_trainable_history_indices(
            list(range(16)),
            1,
            "uniform",
            device=torch.device("cpu"),
        )
        assert len(selected) == 1
        seen.update(selected)
    assert 0 in seen
    assert 15 in seen
    assert len(seen) >= 12


def test_old_gist_keeps_gradient_path_to_future_reader() -> None:
    selected_gist = torch.ones(1, 1, 2, requires_grad=True)

    def make_layer(value: torch.Tensor) -> dict:
        return {
            "k": value,
            "v": value * 2,
            "topk_indices": torch.zeros(1, 1, dtype=torch.long),
            "topk_scores": torch.ones(1, 1),
            "spatial_xy": torch.zeros(1, 1, 2),
            "memory_is_valid": torch.ones(1, 1, dtype=torch.bool),
            "layer_idx": 0,
        }

    entries = []
    for index in range(8):
        frame = torch.full((1, 1, 2), float(index + 1))
        gist = selected_gist if index == 2 else torch.full((1, 1, 2), float(index + 1))
        entries.append(
            {
                "double": [make_layer(frame)],
                "single": [make_layer(frame)],
                "gist_double": [make_layer(gist)],
                "gist_single": [make_layer(gist)],
                "step_start": index,
                "step_end": index,
            }
        )
    maintained, _ = evict_nonpersistent_frame_kv(
        entries,
        anchor_frames=2,
        recent_frames=4,
        gist_tokens_per_frame=8,
    )
    assert maintained[2]["frame_kv_evicted"]
    cache = combine_memory_entries(
        select_hybrid_memory_entries(
            maintained,
            anchor_frames=2,
            recent_frames=4,
            gist_tokens_per_frame=8,
        )
    )
    assert cache is not None
    cache["double"][0]["k"].sum().backward()
    assert selected_gist.grad is not None
    assert bool((selected_gist.grad != 0).all())


class _FakeFluxTransformer(nn.Module):
    hidden_size = 4
    num_heads = 1

    def __init__(self) -> None:
        super().__init__()
        self.double_blocks = nn.ModuleList()
        self.single_blocks = nn.ModuleList()
        self.time_in = nn.Identity()
        self.txt_in = nn.Identity()
        self.img_in = nn.Identity()

    @staticmethod
    def pe_embedder(ids: torch.Tensor) -> torch.Tensor:
        return ids

    @staticmethod
    def double_stream_modulation_img(vec: torch.Tensor) -> torch.Tensor:
        return vec

    @staticmethod
    def double_stream_modulation_txt(vec: torch.Tensor) -> torch.Tensor:
        return vec

    @staticmethod
    def single_stream_modulation(vec: torch.Tensor) -> tuple[torch.Tensor, None]:
        return vec, None

    @staticmethod
    def final_layer(target: torch.Tensor, _vec: torch.Tensor) -> torch.Tensor:
        return target


def test_flux2_prefix_is_ref_then_gist_then_target() -> None:
    flux2_module = types.ModuleType("flux2")
    flux2_model = types.ModuleType("flux2.model")
    flux2_model.timestep_embedding = lambda timestep, dim: torch.zeros(timestep.shape[0], dim)
    flux2_module.model = flux2_model
    sys.modules.setdefault("flux2", flux2_module)
    sys.modules.setdefault("flux2.model", flux2_model)

    expert = Flux2VideoExpert(_FakeFluxTransformer(), variant="test")
    ref = torch.full((1, 3, 4), 1.0)
    gist = torch.full((1, 8, 4), 2.0)
    target = torch.full((1, 2, 4), 3.0)
    pre = expert.pre_dit(
        x=target,
        timestep=torch.zeros(1),
        context=torch.zeros(1, 2, 4),
        context_mask=torch.ones(1, 2, dtype=torch.bool),
        ref_image_hidden_states=ref,
        gist_hidden_states=gist,
        target_img_ids=torch.zeros(1, 2, 4),
        ref_img_ids=torch.zeros(1, 3, 4),
        gist_img_ids=torch.zeros(1, 8, 4),
    )
    image_tokens = pre["tokens"]["img"]
    assert torch.equal(image_tokens[:, :3], ref)
    assert torch.equal(image_tokens[:, 3:11], gist)
    assert torch.equal(image_tokens[:, 11:], target)
    assert (pre["cond_len"], pre["gist_len"], pre["target_len"]) == (3, 8, 2)
    assert torch.equal(expert.post_dit(pre["tokens"], pre), target)


def test_soft_backward_gate_can_recover_a_hard_excluded_token() -> None:
    mot = MoT.__new__(MoT)
    nn.Module.__init__(mot)
    mot.num_heads = 1
    mot.num_kv_heads = 1
    mot.attn_head_dim = 2
    mot.gqa_implementation = "repeat"
    mot.force_flash_attention = False
    mot.mot_checkpoint_mixed_attn = False
    query = torch.tensor([[[1.0, 0.0]]])
    key = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    value = torch.tensor([[[2.0, 0.0], [0.0, 1.0]]])
    gate = torch.tensor(0.0, requires_grad=True)
    mask = torch.stack([-12.0 * (1.0 - gate), gate.new_zeros(())]).reshape(1, 1, 2)
    output = mot._mixed_attention(query, key, value, mask)
    output.sum().backward()
    assert gate.grad is not None
    assert torch.isfinite(gate.grad)
    assert abs(float(gate.grad.item())) > 0.0


def test_flux2_block_checkpoint_recomputes_video_and_action_blocks() -> None:
    flux2_module = sys.modules.get("flux2")
    if flux2_module is None:
        flux2_module = types.ModuleType("flux2")
        sys.modules["flux2"] = flux2_module
    flux2_model = sys.modules.get("flux2.model")
    if flux2_model is None:
        flux2_model = types.ModuleType("flux2.model")
        sys.modules["flux2.model"] = flux2_model
    flux2_module.model = flux2_model
    previous_apply_rope = getattr(flux2_model, "apply_rope", None)
    flux2_model.apply_rope = lambda q, k, _pe: (q, k)

    class FakeTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pe_embedder = nn.Identity()

    class FakeVideoDoubleBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(2, 2, bias=False)
            self.calls = 0

        def _prepare_qkv(self, img, txt, _img_pe, _txt_pe, _img_mod, _txt_mod):
            self.calls += 1
            stream = self.proj(torch.cat([txt, img], dim=1))
            qkv = stream.unsqueeze(1)
            return qkv, qkv, qkv, None, int(txt.shape[1]), None

        @staticmethod
        def _apply_residuals(img, txt, img_attn, txt_attn, _mods):
            return img + img_attn, txt + txt_attn

    class FakeVideoSingleBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(2, 2, bias=False)
            self.calls = 0

        def _qkv(self, stream, _mod):
            self.calls += 1
            qkv = self.proj(stream).unsqueeze(1)
            return qkv, qkv, qkv, torch.zeros_like(stream), torch.ones_like(stream)

        @staticmethod
        def _out(residual, mixed, _mlp, _gate):
            return residual + mixed

    class FakeActionBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(2, 2, bias=False)
            self.calls = 0

        def prepare_qkv(self, action, _pe, _mod):
            self.calls += 1
            qkv = self.proj(action)
            return {"q": qkv, "k": qkv, "v": qkv, "residual": action}

        @staticmethod
        def apply_post(mixed, state):
            return state["residual"] + mixed

    class FakeVideoExpert(nn.Module):
        block_protocol = "flux2"
        num_heads = 1
        num_kv_heads = 1
        attn_head_dim = 2
        double_layers = 1
        single_layers = 1

        def __init__(self) -> None:
            super().__init__()
            self.transformer = FakeTransformer()
            self.double_blocks = nn.ModuleList([FakeVideoDoubleBlock()])
            self.single_blocks = nn.ModuleList([FakeVideoSingleBlock()])
            self.blocks = nn.ModuleList(
                [self.double_blocks[0], self.single_blocks[0]]
            )

    class FakeActionExpert(nn.Module):
        block_protocol = "flux2"
        num_heads = 1
        num_kv_heads = 1
        attn_head_dim = 2

        def __init__(self) -> None:
            super().__init__()
            self.double_blocks = nn.ModuleList([FakeActionBlock()])
            self.single_blocks = nn.ModuleList([FakeActionBlock()])
            self.blocks = nn.ModuleList(
                [self.double_blocks[0], self.single_blocks[0]]
            )

    try:
        video_expert = FakeVideoExpert()
        action_expert = FakeActionExpert()
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=True,
        ).train()
        video_cache = mot.prefill_flux2_video_cache(
            video_tokens={
                "txt": torch.randn(1, 1, 2, requires_grad=True),
                "img": torch.randn(1, 1, 2, requires_grad=True),
            },
            video_freqs={
                "txt": torch.zeros(1, 1, 4),
                "img": torch.zeros(1, 1, 4),
            },
            video_t_mod={
                "double_img": torch.zeros(1, 2),
                "double_txt": torch.zeros(1, 2),
                "single": torch.zeros(1, 2),
            },
            attention_mask={
                "double_joint": torch.ones(2, 2, dtype=torch.bool),
                "single": torch.ones(2, 2, dtype=torch.bool),
            },
        )
        action = mot.forward_flux2_action_with_video_cache(
            action_tokens=torch.randn(1, 1, 2, requires_grad=True),
            action_ids=torch.zeros(1, 1, 4),
            action_t_mod={
                "double_img": torch.zeros(1, 2),
                "single": torch.zeros(1, 2),
            },
            video_kv_cache=video_cache,
            attention_mask={
                "double_joint": torch.ones(3, 3, dtype=torch.bool),
                "single": torch.ones(3, 3, dtype=torch.bool),
            },
            video_seq_len=2,
        )
        loss = video_cache["final_video"]["txt"].sum() + action.sum()
        loss.backward()
        assert video_expert.double_blocks[0].calls >= 2
        assert video_expert.single_blocks[0].calls >= 2
        assert action_expert.double_blocks[0].calls >= 2
        assert action_expert.single_blocks[0].calls >= 2
    finally:
        if previous_apply_rope is None:
            delattr(flux2_model, "apply_rope")
        else:
            flux2_model.apply_rope = previous_apply_rope


def test_checkpoint_roundtrip_includes_memory_parameters() -> None:
    class DummyModel(nn.Module):
        save_checkpoint = ImageWAM.save_checkpoint
        load_checkpoint = ImageWAM.load_checkpoint
        _edit_memory_checkpoint_contract = ImageWAM._edit_memory_checkpoint_contract

        def __init__(self) -> None:
            super().__init__()
            self.mot = nn.Linear(2, 2)
            self.edit_memory_test = nn.Parameter(torch.tensor([3.0]))
            self.proprio_encoder = None
            self.save_lora_merged = False
            self.save_trainable_only = False
            self.torch_dtype = torch.float32
            self.stack = "test"
            self.edit_memory_enabled = True
            self.edit_memory_config = {}

    first = DummyModel()
    second = DummyModel()
    with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
        first.save_checkpoint(checkpoint.name, step=17)
        payload = torch.load(checkpoint.name, map_location="cpu")
        assert "edit_memory" in payload
        assert torch.equal(payload["edit_memory"]["edit_memory_test"], torch.tensor([3.0]))
        second.edit_memory_test.data.zero_()
        second.load_checkpoint(checkpoint.name)
        assert torch.equal(second.edit_memory_test, torch.tensor([3.0]))


def test_checkpoint_rejects_incomplete_memory_state() -> None:
    class DummyModel(nn.Module):
        save_checkpoint = ImageWAM.save_checkpoint
        load_checkpoint = ImageWAM.load_checkpoint
        _edit_memory_checkpoint_contract = ImageWAM._edit_memory_checkpoint_contract

        def __init__(self) -> None:
            super().__init__()
            self.mot = nn.Linear(2, 2)
            self.edit_memory_test = nn.Parameter(torch.tensor([3.0]))
            self.proprio_encoder = None
            self.save_lora_merged = False
            self.save_trainable_only = False
            self.torch_dtype = torch.float32
            self.stack = "test"
            self.edit_memory_enabled = True
            self.edit_memory_config = {}

    model = DummyModel()
    with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
        model.save_checkpoint(checkpoint.name)
        payload = torch.load(checkpoint.name, map_location="cpu")
        payload["edit_memory"] = {}
        torch.save(payload, checkpoint.name)
        try:
            model.load_checkpoint(checkpoint.name)
        except RuntimeError as exc:
            assert "missing=['edit_memory_test']" in str(exc)
        else:
            raise AssertionError("Incomplete edit-memory state must not load silently.")


def test_checkpoint_rejects_replan_contract_mismatch() -> None:
    class DummyModel(nn.Module):
        save_checkpoint = ImageWAM.save_checkpoint
        load_checkpoint = ImageWAM.load_checkpoint
        _edit_memory_checkpoint_contract = ImageWAM._edit_memory_checkpoint_contract

        def __init__(self) -> None:
            super().__init__()
            self.mot = nn.Linear(2, 2)
            self.edit_memory_test = nn.Parameter(torch.tensor([3.0]))
            self.proprio_encoder = None
            self.save_lora_merged = False
            self.save_trainable_only = False
            self.torch_dtype = torch.float32
            self.stack = "test"
            self.edit_memory_enabled = True
            self.edit_memory_config = {}

    previous = os.environ.get("IMAGEWAM_REPLAN_STEPS")
    try:
        os.environ["IMAGEWAM_REPLAN_STEPS"] = "16"
        with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
            DummyModel().save_checkpoint(checkpoint.name)
            os.environ["IMAGEWAM_REPLAN_STEPS"] = "8"
            try:
                DummyModel().load_checkpoint(checkpoint.name)
            except RuntimeError as exc:
                assert "replan_steps" in str(exc)
            else:
                raise AssertionError("A Stage2 checkpoint must reject a different inference replan interval.")
    finally:
        if previous is None:
            os.environ.pop("IMAGEWAM_REPLAN_STEPS", None)
        else:
            os.environ["IMAGEWAM_REPLAN_STEPS"] = previous


def test_memory_prefix_is_target_free_and_uses_inference_timestep() -> None:
    class DummyVideoExpert:
        def __init__(self) -> None:
            self.call = None

        def pre_dit(self, **kwargs):
            self.call = kwargs
            ref = kwargs["ref_image_hidden_states"]
            gist = kwargs["gist_hidden_states"]
            text = kwargs["context"]
            return {
                "tokens": {"txt": text, "img": torch.cat([ref, gist], dim=1)},
                "freqs": {"txt": torch.zeros_like(text), "img": torch.zeros_like(torch.cat([ref, gist], dim=1))},
                "t_mod": {},
                "txt_len": int(text.shape[1]),
                "target_len": 0,
                "cond_len": int(ref.shape[1]),
                "gist_len": int(gist.shape[1]),
                "text_mask": kwargs["context_mask"],
            }

    class DummyMot:
        def __init__(self) -> None:
            self.call = None

        def prefill_flux2_video_cache(self, **kwargs):
            self.call = kwargs
            return {"cache": kwargs["video_tokens"]}

    model = ImageWAM.__new__(ImageWAM)
    nn.Module.__init__(model)
    model.video_expert = DummyVideoExpert()
    model.mot = DummyMot()
    ref = torch.randn(1, 3, 4)
    gist = torch.randn(1, 8, 4)
    text = torch.randn(1, 2, 4)
    memory = {"sentinel": True}
    pre, cache = model._prefill_flux2_memory_prefix(
        ref_tokens=ref,
        ref_img_ids=torch.zeros(1, 3, 4),
        text_hidden=text,
        text_mask=torch.ones(1, 2, dtype=torch.bool),
        gist_tokens=gist,
        gist_ids=torch.zeros(1, 8, 4),
        memory_kv_cache=memory,
    )
    call = model.video_expert.call
    assert call["x"].shape[1] == 0
    assert torch.equal(call["timestep"], torch.zeros(1))
    assert pre["target_len"] == 0
    assert model.mot.call["memory_kv_cache"] is memory
    assert tuple(model.mot.call["attention_mask"]["double_joint"].shape) == (1, 13, 13)
    assert "cache" in cache


def test_flux2_attention_regions_prevent_target_leakage() -> None:
    model = ImageWAM.__new__(ImageWAM)
    nn.Module.__init__(model)
    mask = model._build_mot_attention_mask_flux2(
        batch_size=1,
        txt_len=2,
        cond_len=3,
        gist_len=2,
        target_len=4,
        action_len=2,
        device=torch.device("cpu"),
        text_attention_mask=torch.tensor([[True, False]]),
    )["double_joint"][0]
    prefix = slice(0, 7)
    target = slice(7, 11)
    action = slice(11, 13)
    assert not bool(mask[prefix, target].any())
    assert not bool(mask[prefix, action].any())
    assert bool(mask[target, prefix].any()) and bool(mask[target, target].all())
    assert not bool(mask[target, action].any())
    assert bool(mask[action, prefix].any()) and bool(mask[action, action].all())
    assert not bool(mask[action, target].any())
    assert not bool(mask[:, 1].any())


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
