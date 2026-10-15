import torch

from imagewam.models.backbones.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)


def test_logit_normal_training_sampling_matches_definition() -> None:
    scheduler = WanContinuousFlowMatchScheduler(
        num_train_timesteps=1000,
        shift=5.0,
        sampling_distribution="logit_normal",
        loss_weighting="none",
    )
    generator_a = torch.Generator(device="cpu").manual_seed(17)
    generator_b = torch.Generator(device="cpu").manual_seed(17)
    sampled = scheduler.sample_training_t(
        batch_size=16,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=generator_a,
    )
    normal = torch.randn((16,), generator=generator_b)
    u = torch.sigmoid(normal)
    expected = scheduler._phi(u, 5.0) * 1000.0
    assert torch.equal(sampled, expected)
    assert bool((sampled > 0).all().item())
    assert bool((sampled < 1000).all().item())


def test_shift_one_is_identity_and_memory_loss_weight_is_constant() -> None:
    scheduler = WanContinuousFlowMatchScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        sampling_distribution="logit_normal",
        loss_weighting="none",
    )
    generator = torch.Generator(device="cpu").manual_seed(3)
    sampled = scheduler.sample_training_t(
        batch_size=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=generator,
    )
    generator_check = torch.Generator(device="cpu").manual_seed(3)
    expected = torch.sigmoid(torch.randn((8,), generator=generator_check)) * 1000.0
    assert torch.equal(sampled, expected)
    weights = scheduler.training_weight(torch.tensor([0.0, 500.0, 1000.0]))
    assert torch.equal(weights, torch.ones(3))

