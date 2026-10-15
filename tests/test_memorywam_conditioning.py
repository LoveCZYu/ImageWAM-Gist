import torch

from imagewam.models.backbones.imagewam import ImageWAM


def test_clean_conditioning_augmentation_uses_fixed_interpolation() -> None:
    tokens = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    noise = torch.full_like(tokens, 10.0)
    ratio = torch.tensor([0.25])
    augmented = ImageWAM._augment_clean_conditioning_tokens(
        tokens,
        enabled=True,
        probability=1.0,
        ratio_min=0.0,
        ratio_max=1.0,
        ratio=ratio,
        noise=noise,
    )
    assert torch.equal(augmented, 0.75 * tokens + 0.25 * noise)


def test_clean_conditioning_augmentation_can_be_disabled() -> None:
    tokens = torch.randn(2, 3, 4)
    noise = torch.randn_like(tokens)
    ratio = torch.ones(2)
    unchanged = ImageWAM._augment_clean_conditioning_tokens(
        tokens,
        enabled=False,
        probability=1.0,
        ratio_min=0.0,
        ratio_max=1.0,
        ratio=ratio,
        noise=noise,
    )
    assert torch.equal(unchanged, tokens)

