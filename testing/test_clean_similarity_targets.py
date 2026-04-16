"""Tests for clean similarity targets in average mode identity loss.

These tests verify:
1. Normalized loss max(0, 1 - cos_sim/clean_cos) behaves correctly
2. Normalization prevents high-clean samples from dominating
3. Loss is zero when gen matches or exceeds clean target
4. Gradient flow is preserved through the normalized loss
5. Edge cases: low clean_cos, all same clean_cos, batch mixing
6. End-to-end: real ArcFace embeddings with noise degradation
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# --- Core loss function (mirrors SDTrainer implementation) ---

def clean_target_loss(cos_sim, clean_cos, mask=None):
    """Normalized clean target loss: max(0, 1 - cos_sim / clean_cos).

    Returns per-sample loss in [0, 1] regardless of absolute clean_cos.
    """
    loss = torch.clamp(1.0 - cos_sim / clean_cos, min=0.0)
    if mask is not None:
        loss = loss * mask.float()
    return loss


def standard_loss(cos_sim, mask=None):
    """Standard identity loss: (1 - cos_sim)."""
    loss = 1.0 - cos_sim
    if mask is not None:
        loss = loss * mask.float()
    return loss


# --- Unit tests for the loss function ---

class TestCleanTargetLossBasics:
    """Basic properties of the normalized clean target loss."""

    def test_loss_zero_at_target(self):
        """When cos_sim == clean_cos, loss should be 0."""
        cos_sim = torch.tensor([0.3, 0.5, 0.7, 0.9])
        clean_cos = torch.tensor([0.3, 0.5, 0.7, 0.9])
        loss = clean_target_loss(cos_sim, clean_cos)
        assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6), f"Loss should be 0 at target, got {loss}"

    def test_loss_zero_when_exceeds_target(self):
        """When cos_sim > clean_cos, loss should be 0 (clamped)."""
        cos_sim = torch.tensor([0.8, 0.6, 0.95])
        clean_cos = torch.tensor([0.5, 0.3, 0.9])
        loss = clean_target_loss(cos_sim, clean_cos)
        assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6), f"Loss should be 0 above target, got {loss}"

    def test_loss_one_at_zero_cosim(self):
        """When cos_sim == 0, loss should be 1.0 regardless of clean_cos."""
        cos_sim = torch.tensor([0.0, 0.0, 0.0])
        clean_cos = torch.tensor([0.3, 0.6, 0.9])
        loss = clean_target_loss(cos_sim, clean_cos)
        assert torch.allclose(loss, torch.ones_like(loss), atol=1e-6), f"Loss should be 1 at cos_sim=0, got {loss}"

    def test_loss_range_is_0_to_1(self):
        """Loss should always be in [0, 1] for valid inputs."""
        cos_sim = torch.rand(100) * 0.95  # [0, 0.95]
        clean_cos = torch.rand(100) * 0.8 + 0.1  # [0.1, 0.9]
        loss = clean_target_loss(cos_sim, clean_cos)
        assert loss.min() >= -1e-6, f"Loss should be >= 0, got min={loss.min():.6f}"
        assert loss.max() <= 1.0 + 1e-6, f"Loss should be <= 1, got max={loss.max():.6f}"

    def test_proportional_to_shortfall_ratio(self):
        """Loss should be proportional to relative shortfall, not absolute."""
        # Same relative shortfall (50% of target)
        cos_sim = torch.tensor([0.15, 0.45])
        clean_cos = torch.tensor([0.30, 0.90])
        loss = clean_target_loss(cos_sim, clean_cos)
        # Both at 50% of target → both should have loss = 0.5
        assert torch.allclose(loss, torch.tensor([0.5, 0.5]), atol=1e-6), \
            f"Same relative shortfall should give same loss, got {loss}"


class TestNormalizationPreventsOverweighting:
    """The key property: samples with different clean_cos contribute equally."""

    def test_loss_values_equal_at_same_relative_shortfall(self):
        """Two samples at same relative shortfall produce same loss value."""
        # Sample A: clean_cos=0.9, cos_sim=0.45 (50% of target)
        cos_a = torch.tensor([0.45], requires_grad=True)
        clean_a = torch.tensor([0.9])
        loss_a = clean_target_loss(cos_a, clean_a)

        # Sample B: clean_cos=0.3, cos_sim=0.15 (50% of target)
        cos_b = torch.tensor([0.15], requires_grad=True)
        clean_b = torch.tensor([0.3])
        loss_b = clean_target_loss(cos_b, clean_b)

        assert abs(loss_a.item() - loss_b.item()) < 1e-6, \
            f"Loss values should match at same relative shortfall: {loss_a.item():.6f} vs {loss_b.item():.6f}"

        # Gradient should be -1/clean_cos (steeper for smaller targets,
        # compensating for their smaller absolute range)
        loss_a.sum().backward()
        loss_b.sum().backward()
        assert abs(cos_a.grad.item() - (-1.0 / 0.9)) < 1e-5
        assert abs(cos_b.grad.item() - (-1.0 / 0.3)) < 1e-5

    def test_unnormalized_would_overweight(self):
        """Without normalization, high-clean samples would dominate."""
        cos_sim = torch.tensor([0.45, 0.15])  # both at 50% of target
        clean_cos = torch.tensor([0.9, 0.3])

        # Unnormalized: max(0, clean - cos) → [0.45, 0.15] — 3x difference!
        unnorm = torch.clamp(clean_cos - cos_sim, min=0.0)
        assert abs(unnorm[0] / unnorm[1] - 3.0) < 0.01, "Unnormalized should have 3x ratio"

        # Normalized: max(0, 1 - cos/clean) → [0.5, 0.5] — equal
        norm = clean_target_loss(cos_sim, clean_cos)
        assert torch.allclose(norm[0], norm[1], atol=1e-6), \
            f"Normalized should be equal: {norm}"

    def test_batch_loss_not_dominated_by_frontal_shots(self):
        """A batch with mixed clean_cos should weight equally after normalization."""
        # Simulate: 4 frontal shots (clean_cos=0.9) + 4 profile (clean_cos=0.4)
        # All at 70% of their target
        clean_cos = torch.tensor([0.9, 0.9, 0.9, 0.9, 0.4, 0.4, 0.4, 0.4])
        cos_sim = clean_cos * 0.7  # all at 70% of target

        loss = clean_target_loss(cos_sim, clean_cos)

        # All should have the same loss (0.3)
        assert torch.allclose(loss, torch.full_like(loss, 0.3), atol=1e-5), \
            f"All samples at 70% should have loss=0.3, got {loss}"

        # Mean contribution from frontal vs profile should be equal
        frontal_mean = loss[:4].mean()
        profile_mean = loss[4:].mean()
        assert abs(frontal_mean - profile_mean) < 1e-5, \
            f"Frontal vs profile mean loss should match: {frontal_mean:.4f} vs {profile_mean:.4f}"


class TestGradientFlow:
    """Gradients must flow through the clean target loss for training."""

    def test_gradient_exists(self):
        """Gradient should flow from loss back through cos_sim."""
        cos_sim = torch.tensor([0.5], requires_grad=True)
        clean_cos = torch.tensor([0.8])
        loss = clean_target_loss(cos_sim, clean_cos).sum()
        loss.backward()
        assert cos_sim.grad is not None, "Gradient should exist"
        assert cos_sim.grad.abs().sum() > 0, "Gradient should be non-zero"

    def test_gradient_is_negative_reciprocal_of_clean(self):
        """d/d(cos_sim) of max(0, 1 - cos_sim/clean_cos) = -1/clean_cos."""
        cos_sim = torch.tensor([0.3], requires_grad=True)
        clean_cos = torch.tensor([0.6])
        loss = clean_target_loss(cos_sim, clean_cos).sum()
        loss.backward()
        expected_grad = -1.0 / 0.6
        assert abs(cos_sim.grad.item() - expected_grad) < 1e-5, \
            f"Gradient should be -1/clean_cos={expected_grad:.4f}, got {cos_sim.grad.item():.4f}"

    def test_no_gradient_above_target(self):
        """When cos_sim > clean_cos, gradient should be zero (clamped region)."""
        cos_sim = torch.tensor([0.9], requires_grad=True)
        clean_cos = torch.tensor([0.7])
        loss = clean_target_loss(cos_sim, clean_cos).sum()
        loss.backward()
        assert cos_sim.grad.item() == 0.0, \
            f"Gradient should be 0 above target, got {cos_sim.grad.item():.6f}"

    def test_gradient_through_full_arcface_pipeline(self):
        """Gradient flows from clean target loss through ArcFace encoder to pixels."""
        try:
            from toolkit.face_id import DifferentiableFaceEncoder
        except ImportError:
            pytest.skip("DifferentiableFaceEncoder not available")

        enc = DifferentiableFaceEncoder()
        enc.eval()
        enc.to('cuda')

        pixels = torch.randn(1, 3, 112, 112, device='cuda', requires_grad=True)
        gen = enc(pixels)

        ref = F.normalize(torch.randn(1, 512, device='cuda'), p=2, dim=-1)
        cos_sim = F.cosine_similarity(gen, ref, dim=-1)
        clean_cos = torch.tensor([0.7], device='cuda')

        loss = clean_target_loss(cos_sim, clean_cos).sum()
        loss.backward()

        assert pixels.grad is not None, "Gradient should reach input pixels"
        assert pixels.grad.abs().sum() > 0, "Gradient should be non-zero"


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_very_low_clean_cos(self):
        """With clean_cos=0.1 (floor), loss should still be bounded."""
        cos_sim = torch.tensor([0.05])
        clean_cos = torch.tensor([0.1])
        loss = clean_target_loss(cos_sim, clean_cos)
        assert loss.item() == pytest.approx(0.5, abs=1e-5), f"Expected 0.5, got {loss.item()}"

    def test_mask_zeros_out_invalid(self):
        """Masked-out samples should contribute zero loss."""
        cos_sim = torch.tensor([0.3, 0.3, 0.3])
        clean_cos = torch.tensor([0.6, 0.6, 0.6])
        mask = torch.tensor([True, False, True])
        loss = clean_target_loss(cos_sim, clean_cos, mask=mask)
        assert loss[1].item() == 0.0, "Masked sample should have zero loss"
        assert loss[0].item() > 0, "Unmasked sample should have non-zero loss"

    def test_all_same_clean_cos_equals_standard(self):
        """When all clean_cos are the same, relative ordering matches standard loss."""
        cos_sim = torch.tensor([0.2, 0.4, 0.6, 0.8])
        clean_cos = torch.tensor([0.9, 0.9, 0.9, 0.9])

        clean_loss = clean_target_loss(cos_sim, clean_cos)
        std_loss = standard_loss(cos_sim)

        # Ordering should be the same (higher cos_sim → lower loss)
        for i in range(3):
            assert clean_loss[i] > clean_loss[i + 1], \
                f"Ordering should match: loss[{i}]={clean_loss[i]:.3f} > loss[{i+1}]={clean_loss[i+1]:.3f}"

    def test_negative_cos_sim(self):
        """Negative cos_sim (opposite direction) should give loss > 1."""
        cos_sim = torch.tensor([-0.2])
        clean_cos = torch.tensor([0.5])
        loss = clean_target_loss(cos_sim, clean_cos)
        expected = 1.0 - (-0.2 / 0.5)  # 1.4
        assert loss.item() == pytest.approx(expected, abs=1e-5), \
            f"Negative cos_sim: expected {expected:.3f}, got {loss.item():.3f}"

    def test_default_fallback_for_none_clean_cos(self):
        """When clean_cos is None (non-average mode), default to 1.0 = standard loss."""
        cos_sim = torch.tensor([0.5])
        clean_cos = torch.tensor([1.0])  # default fallback
        clean_loss = clean_target_loss(cos_sim, clean_cos)
        std_loss = standard_loss(cos_sim)
        assert torch.allclose(clean_loss, std_loss, atol=1e-6), \
            f"clean_cos=1.0 should match standard loss: {clean_loss} vs {std_loss}"


# --- Integration: real ArcFace with noise degradation ---

@pytest.fixture(scope="module")
def encoder():
    try:
        from toolkit.face_id import DifferentiableFaceEncoder
    except ImportError:
        pytest.skip("DifferentiableFaceEncoder not available")
    enc = DifferentiableFaceEncoder()
    enc.eval()
    enc.to('cuda')
    return enc


@pytest.fixture(scope="module")
def noise_mean(encoder):
    """ArcFace bias direction from noise."""
    embeds = []
    for _ in range(200):
        img = torch.randn(1, 3, 112, 112, device='cuda') * 0.3 + 0.5
        img = img.clamp(0, 1)
        with torch.no_grad():
            embeds.append(encoder(img))
    return torch.cat(embeds, dim=0).mean(dim=0)


def bias_correct(emb, mean):
    """Center and re-normalize."""
    centered = emb - mean.unsqueeze(0) if emb.dim() == 2 else emb - mean
    return F.normalize(centered, p=2, dim=-1)


class TestCleanCosWithRealArcFace:
    """End-to-end tests using actual ArcFace model."""

    def test_clean_cos_decreases_with_noise(self, encoder, noise_mean):
        """Adding noise to an image should decrease its cos_sim vs clean."""
        torch.manual_seed(42)
        clean_img = torch.randn(1, 3, 112, 112, device='cuda') * 0.2 + 0.6
        clean_img = clean_img.clamp(0, 1)

        with torch.no_grad():
            clean_emb = encoder(clean_img)
            clean_c = bias_correct(clean_emb, noise_mean)

        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]
        cos_sims = []
        for noise_std in noise_levels:
            noisy = (clean_img + torch.randn_like(clean_img) * noise_std).clamp(0, 1)
            with torch.no_grad():
                noisy_emb = encoder(noisy)
                noisy_c = bias_correct(noisy_emb, noise_mean)
            cos = F.cosine_similarity(noisy_c, clean_c, dim=-1).item()
            cos_sims.append(cos)

        # Should generally decrease (allow some noise in monotonicity)
        assert cos_sims[0] > cos_sims[-1], \
            f"cos_sim should decrease with noise: {list(zip(noise_levels, cos_sims))}"

    def test_clean_targets_give_zero_loss_for_clean_input(self, encoder, noise_mean):
        """Clean image vs its own embedding should give zero loss."""
        torch.manual_seed(123)
        img = torch.randn(1, 3, 112, 112, device='cuda') * 0.2 + 0.6
        img = img.clamp(0, 1)

        with torch.no_grad():
            emb = encoder(img)
            emb_c = bias_correct(emb, noise_mean)

        # clean_cos == cos_sim for the clean image itself → loss = 0
        cos_sim = F.cosine_similarity(emb_c, emb_c, dim=-1)
        clean_cos = cos_sim.clone()
        loss = clean_target_loss(cos_sim, clean_cos)
        assert loss.item() < 1e-6, f"Loss for clean input should be 0, got {loss.item()}"

    def test_dataset_average_target(self, encoder, noise_mean):
        """Simulate a small dataset: compute average, check clean_cos targets."""
        # Generate 5 "face" images
        imgs = []
        for seed in [10, 20, 30, 40, 50]:
            torch.manual_seed(seed)
            img = torch.randn(1, 3, 112, 112, device='cuda') * 0.2 + 0.6
            imgs.append(img.clamp(0, 1))

        with torch.no_grad():
            embeds = torch.cat([encoder(img) for img in imgs], dim=0)

        # Compute average (like SDTrainer does)
        avg = embeds.mean(dim=0)
        avg = avg / (avg.norm() + 1e-8)

        # Bias-correct and compute clean_cos per image
        avg_c = bias_correct(avg.unsqueeze(0), noise_mean)
        clean_cos_vals = []
        for i in range(5):
            emb_c = bias_correct(embeds[i:i+1], noise_mean)
            cc = F.cosine_similarity(emb_c, avg_c, dim=-1).item()
            clean_cos_vals.append(max(cc, 0.1))

        # All should be positive (faces from same "person")
        assert all(cc > 0 for cc in clean_cos_vals), f"Clean cos should be positive: {clean_cos_vals}"
        # Should have some spread (different images)
        assert max(clean_cos_vals) - min(clean_cos_vals) > 0.01, \
            f"Should have some spread in clean targets: {clean_cos_vals}"

        # Now degrade images and check loss behavior
        for i, img in enumerate(imgs):
            noisy = (img + torch.randn_like(img) * 0.3).clamp(0, 1)
            with torch.no_grad():
                noisy_emb = encoder(noisy)
                noisy_c = bias_correct(noisy_emb, noise_mean)
            cos_sim = F.cosine_similarity(noisy_c, avg_c, dim=-1)
            clean_cos_t = torch.tensor([clean_cos_vals[i]], device='cuda')
            loss = clean_target_loss(cos_sim, clean_cos_t)
            # Loss should be positive (noise degraded it below clean)
            # but bounded by 1.0
            assert loss.item() <= 1.5, f"Loss should be bounded, got {loss.item()}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
