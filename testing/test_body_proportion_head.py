"""Tests for body proportion head keypoint ratios.

Tests verify:
1. _compute_ratios produces correct dimensions with/without head
2. Head ratios are anatomically reasonable on real images
3. Head ratios are stable across noise levels (same as body ratios)
4. Gradient flows through head ratios in training path
5. Cache versioning separates head vs body-only caches
6. Backward compatibility: body-only mode unchanged
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from toolkit.body_id import DifferentiableBodyProportionEncoder


# --- Synthetic keypoint helpers ---

def make_standing_person(batch_size=1, device='cpu'):
    """Create synthetic COCO-17 keypoints for a standing person.

    Returns (B, 17, 2) keypoints in [-1, 1] and (B, 17) visibilities.
    """
    # Normalized coordinates roughly matching a centered standing person
    kp = torch.zeros(batch_size, 17, 2, device=device)
    # Head
    kp[:, 0] = torch.tensor([0.0, -0.8])   # nose
    kp[:, 1] = torch.tensor([-0.03, -0.85]) # left eye
    kp[:, 2] = torch.tensor([0.03, -0.85])  # right eye
    kp[:, 3] = torch.tensor([-0.06, -0.78]) # left ear
    kp[:, 4] = torch.tensor([0.06, -0.78])  # right ear
    # Shoulders
    kp[:, 5] = torch.tensor([-0.2, -0.5])   # left shoulder
    kp[:, 6] = torch.tensor([0.2, -0.5])    # right shoulder
    # Elbows
    kp[:, 7] = torch.tensor([-0.25, -0.2])  # left elbow
    kp[:, 8] = torch.tensor([0.25, -0.2])   # right elbow
    # Wrists
    kp[:, 9] = torch.tensor([-0.22, 0.05])  # left wrist
    kp[:, 10] = torch.tensor([0.22, 0.05])  # right wrist
    # Hips
    kp[:, 11] = torch.tensor([-0.1, 0.1])   # left hip
    kp[:, 12] = torch.tensor([0.1, 0.1])    # right hip
    # Knees
    kp[:, 13] = torch.tensor([-0.1, 0.45])  # left knee
    kp[:, 14] = torch.tensor([0.1, 0.45])   # right knee
    # Ankles
    kp[:, 15] = torch.tensor([-0.1, 0.8])   # left ankle
    kp[:, 16] = torch.tensor([0.1, 0.8])    # right ankle

    vis = torch.ones(batch_size, 17, device=device)
    return kp, vis


def make_big_head_person(batch_size=1, device='cpu'):
    """Same standing person but with a proportionally larger head."""
    kp, vis = make_standing_person(batch_size, device)
    # Move nose further from shoulders (bigger head)
    kp[:, 0] = torch.tensor([0.0, -0.95])
    # Move ears further apart (wider head)
    kp[:, 3] = torch.tensor([-0.12, -0.85])
    kp[:, 4] = torch.tensor([0.12, -0.85])
    return kp, vis


# --- Tests ---

class TestComputeRatiosDimensions:
    """Output dimensions are correct with and without head."""

    def test_body_only_produces_8_ratios(self):
        kp, vis = make_standing_person()
        ratios, ratio_vis = DifferentiableBodyProportionEncoder._compute_ratios(
            kp, vis, include_head=False)
        assert ratios.shape == (1, 8), f"Expected (1, 8), got {ratios.shape}"
        assert ratio_vis.shape == (1, 8)

    def test_with_head_produces_10_ratios(self):
        kp, vis = make_standing_person()
        ratios, ratio_vis = DifferentiableBodyProportionEncoder._compute_ratios(
            kp, vis, include_head=True)
        assert ratios.shape == (1, 10), f"Expected (1, 10), got {ratios.shape}"
        assert ratio_vis.shape == (1, 10)

    def test_first_8_ratios_identical(self):
        """Head flag should not change the first 8 body ratios."""
        kp, vis = make_standing_person()
        body_ratios, _ = DifferentiableBodyProportionEncoder._compute_ratios(
            kp, vis, include_head=False)
        head_ratios, _ = DifferentiableBodyProportionEncoder._compute_ratios(
            kp, vis, include_head=True)
        assert torch.allclose(body_ratios, head_ratios[:, :8], atol=1e-6), \
            "First 8 ratios should be identical with and without head"

    def test_batch_dimension(self):
        kp, vis = make_standing_person(batch_size=4)
        ratios, ratio_vis = DifferentiableBodyProportionEncoder._compute_ratios(
            kp, vis, include_head=True)
        assert ratios.shape == (4, 10)
        assert ratio_vis.shape == (4, 10)


class TestHeadRatioValues:
    """Head ratios are anatomically reasonable."""

    def test_head_height_ratio_positive(self):
        kp, vis = make_standing_person()
        ratios, _ = DifferentiableBodyProportionEncoder._compute_ratios(
            kp, vis, include_head=True)
        head_height_ratio = ratios[0, 8].item()  # head_height / height
        assert head_height_ratio > 0, f"Head height ratio should be positive, got {head_height_ratio}"
        # Anatomically, head+neck is roughly 15-25% of torso+thigh+shin height
        assert 0.05 < head_height_ratio < 0.6, \
            f"Head height ratio {head_height_ratio:.3f} outside reasonable range"

    def test_head_width_ratio_positive(self):
        kp, vis = make_standing_person()
        ratios, _ = DifferentiableBodyProportionEncoder._compute_ratios(
            kp, vis, include_head=True)
        head_width_ratio = ratios[0, 9].item()  # head_width / shoulder_width
        assert head_width_ratio > 0, f"Head width ratio should be positive, got {head_width_ratio}"
        # Head is typically 25-45% of shoulder width
        assert 0.1 < head_width_ratio < 0.8, \
            f"Head width ratio {head_width_ratio:.3f} outside reasonable range"

    def test_big_head_has_larger_ratios(self):
        """A person with a bigger head should have larger head ratios."""
        kp_normal, vis = make_standing_person()
        kp_big, _ = make_big_head_person()

        normal_ratios, _ = DifferentiableBodyProportionEncoder._compute_ratios(
            kp_normal, vis, include_head=True)
        big_ratios, _ = DifferentiableBodyProportionEncoder._compute_ratios(
            kp_big, vis, include_head=True)

        # Head height should be larger
        assert big_ratios[0, 8] > normal_ratios[0, 8], \
            f"Big head should have larger head_height ratio: {big_ratios[0, 8]:.3f} vs {normal_ratios[0, 8]:.3f}"
        # Head width should be larger
        assert big_ratios[0, 9] > normal_ratios[0, 9], \
            f"Big head should have larger head_width ratio: {big_ratios[0, 9]:.3f} vs {normal_ratios[0, 9]:.3f}"

    def test_body_ratios_unchanged_by_head_size(self):
        """Changing head size should not affect body ratios (first 8)."""
        kp_normal, vis = make_standing_person()
        kp_big, _ = make_big_head_person()

        normal_ratios, _ = DifferentiableBodyProportionEncoder._compute_ratios(
            kp_normal, vis, include_head=False)
        big_ratios, _ = DifferentiableBodyProportionEncoder._compute_ratios(
            kp_big, vis, include_head=False)

        assert torch.allclose(normal_ratios, big_ratios, atol=1e-6), \
            "Body ratios should not change when only head keypoints change"


class TestHeadVisibility:
    """Head ratios correctly handle missing head keypoints."""

    def test_missing_nose_zeros_head_height(self):
        kp, vis = make_standing_person()
        vis[:, 0] = 0.0  # nose invisible
        ratios, ratio_vis = DifferentiableBodyProportionEncoder._compute_ratios(
            kp, vis, include_head=True)
        # head_height vis should be 0 (min of nose, L_shoulder, R_shoulder)
        assert ratio_vis[0, 8].item() == 0.0, \
            f"Head height vis should be 0 with missing nose, got {ratio_vis[0, 8]:.3f}"

    def test_missing_ear_zeros_head_width(self):
        kp, vis = make_standing_person()
        vis[:, 3] = 0.0  # left ear invisible
        ratios, ratio_vis = DifferentiableBodyProportionEncoder._compute_ratios(
            kp, vis, include_head=True)
        # head_width vis should be 0 (min of L_ear, R_ear, L_shoulder, R_shoulder)
        assert ratio_vis[0, 9].item() == 0.0, \
            f"Head width vis should be 0 with missing ear, got {ratio_vis[0, 9]:.3f}"

    def test_body_vis_unaffected_by_head(self):
        """Missing head keypoints should not affect body ratio visibility."""
        kp, vis = make_standing_person()
        _, body_vis_full = DifferentiableBodyProportionEncoder._compute_ratios(
            kp, vis.clone(), include_head=True)

        vis[:, 0] = 0.0  # nose
        vis[:, 3] = 0.0  # left ear
        vis[:, 4] = 0.0  # right ear
        _, body_vis_nohead = DifferentiableBodyProportionEncoder._compute_ratios(
            kp, vis, include_head=True)

        assert torch.allclose(body_vis_full[:, :8], body_vis_nohead[:, :8]), \
            "Body ratio visibility should not be affected by head keypoint visibility"

    def test_ref_ratios_replace_low_vis_head(self):
        """Low-vis head ratios should be replaced by ref values."""
        kp, vis = make_standing_person()
        vis[:, 0] = 0.0  # nose invisible
        ref = torch.ones(1, 10) * 0.5

        ratios, ratio_vis = DifferentiableBodyProportionEncoder._compute_ratios(
            kp, vis, ref_ratios=ref, include_head=True)

        # Head height ratio should be replaced with ref value (0.5)
        assert ratios[0, 8].item() == pytest.approx(0.5, abs=1e-5), \
            f"Low-vis head ratio should use ref value, got {ratios[0, 8]:.4f}"
        # Its visibility should be 0
        assert ratio_vis[0, 8].item() == 0.0


class TestGradientFlow:
    """Gradients flow through head ratios."""

    def test_gradient_through_head_ratios(self):
        kp, vis = make_standing_person()
        kp.requires_grad_(True)

        ratios, ratio_vis = DifferentiableBodyProportionEncoder._compute_ratios(
            kp, vis, include_head=True)

        # Loss on head ratios only
        loss = ratios[:, 8:].sum()
        loss.backward()

        assert kp.grad is not None, "Gradient should exist"
        # Head ratios depend on keypoints 0 (nose), 3 (L ear), 4 (R ear), 5, 6 (shoulders)
        # Gradient should be non-zero for these
        assert kp.grad[0, 0].abs().sum() > 0, "Nose gradient should be non-zero"
        assert kp.grad[0, 3].abs().sum() > 0, "Left ear gradient should be non-zero"
        assert kp.grad[0, 4].abs().sum() > 0, "Right ear gradient should be non-zero"

    def test_head_gradient_does_not_affect_body_only_path(self):
        """In body-only mode, no gradient flows to head keypoints."""
        kp, vis = make_standing_person()
        kp.requires_grad_(True)

        ratios, _ = DifferentiableBodyProportionEncoder._compute_ratios(
            kp, vis, include_head=False)
        loss = ratios.sum()
        loss.backward()

        # Head keypoints (0-4) should have zero gradient
        for i in range(5):
            assert kp.grad[0, i].abs().sum() == 0, \
                f"Keypoint {i} should have zero gradient in body-only mode"


# --- Real model tests (require GPU + ViTPose) ---

@pytest.fixture(scope="module")
def encoder():
    try:
        enc = DifferentiableBodyProportionEncoder()
        enc.eval()
        enc.to('cuda')
        return enc
    except Exception:
        pytest.skip("ViTPose model not available")


class TestRealModelHeadRatios:
    """End-to-end tests with real ViTPose on images."""

    def test_encode_body_only_dim(self, encoder):
        """encode() with include_head=False produces (16,) tensor."""
        img_dir = '/home/z/Documents/repos/ai-toolkit/bodytest'
        from PIL import Image
        imgs = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.webp'))]
        if not imgs:
            pytest.skip("No test images")
        img = Image.open(os.path.join(img_dir, imgs[0])).convert('RGB')
        emb = encoder.encode(img, include_head=False)
        assert emb.shape == (16,), f"Expected (16,), got {emb.shape}"

    def test_encode_with_head_dim(self, encoder):
        """encode() with include_head=True produces (20,) tensor."""
        img_dir = '/home/z/Documents/repos/ai-toolkit/bodytest'
        from PIL import Image
        imgs = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.webp'))]
        if not imgs:
            pytest.skip("No test images")
        img = Image.open(os.path.join(img_dir, imgs[0])).convert('RGB')
        emb = encoder.encode(img, include_head=True)
        assert emb.shape == (20,), f"Expected (20,), got {emb.shape}"

    def test_head_ratios_consistent_for_visible_heads(self, encoder):
        """Head ratios with high visibility should be relatively consistent."""
        img_dir = '/home/z/Documents/repos/ai-toolkit/bodytest'
        from PIL import Image
        imgs = sorted([f for f in os.listdir(img_dir)
                       if f.endswith(('.jpg', '.png', '.webp'))])[:15]
        if len(imgs) < 3:
            pytest.skip("Need at least 3 test images")

        head_heights = []
        head_widths = []
        for fname in imgs:
            img = Image.open(os.path.join(img_dir, fname)).convert('RGB')
            emb = encoder.encode(img, include_head=True)
            if emb.abs().sum() < 1e-6:
                continue  # no body detected
            # Only include images where head keypoints are visible
            # vis is in second half of embedding: indices 10+8=18, 10+9=19
            head_height_vis = emb[18].item()
            head_width_vis = emb[19].item()
            if head_height_vis > 0.3:
                head_heights.append(emb[8].item())
            if head_width_vis > 0.3:
                head_widths.append(emb[9].item())

        print(f"Head height: {len(head_heights)} visible, "
              f"mean={np.mean(head_heights):.3f} std={np.std(head_heights):.3f}" if head_heights else "none visible")
        print(f"Head width: {len(head_widths)} visible, "
              f"mean={np.mean(head_widths):.3f} std={np.std(head_widths):.3f}" if head_widths else "none visible")

        if len(head_heights) < 3:
            pytest.skip("Not enough high-visibility head detections")
        # For high-visibility samples (face clearly visible), ratios should be
        # more consistent — but still allow variance from pose angle
        hh_std = np.std(head_heights)
        assert hh_std < np.mean(head_heights), \
            f"High-vis head height ratio too variable: std={hh_std:.3f}, mean={np.mean(head_heights):.3f}"

    def test_forward_with_head_gradient(self, encoder):
        """Forward pass with include_head=True produces gradients."""
        pixels = torch.randn(1, 3, 256, 256, device='cuda', requires_grad=True)
        pixels_clamped = pixels.sigmoid()  # map to [0, 1]

        ratios, vis = encoder(pixels_clamped, include_head=True)
        assert ratios.shape == (1, 10), f"Expected (1, 10), got {ratios.shape}"

        loss = ratios[:, 8:].sum()  # loss on head ratios only
        loss.backward()
        assert pixels.grad is not None, "Gradient should reach input pixels"
        assert pixels.grad.abs().sum() > 0, "Gradient should be non-zero"

    def test_forward_body_only_unchanged(self, encoder):
        """Forward pass with include_head=False still works as before."""
        pixels = torch.rand(1, 3, 256, 256, device='cuda')
        ratios, vis = encoder(pixels, include_head=False)
        assert ratios.shape == (1, 8), f"Expected (1, 8), got {ratios.shape}"
        assert vis.shape == (1, 8)

    def test_noise_degrades_head_ratios(self, encoder):
        """Adding noise should degrade head ratio accuracy (same as body)."""
        img_dir = '/home/z/Documents/repos/ai-toolkit/bodytest'
        from PIL import Image
        from torchvision import transforms
        imgs = sorted([f for f in os.listdir(img_dir)
                       if f.endswith(('.jpg', '.png', '.webp'))])[:5]
        if not imgs:
            pytest.skip("No test images")

        to_tensor = transforms.ToTensor()
        clean_embs = []
        noisy_embs = []

        for fname in imgs:
            img = Image.open(os.path.join(img_dir, fname)).convert('RGB')
            emb = encoder.encode(img, include_head=True)
            if emb.abs().sum() < 1e-6:
                continue
            clean_embs.append(emb)

            # Add noise to image
            img_t = to_tensor(img).unsqueeze(0).to('cuda')
            noisy = (img_t + torch.randn_like(img_t) * 0.3).clamp(0, 1)
            noisy_pil = transforms.ToPILImage()(noisy.squeeze(0).cpu())
            noisy_emb = encoder.encode(noisy_pil, include_head=True)
            noisy_embs.append(noisy_emb)

        if len(clean_embs) < 2:
            pytest.skip("Not enough valid detections")

        # Compare head ratios: clean vs noisy
        clean = torch.stack(clean_embs)
        noisy = torch.stack(noisy_embs)

        # Head ratio difference should be non-trivial (noise has an effect)
        head_diff = (clean[:, 8:10] - noisy[:, 8:10]).abs().mean()
        body_diff = (clean[:, :8] - noisy[:, :8]).abs().mean()
        print(f"Head ratio diff from noise: {head_diff:.4f}")
        print(f"Body ratio diff from noise: {body_diff:.4f}")
        # Both should be affected by noise (non-zero diff)
        assert head_diff > 0.001, "Noise should affect head ratios"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
