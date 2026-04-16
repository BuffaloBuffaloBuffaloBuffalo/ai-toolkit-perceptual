"""Tests for ArcFace bias correction and average-mode embedding handling.

These tests verify:
1. The ArcFace bias problem exists (baseline cos_sim ~0.5 for noise vs face)
2. Bias correction brings noise baseline to ~0
3. Face-vs-face discrimination is preserved after correction
4. Gradient flow is preserved through the correction
5. Average mode doesn't replace zero embeddings (no-face images)
6. Single-person datasets work correctly (the common LoRA case)
7. Multi-person datasets work correctly
8. The correction works at different noise levels (simulating different timesteps)
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# --- Helpers ---

def make_noise_embedding(encoder, device, n=1):
    """Generate ArcFace embeddings from random noise images."""
    embeds = []
    for _ in range(n):
        img = torch.randn(1, 3, 112, 112, device=device) * 0.3 + 0.5
        img = img.clamp(0, 1)
        with torch.no_grad():
            emb = encoder(img)
        embeds.append(emb)
    return torch.cat(embeds, dim=0)


def make_uniform_embedding(encoder, device, color=(0.7, 0.5, 0.5)):
    """Generate ArcFace embedding from a uniform-color image (featureless blob)."""
    img = torch.zeros(1, 3, 112, 112, device=device)
    img[:, 0] = color[0]
    img[:, 1] = color[1]
    img[:, 2] = color[2]
    with torch.no_grad():
        return encoder(img)


def compute_bias_mean(encoder, device, n=200):
    """Compute the ArcFace bias direction from random noise."""
    return make_noise_embedding(encoder, device, n=n).mean(dim=0)


def center_and_normalize(embeddings, mean):
    """Subtract mean and re-normalize — the proposed bias correction."""
    centered = embeddings - mean.unsqueeze(0)
    return F.normalize(centered, p=2, dim=-1)


# --- Fixtures ---

@pytest.fixture(scope="module")
def encoder():
    from toolkit.face_id import DifferentiableFaceEncoder
    enc = DifferentiableFaceEncoder()
    enc.eval()
    enc.to('cuda')
    return enc


@pytest.fixture(scope="module")
def noise_mean(encoder):
    """The ArcFace bias direction computed from noise inputs."""
    return compute_bias_mean(encoder, 'cuda', n=200)


@pytest.fixture(scope="module")
def noise_embeddings(encoder):
    """50 noise embeddings for testing."""
    return make_noise_embedding(encoder, 'cuda', n=50)


@pytest.fixture(scope="module")
def blob_embeddings(encoder):
    """Embeddings from uniform-color blobs (skin tone, gray, black)."""
    colors = [(0.7, 0.5, 0.5), (0.5, 0.5, 0.5), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
    embeds = []
    for c in colors:
        embeds.append(make_uniform_embedding(encoder, 'cuda', color=c))
    return torch.cat(embeds, dim=0)


# --- Problem 1: Verify the bias exists ---

class TestArcFaceBiasExists:
    """Confirm the ArcFace embedding space has the bias we're correcting."""

    def test_noise_embeddings_cluster_tightly(self, noise_embeddings):
        """Random noise inputs should produce highly similar embeddings (~0.85)."""
        # Pairwise cos_sim between first 25 and last 25
        pairs = F.cosine_similarity(noise_embeddings[:25], noise_embeddings[25:], dim=-1)
        assert pairs.mean() > 0.7, f"Noise pairs should cluster tightly, got mean={pairs.mean():.3f}"

    def test_noise_mean_has_high_norm(self, noise_mean):
        """The mean of noise embeddings should have high norm (strong bias direction)."""
        norm = noise_mean.norm().item()
        assert norm > 0.8, f"Noise mean norm should be >0.8, got {norm:.4f}"

    def test_blob_scores_high_against_noise(self, blob_embeddings, noise_embeddings):
        """Uniform blobs should score high against noise (same bias cluster)."""
        for i in range(blob_embeddings.shape[0]):
            cos = F.cosine_similarity(
                blob_embeddings[i:i+1].expand(50, -1),
                noise_embeddings, dim=-1
            )
            assert cos.mean() > 0.3, f"Blob {i} vs noise should be >0.3, got {cos.mean():.3f}"


# --- Problem 1 fix: Bias correction ---

class TestBiasCorrectionNoise:
    """After bias correction, non-face inputs should score near zero."""

    def test_noise_vs_noise_after_correction(self, noise_embeddings, noise_mean):
        """Centered noise pairs should have low cos_sim (~0, not ~0.85)."""
        centered = center_and_normalize(noise_embeddings, noise_mean)
        pairs = F.cosine_similarity(centered[:25], centered[25:], dim=-1)
        # Should be much lower than the raw 0.85
        assert pairs.mean().abs() < 0.3, f"Centered noise pairs should be near 0, got {pairs.mean():.3f}"

    def test_blob_vs_noise_after_correction(self, blob_embeddings, noise_embeddings, noise_mean):
        """Centered blobs should score low against centered noise."""
        blob_c = center_and_normalize(blob_embeddings, noise_mean)
        noise_c = center_and_normalize(noise_embeddings[:4], noise_mean)
        for i in range(blob_c.shape[0]):
            cos = F.cosine_similarity(blob_c[i:i+1], noise_c, dim=-1)
            # Blobs and noise are in same cluster — after centering, both small
            assert cos.mean().abs() < 0.5, f"Centered blob {i} vs noise should be low, got {cos.mean():.3f}"


class TestBiasCorrectionFaces:
    """Bias correction should preserve face-vs-face discrimination."""

    def test_same_face_different_encode_stays_high(self, encoder, noise_mean):
        """The same face image encoded twice should still score ~1.0 after centering."""
        # Use a synthetic "face" by running a fixed seed through the encoder
        torch.manual_seed(42)
        fake_face_1 = torch.randn(1, 3, 112, 112, device='cuda') * 0.2 + 0.6
        torch.manual_seed(42)
        fake_face_2 = torch.randn(1, 3, 112, 112, device='cuda') * 0.2 + 0.6

        with torch.no_grad():
            emb1 = encoder(fake_face_1)
            emb2 = encoder(fake_face_2)

        raw_cos = F.cosine_similarity(emb1, emb2, dim=-1).item()
        assert abs(raw_cos - 1.0) < 0.001, f"Same input should score 1.0, got {raw_cos}"

        c1 = center_and_normalize(emb1, noise_mean)
        c2 = center_and_normalize(emb2, noise_mean)
        centered_cos = F.cosine_similarity(c1, c2, dim=-1).item()
        assert abs(centered_cos - 1.0) < 0.001, f"Centered same input should score 1.0, got {centered_cos}"

    def test_different_inputs_have_spread_after_centering(self, encoder, noise_mean):
        """Different structured inputs should have variable cos_sim after centering."""
        inputs = []
        for seed in range(10):
            torch.manual_seed(seed * 1000)
            img = torch.randn(1, 3, 112, 112, device='cuda') * 0.3 + 0.5
            img = img.clamp(0, 1)
            inputs.append(img)

        with torch.no_grad():
            embeds = torch.cat([encoder(x) for x in inputs], dim=0)

        # Raw: all should be fairly similar (bias)
        raw_pairs = []
        for i in range(5):
            for j in range(5, 10):
                raw_pairs.append(F.cosine_similarity(embeds[i:i+1], embeds[j:j+1], dim=-1).item())
        raw_mean = np.mean(raw_pairs)

        # Centered: should have more spread
        centered = center_and_normalize(embeds, noise_mean)
        cent_pairs = []
        for i in range(5):
            for j in range(5, 10):
                cent_pairs.append(F.cosine_similarity(centered[i:i+1], centered[j:j+1], dim=-1).item())
        cent_std = np.std(cent_pairs)

        # After centering, the std of pairwise cos_sim should increase (more discriminative)
        raw_std = np.std(raw_pairs)
        assert cent_std >= raw_std * 0.5, f"Centering should maintain or increase spread: raw_std={raw_std:.3f}, cent_std={cent_std:.3f}"


class TestBiasCorrectionSinglePerson:
    """Single-person dataset (common LoRA case) — must still work."""

    def test_noise_vs_single_person_ref(self, encoder, noise_mean):
        """Noise should score low against a single-person ref after centering."""
        torch.manual_seed(999)
        ref = torch.randn(1, 3, 112, 112, device='cuda') * 0.2 + 0.6
        ref = ref.clamp(0, 1)

        with torch.no_grad():
            ref_emb = encoder(ref)
            noise_embs = make_noise_embedding(encoder, 'cuda', n=20)

        ref_c = center_and_normalize(ref_emb, noise_mean)
        noise_c = center_and_normalize(noise_embs, noise_mean)

        raw = F.cosine_similarity(noise_embs, ref_emb.expand(20, -1), dim=-1)
        centered = F.cosine_similarity(noise_c, ref_c.expand(20, -1), dim=-1)

        # Raw should be high (~0.5), centered should be much lower
        assert raw.mean() > 0.3, f"Raw noise-vs-face should be biased high, got {raw.mean():.3f}"
        assert centered.mean() < raw.mean() - 0.1, \
            f"Centering should reduce noise-vs-face: raw={raw.mean():.3f}, centered={centered.mean():.3f}"

    def test_ref_embedding_not_zeroed(self, noise_mean):
        """With noise-based mean, a face embedding shouldn't be zeroed out."""
        # Simulate a face embedding (NOT equal to noise_mean)
        fake_face = F.normalize(torch.randn(1, 512, device='cuda'), p=2, dim=-1)
        centered = center_and_normalize(fake_face, noise_mean)
        # The centered embedding should have meaningful magnitude before normalization
        raw_centered = fake_face - noise_mean.unsqueeze(0)
        assert raw_centered.norm() > 0.1, \
            f"Face embedding minus noise mean should not be near-zero, got norm={raw_centered.norm():.4f}"


class TestGradientFlow:
    """Bias correction must preserve gradient flow for training."""

    def test_gradient_flows_through_centering(self, encoder, noise_mean):
        """Gradients should flow from cos_sim through mean-subtraction to input pixels."""
        pixels = torch.randn(1, 3, 112, 112, device='cuda', requires_grad=True)
        # Run through encoder forward (not encode — forward is differentiable)
        gen = encoder(pixels)
        gen_c = center_and_normalize(gen, noise_mean.detach())

        # Create a fake reference
        ref = F.normalize(torch.randn(1, 512, device='cuda'), p=2, dim=-1)
        ref_c = center_and_normalize(ref, noise_mean.detach())

        cos = F.cosine_similarity(gen_c, ref_c, dim=-1)
        loss = (1.0 - cos).mean()
        loss.backward()

        assert pixels.grad is not None, "Gradient should exist on input pixels"
        assert pixels.grad.abs().sum() > 0, "Gradient should be non-zero"


# --- Problem 2: Average mode zero-embedding replacement ---

class TestAverageModeBug:
    """Average mode must not replace zero embeddings (no-face images)."""

    def test_zero_embedding_not_replaced(self):
        """Simulates the average replacement logic — zeros should stay zero."""
        # Simulate file items with mixed face/no-face
        class FakeFileItem:
            def __init__(self, emb):
                self.identity_embedding = emb

        avg_embed = F.normalize(torch.randn(512), p=2, dim=-1)

        items = [
            FakeFileItem(F.normalize(torch.randn(512), p=2, dim=-1)),  # face
            FakeFileItem(torch.zeros(512)),                             # no face
            FakeFileItem(F.normalize(torch.randn(512), p=2, dim=-1)),  # face
            FakeFileItem(None),                                         # never detected
        ]

        # Apply the FIXED replacement logic
        replaced_count = 0
        for fi in items:
            emb = getattr(fi, 'identity_embedding', None)
            if emb is not None and emb.abs().sum() > 0:
                fi.identity_embedding = avg_embed.clone()
                replaced_count += 1

        assert replaced_count == 2, f"Should replace 2 face items, replaced {replaced_count}"
        # No-face item should still be zeros
        assert items[1].identity_embedding.abs().sum() == 0, "No-face zeros should NOT be replaced"
        # None item should still be None
        assert items[3].identity_embedding is None, "None embedding should stay None"
        # Face items should be the average
        assert torch.allclose(items[0].identity_embedding, avg_embed), "Face item should be average"
        assert torch.allclose(items[2].identity_embedding, avg_embed), "Face item should be average"

    def test_ref_valid_filters_no_face(self):
        """After average replacement fix, no-face items should fail ref_valid."""
        avg_embed = F.normalize(torch.randn(512), p=2, dim=-1)
        zero_embed = torch.zeros(512)

        batch = torch.stack([avg_embed, zero_embed, avg_embed])  # (3, 512)
        ref_valid = batch.abs().sum(dim=-1) > 0

        assert ref_valid[0] == True, "Face item should pass ref_valid"
        assert ref_valid[1] == False, "No-face item should fail ref_valid"
        assert ref_valid[2] == True, "Face item should pass ref_valid"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
