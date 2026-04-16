"""Tests for VAE perceptual anchor loss.

Run with: python testing/test_vae_anchor_loss.py
Requires a Flux 2 VAE safetensors file. The test will look for the VAE at
common locations, or you can set VAE_ANCHOR_PATH environment variable.

Tests:
1. Load encoder and verify multi-scale feature extraction
2. Verify loss=0 for identical inputs
3. Verify loss increases monotonically with noise
4. Verify gradient flows back to pixel input
5. Test on real images (encode clean, add noise, verify correlation)
6. Verify cached features match live-computed features
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from safetensors.torch import save_file, load_file

from toolkit.vae_anchor import (
    VAEAnchorEncoder,
    FEATURE_LEVELS,
    encode_reference_features,
    cache_vae_anchor_features,
)
from toolkit.config_modules import FaceIDConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_vae_path():
    """Find a Flux 2 VAE safetensors file for testing."""
    # Check environment variable first
    env_path = os.getenv('VAE_ANCHOR_PATH', '')
    if env_path and os.path.exists(env_path):
        return env_path

    # Common locations
    candidates = [
        os.path.expanduser('~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/*/ae.safetensors'),
        os.path.expanduser('~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots/*/ae.safetensors'),
    ]
    import glob
    for pattern in candidates:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]

    return None


class _FakeFileItem:
    """Mimics FileItemDTO for caching tests."""
    def __init__(self, path):
        self.path = path
        self.vae_anchor_features = None
        self.dataset_config = type('DC', (), {
            'vae_anchor_loss_weight': 1.0,
            'vae_anchor_loss_min_t': 0.0,
            'vae_anchor_loss_max_t': 0.5,
        })()


def _run_test(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def _make_test_image(size=256):
    """Create a test image with some structure (not just noise)."""
    # Gradient + circles pattern
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    # Horizontal gradient
    for x in range(size):
        arr[:, x, 0] = int(255 * x / size)
    # Vertical gradient
    for y in range(size):
        arr[y, :, 1] = int(255 * y / size)
    # Some circles
    for cx, cy, r in [(64, 64, 30), (192, 192, 40), (128, 128, 50)]:
        yy, xx = np.ogrid[:size, :size]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        arr[mask, 2] = 200
    return Image.fromarray(arr, 'RGB')


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_encoder_loads_and_extracts_features():
    """Test 1: Load VAE encoder and verify feature extraction at multiple scales."""
    vae_path = _find_vae_path()
    assert vae_path is not None, (
        "No VAE model found. Set VAE_ANCHOR_PATH or ensure Flux 2 VAE is cached."
    )

    encoder = VAEAnchorEncoder(vae_path=vae_path)
    encoder.load(device=torch.device('cuda'), dtype=torch.float32)

    # Create test input: (1, 3, 256, 256) in [-1, 1]
    x = torch.randn(1, 3, 256, 256, device='cuda')
    x = x.clamp(-1, 1)

    final, features = encoder.encode_with_features(x)

    # Verify we got all expected levels
    for level in FEATURE_LEVELS:
        assert level in features, f"Missing feature level: {level}"
        feat = features[level]
        assert feat.dim() == 4, f"{level}: expected 4D tensor, got {feat.dim()}D"
        assert feat.shape[0] == 1, f"{level}: batch size should be 1, got {feat.shape[0]}"
        print(f"    {level}: shape={list(feat.shape)}")

    # Verify expected channel counts and spatial sizes
    # ch_mult = [1, 2, 4, 4], ch=128, input 256x256
    # Level 1: 256 ch, 128x128 (after down[1].block[1], before downsample)
    # Level 2: 512 ch, 64x64
    # Level 3: 512 ch, 32x32
    # Mid: 512 ch, 32x32
    assert features['level_1'].shape[1] == 256, f"Level 1 should have 256 channels, got {features['level_1'].shape[1]}"
    assert features['level_2'].shape[1] == 512, f"Level 2 should have 512 channels, got {features['level_2'].shape[1]}"
    assert features['level_3'].shape[1] == 512, f"Level 3 should have 512 channels, got {features['level_3'].shape[1]}"
    assert features['mid'].shape[1] == 512, f"Mid should have 512 channels, got {features['mid'].shape[1]}"

    # Verify spatial sizes decrease at each level
    s1 = features['level_1'].shape[2] * features['level_1'].shape[3]
    s2 = features['level_2'].shape[2] * features['level_2'].shape[3]
    s3 = features['level_3'].shape[2] * features['level_3'].shape[3]
    assert s1 > s2 > s3, f"Spatial sizes should decrease: {s1} > {s2} > {s3}"

    # Final encoding shape check
    print(f"    final: shape={list(final.shape)}")
    assert final.shape[1] == 64, f"Final should have 2*z_channels=64, got {final.shape[1]}"

    encoder.cleanup()
    print(f"    VAE path: {vae_path}")


def test_loss_zero_for_identical_inputs():
    """Test 2: Loss should be exactly 0 when pred == ref."""
    vae_path = _find_vae_path()
    assert vae_path is not None, "No VAE model found."

    encoder = VAEAnchorEncoder(vae_path=vae_path)
    encoder.load(device=torch.device('cuda'), dtype=torch.float32)

    # Create a structured test image
    pil_img = _make_test_image(256)
    import torchvision.transforms.functional as TF
    img = TF.to_tensor(pil_img).unsqueeze(0).cuda() * 2.0 - 1.0  # [-1, 1]

    with torch.no_grad():
        _, features = encoder.encode_with_features(img)

    # Compare features against themselves
    loss, per_level = VAEAnchorEncoder.compute_loss(features, features)

    print(f"    loss for identical inputs: {loss.item():.10f}")
    for k, v in per_level.items():
        print(f"    {k}: {v:.10f}")

    assert loss.item() < 1e-6, f"Loss should be ~0 for identical inputs, got {loss.item()}"
    for k, v in per_level.items():
        assert v < 1e-6, f"Per-level loss for {k} should be ~0, got {v}"

    encoder.cleanup()


def test_loss_increases_with_noise():
    """Test 3: Loss should increase monotonically with added noise level."""
    vae_path = _find_vae_path()
    assert vae_path is not None, "No VAE model found."

    encoder = VAEAnchorEncoder(vae_path=vae_path)
    encoder.load(device=torch.device('cuda'), dtype=torch.float32)

    pil_img = _make_test_image(256)
    import torchvision.transforms.functional as TF
    img = TF.to_tensor(pil_img).unsqueeze(0).cuda() * 2.0 - 1.0

    with torch.no_grad():
        _, ref_features = encoder.encode_with_features(img)

    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5]
    losses = []

    for sigma in noise_levels:
        noisy = img + torch.randn_like(img) * sigma
        noisy = noisy.clamp(-1, 1)
        with torch.no_grad():
            _, pred_features = encoder.encode_with_features(noisy)
        loss, _ = VAEAnchorEncoder.compute_loss(pred_features, ref_features)
        losses.append(loss.item())
        print(f"    sigma={sigma:.2f} -> loss={loss.item():.6f}")

    # Check monotonic increase (allowing small tolerance for noise)
    for i in range(1, len(losses)):
        assert losses[i] >= losses[i - 1] - 0.01, (
            f"Loss should increase with noise: sigma={noise_levels[i-1]:.2f}->{noise_levels[i]:.2f}, "
            f"loss={losses[i-1]:.6f}->{losses[i]:.6f}"
        )

    # First entry should be ~0, last should be significantly larger
    assert losses[0] < 0.01, f"Loss at sigma=0 should be ~0, got {losses[0]}"
    assert losses[-1] > losses[0] * 10, (
        f"Loss at sigma=1.5 should be much larger than sigma=0: {losses[-1]} vs {losses[0]}"
    )

    encoder.cleanup()


def test_gradient_flows_to_input():
    """Test 4: Verify gradients flow from loss back to pixel input."""
    vae_path = _find_vae_path()
    assert vae_path is not None, "No VAE model found."

    encoder = VAEAnchorEncoder(vae_path=vae_path)
    encoder.load(device=torch.device('cuda'), dtype=torch.float32)

    # Input with grad tracking
    x = torch.randn(1, 3, 128, 128, device='cuda', requires_grad=True)

    # Create reference features (detached)
    with torch.no_grad():
        _, ref_features = encoder.encode_with_features(x.detach())

    # Forward with grad
    _, pred_features = encoder.encode_with_features(x)
    loss, _ = VAEAnchorEncoder.compute_loss(pred_features, ref_features)

    # Even with identical features, numerical noise creates tiny loss
    # The important thing is that backward() works and x.grad is not None
    loss.backward()

    assert x.grad is not None, "Gradient should flow back to input"
    assert x.grad.shape == x.shape, f"Gradient shape mismatch: {x.grad.shape} vs {x.shape}"
    grad_norm = x.grad.norm().item()
    print(f"    grad norm on input: {grad_norm:.6f}")
    # Grad norm can be very small for nearly-identical inputs; just verify it exists

    # Now test with a real difference
    x2 = torch.randn(1, 3, 128, 128, device='cuda', requires_grad=True)
    ref_different = torch.randn(1, 3, 128, 128, device='cuda')
    with torch.no_grad():
        _, ref_feat2 = encoder.encode_with_features(ref_different)

    _, pred_feat2 = encoder.encode_with_features(x2)
    loss2, _ = VAEAnchorEncoder.compute_loss(pred_feat2, ref_feat2)
    loss2.backward()

    assert x2.grad is not None, "Gradient should flow for different inputs"
    grad_norm2 = x2.grad.norm().item()
    print(f"    grad norm (different inputs): {grad_norm2:.6f}")
    assert grad_norm2 > 0, "Gradient norm should be > 0 for different inputs"

    encoder.cleanup()


def test_real_images():
    """Test 5: Test on real images with noise correlation."""
    vae_path = _find_vae_path()
    assert vae_path is not None, "No VAE model found."

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bodytest_dir = os.path.join(repo_root, 'bodytest')
    if not os.path.isdir(bodytest_dir):
        # Also check main repo if we're in a worktree
        bodytest_dir = '/home/z/Documents/repos/ai-toolkit/bodytest'
    if not os.path.isdir(bodytest_dir):
        print("    SKIP: bodytest/ directory not found")
        return

    # Find first valid image
    img_path = None
    for f in os.listdir(bodytest_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            img_path = os.path.join(bodytest_dir, f)
            break

    if img_path is None:
        print("    SKIP: no images in bodytest/")
        return

    print(f"    using image: {os.path.basename(img_path)}")

    encoder = VAEAnchorEncoder(vae_path=vae_path)
    encoder.load(device=torch.device('cuda'), dtype=torch.float32)

    pil_img = Image.open(img_path).convert('RGB')
    ref_features = encode_reference_features(encoder, pil_img, target_size=256)

    # Verify features are on CPU fp16
    for level, feat in ref_features.items():
        assert feat.device.type == 'cpu', f"{level} should be on CPU, got {feat.device}"
        assert feat.dtype == torch.float16, f"{level} should be fp16, got {feat.dtype}"
        print(f"    ref {level}: shape={list(feat.shape)}")

    # Add increasing noise and verify loss correlation
    import torchvision.transforms.functional as TF
    w, h = pil_img.size
    scale = 256 / min(w, h)
    new_w = (int(w * scale) // 8) * 8
    new_h = (int(h * scale) // 8) * 8
    pil_resized = pil_img.resize((new_w, new_h))
    img_tensor = TF.to_tensor(pil_resized).unsqueeze(0).cuda() * 2.0 - 1.0

    noise_levels = [0.0, 0.1, 0.3, 0.5, 1.0]
    losses = []

    for sigma in noise_levels:
        noisy = img_tensor + torch.randn_like(img_tensor) * sigma
        noisy = noisy.clamp(-1, 1)
        with torch.no_grad():
            _, pred_features = encoder.encode_with_features(noisy)
        # Convert ref to proper format for comparison
        ref_gpu = {k: v.float().cuda() for k, v in ref_features.items()}
        loss, per_level = VAEAnchorEncoder.compute_loss(pred_features, ref_gpu)
        losses.append(loss.item())
        level_str = ' '.join(f'{k}={v:.4f}' for k, v in per_level.items())
        print(f"    sigma={sigma:.1f} -> total={loss.item():.6f} {level_str}")

    # Verify monotonic correlation
    for i in range(1, len(losses)):
        assert losses[i] >= losses[i - 1] - 0.05, (
            f"Loss should generally increase: sigma {noise_levels[i-1]}->{noise_levels[i]}, "
            f"loss {losses[i-1]:.4f}->{losses[i]:.4f}"
        )

    encoder.cleanup()


def test_cache_matches_live():
    """Test 6: Cached features should match live-computed features."""
    vae_path = _find_vae_path()
    assert vae_path is not None, "No VAE model found."

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test image and save it
        pil_img = _make_test_image(256)
        img_path = os.path.join(tmpdir, 'test_image.png')
        pil_img.save(img_path)

        # Create fake file item
        file_item = _FakeFileItem(img_path)

        # Create config
        config = FaceIDConfig(
            vae_anchor_loss_weight=1.0,
            vae_anchor_model_path=vae_path,
        )

        # Cache features
        cache_vae_anchor_features([file_item], config)

        assert file_item.vae_anchor_features is not None, "Features should be set on file_item"
        cached_features = {k: v.clone() for k, v in file_item.vae_anchor_features.items()}

        # Verify cache file exists
        cache_path = os.path.join(tmpdir, '_face_id_cache', 'test_image_vae_anchor.safetensors')
        assert os.path.exists(cache_path), f"Cache file should exist at {cache_path}"

        # Load from cache (simulate second run)
        file_item2 = _FakeFileItem(img_path)
        cache_vae_anchor_features([file_item2], config)
        assert file_item2.vae_anchor_features is not None, "Features should load from cache"

        # Compare cached vs fresh
        for level in FEATURE_LEVELS:
            if level in cached_features and level in file_item2.vae_anchor_features:
                diff = (cached_features[level].float() - file_item2.vae_anchor_features[level].float()).abs().max()
                print(f"    {level} cache vs reload max diff: {diff.item():.8f}")
                assert diff < 0.01, f"Cached features should match: {level} diff={diff.item()}"

        # Also compare against live computation
        encoder = VAEAnchorEncoder(vae_path=vae_path)
        encoder.load(device=torch.device('cuda'), dtype=torch.float32)
        live_features = encode_reference_features(encoder, pil_img, target_size=512)

        for level in FEATURE_LEVELS:
            if level in cached_features and level in live_features:
                diff = (cached_features[level].float() - live_features[level].float()).abs().max()
                print(f"    {level} cached vs live max diff: {diff.item():.8f}")
                # fp16 rounding can cause small diffs
                assert diff < 0.1, f"Cached should match live: {level} diff={diff.item()}"

        encoder.cleanup()


def test_batch_collection():
    """Test that features collect correctly into DataLoaderBatchDTO format."""
    # Create synthetic features matching expected shapes
    batch_size = 2
    features_per_item = []
    for i in range(batch_size):
        feat = {
            'level_1': torch.randn(256, 128, 128) * (i + 1),
            'level_2': torch.randn(512, 64, 64) * (i + 1),
            'level_3': torch.randn(512, 32, 32) * (i + 1),
            'mid': torch.randn(512, 32, 32) * (i + 1),
        }
        features_per_item.append(feat)

    # Simulate batch collection (same logic as data_loader.py)
    from toolkit.vae_anchor import FEATURE_LEVELS as FL
    ref_feats = features_per_item[0]
    batch_feats = {}
    for level in FL:
        level_tensors = []
        for f in features_per_item:
            if f is not None and level in f:
                t = f[level]
                level_tensors.append(t.unsqueeze(0) if t.dim() == 3 else t)
            else:
                level_tensors.append(torch.zeros_like(
                    ref_feats[level].unsqueeze(0) if ref_feats[level].dim() == 3 else ref_feats[level]
                ))
        batch_feats[level] = torch.cat(level_tensors, dim=0)

    # Verify shapes
    for level in FL:
        assert batch_feats[level].shape[0] == batch_size, (
            f"{level}: batch dim should be {batch_size}, got {batch_feats[level].shape[0]}"
        )
        print(f"    {level}: batch shape={list(batch_feats[level].shape)}")

    # Verify compute_loss works with batched features — returns (B,) per-sample losses
    pred_feats = {k: torch.randn_like(v) for k, v in batch_feats.items()}
    loss, per_level = VAEAnchorEncoder.compute_loss(pred_feats, batch_feats)
    assert loss.shape == (batch_size,), f"Expected shape ({batch_size},), got {loss.shape}"
    print(f"    batch loss (per-sample): {loss.tolist()}")
    assert (loss > 0).all(), "Loss should be > 0 for different features"
    assert not torch.isnan(loss).any(), "Loss should not be NaN"
    assert not torch.isinf(loss).any(), "Loss should not be inf"


def test_size_mismatch_handling():
    """Test that compute_loss handles size mismatch between pred and ref."""
    # Pred at one resolution, ref at another
    pred_features = {
        'level_1': torch.randn(1, 256, 64, 64),
        'level_2': torch.randn(1, 512, 32, 32),
    }
    ref_features = {
        'level_1': torch.randn(1, 256, 128, 128),  # different spatial size
        'level_2': torch.randn(1, 512, 64, 64),     # different spatial size
    }

    loss, per_level = VAEAnchorEncoder.compute_loss(pred_features, ref_features)
    print(f"    size mismatch loss: {loss.item():.6f}")
    assert not torch.isnan(loss), "Loss should not be NaN with size mismatch"
    assert loss.item() > 0, "Loss should be > 0"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("VAE Anchor Loss Tests")
    print("=" * 60)

    vae_path = _find_vae_path()
    if vae_path:
        print(f"Using VAE: {vae_path}")
    else:
        print("WARNING: No VAE model found. GPU tests will be skipped.")
        print("Set VAE_ANCHOR_PATH to point to a Flux 2 ae.safetensors file.")

    results = []

    # Tests that don't require GPU/VAE
    results.append(_run_test("batch_collection", test_batch_collection))
    results.append(_run_test("size_mismatch_handling", test_size_mismatch_handling))

    # Tests that require GPU + VAE model
    if vae_path and torch.cuda.is_available():
        results.append(_run_test("encoder_loads_and_extracts_features", test_encoder_loads_and_extracts_features))
        results.append(_run_test("loss_zero_for_identical_inputs", test_loss_zero_for_identical_inputs))
        results.append(_run_test("loss_increases_with_noise", test_loss_increases_with_noise))
        results.append(_run_test("gradient_flows_to_input", test_gradient_flows_to_input))
        results.append(_run_test("real_images", test_real_images))
        results.append(_run_test("cache_matches_live", test_cache_matches_live))
    else:
        skipped = ["encoder_loads_and_extracts_features", "loss_zero_for_identical_inputs",
                    "loss_increases_with_noise", "gradient_flows_to_input",
                    "real_images", "cache_matches_live"]
        for name in skipped:
            print(f"  SKIP  {name} (no GPU or VAE model)")

    print()
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    if passed < total:
        sys.exit(1)
