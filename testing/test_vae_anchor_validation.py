"""Comprehensive validation tests for VAE anchor loss feature.

Run with: python testing/test_vae_anchor_validation.py

End-to-end tests using real images from /home/z/Documents/repos/ai-toolkit/bodytest.
Tests cover: auto-resolve, encoder loading, caching, loss properties,
gradient flow, batch collection, per-dataset weights, timestep gating,
ref validity, resolution mismatch, and real image perceptual quality.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
import tempfile
import traceback
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
# Constants
# ---------------------------------------------------------------------------

BODYTEST_DIR = '/home/z/Documents/repos/ai-toolkit/bodytest'
HAS_CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_bodytest_images(n=None):
    """Return list of image paths from the bodytest directory."""
    exts = ('.jpg', '.jpeg', '.png', '.webp')
    images = []
    for f in sorted(os.listdir(BODYTEST_DIR)):
        if f.lower().endswith(exts) and not f.startswith('.'):
            images.append(os.path.join(BODYTEST_DIR, f))
            if n is not None and len(images) >= n:
                break
    return images


class _FakeFileItem:
    """Mimics FileItemDTO for caching tests."""
    def __init__(self, path, vae_anchor_loss_weight=None, vae_anchor_loss_min_t=None,
                 vae_anchor_loss_max_t=None):
        self.path = path
        self.vae_anchor_features = None
        self.dataset_config = type('DC', (), {
            'vae_anchor_loss_weight': vae_anchor_loss_weight,
            'vae_anchor_loss_min_t': vae_anchor_loss_min_t,
            'vae_anchor_loss_max_t': vae_anchor_loss_max_t,
        })()
        # Mimic the FileItemDTO per-sample fields (set from dataset_config in __init__)
        self.vae_anchor_loss_weight = vae_anchor_loss_weight
        self.vae_anchor_loss_min_t = vae_anchor_loss_min_t
        self.vae_anchor_loss_max_t = vae_anchor_loss_max_t


# Shared encoder (loaded once to save time across tests)
_shared_encoder = None


def _get_shared_encoder():
    """Lazily load and return a shared VAEAnchorEncoder instance."""
    global _shared_encoder
    if _shared_encoder is None:
        _shared_encoder = VAEAnchorEncoder(vae_path='')
        _shared_encoder.load(device=torch.device('cuda'), dtype=torch.float32)
    return _shared_encoder


_results = []  # (name, status) where status is 'PASS', 'FAIL', or 'SKIP'


def _run_test(name, fn, requires_gpu=False):
    """Run a single test, recording the result."""
    if requires_gpu and not HAS_CUDA:
        print(f"  SKIP  {name} -- no CUDA GPU available")
        _results.append((name, 'SKIP'))
        return
    try:
        fn()
        print(f"  PASS  {name}")
        _results.append((name, 'PASS'))
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()
        _results.append((name, 'FAIL'))


# ---------------------------------------------------------------------------
# 1. auto_resolve_path
# ---------------------------------------------------------------------------

def test_auto_resolve_path():
    """Test that _resolve_vae_path('') successfully resolves to a real file."""
    resolved = VAEAnchorEncoder._resolve_vae_path('')
    assert os.path.exists(resolved), f"Resolved path does not exist: {resolved}"
    assert resolved.endswith('.safetensors'), (
        f"Resolved path should end with .safetensors, got: {resolved}"
    )
    print(f"    resolved: {resolved}")
    fsize_mb = os.path.getsize(resolved) / (1024 * 1024)
    print(f"    file size: {fsize_mb:.1f} MB")
    assert fsize_mb > 10, f"VAE file seems too small ({fsize_mb:.1f} MB), possibly corrupt"


# ---------------------------------------------------------------------------
# 2. encoder_loads_from_auto_path
# ---------------------------------------------------------------------------

def test_encoder_loads_from_auto_path():
    """Load VAEAnchorEncoder with empty vae_path, verify encode works."""
    encoder = _get_shared_encoder()

    # Encode a dummy tensor
    x = torch.randn(1, 3, 256, 256, device='cuda').clamp(-1, 1)
    final, features = encoder.encode_with_features(x)

    # Verify all 4 levels
    for level in FEATURE_LEVELS:
        assert level in features, f"Missing feature level: {level}"
        assert features[level].dim() == 4, f"{level}: expected 4D, got {features[level].dim()}D"
        print(f"    {level}: {list(features[level].shape)}")

    # Verify channel counts
    assert features['level_1'].shape[1] == 256, f"level_1 channels: {features['level_1'].shape[1]}"
    assert features['level_2'].shape[1] == 512, f"level_2 channels: {features['level_2'].shape[1]}"
    assert features['level_3'].shape[1] == 512, f"level_3 channels: {features['level_3'].shape[1]}"
    assert features['mid'].shape[1] == 512, f"mid channels: {features['mid'].shape[1]}"

    # Final encoding should have 2*z_channels=64
    assert final.shape[1] == 64, f"final channels: {final.shape[1]}"
    print(f"    final: {list(final.shape)}")


# ---------------------------------------------------------------------------
# 3. cache_real_images
# ---------------------------------------------------------------------------

def test_cache_real_images():
    """Cache VAE anchor features for real bodytest images, verify structure."""
    images = _get_bodytest_images(n=3)
    assert len(images) >= 2, f"Need at least 2 images in bodytest, found {len(images)}"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy images to temp dir so caching writes there
        file_items = []
        for img_path in images:
            dst = os.path.join(tmpdir, os.path.basename(img_path))
            shutil.copy2(img_path, dst)
            file_items.append(_FakeFileItem(dst))

        config = FaceIDConfig(vae_anchor_loss_weight=1.0, vae_anchor_model_path='')
        cache_vae_anchor_features(file_items, config)

        for fi in file_items:
            fname = os.path.basename(fi.path)
            # Features should be set
            assert fi.vae_anchor_features is not None, f"No features for {fname}"

            # All 4 levels present
            for level in FEATURE_LEVELS:
                assert level in fi.vae_anchor_features, f"{fname} missing {level}"
                feat = fi.vae_anchor_features[level]
                # fp16 on CPU
                assert feat.dtype == torch.float16, f"{fname}/{level}: dtype={feat.dtype}, expected fp16"
                assert feat.device.type == 'cpu', f"{fname}/{level}: device={feat.device}, expected cpu"
                assert feat.dim() == 4, f"{fname}/{level}: dim={feat.dim()}, expected 4"
                print(f"    {fname}/{level}: {list(feat.shape)} {feat.dtype}")

        # Cache files created
        cache_dir = os.path.join(tmpdir, '_face_id_cache')
        assert os.path.isdir(cache_dir), f"Cache dir not created: {cache_dir}"
        cache_files = os.listdir(cache_dir)
        assert len(cache_files) == len(images), (
            f"Expected {len(images)} cache files, got {len(cache_files)}"
        )
        for cf in cache_files:
            assert cf.endswith('_vae_anchor.safetensors'), f"Unexpected cache file: {cf}"
        print(f"    cache dir: {cache_dir} ({len(cache_files)} files)")


# ---------------------------------------------------------------------------
# 4. cache_reload_matches
# ---------------------------------------------------------------------------

def test_cache_reload_matches():
    """Cache features, reload from disk, verify they match within fp16 tolerance."""
    images = _get_bodytest_images(n=2)
    assert len(images) >= 1, "Need at least 1 image in bodytest"

    with tempfile.TemporaryDirectory() as tmpdir:
        dst = os.path.join(tmpdir, os.path.basename(images[0]))
        shutil.copy2(images[0], dst)

        config = FaceIDConfig(vae_anchor_loss_weight=1.0, vae_anchor_model_path='')

        # First pass: compute and cache
        fi1 = _FakeFileItem(dst)
        cache_vae_anchor_features([fi1], config)
        original = {k: v.clone() for k, v in fi1.vae_anchor_features.items()}

        # Second pass: should load from disk
        fi2 = _FakeFileItem(dst)
        cache_vae_anchor_features([fi2], config)
        reloaded = fi2.vae_anchor_features

        for level in FEATURE_LEVELS:
            assert level in original and level in reloaded, f"Missing {level}"
            diff = (original[level].float() - reloaded[level].float()).abs().max().item()
            print(f"    {level} max diff (cached vs reloaded): {diff:.10f}")
            # Exact match expected since both are loaded from the same safetensors file
            assert diff == 0.0, f"{level}: reloaded features differ by {diff}"

        print("    cached == reloaded: exact match")


# ---------------------------------------------------------------------------
# 5. per_sample_loss_shape
# ---------------------------------------------------------------------------

def test_per_sample_loss_shape():
    """Verify compute_loss returns (B,) tensor, not scalar. Test with B=3."""
    batch_size = 3
    pred_features = {
        'level_1': torch.randn(batch_size, 256, 32, 32),
        'level_2': torch.randn(batch_size, 512, 16, 16),
        'level_3': torch.randn(batch_size, 512, 8, 8),
        'mid': torch.randn(batch_size, 512, 8, 8),
    }
    ref_features = {
        'level_1': torch.randn(batch_size, 256, 32, 32),
        'level_2': torch.randn(batch_size, 512, 16, 16),
        'level_3': torch.randn(batch_size, 512, 8, 8),
        'mid': torch.randn(batch_size, 512, 8, 8),
    }

    loss, per_level = VAEAnchorEncoder.compute_loss(pred_features, ref_features)

    assert loss.dim() == 1, f"Loss should be 1D, got {loss.dim()}D"
    assert loss.shape == (batch_size,), f"Loss shape should be ({batch_size},), got {loss.shape}"
    assert not torch.isnan(loss).any(), "Loss contains NaN"
    assert not torch.isinf(loss).any(), "Loss contains Inf"
    assert (loss > 0).all(), f"Loss should be > 0 for random features: {loss.tolist()}"
    print(f"    loss shape: {loss.shape}, values: {[f'{v:.4f}' for v in loss.tolist()]}")

    # Verify per-level dict has all expected keys
    for level in FEATURE_LEVELS:
        assert level in per_level, f"Missing per_level key: {level}"
        assert isinstance(per_level[level], float), f"{level}: expected float, got {type(per_level[level])}"
    print(f"    per_level: {per_level}")


# ---------------------------------------------------------------------------
# 6. loss_zero_identical
# ---------------------------------------------------------------------------

def test_loss_zero_identical():
    """Encode same image twice, loss between them should be exactly 0."""
    encoder = _get_shared_encoder()

    # Use a real bodytest image for realistic feature distributions
    images = _get_bodytest_images(n=1)
    pil_img = Image.open(images[0]).convert('RGB')
    import torchvision.transforms.functional as TF

    # Resize to known dims, multiple of 8
    pil_img = pil_img.resize((256, 256))
    img = TF.to_tensor(pil_img).unsqueeze(0).cuda() * 2.0 - 1.0

    with torch.no_grad():
        _, features = encoder.encode_with_features(img)

    loss, per_level = VAEAnchorEncoder.compute_loss(features, features)

    print(f"    loss for identical features: {loss.item():.15f}")
    for k, v in per_level.items():
        print(f"    {k}: {v:.15f}")

    assert loss.item() == 0.0, f"Loss should be exactly 0.0, got {loss.item()}"


# ---------------------------------------------------------------------------
# 7. loss_monotonic_with_noise
# ---------------------------------------------------------------------------

def test_loss_monotonic_with_noise():
    """Encode clean image, add increasing noise, verify loss increases monotonically."""
    encoder = _get_shared_encoder()

    images = _get_bodytest_images(n=1)
    pil_img = Image.open(images[0]).convert('RGB').resize((256, 256))
    import torchvision.transforms.functional as TF
    img = TF.to_tensor(pil_img).unsqueeze(0).cuda() * 2.0 - 1.0

    with torch.no_grad():
        _, ref_features = encoder.encode_with_features(img)

    sigmas = [0.05, 0.1, 0.2, 0.5, 1.0]
    losses = []

    torch.manual_seed(42)
    for sigma in sigmas:
        noisy = img + torch.randn_like(img) * sigma
        noisy = noisy.clamp(-1, 1)
        with torch.no_grad():
            _, pred_features = encoder.encode_with_features(noisy)
        loss, _ = VAEAnchorEncoder.compute_loss(pred_features, ref_features)
        losses.append(loss.item())
        print(f"    sigma={sigma:.2f} -> loss={loss.item():.6f}")

    # Strict monotonic increase
    for i in range(1, len(losses)):
        assert losses[i] > losses[i - 1], (
            f"Loss not monotonically increasing: "
            f"sigma {sigmas[i-1]:.2f}={losses[i-1]:.6f} -> sigma {sigmas[i]:.2f}={losses[i]:.6f}"
        )

    # Verify meaningful dynamic range
    ratio = losses[-1] / max(losses[0], 1e-10)
    print(f"    dynamic range (sigma 1.0 / sigma 0.05): {ratio:.1f}x")
    assert ratio > 5, f"Expected at least 5x dynamic range, got {ratio:.1f}x"


# ---------------------------------------------------------------------------
# 8. gradient_flow
# ---------------------------------------------------------------------------

def test_gradient_flow():
    """Create input with requires_grad, encode, compute loss, backward. Verify grad exists."""
    encoder = _get_shared_encoder()

    x = torch.randn(1, 3, 128, 128, device='cuda', requires_grad=True)

    # Reference features (detached)
    with torch.no_grad():
        _, ref_features = encoder.encode_with_features(x.detach())

    # Forward with gradients
    _, pred_features = encoder.encode_with_features(x)
    loss, _ = VAEAnchorEncoder.compute_loss(pred_features, ref_features)

    # Even identical features can have tiny numerical differences when
    # one pass has grad tracking. Use sum to ensure scalar for backward.
    total_loss = loss.sum()
    total_loss.backward()

    assert x.grad is not None, "x.grad should not be None after backward()"
    assert x.grad.shape == x.shape, f"grad shape {x.grad.shape} != input shape {x.shape}"
    grad_norm = x.grad.norm().item()
    print(f"    grad norm (same image): {grad_norm:.8f}")

    # Now test with clearly different reference for non-trivial grad
    x2 = torch.randn(1, 3, 128, 128, device='cuda', requires_grad=True)
    different_ref = torch.randn(1, 3, 128, 128, device='cuda')
    with torch.no_grad():
        _, ref2 = encoder.encode_with_features(different_ref)

    _, pred2 = encoder.encode_with_features(x2)
    loss2, _ = VAEAnchorEncoder.compute_loss(pred2, ref2)
    loss2.sum().backward()

    assert x2.grad is not None, "x2.grad should not be None"
    grad_norm2 = x2.grad.norm().item()
    print(f"    grad norm (different ref): {grad_norm2:.6f}")
    assert grad_norm2 > 0, "Gradient norm should be > 0 for different inputs"


# ---------------------------------------------------------------------------
# 9. batch_collection
# ---------------------------------------------------------------------------

def test_batch_collection():
    """Simulate DataLoaderBatchDTO batch collection, verify (B, C, H, W) shapes."""
    batch_size = 4

    # Create per-item features mimicking cached fp16 CPU tensors (3D or 4D)
    items = []
    for i in range(batch_size):
        feat = {
            'level_1': torch.randn(1, 256, 128, 128).half(),
            'level_2': torch.randn(1, 512, 64, 64).half(),
            'level_3': torch.randn(1, 512, 32, 32).half(),
            'mid': torch.randn(1, 512, 32, 32).half(),
        }
        items.append(feat)

    # Replicate the exact batch collection logic from data_loader.py (lines 608-628)
    ref_feats = None
    for f in items:
        if f is not None:
            ref_feats = f
            break

    batch_feats = {}
    for level in FEATURE_LEVELS:
        level_tensors = []
        for f in items:
            if f is not None and level in f:
                t = f[level]
                level_tensors.append(t.unsqueeze(0) if t.dim() == 3 else t)
            else:
                level_tensors.append(torch.zeros_like(
                    ref_feats[level].unsqueeze(0) if ref_feats[level].dim() == 3 else ref_feats[level]
                ))
        batch_feats[level] = torch.cat(level_tensors, dim=0)

    # Verify shapes
    expected = {
        'level_1': (batch_size, 256, 128, 128),
        'level_2': (batch_size, 512, 64, 64),
        'level_3': (batch_size, 512, 32, 32),
        'mid': (batch_size, 512, 32, 32),
    }
    for level in FEATURE_LEVELS:
        assert batch_feats[level].shape == expected[level], (
            f"{level}: expected {expected[level]}, got {tuple(batch_feats[level].shape)}"
        )
        print(f"    {level}: {tuple(batch_feats[level].shape)}")

    # Also test with a mix of 3D (no batch dim) and None features
    items_mixed = []
    for i in range(batch_size):
        if i == 1:
            # Simulate missing features (None)
            items_mixed.append(None)
        elif i == 2:
            # 3D tensors (no batch dim)
            feat = {
                'level_1': torch.randn(256, 128, 128).half(),
                'level_2': torch.randn(512, 64, 64).half(),
                'level_3': torch.randn(512, 32, 32).half(),
                'mid': torch.randn(512, 32, 32).half(),
            }
            items_mixed.append(feat)
        else:
            feat = {
                'level_1': torch.randn(1, 256, 128, 128).half(),
                'level_2': torch.randn(1, 512, 64, 64).half(),
                'level_3': torch.randn(1, 512, 32, 32).half(),
                'mid': torch.randn(1, 512, 32, 32).half(),
            }
            items_mixed.append(feat)

    ref_feats2 = None
    for f in items_mixed:
        if f is not None:
            ref_feats2 = f
            break

    batch_feats2 = {}
    for level in FEATURE_LEVELS:
        level_tensors = []
        for f in items_mixed:
            if f is not None and level in f:
                t = f[level]
                level_tensors.append(t.unsqueeze(0) if t.dim() == 3 else t)
            else:
                level_tensors.append(torch.zeros_like(
                    ref_feats2[level].unsqueeze(0) if ref_feats2[level].dim() == 3 else ref_feats2[level]
                ))
        batch_feats2[level] = torch.cat(level_tensors, dim=0)

    for level in FEATURE_LEVELS:
        assert batch_feats2[level].shape == expected[level], (
            f"mixed {level}: expected {expected[level]}, got {tuple(batch_feats2[level].shape)}"
        )
    print("    mixed batch (None + 3D + 4D): shapes OK")


# ---------------------------------------------------------------------------
# 10. per_dataset_weight_propagation
# ---------------------------------------------------------------------------

def test_per_dataset_weight_propagation():
    """Create FileItemDTO-like objects with various vae_anchor_loss_weight values.
    Verify the weight lists are correctly collected."""
    # Weights: None (use global), 0.0 (disabled), 0.5 (custom)
    file_items = [
        _FakeFileItem('/fake/img1.jpg', vae_anchor_loss_weight=None),
        _FakeFileItem('/fake/img2.jpg', vae_anchor_loss_weight=0.0),
        _FakeFileItem('/fake/img3.jpg', vae_anchor_loss_weight=0.5),
    ]

    # Simulate the batch collection from DataLoaderBatchDTO
    weight_list = [x.vae_anchor_loss_weight for x in file_items]
    min_t_list = [x.vae_anchor_loss_min_t for x in file_items]
    max_t_list = [x.vae_anchor_loss_max_t for x in file_items]

    assert weight_list == [None, 0.0, 0.5], f"Weight list: {weight_list}"
    assert min_t_list == [None, None, None], f"Min t list: {min_t_list}"
    assert max_t_list == [None, None, None], f"Max t list: {max_t_list}"
    print(f"    weight_list: {weight_list}")
    print(f"    min_t_list:  {min_t_list}")
    print(f"    max_t_list:  {max_t_list}")

    # Simulate per-dataset weight gating from SDTrainer (lines 2373-2382)
    global_va_w = 1.0  # global weight from FaceIDConfig
    _has_per_ds_va_w = any(w is not None for w in weight_list)
    assert _has_per_ds_va_w, "Should detect per-dataset weights"

    va_weights = torch.tensor(
        [w if w is not None else global_va_w for w in weight_list],
        dtype=torch.float32,
    )
    expected_weights = torch.tensor([1.0, 0.0, 0.5])
    assert torch.equal(va_weights, expected_weights), (
        f"Resolved weights: {va_weights.tolist()}, expected: {expected_weights.tolist()}"
    )
    print(f"    resolved weights: {va_weights.tolist()}")

    # Sample 1 (w=0.0) should be gated off
    va_valid_mask = torch.ones(3, dtype=torch.bool)
    va_valid_mask = va_valid_mask & (va_weights > 0)
    assert va_valid_mask.tolist() == [True, False, True], (
        f"Valid mask after weight gating: {va_valid_mask.tolist()}"
    )
    print(f"    valid mask after gating: {va_valid_mask.tolist()}")


# ---------------------------------------------------------------------------
# 11. timestep_gating
# ---------------------------------------------------------------------------

def test_timestep_gating():
    """Simulate the per-sample timestep mask logic from SDTrainer."""
    t_ratio = torch.tensor([0.1, 0.3, 0.6, 0.8])
    min_t = 0.0
    max_t = 0.5

    # Replicate _per_sample_mask from SDTrainer (line 1498-1508)
    # Uses strict inequality: (t_ratio > min_vals) & (t_ratio < max_vals)
    batch_min_list = [None, None, None, None]
    batch_max_list = [None, None, None, None]

    min_vals = torch.tensor(
        [v if v is not None else min_t for v in batch_min_list],
        dtype=t_ratio.dtype,
    )
    max_vals = torch.tensor(
        [v if v is not None else max_t for v in batch_max_list],
        dtype=t_ratio.dtype,
    )
    mask = (t_ratio > min_vals) & (t_ratio < max_vals)

    print(f"    t_ratio:  {t_ratio.tolist()}")
    print(f"    min_t={min_t}, max_t={max_t}")
    print(f"    mask:     {mask.tolist()}")

    # t=0.1 -> 0.1 > 0.0 and 0.1 < 0.5 -> True
    # t=0.3 -> 0.3 > 0.0 and 0.3 < 0.5 -> True
    # t=0.6 -> 0.6 > 0.0 and 0.6 < 0.5 -> False (above max)
    # t=0.8 -> 0.8 > 0.0 and 0.8 < 0.5 -> False (above max)
    expected = [True, True, False, False]
    assert mask.tolist() == expected, f"Expected {expected}, got {mask.tolist()}"

    # Test with per-dataset overrides (sample 1 gets wider window)
    batch_min_list_override = [None, 0.0, None, None]
    batch_max_list_override = [None, 0.9, None, None]

    min_vals2 = torch.tensor(
        [v if v is not None else min_t for v in batch_min_list_override],
        dtype=t_ratio.dtype,
    )
    max_vals2 = torch.tensor(
        [v if v is not None else max_t for v in batch_max_list_override],
        dtype=t_ratio.dtype,
    )
    mask2 = (t_ratio > min_vals2) & (t_ratio < max_vals2)
    print(f"    per-ds override mask: {mask2.tolist()}")
    # Sample 1 (t=0.3) uses max_t=0.9 -> still True
    assert mask2[1] == True, "Sample 1 with max_t=0.9 should pass"

    # Test boundary: t=0.0 exactly on min_t=0.0 -> False (strict >)
    t_boundary = torch.tensor([0.0, 0.5])
    min_b = torch.tensor([0.0, 0.0])
    max_b = torch.tensor([0.5, 0.5])
    mask_b = (t_boundary > min_b) & (t_boundary < max_b)
    print(f"    boundary t={t_boundary.tolist()} -> mask={mask_b.tolist()}")
    assert mask_b.tolist() == [False, False], "Boundary values should be excluded (strict inequality)"


# ---------------------------------------------------------------------------
# 12. ref_validity_check
# ---------------------------------------------------------------------------

def test_ref_validity_check():
    """Simulate the reference validity check from SDTrainer."""
    batch_size = 4

    # Create features where sample 1 is all zeros (invalid)
    ref_features = {
        'level_1': torch.randn(batch_size, 256, 32, 32),
        'level_2': torch.randn(batch_size, 512, 16, 16),
        'level_3': torch.randn(batch_size, 512, 8, 8),
        'mid': torch.randn(batch_size, 512, 8, 8),
    }

    # Zero out sample index 1 across all levels
    for level in FEATURE_LEVELS:
        ref_features[level][1] = 0.0

    # Replicate validity check from SDTrainer (lines 2365-2368)
    ref_valid = torch.stack([
        ref_features[level].abs().sum(dim=(1, 2, 3)) > 0
        for level in FEATURE_LEVELS if level in ref_features
    ], dim=0).all(dim=0)  # (B,)

    print(f"    ref_valid: {ref_valid.tolist()}")
    assert ref_valid.shape == (batch_size,), f"Expected shape ({batch_size},), got {ref_valid.shape}"
    assert ref_valid[0] == True, "Sample 0 (non-zero) should be valid"
    assert ref_valid[1] == False, "Sample 1 (all-zero) should be invalid"
    assert ref_valid[2] == True, "Sample 2 (non-zero) should be valid"
    assert ref_valid[3] == True, "Sample 3 (non-zero) should be valid"

    # Also test: one level zero but others non-zero -> should be invalid
    ref_features2 = {
        'level_1': torch.randn(batch_size, 256, 32, 32),
        'level_2': torch.randn(batch_size, 512, 16, 16),
        'level_3': torch.randn(batch_size, 512, 8, 8),
        'mid': torch.randn(batch_size, 512, 8, 8),
    }
    # Zero only mid for sample 2
    ref_features2['mid'][2] = 0.0

    ref_valid2 = torch.stack([
        ref_features2[level].abs().sum(dim=(1, 2, 3)) > 0
        for level in FEATURE_LEVELS if level in ref_features2
    ], dim=0).all(dim=0)

    print(f"    partial zero ref_valid: {ref_valid2.tolist()}")
    assert ref_valid2[2] == False, "Sample 2 with zero mid should be invalid (all() across levels)"

    # Test combination with timestep mask
    va_noise_mask = torch.tensor([True, True, True, False])
    va_valid_mask = va_noise_mask & ref_valid
    print(f"    combined (noise & ref_valid): {va_valid_mask.tolist()}")
    assert va_valid_mask.tolist() == [True, False, True, False], (
        f"Expected [True, False, True, False], got {va_valid_mask.tolist()}"
    )


# ---------------------------------------------------------------------------
# 13. resolution_mismatch
# ---------------------------------------------------------------------------

def test_resolution_mismatch():
    """Encode ref at 256x256, pred at 512x512. Verify compute_loss handles via interpolation."""
    encoder = _get_shared_encoder()

    images = _get_bodytest_images(n=1)
    pil_img = Image.open(images[0]).convert('RGB')

    # Encode at 256x256
    pil_256 = pil_img.resize((256, 256))
    ref_features = encode_reference_features(encoder, pil_256, target_size=256)

    # Encode at 512x512
    pil_512 = pil_img.resize((512, 512))
    import torchvision.transforms.functional as TF
    img_512 = TF.to_tensor(pil_512).unsqueeze(0).cuda() * 2.0 - 1.0

    with torch.no_grad():
        _, pred_features = encoder.encode_with_features(img_512)

    # Print sizes to show mismatch
    for level in FEATURE_LEVELS:
        ref_s = list(ref_features[level].shape)
        pred_s = list(pred_features[level].shape)
        print(f"    {level}: ref={ref_s}, pred={pred_s}")
        assert ref_s[2:] != pred_s[2:], f"{level}: sizes should differ between 256 and 512 input"

    # compute_loss should handle it via interpolation
    ref_gpu = {k: v.float().cuda() for k, v in ref_features.items()}
    loss, per_level = VAEAnchorEncoder.compute_loss(pred_features, ref_gpu)

    print(f"    cross-resolution loss: {loss.item():.6f}")
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be Inf"
    assert loss.item() > 0, "Cross-resolution loss on same image should be > 0 (due to interpolation artifacts)"

    # Compare with same-resolution loss to verify it is reasonable
    pil_256b = pil_img.resize((256, 256))
    img_256 = TF.to_tensor(pil_256b).unsqueeze(0).cuda() * 2.0 - 1.0
    with torch.no_grad():
        _, pred_256 = encoder.encode_with_features(img_256)
    loss_same, _ = VAEAnchorEncoder.compute_loss(pred_256, ref_gpu)
    print(f"    same-resolution loss: {loss_same.item():.6f}")
    # Cross-res loss should be higher than same-res (which should be ~0)
    # But both should be finite and non-NaN
    assert loss.item() >= loss_same.item(), (
        f"Cross-res loss ({loss.item()}) should be >= same-res loss ({loss_same.item()})"
    )


# ---------------------------------------------------------------------------
# 14. real_image_perceptual_quality
# ---------------------------------------------------------------------------

def test_real_image_perceptual_quality():
    """Load 3+ bodytest images, compute pairwise losses, verify perceptual properties."""
    encoder = _get_shared_encoder()

    images = _get_bodytest_images(n=4)
    assert len(images) >= 3, f"Need at least 3 images, found {len(images)}"

    # Encode all images
    all_features = []
    for img_path in images:
        pil_img = Image.open(img_path).convert('RGB')
        feats = encode_reference_features(encoder, pil_img, target_size=256)
        # Move to GPU float32 for compute_loss
        feats_gpu = {k: v.float().cuda() for k, v in feats.items()}
        all_features.append(feats_gpu)
        print(f"    encoded: {os.path.basename(img_path)}")

    n = len(all_features)
    loss_matrix = torch.zeros(n, n)

    for i in range(n):
        for j in range(n):
            loss, _ = VAEAnchorEncoder.compute_loss(all_features[i], all_features[j])
            loss_matrix[i, j] = loss.item()

    # Print matrix
    print("    pairwise loss matrix:")
    names = [os.path.basename(p)[:20] for p in images]
    header = "    " + " " * 22 + "  ".join(f"{n:>8s}" for n in names)
    print(header)
    for i in range(n):
        row = f"    {names[i]:>20s}  " + "  ".join(f"{loss_matrix[i, j]:8.4f}" for j in range(n))
        print(row)

    # Self-comparison should be exactly 0
    for i in range(n):
        assert loss_matrix[i, i] == 0.0, (
            f"Self-comparison [{i}] should be 0, got {loss_matrix[i, i]:.8f}"
        )

    # Cross-image losses should all be > 0
    for i in range(n):
        for j in range(n):
            if i != j:
                assert loss_matrix[i, j] > 0, (
                    f"Cross-image loss [{i},{j}] should be > 0, got {loss_matrix[i, j]:.8f}"
                )

    # Cross-image losses should be significantly above 0
    cross_losses = []
    for i in range(n):
        for j in range(n):
            if i != j:
                cross_losses.append(loss_matrix[i, j].item())
    min_cross = min(cross_losses)
    max_cross = max(cross_losses)
    mean_cross = sum(cross_losses) / len(cross_losses)
    print(f"    cross-image loss: min={min_cross:.4f}, max={max_cross:.4f}, mean={mean_cross:.4f}")
    assert min_cross > 0.01, (
        f"Minimum cross-image loss should be > 0.01 (not trivially small), got {min_cross:.4f}"
    )

    # Verify loss is not symmetric (due to variance normalization using ref)
    # Actually, check: loss(A, B) and loss(B, A) may differ because ref_var differs
    asymmetries = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(loss_matrix[i, j].item() - loss_matrix[j, i].item())
            asymmetries.append(diff)
    if asymmetries:
        print(f"    asymmetry (|L(i,j)-L(j,i)|): max={max(asymmetries):.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 70)
    print("VAE Anchor Loss -- Comprehensive Validation Tests")
    print("=" * 70)
    print(f"CUDA available: {HAS_CUDA}")
    if HAS_CUDA:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"bodytest dir: {BODYTEST_DIR}")
    bodytest_exists = os.path.isdir(BODYTEST_DIR)
    print(f"bodytest exists: {bodytest_exists}")
    if bodytest_exists:
        imgs = _get_bodytest_images()
        print(f"bodytest images: {len(imgs)}")
    print()

    # --- Tests that don't need GPU ---
    print("--- CPU-only tests ---")
    _run_test("per_sample_loss_shape", test_per_sample_loss_shape)
    _run_test("per_dataset_weight_propagation", test_per_dataset_weight_propagation)
    _run_test("timestep_gating", test_timestep_gating)
    _run_test("ref_validity_check", test_ref_validity_check)
    _run_test("batch_collection", test_batch_collection)
    print()

    # --- Tests that need GPU + VAE model ---
    print("--- GPU tests (encoder, loss, caching, real images) ---")
    _run_test("auto_resolve_path", test_auto_resolve_path, requires_gpu=True)
    _run_test("encoder_loads_from_auto_path", test_encoder_loads_from_auto_path, requires_gpu=True)
    _run_test("loss_zero_identical", test_loss_zero_identical, requires_gpu=True)
    _run_test("loss_monotonic_with_noise", test_loss_monotonic_with_noise, requires_gpu=True)
    _run_test("gradient_flow", test_gradient_flow, requires_gpu=True)
    _run_test("resolution_mismatch", test_resolution_mismatch, requires_gpu=True)
    _run_test("cache_real_images", test_cache_real_images, requires_gpu=True)
    _run_test("cache_reload_matches", test_cache_reload_matches, requires_gpu=True)
    _run_test("real_image_perceptual_quality", test_real_image_perceptual_quality, requires_gpu=True)
    print()

    # --- Summary ---
    print("=" * 70)
    n_pass = sum(1 for _, s in _results if s == 'PASS')
    n_fail = sum(1 for _, s in _results if s == 'FAIL')
    n_skip = sum(1 for _, s in _results if s == 'SKIP')
    total = len(_results)
    print(f"TOTAL: {n_pass} passed, {n_fail} failed, {n_skip} skipped (out of {total})")

    if n_fail > 0:
        print("\nFailed tests:")
        for name, status in _results:
            if status == 'FAIL':
                print(f"  - {name}")
        sys.exit(1)
    else:
        print("\nAll tests passed.")

    # Cleanup shared encoder
    if _shared_encoder is not None:
        _shared_encoder.cleanup()
        del _shared_encoder
        torch.cuda.empty_cache()
