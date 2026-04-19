"""Phase 2 tests for subject-mask region loss weighting.

Exercises the loss-weighting path that lives in
``SDTrainer._build_subject_mask_weight`` and the body-restrict path in
``SDTrainer._build_body_restrict_mask``. The tests construct minimal batch +
config objects and call the helpers directly — this avoids having to spin up a
full diffusion model while still exercising the exact code the trainer runs.

Each test case follows the contract in the Phase 2 spec:

1. No-op when subject_mask is disabled: identical loss.
2. No-op when all weights are None: identical loss.
3. bg_w=0 reduces loss on a portrait.
4. body_w=2 boosts loss on a portrait.
5. Per-dataset override wins over global.
6. Weight-map shape / dtype / device match the underlying loss tensor.

Additionally verifies the body-restrict helper used by perceptual losses.

Run: python testing/test_subject_mask_loss.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.config_modules import SubjectMaskConfig


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "test_data" / "scarlett_full"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".avif")


class _FakeFileItem:
    """Minimal FileItemDTO stand-in for batch construction."""
    def __init__(self):
        self.path = None
        self.width = 512
        self.height = 512
        self.scale_to_width = 512
        self.scale_to_height = 512
        self.crop_x = 0
        self.crop_y = 0
        self.crop_width = 512
        self.crop_height = 512
        self.subject_mask = None
        self.body_mask = None
        self.clothing_mask = None
        self.background_loss_weight = None
        self.clothing_loss_weight = None
        self.body_loss_weight = None
        self.perceptual_restrict_to_body = None


class _FakeBatch:
    """Minimal batch that carries just what the helpers read."""
    def __init__(self, file_items, subject_masks, body_masks, clothing_masks,
                 bg_w_list=None, cl_w_list=None, bd_w_list=None,
                 restrict_list=None):
        self.file_items = file_items
        self.subject_masks = subject_masks
        self.body_masks = body_masks
        self.clothing_masks = clothing_masks
        bs = len(file_items)
        self.background_loss_weight_list = bg_w_list or [None] * bs
        self.clothing_loss_weight_list = cl_w_list or [None] * bs
        self.body_loss_weight_list = bd_w_list or [None] * bs
        self.perceptual_restrict_to_body_list = restrict_list or [None] * bs


class _FakeTrainer:
    """Stand-in for SDTrainer that only provides what the helpers need.

    We bind the actual helper methods onto this class so the code exercised is
    exactly what runs in training.
    """
    def __init__(self, subject_mask_config):
        self.subject_mask_config = subject_mask_config
        self.device_torch = torch.device('cpu')


# Attach the real helpers (unbound) to our fake so test code path == prod.
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer as _RealSDTrainer
_FakeTrainer._build_subject_mask_weight = _RealSDTrainer._build_subject_mask_weight
_FakeTrainer._build_body_restrict_mask = _RealSDTrainer._build_body_restrict_mask


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def _load_cached_masks(limit=3):
    """Load a handful of cached masks from scarlett_full.

    Requires test_subject_mask_cache.py to have populated the cache already.
    Returns (subject_masks, body_masks, clothing_masks) each (N, 1, 256, 256).
    """
    from safetensors.torch import load_file
    cache_dir = DATA_DIR / "_face_id_cache"
    mask_files = sorted(cache_dir.glob("*_subject_masks.safetensors"))
    if len(mask_files) < limit:
        raise SystemExit(
            f"Need >= {limit} cached mask files at {cache_dir}; found "
            f"{len(mask_files)}. Run testing/test_subject_mask_cache.py first."
        )
    subj, body, cloth = [], [], []
    for f in mask_files[:limit]:
        d = load_file(str(f))
        subj.append(((d['person'] > 127).to(torch.bool)).unsqueeze(0))
        body.append(((d['body'] > 127).to(torch.bool)).unsqueeze(0))
        cloth.append(((d['clothing'] > 127).to(torch.bool)).unsqueeze(0))
    return (torch.stack(subj, dim=0),
            torch.stack(body, dim=0),
            torch.stack(cloth, dim=0))


def _make_synthetic_masks(batch_size=2):
    """Create plausible synthetic masks for tests that don't need real data.

    Builds a 256x256 mask where the central 50% area is the person, and within
    the person the upper-third is body (head+shoulders) and lower two-thirds
    are clothing.
    """
    subj = torch.zeros(batch_size, 1, 256, 256, dtype=torch.bool)
    body = torch.zeros(batch_size, 1, 256, 256, dtype=torch.bool)
    cloth = torch.zeros(batch_size, 1, 256, 256, dtype=torch.bool)
    subj[:, :, 64:192, 64:192] = True
    body[:, :, 64:128, 64:192] = True
    cloth[:, :, 128:192, 64:192] = True
    return subj, body, cloth


def _simulate_loss(noise_pred, target, mask_multiplier, weight_map):
    """Replicate the SDTrainer loss composition pipeline precisely.

    loss = (noise_pred - target)^2
    loss = loss * mask_multiplier * weight_map  (if weight_map is not None)
    return loss.mean()
    """
    loss = (noise_pred - target) ** 2
    mm = mask_multiplier
    if weight_map is not None:
        mm = mm * weight_map
    loss = loss * mm
    return loss.mean()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_1_noop_when_disabled():
    """subject_mask disabled → weight map is None → identical loss."""
    smc = SubjectMaskConfig(enabled=False)
    trainer = _FakeTrainer(smc)
    subj, body, cloth = _make_synthetic_masks(2)
    batch = _FakeBatch(
        file_items=[_FakeFileItem(), _FakeFileItem()],
        subject_masks=subj, body_masks=body, clothing_masks=cloth,
    )
    noisy_latents_shape = (2, 4, 32, 32)
    w = trainer._build_subject_mask_weight(batch, noisy_latents_shape, dtype=torch.float32)
    assert w is None, f"Expected None when disabled, got tensor of shape {None if w is None else w.shape}"

    torch.manual_seed(0)
    noise_pred = torch.randn(2, 4, 32, 32)
    target = torch.randn(2, 4, 32, 32)
    mask_multiplier = torch.ones(2, 1, 1, 1)
    loss_disabled = _simulate_loss(noise_pred, target, mask_multiplier, None)

    # With enabled but no weights set, should also match exactly
    smc2 = SubjectMaskConfig(enabled=True)
    trainer2 = _FakeTrainer(smc2)
    w2 = trainer2._build_subject_mask_weight(batch, noisy_latents_shape, dtype=torch.float32)
    assert w2 is None, "Enabled with all-None weights should also return None"
    loss_enabled_no_weights = _simulate_loss(noise_pred, target, mask_multiplier, None)
    assert torch.allclose(loss_disabled, loss_enabled_no_weights), (
        f"Loss mismatch: disabled={loss_disabled.item()} vs enabled-no-weights={loss_enabled_no_weights.item()}"
    )
    print(f"  PASS test_1_noop_when_disabled    loss={loss_disabled.item():.8f}")
    return float(loss_disabled.item())


def test_2_noop_when_weights_none():
    """enabled=True, all *_loss_weight=None → still no weight map."""
    smc = SubjectMaskConfig(
        enabled=True,
        background_loss_weight=None,
        clothing_loss_weight=None,
        body_loss_weight=None,
    )
    trainer = _FakeTrainer(smc)
    subj, body, cloth = _make_synthetic_masks(2)
    batch = _FakeBatch(
        file_items=[_FakeFileItem(), _FakeFileItem()],
        subject_masks=subj, body_masks=body, clothing_masks=cloth,
    )
    w = trainer._build_subject_mask_weight(batch, (2, 4, 32, 32), dtype=torch.float32)
    assert w is None, f"Expected None when all weights are None; got {None if w is None else w.shape}"
    print("  PASS test_2_noop_when_weights_none")


def _make_mm_loss(weight_map, noise_pred, target, mask_multiplier):
    return _simulate_loss(noise_pred, target, mask_multiplier, weight_map)


def test_3_bg_zero_reduces_loss():
    """bg_w=0 zeroes out background pixels, strictly reduces loss."""
    smc = SubjectMaskConfig(enabled=True, background_loss_weight=0.0)
    trainer = _FakeTrainer(smc)
    subj, body, cloth = _make_synthetic_masks(2)
    batch = _FakeBatch(
        file_items=[_FakeFileItem(), _FakeFileItem()],
        subject_masks=subj, body_masks=body, clothing_masks=cloth,
    )
    w = trainer._build_subject_mask_weight(batch, (2, 4, 32, 32), dtype=torch.float32)
    assert w is not None, "Expected non-None weight map when bg_w=0"
    assert w.shape == (2, 4, 32, 32), f"shape: {w.shape}"

    torch.manual_seed(0)
    noise_pred = torch.randn(2, 4, 32, 32)
    target = torch.randn(2, 4, 32, 32)
    mask_multiplier = torch.ones(2, 1, 1, 1)

    loss_disabled = _simulate_loss(noise_pred, target, mask_multiplier, None)
    loss_bg0 = _simulate_loss(noise_pred, target, mask_multiplier, w)
    assert loss_bg0 < loss_disabled, (
        f"Expected bg=0 loss < disabled loss; got bg0={loss_bg0.item()} vs disabled={loss_disabled.item()}"
    )
    # Sanity: the weight map equals 1 inside person and 0 outside
    #   person covers 64:192 x 64:192 area in 256x256 → 8:24 in 32x32 after nearest-interp
    # Spot check: weight inside person is 1, outside is 0
    assert torch.all(w[:, :, 12:20, 12:20] == 1.0), "inside person, weight should be 1.0"
    assert torch.all(w[:, :, 0:5, 0:5] == 0.0), "outside person, weight should be 0.0"
    print(f"  PASS test_3_bg_zero_reduces_loss    disabled={loss_disabled.item():.8f} bg=0 loss={loss_bg0.item():.8f}")
    return float(loss_disabled.item()), float(loss_bg0.item())


def test_4_body_w_two_boosts_loss():
    """body_w=2 doubles loss inside body region, strictly boosts total."""
    smc = SubjectMaskConfig(enabled=True, body_loss_weight=2.0)
    trainer = _FakeTrainer(smc)
    subj, body, cloth = _make_synthetic_masks(2)
    batch = _FakeBatch(
        file_items=[_FakeFileItem(), _FakeFileItem()],
        subject_masks=subj, body_masks=body, clothing_masks=cloth,
    )
    w = trainer._build_subject_mask_weight(batch, (2, 4, 32, 32), dtype=torch.float32)
    assert w is not None
    assert w.shape == (2, 4, 32, 32), f"shape: {w.shape}"
    # Inside body: weight=2, outside body: weight=1
    assert torch.all(w[:, :, 10:14, 10:20] == 2.0), "inside body, weight should be 2.0"
    assert torch.all(w[:, :, 0:5, 0:5] == 1.0), "outside body, weight should be 1.0"

    torch.manual_seed(0)
    noise_pred = torch.randn(2, 4, 32, 32)
    target = torch.randn(2, 4, 32, 32)
    mask_multiplier = torch.ones(2, 1, 1, 1)

    loss_disabled = _simulate_loss(noise_pred, target, mask_multiplier, None)
    loss_bd2 = _simulate_loss(noise_pred, target, mask_multiplier, w)
    assert loss_bd2 > loss_disabled, (
        f"Expected body=2 loss > disabled loss; got body2={loss_bd2.item()} vs disabled={loss_disabled.item()}"
    )
    print(f"  PASS test_4_body_w_two_boosts_loss  disabled={loss_disabled.item():.8f} body=2 loss={loss_bd2.item():.8f}")
    return float(loss_disabled.item()), float(loss_bd2.item())


def test_5_per_dataset_override():
    """Per-dataset body_loss_weight overrides global; matches case 4 when override=2.0."""
    # Global says body_w=1.0, override says body_w=2.0 → override should win.
    smc = SubjectMaskConfig(enabled=True, body_loss_weight=1.0)
    trainer = _FakeTrainer(smc)
    subj, body, cloth = _make_synthetic_masks(2)
    batch = _FakeBatch(
        file_items=[_FakeFileItem(), _FakeFileItem()],
        subject_masks=subj, body_masks=body, clothing_masks=cloth,
        bd_w_list=[2.0, 2.0],
    )
    w_override = trainer._build_subject_mask_weight(batch, (2, 4, 32, 32), dtype=torch.float32)
    assert w_override is not None
    # inside body should be 2, outside should be 1
    assert torch.all(w_override[:, :, 10:14, 10:20] == 2.0)
    assert torch.all(w_override[:, :, 0:5, 0:5] == 1.0)

    # Compare to case 4 explicitly.
    smc4 = SubjectMaskConfig(enabled=True, body_loss_weight=2.0)
    trainer4 = _FakeTrainer(smc4)
    batch4 = _FakeBatch(
        file_items=[_FakeFileItem(), _FakeFileItem()],
        subject_masks=subj, body_masks=body, clothing_masks=cloth,
    )
    w_global2 = trainer4._build_subject_mask_weight(batch4, (2, 4, 32, 32), dtype=torch.float32)
    assert torch.allclose(w_override, w_global2), (
        "per-dataset override should produce identical mask to global=2.0"
    )
    print("  PASS test_5_per_dataset_override")


def test_6_shape_dtype_device():
    """Weight map matches the latent loss in shape, dtype and device."""
    smc = SubjectMaskConfig(enabled=True, background_loss_weight=0.5,
                            body_loss_weight=1.5, clothing_loss_weight=0.8)
    trainer = _FakeTrainer(smc)
    subj, body, cloth = _make_synthetic_masks(3)
    batch = _FakeBatch(
        file_items=[_FakeFileItem() for _ in range(3)],
        subject_masks=subj, body_masks=body, clothing_masks=cloth,
    )
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        shape = (3, 16, 48, 48)
        w = trainer._build_subject_mask_weight(batch, shape, dtype=dt)
        assert w is not None
        assert w.shape == shape, f"dtype={dt}: got {w.shape} expected {shape}"
        assert w.dtype == dt, f"dtype mismatch: got {w.dtype} expected {dt}"
        assert w.device == torch.device('cpu'), f"device mismatch: {w.device}"
    print("  PASS test_6_shape_dtype_device")


def test_7_real_masks_from_scarlett():
    """Real cached masks: bg_w=0 reduces loss vs disabled (portrait data)."""
    cache_dir = DATA_DIR / "_face_id_cache"
    if not cache_dir.exists() or not list(cache_dir.glob("*_subject_masks.safetensors")):
        print("  SKIP test_7_real_masks_from_scarlett (no cache)")
        return None, None
    subj, body, cloth = _load_cached_masks(3)
    trainer_disabled = _FakeTrainer(SubjectMaskConfig(enabled=True))
    batch = _FakeBatch(
        file_items=[_FakeFileItem(), _FakeFileItem(), _FakeFileItem()],
        subject_masks=subj, body_masks=body, clothing_masks=cloth,
    )
    w_disabled = trainer_disabled._build_subject_mask_weight(
        batch, (3, 4, 32, 32), dtype=torch.float32)
    assert w_disabled is None, "All-None weights should produce None map"

    trainer_bg0 = _FakeTrainer(SubjectMaskConfig(enabled=True, background_loss_weight=0.0))
    w_bg0 = trainer_bg0._build_subject_mask_weight(batch, (3, 4, 32, 32), dtype=torch.float32)
    assert w_bg0 is not None and w_bg0.shape == (3, 4, 32, 32)

    torch.manual_seed(42)
    noise_pred = torch.randn(3, 4, 32, 32)
    target = torch.randn(3, 4, 32, 32)
    mask_multiplier = torch.ones(3, 1, 1, 1)

    loss_disabled = _simulate_loss(noise_pred, target, mask_multiplier, None)
    loss_bg0 = _simulate_loss(noise_pred, target, mask_multiplier, w_bg0)
    assert loss_bg0 < loss_disabled, (
        f"Real portraits with bg=0 should reduce loss; got {loss_bg0.item()} vs {loss_disabled.item()}"
    )

    trainer_bd2 = _FakeTrainer(SubjectMaskConfig(enabled=True, body_loss_weight=2.0))
    w_bd2 = trainer_bd2._build_subject_mask_weight(batch, (3, 4, 32, 32), dtype=torch.float32)
    loss_bd2 = _simulate_loss(noise_pred, target, mask_multiplier, w_bd2)
    assert loss_bd2 > loss_disabled, (
        f"Real portraits with body=2 should boost loss; got {loss_bd2.item()} vs {loss_disabled.item()}"
    )
    print(f"  PASS test_7_real_masks_from_scarlett disabled={loss_disabled.item():.8f} "
          f"bg=0 {loss_bg0.item():.8f} body=2 {loss_bd2.item():.8f}")
    return float(loss_bg0.item()), float(loss_bd2.item())


def test_8_composition_with_face_suppression():
    """Subject mask weight composes multiplicatively with a prior mask_multiplier."""
    smc = SubjectMaskConfig(enabled=True, body_loss_weight=2.0)
    trainer = _FakeTrainer(smc)
    subj, body, cloth = _make_synthetic_masks(2)
    batch = _FakeBatch(
        file_items=[_FakeFileItem(), _FakeFileItem()],
        subject_masks=subj, body_masks=body, clothing_masks=cloth,
    )
    w = trainer._build_subject_mask_weight(batch, (2, 4, 32, 32), dtype=torch.float32)
    # Mock a face_suppression_like mask_multiplier (half-weight everywhere)
    mm = torch.full((2, 4, 32, 32), 0.5)
    torch.manual_seed(0)
    noise_pred = torch.randn(2, 4, 32, 32)
    target = torch.randn(2, 4, 32, 32)
    loss_only_fs = _simulate_loss(noise_pred, target, mm, None)
    loss_composed = _simulate_loss(noise_pred, target, mm, w)
    # Weight map has body=2 inside body, 1 elsewhere. mean>1 so combined > only_fs only
    # if body region has non-trivial per-pixel error. We can sanity check the
    # math: loss_composed should equal loss with (mm*w) applied.
    composed_mm = mm * w
    expected = ((noise_pred - target) ** 2 * composed_mm).mean()
    assert torch.allclose(loss_composed, expected, atol=1e-6), (
        f"composition mismatch: got {loss_composed.item()} expected {expected.item()}"
    )
    print("  PASS test_8_composition_with_face_suppression")


def test_9_body_restrict_helper():
    """_build_body_restrict_mask returns shape-matched body mask when enabled."""
    smc = SubjectMaskConfig(enabled=True, perceptual_restrict_to_body=True)
    trainer = _FakeTrainer(smc)
    _, body, _ = _make_synthetic_masks(2)
    batch = _FakeBatch(
        file_items=[_FakeFileItem(), _FakeFileItem()],
        subject_masks=None, body_masks=body, clothing_masks=None,
        restrict_list=[None, None],
    )
    # spatial_shape is (B, H, W) of the per-pixel loss
    m = trainer._build_body_restrict_mask(batch, (2, 64, 64))
    assert m is not None, "Expected non-None body mask when enabled"
    assert m.shape == (2, 64, 64), f"shape {m.shape}"
    # Body mask coverage in synthetic body region: rows 64:128 in 256 = 16:32 in 64
    assert torch.all(m[:, 18:30, 20:40] == 1.0)
    assert torch.all(m[:, 0:5, 0:5] == 0.0)

    # Disabled → None
    smc0 = SubjectMaskConfig(enabled=True, perceptual_restrict_to_body=False)
    trainer0 = _FakeTrainer(smc0)
    m0 = trainer0._build_body_restrict_mask(batch, (2, 64, 64))
    assert m0 is None, "Expected None when restrict flag is False"

    # Per-item override: item 0 opts in, item 1 doesn't → item 1 gets all-ones
    smc_off = SubjectMaskConfig(enabled=True, perceptual_restrict_to_body=False)
    trainer_off = _FakeTrainer(smc_off)
    batch2 = _FakeBatch(
        file_items=[_FakeFileItem(), _FakeFileItem()],
        subject_masks=None, body_masks=body, clothing_masks=None,
        restrict_list=[True, False],
    )
    m2 = trainer_off._build_body_restrict_mask(batch2, (2, 64, 64))
    assert m2 is not None
    # Item 1 should be all 1s (didn't opt in); item 0 should mirror body mask
    assert torch.all(m2[1] == 1.0), "Item 1 should be all-ones (no restrict)"
    assert m2[0, 0, 0].item() == 0.0, "Item 0 top-left should be 0 (outside body)"
    assert m2[0, 20, 30].item() == 1.0, "Item 0 body region should be 1"
    print("  PASS test_9_body_restrict_helper")


def test_10_dataset_config_fields():
    """DatasetConfig exposes the new override fields and defaults them to None."""
    from toolkit.config_modules import DatasetConfig
    d = DatasetConfig(folder_path='/tmp')
    assert d.background_loss_weight is None
    assert d.clothing_loss_weight is None
    assert d.body_loss_weight is None
    assert d.perceptual_restrict_to_body is None
    d2 = DatasetConfig(folder_path='/tmp', background_loss_weight=0.3,
                       clothing_loss_weight=0.5, body_loss_weight=2.0,
                       perceptual_restrict_to_body=True)
    assert d2.background_loss_weight == 0.3
    assert d2.clothing_loss_weight == 0.5
    assert d2.body_loss_weight == 2.0
    assert d2.perceptual_restrict_to_body is True
    print("  PASS test_10_dataset_config_fields")


def test_11_subject_mask_config_fields_preserved():
    """Ensure Phase-1 reserved SubjectMaskConfig field names are unchanged."""
    cfg = SubjectMaskConfig(
        enabled=True,
        background_loss_weight=0.1,
        clothing_loss_weight=0.2,
        body_loss_weight=0.3,
        perceptual_restrict_to_body=True,
    )
    assert cfg.background_loss_weight == 0.1
    assert cfg.clothing_loss_weight == 0.2
    assert cfg.body_loss_weight == 0.3
    assert cfg.perceptual_restrict_to_body is True
    # defaults for Phase-2 knobs
    default_cfg = SubjectMaskConfig()
    assert default_cfg.background_loss_weight is None
    assert default_cfg.clothing_loss_weight is None
    assert default_cfg.body_loss_weight is None
    assert default_cfg.perceptual_restrict_to_body is False
    print("  PASS test_11_subject_mask_config_fields_preserved")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Phase 2 subject_mask loss-weighting tests ===")
    loss1 = test_1_noop_when_disabled()
    test_2_noop_when_weights_none()
    d3, b3 = test_3_bg_zero_reduces_loss()
    d4, b4 = test_4_body_w_two_boosts_loss()
    test_5_per_dataset_override()
    test_6_shape_dtype_device()
    sc_bg, sc_bd = test_7_real_masks_from_scarlett()
    test_8_composition_with_face_suppression()
    test_9_body_restrict_helper()
    test_10_dataset_config_fields()
    test_11_subject_mask_config_fields_preserved()

    print("\n=== Monotonicity check (synthetic masks) ===")
    print(f"  loss disabled     = {d3:.8f}")
    print(f"  loss bg=0         = {b3:.8f}   (expect < disabled)")
    print(f"  loss body=2       = {b4:.8f}   (expect > disabled)")
    assert b3 < d3
    assert b4 > d4
    if sc_bg is not None:
        print("\n=== Monotonicity check (real masks) ===")
        print(f"  loss bg=0 (real)  = {sc_bg:.8f}")
        print(f"  loss body=2 (real)= {sc_bd:.8f}")

    print("\n=== ALL PHASE 2 TESTS PASSED ===")
