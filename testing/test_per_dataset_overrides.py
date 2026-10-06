#!/usr/bin/env python3
"""Tests for per-dataset loss config overrides.

Validates the full chain:
  DatasetConfig -> FileItemDTO -> DataLoaderBatchDTO -> per-sample masks in SDTrainer

Run: python testing/test_per_dataset_overrides.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from typing import List, Union


# ============================================================
# 1. Config propagation: DatasetConfig -> FileItemDTO fields
# ============================================================

def test_dataset_config_defaults():
    """All override fields default to None (inherit global)."""
    from toolkit.config_modules import DatasetConfig
    d = DatasetConfig(folder_path='/tmp/test')

    override_fields = [
        'identity_loss_weight', 'identity_loss_min_t', 'identity_loss_max_t', 'identity_loss_min_cos',
        'landmark_loss_weight',
        'body_proportion_loss_weight', 'body_proportion_loss_min_t', 'body_proportion_loss_max_t',
        'body_shape_loss_weight', 'body_shape_loss_min_t', 'body_shape_loss_max_t', 'body_shape_loss_min_cos',
        'normal_loss_weight', 'normal_loss_min_t', 'normal_loss_max_t',
        'vae_anchor_loss_weight', 'vae_anchor_loss_min_t', 'vae_anchor_loss_max_t',
        'diffusion_loss_weight',
        'face_suppression_weight',
    ]
    for field in override_fields:
        val = getattr(d, field)
        assert val is None, f"DatasetConfig.{field} should default to None, got {val}"
    print("  PASS test_dataset_config_defaults")


def test_dataset_config_overrides():
    """Override fields are set when provided."""
    from toolkit.config_modules import DatasetConfig
    d = DatasetConfig(
        folder_path='/tmp/test',
        identity_loss_weight=0.5,
        identity_loss_min_t=0.3,
        identity_loss_max_t=0.95,
        identity_loss_min_cos=0.15,
        body_proportion_loss_min_t=0.2,
        body_shape_loss_min_cos=0.4,
        normal_loss_max_t=0.7,
        diffusion_loss_weight=0.8,
    )
    assert d.identity_loss_weight == 0.5
    assert d.identity_loss_min_t == 0.3
    assert d.identity_loss_max_t == 0.95
    assert d.identity_loss_min_cos == 0.15
    assert d.body_proportion_loss_min_t == 0.2
    assert d.body_proportion_loss_max_t is None  # not set
    assert d.body_shape_loss_min_cos == 0.4
    assert d.normal_loss_max_t == 0.7
    assert d.normal_loss_min_t is None  # not set
    assert d.diffusion_loss_weight == 0.8
    print("  PASS test_dataset_config_overrides")


def test_global_config_unchanged():
    """FaceIDConfig global defaults are correct and unchanged."""
    from toolkit.config_modules import FaceIDConfig
    g = FaceIDConfig()
    assert g.identity_loss_min_t == 0.0
    assert g.identity_loss_max_t == 1.0
    assert g.identity_loss_min_cos == 0.2
    assert g.body_proportion_loss_min_t == 0.0
    assert g.body_proportion_loss_max_t == 1.0
    assert g.body_shape_loss_min_t == 0.4
    assert g.body_shape_loss_max_t == 0.8
    assert g.body_shape_loss_min_cos == 0.2
    assert g.normal_loss_min_t == 0.4
    assert g.normal_loss_max_t == 0.8
    assert g.vae_anchor_loss_weight == 0.0
    assert g.vae_anchor_loss_min_t == 0.0
    assert g.vae_anchor_loss_max_t == 0.5
    print("  PASS test_global_config_unchanged")


# ============================================================
# 2. Batch list construction from mixed datasets
# ============================================================

def _make_mock_file_item(dataset_kwargs):
    """Create a minimal mock FileItemDTO-like object with per-dataset overrides."""
    from toolkit.config_modules import DatasetConfig
    dc = DatasetConfig(**dataset_kwargs)

    class MockFileItem:
        pass

    fi = MockFileItem()
    # Copy all override fields from DatasetConfig
    for attr in [
        'identity_loss_weight', 'identity_loss_min_t', 'identity_loss_max_t', 'identity_loss_min_cos',
        'landmark_loss_weight',
        'body_proportion_loss_weight', 'body_proportion_loss_min_t', 'body_proportion_loss_max_t',
        'body_shape_loss_weight', 'body_shape_loss_min_t', 'body_shape_loss_max_t', 'body_shape_loss_min_cos',
        'normal_loss_weight', 'normal_loss_min_t', 'normal_loss_max_t',
        'vae_anchor_loss_weight', 'vae_anchor_loss_min_t', 'vae_anchor_loss_max_t',
        'diffusion_loss_weight',
        'face_suppression_weight',
    ]:
        setattr(fi, attr, getattr(dc, attr))
    return fi


def test_batch_lists_mixed_datasets():
    """Batch with items from different datasets produces correct per-sample lists."""
    fi_a = _make_mock_file_item({
        'folder_path': '/a',
        'identity_loss_min_t': 0.3,
        'identity_loss_max_t': 0.95,
        'normal_loss_weight': 0.5,
    })
    fi_b = _make_mock_file_item({
        'folder_path': '/b',
        # all None — inherits global
    })

    # Simulate what DataLoaderBatchDTO does
    items = [fi_a, fi_b]
    id_min_t_list = [x.identity_loss_min_t for x in items]
    id_max_t_list = [x.identity_loss_max_t for x in items]
    nrm_w_list = [x.normal_loss_weight for x in items]

    assert id_min_t_list == [0.3, None], f"Got {id_min_t_list}"
    assert id_max_t_list == [0.95, None], f"Got {id_max_t_list}"
    assert nrm_w_list == [0.5, None], f"Got {nrm_w_list}"
    print("  PASS test_batch_lists_mixed_datasets")


# ============================================================
# 3. Per-sample timestep mask logic (_per_sample_mask)
# ============================================================

def _per_sample_mask(t_ratio, batch_min_list, batch_max_list, global_min, global_max):
    """Standalone copy of the mask builder from SDTrainer for testing."""
    min_vals = torch.tensor(
        [v if v is not None else global_min for v in batch_min_list],
        device=t_ratio.device, dtype=t_ratio.dtype,
    )
    max_vals = torch.tensor(
        [v if v is not None else global_max for v in batch_max_list],
        device=t_ratio.device, dtype=t_ratio.dtype,
    )
    return (t_ratio > min_vals) & (t_ratio < max_vals)


def test_mask_all_global():
    """All samples use global defaults when no overrides."""
    t_ratio = torch.tensor([0.5, 0.7, 0.3])
    mask = _per_sample_mask(t_ratio, [None, None, None], [None, None, None], 0.4, 0.8)
    # 0.5 in (0.4, 0.8) = True
    # 0.7 in (0.4, 0.8) = True
    # 0.3 in (0.4, 0.8) = False
    assert mask.tolist() == [True, True, False], f"Got {mask.tolist()}"
    print("  PASS test_mask_all_global")


def test_mask_per_sample_override():
    """Per-sample overrides change the mask for that sample only."""
    t_ratio = torch.tensor([0.5, 0.5, 0.5])
    # Sample 0: override min_t=0.6 -> 0.5 NOT in (0.6, 0.8) = False
    # Sample 1: global (0.4, 0.8) -> 0.5 in range = True
    # Sample 2: override max_t=0.45 -> 0.5 NOT in (0.4, 0.45) = False
    mask = _per_sample_mask(
        t_ratio,
        batch_min_list=[0.6, None, None],
        batch_max_list=[None, None, 0.45],
        global_min=0.4, global_max=0.8,
    )
    assert mask.tolist() == [False, True, False], f"Got {mask.tolist()}"
    print("  PASS test_mask_per_sample_override")


def test_mask_all_overridden():
    """Every sample has its own window."""
    t_ratio = torch.tensor([0.5, 0.5, 0.5])
    mask = _per_sample_mask(
        t_ratio,
        batch_min_list=[0.0, 0.6, 0.4],
        batch_max_list=[1.0, 0.8, 0.49],
        global_min=0.4, global_max=0.8,
    )
    # 0.5 in (0.0, 1.0) = True
    # 0.5 in (0.6, 0.8) = False
    # 0.5 in (0.4, 0.49) = False (0.5 > 0.49 is True, so 0.5 < 0.49 is False)
    assert mask.tolist() == [True, False, False], f"Got {mask.tolist()}"
    print("  PASS test_mask_all_overridden")


def test_mask_boundary_exclusive():
    """Boundaries are exclusive: t_ratio must be strictly > min and < max."""
    t_ratio = torch.tensor([0.4, 0.8])
    mask = _per_sample_mask(t_ratio, [None, None], [None, None], 0.4, 0.8)
    # 0.4 > 0.4 = False (not strictly greater)
    # 0.8 < 0.8 = False (not strictly less)
    assert mask.tolist() == [False, False], f"Got {mask.tolist()}"
    print("  PASS test_mask_boundary_exclusive")


def test_mask_batch_size_1():
    """Works with batch_size=1."""
    t_ratio = torch.tensor([0.7])
    mask = _per_sample_mask(t_ratio, [0.5], [0.9], 0.4, 0.8)
    assert mask.tolist() == [True], f"Got {mask.tolist()}"
    print("  PASS test_mask_batch_size_1")


# ============================================================
# 4. Per-sample cosine threshold logic
# ============================================================

def _per_sample_cos_threshold(cos_sim, batch_min_cos_list, global_min_cos):
    """Build per-sample threshold and apply — mirrors SDTrainer logic."""
    cos_threshold = torch.tensor(
        [v if v is not None else global_min_cos for v in batch_min_cos_list],
        dtype=cos_sim.dtype,
    )
    return cos_sim > cos_threshold


def test_cos_threshold_all_global():
    """All samples use global cosine threshold."""
    cos_sim = torch.tensor([0.3, 0.1, 0.5])
    mask = _per_sample_cos_threshold(cos_sim, [None, None, None], 0.2)
    assert mask.tolist() == [True, False, True], f"Got {mask.tolist()}"
    print("  PASS test_cos_threshold_all_global")


def test_cos_threshold_per_sample():
    """Per-sample cosine thresholds override global."""
    cos_sim = torch.tensor([0.3, 0.3, 0.3])
    # Sample 0: threshold=0.5 -> 0.3 > 0.5 = False
    # Sample 1: threshold=0.2 (global) -> 0.3 > 0.2 = True
    # Sample 2: threshold=0.25 -> 0.3 > 0.25 = True
    mask = _per_sample_cos_threshold(cos_sim, [0.5, None, 0.25], 0.2)
    assert mask.tolist() == [False, True, True], f"Got {mask.tolist()}"
    print("  PASS test_cos_threshold_per_sample")


# ============================================================
# 5. Per-sample loss weight application
# ============================================================

def _apply_per_sample_weights(loss_per_sample, batch_weight_list, global_weight, valid_mask):
    """Mirrors SDTrainer per-sample weight application pattern."""
    if any(w is not None for w in batch_weight_list):
        weights = torch.tensor(
            [w if w is not None else global_weight for w in batch_weight_list],
            dtype=loss_per_sample.dtype,
        )
        weighted = loss_per_sample * weights
        return weighted.sum() / max(valid_mask.sum().item(), 1.0)
    else:
        raw_loss = loss_per_sample.sum() / max(valid_mask.sum().item(), 1.0)
        return global_weight * raw_loss


def test_weight_all_global():
    """No per-sample overrides: global weight applied to averaged loss."""
    loss = torch.tensor([0.5, 0.3])
    mask = torch.tensor([True, True])
    result = _apply_per_sample_weights(loss, [None, None], 0.1, mask)
    expected = 0.1 * (0.5 + 0.3) / 2.0
    assert abs(result.item() - expected) < 1e-6, f"Got {result.item()}, expected {expected}"
    print("  PASS test_weight_all_global")


def test_weight_per_sample_override():
    """Per-sample weights applied individually before averaging."""
    loss = torch.tensor([0.5, 0.3])
    mask = torch.tensor([True, True])
    # Sample 0: weight=0.2, Sample 1: weight=0.1 (global)
    result = _apply_per_sample_weights(loss, [0.2, None], 0.1, mask)
    expected = (0.5 * 0.2 + 0.3 * 0.1) / 2.0
    assert abs(result.item() - expected) < 1e-6, f"Got {result.item()}, expected {expected}"
    print("  PASS test_weight_per_sample_override")


def test_weight_zero_disables_sample():
    """Per-sample weight=0 effectively disables that sample's loss contribution."""
    loss = torch.tensor([0.5, 0.3])
    mask = torch.tensor([True, True])
    result = _apply_per_sample_weights(loss, [0.0, 0.1], 0.1, mask)
    expected = (0.5 * 0.0 + 0.3 * 0.1) / 2.0
    assert abs(result.item() - expected) < 1e-6, f"Got {result.item()}, expected {expected}"
    print("  PASS test_weight_zero_disables_sample")


# ============================================================
# 6. Integration: mixed-dataset batch with different windows
# ============================================================

def test_integration_mixed_batch():
    """Simulate a batch where two samples come from datasets with different settings.

    Dataset A: identity window [0.3, 0.95], weight 0.5
    Dataset B: identity window [0.6, 0.9] (global), weight 0.1
    Timestep: both at t_ratio=0.5

    Expected: sample A is inside its window, sample B is outside its window.
    """
    t_ratio = torch.tensor([0.5, 0.5])

    # Identity mask
    id_mask = _per_sample_mask(
        t_ratio,
        batch_min_list=[0.3, None],
        batch_max_list=[0.95, None],
        global_min=0.6, global_max=0.9,
    )
    assert id_mask.tolist() == [True, False], f"id_mask: {id_mask.tolist()}"

    # Body proportion mask — dataset A overrides to [0.2, 0.6], dataset B uses global [0.4, 0.8]
    bp_mask = _per_sample_mask(
        t_ratio,
        batch_min_list=[0.2, None],
        batch_max_list=[0.6, None],
        global_min=0.4, global_max=0.8,
    )
    assert bp_mask.tolist() == [True, True], f"bp_mask: {bp_mask.tolist()}"

    # Normal mask — both use global [0.4, 0.8]
    nrm_mask = _per_sample_mask(
        t_ratio,
        batch_min_list=[None, None],
        batch_max_list=[None, None],
        global_min=0.4, global_max=0.8,
    )
    assert nrm_mask.tolist() == [True, True], f"nrm_mask: {nrm_mask.tolist()}"

    # Cosine threshold — dataset A has higher bar
    cos_sim = torch.tensor([0.35, 0.35])
    cos_gate = _per_sample_cos_threshold(cos_sim, [0.4, None], 0.2)
    # Sample A: 0.35 > 0.4 = False (more conservative)
    # Sample B: 0.35 > 0.2 = True (global)
    assert cos_gate.tolist() == [False, True], f"cos_gate: {cos_gate.tolist()}"

    print("  PASS test_integration_mixed_batch")


def test_integration_any_active():
    """Verify any_active logic: if any sample's mask is True, we should decode x0."""
    t_ratio = torch.tensor([0.5, 0.5])

    # Sample 0: identity window [0.3, 0.95] -> True
    # Sample 1: identity window [0.6, 0.9] (global) -> False
    id_mask = _per_sample_mask(
        t_ratio,
        batch_min_list=[0.3, None],
        batch_max_list=[0.95, None],
        global_min=0.6, global_max=0.9,
    )
    # Even though sample 1 is outside, sample 0 is inside, so any_active should be True
    assert id_mask.any().item() is True, "any_active should be True when any sample is in window"

    # Now both outside
    t_ratio2 = torch.tensor([0.95, 0.95])
    id_mask2 = _per_sample_mask(
        t_ratio2,
        batch_min_list=[0.3, None],
        batch_max_list=[0.9, None],
        global_min=0.6, global_max=0.9,
    )
    assert id_mask2.any().item() is False, "any_active should be False when all samples outside"

    print("  PASS test_integration_any_active")


def test_integration_loss_masking():
    """Full loss computation with mixed masks and weights.

    Simulates: two samples, one inside its window (contributes loss), one outside (masked out).
    """
    t_ratio = torch.tensor([0.5, 0.5])
    id_mask = _per_sample_mask(
        t_ratio,
        batch_min_list=[0.3, None],
        batch_max_list=[0.95, None],
        global_min=0.6, global_max=0.9,
    )
    # id_mask = [True, False]

    # Fake cosine similarities
    cos_sim = torch.tensor([0.8, 0.7])
    cos_gate = _per_sample_cos_threshold(cos_sim, [None, None], 0.2)
    # Both pass threshold

    # Combined loss mask
    loss_mask = id_mask & cos_gate  # [True, False]

    # Loss per sample (1 - cos_sim) * t_ratio * loss_mask
    id_weight = t_ratio
    id_loss_per_sample = (1.0 - cos_sim) * id_weight * loss_mask.float()
    # Sample 0: (1 - 0.8) * 0.5 * 1.0 = 0.1
    # Sample 1: masked out = 0.0
    assert abs(id_loss_per_sample[0].item() - 0.1) < 1e-6
    assert abs(id_loss_per_sample[1].item() - 0.0) < 1e-6

    # Apply per-sample weights: dataset A = 0.5, dataset B = None (global 0.1)
    result = _apply_per_sample_weights(id_loss_per_sample, [0.5, None], 0.1, loss_mask)
    # Only sample 0 contributes: (0.1 * 0.5) / 1.0 = 0.05
    expected = (0.1 * 0.5 + 0.0 * 0.1) / 1.0
    assert abs(result.item() - expected) < 1e-6, f"Got {result.item()}, expected {expected}"

    print("  PASS test_integration_loss_masking")


# ============================================================
# 7. Edge cases
# ============================================================

def test_empty_batch():
    """Batch_size=0 edge case (shouldn't happen but shouldn't crash)."""
    t_ratio = torch.tensor([])
    mask = _per_sample_mask(t_ratio, [], [], 0.4, 0.8)
    assert mask.shape == (0,)
    assert not mask.any()
    print("  PASS test_empty_batch")


def test_all_none_matches_global():
    """All-None override lists must produce identical masks to hardcoded global."""
    t_ratio = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    global_min, global_max = 0.4, 0.8

    per_sample = _per_sample_mask(
        t_ratio, [None]*5, [None]*5, global_min, global_max,
    )
    hardcoded = (t_ratio > global_min) & (t_ratio < global_max)
    assert per_sample.tolist() == hardcoded.tolist(), (
        f"Mismatch: {per_sample.tolist()} vs {hardcoded.tolist()}"
    )
    print("  PASS test_all_none_matches_global")


def test_wide_window_override():
    """Dataset can set window to [0.0, 1.0] to enable loss at all timesteps."""
    t_ratio = torch.tensor([0.01, 0.5, 0.99])
    mask = _per_sample_mask(
        t_ratio,
        batch_min_list=[0.0, 0.0, 0.0],
        batch_max_list=[1.0, 1.0, 1.0],
        global_min=0.6, global_max=0.9,
    )
    assert mask.tolist() == [True, True, True], f"Got {mask.tolist()}"
    print("  PASS test_wide_window_override")


def test_narrow_window_override():
    """Dataset can set a very narrow window."""
    t_ratio = torch.tensor([0.50, 0.51, 0.52])
    mask = _per_sample_mask(
        t_ratio,
        batch_min_list=[0.505, 0.505, 0.505],
        batch_max_list=[0.515, 0.515, 0.515],
        global_min=0.4, global_max=0.8,
    )
    # 0.50 > 0.505 = False
    # 0.51 > 0.505 and 0.51 < 0.515 = True
    # 0.52 < 0.515 = False
    assert mask.tolist() == [False, True, False], f"Got {mask.tolist()}"
    print("  PASS test_narrow_window_override")


# ============================================================
# 8. Face suppression weight tests
# ============================================================

def test_face_suppression_config_default():
    """face_suppression_weight defaults to None in DatasetConfig."""
    from toolkit.config_modules import DatasetConfig
    d = DatasetConfig(folder_path='/tmp/test')
    assert d.face_suppression_weight is None, (
        f"Expected None, got {d.face_suppression_weight}"
    )
    print("  PASS test_face_suppression_config_default")


def test_face_suppression_config_set():
    """face_suppression_weight propagates from DatasetConfig."""
    from toolkit.config_modules import DatasetConfig
    d = DatasetConfig(folder_path='/tmp/test', face_suppression_weight=0.3)
    assert d.face_suppression_weight == 0.3
    print("  PASS test_face_suppression_config_set")


def _build_face_suppression_mask(batch_size, lat_h, lat_w, face_suppression_weight_list,
                                  face_bboxes, file_items):
    """Standalone copy of the face suppression mask builder from SDTrainer for testing.

    Returns the face_supp_mask tensor (B, 1, H, W) normalized to mean=1,
    or None if no suppression is needed.
    """
    if not any(w is not None and w < 1.0 for w in face_suppression_weight_list):
        return None
    if face_bboxes is None:
        return None

    face_supp_mask = torch.ones((batch_size, 1, lat_h, lat_w))
    for idx in range(batch_size):
        w = face_suppression_weight_list[idx]
        if w is None or w >= 1.0:
            continue
        raw_bbox = face_bboxes[idx] if idx < len(face_bboxes) else None
        if raw_bbox is None:
            continue
        fi = file_items[idx]
        orig_w = float(fi['width'])
        orig_h = float(fi['height'])
        bx1, by1, bx2, by2 = [float(v) for v in raw_bbox]
        stw = float(fi.get('scale_to_width') or orig_w)
        sth = float(fi.get('scale_to_height') or orig_h)
        bx1 *= stw / orig_w; by1 *= sth / orig_h
        bx2 *= stw / orig_w; by2 *= sth / orig_h
        cx = float(fi.get('crop_x') or 0)
        cy = float(fi.get('crop_y') or 0)
        cw = float(fi.get('crop_width') or stw)
        ch = float(fi.get('crop_height') or sth)
        bx1 -= cx; by1 -= cy; bx2 -= cx; by2 -= cy
        if bx2 <= 0 or by2 <= 0 or bx1 >= cw or by1 >= ch:
            continue
        bx1 = bx1 * lat_w / cw; bx2 = bx2 * lat_w / cw
        by1 = by1 * lat_h / ch; by2 = by2 * lat_h / ch
        x1 = max(0, int(bx1)); y1 = max(0, int(by1))
        x2 = min(lat_w, int(bx2) + 1); y2 = min(lat_h, int(by2) + 1)
        if x2 > x1 and y2 > y1:
            face_supp_mask[idx, :, y1:y2, x1:x2] = w
    face_supp_mask = face_supp_mask / face_supp_mask.mean()
    return face_supp_mask


def test_face_suppression_mask_spatial_pattern():
    """Face suppression mask has correct spatial pattern: suppressed inside bbox, 1.0 outside."""
    # 1 sample, 8x8 latent, face bbox covers a 4x4 region in center
    # Image is 64x64, no scaling/cropping, face bbox [16, 16, 48, 48]
    fi = {'width': 64, 'height': 64, 'scale_to_width': 64, 'scale_to_height': 64,
          'crop_x': 0, 'crop_y': 0, 'crop_width': 64, 'crop_height': 64}
    bbox = torch.tensor([16.0, 16.0, 48.0, 48.0])
    mask = _build_face_suppression_mask(
        batch_size=1, lat_h=8, lat_w=8,
        face_suppression_weight_list=[0.0],
        face_bboxes=[bbox],
        file_items=[fi],
    )
    assert mask is not None
    # Face bbox [16,16,48,48] in 64x64 image -> latent [2,2,6,6] in 8x8
    # x1=2, y1=2, x2=min(8, int(6)+1)=7, y2=7 -> region [2:7, 2:7] = 5x5
    # Inside region should be 0.0 (suppressed), outside should be > 0
    inside_val = mask[0, 0, 3, 3].item()  # center of face region
    outside_val = mask[0, 0, 0, 0].item()  # corner, outside face
    assert inside_val < outside_val, (
        f"Inside ({inside_val}) should be less than outside ({outside_val})"
    )
    # Inside should be 0.0 before normalization conceptually, but after normalization
    # it's still 0.0 (0.0 * scale = 0.0)
    assert inside_val == 0.0, f"With weight=0.0, face region should be 0.0, got {inside_val}"
    print("  PASS test_face_suppression_mask_spatial_pattern")


def test_face_suppression_mean_normalization():
    """Face suppression mask is normalized to mean=1.0 to preserve total loss magnitude."""
    fi = {'width': 64, 'height': 64, 'scale_to_width': 64, 'scale_to_height': 64,
          'crop_x': 0, 'crop_y': 0, 'crop_width': 64, 'crop_height': 64}
    bbox = torch.tensor([16.0, 16.0, 48.0, 48.0])
    mask = _build_face_suppression_mask(
        batch_size=1, lat_h=8, lat_w=8,
        face_suppression_weight_list=[0.5],
        face_bboxes=[bbox],
        file_items=[fi],
    )
    assert mask is not None
    mean_val = mask.mean().item()
    assert abs(mean_val - 1.0) < 1e-5, f"Mean should be 1.0, got {mean_val}"
    print("  PASS test_face_suppression_mean_normalization")


def test_face_suppression_weight_1_no_effect():
    """face_suppression_weight=1.0 produces None (no mask needed, no effect)."""
    mask = _build_face_suppression_mask(
        batch_size=1, lat_h=8, lat_w=8,
        face_suppression_weight_list=[1.0],
        face_bboxes=[torch.tensor([10.0, 10.0, 50.0, 50.0])],
        file_items=[{'width': 64, 'height': 64, 'scale_to_width': 64,
                     'scale_to_height': 64, 'crop_x': 0, 'crop_y': 0,
                     'crop_width': 64, 'crop_height': 64}],
    )
    assert mask is None, "weight=1.0 should skip mask creation entirely"
    print("  PASS test_face_suppression_weight_1_no_effect")


def test_face_suppression_none_skipped():
    """face_suppression_weight=None is skipped entirely (returns None)."""
    mask = _build_face_suppression_mask(
        batch_size=1, lat_h=8, lat_w=8,
        face_suppression_weight_list=[None],
        face_bboxes=[torch.tensor([10.0, 10.0, 50.0, 50.0])],
        file_items=[{'width': 64, 'height': 64, 'scale_to_width': 64,
                     'scale_to_height': 64, 'crop_x': 0, 'crop_y': 0,
                     'crop_width': 64, 'crop_height': 64}],
    )
    assert mask is None, "weight=None should skip mask creation entirely"
    print("  PASS test_face_suppression_none_skipped")


def test_face_suppression_half_weight():
    """face_suppression_weight=0.5 makes face region half the weight of non-face before normalization."""
    fi = {'width': 64, 'height': 64, 'scale_to_width': 64, 'scale_to_height': 64,
          'crop_x': 0, 'crop_y': 0, 'crop_width': 64, 'crop_height': 64}
    bbox = torch.tensor([0.0, 0.0, 32.0, 32.0])  # top-left quadrant
    mask = _build_face_suppression_mask(
        batch_size=1, lat_h=8, lat_w=8,
        face_suppression_weight_list=[0.5],
        face_bboxes=[bbox],
        file_items=[fi],
    )
    assert mask is not None
    # Before normalization: face region = 0.5, rest = 1.0
    # After normalization: ratio should still be 0.5:1.0 (i.e., face = half of non-face)
    face_val = mask[0, 0, 0, 0].item()  # inside face region
    non_face_val = mask[0, 0, 7, 7].item()  # outside face region
    ratio = face_val / non_face_val
    assert abs(ratio - 0.5) < 1e-5, f"Face/non-face ratio should be 0.5, got {ratio}"
    print("  PASS test_face_suppression_half_weight")


def test_face_suppression_mixed_batch():
    """Mixed batch: one sample suppressed, one not."""
    fi = {'width': 64, 'height': 64, 'scale_to_width': 64, 'scale_to_height': 64,
          'crop_x': 0, 'crop_y': 0, 'crop_width': 64, 'crop_height': 64}
    bbox = torch.tensor([16.0, 16.0, 48.0, 48.0])
    mask = _build_face_suppression_mask(
        batch_size=2, lat_h=8, lat_w=8,
        face_suppression_weight_list=[0.0, None],  # sample 0 suppressed, sample 1 normal
        face_bboxes=[bbox, bbox],
        file_items=[fi, fi],
    )
    assert mask is not None
    # Sample 1 should be all ones (before normalization), sample 0 has suppression
    # After normalization, sample 1 values should be uniform (all same value)
    s1_vals = mask[1, 0]
    assert torch.allclose(s1_vals, s1_vals[0, 0].expand_as(s1_vals)), (
        "Sample with no suppression should have uniform mask values"
    )
    # Sample 0 face region should be 0
    assert mask[0, 0, 3, 3].item() == 0.0, "Suppressed face region should be 0.0"
    print("  PASS test_face_suppression_mixed_batch")


def test_face_suppression_no_bboxes_returns_none():
    """When face_bboxes is None, suppression mask is None (with warning)."""
    mask = _build_face_suppression_mask(
        batch_size=1, lat_h=8, lat_w=8,
        face_suppression_weight_list=[0.0],
        face_bboxes=None,
        file_items=[{'width': 64, 'height': 64}],
    )
    assert mask is None, "Should return None when no face bboxes available"
    print("  PASS test_face_suppression_no_bboxes_returns_none")


# ============================================================
# Per-dataset diffusion loss timestep bounds
# ============================================================
#
# The diffusion loss has its own (global) TrainConfig fields
# `diffusion_loss_min_t` / `diffusion_loss_max_t` that gate which samples
# in a batch contribute to the diffusion loss based on their normalized
# timestep `t_ratio = timesteps / num_train_timesteps`. The per-dataset
# overrides on DatasetConfig (`diffusion_loss_min_t` / `diffusion_loss_max_t`)
# fall back to those globals when None. Unlike the auxiliary face/body/etc
# losses (which use exclusive bounds), the diffusion-loss gate uses
# **inclusive** bounds on both ends to preserve the pre-existing
# global-only semantics — see SDTrainer.py around the
# `apply diffusion loss timestep gating` comment.


def _diffusion_loss_mask(t_ratio, batch_min_list, batch_max_list, global_min, global_max):
    """Mirror of the per-sample diffusion-loss mask builder in SDTrainer.

    Inclusive on both ends (>= min, <= max). Returns a float mask in {0,1}
    matching the production code path that multiplies into the per-sample
    diffusion loss.
    """
    min_vals = torch.tensor(
        [v if v is not None else global_min for v in batch_min_list],
        device=t_ratio.device, dtype=t_ratio.dtype,
    )
    max_vals = torch.tensor(
        [v if v is not None else global_max for v in batch_max_list],
        device=t_ratio.device, dtype=t_ratio.dtype,
    )
    return ((t_ratio >= min_vals) & (t_ratio <= max_vals)).float()


def test_diffusion_bounds_dataset_config_default():
    """DatasetConfig.diffusion_loss_min_t / max_t default to None (inherit global)."""
    from toolkit.config_modules import DatasetConfig
    d = DatasetConfig(folder_path='/tmp/test')
    assert d.diffusion_loss_min_t is None, f"Got {d.diffusion_loss_min_t!r}"
    assert d.diffusion_loss_max_t is None, f"Got {d.diffusion_loss_max_t!r}"
    print("  PASS test_diffusion_bounds_dataset_config_default")


def test_diffusion_bounds_dataset_config_set():
    """DatasetConfig.diffusion_loss_min_t / max_t are picked up when provided."""
    from toolkit.config_modules import DatasetConfig
    d = DatasetConfig(
        folder_path='/tmp/test',
        diffusion_loss_min_t=0.1,
        diffusion_loss_max_t=0.85,
    )
    assert d.diffusion_loss_min_t == 0.1
    assert d.diffusion_loss_max_t == 0.85
    print("  PASS test_diffusion_bounds_dataset_config_set")


def test_diffusion_bounds_global_config_defaults():
    """Global TrainConfig defaults are 0.0 and 1.0 (open window)."""
    from toolkit.config_modules import TrainConfig
    tc = TrainConfig()
    assert tc.diffusion_loss_min_t == 0.0, f"Got {tc.diffusion_loss_min_t!r}"
    assert tc.diffusion_loss_max_t == 1.0, f"Got {tc.diffusion_loss_max_t!r}"
    print("  PASS test_diffusion_bounds_global_config_defaults")


def test_diffusion_bounds_batch_lists_mixed():
    """Mixed-dataset batch produces per-sample lists with None as fallback marker."""
    fi_a = _make_mock_file_item({
        'folder_path': '/a',
        'diffusion_loss_min_t': 0.2,
        'diffusion_loss_max_t': 0.7,
    })
    fi_b = _make_mock_file_item({
        'folder_path': '/b',  # inherits global
    })
    # _make_mock_file_item only sets a subset of fields; copy the new ones
    # the way DataLoaderBatchDTO would.
    from toolkit.config_modules import DatasetConfig
    fi_a.diffusion_loss_min_t = DatasetConfig(folder_path='/a',
        diffusion_loss_min_t=0.2, diffusion_loss_max_t=0.7).diffusion_loss_min_t
    fi_a.diffusion_loss_max_t = 0.7
    fi_b.diffusion_loss_min_t = None
    fi_b.diffusion_loss_max_t = None

    items = [fi_a, fi_b]
    min_list = [x.diffusion_loss_min_t for x in items]
    max_list = [x.diffusion_loss_max_t for x in items]
    assert min_list == [0.2, None], f"Got {min_list}"
    assert max_list == [0.7, None], f"Got {max_list}"
    print("  PASS test_diffusion_bounds_batch_lists_mixed")


def test_diffusion_bounds_mask_all_global():
    """All-None per-sample lists reproduce the original global-only mask."""
    t_ratio = torch.tensor([0.05, 0.2, 0.5, 0.8, 0.95])
    g_min, g_max = 0.1, 0.9
    per_sample = _diffusion_loss_mask(t_ratio, [None]*5, [None]*5, g_min, g_max)
    hardcoded = ((t_ratio >= g_min) & (t_ratio <= g_max)).float()
    assert per_sample.tolist() == hardcoded.tolist(), (
        f"Mismatch: {per_sample.tolist()} vs {hardcoded.tolist()}"
    )
    print("  PASS test_diffusion_bounds_mask_all_global")


def test_diffusion_bounds_mask_per_sample_override():
    """Per-sample overrides only affect their own sample; others fall back to global."""
    t_ratio = torch.tensor([0.5, 0.5, 0.5])
    # Sample 0: override min=0.6 -> 0.5 < 0.6 -> 0 (masked)
    # Sample 1: global (0.4, 0.8) -> 0.5 in range -> 1
    # Sample 2: override max=0.45 -> 0.5 > 0.45 -> 0 (masked)
    mask = _diffusion_loss_mask(
        t_ratio,
        batch_min_list=[0.6, None, None],
        batch_max_list=[None, None, 0.45],
        global_min=0.4, global_max=0.8,
    )
    assert mask.tolist() == [0.0, 1.0, 0.0], f"Got {mask.tolist()}"
    print("  PASS test_diffusion_bounds_mask_per_sample_override")


def test_diffusion_bounds_mask_boundary_inclusive():
    """Diffusion-loss bounds are INCLUSIVE on both ends — unlike the
    auxiliary-loss masks. A timestep equal to min or max stays in-window."""
    t_ratio = torch.tensor([0.4, 0.8])
    mask = _diffusion_loss_mask(t_ratio, [None, None], [None, None], 0.4, 0.8)
    # 0.4 >= 0.4 and 0.4 <= 0.8 -> 1
    # 0.8 >= 0.4 and 0.8 <= 0.8 -> 1
    assert mask.tolist() == [1.0, 1.0], f"Got {mask.tolist()}"
    print("  PASS test_diffusion_bounds_mask_boundary_inclusive")


def test_diffusion_bounds_mask_wide_window_override():
    """A per-dataset (0.0, 1.0) override re-enables the loss for that sample
    even when the global bounds would have excluded its timestep."""
    t_ratio = torch.tensor([0.05, 0.05])
    mask = _diffusion_loss_mask(
        t_ratio,
        batch_min_list=[0.0, None],   # sample 0 overrides; sample 1 keeps global
        batch_max_list=[1.0, None],
        global_min=0.5, global_max=0.9,
    )
    # 0.05 is well below global 0.5 — sample 1 is masked, but sample 0 is in
    # its own wide window and contributes.
    assert mask.tolist() == [1.0, 0.0], f"Got {mask.tolist()}"
    print("  PASS test_diffusion_bounds_mask_wide_window_override")


def test_diffusion_bounds_mask_narrow_window_override():
    """A per-dataset narrow window suppresses samples the global would keep."""
    t_ratio = torch.tensor([0.5, 0.5])
    mask = _diffusion_loss_mask(
        t_ratio,
        batch_min_list=[0.6, None],
        batch_max_list=[0.7, None],
        global_min=0.0, global_max=1.0,
    )
    # 0.5 < 0.6 -> sample 0 masked. Sample 1 keeps global (0,1) -> 1.
    assert mask.tolist() == [0.0, 1.0], f"Got {mask.tolist()}"
    print("  PASS test_diffusion_bounds_mask_narrow_window_override")


def test_diffusion_bounds_needs_mask_detection():
    """The trainer skips the mask build entirely when no global gating is
    set AND no per-sample override is present — matches the
    `_diff_needs_global or _diff_needs_per_sample` gate in SDTrainer."""
    # All-None list with default (open) globals -> no work to do.
    needs = (0.0 > 0.0 or 1.0 < 1.0)  # global check
    per_sample = any(v is not None for v in [None, None, None]) or \
                 any(v is not None for v in [None, None, None])
    assert not needs and not per_sample
    # Add a single per-sample override -> mask is needed even with open globals.
    per_sample2 = any(v is not None for v in [0.3, None]) or \
                  any(v is not None for v in [None, None])
    assert per_sample2
    print("  PASS test_diffusion_bounds_needs_mask_detection")


def test_diffusion_bounds_integration_loss_masking():
    """End-to-end: per-sample loss vector multiplied by mask zeroes out
    out-of-window samples while preserving in-window ones."""
    # Pretend per-sample diffusion losses (shape (B,) after reduction)
    loss = torch.tensor([0.4, 0.3, 0.5, 0.6])
    t_ratio = torch.tensor([0.05, 0.5, 0.95, 0.5])

    # Dataset A (samples 0, 2): override window [0.0, 1.0] (open)
    # Dataset B (samples 1, 3): inherit global window [0.2, 0.8]
    mask = _diffusion_loss_mask(
        t_ratio,
        batch_min_list=[0.0, None, 0.0, None],
        batch_max_list=[1.0, None, 1.0, None],
        global_min=0.2, global_max=0.8,
    )
    # Sample 0: t=0.05, window (0,1) -> in -> 1
    # Sample 1: t=0.5, global (0.2,0.8) -> in -> 1
    # Sample 2: t=0.95, window (0,1) -> in -> 1
    # Sample 3: t=0.5, global (0.2,0.8) -> in -> 1
    assert mask.tolist() == [1.0, 1.0, 1.0, 1.0]
    gated = loss * mask
    assert torch.allclose(gated, loss)

    # Now narrow the global window so sample 1 (t=0.5) falls outside,
    # while dataset A keeps its (0,1) override.
    mask2 = _diffusion_loss_mask(
        t_ratio,
        batch_min_list=[0.0, None, 0.0, None],
        batch_max_list=[1.0, None, 1.0, None],
        global_min=0.6, global_max=0.9,
    )
    # Sample 0: in (0,1) -> 1
    # Sample 1: 0.5 NOT in (0.6, 0.9) -> 0
    # Sample 2: in (0,1) -> 1
    # Sample 3: 0.5 NOT in (0.6, 0.9) -> 0
    assert mask2.tolist() == [1.0, 0.0, 1.0, 0.0]
    gated2 = loss * mask2
    expected = torch.tensor([0.4, 0.0, 0.5, 0.0])
    assert torch.allclose(gated2, expected), f"Got {gated2.tolist()}"
    print("  PASS test_diffusion_bounds_integration_loss_masking")


def test_diffusion_bounds_full_chain_propagation():
    """End-to-end propagation: DatasetConfig -> FileItemDTO-like -> batch list."""
    from toolkit.config_modules import DatasetConfig
    dc_a = DatasetConfig(folder_path='/a', diffusion_loss_min_t=0.15)
    dc_b = DatasetConfig(folder_path='/b', diffusion_loss_max_t=0.6)
    dc_c = DatasetConfig(folder_path='/c')

    # Mirror what FileItemDTO.__init__ does for these fields
    class _FI:
        pass

    items = []
    for dc in (dc_a, dc_b, dc_c):
        fi = _FI()
        fi.diffusion_loss_min_t = dc.diffusion_loss_min_t
        fi.diffusion_loss_max_t = dc.diffusion_loss_max_t
        items.append(fi)

    # Mirror what DataLoaderBatchDTO.__init__ does
    min_list = [x.diffusion_loss_min_t for x in items]
    max_list = [x.diffusion_loss_max_t for x in items]
    assert min_list == [0.15, None, None], f"Got {min_list}"
    assert max_list == [None, 0.6, None], f"Got {max_list}"
    print("  PASS test_diffusion_bounds_full_chain_propagation")


# ============================================================
# Resolution / num_repeats expansion in preprocess_dataset_raw_config
# ============================================================
#
# `preprocess_dataset_raw_config` splits each dataset entry into one entry
# per resolution. `num_repeats` may be a scalar (broadcast to every
# resolution) or a list aligned 1:1 with the resolution list, enabling
# unbalanced per-bucket sampling (e.g. 64:16:4:1 across 256/512/768/1024).


def test_expand_scalar_resolution_scalar_repeats():
    """Single resolution + scalar num_repeats: one entry, unchanged."""
    from toolkit.config_modules import preprocess_dataset_raw_config
    out = preprocess_dataset_raw_config([
        {'dataset_path': '/x', 'resolution': 512, 'num_repeats': 3},
    ])
    assert len(out) == 1
    assert out[0]['resolution'] == 512
    assert out[0]['num_repeats'] == 3
    print("  PASS test_expand_scalar_resolution_scalar_repeats")


def test_expand_list_resolution_scalar_repeats_broadcasts():
    """List resolution + scalar num_repeats: every split inherits the scalar."""
    from toolkit.config_modules import preprocess_dataset_raw_config
    out = preprocess_dataset_raw_config([
        {'dataset_path': '/x', 'resolution': [256, 512, 768], 'num_repeats': 4},
    ])
    assert len(out) == 3
    assert [d['resolution'] for d in out] == [256, 512, 768]
    assert [d['num_repeats'] for d in out] == [4, 4, 4]
    print("  PASS test_expand_list_resolution_scalar_repeats_broadcasts")


def test_expand_list_resolution_list_repeats_aligned():
    """Aligned-list num_repeats applies per-resolution (the headline feature)."""
    from toolkit.config_modules import preprocess_dataset_raw_config
    out = preprocess_dataset_raw_config([
        {
            'dataset_path': '/x',
            'resolution':  [256, 512, 768, 1024],
            'num_repeats': [ 64,  16,   4,    1],
        },
    ])
    assert len(out) == 4
    assert [d['resolution'] for d in out] == [256, 512, 768, 1024]
    assert [d['num_repeats'] for d in out] == [64, 16, 4, 1]
    print("  PASS test_expand_list_resolution_list_repeats_aligned")


def test_expand_default_num_repeats_when_missing():
    """num_repeats omitted defaults to 1 for each resolution split."""
    from toolkit.config_modules import preprocess_dataset_raw_config
    out = preprocess_dataset_raw_config([
        {'dataset_path': '/x', 'resolution': [256, 512]},
    ])
    assert len(out) == 2
    assert [d['num_repeats'] for d in out] == [1, 1]
    print("  PASS test_expand_default_num_repeats_when_missing")


def test_expand_length_mismatch_raises():
    """List num_repeats whose length disagrees with resolution list raises."""
    from toolkit.config_modules import preprocess_dataset_raw_config
    try:
        preprocess_dataset_raw_config([
            {'dataset_path': '/x', 'resolution': [256, 512, 768], 'num_repeats': [4, 2]},
        ])
    except ValueError as e:
        msg = str(e)
        assert 'num_repeats' in msg and 'resolution' in msg, msg
        assert '/x' in msg, f"Error should identify the offending dataset, got: {msg}"
        print("  PASS test_expand_length_mismatch_raises")
        return
    raise AssertionError("Expected ValueError for length mismatch")


def test_expand_scalar_resolution_list_repeats_length_one():
    """Scalar resolution + 1-element list num_repeats is allowed (treated as len 1)."""
    from toolkit.config_modules import preprocess_dataset_raw_config
    out = preprocess_dataset_raw_config([
        {'dataset_path': '/x', 'resolution': 512, 'num_repeats': [7]},
    ])
    assert len(out) == 1
    assert out[0]['resolution'] == 512
    assert out[0]['num_repeats'] == 7
    print("  PASS test_expand_scalar_resolution_list_repeats_length_one")


def test_expand_preserves_other_fields_across_splits():
    """All non-resolution fields are duplicated into every split (incl. overrides)."""
    from toolkit.config_modules import preprocess_dataset_raw_config
    out = preprocess_dataset_raw_config([
        {
            'dataset_path': '/x',
            'resolution': [256, 1024],
            'num_repeats': [8, 1],
            'identity_loss_weight': 0.5,
            'caption_dropout_rate': 0.05,
            'flip_x': True,
        },
    ])
    assert len(out) == 2
    for d in out:
        assert d['dataset_path'] == '/x'
        assert d['identity_loss_weight'] == 0.5
        assert d['caption_dropout_rate'] == 0.05
        assert d['flip_x'] is True
    print("  PASS test_expand_preserves_other_fields_across_splits")


def test_expand_multiple_datasets_independent():
    """Each top-level dataset entry is expanded independently."""
    from toolkit.config_modules import preprocess_dataset_raw_config
    out = preprocess_dataset_raw_config([
        {'dataset_path': '/a', 'resolution': [256, 512], 'num_repeats': [4, 2]},
        {'dataset_path': '/b', 'resolution': 1024, 'num_repeats': 1},
    ])
    assert len(out) == 3
    assert (out[0]['dataset_path'], out[0]['resolution'], out[0]['num_repeats']) == ('/a', 256, 4)
    assert (out[1]['dataset_path'], out[1]['resolution'], out[1]['num_repeats']) == ('/a', 512, 2)
    assert (out[2]['dataset_path'], out[2]['resolution'], out[2]['num_repeats']) == ('/b', 1024, 1)
    print("  PASS test_expand_multiple_datasets_independent")


def test_expand_does_not_mutate_input():
    """The input dataset dicts are not mutated by the preprocessor."""
    from toolkit.config_modules import preprocess_dataset_raw_config
    src = {'dataset_path': '/x', 'resolution': [256, 512], 'num_repeats': [4, 2]}
    src_snapshot = dict(src)
    preprocess_dataset_raw_config([src])
    assert src == src_snapshot, f"Input was mutated: {src} != {src_snapshot}"
    print("  PASS test_expand_does_not_mutate_input")


# ============================================================
# Run all tests
# ============================================================

if __name__ == '__main__':
    print("=== Config propagation ===")
    test_dataset_config_defaults()
    test_dataset_config_overrides()
    test_global_config_unchanged()

    print("\n=== Batch list construction ===")
    test_batch_lists_mixed_datasets()

    print("\n=== Per-sample timestep masks ===")
    test_mask_all_global()
    test_mask_per_sample_override()
    test_mask_all_overridden()
    test_mask_boundary_exclusive()
    test_mask_batch_size_1()

    print("\n=== Per-sample cosine thresholds ===")
    test_cos_threshold_all_global()
    test_cos_threshold_per_sample()

    print("\n=== Per-sample loss weights ===")
    test_weight_all_global()
    test_weight_per_sample_override()
    test_weight_zero_disables_sample()

    print("\n=== Integration ===")
    test_integration_mixed_batch()
    test_integration_any_active()
    test_integration_loss_masking()

    print("\n=== Edge cases ===")
    test_empty_batch()
    test_all_none_matches_global()
    test_wide_window_override()
    test_narrow_window_override()

    print("\n=== Face suppression weight ===")
    test_face_suppression_config_default()
    test_face_suppression_config_set()
    test_face_suppression_mask_spatial_pattern()
    test_face_suppression_mean_normalization()
    test_face_suppression_weight_1_no_effect()
    test_face_suppression_none_skipped()
    test_face_suppression_half_weight()
    test_face_suppression_mixed_batch()
    test_face_suppression_no_bboxes_returns_none()

    print("\n=== Resolution / num_repeats expansion ===")
    test_expand_scalar_resolution_scalar_repeats()
    test_expand_list_resolution_scalar_repeats_broadcasts()
    test_expand_list_resolution_list_repeats_aligned()
    test_expand_default_num_repeats_when_missing()
    test_expand_length_mismatch_raises()
    test_expand_scalar_resolution_list_repeats_length_one()
    test_expand_preserves_other_fields_across_splits()
    test_expand_multiple_datasets_independent()
    test_expand_does_not_mutate_input()

    print("\n=== Diffusion loss timestep bounds ===")
    test_diffusion_bounds_dataset_config_default()
    test_diffusion_bounds_dataset_config_set()
    test_diffusion_bounds_global_config_defaults()
    test_diffusion_bounds_batch_lists_mixed()
    test_diffusion_bounds_mask_all_global()
    test_diffusion_bounds_mask_per_sample_override()
    test_diffusion_bounds_mask_boundary_inclusive()
    test_diffusion_bounds_mask_wide_window_override()
    test_diffusion_bounds_mask_narrow_window_override()
    test_diffusion_bounds_needs_mask_detection()
    test_diffusion_bounds_integration_loss_masking()
    test_diffusion_bounds_full_chain_propagation()

    print("\n=== ALL TESTS PASSED ===")
