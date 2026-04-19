"""End-to-end test for the subject_mask caching pipeline.

Runs on real images in `test_data/scarlett_full/` (42 images of Clare Bowen).
Loads YOLO + SAM 2 + SegFormer once, caches per-image {person, body, clothing}
masks, and validates shape + coverage + cache-hit behavior.

Phase 1: caching only. No loss logic is exercised.

Usage:
    python testing/test_subject_mask_cache.py
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.config_modules import SubjectMaskConfig
from toolkit.subject_mask import CACHE_VERSION_KEY, cache_subject_masks


IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".avif")
DATA_DIR = Path(__file__).parent.parent / "test_data" / "scarlett_full"


class _FakeFileItem:
    """Mimics FileItemDTO for cache_subject_masks tests — same pattern as test_face_id.py."""
    def __init__(self, path: str):
        self.path = path
        self.subject_mask = None
        self.body_mask = None
        self.clothing_mask = None


def _gather_images(data_dir: Path):
    imgs = sorted(p for p in data_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)
    return imgs


def _clear_cache(data_dir: Path):
    cache_dir = data_dir / "_face_id_cache"
    if not cache_dir.exists():
        return
    for p in cache_dir.glob("*_subject_masks.safetensors"):
        p.unlink()


def _cache_count(data_dir: Path) -> int:
    cache_dir = data_dir / "_face_id_cache"
    if not cache_dir.exists():
        return 0
    return len(list(cache_dir.glob("*_subject_masks.safetensors")))


def main():
    if not DATA_DIR.exists():
        raise SystemExit(f"test data missing: {DATA_DIR}")

    images = _gather_images(DATA_DIR)
    if len(images) < 10:
        raise SystemExit(f"Expected at least 10 images in {DATA_DIR}, found {len(images)}")

    # Clear any previously cached subject mask files so we exercise the full pipeline.
    _clear_cache(DATA_DIR)
    assert _cache_count(DATA_DIR) == 0, "Cache should be empty before first run"

    print(f"=== Subject mask cache test ===")
    print(f"Dataset:  {DATA_DIR}")
    print(f"Images:   {len(images)}")

    cfg = SubjectMaskConfig(enabled=True)
    assert cfg.enabled is True
    assert cfg.cache_resolution == 256
    assert cfg.segformer_res == 768
    assert cfg.sam_size == "small"

    # Build fake file items and run full cache pass
    items = [_FakeFileItem(str(p)) for p in images]

    t0 = time.perf_counter()
    cache_subject_masks(items, cfg)
    t1 = time.perf_counter()
    print(f"First pass: {t1 - t0:.1f}s for {len(items)} images "
          f"({(t1 - t0) / max(len(items), 1):.2f}s/img)")

    # ---- assertions ----
    assert _cache_count(DATA_DIR) == len(items), \
        f"Expected {len(items)} cache files, got {_cache_count(DATA_DIR)}"

    person_pcts = []
    body_pcts = []
    clothing_pcts = []

    for it in items:
        # Attached tensors must be bool, CPU, (256, 256)
        for name in ("subject_mask", "body_mask", "clothing_mask"):
            t = getattr(it, name)
            assert t is not None, f"{name} missing on {it.path}"
            assert isinstance(t, torch.Tensor), f"{name} wrong type on {it.path}"
            assert t.dtype == torch.bool, f"{name} dtype={t.dtype}, expected bool on {it.path}"
            assert t.device.type == "cpu", f"{name} device={t.device}, expected cpu"
            assert t.shape == (256, 256), f"{name} shape={tuple(t.shape)}, expected (256, 256)"

        person_pct = 100.0 * it.subject_mask.float().mean().item()
        body_pct = 100.0 * it.body_mask.float().mean().item()
        clothing_pct = 100.0 * it.clothing_mask.float().mean().item()
        person_pcts.append(person_pct)
        body_pcts.append(body_pct)
        clothing_pcts.append(clothing_pct)

    n = len(items)
    print(f"Coverage: person  mean={np.mean(person_pcts):.1f}%  "
          f"min={np.min(person_pcts):.1f}%  max={np.max(person_pcts):.1f}%")
    print(f"         body    mean={np.mean(body_pcts):.1f}%  "
          f"min={np.min(body_pcts):.1f}%  max={np.max(body_pcts):.1f}%")
    print(f"         cloth   mean={np.mean(clothing_pcts):.1f}%  "
          f"min={np.min(clothing_pcts):.1f}%  max={np.max(clothing_pcts):.1f}%")

    # Sanity bounds: every image must have a plausible subject
    for i, p in enumerate(person_pcts):
        assert 5.0 <= p <= 99.0, (
            f"person_pct={p:.1f}% out of [5, 99] for {items[i].path}")

    # Body (skin/hair) should be non-trivial for most portraits. Allow a few to
    # be fully clothed / body-unseen — require >= 35/42 above 0.
    n_body_nonzero = sum(1 for p in body_pcts if p > 0.0)
    expected_min = min(35, max(0, n - 7))
    assert n_body_nonzero >= expected_min, (
        f"Only {n_body_nonzero}/{n} images have body_pct > 0 (need >= {expected_min})")

    # ---- cache file format spot check on first image ----
    first_stem = Path(items[0].path).stem
    cache_file = DATA_DIR / "_face_id_cache" / f"{first_stem}_subject_masks.safetensors"
    assert cache_file.exists(), f"Cache file missing: {cache_file}"
    data = load_file(str(cache_file))
    for k in ("person", "body", "clothing"):
        assert k in data, f"Cache missing key {k}"
        t = data[k]
        assert t.dtype == torch.uint8, f"Cache key {k} dtype={t.dtype}, expected uint8"
        assert t.shape == (256, 256), f"Cache key {k} shape={tuple(t.shape)}"
        # Binary 0/255 representation
        unique = set(t.unique().tolist())
        assert unique <= {0, 255}, f"Cache key {k} has non-binary values: {unique}"
    assert CACHE_VERSION_KEY in data, f"Cache missing version key {CACHE_VERSION_KEY}"

    # ---- second pass must be cache-only (fast, no GPU needed beyond load_file) ----
    # Rebuild items so attributes are None again
    items2 = [_FakeFileItem(str(p)) for p in images]
    t0 = time.perf_counter()
    cache_subject_masks(items2, cfg)
    t1 = time.perf_counter()
    second_pass = t1 - t0
    print(f"Second pass (cache hit): {second_pass:.2f}s for {n} images")

    # Cache-hit pass must attach masks and skip extraction entirely.
    for it in items2:
        assert it.subject_mask is not None
        assert it.body_mask is not None
        assert it.clothing_mask is not None
        assert it.subject_mask.dtype == torch.bool
        assert it.subject_mask.shape == (256, 256)

    # 0.2s/image is very generous for cache-load path.
    assert second_pass < 0.2 * n + 5.0, (
        f"Second pass too slow ({second_pass:.2f}s); cache-hit path may be broken")

    print(f"\nPASS: subject_mask caching pipeline on {n} images")
    print(f"Cache dir: {DATA_DIR / '_face_id_cache'}")


if __name__ == "__main__":
    main()
