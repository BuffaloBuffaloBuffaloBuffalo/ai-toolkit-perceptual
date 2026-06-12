"""Tests for `subject_mask.mask_source: alpha` (issue #48).

Alpha mode reads masks straight from PNG alpha channels instead of running
YOLO + SAM 2 + SegFormer, so this whole test runs on CPU in seconds with no
model downloads. Covers:

- alpha extraction semantics (threshold, no-alpha fallback, LA/palette modes)
- person == body == clothing in alpha mode
- no segmentation models are imported, let alone loaded
- cache write/read round-trip + bucket-transform (flip/resize/crop) alignment
- cache invalidation when the cached mask_source doesn't match the config
- RGBA-safe debug preview tiles
- preflight script end-to-end in --mask-source alpha mode

Usage:
    python testing/test_subject_mask_alpha.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.config_modules import SubjectMaskConfig
from toolkit.subject_mask import (
    CACHE_VERSION_KEY,
    SubjectMaskExtractor,
    _extract_alpha_masks,
    cache_subject_masks,
)

REPO_ROOT = Path(__file__).parent.parent


class _FakeFileItem:
    """Mimics FileItemDTO — same pattern as test_subject_mask_cache.py."""
    def __init__(self, path: str):
        self.path = path
        self.subject_mask = None
        self.body_mask = None
        self.clothing_mask = None


def _make_rgba_png(path: Path, size=(96, 128), subject_box=(20, 30, 60, 90)):
    """Random RGB with an opaque rectangle in an otherwise transparent alpha."""
    W, H = size
    rng = np.random.RandomState(hash(path.name) % (2**31))
    rgb = rng.randint(0, 255, (H, W, 3), dtype=np.uint8)
    alpha = np.zeros((H, W), dtype=np.uint8)
    x0, y0, x1, y1 = subject_box
    alpha[y0:y1, x0:x1] = 255
    img = Image.fromarray(np.dstack([rgb, alpha]))
    img.save(path)
    return (alpha > 127)


def test_extract_semantics():
    # Opaque box → True inside, False outside; exact threshold boundary at 127.
    W, H = 64, 48
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    alpha = np.zeros((H, W), dtype=np.uint8)
    alpha[10:20, 10:30] = 255
    alpha[30:40, 10:30] = 127  # at threshold → background
    alpha[30:40, 40:60] = 128  # just above → subject
    img = Image.fromarray(np.dstack([rgb, alpha]))
    masks = _extract_alpha_masks(img)

    assert masks["person"].shape == (H, W)
    assert masks["person"].dtype == np.bool_
    assert masks["person"][15, 20] and not masks["person"][0, 0]
    assert not masks["person"][35, 20], "alpha==127 must be background"
    assert masks["person"][35, 50], "alpha==128 must be subject"
    assert np.array_equal(masks["person"], masks["body"])
    assert np.array_equal(masks["person"], masks["clothing"])
    assert masks["class_map"].shape == (H, W) and masks["class_map"].max() == 0
    assert masks["boxes"] == []

    # No alpha channel at all → full coverage (train normally).
    rgb_img = Image.fromarray(rgb)
    assert _extract_alpha_masks(rgb_img)["person"].all()

    # LA mode resolves through convert('RGBA').
    la = Image.merge("LA", (Image.fromarray(alpha), Image.fromarray(alpha)))
    assert _extract_alpha_masks(la)["person"][15, 20]

    # Palette image with transparency.
    pal = Image.fromarray(rgb).convert("P")
    pal.info["transparency"] = 0
    m = _extract_alpha_masks(pal)["person"]
    assert m.shape == (H, W)
    print("PASS: extraction semantics")


def test_extractor_loads_no_models():
    cfg = SubjectMaskConfig(enabled=True, mask_source="alpha")
    before = set(sys.modules)
    ext = SubjectMaskExtractor(cfg)
    assert ext.yolo is None and ext.sam is None and ext.seg is None
    assert ext.num_parse_classes == 1
    assert "ultralytics" not in set(sys.modules) - before, \
        "alpha mode must not import ultralytics"
    img = Image.new("RGBA", (32, 32), (0, 0, 0, 255))
    assert ext.extract(img)["person"].all()
    ext.cleanup()  # must not raise on None models
    print("PASS: no models loaded in alpha mode")


def test_cache_roundtrip_and_invalidation(tmp: Path):
    data_dir = tmp / "ds"
    data_dir.mkdir()
    gt = {}
    for i in range(3):
        p = data_dir / f"img_{i}.png"
        gt[p.stem] = _make_rgba_png(p, subject_box=(10 + i * 5, 20, 50 + i * 5, 100))
    # One image with no alpha — must get a full-coverage mask, not a warning-empty one.
    Image.new("RGB", (96, 128), (10, 20, 30)).save(data_dir / "img_noalpha.jpg")

    cfg = SubjectMaskConfig(enabled=True, mask_source="alpha", cache_resolution=64)
    items = [_FakeFileItem(str(p)) for p in sorted(data_dir.glob("img_*"))]

    t0 = time.perf_counter()
    cache_subject_masks(items, cfg)
    dt = time.perf_counter() - t0
    assert dt < 10.0, f"alpha cache pass should be near-instant, took {dt:.1f}s"

    for it in items:
        for name in ("subject_mask", "body_mask", "clothing_mask"):
            t = getattr(it, name)
            assert isinstance(t, torch.Tensor) and t.dtype == torch.bool
            assert t.shape == (64, 64)
        assert torch.equal(it.subject_mask, it.body_mask)
        assert torch.equal(it.subject_mask, it.clothing_mask)
        stem = Path(it.path).stem
        if stem == "img_noalpha":
            assert it.subject_mask.all(), "no-alpha image must be full coverage"
        else:
            # Downsampled mask coverage should track the authored alpha coverage.
            want = gt[stem].mean()
            got = it.subject_mask.float().mean().item()
            assert abs(got - want) < 0.05, f"{stem}: coverage {got:.3f} vs authored {want:.3f}"

    # Cache file format: binary masks + source tag + version sentinel.
    cache_dir = data_dir / "_face_id_cache"
    cache_files = sorted(cache_dir.glob("*_subject_masks_*.safetensors"))
    assert len(cache_files) == len(items)
    data = load_file(str(cache_files[0]))
    assert CACHE_VERSION_KEY in data
    assert float(data["mask_source"].item()) == 1.0
    assert set(data["person"].unique().tolist()) <= {0, 255}

    # Second pass must be pure cache-hit and attach identical masks.
    items2 = [_FakeFileItem(it.path) for it in items]
    cache_subject_masks(items2, cfg)
    for a, b in zip(items, items2):
        assert torch.equal(a.subject_mask, b.subject_mask)

    # A cache written by the model pipeline (mask_source=0) must NOT be served
    # in alpha mode — flip the tag on one file and confirm it re-extracts.
    victim = cache_files[0]
    tampered = dict(load_file(str(victim)))
    tampered["mask_source"] = torch.tensor([0.0])
    tampered["person"] = torch.zeros_like(tampered["person"])  # poison
    save_file(tampered, str(victim))
    items3 = [_FakeFileItem(items[0].path)]
    cache_subject_masks(items3, cfg)
    assert items3[0].subject_mask.any(), "stale auto-source cache was served in alpha mode"
    assert float(load_file(str(victim))["mask_source"].item()) == 1.0

    # Legacy cache with no mask_source key == auto → also invalid in alpha mode.
    legacy = {k: v for k, v in load_file(str(victim)).items() if k != "mask_source"}
    legacy["person"] = torch.zeros_like(legacy["person"])
    save_file(legacy, str(victim))
    items4 = [_FakeFileItem(items[0].path)]
    cache_subject_masks(items4, cfg)
    assert items4[0].subject_mask.any(), "legacy keyless cache was served in alpha mode"
    print("PASS: cache round-trip + source invalidation")


def test_bucket_transform_alignment(tmp: Path):
    """Mask must go through the same flip+resize+crop chain as training pixels."""
    data_dir = tmp / "ds_bucket"
    data_dir.mkdir()
    p = data_dir / "left_half.png"
    W, H = 100, 80
    alpha = np.zeros((H, W), dtype=np.uint8)
    alpha[:, : W // 2] = 255  # subject = left half
    Image.fromarray(
        np.dstack([np.zeros((H, W, 3), dtype=np.uint8), alpha])
    ).save(p)

    it = _FakeFileItem(str(p))
    it.flip_x = True  # subject becomes the RIGHT half
    it.scale_to_width = 200
    it.scale_to_height = 160
    it.crop_x, it.crop_y, it.crop_width, it.crop_height = 0, 0, 200, 160

    cfg = SubjectMaskConfig(enabled=True, mask_source="alpha")
    cache_subject_masks([it], cfg)
    m = it.subject_mask
    assert m.shape == (160, 200)
    assert not m[:, :90].any(), "left side should be background after flip"
    assert m[:, 110:].all(), "right side should be subject after flip"
    print("PASS: bucket transform alignment")


def test_debug_previews(tmp: Path):
    data_dir = tmp / "ds_prev"
    data_dir.mkdir()
    _make_rgba_png(data_dir / "img.png")
    preview_dir = tmp / "previews"
    cfg = SubjectMaskConfig(
        enabled=True, mask_source="alpha", cache_resolution=64,
        save_debug_previews=True,
    )
    cache_subject_masks([_FakeFileItem(str(data_dir / "img.png"))], cfg,
                        preview_dir=str(preview_dir))
    tile = preview_dir / "img.png"
    assert tile.exists(), "debug preview tile not written"
    assert Image.open(tile).mode == "RGB"
    print("PASS: RGBA-safe debug previews")


def test_preflight_script(tmp: Path):
    data_dir = tmp / "ds_preflight"
    data_dir.mkdir()
    for i in range(2):
        _make_rgba_png(data_dir / f"img_{i}.png")
    out_dir = tmp / "preflight_out"
    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "preflight_subject_masks.py"),
         "--dataset-dir", str(data_dir), "--output-dir", str(out_dir),
         "--mask-source", "alpha"],
        capture_output=True, text=True, timeout=120,
    )
    progress = json.loads((out_dir / "progress.json").read_text())
    assert res.returncode == 0, f"preflight failed: {res.stderr}\n{progress}"
    assert progress["status"] == "done", progress
    assert (out_dir / "done.marker").read_text().strip() == "ok"
    for i in range(2):
        assert (out_dir / f"img_{i}.png").exists()
        assert not (out_dir / f"img_{i}.error.txt").exists()
    assert json.loads((out_dir / "config.json").read_text())["mask_source"] == "alpha"
    print("PASS: preflight script in alpha mode")


def main():
    test_extract_semantics()
    test_extractor_loads_no_models()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_cache_roundtrip_and_invalidation(tmp)
        test_bucket_transform_alignment(tmp)
        test_debug_previews(tmp)
        test_preflight_script(tmp)
    print("\nPASS: all alpha mask_source tests")


if __name__ == "__main__":
    main()
