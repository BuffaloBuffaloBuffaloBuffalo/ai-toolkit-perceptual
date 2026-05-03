#!/usr/bin/env python3
"""Real-pipeline validation of subject-mask alignment (v1 vs v2).

For each test image:
  1. Compute bucket params via toolkit.buckets.get_bucket_for_image_size
     (mirrors dataloader_mixins.setup_buckets non-random-crop branch).
  2. Apply dataloader transforms (flip + resize + crop) → training PIL.
  3. v1 simulation: segment the raw image, downsample person mask to a square,
     upscale to training-PIL shape (what the OLD cache path effectively feeds
     to the loss after F.interpolate at train time).
  4. v2 path: call cache_subject_masks on a mock FileItemDTO with bucket
     params attached — exercises the real new code path, writes to tmp cache.
  5. Reference: segment training PIL directly (what the mask SHOULD be).
  6. Report IoU vs reference for v1-sim and v2-real.
  7. Save 4-panel overlay to output/mask_validation/.

Runs for flip_x in (False, True) so the flip handling is exercised.

Usage: python scripts/validate_subject_mask_alignment.py [--n 3]
"""

import argparse
import math
import os
import shutil
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from PIL.ImageOps import exif_transpose

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from toolkit.buckets import get_bucket_for_image_size  # noqa: E402
from toolkit.config_modules import SubjectMaskConfig  # noqa: E402
from toolkit.subject_mask import (  # noqa: E402
    SubjectMaskExtractor, cache_subject_masks,
)

warnings.filterwarnings("ignore")

TEST_DIR = "test_data/scarlett_full"
OUT_DIR = "output/mask_validation"
RESOLUTION = 512
BUCKET_TOLERANCE = 64


@dataclass
class BucketParams:
    scale_to_width: int
    scale_to_height: int
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int


@dataclass
class MockFileItem:
    """Minimal stand-in with the attributes cache_subject_masks reads."""
    path: str
    scale_to_width: int
    scale_to_height: int
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int
    flip_x: bool = False
    flip_y: bool = False
    subject_mask: Optional[torch.Tensor] = None
    body_mask: Optional[torch.Tensor] = None
    clothing_mask: Optional[torch.Tensor] = None


def compute_bucket_params(orig_w: int, orig_h: int,
                          resolution: int = RESOLUTION,
                          tolerance: int = BUCKET_TOLERANCE) -> BucketParams:
    br = get_bucket_for_image_size(orig_w, orig_h, resolution=resolution,
                                   divisibility=tolerance)
    sx, sy = br["width"] / orig_w, br["height"] / orig_h
    m = max(sx, sy)
    stw = int(math.ceil(orig_w * m))
    sth = int(math.ceil(orig_h * m))
    cw, ch = br["width"], br["height"]
    cx = (stw - cw) // 2
    cy = (sth - ch) // 2
    return BucketParams(stw, sth, cx, cy, cw, ch)


def apply_dataloader_transform(img: Image.Image, flip_x: bool, bp: BucketParams) -> Image.Image:
    img = img.convert("RGB")
    if flip_x:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = img.resize((bp.scale_to_width, bp.scale_to_height), Image.BICUBIC)
    img = img.crop((bp.crop_x, bp.crop_y, bp.crop_x + bp.crop_width,
                    bp.crop_y + bp.crop_height))
    return img


def mask_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.bool)
    b = b.to(torch.bool)
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return inter / max(1, union)


def overlay_mask(pil: Image.Image, mask_bool: torch.Tensor, color=(0, 255, 0),
                 alpha: float = 0.45) -> Image.Image:
    """Blend a bool mask as a semi-transparent colored overlay."""
    pil_rgba = pil.convert("RGBA")
    mask_np = mask_bool.cpu().numpy().astype(np.uint8) * 255
    # Resize mask to pil's size if needed (nearest).
    m = Image.fromarray(mask_np).resize(pil.size, Image.NEAREST)
    m_arr = np.asarray(m)
    over = np.zeros((pil.height, pil.width, 4), dtype=np.uint8)
    over[..., 0] = color[0]; over[..., 1] = color[1]; over[..., 2] = color[2]
    over[..., 3] = (m_arr * alpha).astype(np.uint8)
    over_pil = Image.fromarray(over, mode="RGBA")
    return Image.alpha_composite(pil_rgba, over_pil).convert("RGB")


def simulate_v1_mask_at_training(extractor: SubjectMaskExtractor,
                                 raw_pil: Image.Image,
                                 bp: BucketParams,
                                 flip_x: bool,
                                 cache_res: int) -> torch.Tensor:
    """What v1 effectively applied after F.interpolate at training time.

    v1: segment raw → downsample to (cache_res, cache_res) square → at train
    time F.interpolate to latent (or to training PIL for visual parity here).
    Crucially v1 did NOT flip or crop. We match this behavior for apples-to-
    apples comparison against the training tensor.
    """
    masks = extractor.extract(raw_pil)  # raw-coord bool masks
    person_np = masks["person"].astype(np.uint8)
    # v1 square downsample
    t = torch.from_numpy(person_np).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(cache_res, cache_res), mode="nearest")
    # At training time this mask is F.interpolated to the latent grid, whose
    # aspect matches the training tensor. For visual parity we resize to the
    # training PIL's (h, w) directly — same result as latent+nearest-upscale.
    t = F.interpolate(t, size=(bp.crop_height, bp.crop_width), mode="nearest")
    # Because v1 did not flip, if the training pipeline flipped, the resulting
    # mask covers the mirrored location. We simulate that by NOT applying flip
    # here — the mask stays oriented to raw pixels.
    return (t.squeeze(0).squeeze(0) > 0.5).to(torch.bool)


def segment_reference(extractor: SubjectMaskExtractor,
                      training_pil: Image.Image) -> torch.Tensor:
    """Segment the training PIL directly — this is what the mask SHOULD be."""
    masks = extractor.extract(training_pil)
    person_np = masks["person"].astype(np.uint8)
    t = torch.from_numpy(person_np)
    # extractor returns at input resolution already (it accepts PIL)
    if t.shape != (training_pil.height, training_pil.width):
        t = F.interpolate(
            t.float().unsqueeze(0).unsqueeze(0),
            size=(training_pil.height, training_pil.width),
            mode="nearest",
        ).squeeze(0).squeeze(0)
    return (t > 0.5).to(torch.bool)


def list_images(folder: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(
        [os.path.join(folder, f) for f in os.listdir(folder)
         if os.path.splitext(f)[1].lower() in exts]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--dir", default=TEST_DIR)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    paths = list_images(args.dir)[: args.n]
    if not paths:
        print(f"No images in {args.dir}"); return

    print("Loading SubjectMaskExtractor (YOLO + SAM2 + SegFormer-clothes)...")
    cfg = SubjectMaskConfig(enabled=True)
    extractor = SubjectMaskExtractor(cfg)
    # Also the top-level cache config for cache_subject_masks (doesn't build
    # a new extractor since we prime one here).
    cfg_cache = SubjectMaskConfig(enabled=True, cache_resolution=cfg.cache_resolution)

    rows = []
    for p in paths:
        raw = exif_transpose(Image.open(p)).convert("RGB")
        orig_w, orig_h = raw.size
        bp = compute_bucket_params(orig_w, orig_h)
        stem = os.path.splitext(os.path.basename(p))[0]

        for flip_x in (False, True):
            train_pil = apply_dataloader_transform(raw, flip_x, bp)

            # Reference: segment the training tensor directly.
            ref_mask = segment_reference(extractor, train_pil)

            # v1 simulation: segment raw, downsample to square, interp to train dims.
            v1_mask = simulate_v1_mask_at_training(extractor, raw, bp, flip_x,
                                                   cfg_cache.cache_resolution)

            # v2 real: call cache_subject_masks on a mock file_item pointing to
            # a temp copy of the image (function writes cache next to the file).
            with tempfile.TemporaryDirectory(prefix="smask_v2_") as tmp:
                dst = os.path.join(tmp, os.path.basename(p))
                shutil.copy2(p, dst)
                item = MockFileItem(
                    path=dst,
                    scale_to_width=bp.scale_to_width,
                    scale_to_height=bp.scale_to_height,
                    crop_x=bp.crop_x, crop_y=bp.crop_y,
                    crop_width=bp.crop_width, crop_height=bp.crop_height,
                    flip_x=flip_x, flip_y=False,
                )
                cache_subject_masks([item], cfg_cache, preview_dir=None)
                v2_mask = item.subject_mask.to(torch.bool)

            v1_iou = mask_iou(v1_mask, ref_mask)
            v2_iou = mask_iou(v2_mask, ref_mask)
            rows.append((stem[:28], "flip" if flip_x else "noflip", v1_iou, v2_iou,
                         int(ref_mask.sum()), int(v1_mask.sum()), int(v2_mask.sum())))

            # 4-panel overlay: training | training+v1 (red) | training+v2 (lime) | training+ref (cyan)
            panels = [
                train_pil,
                overlay_mask(train_pil, v1_mask, (255, 80, 80)),
                overlay_mask(train_pil, v2_mask, (60, 255, 60)),
                overlay_mask(train_pil, ref_mask, (80, 200, 255)),
            ]
            labels = ["training", f"v1 IoU={v1_iou:.3f}", f"v2 IoU={v2_iou:.3f}", "reference"]
            target_h = 480
            def fit(im):
                r = target_h / im.height
                return im.resize((int(im.width * r), target_h), Image.BICUBIC)
            panels = [fit(pnl) for pnl in panels]
            combo = Image.new("RGB", (sum(p.width for p in panels), target_h), (0, 0, 0))
            x = 0
            for pnl, lbl in zip(panels, labels):
                combo.paste(pnl, (x, 0))
                d = ImageDraw.Draw(combo)
                d.text((x + 6, 6), lbl, fill=(255, 255, 0))
                x += pnl.width
            combo.save(os.path.join(OUT_DIR, f"{stem}_{'flip' if flip_x else 'noflip'}.jpg"),
                       quality=88)

    print(f"\nResults (overlays in {OUT_DIR}):\n")
    print(f"{'image':30s} {'mode':6s} {'v1 IoU':>7s} {'v2 IoU':>7s}  {'ref':>6s} {'v1':>6s} {'v2':>6s}")
    for r in rows:
        print(f"{r[0]:30s} {r[1]:6s} {r[2]:7.3f} {r[3]:7.3f}  {r[4]:6d} {r[5]:6d} {r[6]:6d}")

    if rows:
        nf = [r for r in rows if r[1] == "noflip"]
        fl = [r for r in rows if r[1] == "flip"]
        if nf:
            print(f"\nnoflip mean IoU  v1={sum(r[2] for r in nf)/len(nf):.3f}  "
                  f"v2={sum(r[3] for r in nf)/len(nf):.3f}")
        if fl:
            print(f"flip   mean IoU  v1={sum(r[2] for r in fl)/len(fl):.3f}  "
                  f"v2={sum(r[3] for r in fl)/len(fl):.3f}")


if __name__ == "__main__":
    main()
