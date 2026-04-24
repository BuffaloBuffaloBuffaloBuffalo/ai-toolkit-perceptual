#!/usr/bin/env python3
"""Real-pipeline validation of SDTrainer's face-bbox transform chain.

For each test image, runs a single round-trip:
  raw PIL → InsightFace face detect → raw_bbox
  raw PIL → dataloader transforms (resize + optional flip + crop) → training tensor
  raw_bbox → SDTrainer bbox math (SDTrainer.py:1910-1938) → predicted_training_bbox
  training tensor → InsightFace face detect → detected_training_bbox
  IoU(predicted, detected) + center offset → confirm or reject the math.

Repeats with flip_x=False and flip_x=True so the missing-flip bug surfaces
numerically. Saves side-by-side debug overlays to output/bbox_validation/.

Usage: python scripts/validate_bbox_transform.py [--n 3]
"""

import argparse
import math
import os
import sys
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from PIL.ImageOps import exif_transpose

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from toolkit.buckets import get_bucket_for_image_size  # noqa: E402

warnings.filterwarnings("ignore")

TEST_DIR = "test_data/scarlett_full"
OUT_DIR = "output/bbox_validation"
RESOLUTION = 512
BUCKET_TOLERANCE = 64
LATENT_DIVISOR = 8  # VAE downscale; x0_pixels has the same H/W as crop in most archs


@dataclass
class BucketParams:
    scale_to_width: int
    scale_to_height: int
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int


def compute_bucket_params(orig_w: int, orig_h: int,
                          resolution: int = RESOLUTION,
                          tolerance: int = BUCKET_TOLERANCE) -> BucketParams:
    """Mirror of dataloader_mixins.setup_buckets (non-random-crop branch)."""
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


def apply_dataloader_transform(img: Image.Image, flip_x: bool,
                               bp: BucketParams) -> Image.Image:
    """Mirror of dataloader_mixins.load_and_process_image lines 763-793."""
    img = img.convert("RGB")
    if flip_x:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = img.resize((bp.scale_to_width, bp.scale_to_height), Image.BICUBIC)
    img = img.crop((bp.crop_x, bp.crop_y,
                    bp.crop_x + bp.crop_width, bp.crop_y + bp.crop_height))
    return img


def transform_bbox_sdtrainer(raw_bbox: Tuple[float, float, float, float],
                             orig_w: int, orig_h: int, bp: BucketParams,
                             px_w: int, px_h: int,
                             flip_x: bool = False, flip_y: bool = False
                             ) -> Optional[Tuple[float, float, float, float]]:
    """Exact mirror of SDTrainer.py:1910-1938 bbox-transform chain (post-flip-fix)."""
    bx1, by1, bx2, by2 = raw_bbox
    # flip in raw coords — matches dataloader flip order
    if flip_x:
        bx1, bx2 = orig_w - bx2, orig_w - bx1
    if flip_y:
        by1, by2 = orig_h - by2, orig_h - by1
    # scale
    bx1 *= bp.scale_to_width / orig_w
    by1 *= bp.scale_to_height / orig_h
    bx2 *= bp.scale_to_width / orig_w
    by2 *= bp.scale_to_height / orig_h
    # crop
    bx1 -= bp.crop_x
    by1 -= bp.crop_y
    bx2 -= bp.crop_x
    by2 -= bp.crop_y
    if bx2 <= 0 or by2 <= 0 or bx1 >= bp.crop_width or by1 >= bp.crop_height:
        return None
    # latent decode scale (x0_pixels grid)
    bx1 *= px_w / bp.crop_width
    by1 *= px_h / bp.crop_height
    bx2 *= px_w / bp.crop_width
    by2 *= px_h / bp.crop_height
    # clamp
    bx1 = max(0.0, min(bx1, float(px_w)))
    by1 = max(0.0, min(by1, float(px_h)))
    bx2 = max(0.0, min(bx2, float(px_w)))
    by2 = max(0.0, min(by2, float(px_h)))
    return (bx1, by1, bx2, by2)


def bbox_iou(a: Tuple, b: Tuple) -> float:
    if a is None or b is None:
        return 0.0
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(1e-9, area_a + area_b - inter)


def bbox_center_offset(a: Tuple, b: Tuple) -> float:
    if a is None or b is None:
        return float("inf")
    ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    return math.hypot(ca[0] - cb[0], ca[1] - cb[1])


def draw_bbox(img: Image.Image, bbox: Optional[Tuple], color: str,
              label: str = "") -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    if bbox is None:
        d.text((4, 4), f"{label} NONE", fill=color)
        return out
    d.rectangle(bbox, outline=color, width=3)
    d.text((bbox[0] + 4, max(0, bbox[1] - 12)), label, fill=color)
    return out


def find_face(detector, pil: Image.Image) -> Optional[Tuple[float, float, float, float]]:
    """Return largest face bbox in pil coords, or None."""
    arr = np.asarray(pil.convert("RGB"))
    bgr = arr[:, :, ::-1].copy()
    faces = detector.get(bgr)
    if not faces:
        return None
    # biggest box
    faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                   reverse=True)
    x1, y1, x2, y2 = faces[0].bbox.tolist()
    return (x1, y1, x2, y2)


def list_images(folder: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    out = []
    for f in sorted(os.listdir(folder)):
        if os.path.splitext(f)[1].lower() in exts:
            out.append(os.path.join(folder, f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="number of images to test")
    ap.add_argument("--dir", default=TEST_DIR)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading InsightFace buffalo_l...")
    from insightface.app import FaceAnalysis
    det = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider",
                                                     "CPUExecutionProvider"])
    det.prepare(ctx_id=0, det_size=(640, 640))

    paths = list_images(args.dir)
    if not paths:
        print(f"No images in {args.dir}"); return

    rows = []
    case_idx = 0
    for p in paths:
        if case_idx >= args.n:
            break
        try:
            raw = exif_transpose(Image.open(p)).convert("RGB")
        except Exception as e:
            print(f"skip {p}: {e}"); continue
        orig_w, orig_h = raw.size
        raw_bbox = find_face(det, raw)
        if raw_bbox is None:
            continue
        bp = compute_bucket_params(orig_w, orig_h)
        stem = os.path.splitext(os.path.basename(p))[0]

        for flip_x in (False, True):
            train_pil = apply_dataloader_transform(raw, flip_x, bp)
            px_w, px_h = train_pil.size  # x0_pixels shape matches crop in practice

            predicted = transform_bbox_sdtrainer(raw_bbox, orig_w, orig_h, bp, px_w, px_h,
                                                 flip_x=flip_x)
            detected = find_face(det, train_pil)

            iou = bbox_iou(predicted, detected)
            coff = bbox_center_offset(predicted, detected)
            rows.append((stem[:30], "flip" if flip_x else "noflip", iou, coff,
                         predicted is not None, detected is not None))

            # composite: raw+raw_bbox | training+predicted(green) | training+detected(red)
            raw_vis = draw_bbox(raw, raw_bbox, "yellow", "raw")
            train_pred = draw_bbox(train_pil, predicted, "lime", "predicted")
            train_det = draw_bbox(train_pil, detected, "red", "detected")
            # normalize visualization widths for display
            target_h = 512
            def fit(im):
                r = target_h / im.height
                return im.resize((int(im.width * r), target_h), Image.BICUBIC)
            panels = [fit(raw_vis), fit(train_pred), fit(train_det)]
            total_w = sum(p.width for p in panels)
            combo = Image.new("RGB", (total_w, target_h), (0, 0, 0))
            x = 0
            for pnl in panels:
                combo.paste(pnl, (x, 0)); x += pnl.width
            d = ImageDraw.Draw(combo)
            tag = f"flip_x={flip_x}  IoU={iou:.3f}  dC={coff:.1f}px"
            d.text((4, 4), tag, fill="white")
            combo.save(os.path.join(OUT_DIR, f"{stem}_{'flip' if flip_x else 'noflip'}.jpg"))
        case_idx += 1

    print(f"\nResults (outputs in {OUT_DIR}):\n")
    print(f"{'image':32s} {'mode':6s} {'IoU':>6s} {'dCenter':>8s}  pred? det?")
    for r in rows:
        print(f"{r[0]:32s} {r[1]:6s} {r[2]:6.3f} {r[3]:8.1f}  {str(r[4]):>5s} {str(r[5]):>5s}")

    if rows:
        noflip = [r for r in rows if r[1] == "noflip" and r[4] and r[5]]
        flip = [r for r in rows if r[1] == "flip" and r[4] and r[5]]
        if noflip:
            print(f"\nmean IoU  noflip: {sum(r[2] for r in noflip)/len(noflip):.3f}")
        if flip:
            print(f"mean IoU    flip: {sum(r[2] for r in flip)/len(flip):.3f}")
        print("\nInterpretation:")
        print("  noflip mean IoU near 1 → scale+crop math is correct")
        print("  flip   mean IoU near 0 → confirms missing flip bug")
        print("  flip   mean IoU near 1 → flip already handled somewhere upstream")


if __name__ == "__main__":
    main()
