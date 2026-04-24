#!/usr/bin/env python3
"""Real-pipeline validation of depth-GT alignment (v1 vs v2).

For each test image:
  1. Compute bucket params via toolkit.buckets.get_bucket_for_image_size.
  2. Apply dataloader transforms (flip + resize + crop) → training PIL.
  3. v1 simulation: DA2(raw PIL) at raw resolution. At training time the
     cached depth is F.interpolate'd to the pred grid — we simulate that to
     a common shape for apples-to-apples comparison.
  4. v2 real: call cache_depth_gt_embeddings with the modified function on a
     mock FileItemDTO carrying bucket params.
  5. Reference: DA2(training PIL) directly — the depth the loss SHOULD see.
  6. Report SSI-L1 between (v1, reference) and (v2, reference).
  7. Save 4-panel overlay to output/depth_validation/.

Runs for flip_x in (False, True).

Usage: python scripts/validate_depth_alignment.py [--n 3]
"""

import argparse
import math
import os
import shutil
import sys
import tempfile
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from PIL.ImageOps import exif_transpose

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from toolkit.buckets import get_bucket_for_image_size  # noqa: E402
from toolkit.config_modules import DepthConsistencyConfig  # noqa: E402
from toolkit.depth_consistency import (  # noqa: E402
    DifferentiableDepthEncoder, cache_depth_gt_embeddings, ssi_l1,
)

warnings.filterwarnings("ignore")

TEST_DIR = "test_data/scarlett_full"
OUT_DIR = "output/depth_validation"
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
    path: str
    scale_to_width: int
    scale_to_height: int
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int
    flip_x: bool = False
    flip_y: bool = False
    depth_gt: Optional[torch.Tensor] = None


def compute_bucket_params(orig_w: int, orig_h: int) -> BucketParams:
    br = get_bucket_for_image_size(orig_w, orig_h, resolution=RESOLUTION,
                                   divisibility=BUCKET_TOLERANCE)
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


def da2_of_pil(enc: DifferentiableDepthEncoder, pil: Image.Image,
               device: torch.device) -> torch.Tensor:
    arr = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0)\
        .permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        d = enc(arr)[0].float().cpu()
    return d


def depth_to_pil(d: torch.Tensor, size_wh: Tuple[int, int]) -> Image.Image:
    """Percentile-normalize depth to grayscale PIL, resize to size."""
    a = d.cpu().numpy()
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    a = np.clip((a - lo) / max(1e-6, hi - lo), 0, 1)
    return Image.fromarray((a * 255).astype(np.uint8)).resize(size_wh, Image.BICUBIC).convert("RGB")


def list_images(folder: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted([os.path.join(folder, f) for f in os.listdir(folder)
                   if os.path.splitext(f)[1].lower() in exts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--dir", default=TEST_DIR)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    paths = list_images(args.dir)[: args.n]
    if not paths:
        print(f"No images in {args.dir}"); return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print("Loading DA2-Small...")
    enc = DifferentiableDepthEncoder(grad_checkpoint=False, device=device)
    cfg = DepthConsistencyConfig(loss_weight=0.05)

    rows = []
    for p in paths:
        raw = exif_transpose(Image.open(p)).convert("RGB")
        orig_w, orig_h = raw.size
        bp = compute_bucket_params(orig_w, orig_h)
        stem = os.path.splitext(os.path.basename(p))[0]

        for flip_x in (False, True):
            train_pil = apply_dataloader_transform(raw, flip_x, bp)

            # Reference: DA2 on training tensor directly.
            d_ref = da2_of_pil(enc, train_pil, device)

            # v1 simulation: DA2(raw), then F.interpolate to d_ref shape.
            d_v1_raw = da2_of_pil(enc, raw, device)
            d_v1 = F.interpolate(d_v1_raw.unsqueeze(0).unsqueeze(0).float(),
                                 size=d_ref.shape, mode="bilinear",
                                 align_corners=True).squeeze(0).squeeze(0)

            # v2 real: call cache_depth_gt_embeddings on a mock file_item in tmpdir.
            with tempfile.TemporaryDirectory(prefix="depth_v2_") as tmp:
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
                cache_depth_gt_embeddings([item], cfg, device=device)
                d_v2 = item.depth_gt.float()
                if d_v2.shape != d_ref.shape:
                    d_v2 = F.interpolate(d_v2.unsqueeze(0).unsqueeze(0),
                                         size=d_ref.shape, mode="bilinear",
                                         align_corners=True).squeeze(0).squeeze(0)

            # SSI-L1 of v1 and v2 against reference — loss that training actually sees.
            ssi_v1 = ssi_l1(d_v1, d_ref)[0].item()
            ssi_v2 = ssi_l1(d_v2, d_ref)[0].item()
            rows.append((stem[:28], "flip" if flip_x else "noflip", ssi_v1, ssi_v2))

            # 4-panel: training | v1 depth | v2 depth | ref depth
            size = train_pil.size
            panels = [train_pil, depth_to_pil(d_v1, size), depth_to_pil(d_v2, size),
                      depth_to_pil(d_ref, size)]
            labels = ["training", f"v1 SSI={ssi_v1:.3f}", f"v2 SSI={ssi_v2:.3f}", "reference"]
            target_h = 480
            def fit(im):
                r = target_h / im.height
                return im.resize((int(im.width * r), target_h), Image.BICUBIC)
            panels = [fit(pn) for pn in panels]
            combo = Image.new("RGB", (sum(p.width for p in panels), target_h), (0, 0, 0))
            x = 0
            for pn, lbl in zip(panels, labels):
                combo.paste(pn, (x, 0))
                ImageDraw.Draw(combo).text((x + 6, 6), lbl, fill=(255, 255, 0))
                x += pn.width
            combo.save(os.path.join(OUT_DIR, f"{stem}_{'flip' if flip_x else 'noflip'}.jpg"),
                       quality=88)

    print(f"\nResults (overlays in {OUT_DIR}):\n")
    print(f"{'image':30s} {'mode':6s} {'v1 SSI':>8s} {'v2 SSI':>8s}")
    for r in rows:
        print(f"{r[0]:30s} {r[1]:6s} {r[2]:8.4f} {r[3]:8.4f}")

    if rows:
        nf = [r for r in rows if r[1] == "noflip"]
        fl = [r for r in rows if r[1] == "flip"]
        if nf:
            print(f"\nnoflip mean SSI  v1={sum(r[2] for r in nf)/len(nf):.4f}  "
                  f"v2={sum(r[3] for r in nf)/len(nf):.4f}")
        if fl:
            print(f"flip   mean SSI  v1={sum(r[2] for r in fl)/len(fl):.4f}  "
                  f"v2={sum(r[3] for r in fl)/len(fl):.4f}")
        print("\n(Lower SSI-L1 = closer to reference. v2 should be ~0 if the "
              "fix aligns cache with training tensor exactly.)")


if __name__ == "__main__":
    main()
