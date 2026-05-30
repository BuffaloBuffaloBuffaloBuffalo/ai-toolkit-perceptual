#!/usr/bin/env python3
"""Diagnose ViTPose keypoint extraction: dsnt(raw) vs dsnt(softmax) vs argmax.

The encoder does `dsntnn.dsnt(heatmaps)` on the RAW ViTPose heatmaps. dsnt is a
soft-argmax that is only valid on a normalized (flat_softmax) heatmap; on raw
heatmaps the centroid is wrong. This compares, in heatmap space, the current
method against the true peak (argmax) and against the correct dsnt(flat_softmax),
and renders all three on the warped ViTPose input.

Output: _debug_video_perceptor/vitpose_keypoints.png
Usage: python scripts/debug_vitpose_keypoints.py [image_or_video]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image, ImageDraw

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT = "test_data/dance_clips/man_dancing_square_16fps_4s.mp4"
OUT = "_debug_video_perceptor/vitpose_keypoints.png"


def load_clean(path, size=512):
    if path.lower().endswith((".mp4", ".mov", ".avi", ".webm")):
        import cv2
        cap = cv2.VideoCapture(path)
        ok, fr = cap.read()
        cap.release()
        assert ok, path
        return Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).resize((size, size))
    return Image.open(path).convert("RGB").resize((size, size))


def draw_kp(img, kp01, color):
    """kp01: (17,2) normalized [0,1] (x,y). Draws dots + index."""
    d = ImageDraw.Draw(img)
    w, h = img.size
    for i in range(kp01.shape[0]):
        x, y = float(kp01[i, 0]) * w, float(kp01[i, 1]) * h
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color, outline=color)
    return img


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    import dsntnn
    from toolkit.body_id import DifferentiableBodyProportionEncoder

    enc = DifferentiableBodyProportionEncoder().to(DEVICE).eval()
    pil = load_clean(path)
    W, H = pil.size
    inputs = enc.processor(images=pil, boxes=[[[0.0, 0.0, float(W), float(H)]]], return_tensors="pt")
    pv = inputs["pixel_values"].to(device=next(enc.model.parameters()).device,
                                   dtype=next(enc.model.parameters()).dtype)  # (1,3,256,192)
    with torch.no_grad():
        heatmaps = enc.model(pv, dataset_index=torch.tensor([0], device=pv.device)).heatmaps.float()
    _, K, hh, hw = heatmaps.shape
    print(f"heatmaps {tuple(heatmaps.shape)}  min={heatmaps.min():.3f} max={heatmaps.max():.3f} "
          f"mean={heatmaps.mean():.4f}  per-kp sum mean={heatmaps.sum(dim=(2,3)).mean():.3f} "
          f"(=1.0 would mean already normalized)")

    # --- argmax (true peak), normalized [0,1] (x,y) ---
    flat = heatmaps.view(K, -1).argmax(dim=1)
    ay, ax = (flat // hw).float(), (flat % hw).float()
    argmax01 = torch.stack([ax / (hw - 1), ay / (hh - 1)], dim=-1)  # (K,2)

    # --- dsnt on RAW heatmaps (the current code) ---
    raw = dsntnn.dsnt(heatmaps)[0]  # (K,2) in [-1,1] (x,y)
    raw01 = (raw + 1) / 2

    # --- dsnt on flat_softmax heatmaps (the correct usage) ---
    soft = dsntnn.dsnt(dsntnn.flat_softmax(heatmaps))[0]
    soft01 = (soft + 1) / 2

    def px_dist(a, b):  # mean keypoint distance in 256x192 input pixels
        d = (a - b).clone()
        d[:, 0] *= 192
        d[:, 1] *= 256
        return d.pow(2).sum(-1).sqrt().mean().item()

    # integral regression: clamp>=0, normalize by sum, then dsnt (standard for
    # Gaussian heatmaps that were NOT trained with the dsnt softmax objective)
    hm_pos = heatmaps.clamp(min=0)
    norm = dsntnn.dsnt(hm_pos / hm_pos.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6))[0]
    norm01 = (norm + 1) / 2

    print(f"\nmean keypoint error vs argmax peak (pixels in 256x192 input):")
    print(f"  dsnt(RAW)            = {px_dist(raw01.cpu(), argmax01.cpu()):.1f} px   <- current code")
    print(f"  dsnt(flat_softmax)   = {px_dist(soft01.cpu(), argmax01.cpu()):.1f} px")
    for T in (10.0, 30.0, 100.0):
        st = dsntnn.dsnt(dsntnn.flat_softmax(heatmaps * T))[0]
        print(f"  dsnt(softmax*{T:>5g})   = {px_dist(((st+1)/2).cpu(), argmax01.cpu()):.1f} px")
    print(f"  dsnt(norm-by-sum)    = {px_dist(norm01.cpu(), argmax01.cpu()):.1f} px   <- integral regression")

    # --- render the warped input with all three overlays ---
    mean = enc._img_mean.float().cpu().view(3, 1, 1)
    std = enc._img_std.float().cpu().view(3, 1, 1)
    warped = (pv[0].float().cpu() * std + mean).clamp(0, 1)
    base = Image.fromarray((warped.permute(1, 2, 0) * 255).byte().numpy()).resize((192 * 2, 256 * 2))

    def panel(kp01, title, color):
        im = base.copy()
        draw_kp(im, kp01.cpu().numpy(), color)
        ImageDraw.Draw(im).text((4, 4), title, fill=color)
        return im

    panels = [panel(argmax01, "argmax (peak)", (0, 128, 255)),
              panel(norm01, "norm-by-sum FIX", (0, 255, 0)),
              panel(raw01, "dsnt(raw) OLD/BROKEN", (255, 0, 0))]
    w = sum(p.width for p in panels) + 6
    h = max(p.height for p in panels)
    out = Image.new("RGB", (w, h), (15, 15, 15))
    x = 0
    for p in panels:
        out.paste(p, (x, 0)); x += p.width + 3
    out.save(OUT)
    print(f"\nwrote {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
