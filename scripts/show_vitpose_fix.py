#!/usr/bin/env python3
"""Validate the fixed forward() (loss path) against the repo's correct ViTPose
convention: processor.post_process_pose_estimation (used by draw_skeletons.py and
the SDTrainer body-proportion preview, "correct coords").

Runs the encoder's REAL forward() (manual affine warp + keypoint decode) and maps
its keypoints back to image space, for the FIX (integral regression) and the OLD
(raw dsnt, via monkeypatch), and overlays both against the post_process reference.

Output: _debug_video_perceptor/vitpose_fix_onimage.png  (+ pixel distances)
Usage: python scripts/show_vitpose_fix.py [image_or_video]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image, ImageDraw

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT = "test_data/dance_clips/man_dancing_square_16fps_4s.mp4"
OUT = "_debug_video_perceptor/vitpose_fix_onimage.png"
SIZE = 512


def load_clean(path, size=SIZE):
    if path.lower().endswith((".mp4", ".mov", ".avi", ".webm")):
        import cv2
        cap = cv2.VideoCapture(path)
        ok, fr = cap.read()
        cap.release()
        assert ok, path
        return Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).resize((size, size))
    return Image.open(path).convert("RGB").resize((size, size))


def compute_theta(s_w, s_h, out_w, out_h):
    """Replicate forward()'s warp → theta (output-norm → input-norm)."""
    from transformers.models.vitpose.image_processing_vitpose import (
        box_to_center_and_scale, get_warp_matrix)
    center, scale = box_to_center_and_scale([0, 0, s_w, s_h], out_w, out_h,
                                            normalize_factor=200.0, padding_factor=1.25)
    warp_mat = get_warp_matrix(0, center * 2.0,
                               np.array([out_w - 1, out_h - 1], dtype=np.float32), scale * 200.0)
    M = np.vstack([warp_mat, [0, 0, 1]])
    S_in = np.array([[2.0 / (s_w - 1), 0, -1], [0, 2.0 / (s_h - 1), -1], [0, 0, 1]])
    S_out_inv = np.array([[(out_w - 1) / 2.0, 0, (out_w - 1) / 2.0],
                          [0, (out_h - 1) / 2.0, (out_h - 1) / 2.0], [0, 0, 1]])
    return (S_in @ np.linalg.inv(M) @ S_out_inv)[:2, :]


def warped01_to_imgpx(kp01, theta, s_w, s_h):
    out = []
    for x, y in kp01:
        ix, iy = theta @ np.array([2 * float(x) - 1, 2 * float(y) - 1, 1.0])
        out.append([(ix + 1) / 2 * (s_w - 1), (iy + 1) / 2 * (s_h - 1)])
    return np.array(out, dtype=np.float32)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    import dsntnn
    from toolkit.body_id import DifferentiableBodyProportionEncoder, draw_skeleton_overlay

    enc = DifferentiableBodyProportionEncoder().to(DEVICE).eval()
    pil = load_clean(path)
    W, H = pil.size
    mdev = next(enc.model.parameters()).device
    mdt = next(enc.model.parameters()).dtype

    # --- REFERENCE: processor.post_process_pose_estimation (repo's correct convention) ---
    boxes = [[[0.0, 0.0, float(W), float(H)]]]
    inputs = enc.processor(images=pil, boxes=boxes, return_tensors="pt")
    with torch.no_grad():
        outputs = enc.model(pixel_values=inputs["pixel_values"].to(mdev, mdt),
                            dataset_index=torch.tensor([0], device=mdev))
        outputs.heatmaps = outputs.heatmaps.float()
        ref = enc.processor.post_process_pose_estimation(outputs, boxes=boxes)[0][0]
    ref_kp = np.asarray(ref["keypoints"], dtype=np.float32)  # (17,2) image px
    vis = np.asarray(ref["scores"], dtype=np.float32)

    # --- run the REAL forward() (manual warp + decode) for FIX and OLD ---
    img_t = torch.from_numpy(np.asarray(pil)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(DEVICE)
    theta = compute_theta(W, H, enc.INPUT_SIZE[1], enc.INPUT_SIZE[0])

    with torch.no_grad():
        enc(img_t, include_head=False)                       # FIX (integral regression)
    fix_px = warped01_to_imgpx(enc._last_keypoints[0].cpu().numpy(), theta, W, H)

    _orig = DifferentiableBodyProportionEncoder._heatmaps_to_coords
    DifferentiableBodyProportionEncoder._heatmaps_to_coords = staticmethod(lambda hm: dsntnn.dsnt(hm))
    try:
        with torch.no_grad():
            enc(img_t, include_head=False)                   # OLD (raw dsnt)
        old_px = warped01_to_imgpx(enc._last_keypoints[0].cpu().numpy(), theta, W, H)
    finally:
        DifferentiableBodyProportionEncoder._heatmaps_to_coords = _orig

    d_fix = float(np.linalg.norm(fix_px - ref_kp, axis=1).mean())
    d_old = float(np.linalg.norm(old_px - ref_kp, axis=1).mean())
    print(f"image {W}x{H}; mean keypoint distance vs post_process_pose_estimation (correct convention):")
    print(f"  forward() integral-regression FIX = {d_fix:6.1f} px")
    print(f"  forward() raw-dsnt OLD            = {d_old:6.1f} px")

    def panel(kp_px, title, color):
        img = draw_skeleton_overlay(pil.copy(), kp_px / np.array([W, H]), vis)
        ImageDraw.Draw(img).rectangle([0, 0, W - 1, 20], fill=(0, 0, 0))
        ImageDraw.Draw(img).text((4, 4), title, fill=color)
        return img

    panels = [panel(ref_kp, "post_process (CONVENTION)", (0, 200, 255)),
              panel(fix_px, f"forward() FIX  {d_fix:.0f}px", (0, 255, 0)),
              panel(old_px, f"forward() OLD  {d_old:.0f}px", (255, 0, 0))]
    out = Image.new("RGB", (sum(p.width for p in panels) + 6, max(p.height for p in panels)), (15, 15, 15))
    x = 0
    for p in panels:
        out.paste(p, (x, 0)); x += p.width + 3
    out.save(OUT)
    print(f"wrote {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
