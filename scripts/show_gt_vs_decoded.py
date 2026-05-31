#!/usr/bin/env python3
"""Side-by-side GT vs TAEHV-decoded, full frame + face crop, labeled.

Renders, at 1024² (best case for the tiny decoder), for a few frames:
    [ GT frame | GT face crop | decoded frame | decoded face crop ]
so the reconstruction quality (whole frame vs the small face) is unambiguous.

Output: _debug_video_perceptor/gt_vs_decoded.png
Usage: python scripts/show_gt_vs_decoded.py [video.mp4] [size]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16
DEFAULT_VIDEO = "test_data/dance_clips/man_dancing_square_16fps_4s.mp4"
OUT = "_debug_video_perceptor/gt_vs_decoded.png"
FRAME_W = 384
CROP_W = 256


class _Item:
    def __init__(self, path, size):
        self.path = path
        self.is_video = True
        self.flip_x = self.flip_y = False
        self.scale_to_width = size
        self.scale_to_height = size
        self.crop_x = self.crop_y = 0
        self.crop_width = self.crop_height = size
        self.width = self.height = size


def to_pil(chw):
    return Image.fromarray((chw.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy())


def label(pil, text):
    out = Image.new("RGB", (pil.width, pil.height + 22), (15, 15, 15))
    out.paste(pil, (0, 22))
    ImageDraw.Draw(out).text((4, 5), text, fill=(0, 255, 0))
    return out


def crop_face(pil, bbox, w=CROP_W):
    if bbox is None:
        return Image.new("RGB", (w, w), (60, 0, 0))  # dark red = no face
    return pil.crop([float(v) for v in bbox]).resize((w, w), Image.NEAREST)


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    assert os.path.exists(video), video

    from toolkit.depth_consistency import decode_wan_x0_to_frames, load_taehv_ltx2
    from toolkit.face_id import FaceIDExtractor
    from toolkit.video_frames import read_video_frames_with_transform

    frames = read_video_frames_with_transform(_Item(video, size), num_frames=9)
    taeltx = load_taehv_ltx2(device=str(DEVICE), dtype=DTYPE, version="2.3")
    with torch.no_grad():
        lat = taeltx.encode_video(frames.unsqueeze(0).to(DEVICE, DTYPE), parallel=True, show_progress_bar=False)
        rec = decode_wan_x0_to_frames(lat.permute(0, 2, 1, 3, 4).contiguous().float(), taeltx)[0].permute(1, 0, 2, 3)
    print(f"size={size}  latent HxW={tuple(lat.shape[-2:])}  recon T={rec.shape[0]}")

    ext = FaceIDExtractor(model_name="buffalo_l")
    rows = []
    for t in [0, 2, 4]:
        op, rp = to_pil(frames[t]), to_pil(rec[min(t, rec.shape[0] - 1)])
        _, obb = ext.extract_with_bbox(cv2.cvtColor(np.array(op), cv2.COLOR_RGB2BGR))
        cells = [
            label(op.resize((FRAME_W, FRAME_W)), f"f{t} GT frame"),
            label(crop_face(op, obb), f"f{t} GT face {None if obb is None else f'{int(obb[2]-obb[0])}px'}"),
            label(rp.resize((FRAME_W, FRAME_W)), f"f{t} DECODED frame"),
            label(crop_face(rp, obb), f"f{t} DECODED face (GT box)"),
        ]
        h = max(c.height for c in cells)
        w = sum(c.width for c in cells) + 2 * (len(cells) + 1)
        row = Image.new("RGB", (w, h + 4), (15, 15, 15))
        x = 2
        for c in cells:
            row.paste(c, (x, 2)); x += c.width + 2
        rows.append(row)

    W = max(r.width for r in rows)
    H = sum(r.height for r in rows) + 2 * (len(rows) + 1)
    out = Image.new("RGB", (W, H), (15, 15, 15))
    y = 2
    for r in rows:
        out.paste(r, (2, y)); y += r.height + 2
    out.save(OUT)
    print(f"wrote {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
