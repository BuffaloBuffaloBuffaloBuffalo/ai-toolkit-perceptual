#!/usr/bin/env python3
"""Decode a clean TAEHV-LTX latent of a real clip and dump frames + bbox crops.

Saves PNG montages so we can eyeball (a) the codec reconstruction quality and
(b) whether the detected face bbox actually lands on the face in BOTH the
original and the decoded frame — to tell a real bug from plain codec blur.

Outputs to _debug_video_perceptor/:
  frames_montage.png   rows = sampled frames; cols = [orig | orig+bbox | recon | recon+bbox]
  arcface_crops.png    rows = sampled frames; cols = [orig crop | recon crop]  (the bbox region)

Usage: python scripts/debug_decode_clean_latent.py [video.mp4]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16
NUM_FRAMES = 17
SIZE = 512
DEFAULT_VIDEO = "test_data/dance_clips/man_dancing_square_16fps_4s.mp4"
OUT = "_debug_video_perceptor"
SAMPLES = [0, 8, 16]  # which frames to visualize


class _Item:
    def __init__(self, path):
        self.path = path
        self.is_video = True
        self.flip_x = self.flip_y = False
        self.scale_to_width = SIZE
        self.scale_to_height = SIZE
        self.crop_x = 0
        self.crop_y = 0
        self.crop_width = SIZE
        self.crop_height = SIZE
        self.width = self.height = SIZE


def to_pil(frame_chw):  # (3,H,W) [0,1] → PIL RGB
    return Image.fromarray((frame_chw.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy())


def draw_bbox(pil, bbox):
    pil = pil.copy()
    if bbox is not None:
        d = ImageDraw.Draw(pil)
        d.rectangle([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    outline=(0, 255, 0), width=4)
    return pil


def hstack(pils):
    h = max(p.height for p in pils)
    w = sum(p.width for p in pils)
    out = Image.new("RGB", (w, h), (20, 20, 20))
    x = 0
    for p in pils:
        out.paste(p, (x, 0))
        x += p.width
    return out


def vstack(pils):
    w = max(p.width for p in pils)
    h = sum(p.height for p in pils)
    out = Image.new("RGB", (w, h), (20, 20, 20))
    y = 0
    for p in pils:
        out.paste(p, (0, y))
        y += p.height
    return out


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    assert os.path.exists(video), video
    os.makedirs(OUT, exist_ok=True)
    print(f"video: {video}  device: {DEVICE}")

    from toolkit.depth_consistency import decode_wan_x0_to_frames, load_taehv_ltx2
    from toolkit.face_id import DifferentiableFaceEncoder, FaceIDExtractor
    from toolkit.video_frames import read_video_frames_with_transform

    frames = read_video_frames_with_transform(_Item(video), num_frames=NUM_FRAMES)  # (T,3,512,512)
    print(f"frames: {tuple(frames.shape)}")

    # Clean latent round-trip via the same TAEHV-LTX codec the perceptors use.
    taeltx = load_taehv_ltx2(device=str(DEVICE), dtype=DTYPE, version="2.3")
    x_ntchw = frames.unsqueeze(0).to(DEVICE, DTYPE)
    with torch.no_grad():
        lat = taeltx.encode_video(x_ntchw, parallel=True, show_progress_bar=False)
        lat_ncthw = lat.permute(0, 2, 1, 3, 4).contiguous().float()
        rec = decode_wan_x0_to_frames(lat_ncthw, taeltx)  # (1,3,T_out,512,512)
    rec = rec[0].permute(1, 0, 2, 3)  # (T_out, 3, 512, 512)
    print(f"latent: {tuple(lat_ncthw.shape)}  recon: {tuple(rec.shape)}  range=[{rec.min():.2f},{rec.max():.2f}]")
    T_out = rec.shape[0]

    extractor = FaceIDExtractor(model_name="buffalo_l")
    enc = DifferentiableFaceEncoder().to(DEVICE)

    rows_frames, rows_crops = [], []
    import cv2
    for fi in SAMPLES:
        oi = min(fi, frames.shape[0] - 1)
        ri = min(fi, T_out - 1)
        opil = to_pil(frames[oi])
        rpil = to_pil(rec[ri])

        # Detect bbox on BOTH original and recon (independently) to compare.
        obgr = cv2.cvtColor(np.array(opil), cv2.COLOR_RGB2BGR)
        rbgr = cv2.cvtColor(np.array(rpil), cv2.COLOR_RGB2BGR)
        _, obbox = extractor.extract_with_bbox(obgr)
        _, rbbox = extractor.extract_with_bbox(rbgr)

        rows_frames.append(hstack([
            opil, draw_bbox(opil, obbox), rpil, draw_bbox(rpil, rbbox),
        ]))

        # The bbox region each side feeds to ArcFace (use the ORIGINAL bbox on both,
        # which is what the loss does — GT bbox applied to the recon frame).
        def crop112(pil, bbox):
            if bbox is None:
                c = pil.resize((112, 112))
            else:
                c = pil.crop((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))).resize((112, 112))
            return c

        rows_crops.append(hstack([crop112(opil, obbox), crop112(rpil, obbox)]))

        # Isolated cosines for this frame.
        with torch.no_grad():
            ft = frames[oi].unsqueeze(0).to(DEVICE)
            rt = rec[ri].unsqueeze(0).to(DEVICE)
            obb = [[float(v) for v in obbox]] if obbox is not None else [None]
            e_o = enc(ft, bboxes=obb, return_crops=False)
            e_r = enc(rt, bboxes=obb, return_crops=False)  # ORIGINAL bbox on recon (loss behavior)
            e_r_own = enc(rt, bboxes=([[float(v) for v in rbbox]] if rbbox is not None else [None]), return_crops=False)
        cos_codec = F.cosine_similarity(e_o, e_r, dim=-1).item()
        cos_codec_ownbb = F.cosine_similarity(e_o, e_r_own, dim=-1).item()
        ob = None if obbox is None else [round(float(v)) for v in obbox]
        rb = None if rbbox is None else [round(float(v)) for v in rbbox]
        print(f"  frame {fi}: orig_bbox={ob} recon_bbox={rb}  "
              f"cos(orig,recon|origbbox)={cos_codec:.3f}  cos(orig,recon|reconbbox)={cos_codec_ownbb:.3f}")

    vstack(rows_frames).save(os.path.join(OUT, "frames_montage.png"))
    vstack(rows_crops).save(os.path.join(OUT, "arcface_crops.png"))
    print(f"\nwrote {OUT}/frames_montage.png  and  {OUT}/arcface_crops.png")


if __name__ == "__main__":
    main()
