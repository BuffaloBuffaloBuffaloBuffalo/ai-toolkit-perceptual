#!/usr/bin/env python3
"""Why is the face so small / does resolution fix identity?  Empirical sweep.

For the same clip at several working resolutions, measures:
  * the detected face size (px and % of frame) at NATIVE res and each working res,
  * the TAEHV-LTX latent spatial size,
  * the ArcFace identity cosine between the ORIGINAL frame and its TAEHV
    encode→decode reconstruction (using the original-frame face bbox on both),
and saves an [orig crop | recon crop] montage per resolution.

This isolates whether the weak identity signal is (a) my 512 downscale + the
tiny TAEHV decoder, recoverable at higher res, or (b) something deeper.

Outputs to _debug_video_perceptor/sweep/.
Usage: python scripts/identity_resolution_sweep.py [video.mp4]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16
NUM_FRAMES = 9
RESOLUTIONS = [512, 768, 1024]
DEFAULT_VIDEO = "test_data/dance_clips/man_dancing_square_16fps_4s.mp4"
OUT = "_debug_video_perceptor/sweep"


class _Item:
    def __init__(self, path, size):
        self.path = path
        self.is_video = True
        self.flip_x = self.flip_y = False
        self.scale_to_width = size
        self.scale_to_height = size
        self.crop_x = 0
        self.crop_y = 0
        self.crop_width = size
        self.crop_height = size
        self.width = self.height = size


def to_pil(chw):
    return Image.fromarray((chw.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy())


def hcat(pils, pad=2):
    h = max(p.height for p in pils)
    w = sum(p.width for p in pils) + pad * (len(pils) + 1)
    out = Image.new("RGB", (w, h + 2 * pad), (20, 20, 20))
    x = pad
    for p in pils:
        out.paste(p, (x, pad)); x += p.width + pad
    return out


def vcat(pils, pad=2):
    w = max(p.width for p in pils)
    h = sum(p.height for p in pils) + pad * (len(pils) + 1)
    out = Image.new("RGB", (w, h), (20, 20, 20))
    y = pad
    for p in pils:
        out.paste(p, (0, y)); y += p.height + pad
    return out


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    assert os.path.exists(video), video
    os.makedirs(OUT, exist_ok=True)

    from toolkit.depth_consistency import decode_wan_x0_to_frames, load_taehv_ltx2
    from toolkit.face_id import DifferentiableFaceEncoder, FaceIDExtractor
    from toolkit.video_frames import read_video_frames_with_transform

    extractor = FaceIDExtractor(model_name="buffalo_l")
    fenc = DifferentiableFaceEncoder().to(DEVICE)
    taeltx = load_taehv_ltx2(device=str(DEVICE), dtype=DTYPE, version="2.3")

    # --- native resolution face size ---
    cap = cv2.VideoCapture(video)
    ok, fr0 = cap.read()
    cap.release()
    nat_h, nat_w = fr0.shape[:2]
    _, nbb = extractor.extract_with_bbox(fr0)
    if nbb is not None:
        fw, fh = nbb[2] - nbb[0], nbb[3] - nbb[1]
        print(f"NATIVE {nat_w}x{nat_h}: face bbox {[round(float(v)) for v in nbb]}  "
              f"face {fw:.0f}x{fh:.0f}px = {100*fw/nat_w:.1f}% of width")
    else:
        print(f"NATIVE {nat_w}x{nat_h}: no face detected")

    print(f"\n{'res':>6} | {'latent HxW':>10} | {'face px':>8} | {'face %':>7} | {'recon faces':>11} | {'identity cosine':>15}")
    print("-" * 78)

    for size in RESOLUTIONS:
        frames = read_video_frames_with_transform(_Item(video, size), num_frames=NUM_FRAMES)  # (T,3,size,size)
        with torch.no_grad():
            lat = taeltx.encode_video(frames.unsqueeze(0).to(DEVICE, DTYPE), parallel=True, show_progress_bar=False)
            lat_ncthw = lat.permute(0, 2, 1, 3, 4).contiguous().float()
            rec = decode_wan_x0_to_frames(lat_ncthw, taeltx)[0].permute(1, 0, 2, 3)  # (T_out,3,size,size)
        T_out = rec.shape[0]
        Hl, Wl = lat_ncthw.shape[-2:]

        cos_list, face_px, recon_found, crop_rows = [], [], 0, []
        for t in range(T_out):
            op = to_pil(frames[min(t, frames.shape[0] - 1)])
            rp = to_pil(rec[t])
            _, obb = extractor.extract_with_bbox(cv2.cvtColor(np.array(op), cv2.COLOR_RGB2BGR))
            _, rbb = extractor.extract_with_bbox(cv2.cvtColor(np.array(rp), cv2.COLOR_RGB2BGR))
            if rbb is not None:
                recon_found += 1
            if obb is None:
                continue
            face_px.append(float(obb[2] - obb[0]))
            with torch.no_grad():
                bb = [[float(v) for v in obb]]
                eo = fenc(frames[min(t, frames.shape[0] - 1)].unsqueeze(0).to(DEVICE), bboxes=bb, return_crops=False)
                er = fenc(rec[t].unsqueeze(0).to(DEVICE), bboxes=bb, return_crops=False)
            cos_list.append(F.cosine_similarity(eo, er, dim=-1).item())
            if t < 4:
                oc = op.crop([float(v) for v in obb]).resize((112, 112))
                rc = rp.crop([float(v) for v in obb]).resize((112, 112))
                crop_rows.append(hcat([oc, rc]))

        mean_cos = float(np.mean(cos_list)) if cos_list else float("nan")
        mean_px = float(np.mean(face_px)) if face_px else float("nan")
        print(f"{size:>6} | {Hl}x{Wl:<8} | {mean_px:>7.0f} | {100*mean_px/size:>6.1f}% | "
              f"{recon_found:>4}/{T_out:<6} | {mean_cos:>15.3f}")
        if crop_rows:
            vcat(crop_rows).save(f"{OUT}/crops_{size}.png")

    print(f"\nsaved per-resolution [orig|recon] crops to {OUT}/  (crops_512.png, crops_768.png, crops_1024.png)")


if __name__ == "__main__":
    main()
