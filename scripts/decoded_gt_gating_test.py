#!/usr/bin/env python3
"""Validate the 'decoded GT + detector gate' redesign of the identity video loss.

Two claims to check, on the real dance clip (TAEHV-LTX round-trip = the decoder
the loss uses), at 1024²:

  1. CALIBRATION: with GT from the ORIGINAL sharp frame, a perfect reconstruction
     (the decoded true latent) only scores ~0.2 vs GT — an unreachable target. With
     GT from the DECODED frame, a perfect reconstruction scores ~1.0 — reachable.
  2. GATE + DISCRIMINATION: the face detector's confidence on the DECODED frame lets
     us drop frames where the codec destroyed the face; on the frames it keeps, a
     different identity still scores lower than self (loss has signal).

Usage: python scripts/decoded_gt_gating_test.py [video.mp4] [personB.jpg]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16
SIZE = 1024
NF = 9
DEFAULT_VIDEO = "test_data/dance_clips/man_dancing_square_16fps_4s.mp4"


class _Item:
    def __init__(self, path, size=SIZE):
        self.path = path
        self.is_video = True
        self.flip_x = self.flip_y = False
        self.scale_to_width = self.scale_to_height = size
        self.crop_x = self.crop_y = 0
        self.crop_width = self.crop_height = size
        self.width = self.height = size


def to_pil(chw):
    return Image.fromarray((chw.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy())


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    personB = sys.argv[2] if len(sys.argv) > 2 else sorted(glob.glob("test_data/scarlett_full/*.jpg"))[0]

    from toolkit.depth_consistency import decode_wan_x0_to_frames, load_taehv_ltx2
    from toolkit.face_id import DifferentiableFaceEncoder, FaceIDExtractor
    from toolkit.video_frames import read_video_frames_with_transform

    ext = FaceIDExtractor(model_name="buffalo_l")
    fenc = DifferentiableFaceEncoder().to(DEVICE)
    taeltx = load_taehv_ltx2(device=str(DEVICE), dtype=DTYPE, version="2.3")

    def roundtrip(frames):
        with torch.no_grad():
            lat = taeltx.encode_video(frames.unsqueeze(0).to(DEVICE, DTYPE), parallel=True, show_progress_bar=False)
            return decode_wan_x0_to_frames(lat.permute(0, 2, 1, 3, 4).contiguous().float(), taeltx)[0].permute(1, 0, 2, 3)

    def detect(pil):
        """Return (det_score, bbox) of the largest face, or (0.0, None)."""
        faces, _ = ext._detect(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
        if not faces:
            return 0.0, None
        f = ext._get_largest_face(faces)
        return float(f.det_score), f.bbox.astype(np.float32)

    def emb(frame_chw, bbox):
        with torch.no_grad():
            bb = [[float(v) for v in bbox]] if bbox is not None else [None]
            return fenc(frame_chw.unsqueeze(0).to(DEVICE), bboxes=bb, return_crops=False)

    # --- person A: the clip ---
    A_orig = read_video_frames_with_transform(_Item(video), num_frames=NF)
    A_dec = roundtrip(A_orig)
    T = A_dec.shape[0]

    print(f"size={SIZE} latent=32x32 frames={T}\n")
    print(f"{'frame':>5} | {'det(orig)':>9} | {'det(decoded)':>12} | {'cos(dec, ORIG-GT)':>17} | {'cos(dec, DECODED-GT)':>20}")
    print("-" * 78)
    old_gaps, det_dec = [], []
    A_dec_emb = []
    for t in range(T):
        op, rp = to_pil(A_orig[min(t, A_orig.shape[0] - 1)]), to_pil(A_dec[t])
        ds_o, bb_o = detect(op)
        ds_r, bb_r = detect(rp)
        det_dec.append((t, ds_r, bb_r))
        # old GT = ArcFace(original frame, original bbox); gen = ArcFace(decoded frame, original bbox)
        if bb_o is not None:
            e_origGT = emb(A_orig[min(t, A_orig.shape[0] - 1)], bb_o)
            e_dec_obb = emb(A_dec[t], bb_o)
            old = F.cosine_similarity(e_origGT, e_dec_obb, dim=-1).item()
        else:
            old = float("nan")
        # new GT = ArcFace(decoded frame, decoded bbox); a perfect recon reuses it → 1.0
        if bb_r is not None:
            e_decGT = emb(A_dec[t], bb_r)
            A_dec_emb.append((t, e_decGT))
            new = F.cosine_similarity(e_decGT, e_decGT, dim=-1).item()
        else:
            new = float("nan")
        if not np.isnan(old):
            old_gaps.append(old)
        print(f"{t:>5} | {ds_o:>9.3f} | {ds_r:>12.3f} | {old:>17.3f} | {new:>20.3f}")

    print(f"\n[1] CALIBRATION  old target gap mean = {np.nanmean(old_gaps):.3f} (unreachable; loss floor ~{1-np.nanmean(old_gaps):.2f})")
    print(f"                 new target            = 1.000 (reachable)")

    print("\n[2] DETECTOR GATE on decoded frames")
    for thr in (0.4, 0.5, 0.6, 0.7):
        kept = sum(1 for _, ds, _ in det_dec if ds >= thr)
        print(f"     det_thresh {thr:.1f}: keep {kept}/{T} frames")

    # discrimination: person B (different identity), decoded
    B_img = Image.open(personB).convert("RGB").resize((SIZE, SIZE))
    B_frames = torch.from_numpy(np.array(B_img)).permute(2, 0, 1).float().div(255).unsqueeze(0).repeat(NF, 1, 1, 1)
    B_dec = roundtrip(B_frames)
    ds_b, bb_b = detect(to_pil(B_dec[0]))
    print(f"\n[3] DISCRIMINATION on decoded faces (personB={os.path.basename(personB)}, det={ds_b:.2f})")
    if bb_b is not None and A_dec_emb:
        e_b = emb(B_dec[0], bb_b)
        # self: A decoded frame vs A decoded GT (other frame) — same identity
        if len(A_dec_emb) >= 2:
            self_cos = F.cosine_similarity(A_dec_emb[0][1], A_dec_emb[1][1], dim=-1).item()
            print(f"     A(dec) vs A(dec, other frame) [same id]   = {self_cos:.3f}")
        cross = np.mean([F.cosine_similarity(e_b, e, dim=-1).item() for _, e in A_dec_emb])
        print(f"     B(dec) vs A(dec) [different id, mean]      = {cross:.3f}")
    else:
        print("     (no detectable face to compare)")


if __name__ == "__main__":
    main()
