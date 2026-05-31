#!/usr/bin/env python3
"""Smoke test for the ArcFace identity video path (decoded-GT design).

Validates the WIRED cache_video_identity_embeddings on real face frames WITHOUT
the 22B LTX model. GT is now built from the TAEHV-DECODED clip (the same decoder
the loss uses) and gated by the decoded-face detector score, so:

  1. cache_video_identity_embeddings runs, gates frames, round-trips safetensors.
  2. Reachable target: re-decoding the clip and scoring it against the cached GT
     gives cosine ~1.0 (the old sharp-original GT capped this at ~0.2).
  3. Discrimination: a random target scores far lower.
  4. Gradient flows frames → chunked ArcFace → cosine → frames.

Usage: python scripts/identity_video_smoke.py [faceA.(jpg|png)]
"""
import glob
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16
T_FRAMES = 9
SIZE = 512


def fmt_mem():
    if not torch.cuda.is_available():
        return "cpu"
    return (f"alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
            f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB")


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def make_video(face_path, out_path, n=T_FRAMES, size=SIZE):
    import cv2
    from PIL import Image
    from PIL.ImageOps import exif_transpose

    img = exif_transpose(Image.open(face_path)).convert("RGB").resize((size, size))
    arr = np.array(img)
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 8, (size, size))
    assert vw.isOpened(), "cv2.VideoWriter failed to open"
    rng = np.random.RandomState(0)
    for _ in range(n):
        noisy = np.clip(arr.astype(np.int16) + rng.randint(-3, 4, arr.shape), 0, 255).astype(np.uint8)
        vw.write(cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR))
    vw.release()


class _Item:
    def __init__(self, path):
        self.path = path
        self.is_video = True
        self.flip_x = self.flip_y = False
        self.scale_to_width = self.scale_to_height = SIZE
        self.crop_x = self.crop_y = 0
        self.crop_width = self.crop_height = SIZE
        self.width = self.height = SIZE


class _Cfg:
    face_model = "buffalo_l"
    identity_loss_decoded_det_threshold = 0.5
    identity_loss_frames_per_chunk = 4


def main():
    a = sys.argv[1] if len(sys.argv) > 1 else None
    if a is None:
        faces = sorted(glob.glob("test_data/scarlett_full/*.jpg"))
        assert faces, "need a face jpg under test_data/scarlett_full"
        a = faces[0]
    print(f"face: {a}\ndevice: {DEVICE}")

    from safetensors.torch import load_file

    from toolkit.depth_consistency import decode_wan_x0_to_frames, load_taehv_ltx2
    from toolkit.face_id import (
        CACHE_VERSION_IDENTITY_VIDEO_KEY,
        DifferentiableFaceEncoder,
        cache_video_identity_embeddings,
    )
    from toolkit.video_frames import read_video_frames_with_transform

    tmp = tempfile.mkdtemp(prefix="identity_video_smoke_")
    try:
        vid = os.path.join(tmp, "faceA.mp4")
        make_video(a, vid)

        # --- 1. wired decoded-GT caching + gating ---
        section("1. cache_video_identity_embeddings (decoded GT + detector gate)")
        item = _Item(vid)
        cache_video_identity_embeddings([item], _Cfg(), device=DEVICE, num_frames=T_FRAMES, arch="ltx2.3")
        gt = item.identity_gt_video
        bb = item.identity_gt_video_bbox
        vd = item.identity_gt_video_valid
        assert gt.shape == (T_FRAMES, 512), gt.shape
        assert float(vd.sum()) > 0, "no decoded face survived the gate (close-up should)"
        cache_path = os.path.join(tmp, "_face_id_cache", "faceA.safetensors")
        data = load_file(cache_path)
        assert "identity_gt_video" in data and CACHE_VERSION_IDENTITY_VIDEO_KEY in data
        print(f"  GT {tuple(gt.shape)} gated-valid={int(vd.sum())}/{T_FRAMES} cache OK  {fmt_mem()}")

        # --- 2 + 3. reachable target vs discrimination ---
        section("2+3. reachable target (decoded vs decoded GT) vs random")
        taeltx = load_taehv_ltx2(device=str(DEVICE), dtype=DTYPE, version="2.3")
        frames = read_video_frames_with_transform(_Item(vid), num_frames=T_FRAMES)
        with torch.no_grad():
            lat = taeltx.encode_video(frames.unsqueeze(0).to(DEVICE, DTYPE), parallel=True, show_progress_bar=False)
            rec = decode_wan_x0_to_frames(lat.permute(0, 2, 1, 3, 4).contiguous(), taeltx)[0].permute(1, 0, 2, 3)
        T_out = rec.shape[0]
        enc = DifferentiableFaceEncoder().to(DEVICE)
        h, w = rec.shape[2], rec.shape[3]
        mask = (vd[:T_out] > 0).to(DEVICE)
        gt_dev = gt[:T_out].float().to(DEVICE)
        boxes = [[float(bb[t][0]) * w, float(bb[t][1]) * h, float(bb[t][2]) * w, float(bb[t][3]) * h]
                 if vd[t] > 0 else None for t in range(T_out)]
        with torch.no_grad():
            gen = enc(rec.to(DEVICE), bboxes=boxes, return_crops=False)
        self_cos = F.cosine_similarity(gen, gt_dev, dim=-1)[mask].mean().item()
        rand_cos = F.cosine_similarity(gen, F.normalize(torch.randn_like(gt_dev), dim=-1), dim=-1)[mask].mean().item()
        print(f"  decoded-vs-DECODED-GT cosine = {self_cos:.4f} (loss {1-self_cos:.4f}) | vs random = {rand_cos:.4f}")
        assert self_cos > 0.97, f"target not reachable on the wired cache: {self_cos:.4f}"
        assert self_cos > rand_cos + 0.1, "no discrimination"
        print("  OK — target reachable (was ~0.2 with sharp-original GT)")

        # --- 4. gradient flow ---
        section("4. gradient flow: frames → chunked ArcFace → cosine → frames.grad")
        from torch.utils.checkpoint import checkpoint as ckpt
        x = rec.detach().clone().to(DEVICE).requires_grad_(True)
        chunk = _Cfg.identity_loss_frames_per_chunk
        embs = []
        for cs in range(0, x.shape[0], chunk):
            sub, sub_bb = x[cs:cs + chunk], boxes[cs:cs + chunk]

            def _fn(z, _bb=sub_bb):
                return enc(z, bboxes=_bb, return_crops=False)

            embs.append(ckpt(_fn, sub, use_reentrant=False))
        g_emb = torch.cat(embs, dim=0)
        target = F.normalize(torch.randn_like(gt_dev), dim=-1)
        loss = ((1.0 - F.cosine_similarity(g_emb, target, dim=-1)) * mask.float()).sum() / mask.float().sum().clamp(min=1)
        loss.backward()
        g = x.grad
        assert g is not None and torch.isfinite(g).all() and (g.abs() > 0).any()
        print(f"  loss={loss.item():.4f}  grad finite+nonzero ✓  |g|mean={g.abs().mean():.2e}  {fmt_mem()}")

        section("IDENTITY-VIDEO SMOKE PASSED (decoded-GT design)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
