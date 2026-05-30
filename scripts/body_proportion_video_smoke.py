#!/usr/bin/env python3
"""Smoke test for the ViTPose body-proportion video path (LTX / Wan 5D).

Validates the body-proportion-specific pieces on real full-body frames WITHOUT
the 22B LTX model (the TAEHV decode itself is covered by
``depth_consistency_video_smoke.py``):

  1. cache_video_body_proportion_embeddings → per-frame GT (T, 2N) cube;
     safetensors round-trip + version key; a body is detected (vis > 0).
  2. Self-consistency: re-encoding the SAME frames (the forward() path the loss
     uses) scores visibility-weighted ratio L1 ~0.
  3. Sensitivity: perturbing the GT ratios makes the L1 jump (loss responds to
     proportion mismatch). [Cross-person isn't used — bone-length ratios are
     pose/scale-invariant, so different photos of similar bodies score alike by
     design.]
  4. Gradient flow: frames → chunked + checkpointed ViTPose → ratio L1 →
     backward reaches the frames (finite, nonzero).
  5. Peak VRAM.

Usage:
    python scripts/body_proportion_video_smoke.py [body.(jpg|png)]
"""
import glob
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_FRAMES = 9
SIZE = 512


def fmt_mem():
    if not torch.cuda.is_available():
        return "cpu"
    return (f"alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
            f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB")


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def make_video(img_path, out_path, n=T_FRAMES, size=SIZE):
    import cv2
    from PIL import Image
    from PIL.ImageOps import exif_transpose

    img = exif_transpose(Image.open(img_path)).convert("RGB").resize((size, size))
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
        self.flip_x = False
        self.flip_y = False
        self.scale_to_width = None
        self.scale_to_height = None
        self.crop_x = self.crop_y = self.crop_width = self.crop_height = None
        self.width = self.height = None


class _Cfg:
    body_proportion_include_head = False
    body_proportion_frames_per_chunk = 2


def _ratio_l1(gen_ratios, ref_ratios, ref_vis, gen_vis):
    comb = torch.min(ref_vis, gen_vis)
    return ((gen_ratios - ref_ratios).abs() * comb).sum() / comb.sum().clamp(min=1e-6)


def main():
    a = sys.argv[1] if len(sys.argv) > 1 else None
    if a is None:
        cands = (sorted(glob.glob("test_data/cartwheel_clean/*.jpg"))
                 or sorted(glob.glob("test_data/scarlett_full/*.jpg")))
        assert cands, "no full-body test images found"
        a = cands[0]
    print(f"body image: {a}\ndevice: {DEVICE}")

    from safetensors.torch import load_file

    from toolkit.body_id import (
        DifferentiableBodyProportionEncoder,
        cache_video_body_proportion_embeddings,
    )
    from toolkit.video_frames import read_video_frames_with_transform

    tmp = tempfile.mkdtemp(prefix="bodyprop_video_smoke_")
    try:
        vid = os.path.join(tmp, "body.mp4")
        make_video(a, vid)

        # --- 1. GT caching round-trip ---
        section("1. cache_video_body_proportion_embeddings round-trip")
        item = _Item(vid)
        cache_video_body_proportion_embeddings([item], _Cfg(), device=DEVICE, num_frames=T_FRAMES)
        gt = item.body_proportion_gt_video
        assert gt is not None and gt.shape[0] == T_FRAMES, getattr(gt, "shape", None)
        n = gt.shape[-1] // 2
        ref_vis_cpu = gt[:, n:]
        assert float(ref_vis_cpu.sum()) > 0, "no body detected in any frame (bad test image?)"
        cache_path = os.path.join(tmp, "_face_id_cache", "body.safetensors")
        data = load_file(cache_path)
        assert "body_proportion_gt_video" in data and "body_proportion_gt_video_v2" in data
        print(f"  GT {tuple(gt.shape)} (N={n} ratios) vis_sum={float(ref_vis_cpu.sum()):.2f} cache OK  {fmt_mem()}")

        # --- 2 + 3. self-consistency + sensitivity ---
        section("2+3. self-consistency (L1 ~0) vs ratio-perturbation sensitivity")
        enc = DifferentiableBodyProportionEncoder().to(DEVICE).eval()
        frames = read_video_frames_with_transform(_Item(vid), num_frames=T_FRAMES)
        ref_ratios = gt[:, :n].to(DEVICE)
        ref_vis = gt[:, n:].to(DEVICE)
        with torch.no_grad():
            gen_r, gen_v = enc(frames.to(DEVICE), ref_ratios=ref_ratios, include_head=False)
        self_l1 = _ratio_l1(gen_r, ref_ratios, ref_vis, gen_v).item()
        print(f"  self ratio-L1 (gen == GT frames): {self_l1:.4f}")
        assert self_l1 < 0.05, f"self-consistency L1 too high: {self_l1:.4f}"

        pert = ref_ratios + 0.3  # shift every ratio
        pert_l1 = _ratio_l1(gen_r, pert, ref_vis, gen_v).item()
        print(f"  perturbed ratio-L1 (GT ratios + 0.3): {pert_l1:.4f}")
        assert pert_l1 > self_l1 + 0.1, "loss not sensitive to ratio perturbation"
        print("  OK — self ~0, sensitive to proportion mismatch")

        # --- 4. gradient flow through chunked + checkpointed ViTPose ---
        section("4. gradient flow: frames → chunked ViTPose → ratio L1 → frames.grad")
        from torch.utils.checkpoint import checkpoint as ckpt

        x = frames.clone().to(DEVICE).requires_grad_(True)
        chunk = _Cfg.body_proportion_frames_per_chunk
        r_chunks = []
        for cs in range(0, x.shape[0], chunk):
            sub = x[cs:cs + chunk]
            sub_ref = ref_ratios[cs:cs + chunk]

            def _fn(z, _r=sub_ref):
                return enc(z, ref_ratios=_r, include_head=False)

            r, _v = ckpt(_fn, sub, use_reentrant=False)
            r_chunks.append(r)
        gen = torch.cat(r_chunks, dim=0)
        # Compare against a perturbed target so the loss (and thus the gradient)
        # is nonzero — gen == GT here would sit exactly at the loss minimum.
        target = (ref_ratios + 0.3).detach()
        loss = (gen - target).abs().mean()
        loss.backward()
        g = x.grad
        assert g is not None, "no gradient on frames"
        assert torch.isfinite(g).all(), "frame grad not finite"
        assert (g.abs() > 0).any(), "frame grad all zero"
        print(f"  loss={loss.item():.4f}  grad finite={torch.isfinite(g).all().item()} "
              f"nonzero={(g.abs() > 0).any().item()}  |g|mean={g.abs().mean():.2e}")
        print("  OK")

        section("ALL BODY-PROPORTION-VIDEO SMOKE TESTS PASSED")
        print(f"  peak VRAM: {fmt_mem()}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
