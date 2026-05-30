#!/usr/bin/env python3
"""Dump inspectable artifacts for every step of the video-perceptor chain.

Walks the full pipeline on a real clip and writes images + a report so each
stage can be eyeballed:

  00_input_grid.png      frames read by read_video_frames_with_transform (T,3,512,512)
  01_latent_grid.png     TAEHV-LTX latent (mean over channels) per latent-frame
  02_recon_grid.png      decode_wan_x0_to_frames(clean latent) — the decoded clean latent
  02_recon_frames/       the same recon frames, one PNG each
  10_identity_bbox.png   recon frames with the cached GT face bbox drawn
  11_identity_crops.png  [orig 112 crop | recon 112 crop] per frame — what ArcFace sees
  20_body_skeleton.png   [orig + ViTPose skeleton | recon + ViTPose skeleton] per frame
  30_depth_{t}.png       render_depth_preview strips [GT rgb|GT depth|recon rgb|recon depth]
  report.txt             per-frame numbers for all three perceptors

Usage: python scripts/dump_video_perceptor_artifacts.py [video.mp4]
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
LOG = []


def log(msg):
    print(msg)
    LOG.append(str(msg))


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


class _Cfg:
    face_model = "buffalo_l"
    identity_loss_min_cos = 0.2
    body_proportion_include_head = False
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    input_size = 518
    pixel_blur_sigma = 0.0


def to_pil(chw):
    return Image.fromarray((chw.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy())


def thumb(pil, s=224):
    return pil.resize((s, s))


def grid(pils, cols, pad=2, bg=(20, 20, 20)):
    if not pils:
        return Image.new("RGB", (cols, 1), bg)
    cw = max(p.width for p in pils)
    ch = max(p.height for p in pils)
    rows = (len(pils) + cols - 1) // cols
    out = Image.new("RGB", (cols * (cw + pad) + pad, rows * (ch + pad) + pad), bg)
    for i, p in enumerate(pils):
        r, c = divmod(i, cols)
        out.paste(p, (pad + c * (cw + pad), pad + r * (ch + pad)))
    return out


def hcat(pils, pad=2, bg=(20, 20, 20)):
    h = max(p.height for p in pils)
    w = sum(p.width for p in pils) + pad * (len(pils) + 1)
    out = Image.new("RGB", (w, h + 2 * pad), bg)
    x = pad
    for p in pils:
        out.paste(p, (x, pad))
        x += p.width + pad
    return out


def draw_bbox(pil, bbox, color=(0, 255, 0)):
    pil = pil.copy()
    if bbox is not None:
        ImageDraw.Draw(pil).rectangle([float(b) for b in bbox], outline=color, width=3)
    return pil


def depth_to_pil(d, size=224):
    d = d.detach().float().cpu().numpy()
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    dn = np.clip((d - lo) / max(1e-6, hi - lo), 0, 1)
    return Image.fromarray((dn * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC).convert("RGB")


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    assert os.path.exists(video), video
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "02_recon_frames"), exist_ok=True)
    log(f"video: {video}  device: {DEVICE}  frames: {NUM_FRAMES}  size: {SIZE}")

    from toolkit.body_id import DifferentiableBodyProportionEncoder, draw_skeleton_overlay
    from toolkit.depth_consistency import (
        DifferentiableDepthEncoder,
        decode_wan_x0_to_frames,
        load_taehv_ltx2,
        ssi_l1,
    )
    from toolkit.face_id import DifferentiableFaceEncoder, FaceIDExtractor
    from toolkit.video_frames import read_video_frames_with_transform

    # ---- STEP 0: read frames ----
    frames = read_video_frames_with_transform(_Item(video), num_frames=NUM_FRAMES)  # (T,3,512,512)
    T = frames.shape[0]
    log(f"[00] frames read: {tuple(frames.shape)}")
    grid([thumb(to_pil(frames[t])) for t in range(T)], cols=6).save(f"{OUT}/00_input_grid.png")

    # ---- STEP 1+2: encode → latent → decode ----
    taeltx = load_taehv_ltx2(device=str(DEVICE), dtype=DTYPE, version="2.3")
    with torch.no_grad():
        lat = taeltx.encode_video(frames.unsqueeze(0).to(DEVICE, DTYPE), parallel=True, show_progress_bar=False)
        lat_ncthw = lat.permute(0, 2, 1, 3, 4).contiguous().float()  # (1,C,Tl,Hl,Wl)
        rec = decode_wan_x0_to_frames(lat_ncthw, taeltx)[0].permute(1, 0, 2, 3)  # (T_out,3,H,W)
    Tl = lat_ncthw.shape[2]
    T_out = rec.shape[0]
    log(f"[01] latent: {tuple(lat_ncthw.shape)}  mean={lat_ncthw.mean():.3f} std={lat_ncthw.std():.3f}")
    log(f"[02] recon: {tuple(rec.shape)}  range=[{rec.min():.2f},{rec.max():.2f}]")

    # latent channel-mean heatmap per latent-frame
    lat_imgs = []
    for t in range(Tl):
        m = lat_ncthw[0, :, t].mean(0)  # (Hl,Wl)
        lat_imgs.append(depth_to_pil(m, size=224))
    hcat(lat_imgs).save(f"{OUT}/01_latent_grid.png")

    grid([thumb(to_pil(rec[t])) for t in range(T_out)], cols=6).save(f"{OUT}/02_recon_grid.png")
    for t in range(T_out):
        to_pil(rec[t]).save(f"{OUT}/02_recon_frames/frame_{t:02d}.png")

    orig_pils = [to_pil(frames[t]) for t in range(T)]
    recon_pils = [to_pil(rec[t]) for t in range(T_out)]
    import cv2

    # ---- IDENTITY ----
    log("\n[10] identity (ArcFace)")
    extractor = FaceIDExtractor(model_name="buffalo_l")
    fenc = DifferentiableFaceEncoder().to(DEVICE)
    id_bbox_cells, id_crop_cells = [], []
    for t in range(T_out):
        op, rp = orig_pils[min(t, T - 1)], recon_pils[t]
        _, obb = extractor.extract_with_bbox(cv2.cvtColor(np.array(op), cv2.COLOR_RGB2BGR))
        # GT bbox (from original) drawn on the recon — exactly what the loss crops.
        id_bbox_cells.append(thumb(draw_bbox(rp, obb)))

        def crop112(pil, bb):
            return (pil.crop([float(b) for b in bb]) if bb is not None else pil).resize((112, 112))

        id_crop_cells.append(hcat([crop112(op, obb), crop112(rp, obb)], pad=1))
        with torch.no_grad():
            bb = [[float(v) for v in obb]] if obb is not None else [None]
            eo = fenc(frames[min(t, T - 1)].unsqueeze(0).to(DEVICE), bboxes=bb, return_crops=False)
            er = fenc(rec[t].unsqueeze(0).to(DEVICE), bboxes=bb, return_crops=False)
        cos = F.cosine_similarity(eo, er, dim=-1).item()
        log(f"   frame {t:02d}: bbox={None if obb is None else [round(float(v)) for v in obb]}  "
            f"cos(orig,recon)={cos:.3f}")
    grid(id_bbox_cells, cols=6).save(f"{OUT}/10_identity_bbox.png")
    grid(id_crop_cells, cols=6).save(f"{OUT}/11_identity_crops.png")

    # ---- BODY-PROPORTION ----
    log("\n[20] body-proportion (ViTPose)")
    benc = DifferentiableBodyProportionEncoder().to(DEVICE).eval()
    with torch.no_grad():
        benc(frames.to(DEVICE), include_head=False)
        okp, ovis = benc._last_keypoints.clone(), benc._last_visibility.clone()
        benc(rec.to(DEVICE), include_head=False)
        rkp, rvis = benc._last_keypoints.clone(), benc._last_visibility.clone()
    body_cells = []
    for t in range(T_out):
        os_ = draw_skeleton_overlay(orig_pils[min(t, T - 1)], okp[min(t, okp.shape[0] - 1)], ovis[min(t, okp.shape[0] - 1)])
        rs_ = draw_skeleton_overlay(recon_pils[t], rkp[t], rvis[t])
        body_cells.append(hcat([thumb(os_), thumb(rs_)], pad=1))
        log(f"   frame {t:02d}: orig_vis_mean={ovis[min(t, okp.shape[0]-1)].mean():.2f}  recon_vis_mean={rvis[t].mean():.2f}")
    grid(body_cells, cols=3).save(f"{OUT}/20_body_skeleton.png")

    # ---- DEPTH ----
    log("\n[30] depth-consistency (DA2)")
    del fenc, benc, extractor      # free face/body models first (shared GPU)
    torch.cuda.empty_cache()
    denc = DifferentiableDepthEncoder(model_id=_Cfg.model_id, input_size=_Cfg.input_size,
                                      grad_checkpoint=False, device=DEVICE)

    def depth_batched(x, bs=2):
        outs = []
        with torch.no_grad():
            for cs in range(0, x.shape[0], bs):
                outs.append(denc(x[cs:cs + bs].to(DEVICE)).float().cpu())
        return torch.cat(outs, 0)

    d_orig = depth_batched(frames)   # (T,Hd,Wd) GT-equivalent
    d_rec = depth_batched(rec)       # (T_out,Hd,Wd)
    from toolkit.depth_consistency import render_depth_preview
    for t in range(0, T_out, max(1, T_out // 5)):
        strip = render_depth_preview(recon_pils[t], orig_pils[min(t, T - 1)], d_rec[t], d_orig[min(t, d_orig.shape[0] - 1)])
        strip.save(f"{OUT}/30_depth_{t:02d}.png")
        ssi = ssi_l1(d_rec[t], d_orig[min(t, d_orig.shape[0] - 1)])[0].item()
        log(f"   frame {t:02d}: depth SSI(recon,orig)={ssi:.3f}")

    with open(f"{OUT}/report.txt", "w") as f:
        f.write("\n".join(LOG) + "\n")
    log(f"\nwrote artifacts under {OUT}/  ({len(os.listdir(OUT))} entries) + report.txt")


if __name__ == "__main__":
    main()
