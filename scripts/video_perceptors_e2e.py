#!/usr/bin/env python3
"""End-to-end validation of the three video perceptors on a real clip.

Runs the FULL per-frame pipeline each video perceptor uses at train time —
without the 22B LTX transformer — by substituting a TAEHV-LTX codec round-trip
for the model's x0 prediction:

    real video → read_video_frames_with_transform → cache_video_*  (GT)
              → TAEHV-LTX encode_video → latents (x0, requires_grad)
              → decode_wan_x0_to_frames (the real LTX decode branch) → frames
              → perceptor.forward (chunked, gradient-checkpointed) → per-frame loss vs GT
              → backward → gradient reaches the latents

For each perceptor it checks:
  * GT caching on the real clip (shape + a body/face detected),
  * the loss vs the correct GT is markedly better than vs a wrong/perturbed GT
    (the loss actually discriminates through the whole pipeline),
  * a finite, nonzero gradient reaches the x0 latents (the loss can train),
  * peak VRAM.

Mirrors the SDTrainer 5D loss blocks (decode-once → flatten B*T → chunked
encoder → per-frame loss vs GT cube). Decode itself = the same
decode_wan_x0_to_frames + taeltx the trainer uses.

Usage:
    python scripts/video_perceptors_e2e.py [video.mp4]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16
NUM_FRAMES = 17  # 8n+1 for LTX
SIZE = 512
DEFAULT_VIDEO = "test_data/dance_clips/man_dancing_square_16fps_4s.mp4"


def fmt_mem():
    if not torch.cuda.is_available():
        return "cpu"
    return (f"alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
            f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB")


def section(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


class _Item:
    """FileItemDTO stand-in: a video resized to SIZE×SIZE (no flip/aug)."""

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
    # face / body
    face_model = "buffalo_l"
    identity_loss_min_cos = 0.2
    identity_loss_frames_per_chunk = 4
    body_proportion_include_head = False
    body_proportion_frames_per_chunk = 2
    # depth
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    input_size = 518
    pixel_blur_sigma = 0.0
    ssi_weight = 1.0
    grad_weight = 0.5
    grad_scales = 4


def align_T(cube, T_out):
    """linspace-resample a (Tg, ...) GT cube along dim 0 to T_out (as the blocks do)."""
    if cube.shape[0] == T_out:
        return cube
    ix = torch.linspace(0, cube.shape[0] - 1, T_out).round().long()
    return cube[ix]


def make_x0_latents(frames, taeltx):
    """real frames (T,3,H,W)[0,1] → TAEHV-LTX latents x0 in NCTHW."""
    x_ntchw = frames.unsqueeze(0).to(DEVICE, DTYPE)  # (1, T, 3, H, W)
    with torch.no_grad():
        lat_ntchw = taeltx.encode_video(x_ntchw, parallel=True, show_progress_bar=False)
    return lat_ntchw.permute(0, 2, 1, 3, 4).contiguous().float()  # (1, C, T, H, W)


def decode_frames(x0_ncthw, taeltx):
    from toolkit.depth_consistency import decode_wan_x0_to_frames
    return decode_wan_x0_to_frames(x0_ncthw, taeltx)  # (1, 3, T_out, H, W) in [0,1]


# ----------------------------- perceptors -----------------------------
def e2e_identity(item, frames, taeltx):
    section("IDENTITY (ArcFace) — end to end")
    from toolkit.face_id import DifferentiableFaceEncoder, cache_video_identity_embeddings

    cache_video_identity_embeddings([item], _Cfg(), device=DEVICE, num_frames=NUM_FRAMES)
    gt = item.identity_gt_video.float()                  # (T, 512)
    bb = item.identity_gt_video_bbox.float()             # (T, 4) normalized
    vd = item.identity_gt_video_valid.float()            # (T,)
    print(f"  GT cube {tuple(gt.shape)}  faces detected {int(vd.sum())}/{gt.shape[0]}")
    assert gt.shape[0] == NUM_FRAMES and float(vd.sum()) > 0

    enc = DifferentiableFaceEncoder().to(DEVICE)

    x0 = make_x0_latents(frames, taeltx).requires_grad_(True)
    rec = decode_frames(x0, taeltx)
    B, _, T_out, H, W = rec.shape
    gt_a = align_T(gt, T_out).to(DEVICE)
    bb_a = align_T(bb, T_out)
    vd_a = align_T(vd, T_out).to(DEVICE)

    flat = rec.permute(0, 2, 1, 3, 4).reshape(B * T_out, 3, H, W)
    boxes = [[float(bb_a[t][0]) * W, float(bb_a[t][1]) * H,
              float(bb_a[t][2]) * W, float(bb_a[t][3]) * H] if vd_a[t] > 0 else None
             for t in range(T_out)]
    chunk = _Cfg.identity_loss_frames_per_chunk
    embs = []
    for cs in range(0, flat.shape[0], chunk):
        sub, sub_bb = flat[cs:cs + chunk], boxes[cs:cs + chunk]

        def fn(z, _bb=sub_bb):
            return enc(z, bboxes=_bb, return_crops=False)

        embs.append(ckpt(fn, sub, use_reentrant=False))
    gen = torch.cat(embs, dim=0)                          # (T_out, 512)
    mask = vd_a > 0

    cos = F.cosine_similarity(gen, gt_a, dim=-1)
    good = cos[mask].mean().item()
    wrong = F.cosine_similarity(gen, F.normalize(torch.randn_like(gt_a), dim=-1), dim=-1)[mask].mean().item()
    print(f"  recon-vs-GT cosine = {good:.4f}  (loss {1-good:.4f}) | recon-vs-random = {wrong:.4f}")
    assert good > wrong + 0.1, "identity loss does not discriminate through the codec round-trip"

    loss = ((1.0 - cos) * mask.float()).sum() / mask.float().sum().clamp(min=1)
    loss.backward()
    g = x0.grad
    assert g is not None and torch.isfinite(g).all() and (g.abs() > 0).any()
    print(f"  loss={loss.item():.4f}  x0.grad finite+nonzero ✓  |g|mean={g.abs().mean():.2e}  {fmt_mem()}")
    print("  IDENTITY OK")


def e2e_body(item, frames, taeltx):
    section("BODY-PROPORTION (ViTPose) — end to end")
    from toolkit.body_id import (
        DifferentiableBodyProportionEncoder,
        cache_video_body_proportion_embeddings,
    )

    cache_video_body_proportion_embeddings([item], _Cfg(), device=DEVICE, num_frames=NUM_FRAMES)
    gt = item.body_proportion_gt_video.float()           # (T, 2N)
    n = gt.shape[-1] // 2
    print(f"  GT cube {tuple(gt.shape)} (N={n})  vis_sum={float(gt[:, n:].sum()):.2f}")
    assert gt.shape[0] == NUM_FRAMES and float(gt[:, n:].sum()) > 0

    enc = DifferentiableBodyProportionEncoder().to(DEVICE).eval()

    x0 = make_x0_latents(frames, taeltx).requires_grad_(True)
    rec = decode_frames(x0, taeltx)
    B, _, T_out, H, W = rec.shape
    gt_a = align_T(gt, T_out).to(DEVICE)
    ref_ratios, ref_vis = gt_a[:, :n], gt_a[:, n:]

    flat = rec.permute(0, 2, 1, 3, 4).reshape(B * T_out, 3, H, W)
    ref_flat = ref_ratios.reshape(B * T_out, n)
    chunk = _Cfg.body_proportion_frames_per_chunk
    r_ch, v_ch = [], []
    for cs in range(0, flat.shape[0], chunk):
        sub, sub_ref = flat[cs:cs + chunk], ref_flat[cs:cs + chunk]

        def fn(z, _r=sub_ref):
            return enc(z, ref_ratios=_r, include_head=False)

        r, v = ckpt(fn, sub, use_reentrant=False)
        r_ch.append(r)
        v_ch.append(v)
    gen_r = torch.cat(r_ch, dim=0)
    gen_v = torch.cat(v_ch, dim=0)

    comb = torch.min(ref_vis, gen_v)
    good = (((gen_r - ref_ratios).abs() * comb).sum() / comb.sum().clamp(min=1e-6)).item()
    wrong = (((gen_r - (ref_ratios + 0.3)).abs() * comb).sum() / comb.sum().clamp(min=1e-6)).item()
    print(f"  recon-vs-GT ratio-L1 = {good:.4f} | recon-vs-(GT+0.3) = {wrong:.4f}")
    assert wrong > good + 0.1, "body-proportion loss not sensitive through the codec round-trip"

    target = (ref_ratios + 0.3).detach()
    loss = (gen_r - target).abs().mean()
    loss.backward()
    g = x0.grad
    assert g is not None and torch.isfinite(g).all() and (g.abs() > 0).any()
    print(f"  loss={loss.item():.4f}  x0.grad finite+nonzero ✓  |g|mean={g.abs().mean():.2e}  {fmt_mem()}")
    print("  BODY-PROPORTION OK")


def e2e_depth(item, frames, taeltx):
    section("DEPTH-CONSISTENCY (DA2) — end to end")
    from toolkit.depth_consistency import (
        DifferentiableDepthEncoder,
        cache_video_depth_gt_embeddings,
        ssi_l1,
    )

    cache_video_depth_gt_embeddings([item], _Cfg(), device=DEVICE, num_frames=NUM_FRAMES, batch_size=4)
    gt = item.depth_gt_video.float()                     # (T, Hd, Wd)
    print(f"  GT cube {tuple(gt.shape)}")
    assert gt.shape[0] == NUM_FRAMES

    enc = DifferentiableDepthEncoder(model_id=_Cfg.model_id, input_size=_Cfg.input_size,
                                     grad_checkpoint=True, device=DEVICE)

    x0 = make_x0_latents(frames, taeltx).requires_grad_(True)
    rec = decode_frames(x0, taeltx)
    B, _, T_out, H, W = rec.shape
    flat = rec.permute(0, 2, 1, 3, 4).reshape(B * T_out, 3, H, W)

    d_ch = []
    for cs in range(0, flat.shape[0], 4):
        d_ch.append(ckpt(enc, flat[cs:cs + 4], use_reentrant=False))
    depth = torch.cat(d_ch, dim=0)
    if depth.dim() == 4:
        depth = depth.squeeze(1)                          # (T_out, Hd, Wd)
    gt_a = align_T(gt, T_out).to(DEVICE)
    if gt_a.shape[-2:] != depth.shape[-2:]:
        gt_a = F.interpolate(gt_a.unsqueeze(1), size=depth.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

    good = torch.stack([ssi_l1(depth[t], gt_a[t])[0] for t in range(T_out)]).mean().item()
    shuf = torch.flip(gt_a, dims=[0])                     # wrong temporal alignment
    wrong = torch.stack([ssi_l1(depth[t], shuf[t])[0] for t in range(T_out)]).mean().item()
    print(f"  recon-vs-GT SSI = {good:.4f} | recon-vs-time-flipped-GT = {wrong:.4f}")
    assert good <= wrong + 1e-4, "depth SSI vs correct GT should be <= vs mismatched GT"

    loss = torch.stack([ssi_l1(depth[t], gt_a[t])[0] for t in range(T_out)]).mean()
    loss.backward()
    g = x0.grad
    assert g is not None and torch.isfinite(g).all() and (g.abs() > 0).any()
    print(f"  loss={loss.item():.4f}  x0.grad finite+nonzero ✓  |g|mean={g.abs().mean():.2e}  {fmt_mem()}")
    print("  DEPTH OK")


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    assert os.path.exists(video), f"video not found: {video}"
    print(f"video: {video}\ndevice: {DEVICE}  frames: {NUM_FRAMES}  size: {SIZE}")

    from toolkit.depth_consistency import load_taehv_ltx2
    from toolkit.video_frames import read_video_frames_with_transform

    section("0. TAEHV-LTX codec + real frames")
    taeltx = load_taehv_ltx2(device=str(DEVICE), dtype=DTYPE, version="2.3")
    frames = read_video_frames_with_transform(_Item(video), num_frames=NUM_FRAMES)
    assert frames is not None and frames.shape[0] == NUM_FRAMES, getattr(frames, "shape", None)
    # sanity: round-trip decodes to valid frames
    x0 = make_x0_latents(frames, taeltx)
    rec = decode_frames(x0, taeltx)
    print(f"  frames {tuple(frames.shape)} → latents {tuple(x0.shape)} → recon {tuple(rec.shape)} "
          f"range=[{rec.min():.2f},{rec.max():.2f}]")

    e2e_identity(_Item(video), frames, taeltx)
    e2e_body(_Item(video), frames, taeltx)
    e2e_depth(_Item(video), frames, taeltx)

    section("ALL THREE PERCEPTORS VALIDATED END-TO-END ON THE REAL CLIP")
    print(f"  peak VRAM: {fmt_mem()}")


if __name__ == "__main__":
    main()
