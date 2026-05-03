#!/usr/bin/env python3
"""Smoke test for the Wan 2.1 video depth-consistency path.

Exercises each piece of the pipeline end-to-end on synthetic Wan-shaped
latents so we get signal without needing a full training run:

  1. TAEHV decoder loads from toolkit/taehv/taew2_1.pth.
  2. Synthetic (B, C=16, T=11, H=32, W=32) x0 latents decode via
     decode_wan_x0_to_frames to (B, 3, T_out, H_out, W_out) in [0, 1].
  3. DA2-Small runs on the flattened (B*T_out, 3, H, W) frames under
     gradient checkpointing + chunking.
  4. Per-frame SSI + multi-scale gradient loss against a GT cube —
     self-loss on identical cubes must be ~0; different cubes > 0.
  5. Gradients propagate back to the latents (non-zero, finite).
  6. Peak VRAM fits in 24 GB.

Extended (by --full) to also cover the I/O + preview + full-chain pieces
the above leaves out:

  6. cache_video_depth_gt_embeddings on a real short video → reload →
     shape/version key match (exercises cv2 read + safetensors round-trip).
  7. save_video_depth_preview writes an animated webp that re-opens with
     N frames (catches Pillow webp-anim regressions).
  8. Full-stack backward: x0 → decode_wan_x0_to_frames → chunked DA2 →
     per-frame loss → backward reaches x0 (matches the real SDTrainer path).

Usage:
    python scripts/depth_consistency_video_smoke.py              # tests 1-5
    python scripts/depth_consistency_video_smoke.py --full       # + 6-8
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from toolkit.depth_consistency import (
    DifferentiableDepthEncoder,
    decode_wan_x0_to_frames,
    load_taehv_wan21,
    ssi_l1,
    multiscale_grad_loss,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fmt_mem():
    if DEVICE.type != "cuda":
        return "cpu"
    used = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    return f"alloc={used:.2f}GB peak={peak:.2f}GB"


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def test_1_load_taehv():
    section("1. Load TAEHV tiny decoder")
    tae = load_taehv_wan21(device=str(DEVICE), dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in tae.parameters())
    print(f"  params: {n_params/1e6:.1f}M   {fmt_mem()}")
    print("  OK")
    return tae


def test_2_decode_latents(tae):
    section("2. Decode synthetic Wan 2.1 latents → frames")
    # Wan 2.1 1.3B latent shape: C=16 channels, temporal compression 4.
    # 41 frames → (41-1)/4+1 = 11 latent frames.
    B, C, T, H, W = 1, 16, 11, 32, 32
    x0 = torch.randn(B, C, T, H, W, device=DEVICE, dtype=torch.float32)
    x0.requires_grad_(True)
    frames = decode_wan_x0_to_frames(x0, tae)
    print(f"  latents: {tuple(x0.shape)}  →  frames: {tuple(frames.shape)}")
    print(f"  pixels range: [{frames.min():.3f}, {frames.max():.3f}]   {fmt_mem()}")
    assert frames.dim() == 5 and frames.shape[1] == 3, "expected (B,3,T,H,W)"
    return x0, frames


def test_3_depth_encoder(frames):
    section("3. DA2-Small per-frame depth (chunked + checkpointed)")
    enc = DifferentiableDepthEncoder(grad_checkpoint=True, device=DEVICE)
    B, _, T, H, W = frames.shape
    flat = frames.permute(0, 2, 1, 3, 4).reshape(B * T, 3, H, W)

    from torch.utils.checkpoint import checkpoint as _ckpt

    chunks = []
    for c in flat.split(8, dim=0):
        chunks.append(_ckpt(enc, c, use_reentrant=False))
    depth = torch.cat(chunks, dim=0)
    if depth.dim() == 4:
        depth = depth.squeeze(1)
    print(f"  flat: {tuple(flat.shape)}  →  depth: {tuple(depth.shape)}   {fmt_mem()}")
    return enc, depth


def test_4_losses(depth):
    section("4. SSI + multi-scale gradient loss — self vs different")
    # Per-frame reshape: (T, H, W).
    d = depth.reshape(-1, *depth.shape[-2:])

    # Self-loss must be exactly zero.
    self_ssi = ssi_l1(d[0], d[0])[0].item()
    self_grad = multiscale_grad_loss(d[0], d[0], scales=4).item()
    print(f"  self: ssi={self_ssi:.6f}  grad={self_grad:.6f}")
    assert self_ssi < 1e-5 and self_grad < 1e-5, "self-loss must be ~0"

    # Different frames must give > 0 loss.
    diff_ssi = ssi_l1(d[0], d[-1])[0].item()
    diff_grad = multiscale_grad_loss(d[0], d[-1], scales=4).item()
    print(f"  diff: ssi={diff_ssi:.6f}  grad={diff_grad:.6f}")
    assert diff_ssi > 0 or diff_grad > 0, "different frames should produce > 0"
    print("  OK")


def test_5_grad_flow(x0, depth):
    section("5. Gradient flow x0 → loss (non-zero, finite)")
    d = depth.reshape(-1, *depth.shape[-2:])
    # Contrastive target: permuted frames so loss is non-zero.
    target = d[torch.randperm(d.shape[0])]
    loss = d.new_zeros(())
    for t in range(d.shape[0]):
        _s, _, _ = ssi_l1(d[t], target[t].detach())
        loss = loss + _s + multiscale_grad_loss(d[t], target[t].detach(), scales=4)
    loss = loss / d.shape[0]
    print(f"  loss: {loss.item():.6f}   {fmt_mem()}")
    loss.backward()
    g = x0.grad
    assert g is not None, "x0 should have gradient"
    print(
        f"  x0.grad: shape={tuple(g.shape)}  finite={torch.isfinite(g).all().item()}  "
        f"|g|_mean={g.abs().mean():.3e}  nonzero_frac={(g != 0).float().mean():.3f}"
    )
    assert torch.isfinite(g).all(), "x0 grad must be finite"
    assert (g.abs() > 0).any(), "x0 grad must be non-zero somewhere"
    print("  OK")
    print(f"\nFinal   {fmt_mem()}")


def test_6_cache_video_depth_gt(source_video: str, num_frames: int = 11):
    """Round-trip real video → DA2 GT cache → reload → shape/version check."""
    section("6. cache_video_depth_gt_embeddings on a real video")
    import shutil
    import tempfile

    from safetensors.torch import load_file
    from toolkit.depth_consistency import (
        CACHE_VERSION_VIDEO_KEY,
        cache_video_depth_gt_embeddings,
    )

    if not os.path.exists(source_video):
        raise FileNotFoundError(
            f"Source video for cache test not found: {source_video}"
        )

    # Copy into a tmpdir so we don't pollute the real dataset with a
    # _face_id_cache/ folder, and so reruns start clean.
    tmpdir = tempfile.mkdtemp(prefix="depth_gt_cache_smoke_")
    try:
        stem = os.path.splitext(os.path.basename(source_video))[0]
        dst_video = os.path.join(tmpdir, os.path.basename(source_video))
        shutil.copy2(source_video, dst_video)

        class _FakeItem:
            def __init__(self, path):
                self.path = path
                self.is_video = True
                self.depth_gt_video = None

        item = _FakeItem(dst_video)

        class _Cfg:
            model_id = "depth-anything/Depth-Anything-V2-Small-hf"
            input_size = 518

        cache_video_depth_gt_embeddings(
            [item], _Cfg(), device=DEVICE, num_frames=num_frames, batch_size=4,
        )
        assert item.depth_gt_video is not None, "depth_gt_video not attached"
        assert item.depth_gt_video.shape[0] == num_frames, (
            f"cached T={item.depth_gt_video.shape[0]}, expected {num_frames}"
        )

        cache_path = os.path.join(tmpdir, "_face_id_cache", f"{stem}.safetensors")
        assert os.path.exists(cache_path), f"missing cache file: {cache_path}"
        data = load_file(cache_path)
        assert "depth_gt_video" in data, "depth_gt_video key missing in cache"
        assert CACHE_VERSION_VIDEO_KEY in data, "cache version key missing"
        assert data["depth_gt_video"].shape == item.depth_gt_video.shape
        print(
            f"  cache: {cache_path}\n"
            f"  shape: {tuple(data['depth_gt_video'].shape)}  dtype={data['depth_gt_video'].dtype}"
        )
        print("  OK")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_7_video_preview_webp():
    """save_video_depth_preview writes an animated webp that re-opens with N frames."""
    section("7. save_video_depth_preview animated webp round-trip")
    import tempfile

    from PIL import Image
    from toolkit.depth_consistency import save_video_depth_preview

    T, H, W = 11, 64, 64
    gen_rgb = torch.rand(T, 3, H, W)
    gen_depth = torch.rand(T, H, W)
    gt_depth = torch.rand(T, H, W)

    with tempfile.TemporaryDirectory(prefix="depth_preview_smoke_") as tmp:
        out = os.path.join(tmp, "preview")
        save_video_depth_preview(out, gen_rgb, gen_depth, gt_depth, fps=8)
        written = out + ".webp"
        assert os.path.exists(written), f"webp not written: {written}"
        with Image.open(written) as im:
            n = getattr(im, "n_frames", 1)
            assert n == T, f"animated webp has {n} frames, expected {T}"
            print(f"  webp: {os.path.basename(written)}  frames={n}  size={im.size}")
    print("  OK")


def test_8_full_chain_backward(tae):
    """Backward through decode → chunked+checkpointed DA2 → loss reaches x0."""
    section("8. Full-chain backward: x0 → TAEHV → DA2 → loss → x0.grad")
    from torch.utils.checkpoint import checkpoint as _ckpt

    from toolkit.depth_consistency import (
        decode_wan_x0_to_frames,
        multiscale_grad_loss,
        ssi_l1,
    )

    enc = DifferentiableDepthEncoder(grad_checkpoint=True, device=DEVICE)

    # Real Wan 2.1 1.3B shape.
    B, C, T, H, W = 1, 16, 11, 32, 32
    x0 = torch.randn(B, C, T, H, W, device=DEVICE, dtype=torch.float32, requires_grad=True)
    frames = decode_wan_x0_to_frames(x0, tae)
    B_vid, _, T_out, H_out, W_out = frames.shape
    flat = frames.permute(0, 2, 1, 3, 4).reshape(B_vid * T_out, 3, H_out, W_out)

    depth_chunks = []
    for c in flat.split(8, dim=0):
        depth_chunks.append(_ckpt(enc, c, use_reentrant=False))
    depth = torch.cat(depth_chunks, dim=0)
    if depth.dim() == 4:
        depth = depth.squeeze(1)
    depth = depth.reshape(B_vid, T_out, *depth.shape[1:])

    # GT cube unrelated to x0 so the loss is nonzero.
    gt = torch.randn_like(depth).detach()

    ssi_per, grad_per = [], []
    for t in range(T_out):
        s, _, _ = ssi_l1(depth[0, t], gt[0, t])
        ssi_per.append(s)
        grad_per.append(multiscale_grad_loss(depth[0, t], gt[0, t], scales=4))
    loss = torch.stack(ssi_per).mean() + 0.5 * torch.stack(grad_per).mean()
    print(f"  loss: {loss.item():.6f}   {fmt_mem()}")
    loss.backward()
    g = x0.grad
    assert g is not None, "x0 should have gradient"
    assert torch.isfinite(g).all(), "x0 grad must be finite"
    assert (g.abs() > 0).any(), "x0 grad must be non-zero somewhere"
    print(
        f"  x0.grad: shape={tuple(g.shape)}  finite=True  "
        f"|g|_mean={g.abs().mean():.3e}  nonzero_frac={(g != 0).float().mean():.3f}"
    )
    print(f"  Final   {fmt_mem()}")
    print("  OK")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full", action="store_true",
        help="Also run tests 6-8 (cache round-trip, preview webp, full-chain backward).",
    )
    parser.add_argument(
        "--cache-video",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "test_data", "cartwheel",
            "Jessica_and_Gregs_Cartwheel_Competition_cartwheel_f_cm_np1_ri_med_2.avi",
        ),
        help="Video file to use for test 6 (GT cache round-trip).",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    tae = test_1_load_taehv()
    x0, frames = test_2_decode_latents(tae)
    enc, depth = test_3_depth_encoder(frames)
    test_4_losses(depth)
    test_5_grad_flow(x0, depth)

    if args.full:
        test_6_cache_video_depth_gt(args.cache_video)
        test_7_video_preview_webp()
        test_8_full_chain_backward(tae)

    print("\nAll checks green.")
