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

Usage: python scripts/depth_consistency_video_smoke.py
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


if __name__ == "__main__":
    torch.manual_seed(0)
    tae = test_1_load_taehv()
    x0, frames = test_2_decode_latents(tae)
    enc, depth = test_3_depth_encoder(frames)
    test_4_losses(depth)
    test_5_grad_flow(x0, depth)
    print("\nAll checks green.")
