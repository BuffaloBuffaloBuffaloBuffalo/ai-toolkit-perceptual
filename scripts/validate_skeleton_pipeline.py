#!/usr/bin/env python3
"""Validate each stage of the skeleton motion loss pipeline independently.

Before launching a full training, verify that:
  1. Reference skeleton loads correctly
  2. ViTPoseExtractor forward pass works with gradients
  3. Wan VAE x0 decode works with gradients
  4. Full chain (x0 → VAE → ViTPose → loss) produces non-zero gradients
  5. Gradient checkpointing fits in memory

Usage:
    python scripts/validate_skeleton_pipeline.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from toolkit.skeleton_motion import (
    ViTPoseExtractor,
    normalize_keypoints,
    skeleton_motion_loss,
    skeleton_velocity_loss,
    decode_wan_x0_to_frames,
)

DEVICE = 'cuda'
CACHED_SKELETON = 'test_data/cartwheel_train/cartwheel_skeleton.safetensors'
NUM_FRAMES = 117
LATENT_T = (NUM_FRAMES - 1) // 4 + 1  # Wan temporal compression = 30
LATENT_H, LATENT_W = 32, 36  # 256/8, 288/8 (approx, depends on bucket)


def fmt_mem():
    used = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    return f"mem_alloc={used:.2f}GB peak={peak:.2f}GB"


def section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


def test_1_load_reference():
    section("1. Load cached reference skeleton")
    if not os.path.exists(CACHED_SKELETON):
        print(f"SKIP: cache not found at {CACHED_SKELETON}")
        return None, None
    data = load_file(CACHED_SKELETON)
    kp = data['keypoints']
    vis = data['visibility']
    print(f"  keypoints: {tuple(kp.shape)} range=[{kp.min():.3f}, {kp.max():.3f}]")
    print(f"  visibility: {tuple(vis.shape)} mean={vis.mean():.3f}")
    print(f"  OK")
    return kp.to(DEVICE), vis.to(DEVICE)


def test_2_vitpose_forward():
    section("2. ViTPoseExtractor forward (no grad)")
    torch.cuda.reset_peak_memory_stats()
    model = ViTPoseExtractor().to(DEVICE).eval()
    frames = torch.rand(4, 3, 256, 288, device=DEVICE)
    with torch.no_grad():
        kp, vis = model(frames)
    print(f"  input: {tuple(frames.shape)}")
    print(f"  keypoints: {tuple(kp.shape)} range=[{kp.min():.3f}, {kp.max():.3f}]")
    print(f"  visibility: {tuple(vis.shape)} mean={vis.mean():.3f}")
    print(f"  {fmt_mem()}")
    print(f"  OK")
    return model


def test_3_vitpose_with_gradients(model):
    section("3. ViTPoseExtractor forward (WITH grad) on single frame")
    torch.cuda.reset_peak_memory_stats()
    frame = torch.rand(1, 3, 256, 288, device=DEVICE, requires_grad=True)
    kp, vis = model(frame)
    # Dummy loss
    loss = kp.abs().mean()
    loss.backward()
    grad_norm = frame.grad.norm().item() if frame.grad is not None else 0.0
    print(f"  loss: {loss.item():.4f}")
    print(f"  input grad norm: {grad_norm:.6f}  (should be > 0)")
    print(f"  {fmt_mem()}")
    assert grad_norm > 0, "Gradient did not flow back to input!"
    print(f"  OK")


def test_4_vitpose_with_checkpointing(model):
    section(f"4. ViTPoseExtractor on ALL {NUM_FRAMES} frames with gradient checkpointing")
    torch.cuda.reset_peak_memory_stats()
    frames = torch.rand(NUM_FRAMES, 3, 256, 288, device=DEVICE, requires_grad=True)
    from torch.utils.checkpoint import checkpoint

    all_kp = []
    all_vis = []
    for t in range(frames.shape[0]):
        frame_t = frames[t:t+1]
        kp_t, vis_t = checkpoint(model, frame_t, use_reentrant=False)
        all_kp.append(kp_t)
        all_vis.append(vis_t)
    gen_kp = torch.cat(all_kp, dim=0)
    gen_vis = torch.cat(all_vis, dim=0)

    # Dummy loss and backward
    loss = gen_kp.abs().mean()
    loss.backward()
    grad_norm = frames.grad.norm().item() if frames.grad is not None else 0.0
    print(f"  output: {tuple(gen_kp.shape)}")
    print(f"  loss: {loss.item():.4f}")
    print(f"  input grad norm: {grad_norm:.4f}  (should be > 0)")
    print(f"  {fmt_mem()}")
    assert grad_norm > 0, "Gradient did not flow back!"
    print(f"  OK")


def test_5_losses_against_reference(ref_kp, ref_vis):
    section("5. Loss functions on cached reference")
    if ref_kp is None:
        print("SKIP: no reference loaded")
        return

    # Self-loss should be ~0
    self_pos = skeleton_motion_loss(ref_kp, ref_vis, ref_kp, ref_vis)
    self_vel = skeleton_velocity_loss(ref_kp, ref_vis, ref_kp, ref_vis)
    print(f"  self position loss: {self_pos.item():.6f}  (should be ~0)")
    print(f"  self velocity loss: {self_vel.item():.6f}  (should be ~0)")

    # Noise
    noisy = ref_kp + torch.randn_like(ref_kp) * 0.1
    noisy_pos = skeleton_motion_loss(noisy, ref_vis, ref_kp, ref_vis)
    noisy_vel = skeleton_velocity_loss(noisy, ref_vis, ref_kp, ref_vis)
    print(f"  noisy (σ=0.1) position: {noisy_pos.item():.4f}  (should be > 0)")
    print(f"  noisy (σ=0.1) velocity: {noisy_vel.item():.4f}  (should be > 0)")
    assert self_pos.item() < 1e-4, "Self-loss too high"
    assert noisy_pos.item() > 0.01, "Noise sensitivity too low"
    print(f"  OK")


def test_6_wan_vae_roundtrip():
    section("6. Wan VAE encode+decode roundtrip (no gradients)")
    torch.cuda.reset_peak_memory_stats()

    # Load Wan VAE
    from diffusers import AutoencoderKLWan
    print("  Loading Wan VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae", torch_dtype=torch.bfloat16,
    ).to(DEVICE).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # Fake normalized latents (shape matches what trainer provides)
    lat = torch.randn(1, vae.config.z_dim, LATENT_T, LATENT_H, LATENT_W,
                      device=DEVICE, dtype=torch.bfloat16)
    print(f"  latent shape: {tuple(lat.shape)}")

    with torch.no_grad():
        frames = decode_wan_x0_to_frames(lat, vae)
    print(f"  decoded frames: {tuple(frames.shape)}  range=[{frames.min():.3f}, {frames.max():.3f}]")
    print(f"  {fmt_mem()}")

    del vae, lat, frames
    torch.cuda.empty_cache()
    print(f"  OK")


def test_7_vae_decode_with_gradients():
    section("7. Wan VAE decode with gradients (chunked backward accumulation)")
    torch.cuda.reset_peak_memory_stats()

    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae", torch_dtype=torch.bfloat16,
    ).to(DEVICE).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    lat = torch.randn(1, vae.config.z_dim, LATENT_T, LATENT_H, LATENT_W,
                      device=DEVICE, dtype=torch.bfloat16, requires_grad=True)

    # Chunked decode: process K latent frames at a time, each chunk with
    # its own autograd graph. Run backward per chunk to accumulate gradients
    # into lat.grad without holding all activations simultaneously.
    chunk_size = 4  # latent frames per chunk
    lat_t = lat.shape[2]
    # Run dummy forward+backward per chunk with a simple loss to validate memory fits
    total_loss_val = 0.0
    for start in range(0, lat_t, chunk_size):
        end = min(start + chunk_size, lat_t)
        chunk = lat[:, :, start:end].contiguous()
        # Need to detach + require_grad for per-chunk grad computation
        # But we want the grad to flow back to lat, so use a grad_hook trick:
        # we compute grad wrt chunk and manually accumulate into lat.grad
        # The cleanest way: just run decode on the chunk and compute a partial loss
        frames_chunk = decode_wan_x0_to_frames(chunk, vae)
        chunk_loss = frames_chunk.abs().mean()
        chunk_loss.backward()
        total_loss_val += chunk_loss.item()
        del frames_chunk, chunk_loss
        torch.cuda.empty_cache()
    grad_norm = lat.grad.norm().item() if lat.grad is not None else 0.0
    print(f"  chunks processed: {lat_t // chunk_size + (1 if lat_t % chunk_size else 0)}")
    print(f"  avg chunk loss: {total_loss_val / max(1, lat_t // chunk_size):.4f}")
    print(f"  latent grad norm: {grad_norm:.6f}  (should be > 0)")
    print(f"  {fmt_mem()}")
    assert grad_norm > 0, "Gradient did not flow through VAE"

    del vae, lat
    torch.cuda.empty_cache()
    print(f"  OK")


def test_8_full_chain(ref_kp, ref_vis):
    section("8. Full chain: x0 → VAE → ViTPose → loss (chunked backward)")
    torch.cuda.reset_peak_memory_stats()

    from diffusers import AutoencoderKLWan
    from torch.utils.checkpoint import checkpoint
    vae = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae", torch_dtype=torch.bfloat16,
    ).to(DEVICE).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    vitpose = ViTPoseExtractor().to(DEVICE).eval()

    x0 = torch.randn(1, vae.config.z_dim, LATENT_T, LATENT_H, LATENT_W,
                     device=DEVICE, dtype=torch.bfloat16, requires_grad=True)

    # Chunked: decode K latent frames → ViTPose → loss → backward, per chunk.
    # Each chunk's autograd graph is freed after its backward, keeping memory bounded.
    chunk_size = 4
    lat_t = x0.shape[2]
    total_loss = 0.0
    n_chunks = 0
    # How many pixel frames each latent frame expands to on average (Wan 4x temporal, -3 for first-chunk offset)
    pixel_frames_per_latent = NUM_FRAMES / lat_t

    for start in range(0, lat_t, chunk_size):
        end = min(start + chunk_size, lat_t)
        chunk = x0[:, :, start:end].contiguous()
        # Decode chunk → (1, 3, T_chunk, H, W)
        frames_chunk = decode_wan_x0_to_frames(chunk, vae)
        # Per-frame ViTPose with checkpointing
        frames_flat = frames_chunk[0].permute(1, 0, 2, 3)  # (T, 3, H, W)
        kp_list, vis_list = [], []
        for t in range(frames_flat.shape[0]):
            ft = frames_flat[t:t+1]
            kp_t, vis_t = checkpoint(vitpose, ft, use_reentrant=False)
            kp_list.append(kp_t); vis_list.append(vis_t)
        gen_kp = torch.cat(kp_list, dim=0)
        gen_vis = torch.cat(vis_list, dim=0)
        # Dummy loss (doesn't depend on visibility so gradient always flows)
        chunk_loss = gen_kp.pow(2).mean()
        chunk_loss.backward()
        total_loss += chunk_loss.item()
        n_chunks += 1
        del frames_chunk, frames_flat, kp_list, vis_list, gen_kp, gen_vis, chunk_loss
        torch.cuda.empty_cache()

    grad_norm = x0.grad.norm().item() if x0.grad is not None else 0.0
    print(f"  chunks: {n_chunks} (chunk_size={chunk_size} latent frames)")
    print(f"  avg chunk loss: {total_loss / n_chunks:.4f}")
    print(f"  x0 grad norm: {grad_norm:.6f}  (should be > 0)")
    print(f"  {fmt_mem()}  ← PEAK")
    assert grad_norm > 0, "Full chain gradient is zero!"
    print(f"  OK — chunked decode+ViTPose with full gradient backprop fits in memory")


def main():
    print(f"Pipeline validation on {DEVICE}")
    print(f"Target: {NUM_FRAMES} frames, latent shape (1, 16, {LATENT_T}, {LATENT_H}, {LATENT_W})")

    ref_kp, ref_vis = test_1_load_reference()
    model = test_2_vitpose_forward()
    test_3_vitpose_with_gradients(model)
    test_4_vitpose_with_checkpointing(model)
    test_5_losses_against_reference(ref_kp, ref_vis)
    del model
    torch.cuda.empty_cache()

    test_6_wan_vae_roundtrip()
    test_7_vae_decode_with_gradients()

    if ref_kp is not None:
        test_8_full_chain(ref_kp, ref_vis)

    print("\nAll validation stages passed.")


if __name__ == '__main__':
    main()
