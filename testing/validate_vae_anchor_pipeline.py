"""Validate the VAE anchor loss pipeline end-to-end.

Simulates the actual training path in SDTrainer.py:
  1. Image -> SDXL VAE encode -> latents (ground truth)
  2. Image -> Flux 2 VAE encode -> reference features (cache simulation)
  3. Latents -> SDXL VAE decode -> pixels -> Flux 2 VAE encode -> pred features -> loss
  4. Verify loss ~0 for clean round-trip, increases with noise
  5. Verify gradients flow through the full chain
  6. Print timing and VRAM usage

In real training, gradient_checkpointing is enabled on the SDXL VAE decoder,
which is critical for VRAM. This script replicates that.
"""

import os
import sys
import time

import torch
import torch.nn.functional as F

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from PIL import Image
from PIL.ImageOps import exif_transpose
import torchvision.transforms.functional as TF


def get_vram_mb():
    """Current VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def get_peak_vram_mb():
    """Peak VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def fmt_vram(mb):
    return f"{mb:.0f} MB ({mb / 1024:.2f} GB)"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: No CUDA device found. Running on CPU (gradients will work, timing meaningless).")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    vram_start = get_vram_mb()
    print(f"VRAM at start: {fmt_vram(vram_start)}")

    # ---------------------------------------------------------------
    # Step 1: Load Flux 2 VAE encoder (VAEAnchorEncoder)
    # ---------------------------------------------------------------
    print("\n=== Step 1: Loading Flux 2 VAE encoder (VAEAnchorEncoder) ===")
    from toolkit.vae_anchor import VAEAnchorEncoder, FEATURE_LEVELS

    flux_encoder = VAEAnchorEncoder()  # auto-resolve path
    flux_encoder.load(device=device, dtype=torch.float32)
    print(f"  Loaded. VRAM: {fmt_vram(get_vram_mb())}")

    # ---------------------------------------------------------------
    # Step 2: Load SDXL VAE (the training VAE)
    # ---------------------------------------------------------------
    print("\n=== Step 2: Loading SDXL VAE (diffusers AutoencoderKL) ===")
    from diffusers import AutoencoderKL

    sdxl_vae = AutoencoderKL.from_pretrained(
        "stabilityai/sdxl-vae", torch_dtype=torch.float32
    )
    sdxl_vae = sdxl_vae.to(device)
    sdxl_vae.eval()
    # In training, the VAE decoder is NOT frozen -- gradients must flow through it.
    # But the weights themselves are not trained (requires_grad=False).
    # Gradient checkpointing is enabled to save VRAM (matches SDTrainer setup).
    sdxl_vae.requires_grad_(False)
    try:
        sdxl_vae.enable_gradient_checkpointing()
        sdxl_vae.train()  # required for gradient checkpointing to work
        print("  Gradient checkpointing: ENABLED (matches training)")
    except Exception as e:
        print(f"  WARNING: Could not enable gradient checkpointing: {e}")

    scaling_factor = sdxl_vae.config["scaling_factor"]
    shift_factor = sdxl_vae.config.get("shift_factor", None)
    print(f"  scaling_factor={scaling_factor}, shift_factor={shift_factor}")
    print(f"  VRAM: {fmt_vram(get_vram_mb())}")

    # ---------------------------------------------------------------
    # Step 3: Load a real image from /bodytest
    # ---------------------------------------------------------------
    print("\n=== Step 3: Loading test image ===")
    bodytest_dir = os.path.join(PROJECT_ROOT, "bodytest")
    img_path = None
    for f in sorted(os.listdir(bodytest_dir)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(bodytest_dir, f)
            break
    assert img_path is not None, f"No image found in {bodytest_dir}"
    print(f"  Using: {os.path.basename(img_path)}")

    pil_image = exif_transpose(Image.open(img_path)).convert("RGB")
    pil_image = pil_image.resize((512, 512))
    print(f"  Image size: {pil_image.size}")

    # Convert to tensor [-1, 1]
    img_tensor = TF.to_tensor(pil_image).unsqueeze(0).to(device)  # (1, 3, 512, 512) [0,1]
    img_pixels = img_tensor * 2.0 - 1.0  # [-1, 1]

    # ---------------------------------------------------------------
    # Step 4: Encode through SDXL VAE encoder -> ground-truth latents
    # ---------------------------------------------------------------
    print("\n=== Step 4: Encoding image through SDXL VAE encoder -> latents ===")
    with torch.no_grad():
        latent_dist = sdxl_vae.encode(img_pixels).latent_dist
        gt_latents = latent_dist.mean  # deterministic (no sampling noise)
        # Apply scaling: latents = scaling_factor * (latents - shift)
        # For SDXL, shift_factor is None so just scale
        shift = shift_factor if shift_factor is not None else 0
        gt_latents_scaled = scaling_factor * (gt_latents - shift)
    print(f"  Latent shape: {gt_latents_scaled.shape}")
    print(f"  Latent range: [{gt_latents_scaled.min().item():.4f}, {gt_latents_scaled.max().item():.4f}]")
    print(f"  VRAM: {fmt_vram(get_vram_mb())}")

    # ---------------------------------------------------------------
    # Step 5: Encode image through Flux 2 VAE encoder -> reference features
    # ---------------------------------------------------------------
    print("\n=== Step 5: Encoding image through Flux 2 VAE encoder -> reference features (cache sim) ===")
    with torch.no_grad():
        _, ref_features = flux_encoder.encode_with_features(img_pixels)
    # Move to CPU fp16 (matches cache_vae_anchor_features behavior)
    ref_features = {k: v.cpu().half() for k, v in ref_features.items()}
    print(f"  Feature levels: {list(ref_features.keys())}")
    for level_name, feat in ref_features.items():
        print(f"    {level_name}: shape={feat.shape}, dtype={feat.dtype}, range=[{feat.min().item():.4f}, {feat.max().item():.4f}]")
    print(f"  VRAM: {fmt_vram(get_vram_mb())}")

    # ---------------------------------------------------------------
    # Step 6: Round-trip: latents -> SDXL VAE decode -> pixels -> Flux 2 encode -> loss
    #         Simulates the exact training path from SDTrainer.py lines 2389-2406
    # ---------------------------------------------------------------
    print("\n=== Step 6: Clean round-trip (encode -> decode -> re-encode) ===")

    # Simulate x0_pred being the ground-truth scaled latents
    x0_pred = gt_latents_scaled.clone().detach().requires_grad_(True)

    # SDTrainer path: x0_unscaled = x0_pred / scaling_factor [+ shift_factor]
    x0_unscaled = x0_pred / scaling_factor
    if shift_factor is not None and shift_factor:
        x0_unscaled = x0_unscaled + shift_factor

    # Decode through SDXL VAE (with gradients flowing through)
    x0_vae_decoded = sdxl_vae.decode(x0_unscaled).sample.float()
    x0_vae_input = x0_vae_decoded.clamp(-1, 1)

    print(f"  Decoded pixel range: [{x0_vae_decoded.min().item():.4f}, {x0_vae_decoded.max().item():.4f}]")
    print(f"  Clamped pixel range: [{x0_vae_input.min().item():.4f}, {x0_vae_input.max().item():.4f}]")

    # Encode through Flux 2 VAE encoder (frozen, but gradients flow through activations)
    _, pred_features = flux_encoder.encode_with_features(x0_vae_input)
    print(f"  Pred feature levels: {list(pred_features.keys())}")

    # Compute loss against reference features
    clean_loss, clean_per_level = VAEAnchorEncoder.compute_loss(pred_features, ref_features)
    clean_loss_val = clean_loss.mean().item()
    print(f"\n  Clean round-trip loss: {clean_loss_val:.6f}")
    for level_name, val in clean_per_level.items():
        print(f"    {level_name}: {val:.6f}")
    print(f"  VRAM after forward: {fmt_vram(get_vram_mb())}")

    # Verify loss is small (not exactly 0 due to VAE reconstruction error)
    if clean_loss_val < 1.0:
        print("  PASS: Clean loss is reasonably small (< 1.0)")
    else:
        print(f"  WARNING: Clean loss is {clean_loss_val:.4f} -- larger than expected")

    # Free backward graph from step 6 before step 7
    del x0_pred, x0_unscaled, x0_vae_decoded, x0_vae_input, pred_features, clean_loss
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---------------------------------------------------------------
    # Step 7: Noisy round-trip: add noise to latents, verify loss increases
    # ---------------------------------------------------------------
    print("\n=== Step 7: Noisy round-trips (loss should increase with noise) ===")
    noise_levels = [0.0, 0.1, 0.5, 1.0, 2.0]
    losses = []
    for noise_scale in noise_levels:
        with torch.no_grad():
            noisy_latents = gt_latents_scaled + noise_scale * torch.randn_like(gt_latents_scaled)
            x0_uns = noisy_latents / scaling_factor
            if shift_factor is not None and shift_factor:
                x0_uns = x0_uns + shift_factor
            decoded = sdxl_vae.decode(x0_uns).sample.float().clamp(-1, 1)
            _, noisy_feats = flux_encoder.encode_with_features(decoded)
            noisy_loss, noisy_per_level = VAEAnchorEncoder.compute_loss(noisy_feats, ref_features)
            loss_val = noisy_loss.mean().item()
            losses.append(loss_val)
            print(f"  noise_scale={noise_scale:.1f}  loss={loss_val:.6f}  per_level={noisy_per_level}")

    # Verify monotonically increasing (with small tolerance)
    monotonic = all(losses[i] <= losses[i + 1] + 0.01 for i in range(len(losses) - 1))
    if monotonic:
        print("  PASS: Loss increases monotonically with noise")
    else:
        print("  WARNING: Loss is NOT monotonically increasing!")
        for i in range(1, len(losses)):
            delta = losses[i] - losses[i - 1]
            print(f"    delta[{noise_levels[i-1]:.1f}->{noise_levels[i]:.1f}]: {delta:+.6f}")

    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---------------------------------------------------------------
    # Step 8: Gradient flow test with dummy trainable parameter
    # ---------------------------------------------------------------
    print("\n=== Step 8: Gradient flow test (simulating LoRA parameter) ===")

    # Create a dummy "trainable" scale parameter (simulates a LoRA weight)
    dummy_param = torch.nn.Parameter(torch.ones(1, device=device))

    # x0_pred = dummy_param * gt_latents (so grads flow through dummy_param)
    # retain_grad on non-leaf so we can verify intermediate gradient flow too
    x0_pred_train = dummy_param * gt_latents_scaled.detach()
    x0_pred_train.retain_grad()

    # Forward through the full pipeline
    x0_uns = x0_pred_train / scaling_factor
    if shift_factor is not None and shift_factor:
        x0_uns = x0_uns + shift_factor

    decoded = sdxl_vae.decode(x0_uns).sample.float()
    pixels = decoded.clamp(-1, 1)
    _, pred_feats = flux_encoder.encode_with_features(pixels)
    loss_train, _ = VAEAnchorEncoder.compute_loss(pred_feats, ref_features)
    loss_scalar = loss_train.mean()

    print(f"  Loss value: {loss_scalar.item():.6f}")
    print(f"  VRAM before backward: {fmt_vram(get_vram_mb())}")

    # Backward
    loss_scalar.backward()

    print(f"  VRAM after backward: {fmt_vram(get_vram_mb())}")

    # Check gradient on dummy_param (the leaf "LoRA weight")
    grad_flows_to_param = False
    if dummy_param.grad is not None:
        grad_norm = dummy_param.grad.norm().item()
        print(f"  dummy_param.grad = {dummy_param.grad.item():.8f} (norm={grad_norm:.8f})")
        if grad_norm > 0:
            grad_flows_to_param = True
            print("  PASS: Gradient flows through the full chain to the trainable parameter")
        else:
            print("  FAIL: Gradient is zero -- no gradient flow!")
    else:
        print("  FAIL: dummy_param.grad is None -- backward did not reach it!")

    # Check intermediate gradient on x0_pred (the latent tensor)
    if x0_pred_train.grad is not None:
        latent_grad_norm = x0_pred_train.grad.norm().item()
        print(f"  x0_pred_train.grad norm: {latent_grad_norm:.8f}")
        print("  PASS: Gradients flow to the latent input")
    else:
        print("  INFO: No gradient on x0_pred_train (non-leaf; dummy_param grad is the key check)")
    print(f"  VRAM: {fmt_vram(get_vram_mb())}")

    # Free backward graph
    del x0_pred_train, x0_uns, decoded, pixels, pred_feats, loss_train, loss_scalar
    dummy_param.grad = None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---------------------------------------------------------------
    # Step 9: Timing for full forward+backward pass
    # ---------------------------------------------------------------
    print("\n=== Step 9: Timing full forward+backward pass ===")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    dummy_param2 = torch.nn.Parameter(torch.ones(1, device=device))

    def run_fwd_bwd():
        x0 = dummy_param2 * gt_latents_scaled.detach()
        x0_u = x0 / scaling_factor
        if shift_factor is not None and shift_factor:
            x0_u = x0_u + shift_factor
        dec = sdxl_vae.decode(x0_u).sample.float().clamp(-1, 1)
        _, pf = flux_encoder.encode_with_features(dec)
        lo, _ = VAEAnchorEncoder.compute_loss(pf, ref_features)
        lo.mean().backward()
        dummy_param2.grad = None

    # Warm up
    for _ in range(2):
        run_fwd_bwd()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Timed run
    N_RUNS = 5
    times = []
    for i in range(N_RUNS):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_fwd_bwd()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_ms = sum(times) / len(times) * 1000
    min_ms = min(times) * 1000
    max_ms = max(times) * 1000
    print(f"  Forward+backward ({N_RUNS} runs, 512x512 input):")
    print(f"    avg: {avg_ms:.1f} ms")
    print(f"    min: {min_ms:.1f} ms")
    print(f"    max: {max_ms:.1f} ms")

    # ---------------------------------------------------------------
    # Step 10: VRAM summary
    # ---------------------------------------------------------------
    print("\n=== Step 10: VRAM summary ===")
    peak_vram = get_peak_vram_mb()
    current_vram = get_vram_mb()
    print(f"  Current VRAM: {fmt_vram(current_vram)}")
    print(f"  Peak VRAM:    {fmt_vram(peak_vram)}")
    if peak_vram < 24 * 1024:
        print(f"  PASS: Peak VRAM ({peak_vram / 1024:.2f} GB) fits in 24GB GPU")
    else:
        print(f"  WARNING: Peak VRAM ({peak_vram / 1024:.2f} GB) exceeds 24GB GPU!")

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Clean round-trip loss:          {clean_loss_val:.6f}")
    print(f"  Loss at noise_scale=2.0:        {losses[-1]:.6f}")
    print(f"  Loss ratio (noisy/clean):       {losses[-1] / max(clean_loss_val, 1e-8):.1f}x")
    print(f"  Gradient flows to param:        {'YES' if grad_flows_to_param else 'NO'}")
    print(f"  Avg forward+backward time:      {avg_ms:.1f} ms")
    print(f"  Peak VRAM:                      {peak_vram / 1024:.2f} GB")
    print(f"  Fits 24GB GPU:                  {'YES' if peak_vram < 24 * 1024 else 'NO'}")

    all_pass = (
        clean_loss_val < 1.0
        and monotonic
        and grad_flows_to_param
        and peak_vram < 24 * 1024
    )
    print(f"\n  ALL CHECKS PASSED: {'YES' if all_pass else 'NO'}")

    # Cleanup
    flux_encoder.cleanup()
    del flux_encoder, sdxl_vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
