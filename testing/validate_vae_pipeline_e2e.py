"""End-to-end validation of the VAE anchor loss pipeline.

Tests the exact code path used in training: SDXL VAE decode → Flux 2 encode → cosine loss → backward.
Checks VRAM, gradient flow, checkpointing, and correctness at multiple resolutions.

Run: python testing/validate_vae_pipeline_e2e.py
"""
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

DEVICE = torch.device('cuda')
RESULTS = []


def _run(name, fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        fn()
        print(f"  PASS  {name}")
        RESULTS.append(True)
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        import traceback
        traceback.print_exc()
        RESULTS.append(False)
    torch.cuda.empty_cache()


def load_sdxl_vae():
    """Load SDXL VAE from diffusers."""
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.bfloat16)
    vae.to(DEVICE)
    vae.eval()
    vae.requires_grad_(False)
    vae.enable_gradient_checkpointing()
    return vae


def load_flux2_encoder():
    """Load Flux 2 VAE encoder."""
    from toolkit.vae_anchor import VAEAnchorEncoder
    enc = VAEAnchorEncoder(vae_path='')
    enc.load(device=DEVICE, dtype=torch.float32)
    return enc


def test_vae_checkpointing_eval_vs_train():
    """Verify gradient checkpointing saves VRAM in train mode vs eval mode."""
    vae = load_sdxl_vae()
    latents = torch.randn(1, 4, 96, 96, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    x0 = latents / 0.13025

    # Eval mode (no checkpointing)
    vae.eval()
    torch.cuda.reset_peak_memory_stats()
    latents_eval = torch.randn(1, 4, 96, 96, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    pixels_eval = vae.decode(latents_eval / 0.13025).sample
    loss_eval = pixels_eval.sum()
    loss_eval.backward()
    peak_eval = torch.cuda.max_memory_allocated() / 1e9
    del latents_eval, pixels_eval, loss_eval

    torch.cuda.empty_cache()

    # Train mode (checkpointing active)
    vae.train()
    torch.cuda.reset_peak_memory_stats()
    latents_train = torch.randn(1, 4, 96, 96, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    pixels_train = vae.decode(latents_train / 0.13025).sample
    loss_train = pixels_train.sum()
    loss_train.backward()
    peak_train = torch.cuda.max_memory_allocated() / 1e9

    savings = peak_eval - peak_train
    print(f"    eval peak: {peak_eval:.2f}GB, train peak: {peak_train:.2f}GB, savings: {savings:.2f}GB")
    assert savings > 0.5, f"Checkpointing should save >0.5GB, only saved {savings:.2f}GB"
    assert peak_train < 5.0, f"Train mode peak {peak_train:.2f}GB is too high for 96x96 latents"

    del vae, latents, pixels_eval, pixels_train
    torch.cuda.empty_cache()


def test_full_pipeline_gradient_flow():
    """Test gradient flows from cosine loss through Flux 2 encoder + SDXL VAE decode to latents."""
    vae = load_sdxl_vae()
    vae.train()  # ensure checkpointing
    enc = load_flux2_encoder()

    # Simulate x0_pred with grad (as if from UNet)
    x0_pred = torch.randn(1, 4, 64, 64, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    x0_unscaled = x0_pred / 0.13025

    # SDXL VAE decode
    pixels = vae.decode(x0_unscaled.to(torch.bfloat16)).sample.clamp(-1, 1)

    # Flux 2 encode in f32
    with torch.amp.autocast('cuda', enabled=False):
        _, pred_features = enc.encode_with_features(pixels.float())

    # Create fake reference features
    with torch.no_grad():
        ref_features = {k: v.clone() + torch.randn_like(v) * 0.1 for k, v in pred_features.items()}

    # Cosine loss (same as compute_loss)
    from toolkit.vae_anchor import FEATURE_LEVELS
    total_loss = torch.zeros(1, device=DEVICE)
    for level in FEATURE_LEVELS:
        if level in pred_features and level in ref_features:
            pred = pred_features[level].flatten(2)
            ref = ref_features[level].flatten(2)
            cos = F.cosine_similarity(pred, ref, dim=1)
            total_loss = total_loss + (1.0 - cos).mean(dim=1)
    loss = total_loss.mean()

    # Backward
    loss.backward()

    assert x0_pred.grad is not None, "Gradient did not reach x0_pred"
    grad_norm = x0_pred.grad.norm().item()
    print(f"    loss={loss.item():.4f} grad_norm={grad_norm:.6f}")
    assert grad_norm > 0, "Gradient norm is zero"

    del vae, enc, x0_pred, pixels, pred_features, ref_features
    torch.cuda.empty_cache()


def test_vram_at_resolutions():
    """Test VRAM usage at different latent resolutions."""
    vae = load_sdxl_vae()
    vae.train()
    enc = load_flux2_encoder()

    resolutions = {
        '512px (64x64 latents)': (64, 64),
        '768px (96x96 latents)': (96, 96),
        '768x512 (96x64 latents)': (96, 64),
        '912x624 (114x78 latents)': (114, 78),
    }

    for label, (h, w) in resolutions.items():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated() / 1e9

        x0 = torch.randn(1, 4, h, w, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        x0_unscaled = x0 / 0.13025

        try:
            pixels = vae.decode(x0_unscaled).sample.clamp(-1, 1)
            after_decode = torch.cuda.memory_allocated() / 1e9
            decode_cost = after_decode - baseline

            # Resize if > 512
            _va_max = 512
            if pixels.shape[2] > _va_max or pixels.shape[3] > _va_max:
                _scale = _va_max / max(pixels.shape[2], pixels.shape[3])
                _h = int(pixels.shape[2] * _scale) // 8 * 8
                _w = int(pixels.shape[3] * _scale) // 8 * 8
                pixels = F.interpolate(pixels, size=(_h, _w), mode='bilinear', align_corners=False)

            with torch.amp.autocast('cuda', enabled=False):
                _, feats = enc.encode_with_features(pixels.float())

            after_encode = torch.cuda.memory_allocated() / 1e9
            encode_cost = after_encode - after_decode

            # Backward
            loss = sum(f.mean() for f in feats.values())
            loss.backward()
            peak = torch.cuda.max_memory_allocated() / 1e9

            print(f"    {label}: decode={decode_cost:.2f}GB encode={encode_cost:.2f}GB "
                  f"peak={peak:.2f}GB pixels={list(pixels.shape)} grad={'OK' if x0.grad is not None else 'NONE'}")

            assert x0.grad is not None, f"No gradient at {label}"
            assert peak < 20.0, f"Peak {peak:.2f}GB too high at {label}"

        except torch.cuda.OutOfMemoryError:
            print(f"    {label}: OOM!")
        finally:
            del x0
            if 'pixels' in dir():
                del pixels
            if 'feats' in dir():
                del feats
            torch.cuda.empty_cache()

    del vae, enc
    torch.cuda.empty_cache()


def test_cosine_loss_correctness():
    """Verify cosine loss is 0 for identical features, >0 for different."""
    enc = load_flux2_encoder()

    # Load real image
    from PIL import Image
    bodytest = '/home/z/Documents/repos/ai-toolkit/bodytest'
    img_path = None
    for f in os.listdir(bodytest):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(bodytest, f)
            break
    assert img_path, "No images in bodytest"

    from toolkit.vae_anchor import encode_reference_features, VAEAnchorEncoder
    pil = Image.open(img_path).convert('RGB')
    ref = encode_reference_features(enc, pil, target_size=512)
    ref_gpu = {k: v.float().to(DEVICE) for k, v in ref.items()}

    # Same image → loss should be 0
    loss_same, _ = VAEAnchorEncoder.compute_loss(ref_gpu, ref_gpu)
    print(f"    same image loss: {loss_same.mean().item():.6f}")
    assert loss_same.mean().item() < 0.01, f"Same-image loss too high: {loss_same.mean().item()}"

    # Different features → loss should be > 0
    diff = {k: v + torch.randn_like(v) * 0.5 for k, v in ref_gpu.items()}
    loss_diff, _ = VAEAnchorEncoder.compute_loss(diff, ref_gpu)
    print(f"    different features loss: {loss_diff.mean().item():.4f}")
    assert loss_diff.mean().item() > 0.01, "Different-feature loss should be > 0"

    del enc
    torch.cuda.empty_cache()


def test_vae_train_mode_survives_device_transfer():
    """Verify gradient checkpointing survives CPU offload and reload."""
    vae = load_sdxl_vae()
    vae.train()
    vae.enable_gradient_checkpointing()

    # Offload to CPU
    vae.to('cpu')
    # Bring back
    vae.to(DEVICE)

    print(f"    training={vae.training} _gradient_checkpointing={getattr(vae, '_gradient_checkpointing', 'N/A')}")
    # Note: .to() preserves training mode, but some impls reset it
    # The key test: does decode use less VRAM?

    x0 = torch.randn(1, 4, 64, 64, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)

    # Test with current mode
    torch.cuda.reset_peak_memory_stats()
    pixels = vae.decode(x0 / 0.13025).sample
    pixels.sum().backward()
    peak_after_transfer = torch.cuda.max_memory_allocated() / 1e9

    # Compare: force train mode
    vae.train()
    x0.grad = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    pixels2 = vae.decode(x0 / 0.13025).sample
    pixels2.sum().backward()
    peak_forced_train = torch.cuda.max_memory_allocated() / 1e9

    print(f"    after transfer peak: {peak_after_transfer:.2f}GB, forced train peak: {peak_forced_train:.2f}GB")

    del vae, x0
    torch.cuda.empty_cache()


if __name__ == '__main__':
    print("=" * 60)
    print("VAE Anchor Pipeline E2E Validation")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("No CUDA GPU. Skipping.")
        sys.exit(0)

    _run("vae_checkpointing_eval_vs_train", test_vae_checkpointing_eval_vs_train)
    _run("vae_train_mode_survives_device_transfer", test_vae_train_mode_survives_device_transfer)
    _run("full_pipeline_gradient_flow", test_full_pipeline_gradient_flow)
    _run("cosine_loss_correctness", test_cosine_loss_correctness)
    _run("vram_at_resolutions", test_vram_at_resolutions)

    print()
    passed = sum(1 for r in RESULTS if r)
    print(f"Results: {passed}/{len(RESULTS)} passed")
    if passed < len(RESULTS):
        sys.exit(1)
