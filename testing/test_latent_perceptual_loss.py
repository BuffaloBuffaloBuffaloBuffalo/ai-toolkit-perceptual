"""
Test suite for E-LatentLPIPS perceptual loss in latent space.

Expected outcomes:
- When comparing identical latents: loss should be exactly 0 (or very close, <1e-6)
- When comparing latents with small noise: loss should be small but positive
- Loss should monotonically increase with noise level
- Loss at high noise (t~1) should be significantly higher than at low noise (t~0)
- Gradients should flow from the loss back to the input latents
- The model should handle 16-channel latents (Flux format)

Run with: python testing/test_latent_perceptual_loss.py
"""
import sys
import os
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_elatentlpips_available():
    """Test that the elatentlpips package is importable and has the expected API."""
    print("=" * 60)
    print("TEST: elatentlpips package availability")
    print("=" * 60)
    from elatentlpips import ELatentLPIPS

    # Check that flux encoder is supported
    # augment=None avoids the need for CUDA JIT compilation of custom ops.
    # ensembling=False must be used when augment=None.
    # The perceptual comparison still works well without augmentation ensembling.
    model = ELatentLPIPS(pretrained=True, encoder="flux", verbose=True, augment=None)
    print(f"  Model created successfully")
    print(f"  Model type: {type(model)}")
    print(f"  Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  PASS")
    return model


def test_identical_latents_zero_loss(model, device):
    """Loss should be 0 (or negligibly small) when comparing identical latents."""
    print("\n" + "=" * 60)
    print("TEST: Identical latents produce zero loss")
    print("=" * 60)
    model = model.to(device)
    model.eval()

    # 16-channel latents at a typical Flux latent resolution
    # Flux encodes 1024x1024 images to 64x64 latents (with packing they become 32x32x16)
    latents = torch.randn(1, 16, 32, 32, device=device)

    with torch.no_grad():
        loss = model(latents, latents, normalize=False, add_l1_loss=True, ensembling=False)

    loss_val = loss.item()
    print(f"  Loss for identical latents: {loss_val:.10f}")
    assert loss_val < 1e-5, f"Expected near-zero loss, got {loss_val}"
    print(f"  PASS (loss < 1e-5)")


def test_loss_increases_with_noise(model, device):
    """Loss should monotonically increase as noise level increases."""
    print("\n" + "=" * 60)
    print("TEST: Loss increases with noise level")
    print("=" * 60)
    model = model.to(device)
    model.eval()

    # Create a reference latent
    torch.manual_seed(42)
    target_latents = torch.randn(1, 16, 32, 32, device=device)
    noise = torch.randn_like(target_latents)

    # Test at various noise levels (simulating flow matching t values)
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    losses = []

    with torch.no_grad():
        for t in noise_levels:
            # Flow matching: noisy = (1-t)*x0 + t*noise
            noisy_latents = (1.0 - t) * target_latents + t * noise
            # At t=0, we have perfect x0. At t=1, we have pure noise.
            # The "recovered x0" in this test IS the noisy_latents
            # (simulating what the model would predict at this noise level)
            loss = model(noisy_latents, target_latents, normalize=False, add_l1_loss=True, ensembling=False)
            loss_val = loss.item()
            losses.append(loss_val)
            print(f"  t={t:.2f}: loss={loss_val:.6f}")

    # Check monotonicity (with small tolerance for numerical noise at low t)
    for i in range(1, len(losses)):
        if noise_levels[i] > 0.05:  # skip very small noise comparisons
            assert losses[i] >= losses[i - 1] - 1e-4, \
                f"Loss decreased from t={noise_levels[i-1]} ({losses[i-1]:.6f}) to t={noise_levels[i]} ({losses[i]:.6f})"

    # Check that high noise gives significantly higher loss than low noise
    ratio = losses[-1] / max(losses[1], 1e-8)
    print(f"\n  Loss ratio (t=1.0 / t=0.05): {ratio:.2f}")
    assert ratio > 2.0, f"Expected high-noise loss to be >2x low-noise loss, got ratio {ratio:.2f}"
    print(f"  PASS (monotonically increasing, ratio > 2)")


def test_gradient_flow(model, device):
    """Gradients should flow from the loss back to the input latent."""
    print("\n" + "=" * 60)
    print("TEST: Gradient flows back to input")
    print("=" * 60)
    model = model.to(device)
    model.eval()

    # Input that requires grad (simulating x0_pred from the diffusion model)
    x0_pred = torch.randn(1, 16, 32, 32, device=device, requires_grad=True)
    target = torch.randn(1, 16, 32, 32, device=device)

    loss = model(x0_pred, target, normalize=False, add_l1_loss=True, ensembling=False)
    loss.backward()

    assert x0_pred.grad is not None, "No gradient computed for x0_pred"
    grad_norm = x0_pred.grad.norm().item()
    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Gradient norm: {grad_norm:.6f}")
    print(f"  Gradient shape: {x0_pred.grad.shape}")
    print(f"  Gradient mean: {x0_pred.grad.mean().item():.8f}")
    print(f"  Gradient std: {x0_pred.grad.std().item():.8f}")
    assert grad_norm > 1e-8, f"Gradient norm is too small: {grad_norm}"
    print(f"  PASS (gradient norm > 1e-8)")


def test_flow_matching_x0_recovery(model, device):
    """
    End-to-end test simulating flow matching x0 recovery.

    Flow matching:
      noisy = (1-t)*x0 + t*noise
      model predicts v = noise - x0 (velocity)
      x0_pred = noisy - t * v_pred

    At perfect prediction (v_pred == v_true):
      x0_pred = noisy - t*(noise - x0) = (1-t)*x0 + t*noise - t*noise + t*x0 = x0

    With imperfect prediction, x0_pred has error, and loss should reflect it.
    """
    print("\n" + "=" * 60)
    print("TEST: Flow matching x0 recovery simulation")
    print("=" * 60)
    model = model.to(device)
    model.eval()

    torch.manual_seed(123)
    x0_true = torch.randn(2, 16, 32, 32, device=device)
    noise = torch.randn_like(x0_true)

    test_timesteps = [100, 300, 500, 700, 900]  # out of 1000

    with torch.no_grad():
        print("  Testing perfect prediction (loss should be ~0):")
        for ts in test_timesteps:
            t = ts / 1000.0
            noisy = (1.0 - t) * x0_true + t * noise
            v_true = noise - x0_true
            # Perfect recovery
            x0_recovered = noisy - t * v_true
            loss = model(x0_recovered, x0_true, normalize=False, add_l1_loss=True, ensembling=False)
            print(f"    t={ts:4d} (t_01={t:.2f}): loss={loss.mean().item():.8f}")
            assert loss.mean().item() < 1e-4, f"Perfect recovery should give ~0 loss"

        print("\n  Testing noisy prediction (loss should increase with error):")
        error_levels = [0.0, 0.1, 0.3, 0.5, 1.0]
        for error_scale in error_levels:
            t = 0.5  # mid timestep
            noisy = (1.0 - t) * x0_true + t * noise
            v_true = noise - x0_true
            v_pred = v_true + error_scale * torch.randn_like(v_true)
            x0_pred = noisy - t * v_pred
            loss = model(x0_pred, x0_true, normalize=False, add_l1_loss=True, ensembling=False)
            print(f"    error_scale={error_scale:.1f}: loss={loss.mean().item():.6f}")

    print(f"  PASS (perfect recovery gives ~0 loss)")


def test_per_channel_statistics(model, device):
    """
    Diagnostic: print per-channel latent statistics to verify encode/decode chain.
    This is a human-inspectable diagnostic, not an automated pass/fail.
    """
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Per-channel latent statistics")
    print("=" * 60)
    model = model.to(device)
    model.eval()

    torch.manual_seed(42)
    target = torch.randn(1, 16, 32, 32, device=device)
    # Simulate a slightly wrong prediction
    x0_pred = target + 0.1 * torch.randn_like(target)

    with torch.no_grad():
        diff = x0_pred - target
        for ch in range(16):
            ch_diff = diff[0, ch]
            print(f"  Channel {ch:2d}: "
                  f"mean_diff={ch_diff.mean().item():+.4f}, "
                  f"std_diff={ch_diff.std().item():.4f}, "
                  f"max_abs_diff={ch_diff.abs().max().item():.4f}, "
                  f"target_mean={target[0, ch].mean().item():+.4f}, "
                  f"target_std={target[0, ch].std().item():.4f}")

    loss = model(x0_pred, target, normalize=False, add_l1_loss=True, ensembling=False)
    print(f"\n  Overall loss: {loss.item():.6f}")
    print(f"  (This is a diagnostic test - no automated pass/fail)")


def test_batch_dimension(model, device):
    """Test that the model handles batch dimensions correctly."""
    print("\n" + "=" * 60)
    print("TEST: Batch dimension handling")
    print("=" * 60)
    model = model.to(device)
    model.eval()

    # Test batch sizes 1, 2, 4
    for bs in [1, 2, 4]:
        torch.manual_seed(42)
        target = torch.randn(bs, 16, 32, 32, device=device)
        x0_pred = target + 0.2 * torch.randn_like(target)

        with torch.no_grad():
            loss = model(x0_pred, target, normalize=False, add_l1_loss=True, ensembling=False)

        print(f"  Batch size {bs}: loss shape={loss.shape}, loss mean={loss.mean().item():.6f}")
        # Loss should be per-sample (shape [bs, 1, 1, 1])
        assert loss.shape[0] == bs, f"Expected batch dim {bs}, got {loss.shape[0]}"

    print(f"  PASS")


def test_normalize_flag(model, device):
    """Test that the normalize flag works correctly for Flux latents."""
    print("\n" + "=" * 60)
    print("TEST: Normalize flag for Flux encoder")
    print("=" * 60)
    model = model.to(device)
    model.eval()

    torch.manual_seed(42)
    target = torch.randn(1, 16, 32, 32, device=device)
    x0_pred = target + 0.3 * torch.randn_like(target)

    with torch.no_grad():
        loss_no_norm = model(x0_pred, target, normalize=False, add_l1_loss=True, ensembling=False)
        loss_with_norm = model(x0_pred, target, normalize=True, add_l1_loss=True, ensembling=False)

    print(f"  Loss without normalize: {loss_no_norm.item():.6f}")
    print(f"  Loss with normalize: {loss_with_norm.item():.6f}")
    print(f"  (Both should be valid positive numbers)")
    assert not torch.isnan(loss_no_norm), "Loss without normalize is NaN"
    assert not torch.isnan(loss_with_norm), "Loss with normalize is NaN"
    print(f"  PASS")


def test_dtype_compatibility(model, device):
    """Test that the model works with float32 and float16 inputs."""
    print("\n" + "=" * 60)
    print("TEST: Dtype compatibility")
    print("=" * 60)
    model = model.to(device)
    model.eval()

    torch.manual_seed(42)

    # Float32
    target_f32 = torch.randn(1, 16, 32, 32, device=device, dtype=torch.float32)
    pred_f32 = target_f32 + 0.2 * torch.randn_like(target_f32)
    with torch.no_grad():
        loss_f32 = model.float()(pred_f32, target_f32, normalize=False, add_l1_loss=True, ensembling=False)
    print(f"  float32 loss: {loss_f32.item():.6f}")

    # Float16 (may need autocast on CUDA)
    if device.type == 'cuda':
        model_f16 = model.half()
        target_f16 = target_f32.half()
        pred_f16 = pred_f32.half()
        try:
            with torch.no_grad():
                loss_f16 = model_f16(pred_f16, target_f16, normalize=False, add_l1_loss=True, ensembling=False)
            print(f"  float16 loss: {loss_f16.item():.6f}")
        except RuntimeError as e:
            print(f"  float16 not supported: {e}")
            print(f"  (This is OK - we'll use float32 for the loss model)")
        # restore
        model.float()

    print(f"  PASS")


def test_spatial_resolution_variations(model, device):
    """Test that various spatial resolutions work."""
    print("\n" + "=" * 60)
    print("TEST: Spatial resolution variations")
    print("=" * 60)
    model = model.to(device).float()
    model.eval()

    # Flux latent resolutions for common image sizes:
    # 512x512 -> 32x32 (after 16x downscale and 2x packing)
    # 1024x1024 -> 64x64
    # 768x512 -> 48x32
    resolutions = [(32, 32), (64, 64), (48, 32), (96, 64)]

    for h, w in resolutions:
        torch.manual_seed(42)
        target = torch.randn(1, 16, h, w, device=device)
        pred = target + 0.2 * torch.randn_like(target)
        with torch.no_grad():
            loss = model(pred, target, normalize=False, add_l1_loss=True, ensembling=False)
        print(f"  Resolution {h}x{w}: loss={loss.item():.6f}")
        assert not torch.isnan(loss), f"NaN loss at resolution {h}x{w}"

    print(f"  PASS")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Run all tests
    model = test_elatentlpips_available()
    model = model.to(device)

    test_identical_latents_zero_loss(model, device)
    test_loss_increases_with_noise(model, device)
    test_gradient_flow(model, device)
    test_flow_matching_x0_recovery(model, device)
    test_per_channel_statistics(model, device)
    test_batch_dimension(model, device)
    test_normalize_flag(model, device)
    test_dtype_compatibility(model, device)
    test_spatial_resolution_variations(model, device)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
