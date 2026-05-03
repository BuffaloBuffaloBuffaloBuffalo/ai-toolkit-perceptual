"""
Validate a Depth-Anything-V2-Small consistency loss for diffusion training.

What this script does:
  1. Loads DA2-Small (frozen, bf16, grad-checkpointed) with a pure-tensor
     differentiable preprocessor (NO HF ImageProcessor, which detaches).
  2. Runs it on a handful of images from test_data/scarlett_full and writes
     colormapped depth PNGs for eyeballing.
  3. Exercises the proposed loss — scale-and-shift-invariant L1 +
     multi-scale gradient matching — on pairs of real images.
  4. Verifies gradients flow back to the input pixels (differentiability).
  5. Reports peak VRAM and wall-clock time per forward/backward.

Run:
    python scripts/depth_loss_validation.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import DepthAnythingForDepthEstimation

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "test_data" / "scarlett_full"
OUT_DIR = DATA_DIR / "depth_validation"
OUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda"
DTYPE = torch.bfloat16
DA2_ID = "depth-anything/Depth-Anything-V2-Small-hf"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 518

TEST_IMAGES = [
    "clare-bowen-blue-dress.jpg",
    "169996580-820x1214.jpg",
    "beauty-trends-blogs-daily-beauty-reporter-2013-05-02-clare-bowen-allure.webp",
    "029d1e9f75a25059d4839d8ab3de4eac.jpg",
    "633493_v9_bc.jpg",
    "CLARE-BOWEN-at-Australians-In-Film-Awards-and-Benefit-Dinner-in-Century-City-2.jpg",
]


class DA2SmallPerceptor(torch.nn.Module):
    """Frozen Depth-Anything-V2-Small with a pure-tensor preprocessor.

    Inputs expected: (B, 3, H, W) float in [0, 1] (or [-1, 1], auto-detected).
    Gradients flow through preprocessing and the full DA2 forward.
    """

    def __init__(
        self,
        model_id: str = DA2_ID,
        input_size: int = INPUT_SIZE,
        dtype: torch.dtype = DTYPE,
        device: str = DEVICE,
        grad_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.model = DepthAnythingForDepthEstimation.from_pretrained(
            model_id, torch_dtype=dtype
        ).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        if grad_checkpoint:
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.model.gradient_checkpointing_enable()
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        )
        self.input_size = input_size

    def _aspect_preserving_hw(self, H: int, W: int) -> tuple[int, int]:
        if H >= W:
            new_h = self.input_size
            new_w = max(14, int(round(W * self.input_size / H / 14)) * 14)
        else:
            new_w = self.input_size
            new_h = max(14, int(round(H * self.input_size / W / 14)) * 14)
        return new_h, new_w

    def preprocess(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.min().item() < -0.05:
            pixels = (pixels + 1.0) * 0.5
        pixels = pixels.clamp(0.0, 1.0)
        _, _, H, W = pixels.shape
        new_h, new_w = self._aspect_preserving_hw(H, W)
        x = F.interpolate(
            pixels,
            size=(new_h, new_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        x = x.to(self.mean.dtype)
        x = (x - self.mean) / self.std
        return x.to(next(self.model.parameters()).dtype)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(pixels)
        out = self.model(pixel_values=x)
        return out.predicted_depth.float()


def ssi_l1(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale-and-shift-invariant L1 (MiDaS / Ranftl et al.).

    Args:
      pred, target: (B, H, W) fp32. Gradients expected through `pred`.
      mask: (B, H, W) fp32 in [0,1]. If None, full mask.

    Returns: (loss, scale, shift)
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    if mask is None:
        mask = torch.ones_like(pred)
    elif mask.dim() == 2:
        mask = mask.unsqueeze(0)
    p = pred.flatten(1)
    g = target.flatten(1)
    m = mask.flatten(1).float()
    n = m.sum(dim=1).clamp_min(1.0)
    mean_p = (p * m).sum(1) / n
    mean_g = (g * m).sum(1) / n
    var_p = (p * p * m).sum(1) / n - mean_p * mean_p
    cov_pg = (p * g * m).sum(1) / n - mean_p * mean_g
    s = cov_pg / var_p.clamp_min(1e-6)
    t = mean_g - s * mean_p
    aligned = s.view(-1, 1, 1) * pred + t.view(-1, 1, 1)
    diff = (aligned - target).abs() * mask
    loss = diff.sum() / mask.sum().clamp_min(1.0)
    return loss, s.detach(), t.detach()


def multiscale_grad_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    scales: int = 4,
) -> torch.Tensor:
    """Multi-scale L1 gradient-matching loss (MiDaS)."""
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    if mask is None:
        mask = torch.ones_like(pred)
    elif mask.dim() == 2:
        mask = mask.unsqueeze(0)
    loss = pred.new_zeros(())
    p, g, m = pred, target, mask.float()
    for k in range(scales):
        if k > 0:
            p = F.avg_pool2d(p.unsqueeze(1), 2).squeeze(1)
            g = F.avg_pool2d(g.unsqueeze(1), 2).squeeze(1)
            m = F.avg_pool2d(m.unsqueeze(1), 2).squeeze(1)
        diff = p - g
        mx = m[:, :, 1:] * m[:, :, :-1]
        my = m[:, 1:, :] * m[:, :-1, :]
        dx = (diff[:, :, 1:] - diff[:, :, :-1]).abs() * mx
        dy = (diff[:, 1:, :] - diff[:, :-1, :]).abs() * my
        loss = loss + (dx.sum() / mx.sum().clamp_min(1.0)) + (
            dy.sum() / my.sum().clamp_min(1.0)
        )
    return loss / scales


def load_image_as_tensor(path: Path, device: str = DEVICE) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return t.to(device)


def depth_to_png(depth: torch.Tensor, path: Path) -> None:
    """Save a (H, W) depth tensor as a grayscale 8-bit PNG, min-max normalized."""
    d = depth.detach().float().cpu().numpy()
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    dn = np.clip((d - lo) / max(1e-6, (hi - lo)), 0, 1)
    im = Image.fromarray((dn * 255).astype(np.uint8))
    im.save(path)


def side_by_side(img_tensor: torch.Tensor, depth: torch.Tensor, path: Path) -> None:
    """Save RGB | depth composite for visual inspection."""
    img = (img_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Himg, Wimg = img.shape[:2]
    d = depth.detach().float().cpu().numpy()
    if d.ndim == 3:
        d = d[0]
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    dn = np.clip((d - lo) / max(1e-6, (hi - lo)), 0, 1)
    d_img = (dn * 255).astype(np.uint8)
    d_pil = Image.fromarray(d_img).resize((Wimg, Himg), Image.BICUBIC)
    d_rgb = np.stack([np.asarray(d_pil)] * 3, axis=-1)
    combo = np.concatenate([img, d_rgb], axis=1)
    Image.fromarray(combo).save(path)


def vram_mb() -> float:
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def reset_peak() -> None:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def main() -> None:
    report: dict = {"images": [], "checks": {}}
    print(f"[setup] loading {DA2_ID} on {DEVICE}, dtype={DTYPE}")
    reset_peak()
    t0 = time.time()
    perceptor = DA2SmallPerceptor().to(DEVICE)
    load_time = time.time() - t0
    model_mem_mb = vram_mb()
    report["model_load_time_s"] = round(load_time, 3)
    report["model_vram_mb"] = round(model_mem_mb, 1)
    print(f"[setup] loaded in {load_time:.2f}s, VRAM after load = {model_mem_mb:.1f} MB")

    # 1. Forward pass on each test image — generate maps.
    depths: dict[str, torch.Tensor] = {}
    pixels: dict[str, torch.Tensor] = {}
    for name in TEST_IMAGES:
        path = DATA_DIR / name
        if not path.exists():
            print(f"[warn] missing: {path}")
            continue
        img = load_image_as_tensor(path)
        pixels[name] = img
        reset_peak()
        t = time.time()
        with torch.no_grad():
            d = perceptor(img)
        fwd_time = time.time() - t
        peak_mb = vram_mb()
        depths[name] = d
        out_png = OUT_DIR / (Path(name).stem + "__depth.png")
        combo_png = OUT_DIR / (Path(name).stem + "__rgb_depth.png")
        depth_to_png(d[0], out_png)
        side_by_side(img, d, combo_png)
        img_info = {
            "file": name,
            "input_hw": list(img.shape[-2:]),
            "depth_hw": list(d.shape[-2:]),
            "depth_min": round(float(d.min()), 4),
            "depth_max": round(float(d.max()), 4),
            "depth_mean": round(float(d.mean()), 4),
            "depth_std": round(float(d.std()), 4),
            "fwd_ms_nograd": round(fwd_time * 1000, 1),
            "peak_vram_mb_nograd": round(peak_mb, 1),
        }
        report["images"].append(img_info)
        print(
            f"[map] {name:55s} input={tuple(img.shape[-2:])} -> "
            f"depth={tuple(d.shape[-2:])} range=[{float(d.min()):.2f},"
            f" {float(d.max()):.2f}]  fwd={fwd_time*1000:.0f}ms  peak={peak_mb:.0f}MB"
        )

    if not depths:
        print("[fatal] no depths computed")
        return

    # 2. Self-loss should be (numerically) zero.
    print("\n[check] self-loss on identical depth maps (should be ~0)")
    name = TEST_IMAGES[0]
    d = depths[name][0]
    dzero_ssi, s, t = ssi_l1(d, d.detach())
    dzero_grad = multiscale_grad_loss(d, d.detach())
    print(f"  ssi={float(dzero_ssi):.6e}  s={float(s):.4f} t={float(t):.4f}  "
          f"grad={float(dzero_grad):.6e}")
    report["checks"]["self_loss_ssi"] = float(dzero_ssi)
    report["checks"]["self_loss_grad"] = float(dzero_grad)

    # 3. Cross-image loss: pred vs a DIFFERENT image's depth (should be > self-loss).
    print("\n[check] cross-image loss (should be >> self-loss)")
    cross_results = []
    names_list = list(depths.keys())
    pred_name = names_list[0]
    d_pred = depths[pred_name][0]
    for tgt_name in names_list[1:]:
        d_tgt = depths[tgt_name][0]
        if d_tgt.shape != d_pred.shape:
            d_tgt_resized = F.interpolate(
                d_tgt.unsqueeze(0).unsqueeze(0),
                size=d_pred.shape,
                mode="bilinear",
                align_corners=True,
            ).squeeze()
        else:
            d_tgt_resized = d_tgt
        ssi_loss, sc, sh = ssi_l1(d_pred, d_tgt_resized)
        g_loss = multiscale_grad_loss(d_pred, d_tgt_resized)
        cross_results.append(
            {
                "pred": pred_name,
                "target": tgt_name,
                "ssi": round(float(ssi_loss), 4),
                "grad": round(float(g_loss), 4),
                "scale": round(float(sc), 3),
                "shift": round(float(sh), 3),
            }
        )
        print(f"  {pred_name} vs {tgt_name:55s}  ssi={float(ssi_loss):.4f}  "
              f"grad={float(g_loss):.4f}")
    report["checks"]["cross_image"] = cross_results

    # 4. Perturbation sensitivity: add Gaussian noise to the input, check monotonic.
    print("\n[check] perturbation sensitivity (loss should grow with noise)")
    img = pixels[TEST_IMAGES[0]]
    with torch.no_grad():
        d_ref = perceptor(img)[0]
    sens = []
    for sigma in [0.0, 0.01, 0.05, 0.10, 0.20]:
        torch.manual_seed(0)
        noise = torch.randn_like(img) * sigma
        img_noisy = (img + noise).clamp(0, 1)
        with torch.no_grad():
            d_noisy = perceptor(img_noisy)[0]
        ssi_loss, _, _ = ssi_l1(d_noisy, d_ref.detach())
        g_loss = multiscale_grad_loss(d_noisy, d_ref.detach())
        sens.append(
            {"sigma": sigma, "ssi": round(float(ssi_loss), 5),
             "grad": round(float(g_loss), 5)}
        )
        print(f"  sigma={sigma:.2f}  ssi={float(ssi_loss):.5f}  "
              f"grad={float(g_loss):.5f}")
    report["checks"]["perturbation"] = sens

    # 5. Differentiability: backward pass, inspect input grad.
    print("\n[check] differentiability (backward must succeed and produce finite grad)")
    reset_peak()
    img = pixels[TEST_IMAGES[0]].clone().detach().requires_grad_(True)
    tgt_name = TEST_IMAGES[1]
    d_tgt = depths[tgt_name][0].detach()
    t0 = time.time()
    d_pred = perceptor(img)[0]
    if d_tgt.shape != d_pred.shape:
        d_tgt = F.interpolate(
            d_tgt.unsqueeze(0).unsqueeze(0),
            size=d_pred.shape,
            mode="bilinear",
            align_corners=True,
        ).squeeze()
    loss = ssi_l1(d_pred, d_tgt)[0] + 0.5 * multiscale_grad_loss(d_pred, d_tgt)
    loss.backward()
    torch.cuda.synchronize()
    bwd_time = time.time() - t0
    peak_bwd = vram_mb()
    g = img.grad
    assert g is not None, "no gradient flowed to input"
    grad_stats = {
        "loss": round(float(loss), 4),
        "grad_finite": bool(torch.isfinite(g).all()),
        "grad_abs_mean": float(g.abs().mean()),
        "grad_abs_max": float(g.abs().max()),
        "grad_nonzero_frac": float((g.abs() > 1e-10).float().mean()),
        "fwd_bwd_ms": round(bwd_time * 1000, 1),
        "peak_vram_mb_with_grad": round(peak_bwd, 1),
    }
    report["checks"]["differentiability"] = grad_stats
    print(
        f"  loss={grad_stats['loss']:.4f}  finite={grad_stats['grad_finite']}  "
        f"|grad|_mean={grad_stats['grad_abs_mean']:.2e}  "
        f"|grad|_max={grad_stats['grad_abs_max']:.2e}  "
        f"nonzero_frac={grad_stats['grad_nonzero_frac']:.3f}  "
        f"fwd+bwd={grad_stats['fwd_bwd_ms']:.0f}ms  peak={grad_stats['peak_vram_mb_with_grad']:.0f}MB"
    )

    # 6. Write report
    with open(OUT_DIR / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[done] report -> {OUT_DIR/'validation_report.json'}")
    print(f"[done] depth maps -> {OUT_DIR}")


if __name__ == "__main__":
    main()
