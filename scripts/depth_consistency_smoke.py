"""Integration smoke test for toolkit/depth_consistency.py.

Exercises the production module directly (not the validation copy):
  1. Cache GT depth via ``cache_depth_gt_embeddings`` on real images.
  2. Round-trip through the safetensors cache.
  3. Run ``compute_depth_consistency_loss`` with and without a subject mask.
  4. Verify gradient flow from x0_pixels through DA2 to the loss.

Uses /test_data/scarlett_full since /bodytest is not on disk here.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from toolkit.config_modules import DepthConsistencyConfig
from toolkit.depth_consistency import (
    DifferentiableDepthEncoder,
    cache_depth_gt_embeddings,
    compute_depth_consistency_loss,
    render_depth_preview,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "test_data" / "scarlett_full"
TEST_NAMES = [
    "clare-bowen-blue-dress.jpg",
    "169996580-820x1214.jpg",
    "CLARE-BOWEN-at-Australians-In-Film-Awards-and-Benefit-Dinner-in-Century-City-2.jpg",
]
DEVICE = torch.device("cuda")


def make_file_items(paths):
    items = []
    for p in paths:
        it = SimpleNamespace()
        it.path = str(p)
        items.append(it)
    return items


def load_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def main() -> None:
    paths = [DATA_DIR / n for n in TEST_NAMES if (DATA_DIR / n).exists()]
    assert paths, f"no test images under {DATA_DIR}"
    print(f"[env] cuda={torch.cuda.is_available()}, {len(paths)} images")

    cfg = DepthConsistencyConfig(
        loss_weight=0.05, loss_min_t=0.0, loss_max_t=0.9,
        ssi_weight=1.0, grad_weight=0.5, grad_scales=4,
        mask_source="none",  # we're testing without subject mask cache here
    )

    # Wipe any prior cache so we exercise the fresh path.
    for p in paths:
        cache_file = Path(os.path.dirname(p)) / "_face_id_cache" / (p.stem + ".safetensors")
        if cache_file.exists():
            from safetensors.torch import load_file, save_file
            data = {k: v.clone() for k, v in load_file(str(cache_file)).items()}
            data.pop("depth_gt", None)
            data.pop("depth_gt_v1", None)
            save_file(data, str(cache_file))

    # --- 1. Caching pass ---
    file_items = make_file_items(paths)
    t0 = time.time()
    cache_depth_gt_embeddings(file_items, cfg, device=DEVICE)
    cache_time = time.time() - t0
    print(f"[cache] extracted {len(file_items)} GT depths in {cache_time:.1f}s")
    for it in file_items:
        assert hasattr(it, "depth_gt"), "cache did not set .depth_gt"
        assert it.depth_gt.dtype == torch.float16, it.depth_gt.dtype
        assert it.depth_gt.dim() == 2, it.depth_gt.shape
        print(f"  {Path(it.path).name:60s} depth {tuple(it.depth_gt.shape)} "
              f"range=[{float(it.depth_gt.min()):.2f}, {float(it.depth_gt.max()):.2f}]")

    # --- 2. Round-trip: re-cache should hit the cache_version key and skip work ---
    file_items2 = make_file_items(paths)
    t0 = time.time()
    cache_depth_gt_embeddings(file_items2, cfg, device=DEVICE)
    roundtrip_time = time.time() - t0
    print(f"[cache] round-trip (cache hit) took {roundtrip_time:.2f}s")
    for it1, it2 in zip(file_items, file_items2):
        assert torch.equal(it1.depth_gt, it2.depth_gt), "cache round-trip mismatch"
    print("[cache] round-trip identical")

    # --- 3. Loss compute with real training perceptor (with grad checkpointing) ---
    print("[loss] loading perceptor with grad_checkpoint=True")
    encoder = DifferentiableDepthEncoder(
        model_id=cfg.model_id,
        input_size=cfg.input_size,
        grad_checkpoint=cfg.grad_checkpoint,
        device=DEVICE,
    )

    # Cross-image: x0 from image 0, GT depth from image 1 — meaningful gradient.
    x0_pixels = load_tensor(paths[0])
    x0_pixels = x0_pixels.clone().detach().requires_grad_(True)
    gt_depth = file_items[1].depth_gt.unsqueeze(0)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    loss, ssi, grd, _, _ = compute_depth_consistency_loss(
        encoder, x0_pixels, gt_depth, mask=None,
        ssi_weight=cfg.ssi_weight,
        grad_weight=cfg.grad_weight,
        grad_scales=cfg.grad_scales,
    )
    loss.backward()
    torch.cuda.synchronize()
    fbp_time = time.time() - t0
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    assert x0_pixels.grad is not None, "no gradient on input"
    assert torch.isfinite(x0_pixels.grad).all(), "non-finite gradient"
    assert float(loss) > 0.1, f"cross-image loss should be meaningful, got {float(loss)}"
    assert x0_pixels.grad.abs().mean() > 1e-7, "cross-image grad too small"
    print(f"[loss] cross-image loss={float(loss):.4f}  ssi={float(ssi):.4f}  grad={float(grd):.4f}")
    print(f"[loss] fwd+bwd={fbp_time*1000:.0f}ms  peak_vram={peak_mb:.0f}MB  "
          f"|g|_mean={x0_pixels.grad.abs().mean():.2e}")

    # --- 4. Mask path: use a dummy subject-style mask (upper-half of image) ---
    H, W = x0_pixels.shape[-2:]
    mask = torch.zeros(1, H, W, device=DEVICE)
    mask[:, : H // 2, :] = 1.0
    x0_pixels2 = load_tensor(paths[0]).clone().detach().requires_grad_(True)
    loss_m, ssi_m, grd_m, _, _ = compute_depth_consistency_loss(
        encoder, x0_pixels2, file_items[1].depth_gt.unsqueeze(0), mask=mask,
        ssi_weight=cfg.ssi_weight,
        grad_weight=cfg.grad_weight,
        grad_scales=cfg.grad_scales,
    )
    loss_m.backward()
    print(f"[mask] loss={float(loss_m):.4f}  ssi={float(ssi_m):.4f}  grad={float(grd_m):.4f}  "
          f"|g|_mean={x0_pixels2.grad.abs().mean():.2e}")
    # Gradient should be heavier in masked region (upper half).
    g = x0_pixels2.grad.abs().mean(dim=(0, 1))  # (H, W)
    upper = g[: H // 2, :].mean().item()
    lower = g[H // 2 :, :].mean().item()
    print(f"[mask] upper-half |g| mean={upper:.2e}  lower-half |g| mean={lower:.2e}  "
          f"ratio={upper / max(lower, 1e-12):.2f}x")
    assert upper > lower, "masked region should have larger gradient magnitude"

    # --- 5. Self-loss sanity: pred-depth vs cached-depth of same image (should be tiny) ---
    x0_same = load_tensor(paths[0]).clone().detach().requires_grad_(True)
    loss_self, ssi_self, grd_self, _, _ = compute_depth_consistency_loss(
        encoder, x0_same, file_items[0].depth_gt.unsqueeze(0), mask=None,
        ssi_weight=cfg.ssi_weight, grad_weight=cfg.grad_weight,
        grad_scales=cfg.grad_scales,
    )
    print(f"[self] same image vs its own cached GT: loss={float(loss_self):.4f}  "
          f"ssi={float(ssi_self):.4f}  grad={float(grd_self):.4f}")
    # Note: not exactly zero because the GT was cached at original image resolution
    # and x0 is loaded at the same resolution, but the bicubic resampler path
    # is not perfectly idempotent. Should still be tiny.
    assert float(loss_self) < 0.01, f"self-loss should be small, got {float(loss_self)}"

    # --- 6. Preview render: use paired image + its own cached GT depth
    # (in production the SDTrainer always pairs file_items[i].path with
    # file_items[i].depth_gt, so GT RGB and GT depth match).
    from torchvision.transforms import functional as TF
    x0_p = load_tensor(paths[0]).clone().detach().requires_grad_(True)
    _, _, _, d_pred_p, d_gt_p = compute_depth_consistency_loss(
        encoder, x0_p, file_items[0].depth_gt.unsqueeze(0), mask=None,
        ssi_weight=cfg.ssi_weight, grad_weight=cfg.grad_weight,
        grad_scales=cfg.grad_scales,
    )
    pred_pil = TF.to_pil_image(x0_p[0].detach().clamp(0, 1).cpu())
    ref_pil = Image.open(paths[0]).convert("RGB")
    combo = render_depth_preview(
        pred_pil, ref_pil,
        d_pred_p.squeeze(0) if d_pred_p.dim() == 3 else d_pred_p,
        d_gt_p.squeeze(0) if d_gt_p.dim() == 3 else d_gt_p,
    )
    out_dir = DATA_DIR / "depth_validation"
    out_dir.mkdir(exist_ok=True)
    preview_path = out_dir / "smoke_preview.jpg"
    combo.save(preview_path)
    print(f"[preview] 4-panel composite: {combo.size} -> {preview_path}")

    print("\n[done] all smoke checks passed")


if __name__ == "__main__":
    main()
