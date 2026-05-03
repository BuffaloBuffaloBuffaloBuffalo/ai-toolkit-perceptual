#!/usr/bin/env python3
"""End-to-end smoke test for the v3 depth cache (VAE-roundtrip GT).

Loads the actual Flux 2 VAE + TAEF2 decoder, mirrors the SDTrainer closure
``_vae_roundtrip_for_depth``, and runs:

    raw pixels (test_data/scarlett_full)
        → arr in [0,1]
        → vae.encode (Flux 2 returns Tensor (B, 128, h, w))
        → 128 → 32 unpack via einops
        → TAEF2 decode → roundtrip pixels
        → DA2(raw)  vs  DA2(roundtrip)  ≡  the new "best achievable" floor

For each image we report SSI-L1(DA2(raw), DA2(roundtrip)) — that's exactly
the loss the *old* v2 cache was applying as a floor at every step. v3 makes
that loss reachable as zero by storing DA2(roundtrip) as the GT.

Saves overlay tiles to output/depth_v3_validation/ for visual inspection.

Usage: python scripts/validate_depth_v3_roundtrip.py [--n 4]
"""

import argparse
import os
import sys
import warnings

import numpy as np
import torch
from PIL import Image, ImageDraw
from PIL.ImageOps import exif_transpose

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Silence the diffusers/transformers chatter so the test output stays clean.
warnings.filterwarnings("ignore")

from einops import rearrange  # noqa: E402

from extensions_built_in.diffusion_models.flux2.src.autoencoder import (  # noqa: E402
    AutoEncoder, AutoEncoderParams,
)
from toolkit.depth_consistency import (  # noqa: E402
    DifferentiableDepthEncoder, ssi_l1,
)

TEST_DIR = "test_data/scarlett_full"
OUT_DIR = "output/depth_v3_validation"


def load_flux2_vae(device, dtype):
    """Load Flux 2 VAE the same way Flux2Model does (HF download + state dict)."""
    import huggingface_hub
    from safetensors.torch import load_file

    vae_path = huggingface_hub.hf_hub_download(
        repo_id="ai-toolkit/flux2_vae",
        filename="ae.safetensors",
    )
    with torch.device("meta"):
        vae = AutoEncoder(AutoEncoderParams())
    sd = load_file(vae_path, device="cpu")
    for k in sd:
        sd[k] = sd[k].to(dtype)
    vae.load_state_dict(sd, assign=True)
    vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def load_taef2(device, dtype):
    """Load the TAEF2 decoder as SDTrainer does (lines 896-916)."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file as load_sf
    from toolkit.taesd import Decoder as TAESDDecoder

    weights_path = hf_hub_download("madebyollin/taef2", "taef2.safetensors")
    sd = load_sf(weights_path)
    decoder_sd = {}
    for k, v in sd.items():
        if k.startswith("decoder.layers."):
            rest = k.replace("decoder.layers.", "")
            parts = rest.split(".", 1)
            idx = int(parts[0]) + 1
            remainder = "." + parts[1] if len(parts) > 1 else ""
            decoder_sd[str(idx) + remainder] = v
    decoder = TAESDDecoder(latent_channels=32, use_midblock_gn=True)
    decoder.load_state_dict(decoder_sd)
    decoder.to(device=device, dtype=dtype).eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    return decoder


def make_roundtrip_fn(vae, taef2, vae_dtype):
    """Mirror SDTrainer's _vae_roundtrip_for_depth closure exactly."""

    def roundtrip(arr: torch.Tensor) -> torch.Tensor:
        arr_norm = (arr * 2.0 - 1.0).to(vae_dtype)
        posterior = vae.encode(arr_norm)
        if hasattr(posterior, 'latent_dist'):
            raw_latent = posterior.latent_dist.mode()
        elif isinstance(posterior, torch.Tensor):
            raw_latent = posterior
        elif hasattr(posterior, 'sample') and callable(getattr(posterior, 'sample')):
            raw_latent = posterior.sample()
        else:
            raw_latent = posterior[0]
        # Flux 2: scaling/shift defaults are 1.0 / 0.0 (no .config attribute).
        scaled = raw_latent
        # Unpack 128 → 32 channels for TAEF2.
        if scaled.shape[1] != 32:
            scaled = rearrange(
                scaled, "b (c p1 p2) h w -> b c (h p1) (w p2)",
                c=32, p1=2, p2=2,
            )
        dec_dtype = next(taef2.parameters()).dtype
        pixels = taef2(scaled.to(dec_dtype)).float()
        return pixels.clamp(0, 1)

    return roundtrip


def da2_of_tensor(enc, arr, device):
    """arr is (1, 3, H, W) in [0,1]; returns DA2 depth on CPU."""
    with torch.no_grad():
        d = enc(arr.to(device))[0].float().cpu()
    return d


def depth_to_pil(d, size_wh):
    a = d.cpu().numpy()
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    a = np.clip((a - lo) / max(1e-6, hi - lo), 0, 1)
    return Image.fromarray((a * 255).astype(np.uint8)).resize(size_wh, Image.BICUBIC).convert("RGB")


def list_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in exts
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--dir", default=TEST_DIR)
    ap.add_argument("--bucket", type=int, default=512,
                    help="square bucket size to test the closure on")
    args = ap.parse_args()

    if not os.path.exists(args.dir):
        print(f"ERR: test dir {args.dir} not found")
        return 1

    paths = list_images(args.dir)[: args.n]
    if not paths:
        print(f"ERR: no images in {args.dir}")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16  # matches user's training run
    print(f"device={device} dtype={dtype}")

    print("Loading Flux 2 VAE...")
    vae = load_flux2_vae(device, dtype)
    print(f"  vae class={type(vae).__name__} dtype={next(vae.parameters()).dtype}")

    print("Loading TAEF2 decoder...")
    taef2 = load_taef2(device, dtype)
    print(f"  taef2 dtype={next(taef2.parameters()).dtype}")

    print("Loading DA2-Small (depth perceptor)...")
    enc = DifferentiableDepthEncoder(grad_checkpoint=False, device=device)

    roundtrip = make_roundtrip_fn(vae, taef2, dtype)

    rows = []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0][:30]
        raw_pil = exif_transpose(Image.open(p)).convert("RGB")
        # Resize to a square bucket the VAE can handle (16-divisible).
        side = (args.bucket // 16) * 16
        train_pil = raw_pil.resize((side, side), Image.BICUBIC)
        arr = torch.from_numpy(
            np.asarray(train_pil, dtype=np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(device)

        # Run the closure
        try:
            with torch.no_grad():
                rt = roundtrip(arr)
        except Exception as e:  # noqa: BLE001
            print(f"  CLOSURE FAILED on {stem}: {type(e).__name__}: {e}")
            return 2

        # Sanity: shapes and value range
        assert rt.shape == arr.shape, f"shape mismatch {rt.shape} != {arr.shape}"
        assert rt.dtype == torch.float32, f"unexpected dtype {rt.dtype}"
        assert 0.0 <= rt.min().item() <= rt.max().item() <= 1.0, \
            f"value range broken: [{rt.min():.3f}, {rt.max():.3f}]"

        # DA2 on raw vs roundtrip
        d_raw = da2_of_tensor(enc, arr, device)
        d_rt = da2_of_tensor(enc, rt, device)
        # Reshape to 2D for ssi_l1
        if d_raw.dim() == 3:
            d_raw = d_raw.squeeze(0)
        if d_rt.dim() == 3:
            d_rt = d_rt.squeeze(0)
        # Pixel-domain L1 between raw and roundtrip
        pix_l1 = (rt.cpu() - arr.cpu()).abs().mean().item()
        # SSI-L1 (depth) — this is the FLOOR of the live training loss before v3.
        ssi = ssi_l1(d_rt, d_raw)[0].item()
        rows.append((stem, ssi, pix_l1))

        # Save 4-panel overlay: raw RGB | roundtrip RGB | DA2(raw) | DA2(roundtrip)
        size = train_pil.size
        rt_pil = Image.fromarray(
            (rt[0].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        )
        panels = [train_pil, rt_pil, depth_to_pil(d_raw, size), depth_to_pil(d_rt, size)]
        labels = [
            "raw pixels", "VAE-roundtrip pixels",
            "DA2(raw)  (old v2 GT)", f"DA2(roundtrip) (new v3 GT)  SSI={ssi:.4f}",
        ]
        target_h = 480

        def fit(im):
            r = target_h / im.height
            return im.resize((int(im.width * r), target_h), Image.BICUBIC)

        panels = [fit(pn) for pn in panels]
        combo = Image.new("RGB", (sum(p.width for p in panels), target_h), (0, 0, 0))
        x = 0
        for pn, lbl in zip(panels, labels):
            combo.paste(pn, (x, 0))
            ImageDraw.Draw(combo).text((x + 6, 6), lbl, fill=(255, 255, 0))
            x += pn.width
        combo.save(os.path.join(OUT_DIR, f"{stem}.jpg"), quality=88)

    print(f"\nResults (overlays in {OUT_DIR}):\n")
    print(f"{'image':32s} {'SSI(raw vs RT)':>16s} {'pix L1':>10s}")
    for stem, ssi, pix in rows:
        print(f"{stem:32s} {ssi:16.4f} {pix:10.4f}")
    if rows:
        ssis = [r[1] for r in rows]
        print(f"\n  mean SSI = {sum(ssis)/len(ssis):.4f}  "
              f"(this is the loss-floor v3 eliminates by caching DA2 of the roundtrip)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
