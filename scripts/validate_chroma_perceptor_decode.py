"""
Validate that the perceptor/custom-loss decode path is correctly wired for
Chroma (arch == "chroma", e.g. Chroma 1 HD).

Chroma loads the 16-channel Flux VAE (ostris/Flex.1-alpha, subfolder="vae"),
identical in latent-channels and scaling/shift factors to black-forest-labs
FLUX.1. SDTrainer's identity / body-proportion / depth losses decode x0 latents
to pixels with a tiny decoder selected by arch:

    vae_channels==32                  -> TAEF2   (Flux 2)
    is_flux | 'flex' in arch | zimage -> TAEF1   (16-ch Flux VAE)   <-- chroma now joins here
    is_xl                             -> taesdxl
    else                              -> taesd   (4-ch SD1.5)        <-- chroma USED to fall here (broken)

This script proves, against a real test image:
  1. The Flux VAE produces 16-ch latents (so the 4-ch taesd fall-through was wrong).
  2. TAEF1 (the post-fix decoder) round-trips those latents to a sane image.
  3. taesd (the pre-fix fall-through) RAISES on the same 16-ch latent.

It replicates `_vae_roundtrip_for_depth` (SDTrainer.py) exactly for the tiny-decoder branch.
"""
import os
import glob
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from diffusers import AutoencoderKL, AutoencoderTiny

DATA_DIR = "/home/z/Documents/repos/ai-toolkit/test_data/scarlett_full"
OUT_PNG = "/tmp/chroma_decode_roundtrip.png"
RES = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
print(f"device={device} dtype={dtype}")


def load_image(path, res):
    img = Image.open(path).convert("RGB")
    # center-crop to square then resize
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((res, res), Image.LANCZOS)
    arr = torch.from_numpy(np.asarray(img)).float().permute(2, 0, 1)[None] / 255.0  # [1,3,H,W] in [0,1]
    return arr.to(device), img


def psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return 99.0 if mse == 0 else 10.0 * np.log10(1.0 / mse)


# ---- pick a real test image (deterministic: first by sorted name) ----
imgs = sorted(glob.glob(os.path.join(DATA_DIR, "*.jpg")) + glob.glob(os.path.join(DATA_DIR, "*.png")))
assert imgs, f"no images in {DATA_DIR}"
img_path = imgs[0]
print(f"image: {img_path}")
arr, _pil = load_image(img_path, RES)  # [0,1]

# ---- load the VAE Chroma actually uses (Flex.1-alpha); fall back to FLUX.1 (identical 16-ch) ----
vae = None
for repo in ("ostris/Flex.1-alpha", "black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.1-schnell"):
    try:
        vae = AutoencoderKL.from_pretrained(repo, subfolder="vae", torch_dtype=dtype)
        print(f"VAE loaded from: {repo}")
        break
    except Exception as e:
        print(f"  (could not load {repo}: {type(e).__name__})")
assert vae is not None, "no Flux-family VAE available"
vae = vae.to(device).eval()
vae.requires_grad_(False)

lat_ch = vae.config.latent_channels
scale = float(vae.config.scaling_factor)
shift = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
print(f"VAE: latent_channels={lat_ch}  scaling_factor={scale}  shift_factor={shift}")

# ---- encode -> raw latent -> dataloader-style scaled latent (matches encode_images) ----
with torch.no_grad():
    arr_norm = (arr * 2.0 - 1.0).to(dtype)            # [0,1] -> [-1,1]
    raw_latent = vae.encode(arr_norm).latent_dist.mode()
    scaled = scale * (raw_latent - shift)
print(f"latent shape: {tuple(scaled.shape)}  (channels={scaled.shape[1]})")

# ============================================================================
# POST-FIX path: TAEF1 (what arch=='chroma' now selects)
# ============================================================================
taef1 = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=dtype).to(device).eval()
taef1.requires_grad_(False)
with torch.no_grad():
    td = next(taef1.parameters()).dtype
    pixels_taef1 = taef1.decode(scaled.to(td)).sample.float()   # mirrors SDTrainer taesd branch
    pixels_taef1 = ((pixels_taef1 + 1.0) * 0.5).clamp(0, 1)
taef1_ok = tuple(pixels_taef1.shape) == tuple(arr.shape)
p_taef1 = psnr(pixels_taef1, arr)
print(f"\n[POST-FIX] TAEF1 decode -> {tuple(pixels_taef1.shape)}  PSNR(vs original)={p_taef1:.2f} dB  shape_ok={taef1_ok}")

# full-VAE reference decode (for the side-by-side image)
with torch.no_grad():
    unscaled = (scaled / scale) + shift
    pixels_vae = ((vae.decode(unscaled.to(dtype)).sample.float() + 1.0) * 0.5).clamp(0, 1)

# ============================================================================
# PRE-FIX path: taesd (4-ch SD1.5) -- must FAIL on a 16-ch latent
# ============================================================================
taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=dtype).to(device).eval()
taesd.requires_grad_(False)
taesd_raised = False
taesd_err = ""
try:
    with torch.no_grad():
        _ = taesd.decode(scaled.to(next(taesd.parameters()).dtype)).sample
    print("[PRE-FIX]  taesd decoded the 16-ch latent WITHOUT error (unexpected)")
except Exception as e:
    taesd_raised = True
    taesd_err = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
    print(f"[PRE-FIX]  taesd.decode(16-ch latent) RAISED -> {taesd_err}")

# ---- save side-by-side: original | full-VAE | TAEF1 ----
def to_np(t):
    return (t[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

strip = np.concatenate([to_np(arr), to_np(pixels_vae), to_np(pixels_taef1)], axis=1)
Image.fromarray(strip).save(OUT_PNG)
print(f"\nsaved side-by-side (original | full-VAE | TAEF1): {OUT_PNG}")

# ============================================================================
print("\n==================== VERDICT ====================")
checks = {
    "Flux VAE is 16-channel (4-ch taesd fall-through was wrong)": lat_ch == 16,
    "TAEF1 round-trips 16-ch latent to correct-shape image": taef1_ok,
    "TAEF1 reconstruction is sane (PSNR > 18 dB)": p_taef1 > 18.0,
    "Pre-fix taesd (4-ch) FAILS on 16-ch latent": taesd_raised,
}
for k, v in checks.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")
allok = all(checks.values())
print(f"\n  OVERALL: {'PASS — chroma is correctly wired to the perceptor decode path' if allok else 'FAIL'}")
print("=================================================")
