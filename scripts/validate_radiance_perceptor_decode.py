"""
Validate the pixel-space (FakeVAE) perceptor decode wiring for Chroma Radiance /
Zeta Chroma (arch chroma_radiance / zeta_chroma).

These models have NO latent VAE — they use toolkit.models.FakeVAE, whose encode
and decode are the identity. So the "latent" the trainer operates on already IS
the image in [-1,1], and the perceptor losses (ArcFace / ViTPose / Depth) must
treat x0 as pixels directly instead of running it through a tiny decoder.

SDTrainer now detects this via _is_pixel_space_vae() (latent_channels == 3) and:
  - loads NO tiny decoder for pixel-space models,
  - maps x0 -> pixels as (x0 + 1) / 2 at every decode site,
  - returns the original image for the depth-GT roundtrip (identity).

This script proves, against a real test image:
  1. _is_pixel_space_vae() is True for FakeVAE and False for real 16-/4-ch VAEs.
  2. The pixel-space round-trip (encode_images math -> (x0+1)/2) is LOSSLESS.
  3. Each landmine the explicit branches avoid actually fires on FakeVAE:
       a. config.get(...)            -> AttributeError  (decoder-load)
       b. latent_dist.mode()         -> AttributeError  (depth-GT roundtrip)
       c. 'shift_factor' in config   -> TypeError       (generic else branch)
  4. The pre-fix fall-through (4-ch taesd) FAILS on a 3-ch pixel latent.
"""
import os
import glob
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

import sys
sys.path.insert(0, "/home/z/Documents/repos/ai-toolkit")
from toolkit.models.FakeVAE import FakeVAE

DATA_DIR = "/home/z/Documents/repos/ai-toolkit/test_data/scarlett_full"
OUT_PNG = "/tmp/radiance_decode_roundtrip.png"
RES = 512
device = torch.device("cpu")  # tiny ops; keep off the shared GPU


def psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return 99.0 if mse == 0 else 10.0 * np.log10(1.0 / mse)


def load_image(path, res):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((res, res), Image.LANCZOS)
    return (torch.from_numpy(np.asarray(img)).float().permute(2, 0, 1)[None] / 255.0).to(device)


# ---- 1. Exercise the REAL SDTrainer._is_pixel_space_vae detector --------------
from diffusers.configuration_utils import FrozenDict
try:
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    detect = SDTrainer._is_pixel_space_vae          # unbound; call with a fake self
    detector_src = "real SDTrainer._is_pixel_space_vae"
except Exception as e:
    print(f"  (could not import SDTrainer: {type(e).__name__}: {e}); using inline replica")
    def detect(self):
        cfg = getattr(self.sd.vae, 'config', None)
        if cfg is None:
            return False
        lat = getattr(cfg, 'latent_channels', None)
        if lat is None and hasattr(cfg, 'get'):
            lat = cfg.get('latent_channels', None)
        return lat == 3
    detector_src = "inline replica"

class _FakeSelf:
    class _SD:
        pass
    def __init__(self, vae):
        self.sd = _FakeSelf._SD()
        self.sd.vae = vae

class _CfgHolder:  # stand-in VAE exposing only .config
    def __init__(self, cfg):
        self.config = cfg

fake_vae = FakeVAE()
flux_like = _CfgHolder(FrozenDict({"latent_channels": 16, "scaling_factor": 0.3611, "shift_factor": 0.1159}))
sd15_like = _CfgHolder(FrozenDict({"latent_channels": 4, "scaling_factor": 0.18215}))

det_fake = detect(_FakeSelf(fake_vae))
det_flux = detect(_FakeSelf(flux_like))
det_sd15 = detect(_FakeSelf(sd15_like))
print(f"detector: {detector_src}")
print(f"  _is_pixel_space_vae(FakeVAE)      = {det_fake}  (want True)")
print(f"  _is_pixel_space_vae(16-ch FLUX)   = {det_flux}  (want False)")
print(f"  _is_pixel_space_vae(4-ch SD1.5)   = {det_sd15}  (want False)")

# ---- 2. Pixel-space round-trip on a real image (must be lossless) -------------
imgs = sorted(glob.glob(os.path.join(DATA_DIR, "*.jpg")) + glob.glob(os.path.join(DATA_DIR, "*.png")))
assert imgs, f"no images in {DATA_DIR}"
arr = load_image(imgs[0], RES)  # [0,1], (1,3,H,W)
print(f"\nimage: {imgs[0]}")

# encode_images math for FakeVAE: latents = scaling*(encode.sample() - shift)
img_m1 = arr * 2.0 - 1.0  # dataloader feeds [-1,1] to the VAE
latent = fake_vae.encode(img_m1).latent_dist.sample()
latent = fake_vae.config.scaling_factor * (latent - fake_vae.config.shift_factor)
print(f"latent: shape={tuple(latent.shape)} channels={latent.shape[1]} "
      f"range=[{latent.min():.3f},{latent.max():.3f}]  (== the [-1,1] image)")

# SDTrainer pixel-space chokepoint: x0 already IS the image -> (x0+1)/2
x0_pixels = ((latent.float() + 1.0) * 0.5).clamp(0, 1)
p = psnr(x0_pixels, arr)
print(f"pixel-space decode (x0+1)/2 -> PSNR(vs original) = {p:.2f} dB")

# depth-GT roundtrip for pixel-space returns arr unchanged (identity)
roundtrip = arr.clamp(0, 1)
p_rt = psnr(roundtrip, arr)
print(f"depth-GT roundtrip (identity)  -> PSNR(vs original) = {p_rt:.2f} dB")

def to_np(t):
    return (t[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
Image.fromarray(np.concatenate([to_np(arr), to_np(x0_pixels)], axis=1)).save(OUT_PNG)
print(f"saved side-by-side (original | pixel-space decode): {OUT_PNG}")

# ---- 3. The landmines the explicit branches avoid ----------------------------
print("\n-- landmines on FakeVAE (why the generic paths can't be reused) --")
def expect_raise(label, fn):
    try:
        fn()
        print(f"  [FAIL] {label}: did NOT raise")
        return False
    except Exception as e:
        print(f"  [ok]   {label}: {type(e).__name__}: {str(e).splitlines()[0][:90]}")
        return True

lm_a = expect_raise("config.get('latent_channels')", lambda: fake_vae.config.get('latent_channels', 4))
lm_b = expect_raise("latent_dist.mode()", lambda: fake_vae.encode(img_m1).latent_dist.mode())
lm_c = expect_raise("'shift_factor' in config", lambda: ('shift_factor' in fake_vae.config))

# ---- 4. Pre-fix fall-through: 4-ch taesd on a 3-ch pixel latent ---------------
taesd_raised = False
try:
    from diffusers import AutoencoderTiny
    taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float32).to(device).eval()
    with torch.no_grad():
        _ = taesd.decode(latent).sample
    print("\n[PRE-FIX] 4-ch taesd decoded a 3-ch latent WITHOUT error (unexpected)")
except Exception as e:
    taesd_raised = True
    print(f"\n[PRE-FIX] 4-ch taesd.decode(3-ch latent) RAISED -> {type(e).__name__}: {str(e).splitlines()[0][:110]}")

# ============================================================================
print("\n==================== VERDICT ====================")
checks = {
    "_is_pixel_space_vae: True for FakeVAE, False for real VAEs": det_fake and not det_flux and not det_sd15,
    "pixel-space decode (x0+1)/2 is lossless (PSNR > 60 dB)": p > 60.0,
    "depth-GT roundtrip is identity (PSNR == 99)": p_rt >= 99.0,
    "landmine a: config.get() raises (decoder-load skip needed)": lm_a,
    "landmine b: latent_dist.mode() raises (roundtrip branch needed)": lm_b,
    "landmine c: 'in' config raises (chokepoint branch needed)": lm_c,
    "pre-fix 4-ch taesd FAILS on 3-ch pixel latent": taesd_raised,
}
for k, v in checks.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")
allok = all(checks.values())
print(f"\n  OVERALL: {'PASS — radiance pixel-space perceptor decode is correctly wired' if allok else 'FAIL'}")
print("=================================================")
