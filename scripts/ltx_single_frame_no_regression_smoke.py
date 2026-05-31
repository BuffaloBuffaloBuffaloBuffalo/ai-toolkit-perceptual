"""Behavior-preservation tests for the LTX/Wan still-image (num_frames=1)
single-frame perceptor routing (depth + ArcFace identity + ViTPose body-prop).

The routing is gated so NON-LTX/Wan models (SD / SD3 / SDXL / Flux / Flux2 /
PixArt / ...) are completely unaffected. This proves that, CPU-only and with the
heavy perceptor models stubbed:

  1. the gate predicate ``arch.startswith(('ltx','wan'))`` is False for every
     image-latent arch (so the new ``elif _vae_wants_5d:`` dispatch branches are
     never taken) and True only for ltx2 / ltx2.3 / wan21 / wan22;
  2. ``cache_depth_gt_embeddings`` in its DEFAULT mode (store_as_single_frame_
     video=False) writes the unchanged 2D image GT — ``depth_gt_*`` key,
     ``CACHE_VERSION_KEY``, ``is_depth_cached`` (NOT the video key/flag);
  3. ``cache_video_identity_embeddings`` / ``cache_video_body_proportion_embeddings``
     in their DEFAULT mode (include_images=False) skip image items entirely —
     only ``is_video`` items are processed, exactly as before;
  4. the depth VAE roundtrip's 5D bridge is a no-op for a 4D (image-latent) VAE:
     no temporal axis is added before encode and none is squeezed after decode.

Each "logic" check is paired with a source-consistency assertion (the real gate /
guard lines must be present verbatim) so the test fails loudly if the gating is
ever weakened. CPU-only, never downloads a model.

Run: ``python scripts/ltx_single_frame_no_regression_smoke.py`` — exits 0 on success.
"""

from __future__ import annotations

import os
import re
import sys
import types
import tempfile

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_SDTRAINER = os.path.join(_HERE, "extensions_built_in", "sd_trainer", "SDTrainer.py")

# Every image-latent arch the toolkit supports (config_modules.ModelArch + flux2);
# all must stay on the unchanged 4D path. ltx/wan are the only 5D-latent archs.
IMAGE_ARCHS = [
    "sd1", "sd2", "sd3", "sdxl", "pixart", "pixart_sigma", "auraflow",
    "flux", "flex1", "flex2", "flux2", "lumina2", "vega", "ssd",
]
VIDEO_ARCHS = ["ltx2", "ltx2.3", "wan21", "wan22"]


def _src(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def test_1_arch_gate_predicate() -> None:
    """The gate decides image (4D, unchanged) vs LTX/Wan (5D, new) routing.

    Mirror SDTrainer's exact predicate AND assert that exact line is present in
    the source — so this test breaks if the gate is ever broadened to other
    archs."""
    gate_line = "_vae_wants_5d = (getattr(self.sd.model_config, 'arch', '') or '').startswith(('ltx', 'wan'))"
    assert gate_line in _src(_SDTRAINER), "gate predicate changed — re-verify this test"

    wants_5d = lambda a: (a or "").startswith(("ltx", "wan"))  # noqa: E731 — mirrors the gate
    for a in IMAGE_ARCHS:
        assert wants_5d(a) is False, f"{a} must stay on the unchanged 4D path"
    for a in VIDEO_ARCHS:
        assert wants_5d(a) is True, f"{a} must take the 5D path"
    # The dispatch branches are `elif _vae_wants_5d:` guards (False → original else)
    src = _src(_SDTRAINER)
    assert src.count("elif _vae_wants_5d:") == 2, "expected 2 gated dispatch branches (face, body)"
    assert "store_as_single_frame_video=_vae_wants_5d," in src, "depth dispatch must pass the gate"
    print("[1] arch gate: image-latent archs -> 4D path (False); ltx/wan -> 5D (True)   OK")


def test_2_depth_cache_default_mode_unchanged() -> None:
    """Real cache_depth_gt_embeddings, default mode (what every non-LTX model
    uses): 2D map under depth_gt_*, CACHE_VERSION_KEY, is_depth_cached — and
    none of the video-cube key/flag."""
    import toolkit.depth_consistency as dc

    class _StubDA2:
        def __init__(self, *a, **k):
            pass

        def __call__(self, arr):
            return torch.ones(arr.shape[0], 16, 20)  # (b, Hd, Wd); [0] -> (16,20)

    class _Cfg:
        model_id = "stub"
        input_size = 384
        pixel_blur_sigma = 0.0

    from toolkit.dataloader_mixins import DepthCachingFileItemDTOMixin as Mixin

    _orig = dc.DifferentiableDepthEncoder
    dc.DifferentiableDepthEncoder = _StubDA2
    try:
        with tempfile.TemporaryDirectory() as td:
            png = os.path.join(td, "img.png")
            Image.new("RGB", (64, 48), (10, 20, 30)).save(png)
            item = Mixin()
            item.path = png
            item.crop_height, item.crop_width = 48, 64
            # default: store_as_single_frame_video omitted (False)
            dc.cache_depth_gt_embeddings([item], _Cfg(), device=torch.device("cpu"),
                                         vae_roundtrip_fn=lambda x: x)
            saved = load_file(os.path.join(td, "_face_id_cache", "img.safetensors"))
            assert item.is_depth_cached is True and item.is_depth_video_cached is False
            assert item._depth_cache_key == "depth_gt_48x64", item._depth_cache_key
            assert "depth_gt_48x64" in saved and dc.CACHE_VERSION_KEY in saved
            assert tuple(saved["depth_gt_48x64"].shape) == (16, 20)  # 2D map, not a cube
            assert "depth_gt_video_48x64" not in saved and dc.CACHE_VERSION_VIDEO_KEY not in saved
            assert tuple(item.get_depth_gt().shape) == (16, 20)
    finally:
        dc.DifferentiableDepthEncoder = _orig
    print("[2] depth cache default mode: 2D image GT, no video key/flag (unchanged)     OK")


def test_3_identity_video_cache_skips_images_by_default() -> None:
    """Real cache_video_identity_embeddings, default (include_images=False):
    image items are NOT processed — only is_video items, as before."""
    import toolkit.face_id as fid

    cfg = types.SimpleNamespace(face_model="x", identity_loss_decoded_det_threshold=0.5)

    class _FI:
        def __init__(self):
            self.path = "/nonexistent.png"
            self.is_video = False
            self.identity_gt_video = None

        def get_latent(self):  # would be used only if processed
            raise AssertionError("image item must not be processed in default mode")

    fi = _FI()
    # No stubs needed: default mode filters images out before any model loads.
    fid.cache_video_identity_embeddings([fi], cfg, device=torch.device("cpu"),
                                        num_frames=1, arch="ltx2.3")
    assert fi.identity_gt_video is None, "image processed without include_images (regression!)"
    print("[3] identity video cache default mode: image items skipped (unchanged)        OK")


def test_4_body_proportion_video_cache_skips_images_by_default() -> None:
    """Real cache_video_body_proportion_embeddings, default (include_images=
    False): image items skipped, only is_video processed."""
    import toolkit.body_id as bid

    cfg = types.SimpleNamespace(body_proportion_include_head=False)
    fi = types.SimpleNamespace(path="/nonexistent.png", is_video=False,
                               body_proportion_gt_video=None)
    # Default mode returns before loading ViTPose when there are no video items.
    bid.cache_video_body_proportion_embeddings([fi], cfg, device=torch.device("cpu"),
                                               num_frames=1)
    assert fi.body_proportion_gt_video is None, "image processed without include_images (regression!)"
    print("[4] body-prop video cache default mode: image items skipped (unchanged)       OK")


def test_5_depth_roundtrip_4d_vae_unchanged() -> None:
    """The 5D bridge in _vae_roundtrip_for_depth is gated `if _vae_wants_5d and
    dim()==4` (encode) and `if pixels.dim()==5` (decode). For a 4D image-latent
    VAE both guards are False, so a 4D tensor passes through untouched. Replicate
    the guarded ops verbatim and assert the guard lines exist in the source."""
    src = _src(_SDTRAINER)
    assert "if _vae_wants_5d and arr_norm.dim() == 4:" in src, "encode-side guard changed"
    assert "arr_norm = arr_norm.unsqueeze(2)" in src, "encode-side unsqueeze changed"
    assert "if pixels.dim() == 5:" in src, "decode-side guard changed"

    saw = {}

    class _Dist:
        def __init__(self, z):
            self._z = z

        def mode(self):
            return self._z

    class _Enc:
        def __init__(self, z):
            self.latent_dist = _Dist(z)

    class _Dec:
        def __init__(self, s):
            self.sample = s

    class Fake4DVae:  # an image-latent (SD/SDXL/Flux-style) VAE
        def encode(self, x):
            saw["encode_ndim"] = x.dim()
            b, c, h, w = x.shape           # 4D — would raise if a T axis were added
            return _Enc(torch.randn(b, 4, h // 8, w // 8))

        def decode(self, z):
            b, c, h, w = z.shape
            return _Dec(torch.randn(b, 3, h * 8, w * 8))

    vae = Fake4DVae()
    _vae_wants_5d = False                  # non-ltx/wan model
    _vae_scale, _vae_shift = 1.0, 0.0

    # --- verbatim copy of the guarded shape ops from _vae_roundtrip_for_depth ---
    arr = torch.rand(1, 3, 256, 192)
    arr_norm = (arr * 2.0 - 1.0)
    if _vae_wants_5d and arr_norm.dim() == 4:
        arr_norm = arr_norm.unsqueeze(2)
    posterior = vae.encode(arr_norm)
    raw_latent = posterior.latent_dist.mode()
    scaled = _vae_scale * (raw_latent - _vae_shift)
    unscaled = scaled / _vae_scale
    if _vae_shift:
        unscaled = unscaled + _vae_shift
    pixels = vae.decode(unscaled).sample.float()
    pixels = (pixels + 1.0) * 0.5
    if pixels.dim() == 5:
        pixels = pixels[:, :, 0]
    pixels = pixels.clamp(0, 1)

    assert saw["encode_ndim"] == 4, "a temporal axis was added for a 4D VAE (regression!)"
    assert pixels.shape == (1, 3, 256, 192), pixels.shape  # 4D in -> 4D out, unchanged
    print("[5] depth roundtrip on a 4D VAE: no unsqueeze/squeeze, shape preserved        OK")


def main() -> None:
    test_1_arch_gate_predicate()
    test_2_depth_cache_default_mode_unchanged()
    test_3_identity_video_cache_skips_images_by_default()
    test_4_body_proportion_video_cache_skips_images_by_default()
    test_5_depth_roundtrip_4d_vae_unchanged()
    print("\n[done] LTX single-frame routing — no regression for other models")


if __name__ == "__main__":
    main()
