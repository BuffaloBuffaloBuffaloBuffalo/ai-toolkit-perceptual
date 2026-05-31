"""Regression tests for the lazy depth-cache *retrieval* path.

Yesterday's fix removed a per-item ``gc.collect()`` from the depth cache-hit
read, but the up-front ``cache_depth_gt_embeddings`` pass still eagerly opened
every cache file and materialized every GT depth tensor into RAM — strictly
more work than the latent / text-embedding caches, whose hit path is a bare
``os.path.exists()`` with the tensor read deferred to the DataLoader worker.
That left the depth pass crawling on large maps while the others ran at
4k+ it/s.

This change makes depth match that lazy contract:

  * ``cache_depth_gt_embeddings`` (image) and ``cache_video_depth_gt_embeddings``
    validate a hit with a header-only ``_cache_header_shape`` (no tensor bytes,
    nothing held resident) and record where to read from on the file item.
  * ``DepthCachingFileItemDTOMixin.get_depth_gt`` / ``get_depth_gt_video`` read
    the single bucket tensor on demand in the worker; ``cleanup_depth``
    releases it between batches.

These checks are pure-stdlib + safetensors, CPU-only, and never load DA2.

Run: ``python scripts/depth_cache_lazy_load_smoke.py`` — exits 0 on success.
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch


def _imports():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)
    from safetensors.torch import load_file
    from toolkit.depth_consistency import (  # noqa: E402
        _atomic_save_file,
        _cache_header_shape,
        CACHE_VERSION_KEY,
        CACHE_VERSION_VIDEO_KEY,
    )
    from toolkit.dataloader_mixins import DepthCachingFileItemDTOMixin  # noqa: E402
    return (
        _atomic_save_file,
        _cache_header_shape,
        CACHE_VERSION_KEY,
        CACHE_VERSION_VIDEO_KEY,
        DepthCachingFileItemDTOMixin,
        load_file,
    )


def _multi_key_cache(path, save_fn, version_key):
    """Write a cache file shaped like the real _face_id_cache entry: face/body
    embeddings + several per-bucket depth maps + the version sentinel. Returns
    the dict of ground-truth tensors written."""
    data = {
        # non-depth neighbours that share the file — the lazy read must skip these
        "face_embedding": torch.randn(512),
        "body_proportion": torch.randn(34),
        # two distinct buckets with deliberately different shapes AND values
        "depth_gt_384x512": (torch.randn(384, 512) * 7.0).to(torch.float16),
        "depth_gt_768x576": (torch.randn(576, 768) * 3.0).to(torch.float16),
        version_key: torch.ones(1),
    }
    save_fn(data, path)
    return data


def smoke_1_header_shape_presence() -> None:
    """``_cache_header_shape`` returns the true shape when key+version present,
    None when either is absent — the up-front pass's hit/miss decision."""
    (save_fn, header_shape, VK, _, _, _) = _imports()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "c.safetensors")
        gt = _multi_key_cache(path, save_fn, VK)

        shp = header_shape(path, "depth_gt_384x512", VK)
        assert shp is not None and list(shp) == list(gt["depth_gt_384x512"].shape), shp
        # the other bucket reports its own (different) shape
        shp2 = header_shape(path, "depth_gt_768x576", VK)
        assert list(shp2) == [576, 768], shp2
        # missing depth key for this bucket → miss (must recompute, not reuse)
        assert header_shape(path, "depth_gt_1024x1024", VK) is None
        # version sentinel absent (stale/foreign cache) → miss
        assert header_shape(path, "depth_gt_384x512", "depth_gt_vNOPE") is None
    print("[1] _cache_header_shape presence / per-bucket / version checks OK")


def smoke_2_header_shape_corrupt() -> None:
    """Corrupt / non-safetensors file → None (caller recomputes), no raise."""
    (_, header_shape, VK, _, _, _) = _imports()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "corrupt.safetensors")
        with open(path, "wb") as f:
            f.write(b"definitely not safetensors")
        assert header_shape(path, "depth_gt_384x512", VK) is None
    print("[2] _cache_header_shape tolerates corrupt cache OK")


def smoke_3_lazy_read_equivalence() -> None:
    """The deferred read returns a tensor bit-identical to a full ``load_file``
    of the same key — correctness of the eager path is preserved, and the read
    pulls THIS bucket (not a neighbour) out of a multi-key file."""
    (save_fn, _, VK, _, Mixin, load_file) = _imports()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "c.safetensors")
        gt = _multi_key_cache(path, save_fn, VK)
        full = load_file(path)
        for key in ("depth_gt_384x512", "depth_gt_768x576"):
            got = Mixin._read_depth_key(path, key)
            assert got is not None, key
            assert torch.equal(got, full[key]), f"{key} differs from full load_file"
            assert torch.equal(got, gt[key]), f"{key} differs from written GT"
        # missing key → None
        assert Mixin._read_depth_key(path, "depth_gt_404") is None
        assert Mixin._read_depth_key(None, "depth_gt_384x512") is None
    print("[3] lazy read is bit-identical to full load_file (per bucket) OK")


def smoke_4_image_lazy_contract() -> None:
    """End-to-end image contract: a not-yet-loaded item reads on first access,
    holds the tensor, releases it on cleanup, and re-reads correctly — the same
    disk-cache contract get_latent uses."""
    (save_fn, header_shape, VK, _, Mixin, load_file) = _imports()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "c.safetensors")
        gt = _multi_key_cache(path, save_fn, VK)

        item = Mixin()
        # not cached → no read, no crash, returns None
        assert item.get_depth_gt() is None
        assert item.depth_gt is None

        # simulate cache_depth_gt_embeddings hit path: header check + metadata,
        # tensor NOT read yet
        assert header_shape(path, "depth_gt_384x512", VK) is not None
        item.is_depth_cached = True
        item._depth_cache_path = path
        item._depth_cache_key = "depth_gt_384x512"
        assert item.depth_gt is None, "tensor must not be resident before first access"

        # first access reads from disk
        d = item.get_depth_gt()
        assert d is not None and torch.equal(d, gt["depth_gt_384x512"])
        assert item.depth_gt is d, "result must be cached on the item after read"

        # cleanup releases it (so it can't re-accumulate across batches)
        item.cleanup_depth()
        assert item.depth_gt is None

        # re-read after cleanup still correct
        d2 = item.get_depth_gt()
        assert d2 is not None and torch.equal(d2, gt["depth_gt_384x512"])
    print("[4] image lazy-load / cleanup / re-read contract OK")


def smoke_5_video_lazy_contract() -> None:
    """Video counterpart: frame-count validated from the header shape, tensor
    deferred, cleanup + re-read correct."""
    (save_fn, header_shape, _, VVK, Mixin, _) = _imports()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "v.safetensors")
        cube = (torch.randn(16, 96, 128) * 2.0).to(torch.float16)  # (T, H, W)
        save_fn({"depth_gt_video": cube, VVK: torch.ones(1)}, path)

        # header reports T from shape[0] without reading bytes
        shp = header_shape(path, "depth_gt_video", VVK)
        assert shp is not None and shp[0] == 16, shp

        item = Mixin()
        item.is_depth_video_cached = True
        item._depth_video_cache_path = path
        item._depth_video_cache_key = "depth_gt_video"
        assert item.depth_gt_video is None

        v = item.get_depth_gt_video()
        assert v is not None and torch.equal(v, cube)
        item.cleanup_depth()
        assert item.depth_gt_video is None
        assert torch.equal(item.get_depth_gt_video(), cube)
    print("[5] video lazy-load / cleanup / re-read contract OK")


def smoke_6_deepcopy_is_cheap() -> None:
    """The shared file_list item must not carry the tensor (it is deep-copied
    every __getitem__). After the hit path runs, depth_gt is None on the
    original, so copy.deepcopy clones only metadata, not a depth map."""
    import copy
    (save_fn, header_shape, VK, _, Mixin, _) = _imports()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "c.safetensors")
        _multi_key_cache(path, save_fn, VK)
        original = Mixin()
        original.is_depth_cached = True
        original._depth_cache_path = path
        original._depth_cache_key = "depth_gt_384x512"
        original.depth_gt = None  # state after the hit path / cleanup

        clone = copy.deepcopy(original)
        assert clone.depth_gt is None, "deepcopy carried a resident tensor"
        assert clone._depth_cache_key == "depth_gt_384x512"
        # the clone (the per-item worker copy) loads independently
        assert clone.get_depth_gt() is not None
        assert original.depth_gt is None, "loading the clone must not touch the shared original"
    print("[6] shared original stays tensor-free; deepcopy clones metadata only OK")


def smoke_7_single_frame_video_cache_mode() -> None:
    """``cache_depth_gt_embeddings(store_as_single_frame_video=True)`` — the
    LTX/Wan still-image path. The same v3 GT must land under the *video* cache
    namespace (key + version + is_depth_video_cached) as a ``(1, H, W)`` cube,
    so the 5D video depth block reads it; the default image mode is unchanged.
    DA2 is stubbed so this stays CPU-only and never downloads a model."""
    import toolkit.depth_consistency as dc
    from PIL import Image as _PILImage

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)
    from safetensors.torch import load_file
    from toolkit.dataloader_mixins import DepthCachingFileItemDTOMixin as Mixin

    class _StubDA2:  # returns a fixed (b, 16, 20) depth; ignores pixels
        def __init__(self, *a, **k):
            pass

        def __call__(self, arr):
            return torch.ones(arr.shape[0], 16, 20)

    class _Cfg:
        model_id = "stub"
        input_size = 384
        pixel_blur_sigma = 0.0

    def _run(td, store_as_video):
        # a real (W=64, H=48) image on disk; the stub ignores its content
        png = os.path.join(td, "img.png")
        _PILImage.new("RGB", (64, 48), (123, 222, 64)).save(png)
        item = Mixin()
        item.path = png
        item.crop_height, item.crop_width = 48, 64  # → bucket-keyed cache
        # the roundtrip is the trainer's; here assert its 4D contract and pass through
        seen = {}

        def _rt(x):
            seen["dim"] = x.dim()
            return x

        dc.cache_depth_gt_embeddings(
            [item], _Cfg(), device=torch.device("cpu"),
            vae_roundtrip_fn=_rt, store_as_single_frame_video=store_as_video,
        )
        assert seen["dim"] == 4, "roundtrip fn must receive 4D (1,3,H,W) pixels"
        cache_path = os.path.join(td, "_face_id_cache", "img.safetensors")
        return item, load_file(cache_path)

    _orig = dc.DifferentiableDepthEncoder
    dc.DifferentiableDepthEncoder = _StubDA2
    try:
        # --- video mode (LTX/Wan still image) ---
        with tempfile.TemporaryDirectory() as td:
            item, saved = _run(td, store_as_video=True)
            assert item.is_depth_video_cached is True
            assert item.is_depth_cached is False, "must not set the image flag"
            assert item._depth_video_cache_key == "depth_gt_video_48x64", item._depth_video_cache_key
            assert "depth_gt_video_48x64" in saved and dc.CACHE_VERSION_VIDEO_KEY in saved
            assert "depth_gt_48x64" not in saved and dc.CACHE_VERSION_KEY not in saved
            cube = saved["depth_gt_video_48x64"]
            assert tuple(cube.shape) == (1, 16, 20), cube.shape  # 1-frame cube
            assert item.depth_gt_video is None, "must not retain the tensor"
            # lazy read-back via the worker path returns the same cube
            assert tuple(item.get_depth_gt_video().shape) == (1, 16, 20)

        # --- default image mode (regression: unchanged) ---
        with tempfile.TemporaryDirectory() as td:
            item, saved = _run(td, store_as_video=False)
            assert item.is_depth_cached is True and item.is_depth_video_cached is False
            assert item._depth_cache_key == "depth_gt_48x64", item._depth_cache_key
            assert "depth_gt_48x64" in saved and dc.CACHE_VERSION_KEY in saved
            assert "depth_gt_video_48x64" not in saved
            assert tuple(saved["depth_gt_48x64"].shape) == (16, 20)  # 2D map
            assert tuple(item.get_depth_gt().shape) == (16, 20)
    finally:
        dc.DifferentiableDepthEncoder = _orig
    print("[7] single-frame-video cache mode (LTX/Wan) + image-mode regression OK")


def main() -> None:
    smoke_1_header_shape_presence()
    smoke_2_header_shape_corrupt()
    smoke_3_lazy_read_equivalence()
    smoke_4_image_lazy_contract()
    smoke_5_video_lazy_contract()
    smoke_6_deepcopy_is_cheap()
    smoke_7_single_frame_video_cache_mode()
    print("\n[done] depth-cache lazy-load smoke tests passed")


if __name__ == "__main__":
    main()
