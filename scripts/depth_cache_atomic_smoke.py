"""Regression tests for atomic/Windows-safe depth-cache I/O helpers.

Covers the two private helpers added in ``toolkit/depth_consistency.py``:

  * ``_atomic_save_file`` — temp-file + ``os.replace`` write so that an
    in-place rewrite of a safetensors cache cannot trip Windows
    ``ERROR_USER_MAPPED_FILE (1224)`` when the destination is mmap'd.
  * ``_load_then_close`` — load a safetensors cache into owned (cloned)
    storage so the returned tensors are detached from the mmap and a
    subsequent overwrite of the same path is safe. (The clones drop the mmap
    by refcounting; no per-item ``gc.collect()`` is needed or used — that was
    removed as it dominated the caching loop under a large resident heap.)

The depth-cache crash on Windows hit the exact load → augment → save path
exercised here (multi-bucket training writes the same per-image cache file
under different bucket keys). These checks are pure-stdlib + safetensors,
CPU-only, and never load DA2, so they belong in fast smoke coverage rather
than the GPU-bound ``depth_consistency_smoke.py``.

Run: ``python scripts/depth_cache_atomic_smoke.py`` — exits 0 on success.
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch


def _import_helpers():
    """Import the patched helpers without pulling in CUDA/diffusers."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)
    from toolkit.depth_consistency import _atomic_save_file, _load_then_close  # noqa: E402
    return _atomic_save_file, _load_then_close


def smoke_step_1_atomic_save_basic() -> None:
    """Fresh write → read-back is bit-identical."""
    _atomic_save_file, _load_then_close = _import_helpers()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "fresh.safetensors")
        data = {"a": torch.randn(3, 4), "b": torch.ones(2, dtype=torch.float32)}
        _atomic_save_file(data, path)
        assert os.path.exists(path), "cache file not created"
        loaded = _load_then_close(path)
        assert set(loaded.keys()) == {"a", "b"}, loaded.keys()
        assert torch.equal(loaded["a"], data["a"]), "tensor 'a' mismatch"
        assert torch.equal(loaded["b"], data["b"]), "tensor 'b' mismatch"
    print("[1] atomic save round-trip OK")


def smoke_step_2_load_then_close_detaches_mmap() -> None:
    """Tensors returned by ``_load_then_close`` survive deletion of the
    source dict. The helper exists to detach mmap-backed tensors so that a
    later overwrite of the file does not crash on Windows; the easiest
    observable consequence is that the returned tensors are independent
    clones, not views into mmap'd storage."""
    _atomic_save_file, _load_then_close = _import_helpers()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "detach.safetensors")
        original = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        _atomic_save_file({"t": original}, path)
        loaded = _load_then_close(path)
        t = loaded["t"]
        del loaded
        # Mutating the returned tensor must not raise and must not touch
        # the on-disk bytes (the clone is independent).
        t.add_(1.0)
        reloaded = _load_then_close(path)
        assert torch.equal(reloaded["t"], original), "on-disk value was mutated through mmap"
    print("[2] _load_then_close returns detached clones OK")


def smoke_step_3_load_modify_save_cycle() -> None:
    """Load → augment → save the same path in the same process. This is
    the exact pattern that triggers Windows OSError 1224 against the old
    ``load_file`` + in-place ``save_file`` sequence; it must succeed and
    preserve prior keys."""
    _atomic_save_file, _load_then_close = _import_helpers()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "cycle.safetensors")
        _atomic_save_file({"keep_me": torch.full((4,), 7.0)}, path)
        save_data = _load_then_close(path)
        save_data["added"] = torch.zeros(3)
        _atomic_save_file(save_data, path)
        final = _load_then_close(path)
        assert set(final.keys()) == {"keep_me", "added"}, final.keys()
        assert torch.equal(final["keep_me"], torch.full((4,), 7.0)), "prior key clobbered"
    print("[3] load → augment → save cycle OK")


def smoke_step_4_per_bucket_merge() -> None:
    """Multi-bucket cache merge — mirrors the per-(crop_h, crop_w) keying
    that ``cache_depth_gt_embeddings`` performs. The Windows crash
    originally fired on the *first image of the second dataset*, i.e. the
    first time this load → merge → resave path ran against an existing
    cache file."""
    _atomic_save_file, _load_then_close = _import_helpers()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "buckets.safetensors")

        # bucket A — fresh write
        _atomic_save_file(
            {"depth_gt_512x384": torch.randn(20, 20).to(torch.float16),
             "depth_gt_v3": torch.ones(1)},
            path,
        )

        # bucket B — load existing, add new bucket, save
        merged = _load_then_close(path)
        merged["depth_gt_768x576"] = torch.randn(30, 30).to(torch.float16)
        _atomic_save_file(merged, path)

        # bucket C — same again
        merged2 = _load_then_close(path)
        merged2["depth_gt_1024x768"] = torch.randn(40, 40).to(torch.float16)
        _atomic_save_file(merged2, path)

        final = _load_then_close(path)
        expected = {"depth_gt_512x384", "depth_gt_768x576", "depth_gt_1024x768", "depth_gt_v3"}
        assert set(final.keys()) == expected, final.keys()
    print("[4] per-bucket merge across three rewrites OK")


def smoke_step_5_no_temp_leak_on_success() -> None:
    """A clean run must leave only the destination file behind. The temp
    file lives under the same directory (so ``os.replace`` is a same-FS
    rename) and must be renamed away, not left as garbage."""
    _atomic_save_file, _ = _import_helpers()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "clean.safetensors")
        for _ in range(5):
            _atomic_save_file({"x": torch.randn(10)}, path)
        listing = os.listdir(td)
        assert listing == ["clean.safetensors"], f"temp files leaked: {listing}"
    print("[5] no temp-file leakage across 5 rewrites OK")


def smoke_step_6_temp_cleaned_on_save_failure() -> None:
    """If the safetensors write itself raises (e.g. an unserializable
    object slipped into save_data), the temp file must be removed and the
    original cache must remain untouched."""
    _atomic_save_file, _load_then_close = _import_helpers()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "guarded.safetensors")
        good = {"a": torch.ones(2)}
        _atomic_save_file(good, path)

        # safetensors.save_file rejects non-tensor values → save_file raises
        # → _atomic_save_file's except block must remove the temp file.
        try:
            _atomic_save_file({"bad": "not a tensor"}, path)  # type: ignore[dict-item]
        except Exception:
            pass
        else:
            raise AssertionError("expected save_file to raise on non-tensor value")

        listing = sorted(os.listdir(td))
        assert listing == ["guarded.safetensors"], f"temp file leaked after failure: {listing}"

        # original cache survived the failed write
        loaded = _load_then_close(path)
        assert torch.equal(loaded["a"], torch.ones(2)), "original cache was corrupted by failed write"
    print("[6] temp file cleaned + original preserved on save failure OK")


def smoke_step_7_nested_cache_dir_created() -> None:
    """``_atomic_save_file`` must create missing parent directories — the
    image- and video-cache call sites no longer call ``os.makedirs``
    themselves before the save."""
    _atomic_save_file, _load_then_close = _import_helpers()
    with tempfile.TemporaryDirectory() as td:
        nested = os.path.join(td, "a", "b", "c", "deep.safetensors")
        _atomic_save_file({"k": torch.ones(1)}, nested)
        assert os.path.exists(nested), "nested cache_dir not created"
        loaded = _load_then_close(nested)
        assert torch.equal(loaded["k"], torch.ones(1))
    print("[7] nested cache_dir auto-created OK")


def smoke_step_8_corrupt_cache_raises() -> None:
    """``_load_then_close`` propagates errors from corrupt files. Call
    sites wrap this in try/except to fall back to a fresh-write path; the
    helper itself must not swallow the error silently."""
    _, _load_then_close = _import_helpers()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "corrupt.safetensors")
        with open(path, "wb") as f:
            f.write(b"not a real safetensors file")
        raised = False
        try:
            _load_then_close(path)
        except Exception:
            raised = True
        assert raised, "_load_then_close should have raised on corrupt input"
    print("[8] corrupt cache propagates exception OK")


def main() -> None:
    smoke_step_1_atomic_save_basic()
    smoke_step_2_load_then_close_detaches_mmap()
    smoke_step_3_load_modify_save_cycle()
    smoke_step_4_per_bucket_merge()
    smoke_step_5_no_temp_leak_on_success()
    smoke_step_6_temp_cleaned_on_save_failure()
    smoke_step_7_nested_cache_dir_created()
    smoke_step_8_corrupt_cache_raises()
    print("\n[done] depth-cache atomic-write smoke tests passed")


if __name__ == "__main__":
    main()
