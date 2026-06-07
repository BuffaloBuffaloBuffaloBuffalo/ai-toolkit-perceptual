#!/usr/bin/env python3
"""Tests for the Flux2 ComfyUI checkpoint key remapper.

``normalize_transformer_state_dict`` maps a ComfyUI / original-format Flux2
checkpoint onto the native (bare) transformer keys, handling three layouts:
  (a) bare DiT keys                        -> returned unchanged
  (b) keys under ``model.diffusion_model.`` -> prefix stripped, count preserved
  (c) a bundled checkpoint with vae.* / text_encoders.* siblings
      -> only the DiT subtree kept, siblings dropped (loaded separately)

These are pure dict/string ops — no model or checkpoint download. Tensors are
1-element placeholders; we only assert on key sets / counts / identity, then
prove a real nn.Module load is strict (no silent wrong-load).

Run: CUDA_VISIBLE_DEVICES="" python -m pytest testing/test_comfy_compat.py
"""
import importlib.util
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import pytest
import torch
import torch.nn as nn

# Load comfy_compat by file path rather than as
# `extensions_built_in.diffusion_models.flux2.comfy_compat`: the package
# __init__ chain eagerly imports omnigen2's triton layer_norm, which calls into
# CUDA at import time and explodes on a CPU-only box. comfy_compat itself is
# pure dict/string + torch ops with no such deps, so a direct file load is both
# correct and keeps this test CPU-only.
_COMFY_COMPAT_PATH = os.path.join(
    _REPO, "extensions_built_in", "diffusion_models", "flux2", "comfy_compat.py"
)
_spec = importlib.util.spec_from_file_location("flux2_comfy_compat", _COMFY_COMPAT_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
normalize_transformer_state_dict = _mod.normalize_transformer_state_dict


def _t(v=0.0):
    return torch.tensor([v])


# Representative bare DiT key set (what the native Flux2 module expects).
_BARE_KEYS = [
    "double_blocks.0.img_attn.qkv.weight",
    "double_blocks.0.img_attn.qkv.bias",
    "single_blocks.0.linear1.weight",
    "img_in.weight",
    "final_layer.linear.weight",
]


# ---------------------------------------------------------------------------
# (a) bare DiT keys -> returned ~unchanged (same keys, same tensor objects)
# ---------------------------------------------------------------------------

def test_bare_keys_returned_unchanged():
    sd = {k: _t(i) for i, k in enumerate(_BARE_KEYS)}
    out = normalize_transformer_state_dict(sd)
    assert set(out.keys()) == set(sd.keys()), "bare keys should be unchanged"
    # Values passed through untouched (same objects).
    for k in sd:
        assert out[k] is sd[k], f"tensor for {k} should pass through by reference"


# ---------------------------------------------------------------------------
# (b) model.diffusion_model.* -> prefix stripped, count preserved
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prefix", ["model.diffusion_model.", "diffusion_model."])
def test_prefix_stripped_count_preserved(prefix):
    sd = {prefix + k: _t(i) for i, k in enumerate(_BARE_KEYS)}
    out = normalize_transformer_state_dict(sd)
    assert set(out.keys()) == set(_BARE_KEYS), (
        f"prefix '{prefix}' not stripped cleanly: {sorted(out.keys())}"
    )
    assert len(out) == len(sd), "key count must be preserved when only stripping a prefix"
    # No key retains the prefix, none is mangled/duplicated.
    assert not any(k.startswith(prefix) for k in out)
    # Values still correct per key (no shuffling).
    for i, k in enumerate(_BARE_KEYS):
        assert out[k].item() == float(i)


# ---------------------------------------------------------------------------
# (c) bundled checkpoint: keep DiT subtree, drop vae.* / text_encoders.*
# ---------------------------------------------------------------------------

def test_bundled_checkpoint_keeps_only_dit():
    prefix = "model.diffusion_model."
    sd = {}
    for i, k in enumerate(_BARE_KEYS):
        sd[prefix + k] = _t(i)
    # sibling subtrees that the toolkit loads separately
    for j in range(3):
        sd[f"vae.decoder.block.{j}.weight"] = _t(100 + j)
    for j in range(2):
        sd[f"text_encoders.t5.layer.{j}.weight"] = _t(200 + j)

    out = normalize_transformer_state_dict(sd)
    assert set(out.keys()) == set(_BARE_KEYS), (
        f"only the DiT subtree should remain, got: {sorted(out.keys())}"
    )
    # No sibling leaked through, and nothing was silently mangled into a DiT key.
    assert not any(k.startswith("vae.") or k.startswith("text_encoders.") for k in out)
    # The kept DiT tensors are the original objects (not corrupted/duplicated).
    for i, k in enumerate(_BARE_KEYS):
        assert out[k] is sd[prefix + k]


def test_log_callback_reports_strip_and_drop():
    """The optional log callback fires and mentions stripping + dropped siblings."""
    prefix = "model.diffusion_model."
    sd = {prefix + _BARE_KEYS[0]: _t(0), "vae.x.weight": _t(1)}
    msgs = []
    normalize_transformer_state_dict(sd, log=msgs.append)
    joined = " ".join(msgs).lower()
    assert "stripped" in joined, f"expected a 'stripped' log line, got {msgs}"
    assert "ignored" in joined and "1" in joined, f"expected dropped-key report, got {msgs}"


def test_bare_keys_log_says_no_rename():
    sd = {k: _t(i) for i, k in enumerate(_BARE_KEYS)}
    msgs = []
    normalize_transformer_state_dict(sd, log=msgs.append)
    assert any("no renaming" in m.lower() or "native key" in m.lower() for m in msgs), msgs


# ---------------------------------------------------------------------------
# Strict-load guard: a wrong key set must NOT silently load.
# ---------------------------------------------------------------------------

class _TinyDiT(nn.Module):
    """A module whose param names mirror the normalized (bare) key style."""

    def __init__(self):
        super().__init__()
        self.img_in = nn.Linear(4, 4, bias=False)
        self.final_layer = nn.Linear(4, 4, bias=False)


def _module_keys(m):
    return set(m.state_dict().keys())


def test_strict_load_rejects_missing_key():
    m = _TinyDiT()
    full = m.state_dict()
    # Build a normalized dict that is MISSING one expected key.
    keys = list(full.keys())
    incomplete = {k: full[k] for k in keys[1:]}  # drop the first
    normalized = normalize_transformer_state_dict(dict(incomplete))
    with pytest.raises(RuntimeError) as ei:
        m.load_state_dict(normalized, strict=True)
    assert "missing" in str(ei.value).lower()


def test_strict_load_rejects_extra_key():
    m = _TinyDiT()
    full = dict(m.state_dict())
    full["img_in.weight_EXTRA_BOGUS"] = torch.zeros(4, 4)
    normalized = normalize_transformer_state_dict(full)
    with pytest.raises(RuntimeError) as ei:
        m.load_state_dict(normalized, strict=True)
    assert "unexpected" in str(ei.value).lower()


def test_strict_load_succeeds_after_prefix_strip():
    """A correct prefixed checkpoint loads strictly once normalized."""
    m = _TinyDiT()
    prefix = "model.diffusion_model."
    prefixed = {prefix + k: v for k, v in m.state_dict().items()}
    normalized = normalize_transformer_state_dict(prefixed)
    # Should not raise — exact key match after stripping.
    missing, unexpected = m.load_state_dict(normalized, strict=True)
    assert not missing and not unexpected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
