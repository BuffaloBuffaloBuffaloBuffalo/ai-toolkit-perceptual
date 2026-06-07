#!/usr/bin/env python3
"""UMT5 (Wan 2.1) tokenizer parity check — CONDITIONING REGRESSION probe.

Background
----------
``toolkit/models/loaders/umt5.py`` was changed (commit 0ac2158, "port upstream
ab1ee4d") from::

    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder=...)

to::

    tokenizer = PatchedT5Tokenizer.from_pretrained(model_path, subfolder=...)

``PatchedT5Tokenizer`` subclasses the SLOW ``T5Tokenizer``. For the shipped
``ai-toolkit/umt5_xxl_encoder`` tokenizer, ``AutoTokenizer.from_pretrained``
resolves to the FAST ``T5TokenizerFast``. The fast and slow SentencePiece
implementations disagree on how leading / repeated / trailing spaces are
encoded (the slow tokenizer emits extra ``▁`` metaspace tokens). So the swap
changes the text-conditioning token ids for any prompt that contains such
whitespace — a real, silent training/inference regression for Wan 2.1.

This module PINS that finding with evidence rather than "fixing" it: the change
is an upstream port and reverting it is a human decision. The test asserts:

  1. the swap DOES change ids on whitespace-y prompts (regression is real), and
  2. the divergence is purely fast-vs-slow — ``PatchedT5Tokenizer`` is
     byte-identical to a plain slow ``T5Tokenizer`` (so the patch's
     ``_spm_precompiled_charsmap=None`` override is NOT the culprit).

If a future change makes Auto/Patched agree again, assertion (1) will fail and
flag that the regression status changed — review then.

Assets: requires ``ai-toolkit/umt5_xxl_encoder`` already in the local HF cache.
No download is attempted (``snapshot_download(local_files_only=True)``); the
test SKIPs if the tokenizer files are absent.

Run: CUDA_VISIBLE_DEVICES="" python -m pytest testing/test_umt5_tokenizer_parity.py -v
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

_REPO_ID = "ai-toolkit/umt5_xxl_encoder"
_SUBFOLDER = "tokenizer"

# Prompts chosen to exercise the exact divergence: plain text, leading spaces,
# trailing spaces, repeated internal spaces, unicode, and the empty string.
_PROMPTS = [
    "a photo of a cat",
    "  leading spaces",
    "trailing spaces  ",
    " multiple   internal   spaces ",
    "café 日本語 mixed unicode 🙂",
    "",
]


def _local_tokenizer_dir():
    """Resolve the cached tokenizer dir without any network call, else None."""
    try:
        from huggingface_hub import snapshot_download
        snap = snapshot_download(
            _REPO_ID, local_files_only=True, allow_patterns=[f"{_SUBFOLDER}/*"]
        )
    except Exception:
        return None
    tok_dir = os.path.join(snap, _SUBFOLDER)
    needed = {"spiece.model", "tokenizer_config.json"}
    if not os.path.isdir(tok_dir) or not needed.issubset(set(os.listdir(tok_dir))):
        return None
    return tok_dir


_TOK_DIR = _local_tokenizer_dir()
_skip = pytest.mark.skipif(
    _TOK_DIR is None,
    reason=f"{_REPO_ID} tokenizer not in local HF cache; skipping (no download).",
)


def _ids(tok, text):
    return tok(text, add_special_tokens=True)["input_ids"]


@_skip
def test_auto_vs_patched_diverge_on_whitespace():
    """REGRESSION (confirmed): AutoTokenizer (fast) != PatchedT5Tokenizer (slow).

    The original loader used AutoTokenizer; the new loader uses
    PatchedT5Tokenizer. Their ids must differ on at least one whitespace-y
    prompt — that divergence IS the conditioning regression. If this ever
    stops being true, the regression status changed and needs review.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from transformers import AutoTokenizer
        from toolkit.models.loaders.umt5 import PatchedT5Tokenizer

        auto = AutoTokenizer.from_pretrained(_TOK_DIR)         # original behaviour
        patched = PatchedT5Tokenizer.from_pretrained(_TOK_DIR)  # new behaviour

    # Sanity: Auto resolves to the fast impl, Patched is slow.
    assert auto.is_fast is True, f"expected fast AutoTokenizer, got {type(auto).__name__}"
    assert patched.is_fast is False, f"expected slow Patched, got {type(patched).__name__}"

    diffs = {p: (_ids(auto, p), _ids(patched, p)) for p in _PROMPTS}
    differing = {p: v for p, v in diffs.items() if v[0] != v[1]}
    assert differing, (
        "Expected AutoTokenizer and PatchedT5Tokenizer ids to DIFFER on "
        "whitespace prompts (the known regression), but they all matched. "
        "Regression status may have changed — review the umt5 loader swap."
    )
    # Spot-pin the canonical case so the evidence is explicit in failures.
    a_lead, p_lead = diffs["  leading spaces"]
    assert a_lead != p_lead, (a_lead, p_lead)
    assert len(p_lead) > len(a_lead), (
        "slow tokenizer should emit MORE tokens (extra metaspace) on leading "
        f"spaces; fast={a_lead} slow={p_lead}"
    )


@_skip
def test_patch_itself_matches_plain_slow_t5():
    """The patch is NOT the culprit: Patched == plain slow T5Tokenizer ids.

    Proves the divergence is purely fast-vs-slow tokenizer choice, not the
    PatchedT5Tokenizer's _spm_precompiled_charsmap=None override.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from transformers import T5Tokenizer
        from toolkit.models.loaders.umt5 import PatchedT5Tokenizer

        slow = T5Tokenizer.from_pretrained(_TOK_DIR)
        patched = PatchedT5Tokenizer.from_pretrained(_TOK_DIR)

    for p in _PROMPTS:
        assert _ids(slow, p) == _ids(patched, p), (
            f"PatchedT5Tokenizer diverged from plain slow T5Tokenizer on {p!r}: "
            f"slow={_ids(slow, p)} patched={_ids(patched, p)} — the patch override "
            "itself changes tokenization (unexpected)."
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
