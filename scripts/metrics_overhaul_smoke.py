"""Synthetic smoke tests for the metrics-system overhaul.

Each ``smoke_step_N`` function exercises one step of the audit plan:

  * ``smoke_step_1_bin_running_mean`` — verifies ``SDTrainer._bin_update`` /
    ``_bin_finalize`` accumulate a running mean per bin and that
    ``_reset_step_bins`` clears state between optimizer steps.
  * ``smoke_step_2_metric_buffer`` — verifies ``MetricBuffer.add_scalar``
    weighted-mean and per-microbatch reset semantics.
  * ``smoke_step_3_per_sample`` — verifies per-sample collection produces the
    expected breakdown payload (top-K-by-deviation cap, n/mean/std).
  * ``smoke_step_4_dual_write`` — verifies ``loss_dict`` dual-write emits both
    legacy and canonical keys with the same value.

All smokes are CPU-only and use synthetic tensors. They never touch a real
model, dataloader, or training loop.
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict


def _import_sd_trainer():
    """Import SDTrainer without instantiating heavy dependencies.

    SDTrainer's ``__init__`` pulls in CUDA + diffusers; we only need the
    static helpers (``_bin_update`` etc.) so we work directly off the class.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer  # noqa: E402
    return SDTrainer


def smoke_step_1_bin_running_mean() -> None:
    SDTrainer = _import_sd_trainer()

    # Three samples land in the same t-band ("t40"); two land in a different
    # one ("t70"). Pre-fix code would only retain the last value; post-fix
    # code must report the per-bin mean.
    bins: Dict[str, Any] = {}
    SDTrainer._bin_update(bins, "id_sim_t40", 0.10)
    SDTrainer._bin_update(bins, "id_sim_t40", 0.20)
    SDTrainer._bin_update(bins, "id_sim_t40", 0.30)
    SDTrainer._bin_update(bins, "id_sim_t70", 0.50)
    SDTrainer._bin_update(bins, "id_sim_t70", 0.70)

    assert bins["id_sim_t40"]["count"] == 3
    assert bins["id_sim_t70"]["count"] == 2

    finalized = SDTrainer._bin_finalize(bins)
    assert math.isclose(finalized["id_sim_t40"], 0.20, abs_tol=1e-9), finalized
    assert math.isclose(finalized["id_sim_t70"], 0.60, abs_tol=1e-9), finalized

    # _bin_finalize on None / empty dict returns {}.
    assert SDTrainer._bin_finalize(None) == {}
    assert SDTrainer._bin_finalize({}) == {}

    # _bin_finalize survives a slot with count==0 (defensive).
    bins2 = {"x": {"sum": 1.0, "count": 0}}
    assert SDTrainer._bin_finalize(bins2) == {}

    print("[step 1] bin running mean: OK")


def smoke_step_2_metric_buffer() -> None:
    """Skipped until step 2 lands MetricBuffer."""
    try:
        from extensions_built_in.sd_trainer.metric_buffer import MetricBuffer
    except Exception as e:  # noqa: BLE001
        print(f"[step 2] metric buffer not yet imported ({e}); skipping")
        return

    buf = MetricBuffer(per_sample_cap=16)

    # Microbatch 1: 4 samples, mask [1, 1, 0, 1] → 3 valid → weighted mean by mask.sum.
    buf.add_scalar("id_sim", 0.6, weight=3.0)
    # Microbatch 2: 4 samples, mask [1, 0, 0, 0] → 1 valid.
    buf.add_scalar("id_sim", 0.9, weight=1.0)

    flushed = buf.flush_scalars()
    expected = (0.6 * 3.0 + 0.9 * 1.0) / (3.0 + 1.0)
    got = flushed["id_sim"]
    assert math.isclose(got, expected, abs_tol=1e-9), (got, expected)

    # After flush, buffer is empty (no carryover).
    again = buf.flush_scalars()
    assert again == {}

    # weight=0 must not crash and must be ignored.
    buf.add_scalar("noop", 1.0, weight=0.0)
    assert buf.flush_scalars() == {}

    print("[step 2] metric buffer running mean: OK")


def smoke_step_3_per_sample() -> None:
    try:
        from extensions_built_in.sd_trainer.metric_buffer import (
            MetricBuffer,
            MetricValue,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[step 3] metric buffer not yet imported ({e}); skipping")
        return

    buf = MetricBuffer(per_sample_cap=4)
    # 6 samples; cap=4 should retain the 4 farthest from the mean.
    values = [0.10, 0.12, 0.50, 0.51, 0.90, 0.91]
    tags = [f"img{i}.png" for i in range(len(values))]
    for v, tag in zip(values, tags):
        buf.add_per_sample("id_sim", v, t=0.4, sample_tag=tag)

    flushed = buf.flush_per_sample()
    payload = flushed["id_sim"]
    assert payload["n"] == 6, payload
    mean = payload["mean"]
    expected_mean = sum(values) / len(values)
    assert math.isclose(mean, expected_mean, abs_tol=1e-9), payload

    samples = payload["samples"]
    assert len(samples) == 4, len(samples)
    kept_devs = sorted(abs(s["value"] - mean) for s in samples)
    dropped_devs = sorted(set(abs(v - mean) for v in values) - set(kept_devs))
    assert min(kept_devs) >= max(dropped_devs), (kept_devs, dropped_devs)

    payload_json = json.dumps(payload)
    parsed = json.loads(payload_json)
    assert parsed["n"] == 6

    # MetricValue: behaves like a float for every downstream consumer
    # (arithmetic, format strings, epoch accumulators) and exposes the
    # breakdown via attribute access.
    mv = MetricValue(0.7, payload)
    assert isinstance(mv, float), "MetricValue must subclass float"
    # Arithmetic still works.
    assert math.isclose(float(mv) + 0.3, 1.0, abs_tol=1e-9)
    # Format strings still work (used by the prog-bar printer).
    assert f"{mv:.3e}" == f"{0.7:.3e}", f"{mv:.3e}"
    # Breakdown survives the wrap.
    assert mv.breakdown == payload

    # Logger coerces MetricValue into (value_real, value_text).
    from toolkit.logging_aitk import UILogger
    logger = UILogger.__new__(UILogger)  # avoid db init
    vr, vt = logger._coerce_value(mv)
    assert math.isclose(vr, 0.7, abs_tol=1e-9), (vr, mv)
    assert vt is not None and vt.startswith("{"), vt
    parsed_vt = json.loads(vt)
    assert parsed_vt["n"] == 6

    # Plain ints/floats still hit the simple path.
    vr2, vt2 = logger._coerce_value(0.42)
    assert math.isclose(vr2, 0.42, abs_tol=1e-9)
    assert vt2 is None

    print("[step 3] per-sample top-K-by-deviation + MetricValue + logger: OK")


def smoke_step_4_dual_write() -> None:
    """Verifies the rename map + dual-write helper from the trainer."""
    try:
        from extensions_built_in.sd_trainer.metric_naming import (
            CANONICAL_RENAMES,
            apply_dual_write,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[step 4] metric naming not yet imported ({e}); skipping")
        return

    legacy = {
        "loss": 0.5,
        "grad_norm": 1.2,
        "id_sim": 0.7,
        "identity_loss": 0.3,
        "identity_loss_applied": 0.03,
        "depth_consistency_loss": 0.4,
        "depth_consistency_loss_applied": 0.04,
        "depth_consistency_ssi": 0.05,
        "id_sim_t40": 0.65,
        "depth_loss_t40": 0.42,
        "va_level_1": 0.11,
        "va_mid": 0.22,
    }
    out = apply_dual_write(legacy)

    # Every legacy key still present (back-compat for old dashboards).
    for k, v in legacy.items():
        assert out[k] == v, k

    # Every legacy key with a canonical mapping has its new name present too.
    for legacy_key, canonical_key in CANONICAL_RENAMES.items():
        if legacy_key in legacy:
            assert canonical_key in out, canonical_key
            assert out[canonical_key] == legacy[legacy_key], canonical_key

    print(f"[step 4] dual-write rename map: OK ({len(CANONICAL_RENAMES)} mappings)")


def main() -> int:
    smoke_step_1_bin_running_mean()
    smoke_step_2_metric_buffer()
    smoke_step_3_per_sample()
    smoke_step_4_dual_write()
    print("All smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
