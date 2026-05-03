"""Smoke for the "Metrics (new)" tab's by_sample-facet pivot logic.

The UI side (`ui/src/components/JobMetricsGraph.tsx`) implements
``pivotBySample(points)`` — pivots a list of LossPoints with breakdowns
from ``(step -> samples[])`` to ``(sample -> [(step, value, t)])``. This
script reimplements the same algorithm in Python (faithful port — same
ranking, tiebreaker, sparse-point semantics) and asserts that:

  1. Each distinct sample becomes its own series.
  2. Each series only carries points for the steps where its sample
     appeared in the breakdown payload (i.e. it is sparse — *not*
     interpolated across steps where the sample was absent).
  3. Series ranking is by descending count, then alphabetical.
  4. Sample names with no value or empty/None tag are dropped.
  5. The stable color-hash returns the same index for the same name
     across runs (dataset-stability check for the legend).

The smoke is CPU-only and uses synthetic data only. It also verifies
against the real DB at
``output/kk_snofs_face_depth_diff_lr05/loss_log.db`` if present.
"""

from __future__ import annotations

import json
import os
import random
import sys
import sqlite3
from typing import Any, Dict, List, Optional


# ----- Faithful Python port of the TS pivotBySample -------------------

def _pivot_by_sample(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mirror of ``pivotBySample`` in JobMetricsGraph.tsx.

    ``points`` is a list of ``{step, value, breakdown?}`` shaped exactly
    like ``LossPoint`` on the TS side. ``breakdown`` is
    ``{samples: [{value, t?, sample?}], n, mean, std}`` or absent.
    """
    by_sample: Dict[str, Dict[str, Any]] = {}
    for p in points:
        bd = p.get("breakdown")
        if not bd or not isinstance(bd.get("samples"), list):
            continue
        for s in bd["samples"]:
            tag = (s.get("sample") or "").strip()
            if not tag:
                continue
            v = s.get("value")
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv != fv or fv in (float("inf"), float("-inf")):
                continue
            entry = by_sample.get(tag)
            if entry is None:
                entry = {"sample": tag, "count": 0, "points": []}
                by_sample[tag] = entry
            entry["points"].append({"step": p["step"], "value": fv, "t": s.get("t")})
            entry["count"] += 1
    for entry in by_sample.values():
        entry["points"].sort(key=lambda x: x["step"])
    return sorted(
        by_sample.values(),
        key=lambda e: (-e["count"], e["sample"]),
    )


def _hash_to_index(s: str, mod: int) -> int:
    """FNV-1a-32 mirror of ``hashToIndex`` in the TS file. Used for
    stable color assignment per sample name."""
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return abs(h) % mod


# ----- Smokes ---------------------------------------------------------

def smoke_basic_pivot() -> None:
    """Build a synthetic stream of 20 steps, each with a breakdown of 3
    samples drawn at random from a 5-name alphabet. Verify per-sample
    series counts add up to 60, per-step union covers all 20 steps,
    and ranking is stable by count then alphabetical."""
    rng = random.Random(0xC0FFEE)
    names = ["a.png", "b.png", "c.png", "d.png", "e.png"]
    steps = list(range(1, 21))
    points: List[Dict[str, Any]] = []
    expected_counts: Dict[str, int] = {n: 0 for n in names}
    expected_steps: Dict[str, set] = {n: set() for n in names}

    for step in steps:
        chosen = rng.sample(names, 3)
        samples = []
        for nm in chosen:
            v = rng.uniform(0.0, 1.0)
            t = rng.uniform(0.0, 1.0)
            samples.append({"value": v, "t": t, "sample": nm})
            expected_counts[nm] += 1
            expected_steps[nm].add(step)
        bd = {"samples": samples, "n": len(samples), "mean": None, "std": None}
        points.append({"step": step, "value": sum(s["value"] for s in samples) / len(samples), "breakdown": bd})

    pivoted = _pivot_by_sample(points)

    # Every sample present.
    assert len(pivoted) == 5, f"expected 5 series, got {len(pivoted)}: {[p['sample'] for p in pivoted]}"

    # Each series's point count matches the expected count.
    for entry in pivoted:
        nm = entry["sample"]
        assert entry["count"] == expected_counts[nm], (
            f"{nm}: count {entry['count']} != expected {expected_counts[nm]}"
        )
        # Step coverage is exactly the steps where the sample appeared —
        # NOT interpolated across the gaps.
        actual_steps = {pt["step"] for pt in entry["points"]}
        assert actual_steps == expected_steps[nm], (
            f"{nm}: steps {actual_steps} != expected {expected_steps[nm]}"
        )
        # Points are sorted by step.
        steps_sorted = [pt["step"] for pt in entry["points"]]
        assert steps_sorted == sorted(steps_sorted), f"{nm}: points not sorted"
        # Each point has a finite value and (in this synthetic) a t.
        for pt in entry["points"]:
            assert isinstance(pt["value"], float)
            assert pt["t"] is not None

    total = sum(e["count"] for e in pivoted)
    assert total == 60, f"total observations {total} != 60"

    # Ranking: (-count, name). Validate descending count then alphabetical.
    for i in range(len(pivoted) - 1):
        a, b = pivoted[i], pivoted[i + 1]
        if a["count"] == b["count"]:
            assert a["sample"] < b["sample"], f"alphabetical tiebreak broken: {a['sample']} vs {b['sample']}"
        else:
            assert a["count"] > b["count"], f"count ordering broken: {a['count']} vs {b['count']}"

    print(f"[basic] OK — 5 series, {total} observations, sparse coverage preserved")


def smoke_drops_invalid_tags() -> None:
    """Empty / None / whitespace-only sample tags must be dropped.
    Non-finite or non-numeric values must also be dropped."""
    points = [
        {
            "step": 1,
            "value": 0.5,
            "breakdown": {
                "samples": [
                    {"value": 0.7, "sample": "good.png"},
                    {"value": 0.4, "sample": ""},          # empty
                    {"value": 0.3, "sample": "   "},         # whitespace
                    {"value": 0.2, "sample": None},          # None
                    {"value": float("nan"), "sample": "nan.png"},  # NaN
                    {"value": "not-a-number", "sample": "str.png"},  # non-numeric
                ],
                "n": 6,
                "mean": None,
                "std": None,
            },
        }
    ]
    pivoted = _pivot_by_sample(points)
    assert len(pivoted) == 1, f"expected 1 valid sample, got {[p['sample'] for p in pivoted]}"
    assert pivoted[0]["sample"] == "good.png"
    assert pivoted[0]["points"] == [{"step": 1, "value": 0.7, "t": None}]
    print("[invalid-tags] OK — empty / None / NaN / non-numeric correctly dropped")


def smoke_color_stability() -> None:
    """The color hash must map the same name to the same index across
    runs and be in [0, mod). Two names that differ should usually map
    to different indices (sanity, not strict)."""
    mod = 8
    for name in ["katie02.jpg", "kk.jpg", "IMG_5358_1.jpg"]:
        a = _hash_to_index(name, mod)
        b = _hash_to_index(name, mod)
        assert a == b
        assert 0 <= a < mod
    # Sanity: 5 names should produce at least 2 distinct indices.
    seen = {_hash_to_index(n, mod) for n in ["a.png", "b.png", "c.png", "d.png", "e.png"]}
    assert len(seen) >= 2
    print(f"[color] OK — stable hash, {len(seen)} distinct indices for 5 names")


def smoke_real_db() -> None:
    """If the test DB is present, run the pivot against the real
    breakdown payload and sanity-check it produces 1+ series with
    sensible counts. Skipped if no DB."""
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output",
        "kk_snofs_face_depth_diff_lr05",
        "loss_log.db",
    )
    if not os.path.exists(db_path):
        print("[real-db] skipped — no test db present")
        return

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        "SELECT step, value_real, value_text FROM metrics "
        "WHERE key = 'identity/sim' AND value_text IS NOT NULL "
        "ORDER BY step ASC"
    )
    rows = cur.fetchall()
    con.close()

    points: List[Dict[str, Any]] = []
    for step, value, vt in rows:
        try:
            bd = json.loads(vt)
        except Exception:
            continue
        points.append({"step": step, "value": value, "breakdown": bd})

    pivoted = _pivot_by_sample(points)
    assert len(pivoted) > 0, "expected at least one sample series in real db"

    total = sum(e["count"] for e in pivoted)
    n_samples = len(pivoted)
    print(f"[real-db] OK — {n_samples} sample series across {len(points)} steps, {total} observations")
    top3_str = ", ".join("{}({})".format(e["sample"], e["count"]) for e in pivoted[:3])
    print(f"          top-3: {top3_str}")

    # Sanity: every series's points are within the step range and sorted.
    min_step = min(p["step"] for p in points)
    max_step = max(p["step"] for p in points)
    for entry in pivoted:
        for pt in entry["points"]:
            assert min_step <= pt["step"] <= max_step
        steps = [pt["step"] for pt in entry["points"]]
        assert steps == sorted(steps)


def main() -> int:
    smoke_basic_pivot()
    smoke_drops_invalid_tags()
    smoke_color_stability()
    smoke_real_db()
    print("All by_sample-pivot smokes passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
