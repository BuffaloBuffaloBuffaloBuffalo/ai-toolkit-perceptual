"""Metric aggregation buffer used by the SDTrainer.

Pre-overhaul, each metric in SDTrainer was stored as a single
``self._last_<name>`` scalar that got *overwritten* on every microbatch.
With ``gradient_accumulation_steps > 1`` (or anything that calls
``train_single_accumulation`` more than once before
``hook_train_loop`` flushes), this meant every metric except ``loss``
reflected only the **final** microbatch.

``MetricBuffer`` solves that by accumulating:

  * scalar metrics — weighted running mean across microbatches
    (``add_scalar`` / ``flush_scalars``).
  * per-sample breakdowns — full sample list capped at ``per_sample_cap``,
    selected by largest absolute deviation from the running mean
    (``add_per_sample`` / ``flush_per_sample``). The flushed payload
    matches the JSON ``breakdown`` shape consumed by the UI tooltip:
    ``{"samples": [...], "n": int, "mean": float, "std": float}``.

The buffer is **display-only** — it never participates in any loss
tensor or gradient computation. SDTrainer mirrors every existing
``self._last_<name> = scalar`` write into the buffer; the existing
``_last_*`` attributes stay live for back-compat. ``get_loss_metrics``
prefers the buffer's flushed mean when available.

Usage::

    buf = MetricBuffer()
    # at the start of every optimizer step:
    buf.reset()
    # during each microbatch:
    buf.add_scalar('id_sim', 0.7, weight=batch_size)
    buf.add_per_sample('id_sim', value=0.7, t=0.4, sample_tag='img.png')
    # at flush time inside hook_train_loop's metric assembly:
    scalars = buf.flush_scalars()        # {name: weighted_mean}
    breakdowns = buf.flush_per_sample()  # {name: {'samples': [...], 'n': ...}}

The class is designed to be safe under repeated ``flush_*`` calls (each
flush clears the corresponding state) and tolerant of unknown / unused
metric names.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class MetricValue(float):
    """Float-with-payload used to ship a per-sample breakdown alongside the
    metric scalar through ``loss_dict`` without breaking any existing
    consumer.

    Subclasses ``float`` so all arithmetic, formatting, comparisons, and
    ``f"{val:.3e}"`` formatting just work. The ``breakdown`` attribute
    carries the JSON-serialisable per-sample payload that the logger picks
    up via ``_coerce_value``.
    """

    breakdown: Dict[str, Any]

    def __new__(cls, value: float, breakdown: Dict[str, Any]):
        inst = super().__new__(cls, value)
        inst.breakdown = breakdown
        return inst

    def __repr__(self) -> str:  # pragma: no cover — cosmetic
        return f"MetricValue({float(self)!r}, breakdown_keys={list(self.breakdown.keys())})"


class _ScalarSlot:
    __slots__ = ("sum", "weight")

    def __init__(self) -> None:
        self.sum: float = 0.0
        self.weight: float = 0.0

    def add(self, value: float, weight: float) -> None:
        if weight <= 0.0:
            return
        self.sum += float(value) * float(weight)
        self.weight += float(weight)

    def mean(self) -> Optional[float]:
        if self.weight <= 0.0:
            return None
        return self.sum / self.weight


class _PerSampleSlot:
    """Holds every sample we've seen so far for a metric, until we flush.

    We keep the full list and let ``flush`` decide which K to retain.
    Per-step counts are tiny (batch_size × gradient_accumulation_steps,
    typically < 64), so the list cost is negligible.
    """

    __slots__ = ("samples",)

    def __init__(self) -> None:
        self.samples: List[Dict[str, Any]] = []

    def add(self, value: float, t: Optional[float], sample_tag: Optional[str]) -> None:
        entry: Dict[str, Any] = {"value": float(value)}
        if t is not None:
            entry["t"] = float(t)
        if sample_tag is not None:
            entry["sample"] = str(sample_tag)
        self.samples.append(entry)

    def finalize(self, cap: int) -> Dict[str, Any]:
        n = len(self.samples)
        if n == 0:
            return {"samples": [], "n": 0, "mean": None, "std": None}

        values = [s["value"] for s in self.samples]
        mean = sum(values) / n
        if n > 1:
            var = sum((v - mean) ** 2 for v in values) / n
            std = math.sqrt(var)
        else:
            std = 0.0

        # Top-K-by-deviation: keep the K samples whose |value - mean| is
        # largest. These are the most informative ones to show the user
        # ("which samples drove the mean up / down?").
        if n > cap:
            ordered = sorted(
                self.samples,
                key=lambda s: abs(s["value"] - mean),
                reverse=True,
            )
            kept = ordered[:cap]
        else:
            kept = list(self.samples)

        return {
            "samples": kept,
            "n": n,
            "mean": mean,
            "std": std,
        }


class MetricBuffer:
    """Accumulates scalar means and per-sample breakdowns across microbatches.

    Parameters
    ----------
    per_sample_cap:
        Maximum number of samples retained per metric in the per-sample
        breakdown payload. Defaults to 16 (matches the audit's spec).
    """

    def __init__(self, per_sample_cap: int = 16) -> None:
        self._scalars: Dict[str, _ScalarSlot] = {}
        self._per_sample: Dict[str, _PerSampleSlot] = {}
        self._per_sample_cap = int(per_sample_cap)

    # --- mutation -----------------------------------------------------

    def reset(self) -> None:
        """Clear all scalar and per-sample state. Call at the start of
        every optimizer step (i.e. once per ``hook_train_loop`` invocation)."""
        self._scalars.clear()
        self._per_sample.clear()

    def add_scalar(self, name: str, value: Any, weight: float = 1.0) -> None:
        """Mirror a metric scalar into the buffer with a weight.

        Non-finite or ``None`` values are ignored. Zero-weight entries are
        ignored (e.g. metric_mask had no valid samples). Existing
        ``self._last_<name>`` writes in SDTrainer are unchanged; the
        buffer just *also* records the value so that
        ``gradient_accumulation_steps > 1`` reports a real mean instead of
        the last microbatch's value.
        """
        if value is None:
            return
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(v):
            return
        if weight is None or weight <= 0.0:
            return
        slot = self._scalars.get(name)
        if slot is None:
            slot = _ScalarSlot()
            self._scalars[name] = slot
        slot.add(v, float(weight))

    def add_per_sample(
        self,
        name: str,
        value: Any,
        t: Optional[float] = None,
        sample_tag: Optional[str] = None,
    ) -> None:
        """Record one sample's value for the per-sample breakdown of a metric.

        The flushed payload reports n=total observed, mean/std over all
        observations, and ``samples`` capped at ``per_sample_cap`` (top-K
        by deviation from the running mean). Non-finite / ``None`` values
        are dropped silently.
        """
        if value is None:
            return
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(v):
            return
        slot = self._per_sample.get(name)
        if slot is None:
            slot = _PerSampleSlot()
            self._per_sample[name] = slot
        slot.add(v, t, sample_tag)

    # --- flush --------------------------------------------------------

    def flush_scalars(self) -> Dict[str, float]:
        """Return ``{name: weighted_mean}`` and clear scalar state."""
        out: Dict[str, float] = {}
        for name, slot in self._scalars.items():
            mean = slot.mean()
            if mean is not None:
                out[name] = mean
        self._scalars.clear()
        return out

    def flush_per_sample(self) -> Dict[str, Dict[str, Any]]:
        """Return ``{name: payload}`` and clear per-sample state.

        ``payload`` matches the JSON shape consumed by the UI tooltip:
        ``{"samples": [...], "n": int, "mean": float, "std": float}``.
        Empty / unused metrics are omitted.
        """
        out: Dict[str, Dict[str, Any]] = {}
        for name, slot in self._per_sample.items():
            payload = slot.finalize(self._per_sample_cap)
            if payload["n"] > 0:
                out[name] = payload
        self._per_sample.clear()
        return out
