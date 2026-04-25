"""Canonical metric naming + legacy compatibility shim.

Pre-overhaul, metric keys were a mix of conventions:

  * top-level:        ``loss``, ``grad_norm``, ``timestep``
  * snake_case loss:  ``identity_loss``, ``identity_loss_applied``,
                      ``id_clean_target``, ``id_clean_delta``
  * t-binned:         ``id_sim_t40``, ``depth_loss_t40``, ``bp_sim_t60``,
                      ``shape_sim_t40``, ``bsh_sim_t40``
  * per-level:        ``va_level_1``, ``va_mid``, ``va_edge``
  * misc:             ``face_token_norm``, ``vision_token_norm``, ...

The audit's recommended canonical shape is ``subsystem/kind/variant``
with ``/`` as the separator and the leading ``subsystem`` token used by
the new metrics dashboard for grouping.

This module exposes:

  * ``CANONICAL_RENAMES`` — ``{legacy_key: canonical_key}`` for one-to-one
    renames. A pattern entry handled in code covers ``id_sim_t<NN>``-style
    bin keys.
  * ``apply_dual_write(loss_dict)`` — returns a new dict that contains
    every legacy key (back-compat for ~one release) **and** the canonical
    key for each mapped metric, both pointing at the same value. Wrapping
    a ``MetricValue`` is preserved (it's still a ``float`` subclass).

Consumers:

  * ``SDTrainer.hook_train_loop`` calls ``apply_dual_write`` on the
    finished ``loss_dict`` before returning.
  * ``BaseSDTrainProcess`` epoch-avg block writes both legacy and
    canonical keys for ``loss/epoch_avg`` etc.
  * ``ui/src/hooks/useJobLossLog.tsx`` carries an inverse map so old
    sqlite logs render under the new namespace transparently.
"""

from __future__ import annotations

import re
from typing import Dict, Mapping, MutableMapping


# --- one-to-one renames ---------------------------------------------------
# Keep this map in lock-step with `LEGACY_TO_CANONICAL` in
# `ui/src/hooks/useJobLossLog.tsx`. When you add an entry here, mirror it
# there so old runs stored under the legacy key surface under the new
# namespace in the new dashboard tab.
CANONICAL_RENAMES: Dict[str, str] = {
    # core
    "loss": "core/loss",
    "grad_norm": "core/grad_norm",
    "timestep": "core/timestep",

    # diffusion / training
    "diffusion_loss": "diffusion/loss_raw",
    "diffusion_loss_applied": "diffusion/loss_applied",

    # identity
    "identity_loss": "identity/loss_raw",
    "identity_loss_applied": "identity/loss_applied",
    "id_sim": "identity/sim",
    "id_clean_target": "identity/clean_target",
    "id_clean_delta": "identity/clean_delta",

    # landmark
    "landmark_loss": "landmark/loss_raw",
    "landmark_loss_applied": "landmark/loss_applied",

    # body proportion
    "body_proportion_loss": "body_proportion/loss_raw",
    "body_proportion_loss_applied": "body_proportion/loss_applied",

    # body shape
    "body_shape_loss": "body_shape/loss_raw",
    "body_shape_loss_applied": "body_shape/loss_applied",
    "body_shape_cos": "body_shape/cos",
    "body_shape_l1": "body_shape/l1",
    "body_shape_gated_pct": "body_shape/gated_pct",

    # normals
    "normal_loss": "normal/loss_raw",
    "normal_loss_applied": "normal/loss_applied",
    "normal_cos": "normal/cos",

    # vae anchor
    "vae_anchor_loss": "vae_anchor/loss_raw",
    "vae_anchor_loss_applied": "vae_anchor/loss_applied",
    "va_level_1": "vae_anchor/level/level_1",
    "va_level_2": "vae_anchor/level/level_2",
    "va_level_3": "vae_anchor/level/level_3",
    "va_mid": "vae_anchor/level/mid",
    "va_edge": "vae_anchor/level/edge",

    # depth consistency
    "depth_consistency_loss": "depth/loss_raw",
    "depth_consistency_loss_applied": "depth/loss_applied",
    "depth_consistency_ssi": "depth/ssi",
    "depth_consistency_grad": "depth/grad",

    # tokens
    "face_token_norm": "tokens/face/norm",
    "vision_token_norm": "tokens/vision/norm",
    "body_token_norm": "tokens/body/norm",
    "txt_token_norm": "tokens/text/norm",

    # aux
    "pure_noise_cos": "aux/pure_noise_cos",

    # epoch-averages (set in BaseSDTrainProcess)
    "loss/epoch_avg": "core/loss/epoch_avg",
    "loss/identity_loss_epoch_avg": "identity/loss_raw/epoch_avg",
    "loss/diffusion_loss_epoch_avg": "diffusion/loss_raw/epoch_avg",
    "id_sim_epoch_avg": "identity/sim/epoch_avg",
    "loss/body_proportion_loss_epoch_avg": "body_proportion/loss_raw/epoch_avg",
    "loss/depth_consistency_loss_epoch_avg": "depth/loss_raw/epoch_avg",
}


# --- pattern renames ------------------------------------------------------
# Pattern-shaped keys (per-t-band bins). We canonicalise to
# ``<subsystem>/<kind>/t<NN>`` so the new dashboard can group + facet
# them. Mirror these in the UI hook's `LEGACY_PREFIXES`.
_BIN_PATTERN = re.compile(r"^(?P<prefix>[a-z_]+?)_t(?P<n>\d{2,3})$")
_BIN_PREFIX_TO_CANONICAL: Dict[str, str] = {
    "id_sim": "identity/sim",
    "shape_sim": "landmark/sim",
    "bp_sim": "body_proportion/sim",
    "bsh_sim": "body_shape/sim",
    "depth_loss": "depth/loss",
}


def _canonicalize(legacy_key: str) -> str | None:
    """Return the canonical name for a legacy key, or None if no mapping
    exists.

    Order: explicit ``CANONICAL_RENAMES`` first, then the t-bin pattern.
    Adding a new explicit mapping always wins over the pattern.
    """
    if legacy_key in CANONICAL_RENAMES:
        return CANONICAL_RENAMES[legacy_key]
    m = _BIN_PATTERN.match(legacy_key)
    if m:
        prefix = m.group("prefix")
        n = m.group("n")
        canonical_prefix = _BIN_PREFIX_TO_CANONICAL.get(prefix)
        if canonical_prefix is not None:
            return f"{canonical_prefix}/t{n}"
    return None


def apply_dual_write(loss_dict: Mapping[str, object]) -> MutableMapping[str, object]:
    """Return a new dict containing every legacy key + its canonical alias.

    The legacy keys are preserved verbatim so existing dashboards / wandb
    runs keep working. Each mapped legacy key gains a canonical sibling
    pointing at the same value (``MetricValue`` instances are passed
    through unchanged so the breakdown payload still rides along).

    Idempotent: re-applying on a dict that already contains canonical
    keys does not duplicate them and does not overwrite values.
    """
    out: MutableMapping[str, object] = dict(loss_dict)
    for legacy_key in list(loss_dict.keys()):
        canonical = _canonicalize(legacy_key)
        if canonical is None:
            continue
        if canonical == legacy_key:
            continue
        # Only emit the canonical alias if it isn't already present (the
        # caller may have constructed `loss_dict` with canonical keys
        # directly).
        if canonical not in out:
            out[canonical] = loss_dict[legacy_key]
    return out


# --- introspection helpers (consumed by step 5 UI tab) --------------------
def subsystem_of(canonical_key: str) -> str:
    """Extract the leading ``subsystem`` segment of a canonical key.

    For an unmapped legacy key, returns the input unchanged (callers can
    fall back to a "Custom" group for these).
    """
    if "/" in canonical_key:
        return canonical_key.split("/", 1)[0]
    return canonical_key


__all__ = [
    "CANONICAL_RENAMES",
    "apply_dual_write",
    "subsystem_of",
]
