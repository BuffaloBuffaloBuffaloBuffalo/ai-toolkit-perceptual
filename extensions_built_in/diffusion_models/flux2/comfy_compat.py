from collections import Counter
from typing import Callable, Dict, List, Optional

import torch

# Prefixes that ComfyUI / original-format checkpoints use to namespace the
# diffusion transformer inside a combined checkpoint. The native Flux2 module
# uses bare keys (e.g. "double_blocks.0.img_attn.qkv.weight"), which is also
# what a ComfyUI "diffusion_models" (transformer-only) export already uses, so
# those load unchanged.
_DIT_PREFIXES = ("model.diffusion_model.", "diffusion_model.")


def _summarize_keys(keys: List[str], max_groups: int = 6) -> str:
    # group by top-level segment so a few hundred dropped keys read as
    # "vae.* x244, text_encoders.* x106" instead of a wall of names
    groups = Counter(key.split(".")[0] for key in keys)
    parts = [f"{name}.* x{count}" for name, count in groups.most_common(max_groups)]
    if len(groups) > max_groups:
        parts.append(f"+{len(groups) - max_groups} more")
    return ", ".join(parts)


def normalize_transformer_state_dict(
    state_dict: Dict[str, torch.Tensor],
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, torch.Tensor]:
    """Map a ComfyUI / original-format Flux2 checkpoint onto the native keys.

    Handles three layouts so the user never has to convert a checkpoint first:
      - bare DiT keys (ComfyUI ``diffusion_models`` export) -> returned as-is
      - DiT namespaced under ``model.diffusion_model.`` / ``diffusion_model.``
        -> prefix stripped
      - a full ComfyUI checkpoint that bundles the vae / text encoders next to a
        prefixed DiT -> only the DiT subtree is kept (the bundled tensors, which
        the toolkit loads separately, are dropped)

    Only key names change; tensor values are passed through untouched. Pass
    ``log`` (e.g. ``self.print_and_status_update``) to report what happened to
    the job log.
    """
    log = log if log is not None else (lambda _msg: None)

    for prefix in _DIT_PREFIXES:
        if any(key.startswith(prefix) for key in state_dict):
            kept = {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            dropped = [key for key in state_dict if not key.startswith(prefix)]
            example = next(iter(kept), None)
            log(
                f"ComfyUI checkpoint detected: stripped '{prefix}' from "
                f"{len(kept)} transformer keys"
                + (f" (e.g. '{prefix}{example}' -> '{example}')" if example else "")
            )
            if dropped:
                log(
                    f"  ignored {len(dropped)} non-transformer keys "
                    f"({_summarize_keys(dropped)}); the VAE and text encoder are "
                    f"loaded separately"
                )
            return kept

    log("Transformer checkpoint already uses native key names; no renaming needed")
    return state_dict
