from typing import Dict

import torch

# Prefixes that ComfyUI / original-format checkpoints use to namespace the
# diffusion transformer inside a combined checkpoint. The native Flux2 module
# uses bare keys (e.g. "double_blocks.0.img_attn.qkv.weight"), which is also
# what a ComfyUI "diffusion_models" (transformer-only) export already uses, so
# those load unchanged.
_DIT_PREFIXES = ("model.diffusion_model.", "diffusion_model.")


def normalize_transformer_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Map a ComfyUI / original-format Flux2 checkpoint onto the native keys.

    Handles three layouts so the user never has to convert a checkpoint first:
      - bare DiT keys (ComfyUI ``diffusion_models`` export) -> returned as-is
      - DiT namespaced under ``model.diffusion_model.`` / ``diffusion_model.``
        -> prefix stripped
      - a full ComfyUI checkpoint that bundles the vae / text encoders next to a
        prefixed DiT -> only the DiT subtree is kept (the bundled tensors, which
        the toolkit loads separately, are dropped)
    """
    for prefix in _DIT_PREFIXES:
        if any(key.startswith(prefix) for key in state_dict):
            return {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
    return state_dict
