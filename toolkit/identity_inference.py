"""Load identity projectors and embeddings from an identity-enhanced LoRA file.

The LoRA safetensors file contains:
  - Standard LoRA keys: lora_unet_*, lora_te_*
  - Face ID projector: face_id_projector.*
  - Vision face projector: vision_face_projector.*
  - Body ID projector: body_id_projector.*
  - Averaged identity embeddings: identity.face_embedding, identity.vision_face_embedding, identity.body_embedding
"""
import torch
from collections import OrderedDict
from typing import Optional, Dict
from safetensors.torch import load_file


def _extract_by_prefix(state_dict: dict, prefix: str) -> dict:
    """Extract and strip keys matching a prefix."""
    return {
        k[len(prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }


def _infer_projector_dims(proj_sd: dict) -> dict:
    """Infer projector constructor args from its state_dict shapes."""
    dims = {}
    if 'norm.weight' in proj_sd:
        dims['id_dim'] = proj_sd['norm.weight'].shape[0]
    # Last linear layer output = hidden_size * num_tokens
    # Find the last proj layer
    proj_keys = sorted([k for k in proj_sd if k.startswith('proj.') and k.endswith('.weight')])
    if proj_keys:
        last_weight = proj_sd[proj_keys[-1]]
        dims['total_output'] = last_weight.shape[0]
    if 'output_scale' in proj_sd:
        dims['has_scale'] = True
    return dims


def _read_metadata(lora_path: str) -> dict:
    """Read safetensors metadata header."""
    import json
    from safetensors import safe_open
    with safe_open(lora_path, framework='pt') as f:
        meta = f.metadata() or {}
    return meta


def load_identity_from_lora(
    lora_path: str,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Optional[torch.Tensor]]:
    """Load identity tokens from an identity-enhanced LoRA file.

    Reconstructs projectors from saved weights, runs the averaged embeddings
    through them, and returns ready-to-use identity tokens.

    Args:
        lora_path: Path to the .safetensors LoRA file
        device: Device for computation
        dtype: Dtype for computation

    Returns:
        dict with keys:
          'face_tokens': (1, num_tokens, hidden_size) or None
          'vision_tokens': (1, num_tokens, hidden_size) or None
          'body_tokens': (1, num_tokens, hidden_size) or None
          'all_tokens': concatenated non-None tokens (1, total_tokens, hidden_size) or None
    """
    import json
    state_dict = load_file(lora_path)
    meta = _read_metadata(lora_path)

    # Extract projector state dicts and embeddings
    face_proj_sd = _extract_by_prefix(state_dict, 'face_id_projector.')
    vision_proj_sd = _extract_by_prefix(state_dict, 'vision_face_projector.')
    body_proj_sd = _extract_by_prefix(state_dict, 'body_id_projector.')

    face_emb = state_dict.get('identity.face_embedding', None)
    vision_emb = state_dict.get('identity.vision_face_embedding', None)
    body_emb = state_dict.get('identity.body_embedding', None)

    result = {
        'face_tokens': None,
        'vision_tokens': None,
        'body_tokens': None,
        'all_tokens': None,
    }

    token_parts = []

    # Face ID projector
    if face_proj_sd and face_emb is not None:
        from toolkit.face_id import FaceIDProjector
        id_dim = face_proj_sd['norm.weight'].shape[0] if 'norm.weight' in face_proj_sd else 512
        # Last linear: output = hidden_size * num_tokens
        # Second linear input = id_dim * 2 (from MLP structure)
        # So last linear shape is (hidden_size * num_tokens, id_dim * 2)
        last_bias = face_proj_sd.get('proj.2.bias', None)
        total_out = last_bias.shape[0] if last_bias is not None else 4096 * 4
        # FaceIDProjector has num_tokens stored implicitly — use first linear
        # to find id_dim, then total_out / possible hidden_sizes
        # Use metadata for num_tokens, fallback to default
        face_cfg = json.loads(meta.get('face_id_config', '{}'))
        num_tokens = face_cfg.get('num_tokens', 4)
        hs = total_out // num_tokens

        proj = FaceIDProjector(id_dim=id_dim, hidden_size=hs, num_tokens=num_tokens)
        proj.load_state_dict(face_proj_sd)
        proj.eval().to(device, dtype)
        with torch.no_grad():
            tokens = proj(face_emb.unsqueeze(0).to(device, dtype))
        result['face_tokens'] = tokens
        token_parts.append(tokens)

    # Vision face projector
    if vision_proj_sd and vision_emb is not None:
        from toolkit.face_id import VisionFaceProjector
        # Infer dims from resampler weights
        proj_in_weight = vision_proj_sd.get('resampler.proj_in.weight', None)
        proj_out_weight = vision_proj_sd.get('resampler.proj_out.weight', None)
        latents = vision_proj_sd.get('resampler.latents', None)

        vision_dim = proj_in_weight.shape[1] if proj_in_weight is not None else 1024
        hidden_size = proj_out_weight.shape[0] if proj_out_weight is not None else 4096
        num_tokens = latents.shape[1] if latents is not None else 4
        max_seq_len = vision_emb.shape[0]

        proj = VisionFaceProjector(
            vision_dim=vision_dim, hidden_size=hidden_size,
            num_tokens=num_tokens, max_seq_len=max_seq_len,
        )
        proj.load_state_dict(vision_proj_sd)
        proj.eval().to(device, dtype)
        with torch.no_grad():
            tokens = proj(vision_emb.unsqueeze(0).to(device, dtype))
        result['vision_tokens'] = tokens
        token_parts.append(tokens)

    # Body ID projector
    if body_proj_sd and body_emb is not None:
        from toolkit.body_id import BodyIDProjector
        id_dim = body_proj_sd['norm.weight'].shape[0] if 'norm.weight' in body_proj_sd else 10
        # Last linear in body projector is proj.4 (3-layer MLP: 0,2,4)
        last_bias = body_proj_sd.get('proj.4.bias', None)
        total_out = last_bias.shape[0] if last_bias is not None else 4096 * 4
        body_cfg = json.loads(meta.get('body_id_config', '{}'))
        num_tokens = body_cfg.get('num_tokens', 4)
        hs = total_out // num_tokens

        proj = BodyIDProjector(id_dim=id_dim, hidden_size=hs, num_tokens=num_tokens)
        proj.load_state_dict(body_proj_sd)
        proj.eval().to(device, dtype)
        with torch.no_grad():
            tokens = proj(body_emb.unsqueeze(0).to(device, dtype))
        result['body_tokens'] = tokens
        token_parts.append(tokens)

    # Concatenate all tokens
    if token_parts:
        result['all_tokens'] = torch.cat(token_parts, dim=1)

    return result


def has_identity(lora_path: str) -> bool:
    """Check if a LoRA file contains identity projectors/embeddings."""
    from safetensors import safe_open
    with safe_open(lora_path, framework='pt') as f:
        keys = f.keys()
        return any(k.startswith('face_id_projector.') or
                   k.startswith('identity.') or
                   k.startswith('body_id_projector.') for k in keys)
