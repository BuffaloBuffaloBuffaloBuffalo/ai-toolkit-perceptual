"""Model-aware cache identities and native video-VAE perceptor decoding."""
import copy
import hashlib
import json
import os

import torch

from toolkit.dto import DTO


def perceptor_cache_suffix(file_item, namespace, **settings):
    """Isolate targets by codec, crop, flips, clip selection and perceptor settings."""
    info = file_item.get_latent_info_dict()
    stat = os.stat(file_item.path)
    payload = dict(namespace=namespace, latent=info, settings=settings,
                   source_size=stat.st_size, source_mtime_ns=stat.st_mtime_ns)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


@torch.no_grad()
def decode_native_perceptor_file(model, file_item, transform, device):
    """Return TCHW [0,1] from the training latent, or the exact dataset transform.

    Never silently fall back after a corrupt latent read. Work on a copy so GT
    extraction doesn't populate shared dataset tensors or alter worker state.
    """
    item = copy.copy(file_item)
    latent = item.get_latent()
    if latent is None:
        item.load_and_process_image(transform, only_load_latents=True)
        latent = model.encode_images(item.tensor.unsqueeze(0), device=device)
    else:
        if isinstance(latent, DTO):
            latent = latent.tensor
        latent = latent.unsqueeze(0)
    if isinstance(latent, DTO):
        latent = latent.tensor
    if latent.ndim != 5:
        raise ValueError(f"Native video perceptor expects BCTHW latents, got {latent.shape}")
    pixels = model.decode_latents(latent, device=device, dtype=torch.float32)
    return ((pixels[0].permute(1, 0, 2, 3) + 1.0) * 0.5).clamp(0, 1)
