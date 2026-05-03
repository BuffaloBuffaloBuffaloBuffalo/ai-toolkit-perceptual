"""VAE Perceptual Anchor Loss.

Uses the Flux 2 VAE encoder as a frozen perceptual discriminator.
Encode x0_pred pixels through the VAE encoder, compare intermediate
features against cached reference encodings at multiple scales.

The loss is differentiable: gradients flow from feature comparison
back through the VAE encoder to x0_pixels to x0_pred to the
diffusion model.
"""

import os
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tqdm import tqdm

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import FileItemDTO
    from toolkit.config_modules import FaceIDConfig


def _apply_dataloader_transform(img, file_item):
    """PIL → PIL: mirror dataloader_mixins.load_and_process_image 774-793.

    Applies flip_x/y + bucket resize + crop when the file_item has the
    bucket params attached (post-setup_buckets); returns the input image
    unchanged when they aren't.
    """
    from PIL import Image as _PILImage
    if getattr(file_item, 'flip_x', False):
        img = img.transpose(_PILImage.FLIP_LEFT_RIGHT)
    if getattr(file_item, 'flip_y', False):
        img = img.transpose(_PILImage.FLIP_TOP_BOTTOM)
    stw = getattr(file_item, 'scale_to_width', None)
    sth = getattr(file_item, 'scale_to_height', None)
    cx = getattr(file_item, 'crop_x', None)
    cy = getattr(file_item, 'crop_y', None)
    cw = getattr(file_item, 'crop_width', None)
    ch = getattr(file_item, 'crop_height', None)
    if None in (stw, sth, cx, cy, cw, ch):
        return img
    img = img.resize((int(stw), int(sth)), _PILImage.BICUBIC)
    img = img.crop((int(cx), int(cy), int(cx) + int(cw), int(cy) + int(ch)))
    return img

# Feature extraction points in the Flux 2 VAE encoder.
# The encoder has 4 resolution levels with ch_mult = [1, 2, 4, 4].
# ch=128, so channels per level: 128, 256, 512, 512.
# Each level has 2 ResnetBlocks, then a Downsample (except last level).
# We capture the output after each level's final ResnetBlock:
#   Level 0: 128 ch, H/1 x W/1 (before first downsample)
#   Level 1: 256 ch, H/2 x W/2 (before second downsample)
#   Level 2: 512 ch, H/4 x W/4 (before third downsample)
#   Level 3: 512 ch, H/8 x W/8 (no downsample, lowest resolution)
# Plus the mid block output: 512 ch, H/8 x W/8.
#
# We skip level 0 (too high res, expensive to cache/compare, and
# captures mostly local pixel details not perceptual structure).

FEATURE_LEVELS = ['level_0', 'level_1', 'level_2', 'level_3', 'mid']

# Channel counts per feature level in Flux 2 encoder (ch_mult = [1, 2, 4, 4], ch=128)
LEVEL_CHANNELS = {
    'level_0': 128,
    'level_1': 256,
    'level_2': 512,
    'level_3': 512,
    'mid': 512,
}


class VAEAnchorProjector(nn.Module):
    """Learned per-level projection heads for VAE anchor loss.

    Small 1x1 convs that learn to align predicted features with reference
    features, absorbing the representation gap between SDXL-decoded pixels
    and clean pixels when encoded through Flux 2. Trains alongside LoRA.
    """

    def __init__(self):
        super().__init__()
        self.projectors = nn.ModuleDict({
            level: nn.Conv2d(ch, ch, 1, bias=False)
            for level, ch in LEVEL_CHANNELS.items()
        })
        # Initialize as identity so projector starts as pass-through
        for proj in self.projectors.values():
            nn.init.eye_(proj.weight.view(proj.weight.shape[0], -1))

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            k: self.projectors[k](v) if k in self.projectors else v
            for k, v in features.items()
        }


class VAEAnchorEncoder(nn.Module):
    """Frozen Flux 2 VAE encoder for perceptual anchor loss.

    Loads only the encoder portion of the Flux 2 VAE. Registers forward
    hooks on intermediate layers to capture multi-scale features.
    """

    def __init__(self, vae_path: str = ''):
        super().__init__()
        self._features: Dict[str, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._encoder = None
        self._loaded = False
        self._vae_path = vae_path

    @staticmethod
    def _resolve_vae_path(vae_path: str) -> str:
        """Resolve VAE path: use explicit path, or auto-download from HuggingFace."""
        if vae_path and os.path.exists(vae_path):
            return vae_path

        # Auto-download from HuggingFace
        from huggingface_hub import hf_hub_download
        print("  VAE anchor: downloading Flux 2 VAE from ai-toolkit/flux2_vae...")
        return hf_hub_download(
            repo_id="ai-toolkit/flux2_vae",
            filename="ae.safetensors",
        )

    def load(self, device: torch.device, dtype: torch.dtype):
        """Load the encoder from a full VAE safetensors file or AutoEncoder state."""
        if self._loaded:
            return

        self._vae_path = self._resolve_vae_path(self._vae_path)

        # Import autoencoder directly to avoid triggering the full
        # extensions_built_in package __init__ chain (which has heavy deps)
        import importlib.util
        _ae_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'extensions_built_in', 'diffusion_models', 'flux2', 'src', 'autoencoder.py'
        )
        _spec = importlib.util.spec_from_file_location('flux2_autoencoder', _ae_path)
        _ae_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_ae_mod)
        AutoEncoder = _ae_mod.AutoEncoder
        AutoEncoderParams = _ae_mod.AutoEncoderParams

        # Build full autoencoder to get the correct architecture
        params = AutoEncoderParams()
        ae = AutoEncoder(params)

        state_dict = load_file(self._vae_path)
        encoder_keys = {k: v for k, v in state_dict.items() if k.startswith('encoder.')}
        if encoder_keys:
            encoder_sd = {k[len('encoder.'):]: v for k, v in encoder_keys.items()}
        else:
            encoder_sd = state_dict

        ae.encoder.load_state_dict(encoder_sd, strict=False)
        bn_keys = {k: v for k, v in state_dict.items() if k.startswith('bn.')}
        if bn_keys:
            bn_sd = {k[len('bn.'):]: v for k, v in bn_keys.items()}
            ae.bn.load_state_dict(bn_sd, strict=False)

        self._encoder = ae.encoder
        self._encoder.to(device=device, dtype=dtype)
        self._encoder.eval()
        self._encoder.requires_grad_(False)

        # Register hooks to capture intermediate features
        self._register_hooks()
        self._loaded = True

    def _register_hooks(self):
        """Register forward hooks on encoder layers to capture features."""
        # Remove any existing hooks
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._features.clear()

        encoder = self._encoder

        # Level 0: after down[0].block[1] (last resblock of level 0)
        # 128 channels, H/1 x W/1 (full resolution)
        def hook_level_0(module, input, output):
            self._features['level_0'] = output
        self._hooks.append(
            encoder.down[0].block[1].register_forward_hook(hook_level_0)
        )

        # Level 1: after down[1].block[1] (last resblock of level 1)
        # 256 channels, H/2 x W/2
        def hook_level_1(module, input, output):
            self._features['level_1'] = output
        self._hooks.append(
            encoder.down[1].block[1].register_forward_hook(hook_level_1)
        )

        # Level 2: after down[2].block[1] (last resblock of level 2)
        # 512 channels, H/4 x W/4
        def hook_level_2(module, input, output):
            self._features['level_2'] = output
        self._hooks.append(
            encoder.down[2].block[1].register_forward_hook(hook_level_2)
        )

        # Level 3: after down[3].block[1] (last resblock of level 3)
        # 512 channels, H/8 x W/8
        def hook_level_3(module, input, output):
            self._features['level_3'] = output
        self._hooks.append(
            encoder.down[3].block[1].register_forward_hook(hook_level_3)
        )

        # Mid block: after mid.block_2
        # 512 channels, H/8 x W/8
        def hook_mid(module, input, output):
            self._features['mid'] = output
        self._hooks.append(
            encoder.mid.block_2.register_forward_hook(hook_mid)
        )

    def encode_with_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Encode image and return final latent + intermediate features.

        Args:
            x: Image tensor in [-1, 1] range, shape (B, 3, H, W).

        Returns:
            Tuple of (final_encoding, features_dict).
            features_dict maps level names to feature tensors.
        """
        assert self._loaded, "Call load() first"
        self._features.clear()

        # Cast input to encoder dtype (encoder may be fp16 to save VRAM)
        enc_dtype = next(self._encoder.parameters()).dtype
        x = x.to(dtype=enc_dtype)

        # Forward through encoder (hooks capture intermediates)
        final = self._encoder(x)

        # Copy features (hooks may be overwritten on next call)
        features = {k: v for k, v in self._features.items()}
        return final, features

    @staticmethod
    def compute_loss(
        pred_features: Dict[str, torch.Tensor],
        ref_features: Dict[str, torch.Tensor],
        level_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute normalized MSE between predicted and reference features.

        For each feature level:
          1. Compute MSE per spatial position
          2. Normalize by the reference feature variance (so each level
             contributes equally regardless of its magnitude)
          3. Average over channels and spatial dimensions (per sample)

        Args:
            pred_features: Features from encoding x0_pred pixels.
            ref_features: Cached reference features from clean images.
            level_weights: Optional per-level weight multipliers.

        Returns:
            Tuple of (per_sample_loss (B,), per_level_losses_dict).
            per_level_losses are detached floats for logging.
        """
        if level_weights is None:
            # Bias toward higher-resolution levels that capture spatial detail
            level_weights = {
                'level_0': 4.0,
                'level_1': 2.0,
                'level_2': 1.0,
                'level_3': 1.0,
                'mid': 1.0,
            }

        device = next(iter(pred_features.values())).device
        batch_size = next(iter(pred_features.values())).shape[0]
        total_loss = torch.zeros(batch_size, device=device)
        per_level = {}
        n_levels = 0

        for level_name in FEATURE_LEVELS:
            if level_name not in pred_features or level_name not in ref_features:
                continue

            pred = pred_features[level_name]  # (B, C, H, W)
            ref = ref_features[level_name].to(pred.device, dtype=pred.dtype)

            # Handle size mismatch (pred may be different resolution than ref
            # if training resolution differs from caching resolution)
            if pred.shape[2:] != ref.shape[2:]:
                ref = F.interpolate(
                    ref, size=pred.shape[2:], mode='bilinear', align_corners=False
                )

            # Cosine similarity per spatial position, averaged over space
            # Less mean-seeking than MSE — focuses on feature direction not magnitude
            pred_flat = pred.flatten(2)  # (B, C, N)
            ref_flat = ref.flatten(2)    # (B, C, N)
            cos_sim = F.cosine_similarity(pred_flat, ref_flat, dim=1)  # (B, N)
            level_loss = (1.0 - cos_sim).mean(dim=1)  # (B,)

            weight = level_weights.get(level_name, 1.0)
            total_loss = total_loss + weight * level_loss
            per_level[level_name] = level_loss.detach().mean().item()
            n_levels += 1

        # Average across levels
        if n_levels > 0:
            total_loss = total_loss / n_levels

        return total_loss, per_level

    def cleanup(self):
        """Remove hooks and free memory."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._features.clear()


def encode_reference_features(
    encoder: VAEAnchorEncoder,
    pil_image,
    target_size: int = 512,
) -> Dict[str, torch.Tensor]:
    """Encode a PIL image and extract multi-scale features for caching.

    Args:
        encoder: Loaded VAEAnchorEncoder.
        pil_image: PIL Image in RGB mode.
        target_size: Resize shortest side to this before encoding.

    Returns:
        Dict mapping level names to feature tensors (on CPU, fp16).
    """
    import torchvision.transforms.functional as TF

    # Resize to standard size (match typical training resolution)
    w, h = pil_image.size
    if min(w, h) != target_size:
        scale = target_size / min(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        # Round to multiple of 8 (VAE requirement: 3 downsample levels)
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        if new_w == 0:
            new_w = 8
        if new_h == 0:
            new_h = 8
        pil_image = pil_image.resize((new_w, new_h))
    else:
        # Still round to multiple of 8
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8
        if new_w != w or new_h != h:
            pil_image = pil_image.resize((new_w, new_h))

    # To tensor [0, 1] then to [-1, 1]
    img_tensor = TF.to_tensor(pil_image).unsqueeze(0)  # (1, 3, H, W)
    img_tensor = img_tensor * 2.0 - 1.0  # [0,1] -> [-1,1]

    device = encoder._encoder.conv_in.weight.device
    dtype = encoder._encoder.conv_in.weight.dtype
    img_tensor = img_tensor.to(device=device, dtype=dtype)

    with torch.no_grad():
        _, features = encoder.encode_with_features(img_tensor)

    # Move to CPU fp16 for storage
    return {k: v.cpu().half() for k, v in features.items()}


def cache_vae_anchor_features(
    file_items: List['FileItemDTO'],
    face_id_config: 'FaceIDConfig',
):
    """Extract and cache VAE encoder features for all file items.

    Caches to {image_dir}/_face_id_cache/{filename}_vae_anchor.safetensors.
    Separate file from other caches due to larger size.
    """
    from PIL import Image
    from PIL.ImageOps import exif_transpose

    CACHE_VERSION_KEY = 'vae_anchor_v4'  # v4: cached from dataloader-transformed pixels

    encoder = VAEAnchorEncoder(vae_path=face_id_config.vae_anchor_model_path)
    encoder.load(device=torch.device('cuda'), dtype=torch.float32)
    print("  -  Loaded VAE encoder for anchor feature caching")

    cached_count = 0
    computed_count = 0

    for file_item in tqdm(file_items, desc="Caching VAE anchor features"):
        img_dir = os.path.dirname(file_item.path)
        cache_dir = os.path.join(img_dir, '_face_id_cache')
        filename_no_ext = os.path.splitext(os.path.basename(file_item.path))[0]
        cache_path = os.path.join(cache_dir, f'{filename_no_ext}_vae_anchor.safetensors')

        # Check cache
        if os.path.exists(cache_path):
            data = load_file(cache_path)
            if CACHE_VERSION_KEY in data and all(
                f'vae_anchor_{level}' in data for level in FEATURE_LEVELS
            ):
                # Load cached features onto file_item
                file_item.vae_anchor_features = {
                    level: data[f'vae_anchor_{level}'].clone()
                    for level in FEATURE_LEVELS
                }
                cached_count += 1
                continue

        # v4: encode the dataloader-transformed pixels so cached features
        # share spatial geometry with the training tensor. Apply the same
        # flip + bucket-resize + crop the dataloader uses.
        raw_pil = exif_transpose(Image.open(file_item.path)).convert('RGB')
        pil_image = _apply_dataloader_transform(raw_pil, file_item)

        # target_size chosen so encode_reference_features doesn't re-resize —
        # set it equal to the shortest side of the already-transformed PIL.
        w, h = pil_image.size
        target_size = min(w, h)
        features = encode_reference_features(encoder, pil_image, target_size=target_size)

        file_item.vae_anchor_features = features
        computed_count += 1

        # Save to cache
        os.makedirs(cache_dir, exist_ok=True)
        save_data = {CACHE_VERSION_KEY: torch.ones(1)}
        for level_name, feat_tensor in features.items():
            save_data[f'vae_anchor_{level_name}'] = feat_tensor
        save_file(save_data, cache_path)

    # Free encoder
    encoder.cleanup()
    del encoder
    torch.cuda.empty_cache()

    print(f"  -  VAE anchor features: {cached_count} cached, {computed_count} computed")
