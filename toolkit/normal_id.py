"""Sapiens 0.3B surface normal estimation for body shape loss during training.

Predicts per-pixel surface normals from images. Compared to SMPL beta regression,
normal maps capture fine body shape detail (musculature, fat distribution, contours)
that 10-dim PCA coefficients cannot represent.

Architecture: ViT-Large (24 layers, 1024 dim, 16 heads) + ConvTranspose decoder.
Trained on 300M human images (MAE pretrain) + 500K synthetic renders (normal finetune).
"""

import os
from typing import Optional, List, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
from tqdm import tqdm

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import FileItemDTO
    from toolkit.config_modules import FaceIDConfig


# ============================================================
# Sapiens 0.3B Normal Model — pure PyTorch, no mmseg dependency
# ============================================================

class _PatchEmbed(nn.Module):
    def __init__(self, in_channels=3, embed_dim=1024, patch_size=16, padding=2):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=padding
        )

    def forward(self, x):
        return self.projection(x)


class _Attention(nn.Module):
    def __init__(self, dim=1024, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _FFN(nn.Module):
    def __init__(self, dim=1024, hidden_dim=4096):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _TransformerBlock(nn.Module):
    def __init__(self, dim=1024, num_heads=16, ffn_dim=4096):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = _FFN(dim, ffn_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class _NormalDecoder(nn.Module):
    def __init__(self, in_channels=1024, mid_channels=768):
        super().__init__()
        deconv_layers = []
        conv_layers = []
        for i in range(3):
            in_c = in_channels if i == 0 else mid_channels
            deconv_layers.extend([
                nn.ConvTranspose2d(in_c, mid_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(mid_channels, affine=False),
                nn.SiLU(inplace=True),
            ])
            conv_layers.extend([
                nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
                nn.InstanceNorm2d(mid_channels, affine=False),
                nn.SiLU(inplace=True),
            ])
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.conv_layers = nn.Sequential(*conv_layers)
        self.conv_seg = nn.Conv2d(mid_channels, 3, kernel_size=1)

    def forward(self, x):
        for i in range(3):
            x = self.deconv_layers[i * 3:(i + 1) * 3](x)
            x = self.conv_layers[i * 3:(i + 1) * 3](x)
        return self.conv_seg(x)


class SapiensNormal(nn.Module):
    """Sapiens 0.3B surface normal estimator — pure PyTorch.

    Input: (B, 3, H, W) in [0, 1] RGB, ideally 1024x768 (native) or 512x384.
    Output: (B, 3, H', W') raw normal vectors (not yet L2-normalized).

    Architecture: ViT-L (24 layers, 1024 dim) + 3x ConvTranspose2d decoder.
    Output resolution is input_H/2 x input_W/2 (8x upsample from patch tokens).
    """

    NATIVE_H = 1024
    NATIVE_W = 768

    def __init__(self, embed_dim=1024, num_layers=24, num_heads=16, ffn_dim=4096):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = _PatchEmbed(3, embed_dim, patch_size=16, padding=2)
        # 1024x768 with padding=2: (1024+4)//16 x (768+4)//16 = 64x48 = 3072
        self.pos_embed = nn.Parameter(torch.zeros(1, 64 * 48, embed_dim))
        self.layers = nn.ModuleList([
            _TransformerBlock(embed_dim, num_heads, ffn_dim) for _ in range(num_layers)
        ])
        self.ln1 = nn.LayerNorm(embed_dim)
        self.decode_head = _NormalDecoder(embed_dim)

        self.register_buffer('img_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('img_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _load_pretrained(self):
        """Download and load Sapiens 0.3B normal weights from HuggingFace."""
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(
            repo_id="facebook/sapiens-normal-0.3b",
            filename="sapiens_0.3b_normal_render_people_epoch_66.pth",
        )
        raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = raw.get('state_dict', raw)

        new_sd = {}
        for k, v in sd.items():
            nk = k
            if nk.startswith('backbone.'):
                nk = nk[len('backbone.'):]
            elif nk.startswith('decode_head.'):
                pass
            else:
                continue
            nk = nk.replace('ffn.layers.0.0.', 'ffn.fc1.')
            nk = nk.replace('ffn.layers.1.', 'ffn.fc2.')
            new_sd[nk] = v

        result = self.load_state_dict(new_sd, strict=False)
        unexpected_missing = [k for k in result.missing_keys
                              if k not in ('img_mean', 'img_std')]
        if unexpected_missing:
            print(f"  [normal_id] Warning: missing keys: {unexpected_missing}")

    def forward(self, x):
        x = (x - self.img_mean.to(x.dtype)) / self.img_std.to(x.dtype)
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        if x.shape[1] != self.pos_embed.shape[1]:
            pe = self.pos_embed.reshape(1, 64, 48, self.embed_dim).permute(0, 3, 1, 2)
            pe = F.interpolate(pe.float(), size=(H, W), mode='bilinear', align_corners=False)
            pe = pe.to(x.dtype).permute(0, 2, 3, 1).reshape(1, H * W, self.embed_dim)
            x = x + pe
        else:
            x = x + self.pos_embed

        for layer in self.layers:
            x = layer(x)

        x = self.ln1(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return self.decode_head(x)


# ============================================================
# Training-facing wrapper
# ============================================================

# Square output — fits both portrait and landscape with minimal padding
NORMAL_SIZE = 256


def _best_orientation(w, h):
    """Return (target_h, target_w) that best fits the image's aspect ratio.

    Uses Sapiens native sizes: 1024x768 (portrait) or 768x1024 (landscape).
    """
    if h >= w:
        return (1024, 768)
    else:
        return (768, 1024)


def _best_orientation_train(w, h):
    """Half-resolution version for training speed."""
    if h >= w:
        return (512, 384)
    else:
        return (384, 512)


class DifferentiableNormalEncoder(nn.Module):
    """Frozen Sapiens encoder for normal map loss during training.

    Chooses portrait (1024x768) or landscape (768x1024) orientation to match
    the input image, minimizing letterbox padding. Output is always square
    (NORMAL_SIZE x NORMAL_SIZE) for consistent batching.
    """

    def __init__(self):
        super().__init__()
        self.model = SapiensNormal()
        print("  Loading Sapiens 0.3B normal model...")
        self.model._load_pretrained()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _letterbox_pil(pil_image, target_w, target_h):
        """Resize PIL image to fit inside target_w x target_h, padding with black."""
        from PIL import Image
        w, h = pil_image.size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = pil_image.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new('RGB', (target_w, target_h), (0, 0, 0))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        canvas.paste(resized, (paste_x, paste_y))
        return canvas

    @staticmethod
    def _letterbox_tensor(pixels, target_h, target_w):
        """Resize (B, 3, H, W) tensor to fit inside target, padding with black.

        Differentiable: uses F.interpolate + F.pad.
        """
        B, C, H, W = pixels.shape
        scale = min(target_w / W, target_h / H)
        new_w, new_h = int(W * scale), int(H * scale)
        resized = F.interpolate(pixels, size=(new_h, new_w), mode='bilinear', align_corners=False)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        pad_r = target_w - new_w - pad_x
        pad_b = target_h - new_h - pad_y
        return F.pad(resized, (pad_x, pad_r, pad_y, pad_b), value=0.0)

    @torch.no_grad()
    def encode(self, pil_image) -> torch.Tensor:
        """Cache path: PIL image -> (3, NORMAL_SIZE, NORMAL_SIZE) unit normals on CPU."""
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        w, h = pil_image.size
        target_h, target_w = _best_orientation(w, h)
        img = self._letterbox_pil(pil_image, target_w, target_h)
        tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(device=device, dtype=dtype)

        raw = self.model(tensor)
        raw = F.interpolate(raw.float(), size=(NORMAL_SIZE, NORMAL_SIZE),
                            mode='bilinear', align_corners=False)
        normals = raw / (raw.norm(dim=1, keepdim=True) + 1e-5)
        return normals.squeeze(0).cpu()  # (3, 256, 256)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Training path: (B, 3, H, W) in [0,1] -> (B, 3, NORMAL_SIZE, NORMAL_SIZE) unit normals.

        Chooses portrait/landscape orientation per batch based on majority aspect ratio.
        Letterboxes to preserve geometry.
        """
        B, C, H, W = pixels.shape
        target_h, target_w = _best_orientation_train(W, H)
        pixels = self._letterbox_tensor(pixels, target_h, target_w)

        with torch.amp.autocast('cuda', enabled=False):
            model_dtype = next(self.model.parameters()).dtype
            raw = self.model(pixels.to(model_dtype))

        raw = F.interpolate(raw.float(), size=(NORMAL_SIZE, NORMAL_SIZE),
                            mode='bilinear', align_corners=False)
        normals = raw / (raw.norm(dim=1, keepdim=True) + 1e-5)
        return normals


# ============================================================
# Caching
# ============================================================

def cache_normal_embeddings(
    file_items: List['FileItemDTO'],
    face_id_config: 'FaceIDConfig',
):
    """Extract and cache Sapiens normal maps for all file items.

    Caches to {image_dir}/_face_id_cache/{filename}_normals.safetensors
    (separate file from face/body caches due to larger size ~300KB per image).
    """
    from PIL import Image
    from PIL.ImageOps import exif_transpose

    CACHE_VERSION_KEY = 'normal_v1'

    encoder = DifferentiableNormalEncoder()
    encoder.to('cuda')

    no_normal_count = 0

    for file_item in tqdm(file_items, desc="Caching normal maps"):
        img_dir = os.path.dirname(file_item.path)
        cache_dir = os.path.join(img_dir, '_face_id_cache')
        filename_no_ext = os.path.splitext(os.path.basename(file_item.path))[0]
        cache_path = os.path.join(cache_dir, f'{filename_no_ext}_normals.safetensors')

        # Check cache
        if os.path.exists(cache_path):
            data = load_file(cache_path)
            if 'normal_embedding' in data and CACHE_VERSION_KEY in data:
                file_item.normal_embedding = data['normal_embedding'].clone()
                continue

        pil_image = exif_transpose(Image.open(file_item.path)).convert('RGB')
        normals = encoder.encode(pil_image)  # (3, 256, 192)

        if normals.abs().sum() < 1e-6:
            no_normal_count += 1

        file_item.normal_embedding = normals

        os.makedirs(cache_dir, exist_ok=True)
        save_data = {
            'normal_embedding': normals.half(),  # fp16 to save disk (~300KB)
            CACHE_VERSION_KEY: torch.ones(1),
        }
        save_file(save_data, cache_path)

    del encoder
    torch.cuda.empty_cache()

    if no_normal_count > 0:
        print(f"  -  Warning: zero normals for {no_normal_count}/{len(file_items)} images")
