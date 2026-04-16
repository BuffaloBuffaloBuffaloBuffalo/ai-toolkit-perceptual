#!/usr/bin/env python3
"""Test Sapiens 0.3B normal estimation: inference, differentiability, and visualization."""

import sys
import math
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from PIL.ImageOps import exif_transpose

# ============================================================
# Sapiens 0.3B Normal Model — pure PyTorch, no mmseg dependency
# ============================================================

class SapiensPatchEmbed(nn.Module):
    def __init__(self, in_channels=3, embed_dim=1024, patch_size=16, padding=2):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=padding
        )

    def forward(self, x):
        # (B, 3, H, W) -> (B, embed_dim, H', W')
        return self.projection(x)


class SapiensAttention(nn.Module):
    def __init__(self, dim=1024, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)
        # Use scaled_dot_product_attention for efficiency
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class SapiensFFN(nn.Module):
    def __init__(self, dim=1024, hidden_dim=4096):
        super().__init__()
        # mmcv FFN stores as layers.0.0 (Linear) and layers.1 (Linear) with GELU between
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class SapiensTransformerBlock(nn.Module):
    def __init__(self, dim=1024, num_heads=16, ffn_dim=4096):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = SapiensAttention(dim, num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = SapiensFFN(dim, ffn_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class SapiensNormalDecoder(nn.Module):
    """Decode head: 3x upsample (ConvTranspose2d + InstanceNorm2d + SiLU) + 3x Conv1x1 + final Conv."""
    def __init__(self, in_channels=1024, mid_channels=768):
        super().__init__()
        # 3 deconv blocks: 1024->768, 768->768, 768->768, each 2x upsample
        # Interleaved with 1x1 conv blocks at 768 channels
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
        # x: (B, C, H, W) from backbone
        # Apply deconv and conv blocks interleaved
        for i in range(3):
            x = self.deconv_layers[i * 3:(i + 1) * 3](x)
            x = self.conv_layers[i * 3:(i + 1) * 3](x)
        return self.conv_seg(x)


class SapiensNormal(nn.Module):
    """Sapiens 0.3B normal estimation model — pure PyTorch."""

    INPUT_H = 1024
    INPUT_W = 768

    def __init__(self, embed_dim=1024, num_layers=24, num_heads=16, ffn_dim=4096):
        super().__init__()
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = SapiensPatchEmbed(3, embed_dim, patch_size=16, padding=2)

        # Positional embedding (no cls token)
        # With padding=2 on 1024x768 input: output is (1024+4)//16 x (768+4)//16 = 64x48 = 3072
        self.num_patches = 64 * 48  # 3072
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer blocks
        self.layers = nn.ModuleList([
            SapiensTransformerBlock(embed_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.ln1 = nn.LayerNorm(embed_dim)

        # Decode head
        self.decode_head = SapiensNormalDecoder(embed_dim)

        # ImageNet normalization
        self.register_buffer('img_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('img_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _load_pretrained(self, ckpt_path):
        """Load Sapiens .pth checkpoint with key remapping."""
        raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = raw['state_dict'] if 'state_dict' in raw else raw

        new_sd = {}
        for k, v in sd.items():
            nk = k
            # Strip backbone. prefix
            if nk.startswith('backbone.'):
                nk = nk[len('backbone.'):]
            # Strip decode_head. prefix (we keep it under self.decode_head)
            elif nk.startswith('decode_head.'):
                pass  # keep as-is
            else:
                continue

            # Remap mmcv FFN keys: ffn.layers.0.0.* -> ffn.fc1.*, ffn.layers.1.* -> ffn.fc2.*
            nk = nk.replace('ffn.layers.0.0.', 'ffn.fc1.')
            nk = nk.replace('ffn.layers.1.', 'ffn.fc2.')

            new_sd[nk] = v

        result = self.load_state_dict(new_sd, strict=False)
        # img_mean/img_std are expected missing (registered buffers)
        unexpected_missing = [k for k in result.missing_keys
                              if k not in ('img_mean', 'img_std')]
        if unexpected_missing:
            print(f"  Warning: missing keys: {unexpected_missing}")
        if result.unexpected_keys:
            print(f"  Warning: unexpected keys: {result.unexpected_keys}")
        print(f"  Loaded {len(new_sd)} keys successfully")

    def forward(self, x):
        """Forward pass. Input: (B, 3, H, W) in [0, 1] RGB. Output: (B, 3, H', W') raw normals."""
        # Normalize
        x = (x - self.img_mean.to(x.dtype)) / self.img_std.to(x.dtype)

        # Patch embed -> (B, C, H', W')
        x = self.patch_embed(x)
        B, C, H, W = x.shape

        # Flatten to sequence: (B, N, C)
        x = x.flatten(2).transpose(1, 2)

        # Add positional embedding (interpolate if input size differs from training)
        if x.shape[1] != self.pos_embed.shape[1]:
            # Reshape pos_embed to spatial, interpolate, flatten back
            pe = self.pos_embed.reshape(1, 64, 48, self.embed_dim).permute(0, 3, 1, 2)
            pe = F.interpolate(pe.float(), size=(H, W), mode='bilinear', align_corners=False)
            pe = pe.to(x.dtype).permute(0, 2, 3, 1).reshape(1, H * W, self.embed_dim)
            x = x + pe
        else:
            x = x + self.pos_embed

        # Transformer blocks
        for layer in self.layers:
            x = layer(x)

        # Final norm
        x = self.ln1(x)

        # Reshape back to spatial: (B, C, H, W)
        x = x.transpose(1, 2).reshape(B, C, H, W)

        # Decode head
        x = self.decode_head(x)
        return x

    @torch.no_grad()
    def predict(self, pil_image):
        """Run inference on a PIL image. Returns (3, H, W) unit normal map on CPU."""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Resize to model input size
        img = pil_image.resize((self.INPUT_W, self.INPUT_H), Image.BILINEAR)
        tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(device=device, dtype=dtype)

        raw = self.forward(tensor)  # (1, 3, 512, 384)

        # Upsample to input resolution
        raw = F.interpolate(raw.float(), size=(self.INPUT_H, self.INPUT_W), mode='bilinear', align_corners=False)

        # Normalize to unit vectors
        normals = raw / (raw.norm(dim=1, keepdim=True) + 1e-5)
        return normals.squeeze(0).cpu()  # (3, H, W)


def normal_to_rgb(normal_map):
    """Convert (3, H, W) unit normal in [-1, 1] to (H, W, 3) uint8 RGB."""
    rgb = ((normal_map.permute(1, 2, 0).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return rgb


# ============================================================
# Main test script
# ============================================================

def main():
    from huggingface_hub import hf_hub_download

    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.avif', '.bmp', '.tiff'}

    bodytest_dir = Path('bodytest')
    output_dir = bodytest_dir / 'normals'
    output_dir.mkdir(exist_ok=True)

    # Collect images
    images = sorted([
        f for f in bodytest_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTS and not f.name.startswith('.')
    ])
    print(f"Found {len(images)} images in {bodytest_dir}")

    # Build model using the training wrapper (with letterboxing + orientation)
    print("Building model...")
    from toolkit.normal_id import DifferentiableNormalEncoder, NORMAL_SIZE, _best_orientation
    encoder = DifferentiableNormalEncoder()
    encoder.half().to('cuda')

    # === Inference on bodytest images ===
    print(f"\n=== Inference on {len(images)} images (auto portrait/landscape) ===")
    for img_path in images:
        try:
            pil = exif_transpose(Image.open(img_path)).convert('RGB')
        except Exception as e:
            print(f"  Skip {img_path.name}: {e}")
            continue

        w, h = pil.size
        target_h, target_w = _best_orientation(w, h)
        orient = "portrait" if target_h > target_w else "landscape"

        normal_map = encoder.encode(pil)  # (3, 256, 256) unit normals

        # Upscale for visualization to the orientation used
        normal_vis = F.interpolate(
            normal_map.unsqueeze(0), size=(target_h, target_w),
            mode='bilinear', align_corners=False
        ).squeeze(0)
        rgb = normal_to_rgb(normal_vis)
        out_path = output_dir / f"{img_path.stem}_normal.png"
        Image.fromarray(rgb).save(str(out_path))

        # Side-by-side: letterboxed original | normal map
        letterboxed = encoder._letterbox_pil(pil, target_w, target_h)
        normal_img = Image.fromarray(rgb)
        combined = Image.new('RGB', (target_w * 2 + 4, target_h))
        combined.paste(letterboxed, (0, 0))
        combined.paste(normal_img, (target_w + 4, 0))
        combined_path = output_dir / f"{img_path.stem}_compare.png"
        combined.save(str(combined_path))

        print(f"  {img_path.name:50s}  ({w}x{h} {orient}) -> {out_path.name}")

    print(f"\nDone. Normals saved to {output_dir}/")


if __name__ == '__main__':
    main()
