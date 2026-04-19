"""Profile Sapiens body-part segmentation on a test image set.

Runs Sapiens seg models (0.3B / 0.6B / 1B) on every image in a directory,
saves per-image visualizations (combined class map, subject foreground mask,
per-part thumbnail grid), and records timing / coverage metrics.

Usage:
    python scripts/profile_sapiens_seg.py \\
        --data-dir test_data/scarlett_full \\
        --sizes 0.3b \\
        --dtype fp16

Outputs land in {data-dir}/seg_output/sapiens_{size}/ and include per-image
folders plus metrics.json and report.md at the root.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageOps import exif_transpose
from tqdm import tqdm


# ============================================================
# Class list (Goliath 28-class, index 0..27)
# ============================================================

GOLIATH_CLASSES: Tuple[str, ...] = (
    "Background",
    "Apparel",
    "Face_Neck",
    "Hair",
    "Left_Foot",
    "Left_Hand",
    "Left_Lower_Arm",
    "Left_Lower_Leg",
    "Left_Shoe",
    "Left_Sock",
    "Left_Upper_Arm",
    "Left_Upper_Leg",
    "Lower_Clothing",
    "Right_Foot",
    "Right_Hand",
    "Right_Lower_Arm",
    "Right_Lower_Leg",
    "Right_Shoe",
    "Right_Sock",
    "Right_Upper_Arm",
    "Right_Upper_Leg",
    "Torso",
    "Upper_Clothing",
    "Lower_Lip",
    "Upper_Lip",
    "Lower_Teeth",
    "Upper_Teeth",
    "Tongue",
)


def _build_palette(n: int) -> np.ndarray:
    """Deterministic bright palette for n classes. Background = black."""
    rng = np.random.RandomState(42)
    palette = np.zeros((n, 3), dtype=np.uint8)
    for i in range(1, n):
        # Use HLS-style bright colors
        h = (i * 37) % 360 / 360.0
        s = 0.8
        l = 0.55
        # HLS -> RGB
        c = (1 - abs(2 * l - 1)) * s
        x = c * (1 - abs((h * 6) % 2 - 1))
        m = l - c / 2
        if h < 1 / 6:
            r, g, b = c, x, 0
        elif h < 2 / 6:
            r, g, b = x, c, 0
        elif h < 3 / 6:
            r, g, b = 0, c, x
        elif h < 4 / 6:
            r, g, b = 0, x, c
        elif h < 5 / 6:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        palette[i] = [
            int((r + m) * 255),
            int((g + m) * 255),
            int((b + m) * 255),
        ]
    return palette


PALETTE = _build_palette(len(GOLIATH_CLASSES))


# ============================================================
# Sapiens ViT backbone + seg decoder (pure PyTorch)
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
    def __init__(self, dim, num_heads):
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
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = _FFN(dim, ffn_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class _SegDecoder(nn.Module):
    def __init__(self, in_channels=1024, mid_channels=768, num_classes=28):
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
        self.conv_seg = nn.Conv2d(mid_channels, num_classes, kernel_size=1)

    def forward(self, x):
        for i in range(3):
            x = self.deconv_layers[i * 3:(i + 1) * 3](x)
            x = self.conv_layers[i * 3:(i + 1) * 3](x)
        return self.conv_seg(x)


@dataclass
class SapiensSizeSpec:
    name: str
    repo_id: str
    filename: str
    embed_dim: int
    num_layers: int
    num_heads: int
    ffn_dim: int


SIZE_SPECS: Dict[str, SapiensSizeSpec] = {
    "0.3b": SapiensSizeSpec(
        name="0.3b",
        repo_id="facebook/sapiens-seg-0.3b",
        filename="sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194.pth",
        embed_dim=1024, num_layers=24, num_heads=16, ffn_dim=4096,
    ),
    "0.6b": SapiensSizeSpec(
        name="0.6b",
        repo_id="facebook/sapiens-seg-0.6b",
        filename="sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178.pth",
        embed_dim=1280, num_layers=32, num_heads=16, ffn_dim=5120,
    ),
    "1b": SapiensSizeSpec(
        name="1b",
        repo_id="facebook/sapiens-seg-1b",
        filename="sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151.pth",
        embed_dim=1536, num_layers=40, num_heads=24, ffn_dim=6144,
    ),
}


class SapiensSeg(nn.Module):
    """Sapiens body-part segmentation (pure PyTorch)."""

    NATIVE_H = 1024
    NATIVE_W = 768

    def __init__(self, spec: SapiensSizeSpec, num_classes: int = 28):
        super().__init__()
        self.spec = spec
        self.num_classes = num_classes
        self.embed_dim = spec.embed_dim

        self.patch_embed = _PatchEmbed(3, spec.embed_dim, patch_size=16, padding=2)
        self.pos_embed = nn.Parameter(torch.zeros(1, 64 * 48, spec.embed_dim))
        self.layers = nn.ModuleList([
            _TransformerBlock(spec.embed_dim, spec.num_heads, spec.ffn_dim)
            for _ in range(spec.num_layers)
        ])
        self.ln1 = nn.LayerNorm(spec.embed_dim)
        self.decode_head = _SegDecoder(spec.embed_dim, 768, num_classes)

        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def load_pretrained(self):
        ckpt_path = hf_hub_download(repo_id=self.spec.repo_id, filename=self.spec.filename)
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = raw.get("state_dict", raw)

        new_sd = {}
        for k, v in sd.items():
            nk = k
            if nk.startswith("backbone."):
                nk = nk[len("backbone."):]
            elif nk.startswith("decode_head."):
                pass
            else:
                continue
            nk = nk.replace("ffn.layers.0.0.", "ffn.fc1.")
            nk = nk.replace("ffn.layers.1.", "ffn.fc2.")
            new_sd[nk] = v

        result = self.load_state_dict(new_sd, strict=False)
        unexpected_missing = [k for k in result.missing_keys if k not in ("img_mean", "img_std")]
        if unexpected_missing:
            print(f"  [warn] missing: {unexpected_missing[:5]}  (+{len(unexpected_missing)-5} more)")
        if result.unexpected_keys:
            print(f"  [warn] unexpected: {result.unexpected_keys[:5]}")

    def forward(self, x):
        x = (x - self.img_mean.to(x.dtype)) / self.img_std.to(x.dtype)
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        if x.shape[1] != self.pos_embed.shape[1]:
            pe = self.pos_embed.reshape(1, 64, 48, self.embed_dim).permute(0, 3, 1, 2)
            pe = F.interpolate(pe.float(), size=(H, W), mode="bilinear", align_corners=False)
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
# Preprocessing / postprocessing
# ============================================================


def best_orientation(w: int, h: int) -> Tuple[int, int]:
    """Return (target_h, target_w) matching native Sapiens shapes."""
    if h >= w:
        return (1024, 768)
    return (768, 1024)


def letterbox(pil: Image.Image, target_w: int, target_h: int) -> Tuple[Image.Image, Dict]:
    w, h = pil.size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = pil.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    meta = dict(scale=scale, paste_x=paste_x, paste_y=paste_y, new_w=new_w, new_h=new_h,
                target_w=target_w, target_h=target_h, orig_w=w, orig_h=h)
    return canvas, meta


def unletterbox_mask(class_map: np.ndarray, meta: Dict) -> np.ndarray:
    """Crop letterbox padding and resize back to original image."""
    y0, x0 = meta["paste_y"], meta["paste_x"]
    y1, x1 = y0 + meta["new_h"], x0 + meta["new_w"]
    cropped = class_map[y0:y1, x0:x1]
    out = np.array(Image.fromarray(cropped.astype(np.uint8)).resize(
        (meta["orig_w"], meta["orig_h"]), Image.NEAREST))
    return out


# ============================================================
# Visualization
# ============================================================


def colormap_from_classes(class_map: np.ndarray) -> np.ndarray:
    return PALETTE[class_map]


def overlay(image_rgb: np.ndarray, color_mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    # Only blend where mask is non-background (non-black)
    nonbg = (color_mask.sum(axis=-1) > 0)[..., None]
    out = image_rgb.astype(np.float32).copy()
    blended = image_rgb.astype(np.float32) * (1 - alpha) + color_mask.astype(np.float32) * alpha
    out = np.where(nonbg, blended, out)
    return np.clip(out, 0, 255).astype(np.uint8)


def binary_overlay(image_rgb: np.ndarray, mask: np.ndarray, color=(255, 64, 64), alpha=0.6) -> np.ndarray:
    """Overlay a single-class binary mask in solid color."""
    out = image_rgb.astype(np.float32).copy()
    color_layer = np.array(color, dtype=np.float32)
    m = mask[..., None].astype(np.float32)
    out = out * (1 - alpha * m) + color_layer * alpha * m
    return np.clip(out, 0, 255).astype(np.uint8)


def make_parts_grid(image_rgb: np.ndarray, class_map: np.ndarray, classes_present: List[int],
                    thumb_w: int = 256) -> Image.Image:
    """Grid: one thumbnail per present class showing that class's pixels overlaid."""
    h, w = image_rgb.shape[:2]
    thumb_h = int(thumb_w * h / w)
    if not classes_present:
        return Image.fromarray(image_rgb).resize((thumb_w, thumb_h))

    cols = min(4, len(classes_present))
    rows = (len(classes_present) + cols - 1) // cols
    label_h = 22
    canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    for i, cls in enumerate(classes_present):
        r, c = i // cols, i % cols
        mask = (class_map == cls).astype(np.uint8)
        color = tuple(int(v) for v in PALETTE[cls])
        thumb = binary_overlay(image_rgb, mask, color=color, alpha=0.65)
        thumb_pil = Image.fromarray(thumb).resize((thumb_w, thumb_h), Image.BILINEAR)
        x = c * thumb_w
        y = r * (thumb_h + label_h)
        canvas.paste(thumb_pil, (x, y))
        pct = 100.0 * mask.mean()
        label = f"{GOLIATH_CLASSES[cls]} ({pct:.1f}%)"
        draw.rectangle([x, y + thumb_h, x + thumb_w, y + thumb_h + label_h], fill=(30, 30, 30))
        draw.text((x + 4, y + thumb_h + 3), label, fill=(230, 230, 230), font=font)
    return canvas


def side_by_side(left: np.ndarray, right: np.ndarray) -> Image.Image:
    h = max(left.shape[0], right.shape[0])
    out = Image.new("RGB", (left.shape[1] + right.shape[1] + 8, h), (20, 20, 20))
    out.paste(Image.fromarray(left), (0, 0))
    out.paste(Image.fromarray(right), (left.shape[1] + 8, 0))
    return out


# ============================================================
# Profiling
# ============================================================


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_forward(model: SapiensSeg, tensor: torch.Tensor, upsample_to: Tuple[int, int]) -> Tuple[np.ndarray, float]:
    """Run model forward, return (class_map HxW uint8, forward_ms)."""
    cuda_sync()
    t0 = time.perf_counter()
    with torch.inference_mode():
        logits = model(tensor)  # (1, C, H/2, W/2)
        logits = F.interpolate(logits.float(), size=upsample_to, mode="bilinear", align_corners=False)
        class_map = logits.argmax(dim=1).squeeze(0).to(torch.uint8).cpu().numpy()
    cuda_sync()
    t1 = time.perf_counter()
    return class_map, (t1 - t0) * 1000.0


def profile_size(spec: SapiensSizeSpec, images: List[Path], out_dir: Path, dtype: torch.dtype,
                 device: str = "cuda", warmup: int = 3, max_images: Optional[int] = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_image").mkdir(exist_ok=True)

    print(f"\n=== Sapiens Seg {spec.name} (dtype={dtype}) ===")
    print("Loading weights...")
    model = SapiensSeg(spec)
    model.load_pretrained()
    model.eval().to(device=device, dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params/1e6:.1f}M")

    if max_images is not None:
        images = images[:max_images]

    # Warmup
    print(f"Warmup ({warmup}x)...")
    dummy = torch.zeros(1, 3, 1024, 768, device=device, dtype=dtype)
    for _ in range(warmup):
        with torch.inference_mode():
            _ = model(dummy)
    cuda_sync()

    per_image_records = []
    total_forward_ms = 0.0
    total_full_ms = 0.0

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    for img_path in tqdm(images, desc=f"seg-{spec.name}"):
        t_start = time.perf_counter()
        pil = exif_transpose(Image.open(img_path)).convert("RGB")
        orig_w, orig_h = pil.size

        target_h, target_w = best_orientation(orig_w, orig_h)
        boxed, meta = letterbox(pil, target_w, target_h)
        tensor = torch.from_numpy(np.array(boxed)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(device=device, dtype=dtype)

        class_map_letterboxed, forward_ms = run_forward(model, tensor, upsample_to=(target_h, target_w))

        # Back to original image space for visualization
        class_map_orig = unletterbox_mask(class_map_letterboxed, meta)

        # Per-class coverage
        image_np = np.array(pil)
        coverage = {}
        classes_present = []
        total_px = class_map_orig.size
        for cls in range(len(GOLIATH_CLASSES)):
            c = int((class_map_orig == cls).sum())
            if c > 0:
                coverage[GOLIATH_CLASSES[cls]] = {"pixels": c, "pct": round(100.0 * c / total_px, 3)}
                if cls != 0 and c / total_px > 0.001:  # ignore background + noise for grid
                    classes_present.append(cls)

        # Save visualizations
        img_stem = Path(img_path).stem
        img_out = out_dir / "per_image" / img_stem
        img_out.mkdir(exist_ok=True)

        color_full = colormap_from_classes(class_map_orig)
        overlay_img = overlay(image_np, color_full, alpha=0.55)
        Image.fromarray(color_full).save(img_out / "class_map.png")
        Image.fromarray(overlay_img).save(img_out / "overlay.png")

        # Subject = not-background
        subj = (class_map_orig != 0).astype(np.uint8)
        subj_pct = float(subj.mean()) * 100
        subj_overlay = binary_overlay(image_np, subj, color=(64, 200, 255), alpha=0.5)
        Image.fromarray(subj_overlay).save(img_out / "subject_mask_overlay.png")
        Image.fromarray((subj * 255).astype(np.uint8)).save(img_out / "subject_mask.png")

        # Side-by-side summary
        side_by_side(image_np, overlay_img).save(img_out / "side_by_side.png")

        # Parts grid (only classes > 0.1% area)
        parts_grid = make_parts_grid(image_np, class_map_orig, classes_present, thumb_w=240)
        parts_grid.save(img_out / "parts_grid.png")

        cuda_sync()
        t_end = time.perf_counter()
        full_ms = (t_end - t_start) * 1000.0

        total_forward_ms += forward_ms
        total_full_ms += full_ms

        per_image_records.append({
            "path": str(img_path.relative_to(img_path.parents[1])),
            "orig_wh": [orig_w, orig_h],
            "target_wh": [target_w, target_h],
            "forward_ms": round(forward_ms, 2),
            "full_pipeline_ms": round(full_ms, 2),
            "subject_pct": round(subj_pct, 2),
            "classes_present": [GOLIATH_CLASSES[c] for c in classes_present],
            "coverage": coverage,
        })

    peak_mem_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0

    # Aggregate timing (skip first image as additional warmup tolerance)
    forward_times = np.array([r["forward_ms"] for r in per_image_records[1:]])
    full_times = np.array([r["full_pipeline_ms"] for r in per_image_records[1:]])

    def stats(a):
        return dict(mean=round(float(a.mean()), 2), median=round(float(np.median(a)), 2),
                    p95=round(float(np.percentile(a, 95)), 2), std=round(float(a.std()), 2),
                    min=round(float(a.min()), 2), max=round(float(a.max()), 2))

    summary = {
        "size": spec.name,
        "dtype": str(dtype).replace("torch.", ""),
        "num_params_M": round(n_params / 1e6, 1),
        "peak_gpu_mem_MB": round(peak_mem_mb, 1),
        "n_images": len(per_image_records),
        "forward_ms": stats(forward_times) if len(forward_times) else None,
        "full_pipeline_ms": stats(full_times) if len(full_times) else None,
        "fps_forward": round(1000.0 / forward_times.mean(), 2) if len(forward_times) else None,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"summary": summary, "per_image": per_image_records}, f, indent=2)

    print(f"  forward: mean={summary['forward_ms']['mean']}ms  p95={summary['forward_ms']['p95']}ms  "
          f"fps={summary['fps_forward']}  peak={peak_mem_mb:.0f}MB")

    del model
    torch.cuda.empty_cache()

    return summary, per_image_records


# ============================================================
# Report
# ============================================================


def write_report(root_out: Path, runs: List[Tuple[SapiensSizeSpec, Dict, List[Dict]]]):
    lines = []
    lines.append("# Sapiens Seg Profiling Report\n")
    lines.append(f"Test set: `{root_out.parent.name}` ({runs[0][1]['n_images']} images)\n")
    lines.append("Hardware: " + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU") + "\n")

    # Timing table
    lines.append("## Timing\n")
    lines.append("| Size | Params (M) | dtype | Peak mem (MB) | Forward mean (ms) | p95 (ms) | FPS |")
    lines.append("|------|-----------:|-------|---------------:|-------------------:|---------:|----:|")
    for spec, s, _ in runs:
        fm = s["forward_ms"]
        lines.append(f"| {s['size']} | {s['num_params_M']} | {s['dtype']} | {s['peak_gpu_mem_MB']} | "
                     f"{fm['mean']} | {fm['p95']} | {s['fps_forward']} |")
    lines.append("")

    # Per-class coverage (aggregated over all images, first run only for brevity)
    lines.append("## Per-class coverage (first size, mean % pixels across images)\n")
    _, _, per_image = runs[0]
    class_totals = {c: [] for c in GOLIATH_CLASSES}
    for rec in per_image:
        for cls, info in rec["coverage"].items():
            class_totals[cls].append(info["pct"])

    lines.append("| Class | Mean % | Images present |")
    lines.append("|-------|-------:|---------------:|")
    for cls in GOLIATH_CLASSES:
        vals = class_totals[cls]
        if not vals:
            continue
        lines.append(f"| {cls} | {np.mean(vals):.2f} | {len(vals)}/{len(per_image)} |")
    lines.append("")

    # Per-image table (first size)
    lines.append("## Per-image subject coverage & latency (first size)\n")
    lines.append("| Image | Orig WxH | Subject % | Forward ms | Full ms |")
    lines.append("|-------|----------|----------:|-----------:|--------:|")
    for rec in per_image:
        p = Path(rec["path"]).name
        lines.append(f"| `{p}` | {rec['orig_wh'][0]}x{rec['orig_wh'][1]} | "
                     f"{rec['subject_pct']} | {rec['forward_ms']} | {rec['full_pipeline_ms']} |")

    (root_out / "report.md").write_text("\n".join(lines))


# ============================================================
# Main
# ============================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="Directory of test images (also parent of output folder)")
    ap.add_argument("--sizes", default="0.3b", help="Comma-separated: 0.3b,0.6b,1b")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--out", default="seg_output", help="Output subfolder name under data-dir")
    args = ap.parse_args()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Find images
    exts = (".jpg", ".jpeg", ".png", ".webp", ".avif")
    images = sorted([p for p in args.data_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in exts])
    if not images:
        raise SystemExit(f"No images found in {args.data_dir}")
    print(f"Found {len(images)} images in {args.data_dir}")

    root_out = args.data_dir / args.out
    root_out.mkdir(exist_ok=True)

    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    runs = []
    for size in sizes:
        if size not in SIZE_SPECS:
            raise SystemExit(f"Unknown size '{size}'. Options: {list(SIZE_SPECS)}")
        spec = SIZE_SPECS[size]
        summary, per_image = profile_size(
            spec, images, root_out / f"sapiens_{size}", dtype,
            max_images=args.max_images,
        )
        runs.append((spec, summary, per_image))

    write_report(root_out, runs)
    print(f"\nReport written to {root_out / 'report.md'}")


if __name__ == "__main__":
    main()
