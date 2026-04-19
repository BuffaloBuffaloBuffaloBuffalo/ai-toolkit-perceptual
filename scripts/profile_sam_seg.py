"""Profile SAM 2 human-subject segmentation on a test image set.

Runs SAM 2 (tiny/small/base+/large) on every image in a directory,
saves per-image visualizations, records timing / coverage metrics.

Prompt strategy options:
  - center_point: single positive point at image center (fastest, works well on portraits)
  - inset_box:    bbox covering 90% of image (tells SAM "segment the thing inside")
  - grid:         3x3 grid of positive points inside a 70% inset

Uses HuggingFace transformers (no Meta sam2 dependency).

Usage:
    python scripts/profile_sam_seg.py \\
        --data-dir test_data/scarlett_full \\
        --sizes small \\
        --prompt center_point \\
        --dtype fp16
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageOps import exif_transpose
from tqdm import tqdm
from transformers import Sam2Model, Sam2Processor


# ============================================================
# Model catalog
# ============================================================


@dataclass
class Sam2Spec:
    name: str
    hf_id: str


SIZE_SPECS: Dict[str, Sam2Spec] = {
    "tiny":      Sam2Spec("tiny",      "facebook/sam2.1-hiera-tiny"),
    "small":     Sam2Spec("small",     "facebook/sam2.1-hiera-small"),
    "base_plus": Sam2Spec("base_plus", "facebook/sam2.1-hiera-base-plus"),
    "large":     Sam2Spec("large",     "facebook/sam2.1-hiera-large"),
}


# ============================================================
# Prompt strategies
# ============================================================


def center_point_prompt(w: int, h: int) -> Tuple[List[List[List[List[float]]]], List[List[List[int]]]]:
    """One positive point at (w/2, h/2)."""
    # transformers expects: input_points = List[List[List[List[float]]]]
    # shape: [batch, n_objects, n_points, 2]
    points = [[[[float(w) / 2, float(h) / 2]]]]
    labels = [[[1]]]
    return points, labels


def inset_box_prompt(w: int, h: int, inset: float = 0.05) -> List[List[List[List[float]]]]:
    """Bbox covering (1-2*inset) of the image."""
    x0 = int(w * inset)
    y0 = int(h * inset)
    x1 = int(w * (1 - inset))
    y1 = int(h * (1 - inset))
    return [[[[float(x0), float(y0), float(x1), float(y1)]]]]


def grid_points_prompt(w: int, h: int, grid: int = 3, inset: float = 0.15):
    """grid x grid positive points inside an inset box."""
    pts = []
    x0, y0 = int(w * inset), int(h * inset)
    x1, y1 = int(w * (1 - inset)), int(h * (1 - inset))
    for i in range(grid):
        for j in range(grid):
            x = x0 + (x1 - x0) * j / (grid - 1)
            y = y0 + (y1 - y0) * i / (grid - 1)
            pts.append([float(x), float(y)])
    points = [[pts]]
    labels = [[[1] * len(pts)]]
    return points, labels


# ============================================================
# Visualization
# ============================================================


def binary_overlay(image_rgb: np.ndarray, mask: np.ndarray, color=(64, 200, 255), alpha=0.55) -> np.ndarray:
    out = image_rgb.astype(np.float32).copy()
    m = mask[..., None].astype(np.float32)
    color_layer = np.array(color, dtype=np.float32)
    out = out * (1 - alpha * m) + color_layer * alpha * m
    # Draw outline
    try:
        from scipy.ndimage import binary_dilation
        outline = binary_dilation(mask, iterations=2) & (~mask.astype(bool))
        out[outline] = np.array([255, 255, 0])
    except Exception:
        pass
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_prompt(image_rgb: np.ndarray, points=None, box=None) -> np.ndarray:
    img = Image.fromarray(image_rgb).copy()
    draw = ImageDraw.Draw(img)
    if points is not None:
        for p in points:
            x, y = p
            r = 8
            draw.ellipse((x - r, y - r, x + r, y + r), outline=(0, 255, 0), width=3)
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 255, 0))
    if box is not None:
        x0, y0, x1, y1 = box
        draw.rectangle((x0, y0, x1, y1), outline=(255, 64, 64), width=3)
    return np.array(img)


def side_by_side(*frames: np.ndarray, labels=None, gap: int = 8) -> Image.Image:
    h = max(f.shape[0] for f in frames)
    w_total = sum(f.shape[1] for f in frames) + gap * (len(frames) - 1)
    out = Image.new("RGB", (w_total, h + (26 if labels else 0)), (20, 20, 20))
    x = 0
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for i, f in enumerate(frames):
        out.paste(Image.fromarray(f), (x, 26 if labels else 0))
        if labels:
            draw.text((x + 6, 4), labels[i], fill=(230, 230, 230), font=font)
        x += f.shape[1] + gap
    return out


# ============================================================
# Main profiling routine
# ============================================================


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def select_best_mask(masks: torch.Tensor, iou_scores: torch.Tensor,
                     img_hw: Tuple[int, int], min_frac: float = 0.01,
                     max_frac: float = 0.95) -> Tuple[np.ndarray, int, float]:
    """From SAM's multimask output, pick the mask with highest iou score
    whose area is within [min_frac, max_frac] of image.
    masks: (n_masks, H, W) bool or float
    iou_scores: (n_masks,)
    """
    masks_np = masks.detach().cpu().numpy() if torch.is_tensor(masks) else masks
    scores_np = iou_scores.detach().cpu().numpy() if torch.is_tensor(iou_scores) else iou_scores

    total = img_hw[0] * img_hw[1]
    candidates = []
    for i, (m, s) in enumerate(zip(masks_np, scores_np)):
        frac = float(m.mean())
        if min_frac <= frac <= max_frac:
            candidates.append((s, i, m, frac))

    if not candidates:
        # Fall back to highest iou regardless
        i = int(scores_np.argmax())
        return masks_np[i].astype(bool), i, float(scores_np[i])

    candidates.sort(key=lambda x: -x[0])
    s, i, m, frac = candidates[0]
    return m.astype(bool), i, float(s)


def profile_size(spec: Sam2Spec, images: List[Path], out_dir: Path, prompt_kind: str,
                 dtype: torch.dtype, device: str = "cuda", warmup: int = 3,
                 max_images: Optional[int] = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_image").mkdir(exist_ok=True)

    print(f"\n=== SAM 2 {spec.name} (prompt={prompt_kind}, dtype={dtype}) ===")
    print(f"Loading {spec.hf_id}...")

    processor = Sam2Processor.from_pretrained(spec.hf_id)
    model = Sam2Model.from_pretrained(spec.hf_id, torch_dtype=dtype).to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params/1e6:.1f}M")

    if max_images is not None:
        images = images[:max_images]

    # Warmup on first image
    warmup_img = exif_transpose(Image.open(images[0])).convert("RGB")
    for _ in range(warmup):
        _run_one(model, processor, warmup_img, prompt_kind, device, dtype)
    cuda_sync()

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    per_image_records = []
    for img_path in tqdm(images, desc=f"sam2-{spec.name}"):
        pil = exif_transpose(Image.open(img_path)).convert("RGB")
        orig_w, orig_h = pil.size
        image_np = np.array(pil)

        (mask, iou, mask_idx, encode_ms, decode_ms, full_ms, prompt_meta) = _run_one(
            model, processor, pil, prompt_kind, device, dtype
        )

        subj_pct = float(mask.mean()) * 100

        # Save visualizations
        img_stem = Path(img_path).stem
        img_out = out_dir / "per_image" / img_stem
        img_out.mkdir(exist_ok=True)

        overlay = binary_overlay(image_np, mask.astype(np.uint8), color=(64, 200, 255), alpha=0.55)
        Image.fromarray(overlay).save(img_out / "overlay.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(img_out / "subject_mask.png")

        # Draw prompt
        prompted = draw_prompt(
            image_np,
            points=prompt_meta.get("points"),
            box=prompt_meta.get("box"),
        )
        side_by_side(prompted, overlay, labels=[f"prompt ({prompt_kind})", f"mask ({subj_pct:.1f}%)"]).save(
            img_out / "side_by_side.png"
        )

        per_image_records.append({
            "path": str(img_path),
            "orig_wh": [orig_w, orig_h],
            "subject_pct": round(subj_pct, 2),
            "iou_score": round(float(iou), 3),
            "mask_idx": int(mask_idx),
            "encode_ms": round(encode_ms, 2),
            "decode_ms": round(decode_ms, 2),
            "full_ms": round(full_ms, 2),
        })

    peak_mem_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0

    enc_times = np.array([r["encode_ms"] for r in per_image_records[1:]])
    dec_times = np.array([r["decode_ms"] for r in per_image_records[1:]])
    full_times = np.array([r["full_ms"] for r in per_image_records[1:]])

    def stats(a):
        return dict(mean=round(float(a.mean()), 2), median=round(float(np.median(a)), 2),
                    p95=round(float(np.percentile(a, 95)), 2), std=round(float(a.std()), 2),
                    min=round(float(a.min()), 2), max=round(float(a.max()), 2))

    summary = {
        "size": spec.name,
        "hf_id": spec.hf_id,
        "prompt": prompt_kind,
        "dtype": str(dtype).replace("torch.", ""),
        "num_params_M": round(n_params / 1e6, 1),
        "peak_gpu_mem_MB": round(peak_mem_mb, 1),
        "n_images": len(per_image_records),
        "encode_ms": stats(enc_times) if len(enc_times) else None,
        "decode_ms": stats(dec_times) if len(dec_times) else None,
        "full_ms": stats(full_times) if len(full_times) else None,
        "fps_full": round(1000.0 / full_times.mean(), 2) if len(full_times) else None,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"summary": summary, "per_image": per_image_records}, f, indent=2)

    print(f"  encode: mean={summary['encode_ms']['mean']}ms  decode: {summary['decode_ms']['mean']}ms  "
          f"full: {summary['full_ms']['mean']}ms  fps={summary['fps_full']}  peak={peak_mem_mb:.0f}MB")

    del model, processor
    torch.cuda.empty_cache()
    return summary, per_image_records


def _run_one(model, processor, pil: Image.Image, prompt_kind: str, device: str, dtype: torch.dtype):
    w, h = pil.size
    prompt_meta = {}
    kwargs = {}

    if prompt_kind == "center_point":
        points, labels = center_point_prompt(w, h)
        kwargs["input_points"] = points
        kwargs["input_labels"] = labels
        prompt_meta["points"] = [(w / 2, h / 2)]
    elif prompt_kind == "inset_box":
        box = inset_box_prompt(w, h, inset=0.05)
        kwargs["input_boxes"] = box
        prompt_meta["box"] = box[0][0][0]  # [x0,y0,x1,y1]
    elif prompt_kind == "grid":
        points, labels = grid_points_prompt(w, h, grid=3, inset=0.15)
        kwargs["input_points"] = points
        kwargs["input_labels"] = labels
        prompt_meta["points"] = points[0][0]
    else:
        raise ValueError(prompt_kind)

    t0 = time.perf_counter()
    inputs = processor(images=pil, **kwargs, return_tensors="pt").to(device)
    # Move tensors to dtype where appropriate
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

    cuda_sync()
    t_enc0 = time.perf_counter()
    with torch.inference_mode():
        # Image encoder pass (expensive)
        image_embeddings = model.get_image_embeddings(inputs["pixel_values"])
        cuda_sync()
        t_enc1 = time.perf_counter()
        # Prompt-conditional mask decoding (cheap)
        dec_inputs = {k: v for k, v in inputs.items() if k != "pixel_values"}
        outputs = model(
            image_embeddings=image_embeddings,
            multimask_output=True,
            **dec_inputs,
        )
        cuda_sync()
    t_dec1 = time.perf_counter()

    # Post-process masks back to original image size
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        original_sizes=inputs["original_sizes"].cpu(),
        reshape_input_sizes=inputs["reshape_input_sizes"].cpu() if "reshape_input_sizes" in inputs else None,
    )[0]  # list of tensors, one per image; take first
    # masks shape: (n_objects=1, n_masks, H, W)
    masks = masks[0]  # (n_masks, H, W)
    iou_scores = outputs.iou_scores[0, 0]  # (n_masks,)

    best_mask, mask_idx, iou = select_best_mask(masks, iou_scores, (h, w))
    cuda_sync()
    t_total = time.perf_counter()

    encode_ms = (t_enc1 - t_enc0) * 1000.0
    decode_ms = (t_dec1 - t_enc1) * 1000.0
    full_ms = (t_total - t0) * 1000.0
    return best_mask, iou, mask_idx, encode_ms, decode_ms, full_ms, prompt_meta


# ============================================================
# Report
# ============================================================


def write_report(root_out: Path, runs):
    lines = []
    lines.append("# SAM 2 Subject Segmentation — Profiling Report\n")
    lines.append(f"Test set: `{root_out.parent.name}` ({runs[0][1]['n_images']} images)\n")
    lines.append("Hardware: " + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU") + "\n")

    lines.append("## Timing\n")
    lines.append("| Size | Params (M) | Prompt | dtype | Peak mem (MB) | Encode mean (ms) | Decode mean (ms) | Full mean (ms) | FPS full |")
    lines.append("|------|-----------:|--------|-------|---------------:|------------------:|------------------:|----------------:|---------:|")
    for spec, s, _ in runs:
        lines.append(f"| {s['size']} | {s['num_params_M']} | {s['prompt']} | {s['dtype']} | {s['peak_gpu_mem_MB']} | "
                     f"{s['encode_ms']['mean']} | {s['decode_ms']['mean']} | {s['full_ms']['mean']} | {s['fps_full']} |")
    lines.append("")

    lines.append("## Per-image subject coverage & latency (first run)\n")
    _, _, per_image = runs[0]
    lines.append("| Image | Orig WxH | Subject % | IoU score | mask_idx | Encode ms | Decode ms | Full ms |")
    lines.append("|-------|----------|----------:|----------:|---------:|----------:|----------:|--------:|")
    for rec in per_image:
        p = Path(rec["path"]).name
        lines.append(f"| `{p}` | {rec['orig_wh'][0]}x{rec['orig_wh'][1]} | {rec['subject_pct']} | "
                     f"{rec['iou_score']} | {rec['mask_idx']} | {rec['encode_ms']} | {rec['decode_ms']} | {rec['full_ms']} |")

    (root_out / "report.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--sizes", default="small", help="tiny,small,base_plus,large")
    ap.add_argument("--prompt", default="center_point", choices=["center_point", "inset_box", "grid"])
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--out", default="seg_output", help="Output subfolder under data-dir")
    args = ap.parse_args()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    exts = (".jpg", ".jpeg", ".png", ".webp", ".avif")
    images = sorted([p for p in args.data_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in exts])
    if not images:
        raise SystemExit(f"No images in {args.data_dir}")
    print(f"Found {len(images)} images")

    root_out = args.data_dir / args.out
    root_out.mkdir(exist_ok=True)

    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    runs = []
    for size in sizes:
        if size not in SIZE_SPECS:
            raise SystemExit(f"Unknown size {size}. options: {list(SIZE_SPECS)}")
        spec = SIZE_SPECS[size]
        summary, per_image = profile_size(
            spec, images,
            root_out / f"sam2_{size}_{args.prompt}",
            args.prompt, dtype,
            max_images=args.max_images,
        )
        runs.append((spec, summary, per_image))

    write_report(root_out, runs)
    print(f"\nReport: {root_out / 'report.md'}")


if __name__ == "__main__":
    main()
