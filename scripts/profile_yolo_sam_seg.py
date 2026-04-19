"""YOLO person detection -> SAM 2 bbox prompt -> subject mask.

Drives SAM 2 with bounding boxes produced by a YOLO person detector, so
prompts always land on actual people (instead of failing on off-center subjects
like center-point prompting does).

Pipeline per image:
  1. YOLO detects all COCO 'person' instances, returns bboxes + confidences.
  2. Filter by confidence threshold; keep N best boxes (default: all >= 0.25).
  3. Feed those boxes to SAM 2 as multi-object prompts, multimask_output=True.
  4. For each object, pick highest-IoU mask.
  5. Union all object masks into a single subject foreground mask.
  6. If YOLO finds nothing, fall back to center-point prompt and flag the image.

Usage:
    python scripts/profile_yolo_sam_seg.py \\
        --data-dir test_data/scarlett_full \\
        --sam-size small \\
        --yolo yolo11n.pt \\
        --conf 0.25 \\
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
from ultralytics import YOLO


# ============================================================
# Model catalog
# ============================================================


@dataclass
class Sam2Spec:
    name: str
    hf_id: str


SAM_SPECS: Dict[str, Sam2Spec] = {
    "tiny":      Sam2Spec("tiny",      "facebook/sam2.1-hiera-tiny"),
    "small":     Sam2Spec("small",     "facebook/sam2.1-hiera-small"),
    "base_plus": Sam2Spec("base_plus", "facebook/sam2.1-hiera-base-plus"),
    "large":     Sam2Spec("large",     "facebook/sam2.1-hiera-large"),
}


# ============================================================
# Visualization helpers
# ============================================================


def binary_overlay(image_rgb: np.ndarray, mask: np.ndarray, color=(64, 200, 255), alpha=0.55,
                   outline=True) -> np.ndarray:
    out = image_rgb.astype(np.float32).copy()
    m = mask[..., None].astype(np.float32)
    color_layer = np.array(color, dtype=np.float32)
    out = out * (1 - alpha * m) + color_layer * alpha * m
    if outline:
        try:
            from scipy.ndimage import binary_dilation
            border = binary_dilation(mask.astype(bool), iterations=2) & (~mask.astype(bool))
            out[border] = np.array([255, 255, 0])
        except Exception:
            pass
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_boxes(image_rgb: np.ndarray, boxes: List[Tuple[float, float, float, float]],
               confs: List[float]) -> np.ndarray:
    img = Image.fromarray(image_rgb).copy()
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for (x0, y0, x1, y1), c in zip(boxes, confs):
        d.rectangle((x0, y0, x1, y1), outline=(255, 64, 64), width=3)
        txt = f"person {c:.2f}"
        tw = int(d.textlength(txt, font=font))
        d.rectangle((x0, max(y0 - 18, 0), x0 + tw + 8, max(y0, 18)), fill=(255, 64, 64))
        d.text((x0 + 4, max(y0 - 18, 0) + 1), txt, fill=(255, 255, 255), font=font)
    return np.array(img)


def side_by_side(*frames, labels=None, gap: int = 8) -> Image.Image:
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
# YOLO person detection
# ============================================================


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def detect_persons(yolo: YOLO, pil: Image.Image, conf: float):
    """Run YOLO, return (boxes, confs, detect_ms) with COCO person class = 0."""
    img_np = np.array(pil)  # RGB HxWx3
    cuda_sync()
    t0 = time.perf_counter()
    results = yolo.predict(img_np, classes=[0], conf=conf, verbose=False, device=0)
    cuda_sync()
    t1 = time.perf_counter()
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return [], [], (t1 - t0) * 1000
    boxes = r.boxes.xyxy.cpu().numpy().tolist()
    confs = r.boxes.conf.cpu().numpy().tolist()
    # Sort by confidence descending
    order = np.argsort(confs)[::-1]
    boxes = [boxes[i] for i in order]
    confs = [confs[i] for i in order]
    return boxes, confs, (t1 - t0) * 1000


# ============================================================
# SAM 2 multi-box segmentation
# ============================================================


def sam_segment_boxes(model, processor, pil: Image.Image, boxes: List[List[float]],
                      device: str, dtype: torch.dtype):
    """Segment with SAM 2 using one or more bboxes. Returns (union_mask HxW bool,
    per_object_masks list, encode_ms, decode_ms)."""
    w, h = pil.size

    # HF SAM 2 expects 3-level nested: [image, box, coords] -> [[[x0,y0,x1,y1], ...]]
    input_boxes = [[list(b) for b in boxes]]

    t0 = time.perf_counter()
    inputs = processor(images=pil, input_boxes=input_boxes, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

    cuda_sync()
    t_enc0 = time.perf_counter()
    with torch.inference_mode():
        image_embeddings = model.get_image_embeddings(inputs["pixel_values"])
        cuda_sync()
        t_enc1 = time.perf_counter()
        dec_inputs = {k: v for k, v in inputs.items() if k != "pixel_values"}
        # multimask_output=False for bbox prompts: box is unambiguous, return single mask per object
        outputs = model(
            image_embeddings=image_embeddings,
            multimask_output=False,
            **dec_inputs,
        )
        cuda_sync()
    t_dec1 = time.perf_counter()

    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        original_sizes=inputs["original_sizes"].cpu(),
        reshape_input_sizes=inputs["reshape_input_sizes"].cpu() if "reshape_input_sizes" in inputs else None,
    )[0]  # first image; shape: (n_objects, n_masks=1, H, W)

    # multimask=False -> single mask per object at index 0
    per_object_masks = [masks[i, 0].bool().numpy() for i in range(masks.shape[0])]

    if per_object_masks:
        union = np.any(np.stack(per_object_masks, axis=0), axis=0)
    else:
        union = np.zeros((h, w), dtype=bool)

    encode_ms = (t_enc1 - t_enc0) * 1000
    decode_ms = (t_dec1 - t_enc1) * 1000
    return union, per_object_masks, encode_ms, decode_ms


def sam_segment_center_point(model, processor, pil: Image.Image,
                             device: str, dtype: torch.dtype):
    """Fallback: segment with single center point."""
    w, h = pil.size
    input_points = [[[[float(w) / 2, float(h) / 2]]]]
    input_labels = [[[1]]]
    inputs = processor(images=pil, input_points=input_points, input_labels=input_labels,
                       return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    cuda_sync()
    t_enc0 = time.perf_counter()
    with torch.inference_mode():
        image_embeddings = model.get_image_embeddings(inputs["pixel_values"])
        cuda_sync()
        t_enc1 = time.perf_counter()
        dec_inputs = {k: v for k, v in inputs.items() if k != "pixel_values"}
        outputs = model(image_embeddings=image_embeddings, multimask_output=True, **dec_inputs)
        cuda_sync()
    t_dec1 = time.perf_counter()
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        original_sizes=inputs["original_sizes"].cpu(),
        reshape_input_sizes=inputs["reshape_input_sizes"].cpu() if "reshape_input_sizes" in inputs else None,
    )[0]
    iou_scores = outputs.iou_scores[0, 0].cpu()
    best_k = int(iou_scores.argmax())
    return masks[0, best_k].bool().numpy(), \
           (t_enc1 - t_enc0) * 1000, (t_dec1 - t_enc1) * 1000


# ============================================================
# Main pipeline
# ============================================================


def run(yolo_ckpt: str, sam_size: str, images: List[Path], out_dir: Path,
        conf: float, dtype: torch.dtype, device: str = "cuda",
        warmup: int = 3, max_images: Optional[int] = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_image").mkdir(exist_ok=True)

    print(f"\n=== YOLO({yolo_ckpt}) + SAM2-{sam_size} (dtype={dtype}) ===")
    print("Loading YOLO...")
    yolo = YOLO(yolo_ckpt)
    # Warm YOLO on a dummy image
    _ = yolo.predict(np.zeros((640, 480, 3), dtype=np.uint8), verbose=False, device=0)

    spec = SAM_SPECS[sam_size]
    print(f"Loading SAM 2 {spec.hf_id}...")
    processor = Sam2Processor.from_pretrained(spec.hf_id)
    sam = Sam2Model.from_pretrained(spec.hf_id, torch_dtype=dtype).to(device).eval()
    n_params = sum(p.numel() for p in sam.parameters())
    print(f"  sam params: {n_params/1e6:.1f}M")

    if max_images is not None:
        images = images[:max_images]

    # Warmup sam
    warm = exif_transpose(Image.open(images[0])).convert("RGB")
    for _ in range(warmup):
        sam_segment_center_point(sam, processor, warm, device, dtype)
    cuda_sync()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    records = []
    n_detected, n_fallback = 0, 0
    for img_path in tqdm(images, desc=f"yolo+sam2-{sam_size}"):
        pil = exif_transpose(Image.open(img_path)).convert("RGB")
        orig_w, orig_h = pil.size
        image_np = np.array(pil)
        img_stem = Path(img_path).stem
        img_out = out_dir / "per_image" / img_stem
        img_out.mkdir(exist_ok=True)

        t_start = time.perf_counter()

        # 1. YOLO person detection
        boxes, confs, yolo_ms = detect_persons(yolo, pil, conf)

        if boxes:
            n_detected += 1
            union, per_obj, enc_ms, dec_ms = sam_segment_boxes(
                sam, processor, pil, boxes, device, dtype)
            source = "yolo_box"
        else:
            # Fallback center point
            n_fallback += 1
            union, enc_ms, dec_ms = sam_segment_center_point(
                sam, processor, pil, device, dtype)
            per_obj = [union]
            source = "center_fallback"

        cuda_sync()
        t_end = time.perf_counter()
        full_ms = (t_end - t_start) * 1000

        subj_pct = float(union.mean()) * 100

        # Save visualizations
        Image.fromarray((union.astype(np.uint8) * 255)).save(img_out / "subject_mask.png")
        overlay = binary_overlay(image_np, union.astype(np.uint8), (64, 200, 255), 0.55)
        Image.fromarray(overlay).save(img_out / "overlay.png")

        prompted = draw_boxes(image_np, boxes, confs) if boxes else image_np.copy()
        side_by_side(prompted, overlay,
                     labels=[f"YOLO n={len(boxes)} ({source})",
                             f"SAM mask ({subj_pct:.1f}%)"]).save(img_out / "side_by_side.png")

        records.append({
            "path": str(img_path),
            "stem": img_stem,
            "orig_wh": [orig_w, orig_h],
            "source": source,
            "n_boxes": len(boxes),
            "max_conf": round(float(max(confs)), 3) if confs else None,
            "subject_pct": round(subj_pct, 2),
            "yolo_ms": round(yolo_ms, 2),
            "encode_ms": round(enc_ms, 2),
            "decode_ms": round(dec_ms, 2),
            "full_ms": round(full_ms, 2),
        })

    peak_mem_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0

    def stats(arr):
        a = np.array(arr)
        return dict(mean=round(float(a.mean()), 2), median=round(float(np.median(a)), 2),
                    p95=round(float(np.percentile(a, 95)), 2),
                    min=round(float(a.min()), 2), max=round(float(a.max()), 2))

    summary = {
        "yolo_ckpt": yolo_ckpt,
        "sam_size": sam_size,
        "sam_hf_id": spec.hf_id,
        "dtype": str(dtype).replace("torch.", ""),
        "conf_threshold": conf,
        "n_images": len(records),
        "n_detected": n_detected,
        "n_fallback": n_fallback,
        "detect_rate": round(n_detected / len(records), 3),
        "peak_gpu_mem_MB": round(peak_mem_mb, 1),
        "yolo_ms": stats([r["yolo_ms"] for r in records[1:]]),
        "encode_ms": stats([r["encode_ms"] for r in records[1:]]),
        "decode_ms": stats([r["decode_ms"] for r in records[1:]]),
        "full_ms": stats([r["full_ms"] for r in records[1:]]),
        "fps_full": round(1000.0 / np.mean([r["full_ms"] for r in records[1:]]), 2),
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"summary": summary, "per_image": records}, f, indent=2)

    print(f"\nDetect rate: {n_detected}/{len(records)} ({100*n_detected/len(records):.1f}%)")
    print(f"Timing:  yolo={summary['yolo_ms']['mean']}ms  encode={summary['encode_ms']['mean']}ms  "
          f"decode={summary['decode_ms']['mean']}ms  full={summary['full_ms']['mean']}ms  fps={summary['fps_full']}")
    print(f"Peak VRAM: {peak_mem_mb:.0f} MB")
    return summary, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--sam-size", default="small", choices=list(SAM_SPECS))
    ap.add_argument("--yolo", default="yolo11n.pt", help="YOLO weights name (auto-downloaded)")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--out", default="seg_output")
    args = ap.parse_args()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    exts = (".jpg", ".jpeg", ".png", ".webp", ".avif")
    images = sorted([p for p in args.data_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in exts])
    if not images:
        raise SystemExit(f"No images in {args.data_dir}")
    print(f"Found {len(images)} images")

    root = args.data_dir / args.out
    out_dir = root / f"yolo_{Path(args.yolo).stem}_sam2_{args.sam_size}"
    run(args.yolo, args.sam_size, images, out_dir, args.conf, dtype,
        max_images=args.max_images)


if __name__ == "__main__":
    main()
