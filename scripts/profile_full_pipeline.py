"""Full subject-extraction pipeline with skin/clothes separation.

Pipeline per image:
  1. YOLO detects all persons -> bboxes sorted by confidence.
  2. SAM 2 runs once (image encoder) and decodes masks for each person bbox.
  3. SegFormer-B2-clothes parses the image at pixel level into 17 ATR classes.
  4. Combine:
       subject      = union of SAM person masks  (full silhouette including clothes)
       skin_only    = subject ∩ {Hair, Face, Left/Right-arm, Left/Right-leg}
       clothing     = subject ∩ {Upper-clothes, Dress, Skirt, Pants, Hat, Scarf,
                                  Sunglasses, Belt, Left/Right-shoe, Bag}
       primary_only = (same as subject) but restricted to largest person only
  5. Save overlays + raw binary masks + class colormap + metrics.

Usage:
    python scripts/profile_full_pipeline.py \\
        --data-dir test_data/scarlett_full \\
        --sam-size small \\
        --yolo yolo11n.pt \\
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
from transformers import (AutoConfig, AutoModelForSemanticSegmentation,
                          Sam2Model, Sam2Processor, SegformerImageProcessor)
from ultralytics import YOLO


# ============================================================
# Config / catalog
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

SEGFORMER_ID = "mattmdjaga/segformer_b2_clothes"

# ATR classes (from SegFormer config) – copied for clarity, verified at runtime.
ATR_CLASSES = [
    "Background", "Hat", "Hair", "Sunglasses", "Upper-clothes", "Skirt",
    "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Face",
    "Left-leg", "Right-leg", "Left-arm", "Right-arm", "Bag", "Scarf",
]
# "Body" = identity-relevant human parts we want to preserve.
# Hair is included because it's part of identity.
BODY_CLASSES = {"Hair", "Face", "Left-arm", "Right-arm", "Left-leg", "Right-leg"}
CLOTHING_CLASSES = {"Hat", "Sunglasses", "Upper-clothes", "Skirt", "Pants",
                    "Dress", "Belt", "Left-shoe", "Right-shoe", "Bag", "Scarf"}


# ============================================================
# Visualization
# ============================================================


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color=(64, 200, 255),
                 alpha=0.55, outline=True) -> np.ndarray:
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


def colormap_from_classes(class_map: np.ndarray, n_classes: int) -> np.ndarray:
    rng = np.random.RandomState(7)
    pal = np.zeros((n_classes, 3), dtype=np.uint8)
    for i in range(1, n_classes):
        pal[i] = rng.randint(40, 230, 3)
    return pal[class_map]


def draw_boxes(image_rgb: np.ndarray, boxes, confs) -> np.ndarray:
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


def tile(imgs, labels, col_w=380) -> Image.Image:
    """Tile images horizontally with labels underneath each."""
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    resized = []
    for a in imgs:
        ratio = col_w / a.shape[1]
        new_h = int(a.shape[0] * ratio)
        resized.append(np.array(Image.fromarray(a).resize((col_w, new_h), Image.BILINEAR)))
    h_max = max(a.shape[0] for a in resized)
    label_h = 28
    canvas = Image.new("RGB", (col_w * len(imgs) + 8 * (len(imgs) - 1), h_max + label_h), (20, 20, 20))
    d = ImageDraw.Draw(canvas)
    x = 0
    for a, lbl in zip(resized, labels):
        canvas.paste(Image.fromarray(a), (x, label_h))
        d.text((x + 6, 6), lbl, fill=(230, 230, 230), font=font)
        x += col_w + 8
    return canvas


# ============================================================
# Models
# ============================================================


def load_yolo(ckpt: str) -> YOLO:
    y = YOLO(ckpt)
    # warmup
    _ = y.predict(np.zeros((640, 480, 3), dtype=np.uint8), verbose=False, device=0)
    return y


def load_sam(spec: Sam2Spec, dtype: torch.dtype, device: str):
    processor = Sam2Processor.from_pretrained(spec.hf_id)
    model = Sam2Model.from_pretrained(spec.hf_id, torch_dtype=dtype).to(device).eval()
    return model, processor


def load_segformer(dtype: torch.dtype, device: str, input_res: int = 768):
    proc = SegformerImageProcessor.from_pretrained(SEGFORMER_ID)
    proc.size = {"height": input_res, "width": input_res}
    model = AutoModelForSemanticSegmentation.from_pretrained(
        SEGFORMER_ID, dtype=dtype
    ).to(device).eval()
    cfg = AutoConfig.from_pretrained(SEGFORMER_ID)
    return model, proc, cfg


# ============================================================
# Per-stage runners
# ============================================================


def run_yolo(yolo: YOLO, pil: Image.Image, conf: float, top_k: Optional[int] = None):
    img_np = np.array(pil)
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
    # Sort by area descending (primary subject first)
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    order = np.argsort(areas)[::-1]
    boxes = [boxes[i] for i in order]
    confs = [confs[i] for i in order]
    if top_k is not None:
        boxes = boxes[:top_k]
        confs = confs[:top_k]
    return boxes, confs, (t1 - t0) * 1000


def run_sam_boxes(sam, sam_processor, pil: Image.Image, boxes, device, dtype,
                  multimask: bool = True):
    """SAM 2 with one or more bbox prompts. Returns list of per-object masks."""
    if not boxes:
        return [], 0.0, 0.0
    input_boxes = [[list(b) for b in boxes]]  # [image, box, coords]
    inputs = sam_processor(images=pil, input_boxes=input_boxes, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    cuda_sync()
    t_enc0 = time.perf_counter()
    with torch.inference_mode():
        image_embeddings = sam.get_image_embeddings(inputs["pixel_values"])
        cuda_sync()
        t_enc1 = time.perf_counter()
        dec_inputs = {k: v for k, v in inputs.items() if k != "pixel_values"}
        outputs = sam(image_embeddings=image_embeddings,
                      multimask_output=multimask, **dec_inputs)
        cuda_sync()
    t_dec1 = time.perf_counter()
    masks = sam_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        original_sizes=inputs["original_sizes"].cpu(),
        reshape_input_sizes=inputs["reshape_input_sizes"].cpu() if "reshape_input_sizes" in inputs else None,
    )[0]  # (n_objects, n_masks, H, W)
    iou_scores = outputs.iou_scores[0].cpu()  # (n_objects, n_masks)
    per_obj = []
    for i in range(masks.shape[0]):
        if multimask:
            # mask index 0 = "whole object" for box prompts
            per_obj.append(masks[i, 0].bool().numpy())
        else:
            per_obj.append(masks[i, 0].bool().numpy())
    return per_obj, (t_enc1 - t_enc0) * 1000, (t_dec1 - t_enc1) * 1000


def run_segformer(model, processor, pil: Image.Image, cfg, device, dtype):
    inputs = processor(images=pil, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    cuda_sync()
    t0 = time.perf_counter()
    with torch.inference_mode():
        logits = model(**inputs).logits
        # Upsample to original image size
        up = F.interpolate(logits.float(), size=(pil.size[1], pil.size[0]),
                           mode="bilinear", align_corners=False)
        class_map = up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)
    cuda_sync()
    t1 = time.perf_counter()
    return class_map, (t1 - t0) * 1000


# ============================================================
# Hole filling (optional post-process)
# ============================================================


def fill_holes(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import binary_fill_holes
        return binary_fill_holes(mask.astype(bool)).astype(np.uint8)
    except Exception:
        return mask.astype(np.uint8)


def smooth_mask(mask: np.ndarray, close_radius: int = 3, do_fill: bool = True) -> np.ndarray:
    """Clean stippling: morphological closing + hole fill.

    close_radius: pixel radius of the structuring disk. 3-5 works well at 1MP.
    """
    try:
        from scipy.ndimage import binary_closing, binary_fill_holes, generate_binary_structure, iterate_structure
        m = mask.astype(bool)
        struct = iterate_structure(generate_binary_structure(2, 2), close_radius)
        m = binary_closing(m, structure=struct)
        if do_fill:
            m = binary_fill_holes(m)
        return m.astype(bool)
    except Exception:
        return mask.astype(bool)


# ============================================================
# Main
# ============================================================


def process_image(yolo, sam, sam_proc, seg, seg_proc, seg_cfg,
                  pil: Image.Image, conf: float, primary_only: bool,
                  device: str, dtype: torch.dtype):
    orig_w, orig_h = pil.size
    boxes, confs, yolo_ms = run_yolo(yolo, pil, conf,
                                     top_k=1 if primary_only else None)

    # 1. Subject silhouette (union of SAM masks)
    if boxes:
        per_obj_masks, enc_ms, dec_ms = run_sam_boxes(
            sam, sam_proc, pil, boxes, device, dtype, multimask=True)
        subject_mask = np.any(np.stack(per_obj_masks, axis=0), axis=0)
        source = "yolo_box"
    else:
        per_obj_masks, enc_ms, dec_ms = [], 0.0, 0.0
        subject_mask = np.zeros((orig_h, orig_w), dtype=bool)
        source = "no_detection"

    subject_mask = fill_holes(subject_mask).astype(bool)

    # 2. SegFormer parsing
    class_map, seg_ms = run_segformer(seg, seg_proc, pil, seg_cfg, device, dtype)

    # 3. Filter by class membership — SegFormer is primary source of truth.
    # SAM would drop pixels when contrast is too low (e.g. white dress on white bg), so we
    # rely on SegFormer's class semantics for "is this a human pixel?" and only use SAM
    # as a reference silhouette. Smooth with fill-holes to remove interior gaps.
    body_ids = {i for i, name in seg_cfg.id2label.items() if name in BODY_CLASSES}
    clothing_ids = {i for i, name in seg_cfg.id2label.items() if name in CLOTHING_CLASSES}
    body_parse = np.isin(class_map, list(body_ids))
    clothing_parse = np.isin(class_map, list(clothing_ids))

    body_mask = smooth_mask(body_parse, close_radius=2)
    clothing_mask = smooth_mask(clothing_parse, close_radius=2)

    # Per-class coverage
    coverage = {}
    total = class_map.size
    for cid, name in seg_cfg.id2label.items():
        c = int((class_map == cid).sum())
        if c:
            coverage[name] = {"pixels": c, "pct": round(100 * c / total, 3)}

    # person = body ∪ clothing (pure SegFormer), then smoothed with closing + fill-holes.
    # We do NOT intersect with SAM because SAM drops pixels on low-contrast boundaries
    # (e.g. white dress on light bg). SegFormer is semantic, not edge-based.
    person_mask = smooth_mask(body_mask | clothing_mask, close_radius=3)

    return dict(
        boxes=boxes, confs=confs, source=source,
        subject=subject_mask, person=person_mask, body=body_mask, clothing=clothing_mask,
        class_map=class_map, coverage=coverage,
        yolo_ms=yolo_ms, sam_encode_ms=enc_ms, sam_decode_ms=dec_ms, seg_ms=seg_ms,
    )


def run(args):
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    exts = (".jpg", ".jpeg", ".png", ".webp", ".avif")
    images = sorted([p for p in args.data_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in exts])
    if not images:
        raise SystemExit(f"No images in {args.data_dir}")
    print(f"Found {len(images)} images")
    if args.max_images:
        images = images[:args.max_images]

    root = args.data_dir / args.out
    out_dir = root / f"full_{Path(args.yolo).stem}_sam2_{args.sam_size}_segformer"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_image").mkdir(exist_ok=True)

    # Load models
    print("Loading YOLO...")
    yolo = load_yolo(args.yolo)
    sam_spec = SAM_SPECS[args.sam_size]
    print(f"Loading SAM 2 {sam_spec.hf_id}...")
    sam, sam_proc = load_sam(sam_spec, dtype, device)
    n_sam = sum(p.numel() for p in sam.parameters())
    print(f"  SAM params: {n_sam/1e6:.1f}M")
    print(f"Loading SegFormer {SEGFORMER_ID} at {args.seg_res}x{args.seg_res}...")
    seg, seg_proc, seg_cfg = load_segformer(dtype, device, input_res=args.seg_res)
    n_seg = sum(p.numel() for p in seg.parameters())
    print(f"  SegFormer params: {n_seg/1e6:.1f}M")

    # Warmup
    warm = exif_transpose(Image.open(images[0])).convert("RGB")
    for _ in range(2):
        process_image(yolo, sam, sam_proc, seg, seg_proc, seg_cfg,
                      warm, args.conf, args.primary_only, device, dtype)
    cuda_sync()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    records = []
    n_detected = 0
    for img_path in tqdm(images, desc="pipeline"):
        pil = exif_transpose(Image.open(img_path)).convert("RGB")
        image_np = np.array(pil)
        img_stem = Path(img_path).stem
        img_out = out_dir / "per_image" / img_stem
        img_out.mkdir(exist_ok=True)

        t_start = time.perf_counter()
        r = process_image(yolo, sam, sam_proc, seg, seg_proc, seg_cfg,
                          pil, args.conf, args.primary_only, device, dtype)
        cuda_sync()
        full_ms = (time.perf_counter() - t_start) * 1000

        if r["boxes"]:
            n_detected += 1

        # Save raw binary masks
        Image.fromarray((r["subject"] * 255).astype(np.uint8)).save(img_out / "subject_mask.png")
        Image.fromarray((r["person"] * 255).astype(np.uint8)).save(img_out / "person_mask.png")
        Image.fromarray((r["body"] * 255).astype(np.uint8)).save(img_out / "body_mask.png")
        Image.fromarray((r["clothing"] * 255).astype(np.uint8)).save(img_out / "clothing_mask.png")

        # Overlays
        ov_subject = overlay_mask(image_np, r["subject"].astype(np.uint8), (64, 200, 255), 0.55)
        ov_person = overlay_mask(image_np, r["person"].astype(np.uint8), (100, 180, 255), 0.55)
        ov_body = overlay_mask(image_np, r["body"].astype(np.uint8), (255, 120, 80), 0.6)
        ov_clothing = overlay_mask(image_np, r["clothing"].astype(np.uint8), (120, 255, 120), 0.5)
        Image.fromarray(ov_subject).save(img_out / "subject_overlay.png")
        Image.fromarray(ov_person).save(img_out / "person_overlay.png")
        Image.fromarray(ov_body).save(img_out / "body_overlay.png")
        Image.fromarray(ov_clothing).save(img_out / "clothing_overlay.png")

        # Parse colormap
        color_map = colormap_from_classes(r["class_map"], seg_cfg.num_labels)
        parse_blend = (image_np.astype(np.float32) * 0.5 + color_map.astype(np.float32) * 0.5).clip(0, 255).astype(np.uint8)
        Image.fromarray(parse_blend).save(img_out / "parse_overlay.png")

        # Tile everything for quick inspection
        prompted = draw_boxes(image_np, r["boxes"], r["confs"]) if r["boxes"] else image_np
        tile([prompted, ov_person, ov_body, ov_clothing, parse_blend],
             ["YOLO boxes", "Person (body+clothing)", "Body (hair+face+limbs)", "Clothing only", "Parse colormap"]
             ).save(img_out / "tile.png")

        records.append({
            "path": str(img_path),
            "stem": img_stem,
            "orig_wh": [pil.size[0], pil.size[1]],
            "n_boxes": len(r["boxes"]),
            "max_conf": round(float(max(r["confs"])), 3) if r["confs"] else None,
            "source": r["source"],
            "subject_pct": round(100 * r["subject"].mean(), 2),
            "person_pct": round(100 * r["person"].mean(), 2),
            "body_pct": round(100 * r["body"].mean(), 2),
            "clothing_pct": round(100 * r["clothing"].mean(), 2),
            "yolo_ms": round(r["yolo_ms"], 2),
            "sam_encode_ms": round(r["sam_encode_ms"], 2),
            "sam_decode_ms": round(r["sam_decode_ms"], 2),
            "seg_ms": round(r["seg_ms"], 2),
            "full_ms": round(full_ms, 2),
            "coverage": r["coverage"],
        })

    peak_mem_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0

    def stats(a):
        a = np.array(a)
        return dict(mean=round(float(a.mean()), 2), median=round(float(np.median(a)), 2),
                    p95=round(float(np.percentile(a, 95)), 2),
                    min=round(float(a.min()), 2), max=round(float(a.max()), 2))

    summary = {
        "yolo_ckpt": args.yolo,
        "sam_size": args.sam_size,
        "segformer": SEGFORMER_ID,
        "dtype": str(dtype).replace("torch.", ""),
        "primary_only": args.primary_only,
        "conf_threshold": args.conf,
        "n_images": len(records),
        "n_detected": n_detected,
        "detect_rate": round(n_detected / len(records), 3),
        "peak_gpu_mem_MB": round(peak_mem_mb, 1),
        "yolo_ms": stats([r["yolo_ms"] for r in records[1:]]),
        "sam_encode_ms": stats([r["sam_encode_ms"] for r in records[1:] if r["sam_encode_ms"] > 0]),
        "sam_decode_ms": stats([r["sam_decode_ms"] for r in records[1:] if r["sam_decode_ms"] > 0]),
        "seg_ms": stats([r["seg_ms"] for r in records[1:]]),
        "full_ms": stats([r["full_ms"] for r in records[1:]]),
        "fps_full": round(1000.0 / np.mean([r["full_ms"] for r in records[1:]]), 2),
    }

    (out_dir / "metrics.json").write_text(json.dumps({"summary": summary, "per_image": records}, indent=2))

    print(f"\nDetect rate: {n_detected}/{len(records)} ({100*n_detected/len(records):.1f}%)")
    print(f"Timing  yolo={summary['yolo_ms']['mean']} ms  "
          f"sam_enc={summary['sam_encode_ms']['mean']} ms  "
          f"sam_dec={summary['sam_decode_ms']['mean']} ms  "
          f"seg={summary['seg_ms']['mean']} ms  "
          f"full={summary['full_ms']['mean']} ms  fps={summary['fps_full']}")
    print(f"Peak VRAM: {peak_mem_mb:.0f} MB")
    print(f"Outputs: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--sam-size", default="small", choices=list(SAM_SPECS))
    ap.add_argument("--yolo", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--primary-only", action="store_true",
                    help="Keep only the largest YOLO detection (drops background strangers)")
    ap.add_argument("--seg-res", type=int, default=768,
                    help="SegFormer input resolution (512/768/1024). Higher = cleaner boundaries, slower.")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--out", default="seg_output")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
