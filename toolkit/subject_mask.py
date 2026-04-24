"""Auto-masking pipeline: YOLO + SAM 2 + SegFormer-clothes.

Extracts per-image `person`, `body`, and `clothing` binary masks for use with
region-aware training losses. Mirrors the reference pipeline in
`scripts/profile_full_pipeline.py`. SegFormer is the primary source of truth;
SAM is loaded for a reference silhouette but not intersected into the final
masks (SAM drops pixels on low-contrast boundaries, SegFormer is semantic).

Phase 1: caching only. The resulting masks are attached to FileItemDTO but
are not consumed by any loss. See `toolkit/config_modules.SubjectMaskConfig`.
"""

import os
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
from tqdm import tqdm

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import FileItemDTO
    from toolkit.config_modules import SubjectMaskConfig


# ============================================================
# Config constants — copied verbatim from scripts/profile_full_pipeline.py
# ============================================================

SAM_HF_IDS: Dict[str, str] = {
    "tiny":      "facebook/sam2.1-hiera-tiny",
    "small":     "facebook/sam2.1-hiera-small",
    "base_plus": "facebook/sam2.1-hiera-base-plus",
    "large":     "facebook/sam2.1-hiera-large",
}

SEGFORMER_ID = "mattmdjaga/segformer_b2_clothes"

# "Body" = identity-relevant human parts we want to preserve.
# Hair is included because it's part of identity.
BODY_CLASSES = {"Hair", "Face", "Left-arm", "Right-arm", "Left-leg", "Right-leg"}
CLOTHING_CLASSES = {"Hat", "Sunglasses", "Upper-clothes", "Skirt", "Pants",
                    "Dress", "Belt", "Left-shoe", "Right-shoe", "Bag", "Scarf"}

CACHE_VERSION_KEY = "subject_mask_v2"  # v2: cached from dataloader-transformed pixels (flip+scale+crop), not raw file


# ============================================================
# Mask post-processing
# ============================================================


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import binary_fill_holes
        return binary_fill_holes(mask.astype(bool)).astype(np.uint8)
    except Exception:
        return mask.astype(np.uint8)


def _smooth_mask(mask: np.ndarray, close_radius: int = 3, do_fill: bool = True) -> np.ndarray:
    """Clean stippling: morphological closing + hole fill.

    close_radius: pixel radius of the structuring disk. 3-5 works well at 1MP.
    """
    try:
        from scipy.ndimage import (binary_closing, binary_fill_holes,
                                   generate_binary_structure, iterate_structure)
        m = mask.astype(bool)
        struct = iterate_structure(generate_binary_structure(2, 2), close_radius)
        m = binary_closing(m, structure=struct)
        if do_fill:
            m = binary_fill_holes(m)
        return m.astype(bool)
    except Exception:
        return mask.astype(bool)


# ============================================================
# Extractor
# ============================================================


class SubjectMaskExtractor:
    """Lazy-loads YOLO, SAM 2, and SegFormer once per instance.

    `.extract(pil_image)` returns a dict with keys:
        person:   np.bool_ (H, W) — body + clothing (pure SegFormer, smoothed)
        body:     np.bool_ (H, W) — hair/face/arms/legs (identity-relevant)
        clothing: np.bool_ (H, W) — upper/pants/skirt/dress/shoes/bag/etc

    All masks are at the ORIGINAL image resolution.
    """

    def __init__(self, config: 'SubjectMaskConfig'):
        self.config = config
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        self.dtype = dtype_map.get(config.dtype, torch.float16)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Lazy imports — keep import cost out of general toolkit import graph.
        from ultralytics import YOLO
        from transformers import (AutoConfig, AutoModelForSemanticSegmentation,
                                  Sam2Model, Sam2Processor, SegformerImageProcessor)

        # YOLO (person detector, COCO class 0)
        self.yolo = YOLO(config.yolo_ckpt)
        # warmup
        try:
            self.yolo.predict(np.zeros((640, 480, 3), dtype=np.uint8),
                              verbose=False, device=0 if self.device == "cuda" else "cpu")
        except Exception:
            # Non-fatal: GPU warmup can fail on some environments; real call will surface errors
            pass

        # SAM 2 (kept for debug / future; not intersected into final masks in Phase 1)
        sam_id = SAM_HF_IDS.get(config.sam_size, SAM_HF_IDS["small"])
        self.sam_processor = Sam2Processor.from_pretrained(sam_id)
        self.sam = Sam2Model.from_pretrained(sam_id, torch_dtype=self.dtype).to(self.device).eval()

        # SegFormer-clothes (primary source of truth for body/clothing semantics)
        self.seg_processor = SegformerImageProcessor.from_pretrained(SEGFORMER_ID)
        self.seg_processor.size = {"height": config.segformer_res, "width": config.segformer_res}
        self.seg = AutoModelForSemanticSegmentation.from_pretrained(
            SEGFORMER_ID, dtype=self.dtype
        ).to(self.device).eval()
        self.seg_cfg = AutoConfig.from_pretrained(SEGFORMER_ID)

        # Precompute body/clothing class id sets from the SegFormer config
        self._body_ids = {i for i, name in self.seg_cfg.id2label.items()
                          if name in BODY_CLASSES}
        self._clothing_ids = {i for i, name in self.seg_cfg.id2label.items()
                              if name in CLOTHING_CLASSES}

    # ------------------------------------------------------------------ #
    # Per-stage
    # ------------------------------------------------------------------ #

    def _run_yolo(self, pil_image):
        """Return a list of [x1,y1,x2,y2] boxes sorted by area desc."""
        img_np = np.array(pil_image)
        device_arg = 0 if self.device == "cuda" else "cpu"
        results = self.yolo.predict(img_np, classes=[0], conf=self.config.yolo_conf,
                                    verbose=False, device=device_arg)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        boxes = r.boxes.xyxy.cpu().numpy().tolist()
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        order = np.argsort(areas)[::-1]
        boxes = [boxes[i] for i in order]
        if self.config.primary_only:
            boxes = boxes[:1]
        return boxes

    def _run_segformer(self, pil_image) -> np.ndarray:
        """Return (H, W) int32 class map at original image resolution."""
        inputs = self.seg_processor(images=pil_image, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        with torch.inference_mode():
            logits = self.seg(**inputs).logits
            up = F.interpolate(logits.float(),
                               size=(pil_image.size[1], pil_image.size[0]),
                               mode="bilinear", align_corners=False)
            class_map = up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)
        return class_map

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def extract(self, pil_image) -> Dict[str, np.ndarray]:
        """Extract {person, body, clothing} bool masks at original resolution.

        Also returns the raw SegFormer ``class_map`` (int32 class ids) and the
        list of YOLO ``boxes`` so callers (e.g. debug preview rendering) can
        visualize detector inputs alongside masks.

        SAM is run (for debug / reference) but NOT intersected into the final
        masks — SegFormer is primary source of truth.
        """
        # YOLO for detection (unused in final mask but kept to signal "no subject")
        boxes = self._run_yolo(pil_image)

        # SegFormer parsing is the semantic source of truth
        class_map = self._run_segformer(pil_image)

        body_parse = np.isin(class_map, list(self._body_ids))
        clothing_parse = np.isin(class_map, list(self._clothing_ids))

        body_mask = _smooth_mask(body_parse, close_radius=2)
        clothing_mask = _smooth_mask(clothing_parse, close_radius=2)
        # person = body ∪ clothing (pure SegFormer), then closed + hole-filled.
        person_mask = _smooth_mask(body_mask | clothing_mask, close_radius=3)

        return {
            "person": person_mask.astype(np.bool_),
            "body": body_mask.astype(np.bool_),
            "clothing": clothing_mask.astype(np.bool_),
            "class_map": class_map.astype(np.int32),
            "boxes": boxes,
        }

    def cleanup(self):
        """Free GPU memory held by loaded models."""
        try:
            del self.yolo
        except Exception:
            pass
        try:
            del self.sam
            del self.sam_processor
        except Exception:
            pass
        try:
            del self.seg
            del self.seg_processor
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# Debug preview rendering
# ============================================================


def _overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color, alpha: float = 0.55) -> np.ndarray:
    """Blend a binary mask onto an RGB image with a solid color + yellow outline."""
    out = image_rgb.astype(np.float32).copy()
    m = mask[..., None].astype(np.float32)
    color_layer = np.array(color, dtype=np.float32)
    out = out * (1 - alpha * m) + color_layer * alpha * m
    try:
        from scipy.ndimage import binary_dilation
        border = binary_dilation(mask.astype(bool), iterations=2) & (~mask.astype(bool))
        out[border] = np.array([255, 255, 0])
    except Exception:
        pass
    return np.clip(out, 0, 255).astype(np.uint8)


def _colormap_from_classes(class_map: np.ndarray, n_classes: int) -> np.ndarray:
    """Render a class map to an RGB color image using a deterministic palette."""
    rng = np.random.RandomState(7)
    pal = np.zeros((n_classes, 3), dtype=np.uint8)
    for i in range(1, n_classes):
        pal[i] = rng.randint(40, 230, 3)
    return pal[class_map.astype(np.int32)]


def _render_preview_tile(pil_image, masks: Dict[str, np.ndarray], n_classes: int,
                         col_width: int = 380):
    """5-panel tile: image | person | body | clothing | parse colormap.

    Returns a PIL Image ready to save.
    """
    from PIL import Image, ImageDraw, ImageFont
    img_np = np.array(pil_image)

    person = masks["person"].astype(np.uint8)
    body = masks["body"].astype(np.uint8)
    clothing = masks["clothing"].astype(np.uint8)
    class_map = masks["class_map"]

    ov_person = _overlay_mask(img_np, person, (100, 180, 255))
    ov_body = _overlay_mask(img_np, body, (255, 120, 80))
    ov_clothing = _overlay_mask(img_np, clothing, (120, 255, 120))
    color_map = _colormap_from_classes(class_map, n_classes)
    parse_blend = (img_np.astype(np.float32) * 0.5 + color_map.astype(np.float32) * 0.5)
    parse_blend = np.clip(parse_blend, 0, 255).astype(np.uint8)

    panels = [img_np, ov_person, ov_body, ov_clothing, parse_blend]
    labels = ["Original", "Person", "Body (hair+face+limbs)", "Clothing", "Parse colormap"]

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    resized = []
    for a in panels:
        r = col_width / a.shape[1]
        new_h = int(a.shape[0] * r)
        resized.append(np.array(Image.fromarray(a).resize((col_width, new_h), Image.BILINEAR)))
    h_max = max(a.shape[0] for a in resized)
    label_h = 26
    canvas = Image.new("RGB", (col_width * len(panels) + 8 * (len(panels) - 1),
                               h_max + label_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for a, lbl in zip(resized, labels):
        canvas.paste(Image.fromarray(a), (x, label_h))
        draw.text((x + 6, 6), lbl, fill=(230, 230, 230), font=font)
        x += col_width + 8
    return canvas


# ============================================================
# Cache helper
# ============================================================


def _downsample_bool(mask: np.ndarray, target_hw: int) -> torch.Tensor:
    """Nearest-neighbor downsample a bool mask to (target_hw, target_hw).

    Returns a torch.bool tensor on CPU.
    """
    # Use torch.nn.functional.interpolate with nearest-exact to get stable
    # downsampling; work in uint8 so we don't fall back to float rounding.
    t = torch.from_numpy(mask.astype(np.uint8)).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    t = F.interpolate(t, size=(target_hw, target_hw), mode="nearest")
    return (t.squeeze(0).squeeze(0) > 0.5).to(torch.bool)


def _resize_bool(mask: np.ndarray, out_h: int, out_w: int) -> torch.Tensor:
    """Nearest-neighbor resize a bool mask to (out_h, out_w). CPU torch.bool."""
    t = torch.from_numpy(mask.astype(np.uint8)).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(out_h, out_w), mode="nearest")
    return (t.squeeze(0).squeeze(0) > 0.5).to(torch.bool)


def _apply_dataloader_transform(
    img,  # PIL.Image.Image in RGB
    file_item: 'FileItemDTO',
):
    """Mirror of dataloader_mixins.load_and_process_image lines 774-793.

    Applies deterministic flips + bucket resize + crop. Falls back to the
    input image unchanged if bucket params aren't attached (non-bucketing
    datasets or pre-setup_buckets invocations).
    """
    from PIL import Image as _Image

    # Per-file deterministic flips (if configured via dataset augments).
    if getattr(file_item, 'flip_x', False):
        img = img.transpose(_Image.FLIP_LEFT_RIGHT)
    if getattr(file_item, 'flip_y', False):
        img = img.transpose(_Image.FLIP_TOP_BOTTOM)

    stw = getattr(file_item, 'scale_to_width', None)
    sth = getattr(file_item, 'scale_to_height', None)
    cx = getattr(file_item, 'crop_x', None)
    cy = getattr(file_item, 'crop_y', None)
    cw = getattr(file_item, 'crop_width', None)
    ch = getattr(file_item, 'crop_height', None)

    if None in (stw, sth, cx, cy, cw, ch):
        # No bucket params — use raw. Caller will downsample to a square.
        return img

    img = img.resize((int(stw), int(sth)), _Image.BICUBIC)
    img = img.crop((int(cx), int(cy), int(cx) + int(cw), int(cy) + int(ch)))
    return img


def _mask_output_hw(file_item: 'FileItemDTO', fallback_hw: int) -> tuple:
    """Preferred output (H, W) for the cached mask.

    If bucket crop dims are known, cache at (crop_h, crop_w) so the mask
    matches the training-tensor aspect ratio and F.interpolate to the latent
    grid at training time is a straight resize. Falls back to a square
    (fallback_hw, fallback_hw) when bucket params are absent.
    """
    cw = getattr(file_item, 'crop_width', None)
    ch = getattr(file_item, 'crop_height', None)
    if cw is not None and ch is not None:
        return int(ch), int(cw)
    return int(fallback_hw), int(fallback_hw)


def cache_subject_masks(
    file_items: List['FileItemDTO'],
    config: 'SubjectMaskConfig',
    preview_dir: Optional[str] = None,
) -> None:
    """Extract and cache subject masks for all file items.

    Caches each image's masks to:
        {image_dir}/_face_id_cache/{stem}_subject_masks.safetensors

    Keys: ``person``, ``body``, ``clothing`` (uint8 0/255 at
    ``config.cache_resolution``), plus a version sentinel
    (``subject_mask_v1 = torch.ones(1)``).

    Attaches to each file item:
        file_item.subject_mask  : torch.bool (H_c, W_c)
        file_item.body_mask     : torch.bool (H_c, W_c)
        file_item.clothing_mask : torch.bool (H_c, W_c)
    where (H_c, W_c) == (cache_resolution, cache_resolution).

    Skips extraction if cache exists and has the matching version key.

    Args:
        preview_dir: If provided AND ``config.save_debug_previews`` is True, a
            5-panel ``{stem}.png`` tile is written here for each extracted image.
            Callers (SDTrainer) should pass the job output folder
            (``save_root/subject_mask_previews``) so previews don't end up
            inside the image dataset itself. If None, previews are skipped even
            when the flag is on.
    """
    from PIL import Image
    from PIL.ImageOps import exif_transpose

    target_hw = int(config.cache_resolution)

    # Determine whether we can skip loading the extractor altogether (all cached).
    extractor: Optional[SubjectMaskExtractor] = None
    empty_count = 0

    for file_item in tqdm(file_items, desc="Caching subject masks"):
        img_dir = os.path.dirname(file_item.path)
        cache_dir = os.path.join(img_dir, '_face_id_cache')
        stem = os.path.splitext(os.path.basename(file_item.path))[0]
        cache_path = os.path.join(cache_dir, f'{stem}_subject_masks.safetensors')

        # ------------------------------------------------------------- cache hit
        if os.path.exists(cache_path):
            try:
                data = load_file(cache_path)
            except Exception:
                data = {}
            has_keys = all(k in data for k in ('person', 'body', 'clothing'))
            has_version = CACHE_VERSION_KEY in data
            if has_keys and has_version:
                person = (data['person'].clone() > 127).to(torch.bool)
                body = (data['body'].clone() > 127).to(torch.bool)
                clothing = (data['clothing'].clone() > 127).to(torch.bool)
                file_item.subject_mask = person
                file_item.body_mask = body
                file_item.clothing_mask = clothing
                continue  # cache hit — no need to run models

        # ------------------------------------------------------------- cache miss
        if extractor is None:
            extractor = SubjectMaskExtractor(config)

        # v2: extract masks from the *dataloader-transformed* pixels so cached
        # masks align with the training tensor (and thus latent grid). Applies
        # the same flip → resize → crop chain as
        # toolkit/dataloader_mixins.load_and_process_image (lines 774-793).
        raw_pil = exif_transpose(Image.open(file_item.path)).convert('RGB')
        pil_image = _apply_dataloader_transform(raw_pil, file_item)
        masks = extractor.extract(pil_image)

        # Cache at training-tensor dimensions (crop_w, crop_h) when known.
        # No square downsample — preserves aspect ratio so F.interpolate to
        # the latent grid at training time is a straight resize.
        out_h, out_w = _mask_output_hw(file_item, fallback_hw=target_hw)
        person_t = _resize_bool(masks['person'], out_h, out_w)
        body_t = _resize_bool(masks['body'], out_h, out_w)
        clothing_t = _resize_bool(masks['clothing'], out_h, out_w)

        if not person_t.any():
            empty_count += 1

        file_item.subject_mask = person_t
        file_item.body_mask = body_t
        file_item.clothing_mask = clothing_t

        os.makedirs(cache_dir, exist_ok=True)
        save_data = {
            'person': (person_t.to(torch.uint8) * 255),
            'body': (body_t.to(torch.uint8) * 255),
            'clothing': (clothing_t.to(torch.uint8) * 255),
            CACHE_VERSION_KEY: torch.ones(1),
        }
        save_file(save_data, cache_path)

        # Optional: write a 5-panel preview tile for visual inspection.
        # Only if both the flag is on AND a target directory was provided —
        # we never write inside the image dataset, so the caller must decide
        # where previews live.
        if getattr(config, 'save_debug_previews', False) and preview_dir:
            os.makedirs(preview_dir, exist_ok=True)
            preview_path = os.path.join(preview_dir, f'{stem}.png')
            try:
                tile = _render_preview_tile(
                    pil_image, masks,
                    n_classes=extractor.seg_cfg.num_labels,
                )
                tile.save(preview_path)
            except Exception as e:
                # Preview failures are non-fatal — the cache is the real artifact.
                print(f"  -  Warning: failed to render preview for {stem}: {e}")

    # Free VRAM held by models (only loaded if we had cache misses)
    if extractor is not None:
        extractor.cleanup()
        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if empty_count > 0:
        print(f"  -  Warning: empty subject mask for {empty_count}/{len(file_items)} images")
