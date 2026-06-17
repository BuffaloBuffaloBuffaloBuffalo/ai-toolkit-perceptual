"""Auto-masking pipeline: YOLO + SAM 2 + SegFormer-clothes.

Extracts per-image `person`, `body`, and `clothing` binary masks for use with
region-aware training losses. Mirrors the reference pipeline in
`scripts/profile_full_pipeline.py`. SegFormer is the primary source of truth;
SAM is loaded for a reference silhouette but not intersected into the final
masks (SAM drops pixels on low-contrast boundaries, SegFormer is semantic).

Phase 1: caching only. The resulting masks are attached to FileItemDTO but
are not consumed by any loss. See `toolkit/config_modules.SubjectMaskConfig`.

`mask_source: alpha` bypasses all three models and reads the mask from each
image's PNG alpha channel instead (kohya-style hand-authored masks).
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

# Alpha pixels above this are subject, at or below are background. Matches the
# >50% opacity convention used by kohya-style alpha masks.
ALPHA_MASK_THRESHOLD = 127

# Numeric codes stored in the cache file so masks extracted by one source are
# never served when the config asks for the other.
MASK_SOURCE_CODES = {"auto": 0.0, "alpha": 1.0}


# ============================================================
# Mask post-processing
# ============================================================


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import binary_fill_holes
        return binary_fill_holes(mask.astype(bool)).astype(np.uint8)
    except Exception:
        return mask.astype(np.uint8)


def _skin_probability(image_rgb_u8: np.ndarray) -> np.ndarray:
    """Soft skin-tone probability map in [0, 1] from a uint8 RGB image.

    Uses the classical YCrCb thresholds (Cr ∈ [133, 173], Cb ∈ [77, 127])
    blurred for a smooth probability. Tone-agnostic across most skin tones;
    over-dark or over-light skin can fall outside the range. Cheap (~ms
    per frame) and adds no model dependencies.
    """
    try:
        import cv2
        ycc = cv2.cvtColor(image_rgb_u8, cv2.COLOR_RGB2YCrCb)
        cr = ycc[..., 1].astype(np.float32)
        cb = ycc[..., 2].astype(np.float32)
        m = ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)).astype(np.float32)
        return cv2.GaussianBlur(m, (7, 7), 0)
    except Exception:
        return np.zeros(image_rgb_u8.shape[:2], dtype=np.float32)


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
# Alpha-channel extraction (no models)
# ============================================================


def _extract_alpha_masks(pil_image) -> Dict[str, np.ndarray]:
    """Read the image's alpha channel as the subject mask.

    person = body = clothing = (alpha > ALPHA_MASK_THRESHOLD), used verbatim —
    no smoothing or dilation, the mask is treated as hand-authored ground
    truth. Images without an alpha channel get a full-coverage mask so they
    train normally. ``class_map``/``boxes`` are empty placeholders so preview
    rendering keeps working.
    """
    W, H = pil_image.size
    if pil_image.mode != "RGBA":
        # Resolves LA / PA / palette transparency too; plain RGB becomes
        # alpha=255 everywhere, i.e. a full-coverage mask.
        pil_image = pil_image.convert("RGBA")
    alpha_np = np.array(pil_image.getchannel("A"))
    final_mask = (alpha_np > ALPHA_MASK_THRESHOLD).astype(np.bool_)
    return {
        "person": final_mask,
        "body": final_mask.copy(),
        "clothing": final_mask.copy(),
        "class_map": np.zeros((H, W), dtype=np.int32),
        "boxes": [],
    }


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
        self.mask_source = getattr(config, 'mask_source', 'auto')

        if self.mask_source == 'alpha':
            # Masks come straight from the image's alpha channel — no model
            # loading at all (the whole point: zero VRAM, instant extraction).
            self.yolo = None
            self.sam = None
            self.sam_processor = None
            self.seg = None
            self.seg_processor = None
            self.seg_cfg = None
            self._body_ids = set()
            self._clothing_ids = set()
            return

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

        # SegFormer-clothes (primary source of truth for body/clothing semantics).
        # We do the resize ourselves (aspect-preserving, longest side =
        # ``segformer_res``) and pass ``do_resize=False`` at call time. The
        # processor's default ``{height,width}`` resize forces a square,
        # which severely distorts tall/wide images at high resolutions and
        # *decreases* mask accuracy as ``segformer_res`` grows.
        self.seg_processor = SegformerImageProcessor.from_pretrained(SEGFORMER_ID)
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
        """Return (H, W) int32 class map at original image resolution.

        Aspect-preserving: longest side resized to ``segformer_res``, the
        other side rounded to a multiple of 32 (SegFormer's stride). Square
        forcing produced large drops in mask coverage on tall portraits at
        high resolutions because the horizontal upscale fed the model
        interpolation artifacts at scales it was never trained on.
        """
        from PIL import Image

        target = int(self.config.segformer_res)
        W, H = pil_image.size
        if H >= W:
            new_h = target
            new_w = max(32, int(round(W * target / H / 32)) * 32)
        else:
            new_w = target
            new_h = max(32, int(round(H * target / W / 32)) * 32)
        pil_in = pil_image if (new_w, new_h) == (W, H) else pil_image.resize(
            (new_w, new_h), Image.BICUBIC
        )

        inputs = self.seg_processor(
            images=pil_in, return_tensors="pt", do_resize=False,
        ).to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        with torch.inference_mode():
            logits = self.seg(**inputs).logits
            up = F.interpolate(logits.float(),
                               size=(H, W),
                               mode="bilinear", align_corners=False)
            # Skin-tone bias: where the image looks like skin in YCrCb,
            # add a positive bias to body-class logits so close-call
            # clothing/body pixels tip into body. Disabled at 0 (default).
            bias = float(getattr(self.config, "skin_bias", 0.0))
            if bias > 0.0 and len(self._body_ids) > 0:
                skin = _skin_probability(np.asarray(pil_image, dtype=np.uint8))
                skin_t = torch.from_numpy(skin).to(up.device, dtype=up.dtype)
                skin_t = skin_t.unsqueeze(0).unsqueeze(0)
                body_idx = torch.tensor(
                    sorted(self._body_ids), device=up.device, dtype=torch.long,
                )
                up[:, body_idx] = up[:, body_idx] + bias * skin_t
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

        In ``mask_source: alpha`` mode none of the models run; the mask is the
        image's alpha channel, used verbatim.
        """
        if self.mask_source == 'alpha':
            return _extract_alpha_masks(pil_image)

        # YOLO for detection (unused in final mask but kept to signal "no subject")
        boxes = self._run_yolo(pil_image)

        # SegFormer parsing is the semantic source of truth
        class_map = self._run_segformer(pil_image)

        body_parse = np.isin(class_map, list(self._body_ids))
        clothing_parse = np.isin(class_map, list(self._clothing_ids))

        body_mask = _smooth_mask(body_parse, close_radius=int(self.config.body_close_radius))
        clothing_mask = _smooth_mask(clothing_parse, close_radius=2)
        # person = body ∪ clothing (pure SegFormer), then closed + hole-filled.
        person_mask = _smooth_mask(body_mask | clothing_mask, close_radius=3)

        # True dilation grows the outer boundary (closing only fills holes),
        # so this is the knob users reach for when they want a padded mask.
        dilate_r = int(getattr(self.config, "mask_dilate_radius", 0))
        if dilate_r > 0:
            try:
                from scipy.ndimage import (binary_dilation,
                                            generate_binary_structure,
                                            iterate_structure)
                struct = iterate_structure(
                    generate_binary_structure(2, 2), dilate_r,
                )
                person_mask = binary_dilation(person_mask, structure=struct)
            except Exception:
                pass

        return {
            "person": person_mask.astype(np.bool_),
            "body": body_mask.astype(np.bool_),
            "clothing": clothing_mask.astype(np.bool_),
            "class_map": class_map.astype(np.int32),
            "boxes": boxes,
        }

    @property
    def num_parse_classes(self) -> int:
        """SegFormer label count for parse-colormap previews (1 in alpha mode)."""
        return int(self.seg_cfg.num_labels) if self.seg_cfg is not None else 1

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
    if image_rgb.shape[-1] == 4:
        # RGBA source (alpha mask source) — previews render in RGB.
        image_rgb = image_rgb[..., :3]
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


def _render_preview_tile_from_cache(
    pil_image,
    person_t: torch.Tensor,
    body_t: torch.Tensor,
    clothing_t: torch.Tensor,
    col_width: int = 380,
    mask_source: str = 'auto',
):
    """4-panel tile from cached bool masks: image | person | body | clothing.

    Used for upfront QC previews on cache hit, where the SegFormer ``class_map``
    isn't stored. Mirrors :func:`_render_preview_tile` minus the parse colormap.

    In alpha mode person == body == clothing, so the tile collapses to two
    panels: image | alpha mask.
    """
    from PIL import Image, ImageDraw, ImageFont
    img_np = np.array(pil_image)
    if img_np.ndim == 3 and img_np.shape[-1] == 4:
        img_np = img_np[..., :3]
    H, W = img_np.shape[:2]

    def _bool_to_np(t: torch.Tensor) -> np.ndarray:
        # _overlay_mask expects 0/1 (it does alpha*m); 0/255 saturates the blend.
        m = t.detach().cpu().to(torch.uint8).numpy()
        if m.shape != (H, W):
            # Resize via 0/255 so PIL.NEAREST works on visible values, then rebinarize.
            m255 = (m * 255)
            m = np.array(Image.fromarray(m255).resize((W, H), Image.NEAREST))
            m = (m > 127).astype(np.uint8)
        return m

    person = _bool_to_np(person_t)
    ov_person = _overlay_mask(img_np, person, (100, 180, 255))

    if mask_source == 'alpha':
        panels = [img_np, ov_person]
        labels = ["Original", "Mask (alpha channel)"]
    else:
        body = _bool_to_np(body_t)
        clothing = _bool_to_np(clothing_t)
        ov_body = _overlay_mask(img_np, body, (255, 120, 80))
        ov_clothing = _overlay_mask(img_np, clothing, (120, 255, 120))
        panels = [img_np, ov_person, ov_body, ov_clothing]
        labels = ["Original", "Person", "Body (hair+face+limbs)", "Clothing"]

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


def _render_preview_tile(pil_image, masks: Dict[str, np.ndarray], n_classes: int,
                         col_width: int = 380, mask_source: str = 'auto'):
    """5-panel tile: image | person | body | clothing | parse colormap.

    In alpha mode person == body == clothing and there is no SegFormer parse,
    so the tile collapses to two panels: image | alpha mask.

    Returns a PIL Image ready to save.
    """
    from PIL import Image, ImageDraw, ImageFont
    img_np = np.array(pil_image)
    if img_np.ndim == 3 and img_np.shape[-1] == 4:
        img_np = img_np[..., :3]

    person = masks["person"].astype(np.uint8)
    ov_person = _overlay_mask(img_np, person, (100, 180, 255))

    if mask_source == 'alpha':
        panels = [img_np, ov_person]
        labels = ["Original", "Mask (alpha channel)"]
    else:
        body = masks["body"].astype(np.uint8)
        clothing = masks["clothing"].astype(np.uint8)
        class_map = masks["class_map"]

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
    
    🛡️ ANTI-FOOL & WINDOWS LOCK DEFENSE EDITION: 
    - In 'alpha' mode, collapses the multi-channel schema into a robust 
      Binary Mask to prevent loss stacking.
    - Added rigorous memory-mapped file disposal to bypass Windows (os error 1224).
    """
    from PIL import Image
    from PIL.ImageOps import exif_transpose

    target_hw = int(config.cache_resolution)
    mask_source = getattr(config, 'mask_source', 'auto')
    source_code = MASK_SOURCE_CODES.get(mask_source, 0.0)

    def _load_pil(path: str):
        img = exif_transpose(Image.open(path))
        return img.convert('RGBA' if mask_source == 'alpha' else 'RGB')

    # ========================================================================
    # 🛡️ [Fool-proof Hijack 1/2] UI Weight Reduction (Only for alpha mode)
    # ========================================================================
    if mask_source == 'alpha':
        w_person = float(getattr(config, "person_loss_weight", 1.0)) 
        w_body = float(getattr(config, "body_loss_weight", 1.0))
        w_cloth = float(getattr(config, "clothing_loss_weight", 1.0))
        
        final_foreground_weight = max(w_person, w_body, w_cloth)
        if final_foreground_weight <= 0.0:
            final_foreground_weight = 1.0 
            
        if hasattr(config, "person_loss_weight"):
            config.person_loss_weight = final_foreground_weight
        else:
            config.body_loss_weight = final_foreground_weight
            final_foreground_weight = 0.0 
            
        if hasattr(config, "body_loss_weight") and final_foreground_weight == 0.0:
            config.body_loss_weight = 0.0
        if hasattr(config, "clothing_loss_weight"):
            config.clothing_loss_weight = 0.0
    # ========================================================================

    extractor: Optional[SubjectMaskExtractor] = None
    empty_count = 0
    cached_count = 0
    extracted_count = 0

    pbar = tqdm(file_items, desc="Caching subject masks", ascii=True)
    for file_item in pbar:
        img_dir = os.path.dirname(file_item.path)
        cache_dir = os.path.join(img_dir, '_face_id_cache')
        stem = os.path.splitext(os.path.basename(file_item.path))[0]
        
        out_h, out_w = _mask_output_hw(file_item, fallback_hw=target_hw)
        cache_path = os.path.join(
            cache_dir, f'{stem}_subject_masks_{out_h}x{out_w}.safetensors',
        )

        # ------------------------------------------------------------- cache hit
        is_cache_valid = False
        if os.path.exists(cache_path):
            try:
                data = load_file(cache_path)
            except Exception:
                data = {}
            has_keys = all(k in data for k in ('person', 'body', 'clothing'))
            has_version = CACHE_VERSION_KEY in data
            cached_bcr = int(data['body_close_radius'].item()) if 'body_close_radius' in data else 2
            radius_match = cached_bcr == int(config.body_close_radius)
            cached_source = float(data['mask_source'].item()) if 'mask_source' in data else 0.0
            source_match = cached_source == source_code
            
            if has_keys and has_version and radius_match and source_match:
                person = (data['person'].clone() > 127).to(torch.bool)
                
                # ========================================================================
                # 🛡️ [Fool-proof Hijack 2/2 - Cache Hit] Isolate Pixel Channels
                # ========================================================================
                if mask_source == 'alpha':
                    body = torch.zeros_like(person)
                    clothing = torch.zeros_like(person)
                    if getattr(config, "body_loss_weight", 0.0) > 0.0:
                        body = person.clone()
                        person = torch.zeros_like(person)
                else:
                    body = (data['body'].clone() > 127).to(torch.bool)
                    clothing = (data['clothing'].clone() > 127).to(torch.bool)
                # ========================================================================

                file_item.subject_mask = person
                file_item.body_mask = body
                file_item.clothing_mask = clothing

                if getattr(config, 'save_debug_previews', False) and preview_dir:
                    os.makedirs(preview_dir, exist_ok=True)
                    preview_path = os.path.join(preview_dir, f'{stem}.png')
                    if not os.path.exists(preview_path):
                        try:
                            raw_pil = exif_transpose(Image.open(file_item.path)).convert('RGB')
                            pil_image = _apply_dataloader_transform(raw_pil, file_item)
                            tile = _render_preview_tile_from_cache(
                                pil_image, person if person.any() else body, body, clothing,
                                mask_source=mask_source,
                            )
                            tile.save(preview_path)
                        except Exception as e:
                            print(f"  -  Warning: failed to render preview for {stem}: {e}")
                cached_count += 1
                pbar.set_postfix(hit=cached_count, miss=extracted_count)
                is_cache_valid = True
            
            del data
            
            if is_cache_valid:
                continue

        # ------------------------------------------------------------- cache miss
        if extractor is None:
            extractor = SubjectMaskExtractor(config)

        raw_pil = _load_pil(file_item.path)
        pil_image = _apply_dataloader_transform(raw_pil, file_item)
        masks = extractor.extract(pil_image)

        # ========================================================================
        # 🛡️ [Fool-proof Hijack 2/2 - Fresh Extraction] Isolate Pixel Channels
        # ========================================================================
        if mask_source == 'alpha':
            alpha_mask = masks['person']
            if getattr(config, "body_loss_weight", 0.0) > 0.0:
                masks['body'] = alpha_mask
                masks['person'] = np.zeros_like(alpha_mask)
            else:
                masks['person'] = alpha_mask
                masks['body'] = np.zeros_like(alpha_mask)
            masks['clothing'] = np.zeros_like(alpha_mask)
        # ========================================================================

        person_t = _resize_bool(masks['person'], out_h, out_w)
        body_t = _resize_bool(masks['body'], out_h, out_w)
        clothing_t = _resize_bool(masks['clothing'], out_h, out_w)

        if not person_t.any() and not body_t.any():
            empty_count += 1

        file_item.subject_mask = person_t
        file_item.body_mask = body_t
        file_item.clothing_mask = clothing_t

        os.makedirs(cache_dir, exist_ok=True)
        save_data = {
            'person': (person_t.to(torch.uint8) * 255) if person_t.any() else (body_t.to(torch.uint8) * 255),
            'body': (body_t.to(torch.uint8) * 255),
            'clothing': (clothing_t.to(torch.uint8) * 255),
            'body_close_radius': torch.tensor([float(config.body_close_radius)]),
            'mask_source': torch.tensor([source_code]),
        }
        
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except Exception:
                import time
                cache_path = cache_path.replace(".safetensors", f"_retry_{int(time.time())}.safetensors")

        save_file(save_data, cache_path)
        extracted_count += 1
        pbar.set_postfix(hit=cached_count, miss=extracted_count)

        if getattr(config, 'save_debug_previews', False) and preview_dir:
            os.makedirs(preview_dir, exist_ok=True)
            preview_path = os.path.join(preview_dir, f'{stem}.png')
            try:
                tile = _render_preview_tile(
                    pil_image, masks,
                    n_classes=extractor.num_parse_classes,
                    mask_source=mask_source,
                )
                tile.save(preview_path)
            except Exception as e:
                print(f"  -  Warning: failed to render preview for {stem}: {e}")

    if extractor is not None:
        extractor.cleanup()
        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        f"  -  Subject masks: {cached_count} cache hit"
        f"{'s' if cached_count != 1 else ''}, "
        f"{extracted_count} extracted"
    )
    if empty_count > 0:
        print(f"  -  Warning: empty subject mask for {empty_count}/{len(file_items)} images")
