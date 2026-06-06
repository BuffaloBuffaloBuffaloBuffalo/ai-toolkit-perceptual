"""Depth-consistency auxiliary loss via a frozen Depth-Anything-V2 perceptor.

Validated in scripts/depth_loss_validation.py: the MiDaS-style SSI-L1 +
multi-scale gradient-matching loss is numerically stable (self-loss = 0
exactly), discriminates content across images, has monotonic perturbation
response, and is fully differentiable w.r.t. input pixels.  VRAM is ~340 MB
peak for DA2-Small with bf16 + gradient checkpointing on 24 GB GPUs.

Reference: Ranftl et al., "Towards Robust Monocular Depth Estimation"
(MiDaS, TPAMI 2022); Yang et al., "Depth Anything V2" (NeurIPS 2024).
"""

from __future__ import annotations

import gc
import os
import tempfile
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CACHE_VERSION_KEY = "depth_gt_v3"  # v3: GT depth from VAE-encode-then-decode roundtrip pixels (zero-floor target)
CACHE_VERSION_VIDEO_KEY = "depth_gt_video_v2"


def _atomic_save_file(save_data: dict, cache_path: str) -> None:
    """Windows-safe atomic safetensors write.

    safetensors.save_file uses memory-mapped I/O on Windows. When the same
    cache file has been read earlier in the same process via load_file, the
    mmap section can stay live and block a subsequent in-place rewrite,
    producing OSError 1224 ("user-mapped section open"). Writing to a temp
    file in the same directory and using os.replace sidesteps that: rename
    succeeds even when the destination is currently mmap'd.
    """
    cache_dir = os.path.dirname(cache_path) or "."
    os.makedirs(cache_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp_", suffix=".safetensors", dir=cache_dir
    )
    os.close(fd)
    try:
        save_file(save_data, tmp_path)
        os.replace(tmp_path, cache_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def _load_then_close(cache_path: str) -> dict:
    """Read every tensor from a cache file into owned (non-mmap) storage.

    The GT-depth cache shares one ``_face_id_cache/{stem}.safetensors`` per
    image with the face/body/shape embeddings and every other bucket's depth
    map, so writing a new bucket's map means reading the existing keys, merging,
    and rewriting the whole file (safetensors has no in-place append).

    ``load_file`` mmaps the file; cloning each tensor copies it into owned
    storage, and once ``existing`` drops, refcounting unmaps the file right away
    (the returned dict has no reference cycle, so a GC pass is not needed to
    release it). The previous per-item ``gc.collect()`` here was the dominant
    cost of the caching loop whenever the cache file already existed — and it
    always does once the face pass has run (it writes these same files first) or
    when a cache was copied in. A full collect per image is an O(N^2) heap walk
    that scales with the resident heap (the multi-billion-param trainer model,
    the VAE, DA2), starving the loop with the GPU idle (~1-3 it/s, ~0% util on
    an H100). Releasing the mmap never required it, and the rewrite goes through
    ``_atomic_save_file`` (temp file + ``os.replace``), which is already safe
    against a live mmap. Mirrors commit a51428c, which removed the same per-item
    collect from the cache-HIT read path.
    """
    existing = load_file(cache_path)
    copied = {k: v.clone() for k, v in existing.items()}
    del existing
    return copied


def _cache_header_shape(cache_path: str, key: str, version_key: str):
    """Header-only cache-HIT validation: return the shape of ``key`` (a list)
    when both ``key`` and ``version_key`` are present, else None.

    This is the up-front pass's hit check. It deliberately does NOT read the
    tensor: ``safe_open`` + ``keys()`` + ``get_slice(key).get_shape()`` touch
    only the safetensors header (a small JSON blob), never the tensor bytes.
    The heavy per-map read is deferred to the DataLoader worker
    (``FileItemDTO.get_depth_gt`` / ``get_depth_gt_video``), so the caching
    pass stays header-bound — like the latent / text-embedding caches, whose
    hit path is a bare ``os.path.exists()`` — instead of materializing every
    map into RAM serially on the main process (the old behaviour, which left
    the pass crawling even after the per-item ``gc.collect()`` was removed).

    Returns None on an unreadable / corrupt file so the caller recomputes.
    """
    try:
        with safe_open(cache_path, framework="pt", device="cpu") as f:
            keys = f.keys()
            if key in keys and version_key in keys:
                return f.get_slice(key).get_shape()
    except Exception:  # noqa: BLE001 — corrupt/unreadable cache → recompute
        return None
    return None


def _blur_cache_suffix(sigma: float) -> str:
    """Cache-key suffix for pixel_blur_sigma. Empty when sigma <= 0 so
    existing (unblurred) caches keep their original keys."""
    if sigma is None or sigma <= 0:
        return ""
    return f"_blur{int(round(float(sigma) * 10)):02d}"


def gaussian_blur_2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Differentiable depthwise Gaussian blur for ``(B, C, H, W)`` tensors.

    Identity passthrough when ``sigma <= 0`` — call sites can wire this
    behind a config flag without branching. Kernel size is
    ``2*ceil(3*sigma) + 1``; padding is ``reflect`` to avoid edge darkening.
    The kernel is built in fp32 and cast to the input dtype so bf16/fp16
    callers stay numerically safe.
    """
    if sigma is None or sigma <= 0:
        return x
    import math
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    k = 2 * radius + 1
    coords = torch.arange(k, device=x.device, dtype=torch.float32) - radius
    g = torch.exp(-(coords ** 2) / (2.0 * float(sigma) * float(sigma)))
    g = g / g.sum()
    kernel_1d = g.view(1, 1, 1, k)
    kernel_2d = (kernel_1d * kernel_1d.transpose(-1, -2)).to(x.dtype)
    C = x.shape[1]
    kernel = kernel_2d.expand(C, 1, k, k).contiguous()
    x_padded = F.pad(x, (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(x_padded, kernel, groups=C)


def _apply_dataloader_transform(img, file_item):
    """PIL → PIL: mirror dataloader_mixins.load_and_process_image 774-793.

    Applies deterministic flips + bucket resize + crop when bucket params are
    attached to ``file_item``. Falls back to the input image unchanged when
    params are missing (non-bucketed datasets).
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


class DifferentiableDepthEncoder(nn.Module):
    """Frozen Depth-Anything-V2 perceptor with a pure-tensor preprocessor.

    Inputs: ``(B, 3, H, W)`` float tensor in ``[0, 1]`` or ``[-1, 1]``
    (auto-detected).  Gradients flow through preprocessing and the full DA2
    forward.  The HF ``DPTImageProcessor`` is intentionally bypassed: it
    round-trips through PIL + numpy and detaches the computation graph.
    """

    def __init__(
        self,
        model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
        input_size: int = 518,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        grad_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        from transformers import DepthAnythingForDepthEstimation  # lazy import

        self.model = DepthAnythingForDepthEstimation.from_pretrained(
            model_id, torch_dtype=dtype
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        if grad_checkpoint:
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.model.gradient_checkpointing_enable()
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        )
        self.input_size = input_size
        if device is not None:
            self.to(device)

    def _aspect_preserving_hw(self, H: int, W: int) -> Tuple[int, int]:
        if H >= W:
            new_h = self.input_size
            new_w = max(14, int(round(W * self.input_size / H / 14)) * 14)
        else:
            new_w = self.input_size
            new_h = max(14, int(round(H * self.input_size / W / 14)) * 14)
        return new_h, new_w

    def preprocess(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.min().item() < -0.05:
            pixels = (pixels + 1.0) * 0.5
        pixels = pixels.clamp(0.0, 1.0)
        _, _, H, W = pixels.shape
        new_h, new_w = self._aspect_preserving_hw(H, W)
        x = F.interpolate(
            pixels,
            size=(new_h, new_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        x = x.to(self.mean.dtype)
        x = (x - self.mean) / self.std
        return x.to(next(self.model.parameters()).dtype)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return (B, Hd, Wd) float32 depth map.  Gradients flow if input has grad."""
        x = self.preprocess(pixels)
        out = self.model(pixel_values=x)
        return out.predicted_depth.float()


def ssi_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale-and-shift-invariant L1 (MiDaS / Ranftl et al.).

    Solves ``min_{s,t} ||s*pred + t - target||_2`` in closed form per-sample
    (differentiable in ``pred``), then returns L1 between the aligned pred
    and the target.  Returns ``(loss, scale, shift)``.
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    if mask is None:
        mask = torch.ones_like(pred)
    elif mask.dim() == 2:
        mask = mask.unsqueeze(0)
    p = pred.flatten(1)
    g = target.flatten(1)
    m = mask.flatten(1).float()
    n = m.sum(dim=1).clamp_min(1.0)
    mean_p = (p * m).sum(1) / n
    mean_g = (g * m).sum(1) / n
    var_p = (p * p * m).sum(1) / n - mean_p * mean_p
    cov_pg = (p * g * m).sum(1) / n - mean_p * mean_g
    s = cov_pg / var_p.clamp_min(1e-6)
    t = mean_g - s * mean_p
    aligned = s.view(-1, 1, 1) * pred + t.view(-1, 1, 1)
    diff = (aligned - target).abs() * mask
    loss = diff.sum() / mask.sum().clamp_min(1.0)
    return loss, s.detach(), t.detach()


def multiscale_grad_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scales: int = 4,
) -> torch.Tensor:
    """Multi-scale L1 gradient-matching loss (MiDaS)."""
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    if mask is None:
        mask = torch.ones_like(pred)
    elif mask.dim() == 2:
        mask = mask.unsqueeze(0)
    loss = pred.new_zeros(())
    p, g, m = pred, target, mask.float()
    for k in range(scales):
        if k > 0:
            p = F.avg_pool2d(p.unsqueeze(1), 2).squeeze(1)
            g = F.avg_pool2d(g.unsqueeze(1), 2).squeeze(1)
            m = F.avg_pool2d(m.unsqueeze(1), 2).squeeze(1)
        diff = p - g
        mx = m[:, :, 1:] * m[:, :, :-1]
        my = m[:, 1:, :] * m[:, :-1, :]
        dx = (diff[:, :, 1:] - diff[:, :, :-1]).abs() * mx
        dy = (diff[:, 1:, :] - diff[:, :-1, :]).abs() * my
        loss = loss + (dx.sum() / mx.sum().clamp_min(1.0)) + (
            dy.sum() / my.sum().clamp_min(1.0)
        )
    return loss / scales


def compute_depth_consistency_loss(
    encoder: DifferentiableDepthEncoder,
    x0_pixels: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    ssi_weight: float = 1.0,
    grad_weight: float = 0.5,
    grad_scales: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full depth-consistency loss for one sample or a batch.

    Args:
        encoder: frozen DA2 perceptor.
        x0_pixels: ``(B, 3, H, W)`` generator output in ``[0, 1]``.
        gt_depth: ``(B, Hd_gt, Wd_gt)`` cached GT depth (any resolution).
        mask: ``(B, Hm, Wm)`` optional spatial weight in ``[0, 1]``; if None,
            full image.
        ssi_weight, grad_weight, grad_scales: loss composition.

    Returns:
        ``(loss, ssi_component, grad_component, d_pred_detached,
        target_resampled)`` — the first is gradient-carrying; the remaining
        are detached for logging / preview rendering.
    """
    d_pred = encoder(x0_pixels)  # (B, Hd, Wd) fp32, gradient flows

    # Resize GT depth and mask to match pred grid.
    target = gt_depth
    if target.dim() == 2:
        target = target.unsqueeze(0)
    if target.shape[-2:] != d_pred.shape[-2:]:
        target = F.interpolate(
            target.unsqueeze(1).float(),
            size=d_pred.shape[-2:],
            mode="bilinear",
            align_corners=True,
        ).squeeze(1)
    target = target.to(d_pred.device, dtype=d_pred.dtype)

    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[-2:] != d_pred.shape[-2:]:
            mask = F.interpolate(
                mask.unsqueeze(1).float(),
                size=d_pred.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        mask = mask.to(d_pred.device, dtype=d_pred.dtype)

    ssi, s_align, t_align = ssi_l1(d_pred, target, mask)
    # Align d_pred to target's scale before grad-matching so per-sample DA2
    # output magnitude doesn't leak into |∇d_pred − ∇d_gt|. s, t come back
    # detached from ssi_l1, so the gradient w.r.t. d_pred reduces to a
    # constant scalar — alignment-fit noise stays out of the backward path.
    d_pred_aligned = s_align.view(-1, 1, 1) * d_pred + t_align.view(-1, 1, 1)
    grd = multiscale_grad_loss(d_pred_aligned, target, mask, scales=grad_scales)
    loss = ssi_weight * ssi + grad_weight * grd
    return loss, ssi.detach(), grd.detach(), d_pred.detach(), target.detach()


def render_depth_preview(
    pred_pil,
    ref_pil,
    d_pred: torch.Tensor,
    d_gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> "Image.Image":
    """Render a composite preview strip.

    Without a mask: ``[GT RGB | GT depth | Pred RGB | Pred depth]`` (4 panels).
    With a mask: a 5th ``Mask`` panel is appended (white = included).

    Depth maps are percentile-normalized (p2-p98) to grayscale, then color
    inverted so nearer surfaces appear brighter.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    def _depth_to_pil(dep: torch.Tensor, size) -> "Image.Image":
        d = dep.detach().float().cpu().numpy()
        if d.ndim == 3:
            d = d[0]
        lo, hi = np.percentile(d, 2), np.percentile(d, 98)
        dn = np.clip((d - lo) / max(1e-6, (hi - lo)), 0, 1)
        im = Image.fromarray((dn * 255).astype(np.uint8))
        return im.resize(size, Image.BICUBIC)

    W, H = pred_pil.size
    ref_pil = ref_pil.resize((W, H), Image.BICUBIC)
    gt_pil = _depth_to_pil(d_gt, (W, H))
    pred_depth_pil = _depth_to_pil(d_pred, (W, H))

    panels = [
        ("GT RGB", ref_pil),
        ("GT depth", gt_pil.convert("RGB")),
        ("Pred RGB", pred_pil),
        ("Pred depth", pred_depth_pil.convert("RGB")),
    ]

    if mask is not None:
        m = mask.detach().float().cpu().numpy()
        if m.ndim == 3:
            m = m[0]
        m = np.clip(m, 0.0, 1.0)
        mask_pil = Image.fromarray((m * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC).convert("RGB")
        panels.append(("Mask", mask_pil))

    combo = Image.new("RGB", (W * len(panels), H), (0, 0, 0))
    for i, (_, panel_pil) in enumerate(panels):
        combo.paste(panel_pil, (W * i, 0))

    draw = ImageDraw.Draw(combo)
    for i, (label, _) in enumerate(panels):
        draw.text((W * i + 4, 4), label, fill=(255, 255, 0))

    return combo


def cache_depth_gt_embeddings(
    file_items: List["FileItemDTO"],  # noqa: F821
    config: "DepthConsistencyConfig",  # noqa: F821
    device: Optional[torch.device] = None,
    vae_roundtrip_fn: Optional[callable] = None,  # noqa: A002 (lower-case callable is fine here)
    store_as_single_frame_video: bool = False,
) -> None:
    """Extract and cache GT depth maps for all file items.

    Caches to ``{image_dir}/_face_id_cache/{filename}.safetensors`` alongside
    the existing face/body proportion/shape embeddings under the key
    ``depth_gt`` (fp16).  Versioned by ``CACHE_VERSION_KEY``; re-runs when
    the version changes.  The cached depth is at DA2's native output
    resolution (aspect-preserving ``input_size`` long side).

    Args:
        vae_roundtrip_fn: optional callable ``(pixels[1,3,H,W] in [0,1]) ->
            pixels[1,3,H,W] in [0,1]`` that runs the trainer's actual VAE
            encode + decode chain. When supplied, GT depth is extracted from
            the round-trip pixels rather than the raw original — that turns
            the floor of the live training loss to zero (the model can
            actually reach the target). Required for v3 caches.
        store_as_single_frame_video: for video-latent models (LTX-2 / Wan)
            trained on stills (num_frames=1). The live depth loss for these
            runs through the 5D video block, which reads a ``(T, H, W)`` cube
            under the video cache key. When set, the single roundtrip frame is
            saved as a 1-frame cube under ``depth_gt_video*`` (versioned by
            ``CACHE_VERSION_VIDEO_KEY``) and ``is_depth_video_cached`` is set,
            instead of the 2D ``depth_gt*`` image map. Same v3 GT, different
            consumer.
    """
    from PIL import Image
    from PIL.ImageOps import exif_transpose

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("  -  Loading Depth-Anything-V2 perceptor for GT depth caching...")
    encoder = DifferentiableDepthEncoder(
        model_id=config.model_id,
        input_size=config.input_size,
        grad_checkpoint=False,  # no grad needed for caching
        device=device,
    )

    _pix_blur_sigma = float(getattr(config, 'pixel_blur_sigma', 0.0) or 0.0)
    _blur_sfx = _blur_cache_suffix(_pix_blur_sigma)
    zero_depth_count = 0

    # Single-frame-video mode stores the same v3 roundtrip GT under the video
    # cache namespace (key + version + lazy-load flag) so the 5D video depth
    # block reads it; otherwise everything below is the standard image path.
    _key_base = "depth_gt_video" if store_as_single_frame_video else "depth_gt"
    _ver_key = CACHE_VERSION_VIDEO_KEY if store_as_single_frame_video else CACHE_VERSION_KEY

    def _record_lazy_meta(fi, cpath, ckey):
        # Drop the resident tensor and point the DataLoader worker at the cache
        # we just wrote (re-read via get_depth_gt / get_depth_gt_video). Same
        # contract for both the hit and miss paths.
        if store_as_single_frame_video:
            fi.depth_gt_video = None
            fi._depth_video_cache_path = cpath
            fi._depth_video_cache_key = ckey
            fi.is_depth_video_cached = True
        else:
            fi.depth_gt = None
            fi._depth_cache_path = cpath
            fi._depth_cache_key = ckey
            fi.is_depth_cached = True

    for file_item in tqdm(file_items, desc="Caching GT depth maps"):
        img_dir = os.path.dirname(file_item.path)
        cache_dir = os.path.join(img_dir, "_face_id_cache")
        filename_no_ext = os.path.splitext(os.path.basename(file_item.path))[0]
        cache_path = os.path.join(cache_dir, f"{filename_no_ext}.safetensors")

        # Per-bucket cache key. Multi-resolution training assigns the same
        # image to different bucket dims across datasets, and the cached
        # depth values are bucket-dependent (DA2 isn't scale-invariant, and
        # its preprocess upsamples from bucket pixels). Without bucket
        # keying, the first dataset's depth_gt gets reused for every other
        # resolution, silently inflating the loss.
        # When pixel_blur_sigma > 0, the suffix isolates blurred caches from
        # sharp ones so users can flip the param without re-extraction.
        _ch = getattr(file_item, 'crop_height', None)
        _cw = getattr(file_item, 'crop_width', None)
        if isinstance(_ch, int) and isinstance(_cw, int) and _ch > 0 and _cw > 0:
            depth_key = f"{_key_base}_{int(_ch)}x{int(_cw)}{_blur_sfx}"
        else:
            depth_key = f"{_key_base}{_blur_sfx}"

        if os.path.exists(cache_path):
            # Header-only hit check — record where to read from and defer the
            # actual tensor load to the DataLoader worker (get_depth_gt). The
            # bucket-specific depth_key must be validated here (one cache file
            # holds every bucket's map), but reading the bytes now would
            # serialize N large reads on the main process and hold them all
            # resident — the stall this lazy path removes.
            if _cache_header_shape(cache_path, depth_key, _ver_key) is not None:
                _record_lazy_meta(file_item, cache_path, depth_key)
                continue

        # v2: run DA2 on the *dataloader-transformed* pixels so cached depth
        # lines up with the training tensor the trainer actually sees. Mirrors
        # toolkit/dataloader_mixins.load_and_process_image lines 774-793.
        raw_pil = exif_transpose(Image.open(file_item.path)).convert("RGB")
        pil_image = _apply_dataloader_transform(raw_pil, file_item)
        import numpy as np

        arr = torch.from_numpy(
            np.asarray(pil_image, dtype=np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(device)

        # v3: round-trip through VAE encode + decode so the cached GT is the
        # cleanest pixel representation the trainer can actually produce.
        if vae_roundtrip_fn is not None:
            with torch.no_grad():
                arr = vae_roundtrip_fn(arr)

        # Symmetric pre-DA2 blur: matches the live blur applied to x0_pixels
        # in SDTrainer's depth block. Without this, pred-depth (blurred) and
        # GT-depth (sharp) would systematically mismatch even at perfect
        # shape alignment.
        if _pix_blur_sigma > 0:
            with torch.no_grad():
                arr = gaussian_blur_2d(arr, _pix_blur_sigma)

        with torch.no_grad():
            depth = encoder(arr)[0].cpu().to(torch.float16)

        if depth.abs().sum() < 1e-6:
            zero_depth_count += 1

        # Single-frame-video mode saves the frame as a (1, H, W) cube so the
        # 5D video depth block (which expects (T, H, W)) reads it as T=1.
        depth_to_save = depth.unsqueeze(0) if store_as_single_frame_video else depth

        save_data = {}
        if os.path.exists(cache_path):
            try:
                save_data = _load_then_close(cache_path)
            except Exception:  # noqa: BLE001 — corrupt cache → rewrite from scratch
                save_data = {}
        save_data[depth_key] = depth_to_save
        save_data[_ver_key] = torch.ones(1)
        _atomic_save_file(save_data, cache_path)

        # Don't retain the tensor on the shared file_list item: it would be
        # deep-copied on every __getitem__ and held resident for every image
        # at once. Record lazy-load metadata so the worker re-reads from the
        # just-written cache on later epochs — same contract as the hit path.
        _record_lazy_meta(file_item, cache_path, depth_key)

    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if zero_depth_count > 0:
        print(
            f"  -  Warning: zero depth for {zero_depth_count}/{len(file_items)} images"
        )


# ---------------------------------------------------------------------------
# Wan 2.1 video path: TAEHV x0 decode + per-frame depth GT caching
# ---------------------------------------------------------------------------


def load_taehv_wan21(device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Load TAEHV (tiny autoencoder) pretrained for Wan 2.1 latents.

    11M-param decoder — decodes 100+ frames with gradients in ~10 GB peak
    (vs ~20 GB+ for the full Wan 3D VAE). Output is [0, 1] directly
    (no latents_mean/std denormalization needed).

    Weights (``taew2_1.pth``) live at ``toolkit/taehv/taew2_1.pth``; they
    are gitignored and must be placed there manually.
    """
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    _taehv_dir = os.path.join(_here, "taehv")
    if _taehv_dir not in sys.path:
        sys.path.insert(0, _taehv_dir)
    from taehv import TAEHV  # noqa: E402
    ckpt = os.path.join(_taehv_dir, "taew2_1.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"TAEHV checkpoint not found at {ckpt}. "
            "Download taew2_1.pth and place it there before training."
        )
    tae = TAEHV(checkpoint_path=ckpt).to(device).to(dtype).eval()
    for p in tae.parameters():
        p.requires_grad_(False)
    return tae


def load_taehv_ltx2(device: str = "cuda", dtype: torch.dtype = torch.bfloat16, version: str = "2.3"):
    """Load TAEHV tiny decoder for LTX-2 / LTX-2.3 latents (128-channel).

    LTX counterpart of load_taehv_wan21. The TAEHV class auto-configures
    patch_size=4 / latent_channels=128 when the checkpoint name contains
    'taeltx'. madebyollin ships separate weights per generation:
    ``taeltx2_3.pth`` (LTX-2.3) and ``taeltx_2.pth`` (LTX-2.0). Output is
    [0, 1] directly (no latents_mean/std denormalization, same as the Wan
    path). Weights live in ``toolkit/taehv/`` and are auto-downloaded if
    missing (they are gitignored).
    """
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    _taehv_dir = os.path.join(_here, "taehv")
    if _taehv_dir not in sys.path:
        sys.path.insert(0, _taehv_dir)
    from taehv import TAEHV  # noqa: E402
    fname = "taeltx2_3.pth" if str(version) == "2.3" else "taeltx_2.pth"
    ckpt = os.path.join(_taehv_dir, fname)
    if not os.path.exists(ckpt):
        import urllib.request
        url = f"https://raw.githubusercontent.com/madebyollin/taehv/main/{fname}"
        print(f"DepthConsistency: downloading LTX tiny decoder {fname} from {url} ...")
        urllib.request.urlretrieve(url, ckpt)
    tae = TAEHV(checkpoint_path=ckpt).to(device).to(dtype).eval()
    for p in tae.parameters():
        p.requires_grad_(False)
    return tae


def decode_wan_x0_to_frames(
    x0_latents: torch.Tensor,
    decoder,
) -> torch.Tensor:
    """Decode Wan video x0 prediction from latent to pixel frames.

    Supports TAEHV (tiny, ~10 GB peak for 100+ frames w/ grads) or a full
    ``AutoencoderKLWan``.

    Args:
        x0_latents: (B, C, T, H, W) in normalized latent space (NCTHW).
        decoder: TAEHV instance OR AutoencoderKLWan instance.

    Returns:
        frames: (B, 3, T_out, H_out, W_out) in [0, 1] (NCTHW).
    """
    is_taehv = hasattr(decoder, "t_upscale") and hasattr(decoder, "decode_video")

    if is_taehv:
        # TAEHV expects NTCHW; our latents are NCTHW → permute.
        x0_ntchw = x0_latents.permute(0, 2, 1, 3, 4).to(
            next(decoder.parameters()).dtype
        )
        frames_ntchw = decoder.decode_video(
            x0_ntchw, parallel=True, show_progress_bar=False
        )
        return frames_ntchw.permute(0, 2, 1, 3, 4).float().clamp(0, 1)

    # Fallback: full Wan VAE — needs denormalization.
    vae = decoder
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(x0_latents.device, x0_latents.dtype)
    )
    latents_std = (
        1.0 / torch.tensor(vae.config.latents_std)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(x0_latents.device, x0_latents.dtype)
    )
    raw_latents = x0_latents / latents_std + latents_mean
    video = vae.decode(raw_latents.to(vae.dtype), return_dict=False)[0]
    pixels = (video.float() + 1.0) * 0.5
    return pixels.clamp(0, 1)


def cache_video_depth_gt_embeddings(
    file_items: List["FileItemDTO"],  # noqa: F821
    config: "DepthConsistencyConfig",  # noqa: F821
    device: Optional[torch.device] = None,
    num_frames: Optional[int] = None,
    batch_size: int = 4,
) -> None:
    """Extract and cache per-frame GT depth maps for video file items.

    Cached at ``{video_dir}/_face_id_cache/{stem}.safetensors`` under the key
    ``depth_gt_video`` with shape ``(T, H, W)`` float16 at DA2's native output
    resolution. Versioned by ``CACHE_VERSION_VIDEO_KEY``.

    Args:
        file_items: items whose ``is_video`` is truthy are processed.
        config: ``DepthConsistencyConfig`` — supplies DA2 model id / input size.
        device: CUDA device for extraction.
        num_frames: if set, uniformly subsamples the video to this many frames
            before caching. Must match the training ``num_frames`` so the
            cached T lines up with the decoded x0 T at training time.
        batch_size: frames per DA2 forward pass during caching.
    """
    from toolkit.video_frames import read_video_frames_with_transform

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_items = [f for f in file_items if getattr(f, "is_video", False)]
    if not video_items:
        return

    print(
        f"  -  Loading Depth-Anything-V2 perceptor for GT video depth caching "
        f"({len(video_items)} videos)..."
    )
    encoder = DifferentiableDepthEncoder(
        model_id=config.model_id,
        input_size=config.input_size,
        grad_checkpoint=False,
        device=device,
    )

    _pix_blur_sigma = float(getattr(config, 'pixel_blur_sigma', 0.0) or 0.0)
    _blur_sfx = _blur_cache_suffix(_pix_blur_sigma)
    video_depth_key = f"depth_gt_video{_blur_sfx}"

    for file_item in tqdm(video_items, desc="Caching GT depth (video)"):
        vid_dir = os.path.dirname(file_item.path)
        cache_dir = os.path.join(vid_dir, "_face_id_cache")
        stem = os.path.splitext(os.path.basename(file_item.path))[0]
        cache_path = os.path.join(cache_dir, f"{stem}.safetensors")

        # Cache hit: reuse if version matches AND cached T == requested T.
        # Header-only — the frame count comes from the stored shape, so the
        # tensor read defers to the worker (get_depth_gt_video).
        if os.path.exists(cache_path):
            _shape = _cache_header_shape(
                cache_path, video_depth_key, CACHE_VERSION_VIDEO_KEY
            )
            if _shape is not None and (
                num_frames is None or _shape[0] == num_frames
            ):
                file_item.depth_gt_video = None
                file_item._depth_video_cache_path = cache_path
                file_item._depth_video_cache_key = video_depth_key
                file_item.is_depth_video_cached = True
                continue

        # Read + transform frames exactly as the dataloader does (flip → resize
        # → crop), uniformly subsampled to num_frames so cached T matches the
        # decoded x0 T at training time.
        video_tensor = read_video_frames_with_transform(file_item, num_frames)
        if video_tensor is None:
            print(f"  -  Warning: cannot read video frames: {file_item.path}")
            continue

        # Per-frame depth via DA2, no-grad, batched. Blur (if configured)
        # is applied per-batch in pixel space so the cached GT depth matches
        # the live blur on x0_pixels at training time.
        depth_frames: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, video_tensor.shape[0], batch_size):
                batch = video_tensor[start:start + batch_size].to(device)
                if _pix_blur_sigma > 0:
                    batch = gaussian_blur_2d(batch, _pix_blur_sigma)
                d = encoder(batch)  # (b, H_out, W_out)
                depth_frames.append(d.detach().cpu().to(torch.float16))
        depth_video = torch.cat(depth_frames, dim=0)  # (T, H_out, W_out)

        # Don't retain on the shared file_list item — re-read in the worker
        # (get_depth_gt_video) to keep deepcopy cheap and RAM bounded.
        file_item.depth_gt_video = None

        save_data = {}
        if os.path.exists(cache_path):
            try:
                save_data = _load_then_close(cache_path)
            except Exception:  # noqa: BLE001 — corrupt cache → rewrite
                save_data = {}
        save_data[video_depth_key] = depth_video
        save_data[CACHE_VERSION_VIDEO_KEY] = torch.ones(1)
        _atomic_save_file(save_data, cache_path)

        # Lazy-load metadata so cleanup_depth releases the resident cube and
        # the worker re-reads from the just-written cache on later epochs.
        file_item._depth_video_cache_path = cache_path
        file_item._depth_video_cache_key = video_depth_key
        file_item.is_depth_video_cached = True

    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_video_depth_preview(
    output_path: str,
    gen_rgb: torch.Tensor,     # (T, 3, H, W) in [0, 1]
    gen_depth: torch.Tensor,   # (T, H_d, W_d) float
    gt_depth: torch.Tensor,    # (T_g, H_d, W_d) float
    fps: int = 16,
) -> None:
    """Write an animated webp: [gen_rgb | gen_depth | gt_depth] per frame.

    ``gt_depth`` is linspace-resampled along T to match ``gen_rgb``.
    Depth frames are percentile-normalized (p2-p98) per frame for contrast.
    """
    import numpy as np
    from PIL import Image

    def _depth_to_pil(dep_np, size):
        d = dep_np
        if d.ndim == 3:
            d = d[0]
        lo, hi = np.percentile(d, 2), np.percentile(d, 98)
        dn = np.clip((d - lo) / max(1e-6, (hi - lo)), 0, 1)
        im = Image.fromarray((dn * 255).astype(np.uint8))
        return im.resize(size, Image.BICUBIC).convert("RGB")

    T = gen_rgb.shape[0]
    T_g = gt_depth.shape[0]
    if T_g != T:
        ix = torch.linspace(0, T_g - 1, T).long()
        gt_depth = gt_depth[ix]

    gen_rgb_np = gen_rgb.detach().float().clamp(0, 1).cpu().numpy()
    gen_d_np = gen_depth.detach().float().cpu().numpy()
    gt_d_np = gt_depth.detach().float().cpu().numpy()

    H, W = gen_rgb_np.shape[2], gen_rgb_np.shape[3]
    pil_frames = []
    for t in range(T):
        rgb = (gen_rgb_np[t].transpose(1, 2, 0) * 255).astype(np.uint8)
        rgb_pil = Image.fromarray(rgb)
        gen_d_pil = _depth_to_pil(gen_d_np[t], (W, H))
        gt_d_pil = _depth_to_pil(gt_d_np[t], (W, H))

        combo = Image.new("RGB", (W * 3 + 8, H), (0, 0, 0))
        combo.paste(rgb_pil, (0, 0))
        combo.paste(gen_d_pil, (W + 4, 0))
        combo.paste(gt_d_pil, (W * 2 + 8, 0))
        pil_frames.append(combo)

    if not output_path.endswith(".webp"):
        output_path = output_path + ".webp"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    duration_ms = int(1000 / max(1, fps))
    pil_frames[0].save(
        output_path,
        format="WEBP",
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
