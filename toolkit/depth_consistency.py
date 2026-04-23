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

import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CACHE_VERSION_KEY = "depth_gt_v1"
CACHE_VERSION_VIDEO_KEY = "depth_gt_video_v1"


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

    ssi, _, _ = ssi_l1(d_pred, target, mask)
    grd = multiscale_grad_loss(d_pred, target, mask, scales=grad_scales)
    loss = ssi_weight * ssi + grad_weight * grd
    return loss, ssi.detach(), grd.detach(), d_pred.detach(), target.detach()


def render_depth_preview(
    pred_pil,
    ref_pil,
    d_pred: torch.Tensor,
    d_gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> "Image.Image":
    """Render a 4-panel composite: [GT RGB | GT depth | Pred RGB | Pred depth].

    Depth maps are percentile-normalized (p2-p98) to grayscale, then color
    inverted so nearer surfaces appear brighter.  When a mask is provided,
    an outline is drawn on each RGB panel.
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

    # Composite
    combo = Image.new("RGB", (W * 4, H), (0, 0, 0))
    combo.paste(ref_pil, (0, 0))
    combo.paste(gt_pil.convert("RGB"), (W, 0))
    combo.paste(pred_pil, (W * 2, 0))
    combo.paste(pred_depth_pil.convert("RGB"), (W * 3, 0))

    draw = ImageDraw.Draw(combo)
    labels = ["GT RGB", "GT depth", "Pred RGB", "Pred depth"]
    for i, label in enumerate(labels):
        draw.text((W * i + 4, 4), label, fill=(255, 255, 0))

    return combo


def cache_depth_gt_embeddings(
    file_items: List["FileItemDTO"],  # noqa: F821
    config: "DepthConsistencyConfig",  # noqa: F821
    device: Optional[torch.device] = None,
) -> None:
    """Extract and cache GT depth maps for all file items.

    Caches to ``{image_dir}/_face_id_cache/{filename}.safetensors`` alongside
    the existing face/body proportion/shape embeddings under the key
    ``depth_gt`` (fp16).  Versioned by ``CACHE_VERSION_KEY``; re-runs when
    the version changes.  The cached depth is at DA2's native output
    resolution (aspect-preserving ``input_size`` long side).
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

    zero_depth_count = 0

    for file_item in tqdm(file_items, desc="Caching GT depth maps"):
        img_dir = os.path.dirname(file_item.path)
        cache_dir = os.path.join(img_dir, "_face_id_cache")
        filename_no_ext = os.path.splitext(os.path.basename(file_item.path))[0]
        cache_path = os.path.join(cache_dir, f"{filename_no_ext}.safetensors")

        if os.path.exists(cache_path):
            data = load_file(cache_path)
            if "depth_gt" in data and CACHE_VERSION_KEY in data:
                file_item.depth_gt = data["depth_gt"].clone()
                continue

        pil_image = exif_transpose(Image.open(file_item.path)).convert("RGB")
        import numpy as np

        arr = torch.from_numpy(
            np.asarray(pil_image, dtype=np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            depth = encoder(arr)[0].cpu().to(torch.float16)

        if depth.abs().sum() < 1e-6:
            zero_depth_count += 1

        file_item.depth_gt = depth

        os.makedirs(cache_dir, exist_ok=True)
        save_data = {}
        if os.path.exists(cache_path):
            existing = load_file(cache_path)
            save_data = {k: v.clone() for k, v in existing.items()}
        save_data["depth_gt"] = depth
        save_data[CACHE_VERSION_KEY] = torch.ones(1)
        save_file(save_data, cache_path)

    del encoder
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
    import cv2
    import numpy as np

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

    for file_item in tqdm(video_items, desc="Caching GT depth (video)"):
        vid_dir = os.path.dirname(file_item.path)
        cache_dir = os.path.join(vid_dir, "_face_id_cache")
        stem = os.path.splitext(os.path.basename(file_item.path))[0]
        cache_path = os.path.join(cache_dir, f"{stem}.safetensors")

        # Cache hit: reuse if version matches AND cached T == requested T.
        if os.path.exists(cache_path):
            data = load_file(cache_path)
            if (
                "depth_gt_video" in data
                and CACHE_VERSION_VIDEO_KEY in data
                and (num_frames is None or data["depth_gt_video"].shape[0] == num_frames)
            ):
                file_item.depth_gt_video = data["depth_gt_video"].clone()
                continue

        # Read frames from disk.
        cap = cv2.VideoCapture(file_item.path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            print(f"  -  Warning: cannot read video frames: {file_item.path}")
            continue

        if num_frames is not None and num_frames < total:
            indices = np.linspace(0, total - 1, num_frames, dtype=int)
        else:
            indices = np.arange(total)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, fr = cap.read()
            if not ok:
                continue
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            frames.append(
                torch.from_numpy(fr).float().permute(2, 0, 1) / 255.0
            )
        cap.release()

        if not frames:
            print(f"  -  Warning: no frames read from {file_item.path}")
            continue

        video_tensor = torch.stack(frames)  # (T, 3, H, W)

        # Per-frame depth via DA2, no-grad, batched.
        depth_frames: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, video_tensor.shape[0], batch_size):
                batch = video_tensor[start:start + batch_size].to(device)
                d = encoder(batch)  # (b, H_out, W_out)
                depth_frames.append(d.detach().cpu().to(torch.float16))
        depth_video = torch.cat(depth_frames, dim=0)  # (T, H_out, W_out)

        file_item.depth_gt_video = depth_video

        os.makedirs(cache_dir, exist_ok=True)
        save_data = {}
        if os.path.exists(cache_path):
            try:
                existing = load_file(cache_path)
                save_data = {k: v.clone() for k, v in existing.items()}
            except Exception:  # noqa: BLE001 — corrupt cache → rewrite
                save_data = {}
        save_data["depth_gt_video"] = depth_video
        save_data[CACHE_VERSION_VIDEO_KEY] = torch.ones(1)
        save_file(save_data, cache_path)

    del encoder
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
