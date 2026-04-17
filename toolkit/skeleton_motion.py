"""Skeleton motion matching loss for video LoRA training.

Extracts per-frame COCO 17 keypoints from both reference and generated video
frames using ViTPose, then computes a visibility-weighted positional loss in
body-centered coordinates.

This captures specific motions (cartwheels, dances, etc.) rather than just
static body proportions (which are pose-invariant ratios).

Usage:
    # 1. Cache reference keypoints from video
    ref_kp, ref_vis = cache_video_skeleton("cartwheel.mp4")

    # 2. During training, decode x0 and extract generated keypoints
    gen_frames = decode_wan_x0(x0_latents, vae)
    gen_kp, gen_vis = run_vitpose(vitpose_model, gen_frames)

    # 3. Compute loss
    loss = skeleton_motion_loss(gen_kp, gen_vis, ref_kp, ref_vis)
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file


# COCO 17 keypoint indices
L_SHOULDER = 5;  R_SHOULDER = 6
L_HIP = 11;      R_HIP = 12
L_KNEE = 13;     R_KNEE = 14
L_ANKLE = 15;    R_ANKLE = 16

# ViTPose input size (H, W)
VITPOSE_SIZE = (256, 192)

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Temperature for flat_softmax before dsntnn.dsnt.
# ViTPose heatmap peaks are ~0.9 in [0,1]; over a 64x48=3072 grid that's
# too flat for softmax to produce a peaked distribution. temp=20 converges
# to correct coordinate ranges.
DSNT_TEMPERATURE = 20.0


# ---------------------------------------------------------------------------
# ViTPose keypoint extraction — clean path, no affine warp
# ---------------------------------------------------------------------------

class ViTPoseExtractor(nn.Module):
    """Thin wrapper around ViTPose for keypoint extraction.

    Resizes input to 256x192, runs the model, and extracts coordinates via
    temperature-scaled dsntnn.  No affine warp, no HF processor — just
    resize, normalize, infer.
    """

    def __init__(self):
        super().__init__()
        from transformers import VitPoseForPoseEstimation
        import dsntnn  # noqa: F401 — verify available

        self.model = VitPoseForPoseEstimation.from_pretrained(
            "usyd-community/vitpose-plus-base",
            dtype=torch.float16,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.register_buffer(
            '_mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer(
            '_std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, pixels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract keypoints from frames.

        Args:
            pixels: (B, 3, H, W) in [0, 1] RGB

        Returns:
            keypoints: (B, 17, 2) in [0, 1] relative to input image dims
            visibility: (B, 17) confidence scores
        """
        import dsntnn

        B = pixels.shape[0]
        all_kp = []
        all_vis = []

        with torch.amp.autocast('cuda', enabled=False):
            for i in range(B):
                sample = pixels[i:i + 1].float()  # (1, 3, H, W)

                # Simple resize to ViTPose input size — differentiable
                sample = F.interpolate(
                    sample, size=VITPOSE_SIZE, mode='bilinear', align_corners=False)

                # ImageNet normalize
                sample = (sample - self._mean.to(sample)) / self._std.to(sample)

                # Infer
                heatmaps = self.model(
                    sample.to(next(self.model.parameters()).dtype),
                    dataset_index=torch.tensor([0], device=sample.device),
                ).heatmaps.float()  # (1, 17, 64, 48)

                # Temperature-scaled soft-argmax → coords in [-1, 1]
                coords = dsntnn.dsnt(dsntnn.flat_softmax(
                    heatmaps * DSNT_TEMPERATURE))  # (1, 17, 2)
                # Confidence from raw heatmap peaks (not softmaxed)
                confidence = heatmaps.flatten(2).max(dim=2).values.detach()

                all_kp.append(coords)
                all_vis.append(confidence)

        keypoints = torch.cat(all_kp, dim=0)    # (B, 17, 2) in [-1, 1]
        visibility = torch.cat(all_vis, dim=0)   # (B, 17)

        # [-1, 1] → [0, 1] in original image space.
        # Since we just resized (no padding/crop), this maps directly.
        keypoints_01 = (keypoints + 1.0) / 2.0

        return keypoints_01, visibility


@torch.no_grad()
def extract_video_keypoints(
    model: ViTPoseExtractor,
    frames: torch.Tensor,
    batch_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract keypoints from all frames of a video.

    Args:
        model: ViTPoseExtractor (on GPU, eval mode)
        frames: (T, 3, H, W) video frames in [0, 1]
        batch_size: max frames per forward pass

    Returns:
        keypoints: (T, 17, 2) in [0, 1] (image space)
        visibility: (T, 17) confidence scores
    """
    T = frames.shape[0]
    all_kp = []
    all_vis = []

    for start in range(0, T, batch_size):
        batch = frames[start:start + batch_size]
        kp, vis = model(batch)
        all_kp.append(kp.cpu())
        all_vis.append(vis.cpu())

    return torch.cat(all_kp, dim=0), torch.cat(all_vis, dim=0)


# ---------------------------------------------------------------------------
# Keypoint normalization
# ---------------------------------------------------------------------------

def normalize_keypoints(
    keypoints: torch.Tensor,
    visibility: torch.Tensor,
    vis_threshold: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize keypoints to body-centered, height-normalized coordinates.

    Centers on the torso midpoint and scales by a pose-invariant height
    proxy (torso + thigh + shin).  Position- and scale-invariant while
    preserving the actual pose.

    Args:
        keypoints: (B, 17, 2) in any consistent coordinate space
        visibility: (B, 17) confidence scores
        vis_threshold: minimum visibility for core keypoints

    Returns:
        normalized: (B, 17, 2) body-centered, height-normalized
        valid: (B,) bool — True if core keypoints visible
    """
    def dist(i, j):
        return (keypoints[:, i] - keypoints[:, j]).pow(2).sum(-1).clamp(min=1e-6).sqrt()

    shoulder_mid = (keypoints[:, L_SHOULDER] + keypoints[:, R_SHOULDER]) / 2
    hip_mid = (keypoints[:, L_HIP] + keypoints[:, R_HIP]) / 2
    body_center = (shoulder_mid + hip_mid) / 2

    torso = (shoulder_mid - hip_mid).pow(2).sum(-1).clamp(min=1e-6).sqrt()
    thigh = (dist(L_HIP, L_KNEE) + dist(R_HIP, R_KNEE)) / 2
    shin = (dist(L_KNEE, L_ANKLE) + dist(R_KNEE, R_ANKLE)) / 2
    height = (torso + thigh + shin).clamp(min=1e-4)

    core_vis = torch.stack([
        visibility[:, L_SHOULDER], visibility[:, R_SHOULDER],
        visibility[:, L_HIP], visibility[:, R_HIP],
    ], dim=-1).min(dim=-1).values
    valid = core_vis >= vis_threshold

    centered = keypoints - body_center.unsqueeze(1)
    normalized = centered / height.unsqueeze(1).unsqueeze(2)

    return normalized, valid


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def skeleton_motion_loss(
    gen_keypoints: torch.Tensor,
    gen_visibility: torch.Tensor,
    ref_keypoints: torch.Tensor,
    ref_visibility: torch.Tensor,
    vis_threshold: float = 0.2,
) -> torch.Tensor:
    """Per-frame skeleton position matching loss.

    Visibility-weighted L1 between body-centered normalized keypoint
    positions, averaged across frames and keypoints.

    Both tensors must have the same temporal length T.
    """
    gen_norm, gen_valid = normalize_keypoints(gen_keypoints, gen_visibility, vis_threshold)
    ref_norm, ref_valid = normalize_keypoints(ref_keypoints, ref_visibility, vis_threshold)

    valid = gen_valid & ref_valid
    if not valid.any():
        return gen_keypoints.new_tensor(0.0)

    combined_vis = torch.min(ref_visibility, gen_visibility)
    vis_weight = combined_vis * (combined_vis >= vis_threshold).float()

    diff = (gen_norm - ref_norm.detach()).abs()
    weighted_diff = diff * vis_weight.unsqueeze(-1)

    per_frame = weighted_diff.sum(dim=(1, 2)) / vis_weight.sum(dim=1).clamp(min=1e-6)
    loss = (per_frame * valid.float()).sum() / valid.float().sum().clamp(min=1.0)
    return loss


def skeleton_translation_loss(
    gen_keypoints: torch.Tensor,
    gen_visibility: torch.Tensor,
    ref_keypoints: torch.Tensor,
    ref_visibility: torch.Tensor,
    vis_threshold: float = 0.2,
) -> torch.Tensor:
    """Absolute body-center translation loss.

    L1 on the body center (torso midpoint) position over time in image
    coordinates — UN-normalized unlike skeleton_motion_loss. Captures
    translation across the frame (e.g. a cartwheel that moves left→right).

    Args:
        gen_keypoints: (T, 17, 2) in [0, 1] image space
        ref_keypoints: (T, 17, 2) in [0, 1] image space
    """
    def body_center(kp):
        shoulder_mid = (kp[:, L_SHOULDER] + kp[:, R_SHOULDER]) / 2
        hip_mid = (kp[:, L_HIP] + kp[:, R_HIP]) / 2
        return (shoulder_mid + hip_mid) / 2  # (T, 2)

    def core_vis(vis):
        return torch.stack([
            vis[:, L_SHOULDER], vis[:, R_SHOULDER],
            vis[:, L_HIP], vis[:, R_HIP],
        ], dim=-1).min(dim=-1).values  # (T,)

    gen_c = body_center(gen_keypoints)
    ref_c = body_center(ref_keypoints)
    valid = (core_vis(gen_visibility) >= vis_threshold) & (core_vis(ref_visibility) >= vis_threshold)
    if not valid.any():
        return gen_keypoints.new_tensor(0.0)

    # L1 in 2D, summed across xy
    diff = (gen_c - ref_c.detach()).abs().sum(-1)  # (T,)
    loss = (diff * valid.float()).sum() / valid.float().sum().clamp(min=1.0)
    return loss


def skeleton_velocity_loss(
    gen_keypoints: torch.Tensor,
    gen_visibility: torch.Tensor,
    ref_keypoints: torch.Tensor,
    ref_visibility: torch.Tensor,
    vis_threshold: float = 0.2,
) -> torch.Tensor:
    """Motion velocity matching loss — penalizes differences in keypoint deltas."""
    if gen_keypoints.shape[0] < 2:
        return gen_keypoints.new_tensor(0.0)

    gen_norm, gen_valid = normalize_keypoints(gen_keypoints, gen_visibility, vis_threshold)
    ref_norm, ref_valid = normalize_keypoints(ref_keypoints, ref_visibility, vis_threshold)

    gen_vel = gen_norm[1:] - gen_norm[:-1]
    ref_vel = ref_norm[1:] - ref_norm[:-1]

    gen_vis_pair = torch.min(gen_visibility[:-1], gen_visibility[1:])
    ref_vis_pair = torch.min(ref_visibility[:-1], ref_visibility[1:])
    combined_vis = torch.min(gen_vis_pair, ref_vis_pair)
    vis_weight = combined_vis * (combined_vis >= vis_threshold).float()

    valid_pair = gen_valid[:-1] & gen_valid[1:] & ref_valid[:-1] & ref_valid[1:]
    if not valid_pair.any():
        return gen_keypoints.new_tensor(0.0)

    diff = (gen_vel - ref_vel.detach()).abs()
    weighted_diff = diff * vis_weight.unsqueeze(-1)
    per_frame = weighted_diff.sum(dim=(1, 2)) / vis_weight.sum(dim=1).clamp(min=1e-6)

    loss = (per_frame * valid_pair.float()).sum() / valid_pair.float().sum().clamp(min=1.0)
    return loss


# ---------------------------------------------------------------------------
# Preview / visualization
# ---------------------------------------------------------------------------

def save_skeleton_preview(
    output_path: str,
    gen_frames: torch.Tensor,
    gen_keypoints: torch.Tensor,
    gen_visibility: torch.Tensor,
    ref_keypoints: torch.Tensor,
    ref_visibility: torch.Tensor,
    max_frames: int = 0,  # 0 = use all frames
    fps: int = 16,
) -> None:
    """Save a side-by-side preview of generated frames with ref/gen skeleton overlays.

    Output is an animated webp showing:
        [generated frame + gen skeleton] | [generated frame + ref skeleton]
    so you can visually compare what the model produced vs what we're asking for.

    Args:
        output_path: path to save .webp (appended if needed)
        gen_frames: (T, 3, H, W) decoded x0 frames in [0, 1]
        gen_keypoints: (T, 17, 2) predicted keypoints from ViTPose on gen frames, [0,1]
        gen_visibility: (T, 17) predicted visibility
        ref_keypoints: (T', 17, 2) cached reference keypoints, [0,1]
        ref_visibility: (T', 17) reference visibility
        max_frames: cap number of frames to render (for speed + file size)
        fps: playback fps
    """
    from PIL import Image as PILImage
    from toolkit.body_id import draw_skeleton_overlay

    T_gen = gen_frames.shape[0]
    T_ref = ref_keypoints.shape[0]

    # Subsample only if max_frames is set (default 0 = keep all)
    if max_frames > 0 and T_gen > max_frames:
        idx = torch.linspace(0, T_gen - 1, max_frames).long()
        gen_frames = gen_frames[idx]
        gen_keypoints = gen_keypoints[idx]
        gen_visibility = gen_visibility[idx]
    T_gen_used = gen_frames.shape[0]

    # Align ref to gen
    if T_ref != T_gen_used:
        idx = torch.linspace(0, T_ref - 1, T_gen_used).long()
        ref_keypoints = ref_keypoints[idx]
        ref_visibility = ref_visibility[idx]

    # Convert frames to PIL, draw both skeletons
    gen_frames_cpu = gen_frames.detach().float().clamp(0, 1).cpu()
    gen_kp_cpu = gen_keypoints.detach().cpu()
    gen_vis_cpu = gen_visibility.detach().cpu()
    ref_kp_cpu = ref_keypoints.detach().cpu()
    ref_vis_cpu = ref_visibility.detach().cpu()

    pil_frames = []
    for t in range(T_gen_used):
        frame_np = (gen_frames_cpu[t].permute(1, 2, 0).numpy() * 255).astype('uint8')
        pil = PILImage.fromarray(frame_np)

        gen_sk = draw_skeleton_overlay(pil, gen_kp_cpu[t].numpy(), gen_vis_cpu[t].numpy())
        ref_sk = draw_skeleton_overlay(pil, ref_kp_cpu[t].numpy(), ref_vis_cpu[t].numpy())

        combined = PILImage.new('RGB', (gen_sk.width * 2 + 4, gen_sk.height))
        combined.paste(gen_sk, (0, 0))
        combined.paste(ref_sk, (gen_sk.width + 4, 0))
        pil_frames.append(combined)

    if not output_path.endswith('.webp'):
        output_path = output_path + '.webp'
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    duration_ms = int(1000 / fps)
    pil_frames[0].save(
        output_path,
        format='WEBP',
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )


# ---------------------------------------------------------------------------
# Wan x0 decode utilities
# ---------------------------------------------------------------------------

def load_taehv_wan21(device='cuda', dtype=torch.bfloat16):
    """Load TAEHV (tiny autoencoder) pretrained for Wan 2.1 latents.

    11M param decoder — decodes 117 frames with gradients in ~10GB peak
    (vs ~20GB+ OOM for the full Wan 3D VAE). Output is [0, 1] directly
    (no latents_mean/std denormalization needed).

    Returns the TAEHV module (frozen, eval mode).
    """
    import sys
    import os as _os
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _taehv_dir = _os.path.join(_here, 'taehv')
    if _taehv_dir not in sys.path:
        sys.path.insert(0, _taehv_dir)
    from taehv import TAEHV
    ckpt = _os.path.join(_taehv_dir, 'taew2_1.pth')
    tae = TAEHV(checkpoint_path=ckpt).to(device).to(dtype).eval()
    for p in tae.parameters():
        p.requires_grad_(False)
    return tae


def decode_wan_x0_to_frames(
    x0_latents: torch.Tensor,
    decoder,
) -> torch.Tensor:
    """Decode Wan video x0 prediction from latent space to pixel frames.

    Supports two decoder types:
    - TAEHV (tiny decoder, ~10GB peak for 117 frames with gradients)
      No latents_mean/std needed — decodes raw diffusion latents directly.
    - Full AutoencoderKLWan (fallback, requires denormalization)

    Args:
        x0_latents: (B, C, T, H, W) in normalized latent space (our NCTHW convention)
        decoder: TAEHV instance OR AutoencoderKLWan instance

    Returns:
        frames: (B, 3, T_out, H_out, W_out) in [0, 1] (NCTHW)
    """
    # Detect TAEHV by its distinctive attributes
    is_taehv = hasattr(decoder, 't_upscale') and hasattr(decoder, 'decode_video')

    if is_taehv:
        # TAEHV expects NTCHW; our latents are NCTHW → permute
        x0_ntchw = x0_latents.permute(0, 2, 1, 3, 4).to(next(decoder.parameters()).dtype)
        frames_ntchw = decoder.decode_video(
            x0_ntchw, parallel=True, show_progress_bar=False)
        # Back to NCTHW
        return frames_ntchw.permute(0, 2, 1, 3, 4).float().clamp(0, 1)

    # Fallback: full Wan VAE — needs denormalization
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


# ---------------------------------------------------------------------------
# Reference keypoint caching
# ---------------------------------------------------------------------------

CACHE_VERSION = 'skeleton_motion_v1'


def cache_video_skeleton(
    video_path: str,
    output_path: Optional[str] = None,
    num_frames: Optional[int] = None,
    device: str = 'cuda',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract and cache skeleton keypoints from a reference video.

    Args:
        video_path: path to video file
        output_path: where to save cache (default: alongside video)
        num_frames: if set, uniformly subsample to this many frames
        device: CUDA device

    Returns:
        keypoints: (T, 17, 2) in [0, 1] image space
        visibility: (T, 17) confidence
    """
    import cv2

    if output_path is None:
        base = os.path.splitext(video_path)[0]
        output_path = f"{base}_skeleton.safetensors"

    # Check cache
    if os.path.exists(output_path):
        data = load_file(output_path)
        if CACHE_VERSION in data:
            print(f"Loaded cached skeleton: {output_path} ({data['keypoints'].shape[0]} frames)")
            return data['keypoints'], data['visibility']
        print(f"Stale cache, re-extracting: {output_path}")
        os.remove(output_path)

    # Load video frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError(f"Cannot read video: {video_path}")

    if num_frames is not None and num_frames < total_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = np.arange(total_frames)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames read from {video_path}")

    video_tensor = torch.stack(frames).to(device)
    _, _, h, w = video_tensor.shape
    print(f"Loaded {len(frames)} frames at {h}x{w} from {video_path}")

    # Extract keypoints
    model = ViTPoseExtractor()
    model.to(device)
    model.eval()

    keypoints, visibility = extract_video_keypoints(model, video_tensor, batch_size=16)

    del model
    torch.cuda.empty_cache()

    # Save cache
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    save_file({
        'keypoints': keypoints,
        'visibility': visibility,
        CACHE_VERSION: torch.ones(1),
    }, output_path)
    print(f"Cached skeleton to {output_path} ({keypoints.shape[0]} frames)")

    return keypoints, visibility
