"""Shared helper for reading a video file into the exact frame tensor the
dataloader produces for training (flip → resize → crop), uniformly subsampled
to ``num_frames``.

Used by the GT-caching paths of the video perceptors (depth-consistency,
ArcFace identity, ViTPose body-proportion) so the frozen perceptor sees the
same frames the model is trained to reconstruct. Keeping this in one place
guarantees the cached GT and the live decoded x0 frames line up geometrically.
"""
import os
from typing import List, Optional, Tuple

import numpy as np
import torch


def read_video_frames_with_transform(
    file_item,
    num_frames: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Decode a video and apply the dataloader's per-frame flip/resize/crop.

    Mirrors ``dataloader_mixins.load_and_process_video``: the flip happens
    before resize+crop, and frames are uniformly subsampled (linspace) to
    ``num_frames`` so the cached T matches the decoded x0 T at training time.

    Args:
        file_item: a ``FileItemDTO`` with ``path`` and the augmentation fields
            (``flip_x/flip_y``, ``scale_to_width/height``, ``crop_x/y/width/height``).
        num_frames: if set and smaller than the clip length, uniformly subsample
            to this many frames; otherwise keep every frame.

    Returns:
        ``(T, 3, H, W)`` float32 tensor in [0, 1], or ``None`` if the video has
        no readable frames.
    """
    import cv2
    from PIL import Image as _PILImage

    # Read frames sequentially — cv2's CAP_PROP_FRAME_COUNT over-reports by 1 on
    # some AVI containers and POS_FRAMES seek to the reported last frame fails
    # silently. Sequential decode gives the actual count.
    cap = cv2.VideoCapture(file_item.path)
    all_frames_bgr = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        all_frames_bgr.append(fr)
    cap.release()
    total = len(all_frames_bgr)
    if total == 0:
        return None

    if num_frames is not None and num_frames < total:
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
    else:
        indices = np.arange(total)

    flip_x = bool(getattr(file_item, 'flip_x', False))
    flip_y = bool(getattr(file_item, 'flip_y', False))

    frames = []
    for idx in indices:
        fr = all_frames_bgr[int(idx)]
        fr_rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        pil = _PILImage.fromarray(fr_rgb)
        # Per-frame transform: flip before resize+crop (same as the dataloader).
        if flip_x:
            pil = pil.transpose(_PILImage.FLIP_LEFT_RIGHT)
        if flip_y:
            pil = pil.transpose(_PILImage.FLIP_TOP_BOTTOM)
        stw = getattr(file_item, 'scale_to_width', None)
        sth = getattr(file_item, 'scale_to_height', None)
        cx = getattr(file_item, 'crop_x', None)
        cy = getattr(file_item, 'crop_y', None)
        cw = getattr(file_item, 'crop_width', None)
        ch = getattr(file_item, 'crop_height', None)
        if None not in (stw, sth, cx, cy, cw, ch):
            pil = pil.resize((int(stw), int(sth)), _PILImage.BICUBIC)
            pil = pil.crop((int(cx), int(cy),
                            int(cx) + int(cw), int(cy) + int(ch)))
        frame_arr = np.asarray(pil, dtype=np.float32) / 255.0
        frames.append(torch.from_numpy(frame_arr).permute(2, 0, 1))

    if not frames:
        return None
    return torch.stack(frames)  # (T, 3, H, W)


def sample_video_frames_pil(
    path: str,
    num_frames: Optional[int] = None,
) -> List[Tuple[int, "object"]]:
    """Decode a video and return up to ``num_frames`` RGB frames for visual QC.

    Unlike :func:`read_video_frames_with_transform` this applies *no* augment /
    resize / crop — the preflight tools want the raw frames at native resolution
    so the user sees exactly what the perceptor sees per frame. Frames are
    uniformly sampled across the whole clip (``linspace``) so the start, middle
    and end are all represented.

    Args:
        path: path to a video file.
        num_frames: if set and smaller than the clip length, uniformly subsample
            to this many frames; otherwise return every frame.

    Returns:
        A list of ``(frame_index, PIL.Image)`` for the sampled frames (the index
        is into the original clip), or ``[]`` if the video has no readable
        frames.
    """
    import cv2
    from PIL import Image as _PILImage

    cap = cv2.VideoCapture(path)
    all_frames_bgr = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        all_frames_bgr.append(fr)
    cap.release()
    total = len(all_frames_bgr)
    if total == 0:
        return []

    if num_frames is not None and 0 < num_frames < total:
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
    else:
        indices = np.arange(total)

    out = []
    for idx in indices:
        fr_rgb = cv2.cvtColor(all_frames_bgr[int(idx)], cv2.COLOR_BGR2RGB)
        out.append((int(idx), _PILImage.fromarray(fr_rgb).convert('RGB')))
    return out


def _qc_font(size: int):
    """Load a bold TTF if present, else PIL's bitmap default (mirrors the
    font lookup the preflight tile renderers use)."""
    from PIL import ImageFont

    for fp in (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/TTF/DejaVuSans-Bold.ttf',
    ):
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()


def vstack_labeled_tiles(
    tiles: list,
    labels: list,
    header_h: int = 24,
    gutter: int = 4,
    max_total_height: int = 6000,
):
    """Vertically stack per-frame QC tiles into one montage PIL image.

    Each frame's tile gets a thin header strip above it carrying ``labels[i]``
    (e.g. ``"frame 120 / 484"``) so a clip's sampled frames read top-to-bottom
    as one image — keeping the preflight output one-PNG-per-source-file. Tiles
    of unequal width are centered on black. The montage is downscaled if it
    would exceed ``max_total_height`` pixels.

    Args:
        tiles: list of PIL images (one per sampled frame).
        labels: per-tile header strings (same length as ``tiles``).
        header_h: height in px of each frame's label strip.
        gutter: vertical gap in px between frame rows.
        max_total_height: downscale the final montage past this height.

    Returns:
        A single RGB ``PIL.Image``, or ``None`` if ``tiles`` is empty.
    """
    from PIL import Image, ImageDraw

    if not tiles:
        return None

    width = max(t.size[0] for t in tiles)
    font = _qc_font(16)

    rows = []
    for tile, label in zip(tiles, labels):
        if tile.size[0] != width:
            centered = Image.new('RGB', (width, tile.size[1]), (0, 0, 0))
            centered.paste(tile, ((width - tile.size[0]) // 2, 0))
            tile = centered
        row = Image.new('RGB', (width, header_h + tile.size[1]), (0, 0, 0))
        ImageDraw.Draw(row).text((6, 3), str(label), fill=(0, 220, 255), font=font)
        row.paste(tile, (0, header_h))
        rows.append(row)

    total_h = sum(r.size[1] for r in rows) + gutter * (len(rows) - 1)
    canvas = Image.new('RGB', (width, total_h), (20, 20, 20))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.size[1] + gutter

    if canvas.size[1] > max_total_height:
        ratio = max_total_height / canvas.size[1]
        canvas = canvas.resize(
            (max(1, int(canvas.size[0] * ratio)), max_total_height),
            Image.LANCZOS,
        )
    return canvas
