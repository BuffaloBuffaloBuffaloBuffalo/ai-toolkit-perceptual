"""Shared helper for reading a video file into the exact frame tensor the
dataloader produces for training (flip → resize → crop), uniformly subsampled
to ``num_frames``.

Used by the GT-caching paths of the video perceptors (depth-consistency,
ArcFace identity, ViTPose body-proportion) so the frozen perceptor sees the
same frames the model is trained to reconstruct. Keeping this in one place
guarantees the cached GT and the live decoded x0 frames line up geometrically.
"""
from typing import Optional

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
