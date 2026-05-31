"""Face-detection preflight: run InsightFace detection on a dataset folder for visual QC.

For images, writes one annotated PNG (original with bbox + keypoints overlaid
and a status banner). For videos, uniformly samples ``--video-frames`` frames
across the clip, runs detection on each, and stacks the annotated frames into
one labelled montage PNG (each row's header carries that frame's OK / NO FACE
status). The detected/failed/padded counters are tallied per source file (a
video counts as detected if any sampled frame has a face). Writes
``progress.json`` and ``done.marker`` in ``<output_dir>``. Pure inspection —
does NOT touch the dataset's ``_face_id_cache/``.

Invoked by the UI's POST /api/dataset-tools/face-detect/start route.
"""

import argparse
import json
import os
import sys
import time
import traceback
from glob import glob

# Ensure repo root is on sys.path so `toolkit` imports resolve when invoked
# from the UI subprocess.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
# Mirror toolkit.data_loader.video_extensions so the preflight sees exactly the
# files the dataloader would treat as videos.
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.webm', '.mkv', '.wmv', '.m4v', '.flv')


def _is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS


def _list_media(dataset_dir: str):
    out = []
    for ext in IMAGE_EXTS + VIDEO_EXTS:
        out.extend(glob(os.path.join(dataset_dir, f'*{ext}')))
        out.extend(glob(os.path.join(dataset_dir, f'*{ext.upper()}')))
    return sorted(set(out))


def _write_progress(progress_path: str, payload: dict) -> None:
    tmp = progress_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(payload, f)
    os.replace(tmp, progress_path)


def _render_tile(pil_image, bbox, kps, status: str, color):
    """Annotate the image with bbox + keypoints and append a status banner."""
    from PIL import Image, ImageDraw, ImageFont

    img = pil_image.copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)

    stroke = max(2, min(w, h) // 250)
    if bbox is not None:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=stroke)
    if kps is not None:
        r = max(3, min(w, h) // 200)
        for kp in kps:
            kx, ky = float(kp[0]), float(kp[1])
            draw.ellipse((kx - r, ky - r, kx + r, ky + r), fill=color)

    banner_h = max(22, h // 28)
    banner = Image.new('RGB', (w, banner_h), color=(0, 0, 0))
    bdraw = ImageDraw.Draw(banner)
    font = None
    for path in (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/TTF/DejaVuSans-Bold.ttf',
    ):
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, banner_h - 6)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()
    bdraw.text((8, 2), status, fill=color, font=font)

    canvas = Image.new('RGB', (w, h + banner_h), color=(0, 0, 0))
    canvas.paste(img, (0, 0))
    canvas.paste(banner, (0, h))

    # Downscale wide images so the UI fetch is snappy.
    max_w = 900
    if canvas.size[0] > max_w:
        ratio = max_w / canvas.size[0]
        canvas = canvas.resize((max_w, int(canvas.size[1] * ratio)), Image.LANCZOS)
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', required=True, help='Folder of images to inspect')
    p.add_argument('--output-dir', required=True, help='Where to write tile PNGs + progress.json')
    p.add_argument('--face-model', default='buffalo_l', help='InsightFace model pack name')
    p.add_argument('--det-size', type=int, default=640, help='RetinaFace det_size (square)')
    p.add_argument('--limit', type=int, default=0, help='If >0, only process the first N files')
    p.add_argument('--video-frames', type=int, default=4,
                   help='Frames uniformly sampled per video for the QC montage')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    progress_path = os.path.join(args.output_dir, 'progress.json')
    done_path = os.path.join(args.output_dir, 'done.marker')

    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    files = _list_media(args.dataset_dir)
    if args.limit > 0:
        files = files[:args.limit]
    total = len(files)
    dataset_name = os.path.basename(os.path.normpath(args.dataset_dir))

    if total == 0:
        _write_progress(progress_path, {
            'status': 'error',
            'message': f'No images or videos found under {args.dataset_dir}',
            'done': 0, 'total': 0, 'dataset': dataset_name,
        })
        with open(done_path, 'w') as f:
            f.write('error\n')
        return 1

    _write_progress(progress_path, {
        'status': 'starting',
        'message': 'Loading InsightFace detector...',
        'done': 0, 'total': total, 'started_at': time.time(),
        'dataset': dataset_name,
    })

    try:
        from PIL import Image
        from PIL.ImageOps import exif_transpose
        import numpy as np
        import cv2
        from toolkit.face_id import FaceIDExtractor
        from toolkit.video_frames import (
            sample_video_frames_pil, vstack_labeled_tiles,
        )

        extractor = FaceIDExtractor(model_name=args.face_model)
        # Honor user-tunable det-size by re-preparing with the requested square.
        if args.det_size != 640:
            extractor.app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))

        def _detect_and_render(pil):
            """Detect on one RGB PIL frame. Returns (tile, detected, padded)."""
            bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            faces, pad = extractor._detect(bgr)
            if len(faces) == 0:
                return _render_tile(pil, None, None, 'NO FACE', (220, 60, 60)), False, False
            face = extractor._get_largest_face(faces)
            used_padding = pad > 0
            color = (255, 165, 0) if used_padding else (60, 200, 60)
            label = (f'OK (padded retry, faces={len(faces)})'
                     if used_padding else f'OK (faces={len(faces)})')
            kps = getattr(face, 'kps', None)
            if kps is not None and used_padding:
                # _detect already shifted bbox; kps came from the padded image
                # so we shift them too for display.
                kps = np.asarray(kps, dtype=np.float32) - np.array([pad, pad], dtype=np.float32)
            return _render_tile(pil, face.bbox, kps, label, color), True, used_padding

        n_detected = 0
        n_failed = 0
        n_padded = 0

        for i, path in enumerate(files):
            stem = os.path.splitext(os.path.basename(path))[0]
            is_video = _is_video(path)
            _write_progress(progress_path, {
                'status': 'running',
                'message': f'Processing {os.path.basename(path)}'
                           + (' (video)' if is_video else ''),
                'done': i, 'total': total,
                'current': os.path.basename(path),
                'detected': n_detected, 'failed': n_failed, 'padded': n_padded,
                'dataset': dataset_name,
            })
            try:
                if is_video:
                    frames = sample_video_frames_pil(path, args.video_frames)
                    if not frames:
                        raise Exception('No readable frames in video')
                    tiles, labels = [], []
                    any_detected = any_padded = False
                    for fidx, frame_pil in frames:
                        tile, det, pad_used = _detect_and_render(frame_pil)
                        tiles.append(tile)
                        status = ('OK' if det else 'NO FACE') + (' (padded)' if pad_used else '')
                        labels.append(f'frame {fidx}: {status}')
                        any_detected = any_detected or det
                        any_padded = any_padded or pad_used
                    vstack_labeled_tiles(tiles, labels).save(
                        os.path.join(args.output_dir, f'{stem}.png'))
                    # File-level tally so detected/failed/padded stay <= total.
                    if any_detected:
                        n_detected += 1
                        if any_padded:
                            n_padded += 1
                    else:
                        n_failed += 1
                else:
                    pil = exif_transpose(Image.open(path)).convert('RGB')
                    tile, det, pad_used = _detect_and_render(pil)
                    tile.save(os.path.join(args.output_dir, f'{stem}.png'))
                    if det:
                        n_detected += 1
                        if pad_used:
                            n_padded += 1
                    else:
                        n_failed += 1
            except Exception as e:  # noqa: BLE001
                err_path = os.path.join(args.output_dir, f'{stem}.error.txt')
                with open(err_path, 'w') as ef:
                    ef.write(f'{e}\n\n{traceback.format_exc()}')

        _write_progress(progress_path, {
            'status': 'done',
            'message': (
                f'Detected {n_detected}/{total} ({n_padded} via padding fallback); '
                f'{n_failed} failed.'
            ),
            'done': total, 'total': total, 'finished_at': time.time(),
            'detected': n_detected, 'failed': n_failed, 'padded': n_padded,
            'dataset': dataset_name,
        })
        with open(done_path, 'w') as f:
            f.write('ok\n')
        return 0

    except Exception as e:  # noqa: BLE001
        _write_progress(progress_path, {
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc(),
            'done': 0, 'total': total, 'dataset': dataset_name,
        })
        with open(done_path, 'w') as f:
            f.write('error\n')
        return 1


if __name__ == '__main__':
    sys.exit(main())
