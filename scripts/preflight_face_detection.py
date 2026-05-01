"""Face-detection preflight: run InsightFace detection on a dataset folder for visual QC.

Writes one annotated PNG per image (original with bbox + keypoints overlaid and
a status banner), plus ``progress.json`` and ``done.marker`` in ``<output_dir>``.
Pure inspection — does NOT touch the dataset's ``_face_id_cache/``.

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


def _list_images(dataset_dir: str):
    out = []
    for ext in IMAGE_EXTS:
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
    p.add_argument('--limit', type=int, default=0, help='If >0, only process the first N images')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    progress_path = os.path.join(args.output_dir, 'progress.json')
    done_path = os.path.join(args.output_dir, 'done.marker')

    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    files = _list_images(args.dataset_dir)
    if args.limit > 0:
        files = files[:args.limit]
    total = len(files)
    dataset_name = os.path.basename(os.path.normpath(args.dataset_dir))

    if total == 0:
        _write_progress(progress_path, {
            'status': 'error',
            'message': f'No images found under {args.dataset_dir}',
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

        extractor = FaceIDExtractor(model_name=args.face_model)
        # Honor user-tunable det-size by re-preparing with the requested square.
        if args.det_size != 640:
            extractor.app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))

        n_detected = 0
        n_failed = 0
        n_padded = 0

        for i, path in enumerate(files):
            stem = os.path.splitext(os.path.basename(path))[0]
            _write_progress(progress_path, {
                'status': 'running',
                'message': f'Processing {os.path.basename(path)}',
                'done': i, 'total': total,
                'current': os.path.basename(path),
                'detected': n_detected, 'failed': n_failed, 'padded': n_padded,
                'dataset': dataset_name,
            })
            try:
                pil = exif_transpose(Image.open(path)).convert('RGB')
                bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                faces, pad = extractor._detect(bgr)
                if len(faces) == 0:
                    n_failed += 1
                    tile = _render_tile(pil, None, None, 'NO FACE', (220, 60, 60))
                else:
                    face = extractor._get_largest_face(faces)
                    used_padding = pad > 0
                    if used_padding:
                        n_padded += 1
                    n_detected += 1
                    color = (255, 165, 0) if used_padding else (60, 200, 60)
                    label = f'OK (padded retry, faces={len(faces)})' if used_padding else f'OK (faces={len(faces)})'
                    kps = getattr(face, 'kps', None)
                    if kps is not None and used_padding:
                        # _detect already shifted bbox; kps came from the
                        # padded image so we shift them too for display.
                        import numpy as _np
                        kps = _np.asarray(kps, dtype=_np.float32) - _np.array([pad, pad], dtype=_np.float32)
                    tile = _render_tile(pil, face.bbox, kps, label, color)
                tile.save(os.path.join(args.output_dir, f'{stem}.png'))
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
