"""Subject-mask preflight: run extraction on a dataset folder for visual QC.

For images, writes a 5-panel tile PNG ([image | person | body | clothing |
parse colormap]). For videos, uniformly samples ``--video-frames`` frames across
the clip, runs extraction on each, and vertically stacks the per-frame tiles
into one labelled montage PNG so a clip's temporal behaviour reads as a single
image. Either way the output is one ``<stem>.png`` per source file, plus
``progress.json`` and ``done.marker`` in ``<output_dir>``. Pure inspection —
does NOT touch the dataset's ``_face_id_cache/`` and does not write any
safetensors. Re-runs with different CLI args overwrite tiles in place.

Invoked by the UI's POST /api/dataset-tools/preflight/start route.
"""

import argparse
import json
import os
import sys
import time
import traceback
from glob import glob

# Ensure repo root is on sys.path so `toolkit` imports resolve when invoked
# from the UI subprocess (cwd=TOOLKIT_ROOT, but PYTHONPATH may not be set).
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', required=True, help='Folder of images to inspect')
    p.add_argument('--output-dir', required=True, help='Where to write tile PNGs + progress.json')
    p.add_argument('--segformer-res', type=int, default=768)
    p.add_argument('--body-close-radius', type=int, default=2)
    p.add_argument('--mask-dilate-radius', type=int, default=0)
    p.add_argument('--skin-bias', type=float, default=0.0)
    p.add_argument('--yolo-conf', type=float, default=0.25)
    p.add_argument('--primary-only', type=int, default=1, help='1 = use only the largest YOLO box')
    p.add_argument('--sam-size', default='small', choices=['tiny', 'small', 'base_plus', 'large'])
    p.add_argument('--dtype', default='fp16', choices=['fp16', 'bf16', 'fp32'])
    p.add_argument('--limit', type=int, default=0, help='If >0, only process the first N files')
    p.add_argument('--video-frames', type=int, default=4,
                   help='Frames uniformly sampled per video for the QC montage')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    progress_path = os.path.join(args.output_dir, 'progress.json')
    done_path = os.path.join(args.output_dir, 'done.marker')

    # Persist the resolved config alongside outputs so the UI can read back what
    # was actually run (UI form state may have changed by the time results show).
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    files = _list_media(args.dataset_dir)
    if args.limit > 0:
        files = files[:args.limit]
    total = len(files)

    if total == 0:
        _write_progress(progress_path, {
            'status': 'error',
            'message': f'No images or videos found under {args.dataset_dir}',
            'done': 0, 'total': 0,
        })
        with open(done_path, 'w') as f:
            f.write('error\n')
        return 1

    _write_progress(progress_path, {
        'status': 'starting',
        'message': 'Loading models (YOLO + SegFormer)...',
        'done': 0, 'total': total, 'started_at': time.time(),
    })

    try:
        # Defer heavy imports until after progress.json is written so the UI
        # gets fast feedback that the run is alive.
        from PIL import Image
        from PIL.ImageOps import exif_transpose
        from toolkit.config_modules import SubjectMaskConfig
        from toolkit.subject_mask import (
            SubjectMaskExtractor, _render_preview_tile,
        )
        from toolkit.video_frames import (
            sample_video_frames_pil, vstack_labeled_tiles,
        )

        cfg = SubjectMaskConfig(
            enabled=True,
            segformer_res=args.segformer_res,
            body_close_radius=args.body_close_radius,
            mask_dilate_radius=args.mask_dilate_radius,
            skin_bias=args.skin_bias,
            yolo_conf=args.yolo_conf,
            primary_only=bool(args.primary_only),
            sam_size=args.sam_size,
            dtype=args.dtype,
        )
        extractor = SubjectMaskExtractor(cfg)

        for i, path in enumerate(files):
            stem = os.path.splitext(os.path.basename(path))[0]
            is_video = _is_video(path)
            _write_progress(progress_path, {
                'status': 'running',
                'message': f'Processing {os.path.basename(path)}'
                           + (' (video)' if is_video else ''),
                'done': i, 'total': total, 'current': os.path.basename(path),
            })
            try:
                if is_video:
                    frames = sample_video_frames_pil(path, args.video_frames)
                    if not frames:
                        raise Exception('No readable frames in video')
                else:
                    frames = [(None, exif_transpose(Image.open(path)).convert('RGB'))]

                tiles, labels = [], []
                for fidx, frame_pil in frames:
                    masks = extractor.extract(frame_pil)
                    tiles.append(_render_preview_tile(
                        frame_pil, masks, n_classes=extractor.seg_cfg.num_labels,
                    ))
                    labels.append(f'frame {fidx}')

                if is_video:
                    montage = vstack_labeled_tiles(tiles, labels)
                    montage.save(os.path.join(args.output_dir, f'{stem}.png'))
                else:
                    tiles[0].save(os.path.join(args.output_dir, f'{stem}.png'))
            except Exception as e:  # noqa: BLE001
                # Per-image failures are non-fatal; record and continue so
                # one bad file doesn't kill the whole run.
                err_path = os.path.join(args.output_dir, f'{stem}.error.txt')
                with open(err_path, 'w') as ef:
                    ef.write(f'{e}\n\n{traceback.format_exc()}')

        _write_progress(progress_path, {
            'status': 'done',
            'message': f'Completed {total} images',
            'done': total, 'total': total, 'finished_at': time.time(),
        })
        with open(done_path, 'w') as f:
            f.write('ok\n')

        try:
            extractor.cleanup()
        except Exception:
            pass
        return 0

    except Exception as e:  # noqa: BLE001
        _write_progress(progress_path, {
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc(),
            'done': 0, 'total': total,
        })
        with open(done_path, 'w') as f:
            f.write('error\n')
        return 1


if __name__ == '__main__':
    sys.exit(main())
