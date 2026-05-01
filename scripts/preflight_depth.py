"""Depth (and depth+mask) preflight: run Depth-Anything-V2 on a dataset folder for visual QC.

Without ``--use-mask``: writes a 2-panel tile per image: ``[ original | depth ]``.
With ``--use-mask``: also loads the subject-mask extractor and writes a 4-panel
tile: ``[ original | depth | subject mask | depth × mask ]`` so the user can
verify the spatial region the depth-consistency loss is restricted to.

Pure inspection — does NOT touch the dataset's ``_face_id_cache/``. Tiles are
overwritten in place on re-runs with the same runId.

Invoked by the UI's POST /api/dataset-tools/depth/start route.
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


def _depth_to_gray_pil(depth_np, size):
    """Percentile-normalize a depth map and return a grayscale PIL Image.

    Mirrors ``toolkit.depth_consistency.render_depth_preview._depth_to_pil``.
    Brighter = nearer (DA2 outputs higher values for closer surfaces).
    """
    import numpy as np
    from PIL import Image

    d = depth_np.astype('float32', copy=False)
    if d.ndim == 3:
        d = d[0]
    lo, hi = float(np.percentile(d, 2)), float(np.percentile(d, 98))
    dn = ((d - lo) / max(1e-6, hi - lo)).clip(0, 1)
    im = Image.fromarray((dn * 255).astype('uint8'))
    return im.resize(size, Image.BICUBIC).convert('RGB')


def _mask_to_pil(mask_bool, size):
    import numpy as np
    from PIL import Image

    m = (np.asarray(mask_bool, dtype='uint8') * 255)
    return Image.fromarray(m).resize(size, Image.NEAREST).convert('RGB')


def _depth_times_mask_pil(depth_gray_pil, mask_bool, size):
    """Show depth gated by the subject mask: outside the mask is dimmed."""
    import numpy as np
    from PIL import Image

    d = np.asarray(depth_gray_pil.convert('L'), dtype='float32') / 255.0
    m = np.asarray(
        Image.fromarray((np.asarray(mask_bool, dtype='uint8') * 255)).resize(size, Image.NEAREST),
        dtype='float32',
    ) / 255.0
    # Outside mask: 15% brightness so the silhouette is still legible.
    gated = d * (m * 0.85 + 0.15)
    out = (gated * 255).clip(0, 255).astype('uint8')
    return Image.fromarray(out).convert('RGB')


def _render_tile(panels, max_total_w=1800):
    """Hstack labelled panels (all sharing height = panels[0].height).

    Each panel is a ``(label, PIL.Image)`` pair. Output is downscaled if the
    horizontal stack would exceed ``max_total_w`` pixels.
    """
    from PIL import Image, ImageDraw, ImageFont

    h = panels[0][1].size[1]
    total_w = sum(p[1].size[0] for p in panels)
    canvas = Image.new('RGB', (total_w, h), (0, 0, 0))
    x = 0
    for _, im in panels:
        canvas.paste(im, (x, 0))
        x += im.size[0]

    draw = ImageDraw.Draw(canvas)
    font = None
    for path in (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/TTF/DejaVuSans-Bold.ttf',
    ):
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, 18)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    x = 0
    for label, im in panels:
        # Label background for readability on bright images.
        tw = max(60, len(label) * 11)
        draw.rectangle((x + 2, 2, x + tw + 8, 26), fill=(0, 0, 0))
        draw.text((x + 6, 4), label, fill=(255, 255, 0), font=font)
        x += im.size[0]

    if canvas.size[0] > max_total_w:
        ratio = max_total_w / canvas.size[0]
        canvas = canvas.resize(
            (max_total_w, int(canvas.size[1] * ratio)),
            Image.LANCZOS,
        )
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', required=True, help='Folder of images to inspect')
    p.add_argument('--output-dir', required=True, help='Where to write tile PNGs + progress.json')
    p.add_argument('--depth-model', default='depth-anything/Depth-Anything-V2-Small-hf')
    p.add_argument('--input-size', type=int, default=518, help='DA2 long-side input resolution')
    p.add_argument('--use-mask', type=int, default=0, help='1 = also extract subject mask + render depth × mask')
    p.add_argument('--segformer-res', type=int, default=768)
    p.add_argument('--body-close-radius', type=int, default=2)
    p.add_argument('--mask-dilate-radius', type=int, default=0)
    p.add_argument('--skin-bias', type=float, default=0.0)
    p.add_argument('--yolo-conf', type=float, default=0.25)
    p.add_argument('--primary-only', type=int, default=1)
    p.add_argument('--sam-size', default='small', choices=['tiny', 'small', 'base_plus', 'large'])
    p.add_argument('--dtype', default='fp16', choices=['fp16', 'bf16', 'fp32'])
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
    use_mask = bool(args.use_mask)

    if total == 0:
        _write_progress(progress_path, {
            'status': 'error',
            'message': f'No images found under {args.dataset_dir}',
            'done': 0, 'total': 0, 'dataset': dataset_name, 'use_mask': use_mask,
        })
        with open(done_path, 'w') as f:
            f.write('error\n')
        return 1

    _write_progress(progress_path, {
        'status': 'starting',
        'message': 'Loading Depth-Anything-V2' + (' + SubjectMaskExtractor...' if use_mask else '...'),
        'done': 0, 'total': total, 'started_at': time.time(),
        'dataset': dataset_name, 'use_mask': use_mask,
    })

    try:
        import numpy as np
        import torch
        from PIL import Image
        from PIL.ImageOps import exif_transpose
        from toolkit.depth_consistency import DifferentiableDepthEncoder

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        depth_dtype = {
            'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32,
        }[args.dtype]
        encoder = DifferentiableDepthEncoder(
            model_id=args.depth_model,
            input_size=args.input_size,
            dtype=depth_dtype,
            grad_checkpoint=False,
            device=device,
        )

        mask_extractor = None
        if use_mask:
            from toolkit.config_modules import SubjectMaskConfig
            from toolkit.subject_mask import SubjectMaskExtractor
            mask_cfg = SubjectMaskConfig(
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
            mask_extractor = SubjectMaskExtractor(mask_cfg)

        n_processed = 0
        n_empty_mask = 0

        for i, path in enumerate(files):
            stem = os.path.splitext(os.path.basename(path))[0]
            _write_progress(progress_path, {
                'status': 'running',
                'message': f'Processing {os.path.basename(path)}',
                'done': i, 'total': total,
                'current': os.path.basename(path),
                'processed': n_processed,
                'empty_mask': n_empty_mask,
                'dataset': dataset_name,
                'use_mask': use_mask,
            })
            try:
                pil = exif_transpose(Image.open(path)).convert('RGB')
                W, H = pil.size

                arr = (
                    torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device)
                )
                with torch.no_grad():
                    depth = encoder(arr)[0].float().cpu().numpy()  # (Hd, Wd)

                depth_pil = _depth_to_gray_pil(depth, (W, H))

                if use_mask and mask_extractor is not None:
                    masks = mask_extractor.extract(pil)
                    person = masks['person']  # (H, W) bool, original res
                    if not bool(person.any()):
                        n_empty_mask += 1
                    mask_pil = _mask_to_pil(person, (W, H))
                    masked_depth_pil = _depth_times_mask_pil(depth_pil, person, (W, H))
                    panels = [
                        ('Original', pil),
                        ('Depth', depth_pil),
                        ('Subject mask', mask_pil),
                        ('Depth × mask', masked_depth_pil),
                    ]
                else:
                    panels = [
                        ('Original', pil),
                        ('Depth', depth_pil),
                    ]

                tile = _render_tile(panels)
                tile.save(os.path.join(args.output_dir, f'{stem}.png'))
                n_processed += 1
            except Exception as e:  # noqa: BLE001
                err_path = os.path.join(args.output_dir, f'{stem}.error.txt')
                with open(err_path, 'w') as ef:
                    ef.write(f'{e}\n\n{traceback.format_exc()}')

        msg = f'Processed {n_processed}/{total} images'
        if use_mask:
            msg += f' ({n_empty_mask} with empty subject mask)'
        _write_progress(progress_path, {
            'status': 'done',
            'message': msg,
            'done': total, 'total': total, 'finished_at': time.time(),
            'processed': n_processed, 'empty_mask': n_empty_mask,
            'dataset': dataset_name, 'use_mask': use_mask,
        })
        with open(done_path, 'w') as f:
            f.write('ok\n')

        if mask_extractor is not None:
            try:
                mask_extractor.cleanup()
            except Exception:
                pass
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return 0

    except Exception as e:  # noqa: BLE001
        _write_progress(progress_path, {
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc(),
            'done': 0, 'total': total, 'dataset': dataset_name, 'use_mask': use_mask,
        })
        with open(done_path, 'w') as f:
            f.write('error\n')
        return 1


if __name__ == '__main__':
    sys.exit(main())
