#!/usr/bin/env python3
"""Sample N random images from a source directory into a new dataset folder.

Useful for building a smaller dataset (e.g., a regularization subset) from a
large image directory. Each sampled image is paired with its matching caption
file (`.txt` or `.caption` with the same stem) when one exists, so the
resulting folder is a self-contained ai-toolkit dataset.

Usage:
    python scripts/sample_dataset.py \\
        --source /path/to/source \\
        --output /path/to/output \\
        --count 50

Options:
    --seed SEED       Random seed for reproducible sampling (default: random)
    --symlink         Symlink files instead of copying (saves disk space, but
                      breaks if the source ever moves)
    --recursive       Recurse into subdirectories of --source
    --extensions ...  Override the default image extensions (with leading dots)
    --overwrite       Allow writing into a non-empty output directory
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

DEFAULT_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')
CAPTION_EXTENSIONS = ('.txt', '.caption')


def collect_images(source: Path, extensions, recursive: bool):
    """Return a sorted list of image paths in `source`."""
    if recursive:
        files = (p for p in source.rglob('*') if p.is_file())
    else:
        files = (p for p in source.iterdir() if p.is_file())
    return sorted(p for p in files if p.suffix.lower() in extensions)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--source', type=Path, required=True,
                        help='Source directory containing images.')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output directory (will be created).')
    parser.add_argument('--count', type=int, required=True,
                        help='Number of images to sample.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible sampling.')
    parser.add_argument('--symlink', action='store_true',
                        help='Symlink files instead of copying.')
    parser.add_argument('--recursive', action='store_true',
                        help='Recurse into subdirectories of --source.')
    parser.add_argument('--extensions', nargs='+', default=None,
                        help=f'Image extensions. Default: {DEFAULT_EXTENSIONS}')
    parser.add_argument('--overwrite', action='store_true',
                        help='Allow writing into a non-empty output directory.')
    args = parser.parse_args()

    if not args.source.is_dir():
        sys.exit(f'Source not found or not a directory: {args.source}')

    if args.count <= 0:
        sys.exit(f'--count must be positive, got {args.count}')

    extensions = tuple(
        (e if e.startswith('.') else f'.{e}').lower()
        for e in (args.extensions or DEFAULT_EXTENSIONS)
    )

    images = collect_images(args.source, extensions, args.recursive)
    if not images:
        sys.exit(f'No images with extensions {extensions} found in {args.source}')

    print(f'Found {len(images)} image(s) in {args.source}')

    if args.count > len(images):
        sys.exit(f'Requested {args.count} but only {len(images)} available')

    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        sys.exit(
            f'Output directory {args.output} is not empty. '
            'Use --overwrite to write into it anyway.'
        )

    rng = random.Random(args.seed)
    sample = rng.sample(images, args.count)

    args.output.mkdir(parents=True, exist_ok=True)
    print(f'Writing {len(sample)} sample(s) to {args.output}')

    op = os.symlink if args.symlink else shutil.copy2
    op_label = 'symlinked' if args.symlink else 'copied'

    n_images = 0
    n_captions = 0
    for src_img in sample:
        dst_img = args.output / src_img.name
        if dst_img.exists():
            continue
        op(str(src_img.resolve() if args.symlink else src_img), str(dst_img))
        n_images += 1
        for cap_ext in CAPTION_EXTENSIONS:
            src_cap = src_img.with_suffix(cap_ext)
            if src_cap.exists():
                dst_cap = dst_img.with_suffix(cap_ext)
                if not dst_cap.exists():
                    op(
                        str(src_cap.resolve() if args.symlink else src_cap),
                        str(dst_cap),
                    )
                    n_captions += 1
                break

    print(f'Done: {op_label} {n_images} image(s), {n_captions} caption(s)')


if __name__ == '__main__':
    main()
