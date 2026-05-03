"""
Generate RGB|depth side-by-side composites for every image in a folder
using DA2-Small, and write them alongside the originals.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from depth_loss_validation import DA2SmallPerceptor, load_image_as_tensor, side_by_side

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp"}


def main(folder: Path) -> None:
    imgs = sorted(
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and not p.stem.endswith("__depth")
    )
    print(f"[info] {len(imgs)} images in {folder}")
    perceptor = DA2SmallPerceptor()
    t0 = time.time()
    for i, path in enumerate(imgs, 1):
        out = folder / f"{path.stem}__depth.png"
        try:
            img = load_image_as_tensor(path)
            import torch
            with torch.no_grad():
                d = perceptor(img)
            side_by_side(img, d, out)
            print(f"[{i:02d}/{len(imgs)}] {path.name:60s} -> {out.name}")
        except Exception as e:  # noqa: BLE001
            print(f"[{i:02d}/{len(imgs)}] {path.name:60s} FAILED: {e}")
    print(f"[done] {len(imgs)} images in {time.time()-t0:.1f}s -> {folder}")


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parents[1] / "test_data" / "scarlett_full"
    )
    main(folder)
