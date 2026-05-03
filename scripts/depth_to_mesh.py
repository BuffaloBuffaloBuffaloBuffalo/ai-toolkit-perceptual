"""Turn cached DA2 depth maps into viewable .ply meshes / point clouds.

Usage
-----
Single image:
    PYTHONPATH=. python scripts/depth_to_mesh.py \\
        --image test_data/scarlett_full/clare-bowen-blue-dress.jpg \\
        --out test_data/scarlett_full/depth_meshes/clare-bowen-blue-dress.ply

Bulk (all images in a folder that have cached GT depths):
    PYTHONPATH=. python scripts/depth_to_mesh.py \\
        --folder test_data/scarlett_full

By default emits textured meshes.  Pass ``--pointcloud`` to emit a point
cloud instead.  ``--compute`` re-runs DA2 on the image when no cached
depth is found.

Viewing
-------
The .ply files load in:
    meshlab <file.ply>          # recommended — full 3D viewer
    blender (File > Import > PLY)
    https://3dviewer.net (drag-and-drop)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from PIL.ImageOps import exif_transpose
from safetensors.torch import load_file

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp"}


def load_depth_and_rgb(
    image_path: Path,
    compute_if_missing: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (depth[H,W] float32, rgb[H,W,3] uint8), both at depth resolution."""
    image_path = Path(image_path)
    cache_file = image_path.parent / "_face_id_cache" / (image_path.stem + ".safetensors")

    depth: np.ndarray | None = None
    if cache_file.exists():
        data = load_file(str(cache_file))
        if "depth_gt" in data:
            depth = data["depth_gt"].float().cpu().numpy()

    if depth is None:
        if not compute_if_missing:
            raise FileNotFoundError(
                f"no cached depth_gt for {image_path.name} — "
                f"pass --compute to run DA2 fresh"
            )
        print(f"[compute] running DA2 on {image_path.name}")
        from toolkit.depth_consistency import DifferentiableDepthEncoder

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = DifferentiableDepthEncoder(grad_checkpoint=False, device=device)
        img = exif_transpose(Image.open(image_path)).convert("RGB")
        arr = torch.from_numpy(
            np.asarray(img, dtype=np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            depth = encoder(arr)[0].cpu().numpy()

    img = exif_transpose(Image.open(image_path)).convert("RGB")
    # Resize RGB to match depth's spatial size (depth is at DA2 native grid).
    Hd, Wd = depth.shape
    img = img.resize((Wd, Hd), Image.BICUBIC)
    return depth.astype(np.float32), np.asarray(img, dtype=np.uint8)


def unproject(
    depth: np.ndarray,
    rgb: np.ndarray,
    fov_deg: float = 55.0,
    convention: str = "disparity",
    near: float = 0.5,
    far: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole unproject DA2 depth to 3D.

    DA2's relative-depth output is disparity-like (higher values = closer).
    ``convention="disparity"`` inverts to linear depth before unprojecting;
    ``convention="depth"`` treats the output as linear depth directly.
    Output Z is normalised into ``[near, far]`` in world units.
    """
    H, W = depth.shape
    if convention == "disparity":
        z = 1.0 / (np.maximum(depth, 1e-6))
    else:
        z = depth.copy()
    zmin, zmax = float(z.min()), float(z.max())
    if zmax > zmin:
        z = (z - zmin) / (zmax - zmin)
    z = near + z * (far - near)

    fov = np.deg2rad(fov_deg)
    f = 0.5 * max(W, H) / np.tan(0.5 * fov)
    cx, cy = W / 2.0, H / 2.0

    xv, yv = np.meshgrid(np.arange(W), np.arange(H))
    X = (xv - cx) * z / f
    Y = -(yv - cy) * z / f  # image y grows downward; flip so world-up = up
    Z = -z  # camera looks along -Z

    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1).astype(np.float32)
    colors = rgb.reshape(-1, 3).astype(np.uint8)
    return points, colors


def grid_faces(H: int, W: int, depth: np.ndarray, disc_thresh: float) -> np.ndarray:
    """Two triangles per 2x2 cell, with discontinuity-based edge culling."""
    idx = np.arange(H * W).reshape(H, W)
    tl, tr = idx[:-1, :-1], idx[:-1, 1:]
    bl, br = idx[1:, :-1], idx[1:, 1:]

    d = depth
    d_tl, d_tr = d[:-1, :-1], d[:-1, 1:]
    d_bl, d_br = d[1:, :-1], d[1:, 1:]
    d_max = np.maximum.reduce([d_tl, d_tr, d_bl, d_br])
    d_min = np.minimum.reduce([d_tl, d_tr, d_bl, d_br])
    # Fraction-of-max threshold; DA2 output is non-negative so d_max >= 0.
    ok = (d_max - d_min) < (disc_thresh * np.maximum(d_max, 1e-6))

    tri1 = np.stack([tl, bl, tr], axis=-1)[ok]
    tri2 = np.stack([tr, bl, br], axis=-1)[ok]
    return np.concatenate([tri1, tri2], axis=0).astype(np.int64)


def process_one(
    image_path: Path,
    out_path: Path,
    *,
    as_mesh: bool,
    fov_deg: float,
    convention: str,
    disc_thresh: float,
    compute_if_missing: bool,
) -> None:
    depth, rgb = load_depth_and_rgb(image_path, compute_if_missing=compute_if_missing)
    points, colors = unproject(depth, rgb, fov_deg=fov_deg, convention=convention)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if as_mesh:
        H, W = depth.shape
        faces = grid_faces(H, W, depth, disc_thresh)
        mesh = trimesh.Trimesh(
            vertices=points, faces=faces, vertex_colors=colors, process=False
        )
        mesh.export(str(out_path))
        print(
            f"[mesh] {image_path.name:50s} -> {out_path.name}  "
            f"{len(points):,} verts, {len(faces):,} tris"
        )
    else:
        cloud = trimesh.PointCloud(vertices=points, colors=colors)
        cloud.export(str(out_path))
        print(
            f"[pc]   {image_path.name:50s} -> {out_path.name}  "
            f"{len(points):,} pts"
        )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, help="Single image path")
    src.add_argument("--folder", type=Path, help="Folder of images (bulk mode)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output .ply path (single) or output directory (folder)")
    p.add_argument("--pointcloud", action="store_true",
                   help="Emit point cloud instead of mesh")
    p.add_argument("--fov", type=float, default=55.0, help="Camera FOV degrees (default 55)")
    p.add_argument("--convention", choices=["disparity", "depth"], default="disparity",
                   help="How to interpret DA2 output (default: disparity)")
    p.add_argument("--disc-thresh", type=float, default=0.05,
                   help="Depth discontinuity threshold for mesh edge culling (default 0.05)")
    p.add_argument("--compute", action="store_true",
                   help="Run DA2 fresh when no cached depth is found")
    args = p.parse_args(argv)

    as_mesh = not args.pointcloud

    if args.image:
        out = args.out
        if out is None:
            out = args.image.parent / "depth_meshes" / (args.image.stem + ".ply")
        process_one(
            args.image, out, as_mesh=as_mesh,
            fov_deg=args.fov, convention=args.convention,
            disc_thresh=args.disc_thresh, compute_if_missing=args.compute,
        )
    else:
        folder = args.folder
        out_dir = args.out or (folder / "depth_meshes")
        out_dir.mkdir(parents=True, exist_ok=True)
        imgs = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            and not p.stem.endswith("__depth")
        )
        print(f"[bulk] {len(imgs)} images; output -> {out_dir}")
        ok = 0
        for img_path in imgs:
            try:
                process_one(
                    img_path, out_dir / (img_path.stem + ".ply"),
                    as_mesh=as_mesh, fov_deg=args.fov, convention=args.convention,
                    disc_thresh=args.disc_thresh, compute_if_missing=args.compute,
                )
                ok += 1
            except Exception as e:  # noqa: BLE001
                print(f"[skip] {img_path.name}: {e}")
        print(f"[done] {ok}/{len(imgs)} exported")


if __name__ == "__main__":
    main()
