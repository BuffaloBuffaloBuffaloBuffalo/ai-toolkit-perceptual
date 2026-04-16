#!/usr/bin/env python3
"""Predict SMPL body shape (betas) from images and optionally export meshes.

Usage:
    python scripts/predict_body_shape.py path/to/images/
    python scripts/predict_body_shape.py path/to/images/ --mesh       # export .obj meshes
    python scripts/predict_body_shape.py path/to/images/ --render     # render side-view PNGs
    python scripts/predict_body_shape.py img1.jpg img2.jpg --mesh

Mesh export requires:
    pip install smplx trimesh
    Download SMPL model from https://smpl.is.tue.mpg.de/ and set --smpl-dir

Render requires additionally:
    pip install pyrender  (or uses trimesh's built-in renderer)
"""

import argparse
import sys
import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from PIL.ImageOps import exif_transpose

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from toolkit.body_shape import DifferentiableBodyShapeEncoder

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.avif', '.bmp', '.tiff'}


def collect_images(paths):
    """Collect image paths from files and directories."""
    images = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.suffix.lower() in IMAGE_EXTS and not f.name.startswith('.'):
                    images.append(f)
        elif p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images.append(p)
    return images


def export_mesh(betas, output_path, smpl_dir, gender='neutral'):
    """Export SMPL mesh as .obj from beta parameters."""
    import smplx
    import trimesh

    model = smplx.create(
        smpl_dir, model_type='smpl', gender=gender,
        num_betas=10, batch_size=1,
    )
    betas_tensor = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)
    # Neutral pose (T-pose)
    body_pose = torch.zeros(1, 69)
    output = model(betas=betas_tensor, body_pose=body_pose)
    vertices = output.vertices.detach().numpy()[0]
    faces = model.faces

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(str(output_path))
    return mesh


def render_mesh(mesh, output_path, width=512, height=768, reference_mesh=None):
    """Render a mesh to an image from front and side views.

    Uses FIXED camera position (not scaled to bounding box) so that body size
    differences are visible. Applies proper 3-point lighting to reveal surface
    curvature. Optionally colors vertices by displacement from a reference mesh.

    Args:
        mesh: trimesh.Trimesh object
        output_path: path to save the PNG
        width: width of each view panel
        height: height of each view panel
        reference_mesh: if provided, color vertices by displacement from this mesh
    """
    import trimesh

    try:
        import pyrender
        from pyrender import (
            Mesh as PRMesh, Scene, PerspectiveCamera,
            DirectionalLight, SpotLight, OffscreenRenderer,
        )
    except ImportError:
        # Fallback: save mesh screenshot via trimesh
        try:
            png = mesh.scene().save_image(resolution=(width, height))
            with open(str(output_path), 'wb') as f:
                f.write(png)
            return True
        except Exception as e:
            print(f"  Render failed ({e}). Install pyrender for rendering.")
            return False

    # --- Vertex coloring by displacement from reference ---
    if reference_mesh is not None and reference_mesh.vertices.shape == mesh.vertices.shape:
        disp = np.linalg.norm(mesh.vertices - reference_mesh.vertices, axis=1)
        # Normalize: 0 = no change (blue), max = max change (red)
        max_disp = max(disp.max(), 1e-6)
        # Clamp at a reasonable scale (3cm is a lot for SMPL body shape)
        disp_norm = np.clip(disp / 0.03, 0, 1)
        # Blue -> Cyan -> Green -> Yellow -> Red colormap
        colors = np.zeros((len(disp_norm), 4), dtype=np.uint8)
        for i, d in enumerate(disp_norm):
            if d < 0.25:
                t = d / 0.25
                colors[i] = [0, int(255 * t), 255, 255]  # blue -> cyan
            elif d < 0.5:
                t = (d - 0.25) / 0.25
                colors[i] = [0, 255, int(255 * (1 - t)), 255]  # cyan -> green
            elif d < 0.75:
                t = (d - 0.5) / 0.25
                colors[i] = [int(255 * t), 255, 0, 255]  # green -> yellow
            else:
                t = (d - 0.75) / 0.25
                colors[i] = [255, int(255 * (1 - t)), 0, 255]  # yellow -> red
        render_mesh_obj = trimesh.Trimesh(
            vertices=mesh.vertices, faces=mesh.faces,
            vertex_colors=colors, process=False,
        )
    else:
        # Skin-tone coloring instead of flat gray
        skin_color = [220, 185, 155, 255]
        render_mesh_obj = trimesh.Trimesh(
            vertices=mesh.vertices, faces=mesh.faces,
            vertex_colors=np.tile(skin_color, (len(mesh.vertices), 1)),
            process=False,
        )

    pr_mesh = PRMesh.from_trimesh(render_mesh_obj, smooth=True)

    # --- FIXED camera parameters (same for all subjects) ---
    # Fixed position ensures body SIZE differences are visible across renders.
    # SMPL T-pose body center is at roughly (0, -0.29, 0).
    FIXED_CAM_Y = -0.29        # vertical center of a typical SMPL T-pose body
    FIXED_CAM_DIST_FRONT = 3.0  # fixed distance for front view
    FIXED_CAM_DIST_SIDE = 3.0   # fixed distance for side view
    YFOV = np.pi / 4.0          # 45 degrees — wide enough to fit T-pose arms

    def build_scene_and_render(renderer, cam_pose):
        """Build a pyrender scene with 3-point lighting and render."""
        sc = Scene(ambient_light=[0.15, 0.15, 0.17], bg_color=[255, 255, 255, 255])
        sc.add(pr_mesh)

        camera = PerspectiveCamera(yfov=YFOV, znear=0.1, zfar=20.0)
        sc.add(camera, pose=cam_pose)

        # Key light (warm, from upper-right-front)
        key_pose = np.eye(4)
        key_pose[:3, 3] = [1.5, 1.0, 3.0]
        sc.add(DirectionalLight(color=[1.0, 0.95, 0.9], intensity=4.0), pose=key_pose)

        # Fill light (cool, from left)
        fill_pose = np.eye(4)
        fill_pose[:3, 3] = [-2.0, 0.0, 2.0]
        sc.add(DirectionalLight(color=[0.85, 0.9, 1.0], intensity=2.0), pose=fill_pose)

        # Rim/back light (from behind, slightly above)
        rim_pose = np.eye(4)
        rim_pose[:3, 3] = [0.0, 1.0, -3.0]
        sc.add(DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.5), pose=rim_pose)

        color, _ = renderer.render(sc)
        return color

    r = OffscreenRenderer(width, height)

    # Front view: camera on +Z axis looking at body center
    cam_front = np.eye(4)
    cam_front[:3, 3] = [0.0, FIXED_CAM_Y, FIXED_CAM_DIST_FRONT]
    color_front = build_scene_and_render(r, cam_front)

    # Side view: camera on +X axis, rotated 90 degrees to look at body
    cam_side = np.eye(4)
    cam_side[:3, 3] = [FIXED_CAM_DIST_SIDE, FIXED_CAM_Y, 0.0]
    cam_side[:3, :3] = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0],
    ], dtype=float)
    color_side = build_scene_and_render(r, cam_side)

    r.delete()

    # Combine front + side with a thin separator line
    separator = np.full((height, 2, 3), 200, dtype=np.uint8)
    combined = np.concatenate([color_front, separator, color_side], axis=1)

    # Add label if the name is extractable
    stem = Path(output_path).stem
    label = stem.split('_vid')[0].replace('_', ' ').title() if '_vid' in stem else stem

    # Draw label using PIL
    img = Image.fromarray(combined)
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except (IOError, OSError):
            font = ImageFont.load_default()
        draw.text((10, 10), label, fill=(40, 40, 40), font=font)

        # Add scale bar (10cm = how many pixels?)
        # At FIXED_CAM_DIST with YFOV, visible height = 2 * dist * tan(yfov/2)
        import math
        visible_h = 2 * FIXED_CAM_DIST_FRONT * math.tan(YFOV / 2)
        px_per_meter = height / visible_h
        bar_px = int(0.10 * px_per_meter)  # 10cm bar
        bar_y = height - 30
        bar_x = 10
        draw.rectangle([bar_x, bar_y, bar_x + bar_px, bar_y + 4], fill=(40, 40, 40))
        draw.text((bar_x, bar_y - 18), "10 cm", fill=(40, 40, 40), font=font)
    except ImportError:
        pass

    img.save(str(output_path))
    return True


def main():
    parser = argparse.ArgumentParser(description='Predict body shape from images')
    parser.add_argument('inputs', nargs='+', help='Image files or directories')
    parser.add_argument('--mesh', action='store_true', help='Export .obj meshes')
    parser.add_argument('--render', action='store_true', help='Render preview PNGs')
    parser.add_argument('--smpl-dir', type=str, default=os.path.expanduser('~/smpl_models'),
                        help='Directory containing SMPL model files')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Output directory (default: {input_dir}/body_shape_output/)')
    parser.add_argument('--gender', type=str, default='neutral', choices=['neutral', 'male', 'female'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    images = collect_images(args.inputs)
    if not images:
        print("No images found.")
        return

    print(f"Found {len(images)} images")

    # Output dir
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = images[0].parent / 'body_shape_output'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading HybrIK body shape encoder...")
    encoder = DifferentiableBodyShapeEncoder()
    encoder.to(args.device)

    # Process images
    all_results = []
    for img_path in images:
        pil = exif_transpose(Image.open(img_path)).convert('RGB')
        betas = encoder.encode(pil)
        betas_np = betas.numpy()
        all_results.append((img_path, betas_np))

        beta_str = ' '.join(f'{b:+.3f}' for b in betas_np)
        print(f"  {img_path.name:50s}  betas=[{beta_str}]")

    # Save betas as numpy
    names = [r[0].name for r in all_results]
    betas_array = np.stack([r[1] for r in all_results])
    np.savez(out_dir / 'betas.npz', fnames=names, betas=betas_array)
    print(f"\nSaved betas to {out_dir / 'betas.npz'} ({len(all_results)} images)")

    # Pairwise similarity matrix
    if len(all_results) > 1:
        from torch.nn.functional import cosine_similarity
        bt = torch.from_numpy(betas_array)
        print(f"\nPairwise cosine similarity:")
        header = "              " + "".join(f"{n[:8]:>9s}" for n in names)
        print(header)
        for i in range(len(names)):
            row = f"{names[i][:14]:14s}"
            for j in range(len(names)):
                cos = cosine_similarity(bt[i:i+1], bt[j:j+1]).item()
                row += f"  {cos:.4f}"
            print(row)

        print(f"\nPairwise L1 distance:")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                l1 = np.abs(betas_array[i] - betas_array[j]).mean()
                print(f"  {names[i][:25]:25s} vs {names[j][:25]:25s}  L1={l1:.4f}")

    # Export meshes
    if args.mesh:
        smpl_dir = Path(args.smpl_dir)
        if not smpl_dir.exists():
            print(f"\nSMPL model directory not found: {smpl_dir}")
            print(f"Download SMPL from https://smpl.is.tue.mpg.de/")
            print(f"Extract and pass --smpl-dir /path/to/smpl/models")
            print(f"Expected structure: {smpl_dir}/smpl/SMPL_NEUTRAL.pkl")
        else:
            try:
                import smplx, trimesh
            except ImportError:
                print("\nInstall mesh dependencies: pip install smplx trimesh")
                return

            mesh_dir = out_dir / 'meshes'
            mesh_dir.mkdir(exist_ok=True)
            print(f"\nExporting meshes to {mesh_dir}/")

            # Build all meshes first, then compute mean for displacement viz
            all_meshes = []
            for img_path, betas_np in all_results:
                stem = img_path.stem
                obj_path = mesh_dir / f'{stem}.obj'
                mesh = export_mesh(betas_np, obj_path, str(smpl_dir), args.gender)
                all_meshes.append(mesh)
                print(f"  {obj_path.name}  ({mesh.vertices.shape[0]} vertices)")

            # Compute mean mesh as reference for displacement coloring
            ref_mesh = None
            if args.render and len(all_meshes) > 1:
                mean_verts = np.mean([m.vertices for m in all_meshes], axis=0)
                ref_mesh = trimesh.Trimesh(
                    vertices=mean_verts, faces=all_meshes[0].faces, process=False,
                )
                print(f"  (using mean mesh as reference for displacement coloring)")

            if args.render:
                for (img_path, _), mesh in zip(all_results, all_meshes):
                    png_path = mesh_dir / f'{img_path.stem}.png'
                    render_mesh(mesh, png_path, reference_mesh=ref_mesh)
                    print(f"  {png_path.name}")

    print("\nDone.")


if __name__ == '__main__':
    main()
