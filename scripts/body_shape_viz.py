"""Body-shape visualization prototype.

Runs a body-shape regressor on a directory of images and renders the predicted
SMPL mesh from multiple viewpoints so you can eyeball how accurate the body
shape recovery is.

Current regressor: HMR2 / 4D-Humans (already installed in the repo venv).
SMPL output: 10 betas (the current repo baseline).

Per image, writes:
    <out>/<stem>/original.png              # input
    <out>/<stem>/overlay.png               # SMPL mesh projected onto the image
    <out>/<stem>/three_view.png            # front | side | back standalone renders
    <out>/<stem>/tile.png                  # original | overlay | three_view
    <out>/<stem>/params.json               # betas, pose, measurements
And an HTML index at <out>/index.html.

Usage:
    python scripts/body_shape_viz.py --data-dir test_data/scarlett_full \\
        --out test_data/scarlett_full/body_shape_preview

Environment:
    Set PYOPENGL_PLATFORM=egl before importing pyrender on headless machines.
"""
from __future__ import annotations

import argparse
import json
import os
# Must be set BEFORE any pyrender / PyOpenGL import
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh
import pyrender
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageOps import exif_transpose
from tqdm import tqdm


# ============================================================
# HMR2 loader (mirrors toolkit/body_id.py:52 pattern)
# ============================================================


def load_hmr2_safely():
    """Load HMR2 model + cfg past the torch.load weights_only=True default."""
    from hmr2.models import load_hmr2
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
    try:
        model, cfg = load_hmr2()
    finally:
        torch.load = _orig
    return model, cfg


# ============================================================
# Person detection (same as body_id.py)
# ============================================================


def load_person_detector(device, threshold: float = 0.5):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    m = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT').to(device).eval()
    return m, threshold


@torch.no_grad()
def detect_person(detector, threshold: float, pil_image, device) -> Optional[np.ndarray]:
    from torchvision.transforms import functional as TF
    t = TF.to_tensor(pil_image).unsqueeze(0).to(device)
    out = detector(t)[0]
    mask = (out['labels'] == 1) & (out['scores'] >= threshold)
    if not mask.any():
        return None
    boxes = out['boxes'][mask].cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return boxes[int(areas.argmax())].astype(np.float32)


# ============================================================
# HMR2 inference
# ============================================================


@torch.no_grad()
def run_hmr2(model, cfg, pil_image, bbox, device):
    """Run HMR2 for one image + bbox. Returns dict with vertices, faces, betas,
    the weak-persp pred_cam + per-batch box info (all needed to call
    ``cam_crop_to_full`` for full-image rendering)."""
    from hmr2.utils import recursive_to
    from hmr2.datasets.vitdet_dataset import ViTDetDataset
    from hmr2.utils.renderer import cam_crop_to_full
    img_cv2 = np.array(pil_image.convert('RGB'))[:, :, ::-1].copy()  # RGB->BGR
    boxes = np.array([bbox])
    ds = ViTDetDataset(cfg, img_cv2, boxes)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    for batch in loader:
        batch = recursive_to(batch, device)
        out = model(batch)
        vertices = out['pred_vertices'][0].cpu().numpy()          # (6890, 3)
        betas = out['pred_smpl_params']['betas'][0].cpu().numpy()
        # Full-image camera translation via HMR2's helper
        pred_cam = out['pred_cam'].float()
        box_center = batch['box_center'].float()
        box_size = batch['box_size'].float()
        img_size = batch['img_size'].float()
        scaled_focal = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * img_size.max()
        cam_t_full = cam_crop_to_full(
            pred_cam, box_center, box_size, img_size, scaled_focal
        )[0].cpu().numpy()
        return dict(
            vertices=vertices,
            faces=model.smpl.faces,
            betas=betas,
            cam_t_full=cam_t_full,
            scaled_focal=float(scaled_focal.item()),
        )
    return None


# ============================================================
# Mesh rendering
# ============================================================


def _create_raymond_lights():
    """Copy of hmr2.utils.renderer.create_raymond_lights (inlined to avoid import chain)."""
    thetas = np.pi * np.array([1.0 / 6.0] * 3)
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
    nodes = []
    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)
        z = np.array([xp, yp, zp]); z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix,
        ))
    return nodes


def _build_trimesh(vertices, faces, extra_rotation=None, color=(0.85, 0.88, 1.0)):
    """Build a trimesh with SMPL's required 180° X-flip applied (mirrors HMR2)."""
    mesh = trimesh.Trimesh(vertices.copy(), faces.copy(), process=False)
    if extra_rotation is not None:
        mesh.apply_transform(extra_rotation)
    # SMPL canonical is Y-down in camera frame; flip
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0]))
    vc = np.tile(np.array((*color, 1.0), dtype=np.float32), (len(mesh.vertices), 1))
    mesh.visual.vertex_colors = (vc * 255).astype(np.uint8)
    return mesh


def render_overlay(image_np: np.ndarray, hmr2_out: Dict, alpha=0.7) -> np.ndarray:
    """Project mesh onto original image using HMR2's full-image camera.

    Mirrors hmr2.utils.renderer.Renderer.__call__ exactly.
    """
    img_h, img_w = image_np.shape[:2]
    cam_t = hmr2_out['cam_t_full'].copy()
    cam_t[0] *= -1.0  # HMR2 convention

    mesh_t = _build_trimesh(hmr2_out['vertices'], hmr2_out['faces'])
    mesh = pyrender.Mesh.from_trimesh(mesh_t)

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = cam_t
    cam = pyrender.IntrinsicsCamera(
        fx=hmr2_out['scaled_focal'], fy=hmr2_out['scaled_focal'],
        cx=img_w / 2.0, cy=img_h / 2.0, zfar=1e12,
    )
    scene.add(cam, pose=cam_pose)
    for node in _create_raymond_lights():
        scene.add_node(node)

    r = pyrender.OffscreenRenderer(int(img_w), int(img_h), point_size=1.0)
    try:
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        r.delete()

    valid = (color[..., 3:4].astype(np.float32) / 255.0)
    a = valid * alpha
    out = image_np.astype(np.float32) * (1.0 - a) + color[..., :3].astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def render_three_view(vertices, faces, size: int = 400) -> np.ndarray:
    """Front | side | back standalone renders with tight orthographic framing
    (same across all three views so body proportions look consistent)."""
    # Compute a single bounding box from the canonical (front-view) mesh so
    # every view uses identical framing — side and back don't zoom differently.
    canonical = _build_trimesh(vertices, faces)
    V0 = np.asarray(canonical.vertices)
    center_y = (V0[:, 1].max() + V0[:, 1].min()) / 2.0
    half_height = (V0[:, 1].max() - V0[:, 1].min()) / 2.0
    # Slight horizontal padding, keyed off height so aspect stays 1:1
    mag = float(half_height) * 1.05

    views = [('front', 0), ('side', 90), ('back', 180)]
    panels = []
    for name, yaw_deg in views:
        rot = trimesh.transformations.rotation_matrix(np.radians(yaw_deg), [0, 1, 0])
        mesh_t = _build_trimesh(vertices, faces, extra_rotation=rot)

        scene = pyrender.Scene(bg_color=[30, 30, 35, 255], ambient_light=(0.4, 0.4, 0.4))
        scene.add(pyrender.Mesh.from_trimesh(mesh_t))
        cam_pose = np.eye(4)
        cam_pose[:3, 3] = [0.0, center_y, 3.0]  # +Z in front of mesh, Y centered
        cam = pyrender.OrthographicCamera(xmag=mag, ymag=mag, znear=0.01, zfar=50.0)
        scene.add(cam, pose=cam_pose)
        for node in _create_raymond_lights():
            scene.add_node(node)

        r = pyrender.OffscreenRenderer(size, size)
        try:
            color, _ = r.render(scene)
        finally:
            r.delete()
        panels.append((name, color))

    pad = 4
    label_h = 24
    canvas = np.full((size + label_h, size * len(panels) + pad * (len(panels) - 1), 3), 20, dtype=np.uint8)
    for i, (_, img) in enumerate(panels):
        x = i * (size + pad)
        canvas[label_h:, x:x + size] = img
    canvas_pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(canvas_pil)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
    except Exception:
        font = ImageFont.load_default()
    for i, (name, _) in enumerate(panels):
        x = i * (size + pad)
        draw.text((x + 6, 4), name, fill=(230, 230, 230), font=font)
    return np.array(canvas_pil)


def tile_horiz(imgs: List[np.ndarray], labels: List[str], col_w: int = 400) -> Image.Image:
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
    except Exception:
        font = ImageFont.load_default()
    resized = []
    for a in imgs:
        if a.shape[1] == col_w:
            resized.append(a)
        else:
            ratio = col_w / a.shape[1]
            h = int(a.shape[0] * ratio)
            resized.append(np.array(Image.fromarray(a).resize((col_w, h), Image.BILINEAR)))
    h_max = max(a.shape[0] for a in resized)
    label_h = 24
    canvas = Image.new('RGB', (sum(a.shape[1] for a in resized) + 6 * (len(resized) - 1), h_max + label_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for a, lbl in zip(resized, labels):
        canvas.paste(Image.fromarray(a), (x, label_h))
        draw.text((x + 6, 4), lbl, fill=(230, 230, 230), font=font)
        x += a.shape[1] + 6
    return canvas


# ============================================================
# Main
# ============================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-images', type=int, default=None)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    exts = ('.jpg', '.jpeg', '.png', '.webp', '.avif')
    images = sorted([p for p in args.data_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in exts])
    if args.max_images:
        images = images[:args.max_images]
    print(f"Found {len(images)} images")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'per_image').mkdir(exist_ok=True)

    print("Loading FasterRCNN person detector...")
    detector, threshold = load_person_detector(args.device)
    print("Loading HMR2...")
    hmr2_model, hmr2_cfg = load_hmr2_safely()
    hmr2_model.eval().to(args.device).requires_grad_(False)

    records = []
    fail_no_person = 0
    fail_hmr2 = 0

    for img_path in tqdm(images, desc='body-viz'):
        pil = exif_transpose(Image.open(img_path)).convert('RGB')
        image_np = np.array(pil)
        stem = img_path.stem
        sub = args.out / 'per_image' / stem
        sub.mkdir(exist_ok=True)

        # Save original for reference
        Image.fromarray(image_np).save(sub / 'original.png')

        bbox = detect_person(detector, threshold, pil, args.device)
        if bbox is None:
            fail_no_person += 1
            records.append({'stem': stem, 'status': 'no_person_detected'})
            continue

        t0 = time.perf_counter()
        try:
            out = run_hmr2(hmr2_model, hmr2_cfg, pil, bbox, args.device)
        except Exception as e:
            fail_hmr2 += 1
            records.append({'stem': stem, 'status': f'hmr2_error: {e}'})
            continue
        hmr2_ms = (time.perf_counter() - t0) * 1000

        # Render
        overlay = render_overlay(image_np, out, alpha=0.7)
        three_view = render_three_view(out['vertices'], out['faces'], size=420)

        # Tile: row 1 = [Original | Mesh overlay] (matched height)
        #       row 2 = three_view at full width (1260×444 native)
        row1 = tile_horiz([image_np, overlay], ['Original', 'Mesh overlay'], col_w=420)
        # three_view is 3*420 + 2*4 = 1268 wide × 444 tall
        row1_np = np.array(row1)
        # Pad row1 to match three_view width if needed
        target_w = three_view.shape[1]
        if row1_np.shape[1] < target_w:
            pad = np.full((row1_np.shape[0], target_w - row1_np.shape[1], 3), 20, dtype=np.uint8)
            row1_np = np.concatenate([row1_np, pad], axis=1)
        elif row1_np.shape[1] > target_w:
            # Pad three_view instead
            pad_w = row1_np.shape[1] - target_w
            pad = np.full((three_view.shape[0], pad_w, 3), 20, dtype=np.uint8)
            three_view = np.concatenate([three_view, pad], axis=1)

        tile_np = np.concatenate([row1_np, three_view], axis=0)
        tile = Image.fromarray(tile_np)

        Image.fromarray(overlay).save(sub / 'overlay.png')
        Image.fromarray(three_view).save(sub / 'three_view.png')
        tile.save(sub / 'tile.png')

        params = {
            'stem': stem,
            'status': 'ok',
            'bbox': bbox.tolist(),
            'betas': out['betas'].tolist(),
            'hmr2_ms': round(hmr2_ms, 2),
        }
        (sub / 'params.json').write_text(json.dumps(params, indent=2))
        records.append(params)

    # Save metrics + HTML index
    (args.out / 'metrics.json').write_text(json.dumps({
        'n_images': len(records),
        'n_ok': sum(1 for r in records if r.get('status') == 'ok'),
        'n_no_person': fail_no_person,
        'n_hmr2_error': fail_hmr2,
        'records': records,
    }, indent=2))

    html = ['<!doctype html><html><head><meta charset="utf-8"><title>Body-Shape Preview</title>',
            '<style>body{font-family:monospace;background:#111;color:#ddd;padding:16px;max-width:1800px;margin:0 auto;}',
            'img{max-width:100%;margin:6px 0;border:1px solid #333;} h2{color:#6cf;margin:24px 0 2px;font-size:15px;}',
            '.row{margin-bottom:18px;padding-bottom:6px;border-bottom:1px solid #333;}',
            '.stats{color:#999;font-size:12px;}</style></head><body>',
            '<h1>HMR2 Body-Shape Preview — scarlett_full</h1>',
            f'<p>{sum(1 for r in records if r.get("status")=="ok")}/{len(records)} images processed</p>',
            '<p>Left = original · Middle = SMPL mesh projected onto image · Right = Front | Side | Back standalone renders</p>']
    for r in sorted(records, key=lambda x: x['stem']):
        if r.get('status') != 'ok':
            continue
        html.append(f'<div class="row"><h2>{r["stem"]}</h2>'
                    f'<div class="stats">hmr2 {r["hmr2_ms"]} ms · betas {[round(b,2) for b in r["betas"]]}</div>'
                    f'<img src="per_image/{r["stem"]}/tile.png"></div>')
    html.append('</body></html>')
    (args.out / 'index.html').write_text('\n'.join(html))

    print(f"\nDone: {args.out / 'index.html'}")
    print(f"  ok: {sum(1 for r in records if r.get('status')=='ok')}/{len(records)}")


if __name__ == '__main__':
    main()
