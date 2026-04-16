"""Analyze clean similarity targets on real images from /bodytest.

Loads all face images, computes:
1. Per-image ArcFace embedding (bias-corrected)
2. Dataset average embedding
3. Clean cos_sim of each image vs average (the target)
4. Degraded cos_sim at various noise levels (simulating diffusion timesteps)
5. The normalized clean target loss at each noise level

Prints a table so we can verify:
- Clean images hit their targets (loss ~0)
- Noise degrades cos_sim monotonically
- Normalized loss stays in [0,1] and is proportional to degradation
- Profile/unusual shots have lower targets than frontal shots
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms

# Load ArcFace encoder
from toolkit.face_id import DifferentiableFaceEncoder

# insightface for face detection + alignment
from insightface.app import FaceAnalysis

def main():
    device = 'cuda'
    img_dir = '/home/z/Documents/repos/ai-toolkit/bodytest'

    # Load encoder
    print("Loading ArcFace encoder...")
    encoder = DifferentiableFaceEncoder()
    encoder.eval()
    encoder.to(device)

    # Compute bias mean
    print("Computing bias mean from 200 noise samples...")
    with torch.no_grad():
        noise_embeds = []
        for _ in range(200):
            noise_img = torch.randn(1, 3, 112, 112, device=device) * 0.3 + 0.5
            noise_img = noise_img.clamp(0, 1)
            noise_embeds.append(encoder(noise_img))
        noise_embeds = torch.cat(noise_embeds, dim=0)
        bias_mean = noise_embeds.mean(dim=0)
    print(f"Bias mean norm: {bias_mean.norm():.4f}")

    # Load face detector for getting face bboxes
    print("Loading InsightFace detector...")
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Load and encode all images
    to_tensor = transforms.ToTensor()
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.avif'}

    results = []
    print(f"\nProcessing images from {img_dir}...")
    for fname in sorted(os.listdir(img_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in valid_exts:
            continue
        fpath = os.path.join(img_dir, fname)
        try:
            img = Image.open(fpath).convert('RGB')
        except Exception:
            continue

        # Detect face
        img_np = np.array(img)
        faces = app.get(img_np[:, :, ::-1])  # RGB->BGR for insightface
        if not faces:
            continue

        # Use largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = face.bbox.astype(int)

        # Crop face, pad to square, resize to 112x112
        img_t = to_tensor(img).unsqueeze(0).to(device)  # (1, 3, H, W)
        h, w = img_t.shape[2], img_t.shape[3]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        crop = img_t[:, :, y1c:y2c, x1c:x2c]

        # Pad to square
        _, _, ch, cw = crop.shape
        if cw != ch:
            diff = abs(cw - ch)
            if cw > ch:
                pad_top = diff // 2
                pad_bot = diff - pad_top
                crop = F.pad(crop, (0, 0, pad_top, pad_bot), mode='constant', value=0)
            else:
                pad_left = diff // 2
                pad_right = diff - pad_left
                crop = F.pad(crop, (pad_left, pad_right, 0, 0), mode='constant', value=0)

        crop = F.interpolate(crop, size=(112, 112), mode='bilinear', align_corners=False)

        with torch.no_grad():
            emb = encoder(crop)

        results.append({
            'name': fname[:40],
            'embedding': emb,
            'crop': crop,
        })

    print(f"Successfully encoded {len(results)} face images\n")
    if len(results) < 2:
        print("Need at least 2 face images for analysis")
        return

    # Compute dataset average
    all_embeds = torch.cat([r['embedding'] for r in results], dim=0)  # (N, 512)
    avg_embed = all_embeds.mean(dim=0)
    avg_embed = avg_embed / (avg_embed.norm() + 1e-8)

    # Bias-correct average
    avg_c = F.normalize(avg_embed - bias_mean, p=2, dim=-1)

    # Compute clean_cos for each image
    print("=" * 110)
    print(f"{'Image':<42} {'clean_cos':>9} {'target':>7} | ", end="")
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    for nl in noise_levels:
        print(f"σ={nl:<4}", end=" ")
    print()
    print(f"{'':<42} {'':>9} {'':>7} | ", end="")
    for _ in noise_levels:
        print(f"{'cos':>5}", end=" ")
    print()
    print("-" * 110)

    all_clean_cos = []
    all_noise_data = {nl: [] for nl in noise_levels}

    for r in results:
        emb = r['embedding']
        crop = r['crop']
        emb_c = F.normalize(emb - bias_mean.unsqueeze(0), p=2, dim=-1)
        clean_cos = F.cosine_similarity(emb_c, avg_c.unsqueeze(0), dim=-1).item()
        target = max(clean_cos, 0.1)
        all_clean_cos.append(target)

        print(f"{r['name']:<42} {clean_cos:>9.4f} {target:>7.3f} | ", end="")

        for nl in noise_levels:
            if nl == 0:
                cos_val = clean_cos
            else:
                noisy = (crop + torch.randn_like(crop) * nl).clamp(0, 1)
                with torch.no_grad():
                    noisy_emb = encoder(noisy)
                noisy_c = F.normalize(noisy_emb - bias_mean.unsqueeze(0), p=2, dim=-1)
                cos_val = F.cosine_similarity(noisy_c, avg_c.unsqueeze(0), dim=-1).item()
            all_noise_data[nl].append(cos_val)
            print(f"{cos_val:>5.2f}", end=" ")
        print()

    print("-" * 110)

    # Summary statistics
    print(f"\n{'=== Summary Statistics ==='}")
    print(f"  Images:    {len(results)}")
    print(f"  clean_cos: min={min(all_clean_cos):.3f}  max={max(all_clean_cos):.3f}  "
          f"mean={np.mean(all_clean_cos):.3f}  std={np.std(all_clean_cos):.3f}")

    # Show normalized loss at each noise level
    print(f"\n{'=== Normalized Loss: max(0, 1 - cos_sim/clean_cos) ==='}")
    print(f"{'σ':>6} | {'mean_cos':>8} {'mean_loss':>9} {'max_loss':>8} {'pct_at_target':>13}")
    print("-" * 55)
    for nl in noise_levels:
        cos_vals = np.array(all_noise_data[nl])
        targets = np.array(all_clean_cos)
        losses = np.clip(1.0 - cos_vals / targets, 0, None)
        at_target_pct = (losses < 0.01).mean() * 100
        print(f"{nl:>6.2f} | {cos_vals.mean():>8.4f} {losses.mean():>9.4f} {losses.max():>8.4f} {at_target_pct:>12.1f}%")

    # Comparison: what would the old loss look like?
    print(f"\n{'=== Comparison: Old Loss (1 - cos_sim) vs New Normalized Loss ==='}")
    print(f"{'σ':>6} | {'old_mean':>8} {'new_mean':>8} {'old_std':>8} {'new_std':>8} {'old_max':>8} {'new_max':>8}")
    print("-" * 65)
    for nl in noise_levels:
        cos_vals = np.array(all_noise_data[nl])
        targets = np.array(all_clean_cos)
        old_losses = 1.0 - cos_vals
        new_losses = np.clip(1.0 - cos_vals / targets, 0, None)
        print(f"{nl:>6.2f} | {old_losses.mean():>8.4f} {new_losses.mean():>8.4f} "
              f"{old_losses.std():>8.4f} {new_losses.std():>8.4f} "
              f"{old_losses.max():>8.4f} {new_losses.max():>8.4f}")


if __name__ == '__main__':
    main()
