#!/usr/bin/env python3
"""Test skeleton motion extraction and visualization on a video file.

Usage:
    python scripts/test_skeleton_motion.py --video /path/to/cartwheel.mp4 [--num_frames 121]

Outputs:
    - {video}_skeleton.safetensors: cached keypoint data
    - {video}_skeleton_vis/: per-frame skeleton overlay images
    - {video}_skeleton_vis/motion_summary.png: normalized keypoint trajectory plot
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch

from toolkit.skeleton_motion import (
    cache_video_skeleton,
    normalize_keypoints,
    skeleton_motion_loss,
    skeleton_velocity_loss,
)
from toolkit.body_id import draw_skeleton_overlay
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Test skeleton motion extraction")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--num_frames", type=int, default=None,
                        help="Subsample to N frames (default: all)")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--skip_vis", action="store_true",
                        help="Skip per-frame visualization (just cache + stats)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: video not found: {args.video}")
        return 1

    # 1. Extract and cache keypoints
    print("=" * 60)
    print("Extracting skeleton keypoints...")
    print("=" * 60)
    keypoints, visibility = cache_video_skeleton(
        args.video,
        num_frames=args.num_frames,
        device=args.device,
    )
    T = keypoints.shape[0]
    print(f"\nKeypoints shape: {keypoints.shape}")  # (T, 17, 2) in [0,1]
    print(f"Visibility shape: {visibility.shape}")   # (T, 17)

    # 2. Normalization stats
    norm_kp, valid = normalize_keypoints(keypoints, visibility)
    n_valid = valid.sum().item()
    print(f"\nValid frames (core keypoints visible): {n_valid}/{T} ({100*n_valid/T:.0f}%)")

    kp_names = [
        'nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear',
        'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow',
        'l_wrist', 'r_wrist', 'l_hip', 'r_hip',
        'l_knee', 'r_knee', 'l_ankle', 'r_ankle',
    ]
    print("\nPer-keypoint mean visibility:")
    for i, name in enumerate(kp_names):
        mean_vis = visibility[:, i].mean().item()
        print(f"  {name:>12s}: {mean_vis:.3f}")

    # 3. Sanity checks
    self_loss = skeleton_motion_loss(keypoints, visibility, keypoints, visibility)
    self_vel_loss = skeleton_velocity_loss(keypoints, visibility, keypoints, visibility)
    print(f"\nSelf-loss (should be ~0): position={self_loss.item():.6f}, velocity={self_vel_loss.item():.6f}")

    print("\nNoise sensitivity (L1 loss with increasing keypoint noise):")
    for noise_std in [0.01, 0.05, 0.1, 0.2]:
        noisy_kp = keypoints + torch.randn_like(keypoints) * noise_std
        pos_loss = skeleton_motion_loss(noisy_kp, visibility, keypoints, visibility)
        vel_loss = skeleton_velocity_loss(noisy_kp, visibility, keypoints, visibility)
        print(f"  noise={noise_std:.2f}: position={pos_loss.item():.4f}, velocity={vel_loss.item():.4f}")

    # 4. Per-frame skeleton visualization
    if not args.skip_vis:
        vis_dir = os.path.splitext(args.video)[0] + '_skeleton_vis'
        os.makedirs(vis_dir, exist_ok=True)
        print(f"\nSaving per-frame skeleton overlays to {vis_dir}/")

        cap = cv2.VideoCapture(args.video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if args.num_frames and args.num_frames < total_frames:
            indices = np.linspace(0, total_frames - 1, args.num_frames, dtype=int)
        else:
            indices = np.arange(total_frames)

        for frame_idx, vid_idx in enumerate(indices):
            if frame_idx >= T:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(vid_idx))
            ret, frame = cap.read()
            if not ret:
                continue
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # keypoints are already in [0, 1] image space
            overlay = draw_skeleton_overlay(
                pil_frame, keypoints[frame_idx].numpy(), visibility[frame_idx].numpy())
            overlay.save(os.path.join(vis_dir, f'frame_{frame_idx:04d}.png'))

        cap.release()

        # 5. Motion trajectory summary
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            highlight_kps = {
                'l_wrist': (9, 'tab:blue'),
                'r_wrist': (10, 'tab:cyan'),
                'l_ankle': (15, 'tab:red'),
                'r_ankle': (16, 'tab:orange'),
                'nose': (0, 'tab:green'),
            }

            ax = axes[0]
            for name, (idx, color) in highlight_kps.items():
                ax.plot(norm_kp[:, idx, 0].numpy(), label=name, color=color, alpha=0.8)
            ax.set_title('Normalized X position over time')
            ax.set_xlabel('Frame')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            ax = axes[1]
            for name, (idx, color) in highlight_kps.items():
                ax.plot(norm_kp[:, idx, 1].numpy(), label=name, color=color, alpha=0.8)
            ax.set_title('Normalized Y position over time')
            ax.set_xlabel('Frame')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            ax = axes[2]
            if T > 1:
                vel = (norm_kp[1:] - norm_kp[:-1]).pow(2).sum(-1).sqrt()
                for name, (idx, color) in highlight_kps.items():
                    ax.plot(vel[:, idx].numpy(), label=name, color=color, alpha=0.8)
                ax.set_title('Keypoint velocity over time')
                ax.set_xlabel('Frame')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            summary_path = os.path.join(vis_dir, 'motion_summary.png')
            plt.savefig(summary_path, dpi=150)
            plt.close()
            print(f"Motion summary saved to {summary_path}")

        except ImportError:
            print("(matplotlib not available — skipping trajectory plot)")

    print("\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
