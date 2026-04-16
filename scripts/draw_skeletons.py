#!/usr/bin/env python3
"""Draw skeleton overlays on all images in a directory (recursive).

Uses the MediaPipe person detector for bounding boxes and ViTPose for
COCO-format 17-keypoint pose estimation.

Usage:
    python scripts/draw_skeletons.py /path/to/images
    python scripts/draw_skeletons.py /path/to/images --threshold 0.3

Output goes to a 'skeletons' subdirectory in each folder containing images.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.body_id import DifferentiableBodyProportionEncoder, draw_skeleton_overlay

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}


def main():
    parser = argparse.ArgumentParser(description='Draw skeleton overlays on images')
    parser.add_argument('directory', help='Directory to scan for images (recursive)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Person detection confidence threshold (default: 0.5)')
    args = parser.parse_args()

    root = Path(args.directory)
    if not root.is_dir():
        print(f"Error: {root} is not a directory")
        sys.exit(1)

    # Find all images recursively, excluding skeleton output dirs
    images = []
    for ext in IMAGE_EXTENSIONS:
        for p in root.rglob(f'*{ext}'):
            if 'skeletons' not in p.parts:
                images.append(p)
    images.sort()

    if not images:
        print(f"No images found in {root}")
        sys.exit(0)

    print(f"Found {len(images)} images in {root}")

    # Load MediaPipe person detector for bounding boxes
    from huggingface_hub import hf_hub_download

    det_code_path = hf_hub_download('opencv/person_detection_mediapipe', 'mp_persondet.py')
    det_model_path = hf_hub_download('opencv/person_detection_mediapipe', 'person_detection_mediapipe_2023mar.onnx')

    sys.path.insert(0, os.path.dirname(det_code_path))

    from mp_persondet import MPPersonDet

    print("Loading person detector + ViTPose...")
    detector = MPPersonDet(det_model_path, scoreThreshold=args.threshold)

    # Load ViTPose encoder
    encoder = DifferentiableBodyProportionEncoder()
    encoder.eval()

    no_person_count = 0

    for img_path in tqdm(images, desc="Drawing skeletons"):
        try:
            pil_img = exif_transpose(Image.open(img_path)).convert('RGB')
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # Detect persons
            persons = detector.infer(cv_img)
            if len(persons) == 0:
                no_person_count += 1
                continue

            # Use the highest-confidence person
            best_person = max(persons, key=lambda p: p[-1])

            # Use full image as bbox — ViTPose handles full frames well
            # and tight crops can miss legs/feet
            w, h = pil_img.size
            person_bbox = [0, 0, w, h]

            w, h = pil_img.size
            boxes = [[[float(v) for v in person_bbox]]]
            inputs = encoder.processor(images=pil_img, boxes=boxes, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(
                device=next(encoder.model.parameters()).device,
                dtype=next(encoder.model.parameters()).dtype,
            )

            with torch.no_grad():
                outputs = encoder.model(pixel_values=pixel_values, dataset_index=torch.tensor([0], device=pixel_values.device))
                # Cast to float32 for post-processing (fp16 not supported by numpy)
                outputs.heatmaps = outputs.heatmaps.float()
                pose_results = encoder.processor.post_process_pose_estimation(
                    outputs, boxes=boxes
                )[0]

            if not pose_results:
                no_person_count += 1
                continue

            kp = pose_results[0]['keypoints']    # (17, 2) in original image pixels
            scores = pose_results[0]['scores']   # (17,)

            # Normalize to [0, 1] for draw_skeleton_overlay
            kp_normalized = torch.zeros(17, 2)
            kp_normalized[:, 0] = kp[:, 0] / w
            kp_normalized[:, 1] = kp[:, 1] / h
            vis = scores

            skeleton_img = draw_skeleton_overlay(pil_img, kp_normalized, vis)

            # Save to skeletons subdirectory
            out_dir = img_path.parent / 'skeletons'
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / f'{img_path.stem}_skeleton{img_path.suffix}'
            skeleton_img.save(out_path)

        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")

    print(f"Done! ({no_person_count} no person)")


if __name__ == '__main__':
    main()
