import os
from typing import Optional, List, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file
from tqdm import tqdm

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import FileItemDTO
    from toolkit.config_modules import BodyIDConfig, FaceIDConfig


class BodyIDExtractor:
    """Extracts 10-dim SMPL body shape parameters (betas) using HMR2.

    Uses torchvision FasterRCNN for person detection and HMR2 (4D-Humans)
    for SMPL regression. HMR2 is an optional dependency.
    """

    def __init__(self, detection_threshold: float = 0.5, device_id: int = 0):
        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        self.detection_threshold = detection_threshold

        # Person detector: torchvision FasterRCNN (no extra deps)
        import torchvision
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        )
        self.detector.eval()
        self.detector.to(self.device)
        self.detector.requires_grad_(False)

        # HMR2 model (optional dependency)
        try:
            from hmr2.models import load_hmr2
            from hmr2.utils import recursive_to
            from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
        except ImportError:
            raise ImportError(
                "body_id requires 4D-Humans (HMR2). Install with:\n"
                "  pip install git+https://github.com/shubham-goel/4D-Humans.git\n"
                "See https://github.com/shubham-goel/4D-Humans for details."
            )

        self._recursive_to = recursive_to
        self._ViTDetDataset = ViTDetDataset
        self._DEFAULT_MEAN = DEFAULT_MEAN
        self._DEFAULT_STD = DEFAULT_STD

        # HMR2 checkpoint contains OmegaConf objects that PyTorch 2.6+ blocks.
        # Temporarily allow weights_only=False for this trusted checkpoint.
        _orig_torch_load = torch.load
        torch.load = lambda *args, **kwargs: _orig_torch_load(*args, **{**kwargs, 'weights_only': False})
        try:
            self.hmr2_model, self.hmr2_cfg = load_hmr2()
        finally:
            torch.load = _orig_torch_load
        self.hmr2_model.eval()
        self.hmr2_model.to(self.device)
        self.hmr2_model.requires_grad_(False)

    @torch.no_grad()
    def detect_person(self, pil_image) -> Optional[np.ndarray]:
        """Detect the largest person in a PIL image.

        Returns [x1, y1, x2, y2] bbox or None if no person found.
        """
        from torchvision.transforms import functional as TF
        img_tensor = TF.to_tensor(pil_image).unsqueeze(0).to(self.device)
        outputs = self.detector(img_tensor)[0]

        # Filter for person class (COCO label 1) above threshold
        person_mask = (outputs['labels'] == 1) & (outputs['scores'] >= self.detection_threshold)
        if not person_mask.any():
            return None

        boxes = outputs['boxes'][person_mask].cpu().numpy()
        scores = outputs['scores'][person_mask].cpu().numpy()

        # Return largest person by bbox area
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        largest_idx = areas.argmax()
        return boxes[largest_idx].astype(np.float32)

    @torch.no_grad()
    def extract_from_pil(self, pil_image) -> Optional[np.ndarray]:
        """Extract SMPL beta shape parameters from a PIL Image.

        Returns 10-dim float32 array, or None if no person detected.
        """
        bbox = self.detect_person(pil_image)
        if bbox is None:
            return None

        # Prepare HMR2 input using ViTDetDataset preprocessing
        img_cv2 = np.array(pil_image.convert('RGB'))[:, :, ::-1].copy()  # RGB→BGR
        boxes = np.array([bbox])

        dataset = self._ViTDetDataset(self.hmr2_cfg, img_cv2, boxes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        for batch in dataloader:
            batch = self._recursive_to(batch, self.device)
            out = self.hmr2_model(batch)
            betas = out['pred_smpl_params']['betas'][0].cpu().numpy()  # (10,)
            return betas.astype(np.float32)

        return None


class BodyIDProjector(nn.Module):
    """Projects 10-dim SMPL beta shape parameters to model hidden space tokens.

    Deeper MLP than FaceIDProjector (3 layers vs 2) because 10-dim input
    needs more expansion to reach hidden_size * num_tokens.
    """

    def __init__(self, id_dim: int = 10, hidden_size: int = 4096, num_tokens: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens

        self.norm = nn.LayerNorm(id_dim)
        self.proj = nn.Sequential(
            nn.Linear(id_dim, 64),
            nn.GELU(),
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Linear(256, hidden_size * num_tokens),
        )
        self.output_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 10) SMPL beta tensor
        Returns:
            (B, num_tokens, hidden_size) projected body tokens
        """
        x = self.norm(x)
        x = self.proj(x)
        x = x.reshape(-1, self.num_tokens, self.hidden_size)
        return x * self.output_scale


class DifferentiableBodyProportionEncoder(nn.Module):
    """ViTPose-based encoder for body proportion loss during training.

    Uses ViTPose Base Simple from HuggingFace Transformers with dsntnn for
    differentiable coordinate regression.  Produces 8 pose-invariant
    bone-length ratios that characterise a person's body proportions
    independently of their pose.
    """

    # COCO 17 keypoint indices
    L_SHOULDER = 5;  R_SHOULDER = 6
    L_ELBOW = 7;     R_ELBOW = 8
    L_WRIST = 9;     R_WRIST = 10
    L_HIP = 11;      R_HIP = 12
    L_KNEE = 13;     R_KNEE = 14
    L_ANKLE = 15;    R_ANKLE = 16

    RATIO_DIM = 8  # number of body proportion ratios

    # Bone connections for skeleton drawing: (start_idx, end_idx) using COCO indices
    SKELETON_BONES = [
        (5, 7), (7, 9),      # left arm: shoulder->elbow->wrist
        (6, 8), (8, 10),     # right arm
        (11, 13), (13, 15),  # left leg: hip->knee->ankle
        (12, 14), (14, 16),  # right leg
        (5, 6),              # shoulders
        (11, 12),            # hips
        (5, 11),             # left torso
        (6, 12),             # right torso
    ]

    # ImageNet normalization constants
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # ViTPose input size: (height, width)
    INPUT_SIZE = (256, 192)

    def __init__(self):
        super().__init__()
        from transformers import VitPoseForPoseEstimation, VitPoseImageProcessor
        import dsntnn  # noqa: F401 — verify available

        self.processor = VitPoseImageProcessor.from_pretrained(
            "usyd-community/vitpose-plus-base"
        )
        self.model = VitPoseForPoseEstimation.from_pretrained(
            "usyd-community/vitpose-plus-base",
            torch_dtype=torch.float16,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Register ImageNet normalization buffers (will move with .to())
        self.register_buffer(
            '_img_mean',
            torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            '_img_std',
            torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

        # Last-computed raw keypoints/visibility, populated during forward()
        self._last_keypoints = None   # (B, 17, 2) in [0, 1] normalized
        self._last_visibility = None  # (B, 17) confidence scores

    VIS_THRESHOLD = 0.2  # minimum visibility to trust a keypoint

    @staticmethod
    def _compute_ratios(
        keypoints: torch.Tensor,
        visibilities: torch.Tensor,
        ref_ratios: torch.Tensor = None,
        include_head: bool = False,
    ):
        """Compute pose-invariant bone-length ratios from keypoints.

        Args:
            keypoints: (B, 17, 2) x/y coordinates (2D only, COCO format)
            visibilities: (B, 17) confidence scores in [0, 1]
            ref_ratios: (B, N) or None -- if provided, ratios with low-confidence
                keypoints are replaced with the reference value (zero gradient)
            include_head: if True, add head ratios using keypoints 0,3,4
        Returns:
            ratios: (B, N) body proportion ratios (N=8 body, or 10 with head)
            ratio_vis: (B, N) visibility weights for each ratio
        """
        kp = keypoints
        vis = visibilities
        threshold = DifferentiableBodyProportionEncoder.VIS_THRESHOLD

        def dist(i, j):
            return (kp[:, i] - kp[:, j]).pow(2).sum(-1).clamp(min=1e-6).sqrt()

        def min_vis(*indices):
            return torch.stack([vis[:, i] for i in indices], dim=-1).min(dim=-1).values

        # Bilateral averaged bone lengths (COCO indices)
        upper_arm = (dist(5, 7) + dist(6, 8)) / 2
        forearm = (dist(7, 9) + dist(8, 10)) / 2
        thigh = (dist(11, 13) + dist(12, 14)) / 2
        shin = (dist(13, 15) + dist(14, 16)) / 2

        shoulder_mid = (kp[:, 5] + kp[:, 6]) / 2
        hip_mid = (kp[:, 11] + kp[:, 12]) / 2
        torso = (shoulder_mid - hip_mid).pow(2).sum(-1).clamp(min=1e-6).sqrt()

        shoulder_w = dist(5, 6)
        hip_w = dist(11, 12)

        height = torso + thigh + shin  # pose-invariant height proxy
        height = height.clamp(min=1e-4)

        ratio_list = [
            upper_arm / height,
            forearm / height,
            thigh / height,
            shin / height,
            torso / height,
            shoulder_w / hip_w.clamp(min=1e-4),
            upper_arm / forearm.clamp(min=1e-4),
            thigh / shin.clamp(min=1e-4),
        ]

        vis_list = [
            min_vis(5, 6, 7, 8),             # upper_arm (bilateral avg)
            min_vis(7, 8, 9, 10),            # forearm
            min_vis(11, 12, 13, 14),         # thigh
            min_vis(13, 14, 15, 16),         # shin
            min_vis(5, 6, 11, 12),           # torso
            min_vis(5, 6, 11, 12),           # shoulder/hip ratio
            min_vis(5, 6, 7, 8, 9, 10),     # upper_arm/forearm
            min_vis(11, 12, 13, 14, 15, 16), # thigh/shin
        ]

        if include_head:
            # COCO: 0=nose, 3=left_ear, 4=right_ear
            # Head height: nose to shoulder midpoint, normalized by body height
            head_height = (kp[:, 0] - shoulder_mid).pow(2).sum(-1).clamp(min=1e-6).sqrt()
            ratio_list.append(head_height / height)
            vis_list.append(min_vis(0, 5, 6))  # nose + both shoulders

            # Head width: ear to ear, normalized by shoulder width
            head_width = dist(3, 4)
            ratio_list.append(head_width / shoulder_w.clamp(min=1e-4))
            vis_list.append(min_vis(3, 4, 5, 6))  # both ears + both shoulders

        ratios = torch.stack(ratio_list, dim=-1)  # (B, N)
        ratio_vis = torch.stack(vis_list, dim=-1)  # (B, N)

        # Replace low-confidence ratios with reference values (zero gradient)
        if ref_ratios is not None:
            low_conf = ratio_vis < threshold
            ratios = torch.where(low_conf, ref_ratios.detach(), ratios)
            ratio_vis = torch.where(low_conf, torch.zeros_like(ratio_vis), ratio_vis)

        return ratios, ratio_vis

    NUM_BODY_RATIOS = 8
    NUM_HEAD_RATIOS = 2

    @torch.no_grad()
    def encode(self, pil_image, person_bbox=None, include_head=False):
        """Encode a PIL image to body proportion ratios for caching.

        Uses the HuggingFace processor for preprocessing (handles cropping
        to person_bbox and resizing to ViTPose input size).

        Args:
            pil_image: PIL Image (RGB)
            person_bbox: [x1, y1, x2, y2] person bounding box or None
            include_head: if True, include head keypoint ratios (nose-to-shoulder, ear-to-ear)
        Returns:
            (2*N,) tensor: first N are ratios, last N are visibility weights.
            N=8 (body only) or N=10 (body + head). Returns zeros if no body detected.
        """
        import dsntnn

        img = pil_image.convert('RGB')

        # Prepare boxes for processor: list of list of [x1,y1,x2,y2]
        if person_bbox is not None:
            boxes = [[[float(v) for v in person_bbox]]]
        else:
            # Full image as bbox
            w, h = img.size
            boxes = [[[0.0, 0.0, float(w), float(h)]]]

        inputs = self.processor(images=img, boxes=boxes, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(
            device=next(self.model.parameters()).device,
            dtype=next(self.model.parameters()).dtype,
        )

        outputs = self.model(pixel_values, dataset_index=torch.tensor([0], device=pixel_values.device))
        heatmaps = outputs.heatmaps.float()  # (1, 17, 64, 48)

        # Differentiable coordinates in [-1, 1]
        coords = dsntnn.dsnt(heatmaps)  # (1, 17, 2)
        # Confidence from heatmap peaks
        confidence = heatmaps.flatten(2).max(dim=2).values  # (1, 17)

        ratios, ratio_vis = self._compute_ratios(coords, confidence, include_head=include_head)
        n = ratios.shape[-1]

        if ratio_vis.mean().item() < 0.1:
            return torch.zeros(n * 2)

        return torch.cat([ratios.squeeze(0), ratio_vis.squeeze(0)], dim=0).cpu()  # (2*N,)

    def forward(self, pixels: torch.Tensor, ref_ratios: torch.Tensor = None,
                person_bboxes: list = None, include_head: bool = False):
        """Differentiable forward pass for training.

        Applies the same affine transform as the HF processor (differentiably
        via affine_grid + grid_sample), normalizes with ImageNet stats, and runs
        the model.  Uses dsntnn for differentiable coordinate extraction.

        Args:
            pixels: (B, 3, H, W) in [0, 1] range (RGB, NCHW)
            ref_ratios: (B, 8) optional cached reference ratios
            person_bboxes: ignored (kept for API compatibility, full image is used)
        Returns:
            ratios: (B, 8) body proportion ratios
            ratio_vis: (B, 8) visibility weights
        """
        import dsntnn
        import math as _math
        from transformers.models.vitpose.image_processing_vitpose import box_to_center_and_scale, get_warp_matrix

        pixels = pixels.float()
        B, C, H, W = pixels.shape

        self._last_crop_bboxes = [None] * B

        all_kp = []
        all_vis = []
        with torch.amp.autocast('cuda', enabled=False):
            for i in range(B):
                sample = pixels[i:i+1]  # (1, 3, H, W)
                s_h, s_w = H, W

                # Replicate the HF processor's affine transform differentiably
                # Uses full image as bbox (ViTPose handles full frames well)
                bbox_coco = [0, 0, s_w, s_h]  # COCO format: x, y, w, h
                out_w, out_h = self.INPUT_SIZE[1], self.INPUT_SIZE[0]  # (192, 256)
                center, scale = box_to_center_and_scale(
                    bbox_coco, out_w, out_h, normalize_factor=200.0, padding_factor=1.25
                )
                warp_mat = get_warp_matrix(
                    0, center * 2.0,
                    np.array([out_w - 1, out_h - 1], dtype=np.float32),
                    scale * 200.0
                )

                # Convert warp matrix to PyTorch grid_sample theta
                M = np.vstack([warp_mat, [0, 0, 1]])
                M_inv = np.linalg.inv(M)
                S_in = np.array([[2.0/(s_w-1), 0, -1], [0, 2.0/(s_h-1), -1], [0, 0, 1]])
                S_out_inv = np.array([[(out_w-1)/2.0, 0, (out_w-1)/2.0],
                                      [0, (out_h-1)/2.0, (out_h-1)/2.0],
                                      [0, 0, 1]])
                theta_np = (S_in @ M_inv @ S_out_inv)[:2, :]
                theta = torch.from_numpy(theta_np).float().unsqueeze(0).to(sample.device)

                grid = torch.nn.functional.affine_grid(theta, (1, C, out_h, out_w), align_corners=True)
                sample = torch.nn.functional.grid_sample(sample, grid, align_corners=True, mode='bilinear', padding_mode='zeros')

                # Apply ImageNet normalization differentiably
                mean = self._img_mean.to(sample.device, sample.dtype)
                std = self._img_std.to(sample.device, sample.dtype)
                sample = (sample - mean) / std

                # Run model
                model_dtype = next(self.model.parameters()).dtype
                heatmaps = self.model(sample.to(model_dtype), dataset_index=torch.tensor([0], device=sample.device)).heatmaps.float()

                # Differentiable coordinates in [-1, 1]
                coords = dsntnn.dsnt(heatmaps)  # (1, 17, 2)
                confidence = heatmaps.flatten(2).max(dim=2).values.detach()  # (1, 17) — detach to prevent gradient through peak values (causes green dot artifacts)

                all_kp.append(coords)
                all_vis.append(confidence)

        keypoints = torch.cat(all_kp, dim=0)      # (B, 17, 2) in [-1, 1]
        visibilities = torch.cat(all_vis, dim=0)   # (B, 17)

        self._last_keypoints = ((keypoints.detach() + 1.0) / 2.0)  # (B, 17, 2) in [0, 1] warped space
        self._last_visibility = visibilities.detach()                # (B, 17)

        ratios, ratio_vis = self._compute_ratios(keypoints, visibilities, ref_ratios=ref_ratios,
                                                    include_head=include_head)

        return ratios, ratio_vis


def draw_skeleton_overlay(pil_image, keypoints, visibility):
    """Draw a skeleton overlay on a PIL image.

    Args:
        pil_image: PIL Image (RGB)
        keypoints: (17, 2) numpy array or tensor, normalized [0, 1] coordinates (x, y)
        visibility: (17,) numpy array or tensor, visibility/confidence scores in [0, 1]

    Returns:
        PIL Image with skeleton drawn on it
    """
    from PIL import ImageDraw

    img = pil_image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    if hasattr(keypoints, 'cpu'):
        keypoints = keypoints.cpu().numpy()
    if hasattr(visibility, 'cpu'):
        visibility = visibility.cpu().numpy()

    # Use first 17 keypoints (COCO format)
    n_kp = min(len(keypoints), 17)
    kp = keypoints[:n_kp]
    vis = visibility[:n_kp]

    def vis_color(v):
        if v > 0.5:
            return (0, 255, 0)      # green -- high
        elif v > 0.2:
            return (255, 255, 0)    # yellow -- medium
        else:
            return (255, 0, 0)      # red -- low

    # Draw bones
    for i, j in DifferentiableBodyProportionEncoder.SKELETON_BONES:
        if i >= n_kp or j >= n_kp:
            continue
        min_v = min(vis[i], vis[j])
        color = vis_color(min_v)
        x1, y1 = float(kp[i, 0]) * w, float(kp[i, 1]) * h
        x2, y2 = float(kp[j, 0]) * w, float(kp[j, 1]) * h
        draw.line([(x1, y1), (x2, y2)], fill=color, width=2)

    # Draw keypoints on top of bones
    r = 4
    for idx in range(n_kp):
        color = vis_color(vis[idx])
        x, y = float(kp[idx, 0]) * w, float(kp[idx, 1]) * h
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=color)

    return img


def cache_body_proportion_embeddings(
    file_items: List['FileItemDTO'],
    face_id_config: 'FaceIDConfig',
):
    """Extract and cache body proportion embeddings for all file items.

    Uses the MediaPipe person detector to get person bboxes, then ViTPose
    (via DifferentiableBodyProportionEncoder.encode) for skeleton extraction.
    Caches both body_proportion_embedding (16,) and person_bbox (4,) for
    training-time person cropping.

    Caches to {image_dir}/_face_id_cache/{filename}.safetensors alongside
    existing face embeddings.  Sets file_item.body_proportion_embedding and
    file_item.person_bbox.
    """
    import sys
    import cv2
    from PIL import Image
    from PIL.ImageOps import exif_transpose
    from huggingface_hub import hf_hub_download

    # Download MediaPipe person detector from HuggingFace (for person bbox)
    det_code_path = hf_hub_download('opencv/person_detection_mediapipe', 'mp_persondet.py')
    det_model_path = hf_hub_download('opencv/person_detection_mediapipe', 'person_detection_mediapipe_2023mar.onnx')

    det_dir = os.path.dirname(det_code_path)
    if det_dir not in sys.path:
        sys.path.insert(0, det_dir)

    from mp_persondet import MPPersonDet

    print("Loading MediaPipe person detector + ViTPose...")
    detector = MPPersonDet(det_model_path, scoreThreshold=0.5)

    # Load ViTPose encoder for pose estimation
    encoder = DifferentiableBodyProportionEncoder()
    encoder.eval()

    no_body_count = 0

    include_head = getattr(face_id_config, 'body_proportion_include_head', False)
    # Cache version key: v3 = with head ratios, v2 = body only
    CACHE_VERSION_KEY = 'body_proportion_v3_head' if include_head else 'body_proportion_v2'

    for file_item in tqdm(file_items, desc="Caching body proportion embeddings"):
        img_dir = os.path.dirname(file_item.path)
        cache_dir = os.path.join(img_dir, '_face_id_cache')
        filename_no_ext = os.path.splitext(os.path.basename(file_item.path))[0]
        cache_path = os.path.join(cache_dir, f'{filename_no_ext}.safetensors')

        # Check if cache exists with ViTPose version
        if os.path.exists(cache_path):
            data = load_file(cache_path)
            if 'body_proportion_embedding' in data and 'person_bbox' in data and CACHE_VERSION_KEY in data:
                file_item.body_proportion_embedding = data['body_proportion_embedding'].clone()
                pb = data['person_bbox'].clone()
                file_item.person_bbox = pb if pb.abs().sum() > 0 else None
                continue

        # Need to extract
        pil_image = exif_transpose(Image.open(file_item.path)).convert('RGB')
        cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Detect persons using MediaPipe
        persons = detector.infer(cv_img)

        n_ratios = DifferentiableBodyProportionEncoder.NUM_BODY_RATIOS + (
            DifferentiableBodyProportionEncoder.NUM_HEAD_RATIOS if include_head else 0)
        if len(persons) == 0:
            no_body_count += 1
            prop_tensor = torch.zeros(n_ratios * 2)
            file_item.body_proportion_embedding = prop_tensor
            file_item.person_bbox = None

            os.makedirs(cache_dir, exist_ok=True)
            save_data = {}
            if os.path.exists(cache_path):
                save_data = load_file(cache_path)
            save_data['body_proportion_embedding'] = prop_tensor
            save_data['person_bbox'] = torch.zeros(4)
            save_data[CACHE_VERSION_KEY] = torch.ones(1)
            save_file(save_data, cache_path)
            continue

        # Use highest-confidence person
        best_person = max(persons, key=lambda p: p[-1])

        # Use full image as bbox — ViTPose handles full frames well
        # and tight/ROI crops can miss legs/feet
        img_w, img_h = pil_image.size
        full_img_bbox = [0, 0, img_w, img_h]
        person_bbox_tensor = torch.tensor(full_img_bbox, dtype=torch.float32)

        # Run ViTPose via encode()
        prop_tensor = encoder.encode(pil_image, person_bbox=full_img_bbox, include_head=include_head)

        if prop_tensor.abs().sum() < 1e-6:
            no_body_count += 1

        file_item.body_proportion_embedding = prop_tensor
        file_item.person_bbox = person_bbox_tensor if prop_tensor.abs().sum() > 0 else None

        # Save alongside existing cache data
        os.makedirs(cache_dir, exist_ok=True)
        save_data = {}
        if os.path.exists(cache_path):
            save_data = load_file(cache_path)
        save_data['body_proportion_embedding'] = prop_tensor
        save_data['person_bbox'] = person_bbox_tensor if file_item.person_bbox is not None else torch.zeros(4)
        save_data[CACHE_VERSION_KEY] = torch.ones(1)
        save_file(save_data, cache_path)

    # Free models
    del detector, encoder
    torch.cuda.empty_cache()

    if no_body_count > 0:
        print(f"  -  Warning: no body detected in {no_body_count}/{len(file_items)} images (using zero vector)")


def cache_body_embeddings(
    file_items: List['FileItemDTO'],
    body_id_config: 'BodyIDConfig',
):
    """Extract and cache SMPL body shape parameters for all file items.

    Caches to {image_dir}/_body_id_cache/{filename}.safetensors.
    Sets file_item.body_embedding for each item.
    """
    from PIL import Image
    from PIL.ImageOps import exif_transpose

    extractor = BodyIDExtractor(
        detection_threshold=body_id_config.detection_threshold,
    )
    no_person_count = 0

    for file_item in tqdm(file_items, desc="Caching body shape embeddings"):
        img_dir = os.path.dirname(file_item.path)
        cache_dir = os.path.join(img_dir, '_body_id_cache')
        filename_no_ext = os.path.splitext(os.path.basename(file_item.path))[0]
        cache_path = os.path.join(cache_dir, f'{filename_no_ext}.safetensors')

        if os.path.exists(cache_path):
            data = load_file(cache_path)
            file_item.body_embedding = data['body_embedding']
        else:
            pil_image = exif_transpose(Image.open(file_item.path)).convert('RGB')
            betas = extractor.extract_from_pil(pil_image)

            if betas is None:
                no_person_count += 1
                betas = np.zeros(10, dtype=np.float32)

            tensor = torch.from_numpy(betas)
            file_item.body_embedding = tensor

            os.makedirs(cache_dir, exist_ok=True)
            save_file({'body_embedding': tensor}, cache_path)

    # Free extractor VRAM
    del extractor
    torch.cuda.empty_cache()

    if no_person_count > 0:
        print(f"  -  Warning: no person detected in {no_person_count}/{len(file_items)} images (using zero vector)")
