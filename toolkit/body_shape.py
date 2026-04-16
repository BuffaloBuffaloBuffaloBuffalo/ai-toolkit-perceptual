import os
from typing import Optional, List, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
from tqdm import tqdm
import torchvision.models as models

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import FileItemDTO
    from toolkit.config_modules import FaceIDConfig


class DifferentiableBodyShapeEncoder(nn.Module):
    """HybrIK-based encoder for body shape loss during training.

    Uses HybrIK's ResNet-34 backbone with global average pooling to predict
    10-dim SMPL beta parameters (pose-invariant body shape).  Only the beta
    prediction head is loaded — the SMPL mesh reconstruction layer is not
    needed, eliminating the SMPL .pkl dependency entirely.

    Architecture: ResNet-34 → AdaptiveAvgPool2d → FC(512→1024) → FC(1024→1024)
    → FC(1024→10) + init_shape bias.

    Global avg pool spreads gradients evenly across all spatial positions
    (same as ArcFace), avoiding spatial concentration artifacts.
    """

    BETA_DIM = 10
    INPUT_SIZE = 256  # HybrIK expects square 256x256 input

    def __init__(self):
        super().__init__()

        # ResNet-34 backbone (matches HybrIK preact.* keys)
        resnet = models.resnet34(weights=None)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # FC shape prediction head
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.decshape = nn.Linear(1024, 10)
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)

        # Mean shape from HybrIK training data (additive bias)
        self.register_buffer('init_shape', torch.zeros(1, 10))

        # HybrIK normalization — applied to RGB data (verified from
        # simple_transform_3d_smpl_cam.py test_transform: img[0] is R)
        self.register_buffer(
            'img_mean', torch.tensor([0.406, 0.457, 0.480]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'img_std', torch.tensor([0.225, 0.224, 0.229]).view(1, 3, 1, 1)
        )

        # Load pretrained weights
        self._load_pretrained()

        # Freeze all parameters
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def _load_pretrained(self):
        """Download and load HybrIK ResNet-34 weights."""
        # Search common locations for the ResNet-34 checkpoint
        search_paths = [
            os.path.expanduser('~/.cache/hybrik/hybrik_resnet34.pth'),
            '/tmp/hybrik_resnet34.pth',
        ]
        ckpt_path = None
        for p in search_paths:
            if os.path.exists(p):
                ckpt_path = p
                break

        if ckpt_path is None:
            # Try downloading via gdown (Google Drive)
            cache_dir = os.path.expanduser('~/.cache/hybrik')
            os.makedirs(cache_dir, exist_ok=True)
            ckpt_path = os.path.join(cache_dir, 'hybrik_resnet34.pth')
            try:
                import gdown
                gdown.download(id='19ktHbERz0Un5EzJYZBdzdzTrFyd9gLCx', output=ckpt_path, quiet=False)
            except Exception as e:
                raise FileNotFoundError(
                    f"HybrIK ResNet-34 weights not found. Download from Google Drive "
                    f"(ID: 19ktHbERz0Un5EzJYZBdzdzTrFyd9gLCx) to {ckpt_path}. "
                    f"Install gdown for automatic download: pip install gdown"
                ) from e

        sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        # Map HybrIK key names to our module names
        key_map = {
            'preact.conv1.weight': 'conv1.weight',
            'preact.bn1.weight': 'bn1.weight',
            'preact.bn1.bias': 'bn1.bias',
            'preact.bn1.running_mean': 'bn1.running_mean',
            'preact.bn1.running_var': 'bn1.running_var',
            'preact.bn1.num_batches_tracked': 'bn1.num_batches_tracked',
        }

        new_sd = {}
        for old_key, tensor in sd.items():
            # Skip SMPL, deconv, final_layer, camera, twist heads
            if old_key.startswith(('smpl.', 'deconv_', 'final_layer', 'deccam', 'decphi')):
                continue

            if old_key == 'init_shape':
                new_sd['init_shape'] = tensor.unsqueeze(0) if tensor.dim() == 1 else tensor
                continue
            if old_key == 'init_cam':
                continue

            if old_key in key_map:
                new_sd[key_map[old_key]] = tensor
            elif old_key.startswith('preact.'):
                new_sd[old_key.replace('preact.', '')] = tensor
            elif old_key.startswith(('fc1.', 'fc2.', 'decshape.', 'drop1.', 'drop2.')):
                new_sd[old_key] = tensor

        result = self.load_state_dict(new_sd, strict=False)
        # img_mean/img_std are expected missing (registered buffers, not in checkpoint)
        unexpected_missing = [k for k in result.missing_keys
                              if k not in ('img_mean', 'img_std')]
        if unexpected_missing:
            print(f"  [body_shape] Warning: missing keys: {unexpected_missing}")

    def _backbone(self, x):
        """Run ResNet-34 backbone → global avg pool."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)  # (B, 512)

    def _predict_betas(self, features):
        """FC head: features → 10 betas.

        No ReLU between FC layers — HybrIK's shape head uses linear + dropout
        only, preserving negative activations that carry shape information.
        """
        x = self.drop1(self.fc1(features))
        x = self.drop2(self.fc2(x))
        return self.decshape(x) + self.init_shape

    @staticmethod
    def _square_crop(pil_image, bbox=None):
        """Crop to square aspect-ratio-preserving region around person.

        Mimics HybrIK's _box_to_center_scale with scale_mult=1.25:
        compute center + scale from bbox, make square, pad by 25%.
        """
        w, h = pil_image.size
        if bbox is not None:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            bw = x2 - x1
            bh = y2 - y1
        else:
            cx, cy = w / 2, h / 2
            bw, bh = float(w), float(h)

        # Make square (use max dimension) and pad by 25%
        size = max(bw, bh) * 1.25
        half = size / 2

        x1 = max(0, int(cx - half))
        y1 = max(0, int(cy - half))
        x2 = min(w, int(cx + half))
        y2 = min(h, int(cy + half))

        crop = pil_image.crop((x1, y1, x2, y2))
        return crop

    @torch.no_grad()
    def encode(self, pil_image, person_bbox=None) -> torch.Tensor:
        """Encode a PIL image to 10-dim SMPL betas for caching.

        Args:
            pil_image: PIL Image (RGB)
            person_bbox: optional [x1, y1, x2, y2] person bounding box
        Returns:
            (10,) beta tensor on CPU
        """
        crop = self._square_crop(pil_image, person_bbox)

        # Convert to tensor, resize to 256x256 with F.interpolate
        tensor = torch.from_numpy(np.array(crop)).permute(2, 0, 1).float()
        tensor = tensor.unsqueeze(0) / 255.0
        tensor = F.interpolate(
            tensor, size=(self.INPUT_SIZE, self.INPUT_SIZE),
            mode='bilinear', align_corners=False,
        )

        # HybrIK normalization
        device = next(self.parameters()).device
        tensor = tensor.to(device)
        tensor = (tensor - self.img_mean) / self.img_std

        features = self._backbone(tensor)
        betas = self._predict_betas(features)
        return betas.squeeze(0).cpu()  # (10,)

    def forward(
        self, pixels: torch.Tensor, person_bboxes: Optional[List] = None
    ) -> torch.Tensor:
        """Differentiable forward pass for training.

        Args:
            pixels: (B, 3, H, W) in [0, 1] range (RGB)
            person_bboxes: optional list of [x1, y1, x2, y2] per batch item
        Returns:
            (B, 10) SMPL betas
        """
        pixels = pixels.float()

        if person_bboxes is not None:
            crops = []
            for i in range(pixels.shape[0]):
                bbox = person_bboxes[i]
                if bbox is not None:
                    ph, pw = pixels.shape[2], pixels.shape[3]
                    x1, y1, x2, y2 = bbox
                    # Square crop with 25% padding (matching HybrIK preprocessing)
                    bw, bh = x2 - x1, y2 - y1
                    cx_bbox = (x1 + x2) / 2
                    cy_bbox = (y1 + y2) / 2
                    size = max(bw, bh) * 1.25
                    half = size / 2
                    cx1 = max(0, int(round(float(cx_bbox - half))))
                    cy1 = max(0, int(round(float(cy_bbox - half))))
                    cx2 = min(pw, int(round(float(cx_bbox + half))))
                    cy2 = min(ph, int(round(float(cy_bbox + half))))
                    if cx2 > cx1 and cy2 > cy1:
                        crop = pixels[i:i+1, :, cy1:cy2, cx1:cx2]
                    else:
                        crop = pixels[i:i+1]
                else:
                    crop = pixels[i:i+1]
                crop = F.interpolate(
                    crop, size=(self.INPUT_SIZE, self.INPUT_SIZE),
                    mode='bilinear', align_corners=False,
                )
                crops.append(crop)
            pixels = torch.cat(crops, dim=0)
        else:
            # No bboxes: make square center crop then resize
            _, _, h, w = pixels.shape
            if h != w:
                s = min(h, w)
                y_off = (h - s) // 2
                x_off = (w - s) // 2
                pixels = pixels[:, :, y_off:y_off+s, x_off:x_off+s]
            pixels = F.interpolate(
                pixels, size=(self.INPUT_SIZE, self.INPUT_SIZE),
                mode='bilinear', align_corners=False,
            )

        # HybrIK normalization
        pixels = (pixels - self.img_mean) / self.img_std

        with torch.amp.autocast('cuda', enabled=False):
            features = self._backbone(pixels)
            betas = self._predict_betas(features)

        return betas


def cache_body_shape_embeddings(
    file_items: List['FileItemDTO'],
    face_id_config: 'FaceIDConfig',
):
    """Extract and cache HybrIK SMPL beta embeddings for all file items.

    Caches to {image_dir}/_face_id_cache/{filename}.safetensors alongside
    existing face/body proportion embeddings.  Reuses person_bbox from
    body proportion caching when available.
    """
    from PIL import Image
    from PIL.ImageOps import exif_transpose

    CACHE_VERSION_KEY = 'body_shape_v1'

    encoder = DifferentiableBodyShapeEncoder()
    print("  -  Loading HybrIK encoder for body shape embeddings...")

    no_body_count = 0

    for file_item in tqdm(file_items, desc="Caching body shape embeddings"):
        img_dir = os.path.dirname(file_item.path)
        cache_dir = os.path.join(img_dir, '_face_id_cache')
        filename_no_ext = os.path.splitext(os.path.basename(file_item.path))[0]
        cache_path = os.path.join(cache_dir, f'{filename_no_ext}.safetensors')

        # Check if cache exists with this version
        if os.path.exists(cache_path):
            data = load_file(cache_path)
            if 'body_shape_embedding' in data and CACHE_VERSION_KEY in data:
                file_item.body_shape_embedding = data['body_shape_embedding'].clone()
                continue

        # Load image
        pil_image = exif_transpose(Image.open(file_item.path)).convert('RGB')

        # Reuse person_bbox if available on the file item
        person_bbox = getattr(file_item, 'person_bbox', None)
        if person_bbox is not None:
            person_bbox = person_bbox.tolist()

        # Run encoder
        betas = encoder.encode(pil_image, person_bbox=person_bbox)

        if betas.abs().sum() < 1e-6:
            no_body_count += 1

        file_item.body_shape_embedding = betas

        # Save alongside existing cache data
        os.makedirs(cache_dir, exist_ok=True)
        save_data = {}
        if os.path.exists(cache_path):
            existing = load_file(cache_path)
            # Clone all existing tensors to avoid mmap invalidation
            save_data = {k: v.clone() for k, v in existing.items()}
        save_data['body_shape_embedding'] = betas
        save_data[CACHE_VERSION_KEY] = torch.ones(1)
        save_file(save_data, cache_path)

    # Free encoder VRAM
    del encoder
    torch.cuda.empty_cache()

    if no_body_count > 0:
        print(f"  -  Warning: zero body shape for {no_body_count}/{len(file_items)} images")
