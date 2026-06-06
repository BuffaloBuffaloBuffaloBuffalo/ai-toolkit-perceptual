import os
import shutil
import tempfile
from typing import Optional, List, Tuple, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file
from tqdm import tqdm

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import FileItemDTO
    from toolkit.config_modules import FaceIDConfig


class FaceIDExtractor:
    """Extracts 512-dim ArcFace face embeddings using InsightFace."""

    def __init__(self, model_name: str = 'buffalo_l', device_id: int = 0):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(
            name=model_name,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        self.app.prepare(ctx_id=device_id, det_size=(640, 640))
        self._warn_if_cpu_only()

    def _warn_if_cpu_only(self):
        # InsightFace silently falls back to CPU when onnxruntime can't bind the
        # CUDA provider — commonly the CPU-only `onnxruntime` wheel shadowing
        # `onnxruntime-gpu`, or cuDNN (libcudnn.so.9) missing from the loader path.
        # On a large dataset that turns face caching from seconds into days, so
        # surface it loudly in the job log instead of leaving a silent ETA.
        active = set()
        for m in self.app.models.values():
            try:
                active.update(m.session.get_providers())
            except Exception:
                pass
        if 'CUDAExecutionProvider' not in active:
            try:
                import onnxruntime as ort
                avail = ort.get_available_providers()
            except Exception:
                avail = ['<unknown>']
            bar = '!' * 80
            print(
                f"\n{bar}\n"
                f"[face_id] WARNING: InsightFace is running on CPU (active providers: "
                f"{sorted(active)}).\n"
                f"          onnxruntime available providers: {avail}\n"
                f"          Face-embedding caching will be EXTREMELY slow (seconds per\n"
                f"          image). Usually the CPU-only `onnxruntime` package is\n"
                f"          shadowing `onnxruntime-gpu`, or cuDNN is not on the library\n"
                f"          path. See docker/Dockerfile for the fix.\n{bar}\n"
            )

    def _get_largest_face(self, faces):
        """Return the largest face by bounding box area."""
        return sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)[0]

    def _detect(self, image: np.ndarray):
        """Run detection with a padding fallback for tight close-ups.

        RetinaFace's anchors don't fire when a face fills the frame, so on a
        zero-face result we retry on a 25%-padded copy and subtract the pad
        offset so bboxes stay in original-image coordinates.
        """
        faces = self.app.get(image)
        if len(faces) > 0:
            return faces, 0
        h, w = image.shape[:2]
        pad = max(h, w) // 4
        padded = np.full((h + 2 * pad, w + 2 * pad, 3), 128, dtype=image.dtype)
        padded[pad:pad + h, pad:pad + w] = image
        faces = self.app.get(padded)
        if len(faces) == 0:
            return [], 0
        for f in faces:
            f.bbox = f.bbox - np.array([pad, pad, pad, pad], dtype=f.bbox.dtype)
        return faces, pad

    def extract(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding from a BGR numpy image (OpenCV format).

        Returns 512-dim L2-normalized embedding, or None if no face detected.
        """
        faces, _ = self._detect(image)
        if len(faces) == 0:
            return None
        face = self._get_largest_face(faces)
        return face.normed_embedding.astype(np.float32)

    def extract_with_bbox(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract face embedding and bounding box from a BGR numpy image.

        Returns (512-dim embedding, [x1,y1,x2,y2] bbox) or (None, None).
        """
        faces, _ = self._detect(image)
        if len(faces) == 0:
            return None, None
        face = self._get_largest_face(faces)
        return face.normed_embedding.astype(np.float32), face.bbox.astype(np.float32)

    def extract_from_pil(self, pil_image) -> Optional[np.ndarray]:
        """Extract face embedding from a PIL Image."""
        import cv2
        pil_image = pil_image.convert('RGB')
        image = np.array(pil_image)
        # PIL is RGB, InsightFace expects BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return self.extract(image)

    def extract_from_pil_with_bbox(self, pil_image) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract face embedding and bbox from a PIL Image."""
        import cv2
        pil_image = pil_image.convert('RGB')
        image = np.array(pil_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return self.extract_with_bbox(image)


def crop_face_for_vision(pil_image, bbox: np.ndarray, padding: float = 0.3):
    """Crop face region from PIL image with padding for vision encoder input.

    Args:
        pil_image: PIL Image (RGB)
        bbox: [x1, y1, x2, y2] face bounding box
        padding: fraction of bbox size to add on each side
    Returns:
        Cropped PIL Image
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    pad_w, pad_h = w * padding, h * padding
    img_w, img_h = pil_image.size

    # Expand bbox with padding, clip to image bounds
    x1 = max(0, int(x1 - pad_w))
    y1 = max(0, int(y1 - pad_h))
    x2 = min(img_w, int(x2 + pad_w))
    y2 = min(img_h, int(y2 + pad_h))

    return pil_image.crop((x1, y1, x2, y2))


class VisionFaceEncoder:
    """Encodes face crops using a frozen vision model (CLIP or DINOv2).

    Extracts penultimate hidden states as spatial tokens.
    """

    def __init__(self, model_path: str = 'openai/clip-vit-large-patch14'):
        from transformers import AutoConfig
        self.model_path = model_path

        config = AutoConfig.from_pretrained(model_path)
        self.model_type = config.model_type  # 'clip_vision_model', 'dinov2', etc.

        if 'clip' in self.model_type:
            from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
            self.model = CLIPVisionModelWithProjection.from_pretrained(model_path)
            self.processor = CLIPImageProcessor.from_pretrained(model_path)
        elif 'dinov2' in self.model_type:
            from transformers import Dinov2Model, AutoImageProcessor
            self.model = Dinov2Model.from_pretrained(model_path)
            self.processor = AutoImageProcessor.from_pretrained(model_path)
        else:
            # Generic fallback for other ViT models
            from transformers import AutoModel, AutoImageProcessor
            self.model = AutoModel.from_pretrained(model_path)
            self.processor = AutoImageProcessor.from_pretrained(model_path)

        self.model.eval()
        self.model.requires_grad_(False)
        # CLIP wraps vision config inside a parent CLIPConfig
        if hasattr(config, 'vision_config'):
            self.hidden_size = config.vision_config.hidden_size
        else:
            self.hidden_size = config.hidden_size

    @torch.no_grad()
    def encode(self, pil_image) -> torch.Tensor:
        """Encode a PIL image crop to spatial tokens.

        Returns: (1, num_tokens, hidden_size) tensor from penultimate hidden states.
        """
        inputs = self.processor(images=pil_image, return_tensors='pt')
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True)
        # Penultimate hidden states: richer spatial features than last layer
        hidden = outputs.hidden_states[-2]  # (1, num_tokens, hidden_size)
        return hidden.cpu()


class FaceIDProjector(nn.Module):
    """Projects 512-dim ArcFace face embeddings to model hidden space tokens.

    Based on MLPProjModelClipFace pattern from ip_adapter.py.
    """

    def __init__(self, id_dim: int = 512, hidden_size: int = 4096, num_tokens: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens

        self.norm = nn.LayerNorm(id_dim)
        self.proj = nn.Sequential(
            nn.Linear(id_dim, id_dim * 2),
            nn.GELU(),
            nn.Linear(id_dim * 2, hidden_size * num_tokens),
        )
        # Learnable output scale — starts near zero so face conditioning
        # doesn't disrupt the pretrained model at init
        self.output_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 512) face embedding tensor
        Returns:
            (B, num_tokens, hidden_size) projected face tokens
        """
        x = self.norm(x)
        x = self.proj(x)
        x = x.reshape(-1, self.num_tokens, self.hidden_size)
        return x * self.output_scale


class VisionFaceProjector(nn.Module):
    """Projects vision encoder spatial tokens to model hidden space using a Resampler.

    Compresses ~257 spatial tokens from CLIP/DINOv2 into a configurable number
    of output tokens for injection into the text stream.
    """

    def __init__(self, vision_dim: int = 1024, hidden_size: int = 4096,
                 num_tokens: int = 4, max_seq_len: int = 257):
        super().__init__()
        from toolkit.resampler import Resampler
        self.resampler = Resampler(
            dim=vision_dim,
            depth=4,
            dim_head=64,
            heads=8,
            num_queries=num_tokens,
            embedding_dim=vision_dim,
            output_dim=hidden_size,
            ff_mult=4,
            max_seq_len=max_seq_len,
        )
        self.output_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_vision_tokens, vision_dim) spatial tokens
        Returns:
            (B, num_tokens, hidden_size) projected face tokens
        """
        x = self.resampler(x)
        return x * self.output_scale


class DifferentiableFaceEncoder(nn.Module):
    """PyTorch ArcFace encoder for identity loss during training.

    Converts the same w600k_r50 ONNX model used for conditioning into a
    differentiable PyTorch module. Produces 512-dim embeddings in the same
    space as the ArcFace conditioning embeddings.
    """

    def __init__(self, model_name: str = 'buffalo_l'):
        super().__init__()
        import onnx2torch
        onnx_path = os.path.join(
            os.path.expanduser('~'), '.insightface', 'models', model_name, 'w600k_r50.onnx'
        )
        if not os.path.exists(onnx_path):
            # Trigger InsightFace auto-download by initializing FaceAnalysis
            print(f"  [face_id] ArcFace model not found at {onnx_path}, downloading via InsightFace...")
            try:
                from insightface.app import FaceAnalysis
                app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
                app.prepare(ctx_id=-1, det_size=(160, 160))
                del app
            except Exception as e:
                raise FileNotFoundError(
                    f"ArcFace ONNX model not found at {onnx_path}. "
                    f"Install insightface for automatic download: pip install insightface onnxruntime-gpu\n"
                    f"Or manually download the buffalo_l model pack to ~/.insightface/models/"
                ) from e
        # onnx2torch.safe_shape_inference writes a temp file next to the
        # source ONNX → PermissionError [Errno 13] on Windows when the
        # InsightFace model dir has a live handle on the ONNX (Defender
        # scan, ORT session from auto-download). Staging avoids contention.
        with tempfile.TemporaryDirectory() as _td:
            _local_onnx = os.path.join(_td, os.path.basename(onnx_path))
            shutil.copy2(onnx_path, _local_onnx)
            self.model = onnx2torch.convert(_local_onnx)
        self.model.eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def encode(self, pil_image) -> torch.Tensor:
        """Encode a PIL image to a 512-dim face embedding for caching.

        Expects a face crop (or full image if no bbox available).
        The image is resized to 112x112 for ArcFace.

        Args:
            pil_image: PIL Image (RGB), ideally a face crop
        Returns:
            (512,) L2-normalized embedding tensor
        """
        # Convert to tensor first, pad to square, then resize with bilinear
        # interpolation to match forward()'s preprocessing exactly
        tensor = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float()  # (3, H, W)
        tensor = tensor.unsqueeze(0)  # (1, 3, H, W)
        # Pad to square to preserve facial proportions
        _, _, th, tw = tensor.shape
        if tw != th:
            diff = abs(tw - th)
            if tw > th:
                pad_top = diff // 2
                pad_bot = diff - pad_top
                tensor = torch.nn.functional.pad(tensor, (0, 0, pad_top, pad_bot), mode='constant', value=0)
            else:
                pad_left = diff // 2
                pad_right = diff - pad_left
                tensor = torch.nn.functional.pad(tensor, (pad_left, pad_right, 0, 0), mode='constant', value=0)
        tensor = torch.nn.functional.interpolate(tensor, size=(112, 112), mode='bilinear', align_corners=False)
        tensor = tensor.squeeze(0)  # (3, 112, 112)
        # RGB → BGR: ArcFace (w600k_r50) expects BGR input
        tensor = tensor.flip(0)
        tensor = (tensor - 127.5) / 127.5  # [0,255] → [-1,1]
        tensor = tensor.unsqueeze(0).to(next(self.model.parameters()).device)
        emb = self.model(tensor)  # (1, 512)
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        return emb.squeeze(0).cpu()

    def forward(self, pixels: torch.Tensor, bboxes: Optional[List] = None,
                return_crops: bool = False):
        """Forward pass for training -- gradients flow through the input pixels.

        Args:
            pixels: (B, 3, H, W) in [0, 1] range (RGB)
            bboxes: optional list of [x1, y1, x2, y2] per batch item, in
                    pixel coordinates of `pixels`.  When provided, each image
                    is cropped to its face region (with 15 % padding), padded
                    to square, then resized to 112x112.  ``None`` entries fall
                    back to the full image.
            return_crops: if True, also return the (B, 3, 112, 112) RGB crops
                    that were fed to ArcFace (before BGR/normalization).
        Returns:
            (B, 512) L2-normalized embeddings, or tuple of (embeddings, crops)
        """
        if bboxes is not None:
            crops = []
            for i in range(pixels.shape[0]):
                bbox = bboxes[i]
                if bbox is not None:
                    ph, pw = pixels.shape[2], pixels.shape[3]
                    x1, y1, x2, y2 = bbox
                    bw, bh = x2 - x1, y2 - y1
                    pad_w, pad_h = bw * 0.15, bh * 0.15
                    # Clamp to image bounds
                    cx1 = max(0, int(round(float(x1 - pad_w))))
                    cy1 = max(0, int(round(float(y1 - pad_h))))
                    cx2 = min(pw, int(round(float(x2 + pad_w))))
                    cy2 = min(ph, int(round(float(y2 + pad_h))))
                    # Ensure a valid crop (at least 1x1)
                    if cx2 > cx1 and cy2 > cy1:
                        crop = pixels[i:i+1, :, cy1:cy2, cx1:cx2]
                        # Pad to square to preserve facial proportions
                        _, _, ch, cw = crop.shape
                        if cw != ch:
                            diff = abs(cw - ch)
                            if cw > ch:
                                pad_top = diff // 2
                                pad_bot = diff - pad_top
                                crop = torch.nn.functional.pad(crop, (0, 0, pad_top, pad_bot), mode='constant', value=0)
                            else:
                                pad_left = diff // 2
                                pad_right = diff - pad_left
                                crop = torch.nn.functional.pad(crop, (pad_left, pad_right, 0, 0), mode='constant', value=0)
                    else:
                        crop = pixels[i:i+1]
                else:
                    crop = pixels[i:i+1]
                crop = torch.nn.functional.interpolate(crop, size=(112, 112), mode='bilinear', align_corners=False)
                crops.append(crop)
            pixels = torch.cat(crops, dim=0)
        else:
            pixels = torch.nn.functional.interpolate(pixels, size=(112, 112), mode='bilinear', align_corners=False)

        # Save RGB crops before BGR conversion for diagnostics
        rgb_crops = pixels.detach() if return_crops else None

        # RGB → BGR: ArcFace (w600k_r50) expects BGR input
        pixels = pixels.flip(1)
        pixels = (pixels * 255.0 - 127.5) / 127.5  # [0,1] → [-1,1] matching ArcFace normalization
        emb = self.model(pixels)  # (B, 512)
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        if return_crops:
            return emb, rgb_crops
        return emb


class DifferentiableLandmarkEncoder(nn.Module):
    """MediaPipe FaceMesh V2 encoder for landmark shape loss during training.

    Uses the MediaPipe FaceMesh V2 model (1.2M params) to predict 478 facial
    landmarks via direct coordinate regression. Landmarks are normalized
    (centered on nose tip, scaled by inter-eye distance) so the loss is
    invariant to face position and size.
    """

    # MediaPipe FaceMesh region indices
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269,
            267, 0, 37, 39, 40, 185]
    LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159,
                160, 161, 246]
    RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387,
                 386, 385, 384, 398]
    NOSE = [1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 19, 94, 370]
    MIDFACE = LEFT_EYE + RIGHT_EYE + NOSE

    def __init__(self):
        super().__init__()
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            'py-feat/mp_facemesh_v2',
            'face_landmarks_detector_Nx3x256x256_onnx.pth',
        )
        self.model = torch.load(model_path, map_location='cpu', weights_only=False)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _normalize_landmarks(landmarks: torch.Tensor) -> torch.Tensor:
        """Normalize landmarks: center on landmark 1 (nose tip), scale by inter-eye distance.

        Args:
            landmarks: (..., 478, 2) landmark coordinates
        Returns:
            (..., 478, 2) normalized landmarks
        """
        nose_tip = landmarks[..., 1:2, :]  # (..., 1, 2)
        centered = landmarks - nose_tip
        left_inner_eye = landmarks[..., 133, :]   # (..., 2)
        right_inner_eye = landmarks[..., 362, :]  # (..., 2)
        inter_eye = (left_inner_eye - right_inner_eye).norm(dim=-1, keepdim=True).unsqueeze(-2)  # (..., 1, 1)
        inter_eye = inter_eye.clamp(min=0.01)
        return centered / inter_eye

    @torch.no_grad()
    def encode(self, pil_image) -> torch.Tensor:
        """Encode a PIL image (face crop) to normalized (478, 2) landmarks for caching.

        Args:
            pil_image: PIL Image (RGB), ideally a face crop
        Returns:
            (478, 2) normalized landmarks tensor on CPU
        """
        from torchvision.transforms import functional as TF

        img = pil_image.convert('RGB').resize((256, 256))
        tensor = TF.to_tensor(img).unsqueeze(0)  # (1, 3, 256, 256)
        tensor = tensor.to(next(self.model.parameters()).device)

        out = self.model(tensor)
        raw = out[0]  # (1, 1, 1, 1434)
        landmarks = raw.reshape(1, 478, 3)[..., :2]  # (1, 478, 2) — x,y only

        landmarks = self._normalize_landmarks(landmarks)
        return landmarks.squeeze(0).cpu()  # (478, 2)

    def forward(self, pixels: torch.Tensor, bboxes: Optional[List] = None) -> torch.Tensor:
        """Differentiable forward pass for training.

        Args:
            pixels: (B, 3, H, W) in [0, 1] range (RGB)
            bboxes: optional list of [x1, y1, x2, y2] per batch item
        Returns:
            (B, 478, 2) normalized landmarks
        """
        # Force float32: the onnx2torch GraphModule uses index_put ops that
        # crash under autocast (fp16/bf16 dtype mismatch in PReLU).
        pixels = pixels.float()

        if bboxes is not None:
            crops = []
            for i in range(pixels.shape[0]):
                bbox = bboxes[i]
                if bbox is not None:
                    ph, pw = pixels.shape[2], pixels.shape[3]
                    x1, y1, x2, y2 = bbox
                    bw, bh = x2 - x1, y2 - y1
                    pad_w, pad_h = bw * 0.15, bh * 0.15
                    cx1 = max(0, int(round(float(x1 - pad_w))))
                    cy1 = max(0, int(round(float(y1 - pad_h))))
                    cx2 = min(pw, int(round(float(x2 + pad_w))))
                    cy2 = min(ph, int(round(float(y2 + pad_h))))
                    if cx2 > cx1 and cy2 > cy1:
                        crop = pixels[i:i+1, :, cy1:cy2, cx1:cx2]
                    else:
                        crop = pixels[i:i+1]
                else:
                    crop = pixels[i:i+1]
                crop = torch.nn.functional.interpolate(
                    crop, size=(256, 256), mode='bilinear', align_corners=False
                )
                crops.append(crop)
            pixels = torch.cat(crops, dim=0)
        else:
            pixels = torch.nn.functional.interpolate(
                pixels, size=(256, 256), mode='bilinear', align_corners=False
            )

        with torch.amp.autocast('cuda', enabled=False):
            out = self.model(pixels)
        raw = out[0]  # (B, 1, 1, 1434)
        landmarks = raw.reshape(pixels.shape[0], 478, 3)[..., :2]  # (B, 478, 2)

        landmarks = self._normalize_landmarks(landmarks)
        return landmarks


def cache_face_embeddings(
    file_items: List['FileItemDTO'],
    face_id_config: 'FaceIDConfig',
):
    """Extract and cache face embeddings for all file items.

    Caches to {image_dir}/_face_id_cache/{filename}.safetensors.
    Sets file_item.face_embedding (and file_item.vision_face_embedding if vision_enabled).
    Also caches identity_embedding (ArcFace) when identity_loss_weight > 0.
    """
    from PIL import Image
    from PIL.ImageOps import exif_transpose

    vision_enabled = face_id_config.vision_enabled
    identity_loss_enabled = face_id_config.identity_loss_weight > 0 or face_id_config.identity_metrics
    landmark_loss_enabled = face_id_config.landmark_loss_weight > 0
    need_bbox = vision_enabled or identity_loss_enabled or landmark_loss_enabled

    extractor = FaceIDExtractor(model_name=face_id_config.face_model)
    vision_encoder = None
    if vision_enabled:
        print("  -  Loading vision encoder for face crop embeddings...")
        vision_encoder = VisionFaceEncoder(model_path=face_id_config.vision_model)

    identity_encoder = None
    if identity_loss_enabled:
        print("  -  Loading ArcFace encoder for identity loss embeddings...")
        identity_encoder = DifferentiableFaceEncoder()

    landmark_encoder = None
    if landmark_loss_enabled:
        print("  -  Loading MediaPipe FaceMesh encoder for landmark loss embeddings...")
        landmark_encoder = DifferentiableLandmarkEncoder()

    no_face_count = 0

    for file_item in tqdm(file_items, desc="Caching face embeddings"):
        img_dir = os.path.dirname(file_item.path)
        cache_dir = os.path.join(img_dir, '_face_id_cache')
        filename_no_ext = os.path.splitext(os.path.basename(file_item.path))[0]
        cache_path = os.path.join(cache_dir, f'{filename_no_ext}.safetensors')

        # Check if cache exists and has all needed keys
        if os.path.exists(cache_path):
            data = load_file(cache_path)
            has_arcface = 'face_embedding' in data
            has_vision = 'vision_face_embedding' in data
            has_identity = 'identity_embedding' in data
            has_landmark = 'landmark_embedding' in data
            has_bbox = 'face_bbox' in data

            if (has_arcface
                    and (not vision_enabled or has_vision)
                    and (not identity_loss_enabled or has_identity)
                    and (not landmark_loss_enabled or has_landmark)
                    and (not need_bbox or has_bbox)):
                # Cache is complete — clone() all tensors because safetensors
                # memory-maps files; if another caching pass (e.g. body
                # proportion) overwrites this file, mmap'd tensors go stale.
                file_item.face_embedding = data['face_embedding'].clone()
                if has_bbox:
                    file_item.face_bbox = data['face_bbox'].clone()
                if vision_enabled and has_vision:
                    file_item.vision_face_embedding = data['vision_face_embedding'].clone()
                if identity_loss_enabled and has_identity:
                    file_item.identity_embedding = data['identity_embedding'].clone()
                if landmark_loss_enabled and has_landmark:
                    file_item.landmark_embedding = data['landmark_embedding'].clone()
                continue

        # Need to extract (either no cache or missing embeddings)
        pil_image = exif_transpose(Image.open(file_item.path)).convert('RGB')

        if need_bbox:
            embedding, bbox = extractor.extract_from_pil_with_bbox(pil_image)
        else:
            embedding = extractor.extract_from_pil(pil_image)
            bbox = None

        face_detected = embedding is not None
        if embedding is None:
            no_face_count += 1
            embedding = np.zeros(512, dtype=np.float32)
            bbox = None

        tensor = torch.from_numpy(embedding)
        file_item.face_embedding = tensor

        save_data = {'face_embedding': tensor}

        # Store face bbox for identity loss face cropping
        if bbox is not None:
            bbox_tensor = torch.from_numpy(bbox)
            file_item.face_bbox = bbox_tensor
            save_data['face_bbox'] = bbox_tensor
        else:
            file_item.face_bbox = None

        # Vision encoder embedding from face crop
        if vision_enabled and vision_encoder is not None:
            if bbox is not None:
                face_crop = crop_face_for_vision(pil_image, bbox, padding=face_id_config.vision_crop_padding)
                vision_emb = vision_encoder.encode(face_crop)  # (1, num_tokens, hidden_size)
                vision_tensor = vision_emb.squeeze(0)  # (num_tokens, hidden_size)
            else:
                # No face detected — zero vector matching encoder's output shape
                vision_tensor = torch.zeros(257, vision_encoder.hidden_size)
            file_item.vision_face_embedding = vision_tensor
            save_data['vision_face_embedding'] = vision_tensor

        # ArcFace embedding for identity loss (only if face was detected)
        if identity_loss_enabled and identity_encoder is not None:
            if face_detected and bbox is not None:
                face_crop = crop_face_for_vision(pil_image, bbox, padding=0.15)
                identity_tensor = identity_encoder.encode(face_crop)  # (512,)
            elif face_detected:
                identity_tensor = identity_encoder.encode(pil_image)  # (512,)
            else:
                identity_tensor = torch.zeros(512)
            file_item.identity_embedding = identity_tensor
            save_data['identity_embedding'] = identity_tensor

        # MediaPipe landmark embedding for landmark shape loss (only if face was detected)
        if landmark_loss_enabled and landmark_encoder is not None:
            if face_detected and bbox is not None:
                face_crop = crop_face_for_vision(pil_image, bbox, padding=0.15)
                landmark_tensor = landmark_encoder.encode(face_crop)  # (478, 2)
            elif face_detected:
                landmark_tensor = landmark_encoder.encode(pil_image)  # (478, 2)
            else:
                landmark_tensor = torch.zeros(478, 2)
            file_item.landmark_embedding = landmark_tensor
            save_data['landmark_embedding'] = landmark_tensor

        os.makedirs(cache_dir, exist_ok=True)
        save_file(save_data, cache_path)

    # Free encoder VRAM
    if vision_encoder is not None:
        del vision_encoder
    if identity_encoder is not None:
        del identity_encoder
    if landmark_encoder is not None:
        del landmark_encoder
    torch.cuda.empty_cache()

    if no_face_count > 0:
        print(f"  -  Warning: no face detected in {no_face_count}/{len(file_items)} images (using zero vector)")


# Bump to invalidate cached per-frame video identity GT (e.g. if the crop
# padding or the encoder normalization changes).
# v2: GT is computed on the DECODED frames (same tiny decoder the loss uses),
# not the sharp originals, so a perfect reconstruction scores cosine 1.0 and the
# target is reachable. Frames whose face does not survive decoding are gated out.
CACHE_VERSION_IDENTITY_VIDEO_KEY = "identity_gt_video_v2"


def _ltx_taehv_version(arch: str) -> str:
    return "2.3" if "2.3" in str(arch) else "2.0"


def cache_video_identity_embeddings(
    file_items: List['FileItemDTO'],
    face_id_config: 'FaceIDConfig',
    device: Optional[torch.device] = None,
    num_frames: Optional[int] = None,
    arch: str = "ltx2.3",
    include_images: bool = False,
):
    """Cache per-frame GT ArcFace identity embeddings from the DECODED clip.

    The identity loss compares ``ArcFace(TAEHV-decode(x0))`` against this GT, so
    the GT is computed the SAME way: decode the clip through the same tiny
    decoder the loss uses (the cached latent if available, else a TAEHV
    round-trip of the frames), then run ArcFace on the decoded faces. This makes
    a perfect reconstruction score cosine 1.0 — unlike a GT taken from the sharp
    original, where even a perfect x0 only scored ~0.2 against the blurry decode.

    A frame is only used (``valid = 1``) if the detector finds a face in the
    DECODED frame with ``det_score >= identity_loss_decoded_det_threshold`` —
    frames the codec blurs past recognition are dropped (the user's "sensitivity"
    knob). Cached at ``{video_dir}/_face_id_cache/{stem}.safetensors``:

      - ``identity_gt_video``       ``(T, 512)`` fp16, L2-normalized; zeros where gated out.
      - ``identity_gt_video_bbox``  ``(T, 4)``  float32, normalized [0,1] box in the decoded
                                    frame (loss scales it to the decoded x0 resolution).
      - ``identity_gt_video_valid`` ``(T,)``    float32 (1.0 where the decoded face passed the gate).

    Versioned by ``CACHE_VERSION_IDENTITY_VIDEO_KEY``.

    Args:
        file_items: items whose ``is_video`` is truthy are processed (plus all
            items when ``include_images`` is set).
        face_id_config: InsightFace model name + ``identity_loss_decoded_det_threshold``.
        device: CUDA device for extraction.
        num_frames: uniformly subsample to this many frames (round-trip fallback).
        arch: model arch, selects the tiny decoder (LTX vs Wan).
        include_images: also process non-video items (LTX/Wan still images, where
            num_frames=1). Each image's cached 5D latent decodes as a T=1 clip via
            ``_decode_clip``, so the identity loss can run through the 5D video
            block. The GT is still the decoded-then-ArcFace embedding (zero-floor).
    """
    import cv2
    from PIL import Image
    from toolkit.depth_consistency import (
        decode_wan_x0_to_frames,
        load_taehv_ltx2,
        load_taehv_wan21,
    )
    from toolkit.video_frames import read_video_frames_with_transform

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # include_images routes LTX/Wan still-image datasets (num_frames=1) here too:
    # _decode_clip uses each item's cached 5D latent, so a single image is a T=1
    # clip and the identity loss can run through the 5D video block.
    if include_images:
        video_items = list(file_items)
    else:
        video_items = [f for f in file_items if getattr(f, "is_video", False)]
    if not video_items:
        return

    det_thresh = float(getattr(face_id_config, 'identity_loss_decoded_det_threshold', 0.5))

    print(
        f"  -  Loading ArcFace + InsightFace + tiny decoder for GT video identity "
        f"caching ({len(video_items)} videos, det_thresh={det_thresh})..."
    )
    extractor = FaceIDExtractor(model_name=face_id_config.face_model)
    identity_encoder = DifferentiableFaceEncoder().to(device)
    if str(arch).startswith("ltx"):
        decoder = load_taehv_ltx2(device=str(device), dtype=torch.bfloat16,
                                  version=_ltx_taehv_version(arch))
    else:
        decoder = load_taehv_wan21(device=str(device), dtype=torch.bfloat16)

    def _score_bbox(pil):
        """(det_score, bbox) of the largest face in a PIL frame, or (0.0, None)."""
        faces, _ = extractor._detect(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
        if not faces:
            return 0.0, None
        f = extractor._get_largest_face(faces)
        return float(f.det_score), f.bbox.astype(np.float32)

    def _decode_clip(file_item):
        """Decode the clip through the tiny decoder → (T_out, 3, H, W) in [0,1].

        Prefers the cached real-VAE latent (exact match to what the loss decodes);
        falls back to a TAEHV round-trip of the read frames.
        """
        latent = None
        try:
            _l = file_item.get_latent()
            if _l is not None:
                _l = _l.to(device, torch.bfloat16)
                if _l.dim() == 4:
                    _l = _l.unsqueeze(0)
                if _l.dim() == 5:
                    latent = _l
        except Exception:  # noqa: BLE001 — fall back to round-trip
            latent = None
        with torch.no_grad():
            if latent is not None:
                dec = decode_wan_x0_to_frames(latent, decoder)
            else:
                frames = read_video_frames_with_transform(file_item, num_frames)
                if frames is None:
                    return None
                lat = decoder.encode_video(frames.unsqueeze(0).to(device, torch.bfloat16),
                                           parallel=True, show_progress_bar=False)
                dec = decode_wan_x0_to_frames(lat.permute(0, 2, 1, 3, 4).contiguous(), decoder)
        return dec[0].permute(1, 0, 2, 3).float().clamp(0, 1)  # (T_out, 3, H, W)

    no_face_frames = 0
    total_frames = 0

    for file_item in tqdm(video_items, desc="Caching GT identity (video, decoded)"):
        vid_dir = os.path.dirname(file_item.path)
        cache_dir = os.path.join(vid_dir, "_face_id_cache")
        stem = os.path.splitext(os.path.basename(file_item.path))[0]
        cache_path = os.path.join(cache_dir, f"{stem}.safetensors")

        if os.path.exists(cache_path):
            data = load_file(cache_path)
            if ('identity_gt_video' in data
                    and CACHE_VERSION_IDENTITY_VIDEO_KEY in data
                    and (num_frames is None
                         or data['identity_gt_video'].shape[0] == num_frames)):
                file_item.identity_gt_video = data['identity_gt_video'].clone()
                file_item.identity_gt_video_bbox = data['identity_gt_video_bbox'].clone()
                file_item.identity_gt_video_valid = data['identity_gt_video_valid'].clone()
                continue

        decoded = _decode_clip(file_item)
        if decoded is None:
            print(f"  -  Warning: cannot read/decode video: {file_item.path}")
            continue

        embs, bboxes, valids = [], [], []
        for t in range(decoded.shape[0]):
            frame = decoded[t]  # (3, H, W) in [0, 1] — the DECODED frame
            _, H, W = frame.shape
            pil = Image.fromarray(
                (frame.permute(1, 2, 0).clamp(0, 1) * 255.0).byte().cpu().numpy()
            )
            score, bbox = _score_bbox(pil)
            total_frames += 1
            if bbox is not None and score >= det_thresh:
                bbox_px = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                with torch.no_grad():
                    emb = identity_encoder(frame.unsqueeze(0).to(device),
                                           bboxes=[bbox_px], return_crops=False)
                embs.append(emb.squeeze(0).detach().cpu().to(torch.float16))  # (512,)
                bboxes.append(torch.tensor(
                    [bbox_px[0] / W, bbox_px[1] / H, bbox_px[2] / W, bbox_px[3] / H],
                    dtype=torch.float32,
                ))
                valids.append(1.0)
            else:
                embs.append(torch.zeros(512, dtype=torch.float16))
                bboxes.append(torch.zeros(4, dtype=torch.float32))
                valids.append(0.0)
                no_face_frames += 1

        identity_gt_video = torch.stack(embs)               # (T, 512) fp16
        identity_gt_video_bbox = torch.stack(bboxes)        # (T, 4)
        identity_gt_video_valid = torch.tensor(valids, dtype=torch.float32)  # (T,)

        file_item.identity_gt_video = identity_gt_video
        file_item.identity_gt_video_bbox = identity_gt_video_bbox
        file_item.identity_gt_video_valid = identity_gt_video_valid

        save_data = {}
        if os.path.exists(cache_path):
            try:
                save_data = {k: v.clone() for k, v in load_file(cache_path).items()}
            except Exception:  # noqa: BLE001 — corrupt cache → rewrite
                save_data = {}
        save_data['identity_gt_video'] = identity_gt_video
        save_data['identity_gt_video_bbox'] = identity_gt_video_bbox
        save_data['identity_gt_video_valid'] = identity_gt_video_valid
        save_data[CACHE_VERSION_IDENTITY_VIDEO_KEY] = torch.ones(1)
        os.makedirs(cache_dir, exist_ok=True)
        save_file(save_data, cache_path)

    del extractor, identity_encoder, decoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if no_face_frames > 0:
        print(
            f"  -  Note: face gated out in {no_face_frames}/{total_frames} decoded "
            f"video frames (det_score < {det_thresh}; excluded from the identity loss)"
        )
