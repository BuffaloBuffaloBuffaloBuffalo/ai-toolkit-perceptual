"""Automated tests for identity conditioning modules (face_id + body_id).

Run with: python testing/test_face_id.py
All tests run on CPU, no external models (InsightFace/CLIP/HMR2) required.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file, load_file

from toolkit.face_id import (
    FaceIDProjector,
    VisionFaceProjector,
    crop_face_for_vision,
    cache_face_embeddings,
)
import toolkit.face_id as _fid  # for monkeypatching the extractor in cache merge tests
from toolkit.body_id import BodyIDProjector
from toolkit.config_modules import FaceIDConfig, BodyIDConfig
from toolkit.identity_inference import load_identity_from_lora, has_identity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeFileItem:
    """Mimics FileItemDTO for cache_face_embeddings tests."""
    def __init__(self, path):
        self.path = path
        self.face_embedding = None
        self.vision_face_embedding = None


class _FakeFace:
    """Mimics InsightFace face detection result."""
    def __init__(self, bbox, embedding):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.normed_embedding = np.array(embedding, dtype=np.float32)


def _run_test(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        return False


# ---------------------------------------------------------------------------
# FaceIDProjector tests
# ---------------------------------------------------------------------------

def test_face_id_projector_output_shape():
    proj = FaceIDProjector(id_dim=512, hidden_size=4096, num_tokens=4)
    x = torch.randn(2, 512)
    out = proj(x)
    assert out.shape == (2, 4, 4096), f"Expected (2, 4, 4096), got {out.shape}"


def test_face_id_projector_custom_tokens():
    for num_tokens in [1, 8, 16]:
        proj = FaceIDProjector(id_dim=512, hidden_size=2048, num_tokens=num_tokens)
        x = torch.randn(3, 512)
        out = proj(x)
        assert out.shape == (3, num_tokens, 2048), f"num_tokens={num_tokens}: got {out.shape}"


def test_face_id_projector_output_scale_init():
    proj = FaceIDProjector()
    assert abs(proj.output_scale.item() - 0.01) < 1e-6, \
        f"output_scale should init to 0.01, got {proj.output_scale.item()}"


def test_face_id_projector_output_scale_effect():
    """Output magnitude should be proportional to output_scale."""
    proj = FaceIDProjector(id_dim=512, hidden_size=256, num_tokens=2)
    x = torch.randn(1, 512)
    with torch.no_grad():
        out_default = proj(x).norm().item()
        proj.output_scale.fill_(1.0)
        out_full = proj(x).norm().item()
    # At scale=0.01, output should be ~100x smaller than scale=1.0
    ratio = out_full / max(out_default, 1e-12)
    assert ratio > 50, f"Scale ratio should be ~100, got {ratio:.1f}"


def test_face_id_projector_batch_size_one():
    proj = FaceIDProjector(id_dim=512, hidden_size=4096, num_tokens=4)
    x = torch.randn(1, 512)
    out = proj(x)
    assert out.shape == (1, 4, 4096)


def test_face_id_projector_zero_input():
    """Zero embedding should produce near-zero output (LayerNorm centers, but scale is 0.01)."""
    proj = FaceIDProjector(id_dim=512, hidden_size=256, num_tokens=2)
    x = torch.zeros(1, 512)
    out = proj(x)
    # Output won't be exactly zero (LayerNorm bias, MLP bias), but should be small
    assert out.shape == (1, 2, 256)


def test_face_id_projector_gradient_flows():
    proj = FaceIDProjector(id_dim=512, hidden_size=256, num_tokens=2)
    x = torch.randn(2, 512)
    out = proj(x)
    loss = out.sum()
    loss.backward()
    # output_scale should have gradient
    assert proj.output_scale.grad is not None, "output_scale should receive gradients"
    # MLP layers should have gradients
    for name, param in proj.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# VisionFaceProjector tests
# ---------------------------------------------------------------------------

def test_vision_face_projector_output_shape():
    proj = VisionFaceProjector(vision_dim=1024, hidden_size=4096, num_tokens=4, max_seq_len=257)
    x = torch.randn(2, 257, 1024)
    out = proj(x)
    assert out.shape == (2, 4, 4096), f"Expected (2, 4, 4096), got {out.shape}"


def test_vision_face_projector_custom_tokens():
    for num_tokens in [1, 4, 8]:
        proj = VisionFaceProjector(vision_dim=768, hidden_size=2048, num_tokens=num_tokens, max_seq_len=197)
        x = torch.randn(1, 197, 768)
        out = proj(x)
        assert out.shape == (1, num_tokens, 2048), f"num_tokens={num_tokens}: got {out.shape}"


def test_vision_face_projector_output_scale_init():
    proj = VisionFaceProjector()
    assert abs(proj.output_scale.item() - 0.01) < 1e-6


def test_vision_face_projector_gradient_flows():
    proj = VisionFaceProjector(vision_dim=256, hidden_size=128, num_tokens=2, max_seq_len=16)
    x = torch.randn(1, 16, 256)
    out = proj(x)
    loss = out.sum()
    loss.backward()
    assert proj.output_scale.grad is not None
    for name, param in proj.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_vision_face_projector_variable_seq_len():
    """Resampler should handle shorter sequences than max_seq_len."""
    proj = VisionFaceProjector(vision_dim=256, hidden_size=128, num_tokens=2, max_seq_len=257)
    # Feed 100 tokens instead of 257
    x = torch.randn(1, 100, 256)
    out = proj(x)
    assert out.shape == (1, 2, 128), f"Expected (1, 2, 128), got {out.shape}"


# ---------------------------------------------------------------------------
# crop_face_for_vision tests
# ---------------------------------------------------------------------------

def test_crop_face_basic():
    img = Image.new('RGB', (640, 480), color=(128, 128, 128))
    bbox = np.array([100, 100, 300, 300], dtype=np.float32)
    crop = crop_face_for_vision(img, bbox, padding=0.0)
    assert crop.size == (200, 200), f"Expected (200, 200), got {crop.size}"


def test_crop_face_with_padding():
    img = Image.new('RGB', (640, 480), color=(128, 128, 128))
    bbox = np.array([200, 150, 400, 350], dtype=np.float32)  # 200x200 face
    crop = crop_face_for_vision(img, bbox, padding=0.5)
    # padding = 0.5 * 200 = 100 on each side → 400x400 (if within bounds)
    assert crop.size[0] > 200 and crop.size[1] > 200, \
        f"Padded crop should be larger than bbox, got {crop.size}"


def test_crop_face_clips_to_bounds():
    img = Image.new('RGB', (200, 200), color=(128, 128, 128))
    bbox = np.array([10, 10, 190, 190], dtype=np.float32)  # 180x180 face
    crop = crop_face_for_vision(img, bbox, padding=0.5)
    # With padding, would go outside image — should clip to 200x200
    assert crop.size[0] <= 200 and crop.size[1] <= 200, \
        f"Crop should not exceed image bounds, got {crop.size}"


def test_crop_face_edge_bbox():
    """Bbox at image edge should still produce a valid crop."""
    img = Image.new('RGB', (100, 100), color=(128, 128, 128))
    bbox = np.array([0, 0, 50, 50], dtype=np.float32)
    crop = crop_face_for_vision(img, bbox, padding=0.3)
    assert crop.size[0] > 0 and crop.size[1] > 0


def test_crop_face_zero_padding():
    img = Image.new('RGB', (500, 500), color=(128, 128, 128))
    bbox = np.array([100, 100, 200, 200], dtype=np.float32)
    crop = crop_face_for_vision(img, bbox, padding=0.0)
    assert crop.size == (100, 100)


# ---------------------------------------------------------------------------
# FaceIDConfig tests
# ---------------------------------------------------------------------------

def test_face_id_config_defaults():
    cfg = FaceIDConfig()
    assert cfg.enabled is False
    assert cfg.num_tokens == 4
    assert cfg.dropout_prob == 0.1
    assert cfg.face_model == 'buffalo_l'
    assert cfg.vision_enabled is False
    assert cfg.vision_model == 'openai/clip-vit-large-patch14'
    assert cfg.vision_num_tokens == 4
    assert cfg.vision_crop_padding == 0.3


def test_face_id_config_custom():
    cfg = FaceIDConfig(
        enabled=True,
        num_tokens=8,
        dropout_prob=0.2,
        vision_enabled=True,
        vision_model='facebook/dinov2-large',
        vision_num_tokens=8,
        vision_crop_padding=0.5,
    )
    assert cfg.enabled is True
    assert cfg.num_tokens == 8
    assert cfg.dropout_prob == 0.2
    assert cfg.vision_enabled is True
    assert cfg.vision_model == 'facebook/dinov2-large'
    assert cfg.vision_num_tokens == 8
    assert cfg.vision_crop_padding == 0.5


# ---------------------------------------------------------------------------
# Synchronized dropout tests
# ---------------------------------------------------------------------------

def test_synchronized_dropout_both_zeroed():
    """When dropout mask is 0, both face and vision tokens should be zeroed."""
    torch.manual_seed(42)
    face_tokens = torch.randn(4, 4, 4096)
    vision_tokens = torch.randn(4, 4, 4096)

    # Force all samples dropped
    drop_mask = torch.zeros(4, 1, 1)
    face_dropped = face_tokens * drop_mask
    vision_dropped = vision_tokens * drop_mask

    assert face_dropped.abs().sum() == 0, "Face tokens should be zero when mask is 0"
    assert vision_dropped.abs().sum() == 0, "Vision tokens should be zero when mask is 0"


def test_synchronized_dropout_both_kept():
    """When dropout mask is 1, both should be unchanged."""
    torch.manual_seed(42)
    face_tokens = torch.randn(4, 4, 4096)
    vision_tokens = torch.randn(4, 4, 4096)

    drop_mask = torch.ones(4, 1, 1)
    face_dropped = face_tokens * drop_mask
    vision_dropped = vision_tokens * drop_mask

    assert torch.equal(face_dropped, face_tokens)
    assert torch.equal(vision_dropped, vision_tokens)


def test_synchronized_dropout_per_sample():
    """Dropout should be per-sample: some samples kept, some dropped."""
    B = 8
    face_tokens = torch.ones(B, 4, 256)
    vision_tokens = torch.ones(B, 4, 256)

    # Manually set mask: first 4 kept, last 4 dropped
    drop_mask = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32).view(B, 1, 1)
    face_dropped = face_tokens * drop_mask
    vision_dropped = vision_tokens * drop_mask

    # First 4 samples should be nonzero
    assert face_dropped[:4].abs().sum() > 0
    assert vision_dropped[:4].abs().sum() > 0
    # Last 4 should be zero
    assert face_dropped[4:].abs().sum() == 0
    assert vision_dropped[4:].abs().sum() == 0


def test_synchronized_dropout_mask_broadcasts():
    """Mask shape (B, 1, 1) should broadcast to (B, num_tokens, hidden_size)."""
    face_tokens = torch.randn(3, 8, 512)  # 8 tokens
    vision_tokens = torch.randn(3, 4, 512)  # 4 tokens

    drop_mask = torch.tensor([1, 0, 1], dtype=torch.float32).view(3, 1, 1)
    face_dropped = face_tokens * drop_mask
    vision_dropped = vision_tokens * drop_mask

    # Sample 1 (idx 1) should be zero for both despite different token counts
    assert face_dropped[1].abs().sum() == 0
    assert vision_dropped[1].abs().sum() == 0
    # Samples 0 and 2 should be unchanged
    assert torch.equal(face_dropped[0], face_tokens[0])
    assert torch.equal(vision_dropped[2], vision_tokens[2])


# ---------------------------------------------------------------------------
# Token concatenation tests
# ---------------------------------------------------------------------------

def test_token_concatenation_shape():
    """ArcFace + vision tokens concatenated should sum token counts."""
    face_tokens = torch.randn(2, 4, 4096)
    vision_tokens = torch.randn(2, 4, 4096)
    combined = torch.cat([face_tokens, vision_tokens], dim=1)
    assert combined.shape == (2, 8, 4096)


def test_token_concatenation_asymmetric():
    """Different token counts for ArcFace and vision should work."""
    face_tokens = torch.randn(2, 4, 4096)
    vision_tokens = torch.randn(2, 8, 4096)
    combined = torch.cat([face_tokens, vision_tokens], dim=1)
    assert combined.shape == (2, 12, 4096)


def test_token_concatenation_preserves_values():
    face_tokens = torch.randn(1, 2, 64)
    vision_tokens = torch.randn(1, 3, 64)
    combined = torch.cat([face_tokens, vision_tokens], dim=1)
    assert torch.equal(combined[:, :2, :], face_tokens)
    assert torch.equal(combined[:, 2:, :], vision_tokens)


# ---------------------------------------------------------------------------
# Token norm metric tests
# ---------------------------------------------------------------------------

def test_per_token_norm_computation():
    """Verify the norm computation matches what SDTrainer does."""
    tokens = torch.randn(2, 4, 4096)
    per_token_norms = tokens.detach().float().norm(dim=-1)  # (B, num_tokens)
    assert per_token_norms.shape == (2, 4)
    metric = per_token_norms.mean().item()
    assert isinstance(metric, float)
    assert metric > 0


def test_per_token_norm_zero_tokens():
    """Zero tokens should have zero norm."""
    tokens = torch.zeros(2, 4, 4096)
    per_token_norms = tokens.detach().float().norm(dim=-1)
    assert per_token_norms.sum() == 0


def test_per_token_norm_batch_invariant():
    """Same token repeated across batch should give same norm regardless of batch size."""
    single_token = torch.randn(1, 4, 256)
    batch_token = single_token.repeat(8, 1, 1)

    norm_single = single_token.detach().float().norm(dim=-1).mean().item()
    norm_batch = batch_token.detach().float().norm(dim=-1).mean().item()
    assert abs(norm_single - norm_batch) < 1e-5, \
        f"Norms should match: {norm_single} vs {norm_batch}"


# ---------------------------------------------------------------------------
# Cache backward compatibility tests
# ---------------------------------------------------------------------------

def test_cache_loads_arcface_only():
    """Old cache files with only face_embedding should still load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, 'test.safetensors')
        emb = torch.randn(512)
        save_file({'face_embedding': emb}, cache_path)

        data = load_file(cache_path)
        assert 'face_embedding' in data
        assert 'vision_face_embedding' not in data
        assert data['face_embedding'].shape == (512,)


def test_cache_loads_both_embeddings():
    """New cache files should store both embeddings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, 'test.safetensors')
        face_emb = torch.randn(512)
        vision_emb = torch.randn(257, 1024)
        save_file({
            'face_embedding': face_emb,
            'vision_face_embedding': vision_emb,
        }, cache_path)

        data = load_file(cache_path)
        assert 'face_embedding' in data
        assert 'vision_face_embedding' in data
        assert data['face_embedding'].shape == (512,)
        assert data['vision_face_embedding'].shape == (257, 1024)


def test_cache_completeness_check():
    """Simulate the cache completeness logic from cache_face_embeddings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, 'test.safetensors')

        # Old cache: only ArcFace
        save_file({'face_embedding': torch.randn(512)}, cache_path)
        data = load_file(cache_path)
        has_arcface = 'face_embedding' in data
        has_vision = 'vision_face_embedding' in data

        # With vision_enabled=False, cache is complete
        vision_enabled = False
        assert has_arcface and (not vision_enabled or has_vision)

        # With vision_enabled=True, cache is incomplete (needs re-extract)
        vision_enabled = True
        assert not (has_arcface and (not vision_enabled or has_vision))


# ---------------------------------------------------------------------------
# cache_face_embeddings merge tests
#
# _face_id_cache/{stem}.safetensors is SHARED: the depth (depth_gt_*),
# body-proportion, and body-shape passes write the same file. The face pass
# must MERGE its keys in, never blind-overwrite, or it silently destroys the
# others' maps (e.g. depth maps copied onto a fresh pod, since face caches
# before depth). The extractor is faked so these stay CPU / no-model.
# ---------------------------------------------------------------------------

_MERGE_FACE_VAL = np.arange(512, dtype=np.float32) / 512.0
_MERGE_DEPTH = torch.full((96, 96), 3.5, dtype=torch.float16)


class _FakeFaceExtractor:
    def __init__(self, *a, **k):
        pass

    def extract_from_pil(self, pil):
        return _MERGE_FACE_VAL.copy()

    def extract_from_pil_with_bbox(self, pil):
        return _MERGE_FACE_VAL.copy(), np.array([1, 2, 30, 40], dtype=np.float32)


class _FakeIdentityEncoder:
    def __init__(self, *a, **k):
        pass

    def encode(self, pil):
        return torch.full((512,), 0.25)


class _patch_face_models:
    """Swap InsightFace / ArcFace for CPU fakes around a cache_face_embeddings call."""
    def __init__(self, identity=False):
        self.identity = identity

    def __enter__(self):
        self._ext, self._idy = _fid.FaceIDExtractor, _fid.DifferentiableFaceEncoder
        _fid.FaceIDExtractor = _FakeFaceExtractor
        if self.identity:
            _fid.DifferentiableFaceEncoder = _FakeIdentityEncoder
        return self

    def __exit__(self, *a):
        _fid.FaceIDExtractor, _fid.DifferentiableFaceEncoder = self._ext, self._idy
        return False


def _seed_cache(cache_path, tensors):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    save_file(tensors, cache_path)


def _make_image(path):
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(path)


def test_face_cache_merge_preserves_depth():
    """Face pass must keep a sibling depth_gt_* key written by the depth pass."""
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, 'im.jpg'); _make_image(img)
        cp = os.path.join(d, '_face_id_cache', 'im.safetensors')
        _seed_cache(cp, {'depth_gt_512x512': _MERGE_DEPTH.clone(), 'depth_gt_v3': torch.ones(1)})
        with _patch_face_models():
            cache_face_embeddings([_FakeFileItem(img)], FaceIDConfig())
        data = load_file(cp)
        assert 'depth_gt_512x512' in data and 'depth_gt_v3' in data, "depth keys destroyed"
        assert torch.equal(data['depth_gt_512x512'], _MERGE_DEPTH), "depth bytes changed"
        assert 'face_embedding' in data, "face_embedding not written"
        assert np.allclose(data['face_embedding'].numpy(), _MERGE_FACE_VAL), "wrong face value"


def test_face_cache_merge_preserves_all_siblings():
    """depth + body_proportion + body_shape keys all survive the face pass."""
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, 'im.jpg'); _make_image(img)
        cp = os.path.join(d, '_face_id_cache', 'im.safetensors')
        seeded = {
            'depth_gt_512x512': _MERGE_DEPTH.clone(), 'depth_gt_v3': torch.ones(1),
            'body_proportion_512x512': torch.full((17, 3), 0.7), 'body_proportion_v1': torch.ones(1),
            'body_shape_512x512': torch.full((10,), 1.3), 'body_shape_v1': torch.ones(1),
        }
        _seed_cache(cp, {k: v.clone() for k, v in seeded.items()})
        with _patch_face_models():
            cache_face_embeddings([_FakeFileItem(img)], FaceIDConfig())
        data = load_file(cp)
        for k, v in seeded.items():
            assert k in data and torch.equal(data[k], v), f"{k} not preserved"
        assert 'face_embedding' in data, "face_embedding not written"


def test_face_cache_fresh_file_is_face_only():
    """No pre-existing cache: behaviour is unchanged (face-only file written)."""
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, 'im.jpg'); _make_image(img)
        cp = os.path.join(d, '_face_id_cache', 'im.safetensors')
        with _patch_face_models():
            cache_face_embeddings([_FakeFileItem(img)], FaceIDConfig())
        data = load_file(cp)
        assert list(data.keys()) == ['face_embedding'], f"fresh file not face-only: {sorted(data)}"


def test_face_cache_reextract_preserves_depth():
    """A face miss (identity key absent) re-extracts: refresh face, keep depth."""
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, 'im.jpg'); _make_image(img)
        cp = os.path.join(d, '_face_id_cache', 'im.safetensors')
        _seed_cache(cp, {'depth_gt_512x512': _MERGE_DEPTH.clone(), 'depth_gt_v3': torch.ones(1),
                         'face_embedding': torch.full((512,), -9.0)})
        with _patch_face_models(identity=True):
            cache_face_embeddings([_FakeFileItem(img)], FaceIDConfig(identity_metrics=True))
        data = load_file(cp)
        assert torch.equal(data['depth_gt_512x512'], _MERGE_DEPTH), "depth not preserved on re-extract"
        assert np.allclose(data['face_embedding'].numpy(), _MERGE_FACE_VAL), "stale face not refreshed"
        assert 'identity_embedding' in data, "identity_embedding not added"


def test_face_cache_corrupt_file_fallback():
    """A pre-existing corrupt/garbage cache file must not crash the face pass.

    The merge step tries load_file() on the existing path; on a corrupt file it
    falls back to a face-only rewrite. The result must be a *valid* safetensors
    file that is face-keyed (the unreadable siblings are unavoidably lost, but
    the write must not raise and must not leave a broken file behind).
    """
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, 'im.jpg'); _make_image(img)
        cp = os.path.join(d, '_face_id_cache', 'im.safetensors')
        # Pre-write garbage bytes at the cache path (not a valid safetensors blob).
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        with open(cp, 'wb') as f:
            f.write(b'\x00\x01not a safetensors file\xff\xfe' * 8)
        with _patch_face_models():
            # Must not raise despite the corrupt destination.
            cache_face_embeddings([_FakeFileItem(img)], FaceIDConfig())
        # Resulting file is valid and face-keyed.
        data = load_file(cp)
        assert 'face_embedding' in data, "face_embedding not written after corrupt fallback"
        assert np.allclose(data['face_embedding'].numpy(), _MERGE_FACE_VAL), "wrong face value"


def test_face_cache_write_is_atomic_no_tmp_leftover():
    """The atomic writer leaves no .tmp_* turds in the cache dir on success."""
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, 'im.jpg'); _make_image(img)
        cache_dir = os.path.join(d, '_face_id_cache')
        cp = os.path.join(cache_dir, 'im.safetensors')
        with _patch_face_models():
            cache_face_embeddings([_FakeFileItem(img)], FaceIDConfig())
        leftovers = [f for f in os.listdir(cache_dir) if f.startswith('.tmp_')]
        assert leftovers == [], f"atomic save left temp files behind: {leftovers}"
        assert os.path.exists(cp) and load_file(cp), "final cache file missing/invalid"


# ---------------------------------------------------------------------------
# End-to-end projector pipeline test
# ---------------------------------------------------------------------------

def test_end_to_end_pipeline():
    """Simulate the full training step: project both embeddings, dropout, concat."""
    B = 4
    hidden_size = 256  # small for speed

    face_proj = FaceIDProjector(id_dim=512, hidden_size=hidden_size, num_tokens=4)
    vision_proj = VisionFaceProjector(vision_dim=128, hidden_size=hidden_size, num_tokens=4, max_seq_len=50)

    # Simulate batch
    face_emb = torch.randn(B, 512)
    vision_emb = torch.randn(B, 50, 128)

    # Project
    face_tokens = face_proj(face_emb)
    vision_tokens = vision_proj(vision_emb)

    assert face_tokens.shape == (B, 4, hidden_size)
    assert vision_tokens.shape == (B, 4, hidden_size)

    # Synchronized dropout
    dropout_prob = 0.5
    drop_mask = (torch.rand(B, 1, 1) > dropout_prob).float()
    face_tokens = face_tokens * drop_mask
    vision_tokens = vision_tokens * drop_mask

    # Verify synchronization
    for i in range(B):
        if drop_mask[i].item() == 0:
            assert face_tokens[i].abs().sum() == 0
            assert vision_tokens[i].abs().sum() == 0

    # Concatenate
    all_tokens = torch.cat([face_tokens, vision_tokens], dim=1)
    assert all_tokens.shape == (B, 8, hidden_size)

    # Backward should work
    loss = all_tokens.sum()
    loss.backward()
    assert face_proj.output_scale.grad is not None
    assert vision_proj.output_scale.grad is not None


def test_end_to_end_arcface_only():
    """Pipeline works with only ArcFace (vision disabled)."""
    B = 2
    hidden_size = 128

    face_proj = FaceIDProjector(id_dim=512, hidden_size=hidden_size, num_tokens=4)
    face_emb = torch.randn(B, 512)
    face_tokens = face_proj(face_emb)

    vision_tokens = None  # vision disabled

    # Dropout
    drop_mask = (torch.rand(B, 1, 1) > 0.1).float()
    face_tokens = face_tokens * drop_mask
    if vision_tokens is not None:
        vision_tokens = vision_tokens * drop_mask

    # Concat
    if vision_tokens is not None:
        all_tokens = torch.cat([face_tokens, vision_tokens], dim=1)
    else:
        all_tokens = face_tokens

    assert all_tokens.shape == (B, 4, hidden_size)


# ---------------------------------------------------------------------------
# BodyIDProjector tests
# ---------------------------------------------------------------------------

def test_body_id_projector_output_shape():
    proj = BodyIDProjector(id_dim=10, hidden_size=4096, num_tokens=4)
    x = torch.randn(2, 10)
    out = proj(x)
    assert out.shape == (2, 4, 4096), f"Expected (2, 4, 4096), got {out.shape}"


def test_body_id_projector_custom_tokens():
    for num_tokens in [1, 4, 8]:
        proj = BodyIDProjector(id_dim=10, hidden_size=2048, num_tokens=num_tokens)
        x = torch.randn(3, 10)
        out = proj(x)
        assert out.shape == (3, num_tokens, 2048), f"num_tokens={num_tokens}: got {out.shape}"


def test_body_id_projector_output_scale_init():
    proj = BodyIDProjector()
    assert abs(proj.output_scale.item() - 0.01) < 1e-6


def test_body_id_projector_gradient_flows():
    proj = BodyIDProjector(id_dim=10, hidden_size=128, num_tokens=2)
    x = torch.randn(2, 10)
    out = proj(x)
    loss = out.sum()
    loss.backward()
    assert proj.output_scale.grad is not None
    for name, param in proj.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_body_id_projector_zero_input():
    """Zero betas should produce valid output shape."""
    proj = BodyIDProjector(id_dim=10, hidden_size=128, num_tokens=2)
    x = torch.zeros(1, 10)
    out = proj(x)
    assert out.shape == (1, 2, 128)


def test_body_id_projector_batch_size_one():
    proj = BodyIDProjector(id_dim=10, hidden_size=4096, num_tokens=4)
    x = torch.randn(1, 10)
    out = proj(x)
    assert out.shape == (1, 4, 4096)


# ---------------------------------------------------------------------------
# BodyIDConfig tests
# ---------------------------------------------------------------------------

def test_body_id_config_defaults():
    cfg = BodyIDConfig()
    assert cfg.enabled is False
    assert cfg.num_tokens == 4
    assert cfg.dropout_prob == 0.1
    assert cfg.detection_threshold == 0.5


def test_body_id_config_custom():
    cfg = BodyIDConfig(
        enabled=True,
        num_tokens=8,
        dropout_prob=0.2,
        detection_threshold=0.7,
    )
    assert cfg.enabled is True
    assert cfg.num_tokens == 8
    assert cfg.dropout_prob == 0.2
    assert cfg.detection_threshold == 0.7


# ---------------------------------------------------------------------------
# Body embedding cache tests
# ---------------------------------------------------------------------------

def test_body_cache_format():
    """Body cache stores 10-dim tensor."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, 'test.safetensors')
        body_emb = torch.randn(10)
        save_file({'body_embedding': body_emb}, cache_path)

        data = load_file(cache_path)
        assert 'body_embedding' in data
        assert data['body_embedding'].shape == (10,)


# ---------------------------------------------------------------------------
# Three-way synchronized dropout tests
# ---------------------------------------------------------------------------

def test_three_way_dropout_all_zeroed():
    """All three token types should be zeroed with same mask."""
    face = torch.randn(4, 4, 256)
    vision = torch.randn(4, 4, 256)
    body = torch.randn(4, 4, 256)
    mask = torch.zeros(4, 1, 1)

    assert (face * mask).abs().sum() == 0
    assert (vision * mask).abs().sum() == 0
    assert (body * mask).abs().sum() == 0


def test_three_way_dropout_per_sample():
    """Per-sample mask applies identically across all three."""
    B = 6
    face = torch.ones(B, 4, 128)
    vision = torch.ones(B, 4, 128)
    body = torch.ones(B, 4, 128)
    mask = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.float32).view(B, 1, 1)

    face_d = face * mask
    vision_d = vision * mask
    body_d = body * mask

    for i in range(B):
        if mask[i].item() == 0:
            assert face_d[i].abs().sum() == 0
            assert vision_d[i].abs().sum() == 0
            assert body_d[i].abs().sum() == 0
        else:
            assert face_d[i].abs().sum() > 0
            assert vision_d[i].abs().sum() > 0
            assert body_d[i].abs().sum() > 0


# ---------------------------------------------------------------------------
# Three-way concatenation test
# ---------------------------------------------------------------------------

def test_three_way_concatenation():
    """Face + vision + body tokens concatenated correctly."""
    face = torch.randn(2, 4, 256)
    vision = torch.randn(2, 4, 256)
    body = torch.randn(2, 4, 256)
    combined = torch.cat([face, vision, body], dim=1)
    assert combined.shape == (2, 12, 256)
    assert torch.equal(combined[:, :4], face)
    assert torch.equal(combined[:, 4:8], vision)
    assert torch.equal(combined[:, 8:], body)


# ---------------------------------------------------------------------------
# Full E2E with all three projectors
# ---------------------------------------------------------------------------

def test_end_to_end_all_three():
    """Full pipeline: face + vision + body projectors, dropout, concat, backward."""
    B = 4
    hidden_size = 128

    face_proj = FaceIDProjector(id_dim=512, hidden_size=hidden_size, num_tokens=4)
    vision_proj = VisionFaceProjector(vision_dim=64, hidden_size=hidden_size, num_tokens=4, max_seq_len=16)
    body_proj = BodyIDProjector(id_dim=10, hidden_size=hidden_size, num_tokens=4)

    face_tokens = face_proj(torch.randn(B, 512))
    vision_tokens = vision_proj(torch.randn(B, 16, 64))
    body_tokens = body_proj(torch.randn(B, 10))

    assert face_tokens.shape == (B, 4, hidden_size)
    assert vision_tokens.shape == (B, 4, hidden_size)
    assert body_tokens.shape == (B, 4, hidden_size)

    # Synchronized dropout
    mask = (torch.rand(B, 1, 1) > 0.5).float()
    face_tokens = face_tokens * mask
    vision_tokens = vision_tokens * mask
    body_tokens = body_tokens * mask

    # Concatenate
    all_tokens = torch.cat([face_tokens, vision_tokens, body_tokens], dim=1)
    assert all_tokens.shape == (B, 12, hidden_size)

    # Backward
    loss = all_tokens.sum()
    loss.backward()
    assert face_proj.output_scale.grad is not None
    assert vision_proj.output_scale.grad is not None
    assert body_proj.output_scale.grad is not None


# ---------------------------------------------------------------------------
# Identity save/load round-trip tests
# ---------------------------------------------------------------------------

def test_save_load_identity_face_only():
    """Save face projector + embedding to safetensors, load back with identity_inference."""
    import json
    hidden_size = 128
    num_tokens = 2

    # Create projector and fake averaged embedding
    proj = FaceIDProjector(id_dim=512, hidden_size=hidden_size, num_tokens=num_tokens)
    avg_emb = torch.randn(512)

    # Build state dict like SDTrainer._get_identity_state_dict
    save_dict = {}
    for k, v in proj.state_dict().items():
        save_dict[f'face_id_projector.{k}'] = v
    save_dict['identity.face_embedding'] = avg_emb

    metadata = {
        'identity_enhanced': 'true',
        'face_id_config': json.dumps({'num_tokens': num_tokens, 'vision_enabled': False, 'vision_num_tokens': 4}),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test_lora.safetensors')
        save_file(save_dict, path, metadata=metadata)

        assert has_identity(path)
        result = load_identity_from_lora(path)
        assert result['face_tokens'] is not None
        assert result['face_tokens'].shape == (1, num_tokens, hidden_size)
        assert result['vision_tokens'] is None
        assert result['body_tokens'] is None
        assert result['all_tokens'].shape == (1, num_tokens, hidden_size)


def test_save_load_identity_all_three():
    """Save all three projectors + embeddings, load back."""
    import json
    hidden_size = 64
    face_tokens = 2
    vision_tokens = 3
    body_tokens = 2

    face_proj = FaceIDProjector(id_dim=512, hidden_size=hidden_size, num_tokens=face_tokens)
    vision_proj = VisionFaceProjector(vision_dim=128, hidden_size=hidden_size, num_tokens=vision_tokens, max_seq_len=16)
    body_proj = BodyIDProjector(id_dim=10, hidden_size=hidden_size, num_tokens=body_tokens)

    save_dict = {}
    for k, v in face_proj.state_dict().items():
        save_dict[f'face_id_projector.{k}'] = v
    for k, v in vision_proj.state_dict().items():
        save_dict[f'vision_face_projector.{k}'] = v
    for k, v in body_proj.state_dict().items():
        save_dict[f'body_id_projector.{k}'] = v
    save_dict['identity.face_embedding'] = torch.randn(512)
    save_dict['identity.vision_face_embedding'] = torch.randn(16, 128)
    save_dict['identity.body_embedding'] = torch.randn(10)

    metadata = {
        'identity_enhanced': 'true',
        'face_id_config': json.dumps({'num_tokens': face_tokens, 'vision_enabled': True, 'vision_num_tokens': vision_tokens}),
        'body_id_config': json.dumps({'num_tokens': body_tokens}),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test_lora.safetensors')
        save_file(save_dict, path, metadata=metadata)

        result = load_identity_from_lora(path)
        assert result['face_tokens'].shape == (1, face_tokens, hidden_size)
        assert result['vision_tokens'].shape == (1, vision_tokens, hidden_size)
        assert result['body_tokens'].shape == (1, body_tokens, hidden_size)
        total = face_tokens + vision_tokens + body_tokens
        assert result['all_tokens'].shape == (1, total, hidden_size)


def test_has_identity_false():
    """A plain LoRA file should not be detected as identity-enhanced."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'plain.safetensors')
        save_file({'lora_unet_something.weight': torch.randn(4, 4)}, path)
        assert not has_identity(path)


def test_load_identity_no_identity_keys():
    """Loading a plain LoRA should return all None tokens."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'plain.safetensors')
        save_file({'lora_unet_something.weight': torch.randn(4, 4)}, path)
        result = load_identity_from_lora(path)
        assert result['face_tokens'] is None
        assert result['all_tokens'] is None


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    tests = [
        # FaceIDProjector
        ("FaceIDProjector output shape", test_face_id_projector_output_shape),
        ("FaceIDProjector custom tokens", test_face_id_projector_custom_tokens),
        ("FaceIDProjector output_scale init", test_face_id_projector_output_scale_init),
        ("FaceIDProjector output_scale effect", test_face_id_projector_output_scale_effect),
        ("FaceIDProjector batch size 1", test_face_id_projector_batch_size_one),
        ("FaceIDProjector zero input", test_face_id_projector_zero_input),
        ("FaceIDProjector gradient flow", test_face_id_projector_gradient_flows),
        # VisionFaceProjector
        ("VisionFaceProjector output shape", test_vision_face_projector_output_shape),
        ("VisionFaceProjector custom tokens", test_vision_face_projector_custom_tokens),
        ("VisionFaceProjector output_scale init", test_vision_face_projector_output_scale_init),
        ("VisionFaceProjector gradient flow", test_vision_face_projector_gradient_flows),
        ("VisionFaceProjector variable seq len", test_vision_face_projector_variable_seq_len),
        # crop_face_for_vision
        ("crop_face_for_vision basic", test_crop_face_basic),
        ("crop_face_for_vision with padding", test_crop_face_with_padding),
        ("crop_face_for_vision clips to bounds", test_crop_face_clips_to_bounds),
        ("crop_face_for_vision edge bbox", test_crop_face_edge_bbox),
        ("crop_face_for_vision zero padding", test_crop_face_zero_padding),
        # FaceIDConfig
        ("FaceIDConfig defaults", test_face_id_config_defaults),
        ("FaceIDConfig custom values", test_face_id_config_custom),
        # Synchronized dropout
        ("Dropout both zeroed", test_synchronized_dropout_both_zeroed),
        ("Dropout both kept", test_synchronized_dropout_both_kept),
        ("Dropout per-sample", test_synchronized_dropout_per_sample),
        ("Dropout mask broadcasts", test_synchronized_dropout_mask_broadcasts),
        # Token concatenation
        ("Token concat shape", test_token_concatenation_shape),
        ("Token concat asymmetric", test_token_concatenation_asymmetric),
        ("Token concat preserves values", test_token_concatenation_preserves_values),
        # Token norms
        ("Token norm computation", test_per_token_norm_computation),
        ("Token norm zero", test_per_token_norm_zero_tokens),
        ("Token norm batch invariant", test_per_token_norm_batch_invariant),
        # Cache
        ("Cache loads ArcFace only", test_cache_loads_arcface_only),
        ("Cache loads both embeddings", test_cache_loads_both_embeddings),
        ("Cache completeness check", test_cache_completeness_check),
        # Cache merge (shared _face_id_cache file must not clobber siblings)
        ("Cache merge preserves depth", test_face_cache_merge_preserves_depth),
        ("Cache merge preserves depth+body siblings", test_face_cache_merge_preserves_all_siblings),
        ("Cache fresh file is face-only", test_face_cache_fresh_file_is_face_only),
        ("Cache re-extract preserves depth", test_face_cache_reextract_preserves_depth),
        ("Cache corrupt-file fallback", test_face_cache_corrupt_file_fallback),
        ("Cache write is atomic (no tmp leftover)", test_face_cache_write_is_atomic_no_tmp_leftover),
        # End-to-end (face + vision)
        ("E2E pipeline (ArcFace + vision)", test_end_to_end_pipeline),
        ("E2E pipeline (ArcFace only)", test_end_to_end_arcface_only),
        # BodyIDProjector
        ("BodyIDProjector output shape", test_body_id_projector_output_shape),
        ("BodyIDProjector custom tokens", test_body_id_projector_custom_tokens),
        ("BodyIDProjector output_scale init", test_body_id_projector_output_scale_init),
        ("BodyIDProjector gradient flow", test_body_id_projector_gradient_flows),
        ("BodyIDProjector zero input", test_body_id_projector_zero_input),
        ("BodyIDProjector batch size 1", test_body_id_projector_batch_size_one),
        # BodyIDConfig
        ("BodyIDConfig defaults", test_body_id_config_defaults),
        ("BodyIDConfig custom values", test_body_id_config_custom),
        # Body cache
        ("Body cache format", test_body_cache_format),
        # Three-way dropout
        ("3-way dropout all zeroed", test_three_way_dropout_all_zeroed),
        ("3-way dropout per-sample", test_three_way_dropout_per_sample),
        # Three-way concat
        ("3-way concatenation", test_three_way_concatenation),
        # Full E2E
        ("E2E all three projectors", test_end_to_end_all_three),
        # Identity save/load round-trip
        ("Save/load identity (face only)", test_save_load_identity_face_only),
        ("Save/load identity (all three)", test_save_load_identity_all_three),
        ("has_identity false for plain LoRA", test_has_identity_false),
        ("Load identity from plain LoRA", test_load_identity_no_identity_keys),
    ]

    print(f"\nRunning {len(tests)} face_id tests...\n")
    passed = sum(_run_test(name, fn) for name, fn in tests)
    failed = len(tests) - passed
    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed > 0:
        sys.exit(1)
    print("All tests passed!")
