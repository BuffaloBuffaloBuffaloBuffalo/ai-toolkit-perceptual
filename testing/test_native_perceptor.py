"""Codec routing, gradient and cache regressions (CPU, no model downloads)."""
from types import SimpleNamespace

import pytest
import torch

from toolkit.perceptor_utils import decode_native_perceptor_file, perceptor_cache_suffix


class FakeItem:
    def __init__(self, path, latent=None):
        self.path = str(path)
        self.latent = latent
        self.is_video = False
        self.flip_x = False
        self.tensor = None

    def get_latent(self):
        return self.latent

    def get_latent_info_dict(self):
        return dict(crop_width=32, crop_height=32, flip_x=self.flip_x)

    def load_and_process_image(self, transform, only_load_latents=False):
        assert only_load_latents
        self.tensor = transform(torch.ones(3, 8, 8))


class FakeNativeModel:
    perceptor_decode_mode = 'native_video'
    is_flow_matching = True
    noise_scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))

    def encode_images(self, images, device=None):
        self.encoded = images
        return images.unsqueeze(2).repeat(1, 8, 1, 1, 1)

    def decode_latents(self, latents, device=None, dtype=None):
        self.decoded = latents
        return latents[:, :3].to(device=device, dtype=dtype)


def test_native_target_uses_cached_training_latent(tmp_path):
    item = FakeItem(tmp_path / 'image.jpg', torch.full((24, 1, 4, 4), 0.4))
    model = FakeNativeModel()
    result = decode_native_perceptor_file(model, item, None, 'cpu')
    assert result.shape == (1, 3, 4, 4)
    assert torch.allclose(result, torch.full_like(result, 0.7))
    assert not hasattr(model, 'encoded')


def test_native_target_fallback_uses_dataset_transform_without_mutation(tmp_path):
    item = FakeItem(tmp_path / 'image.jpg')
    model = FakeNativeModel()
    result = decode_native_perceptor_file(model, item, lambda x: x * -0.5, 'cpu')
    assert result.shape == (1, 3, 8, 8)
    assert torch.all(result == 0.25)
    assert torch.all(model.encoded == -0.5)
    assert item.tensor is None


def test_corrupt_training_latent_is_not_silently_reencoded(tmp_path):
    item = FakeItem(tmp_path / 'image.jpg')
    def broken():
        raise RuntimeError('corrupt training latent')
    item.get_latent = broken
    with pytest.raises(RuntimeError, match='corrupt training latent'):
        decode_native_perceptor_file(FakeNativeModel(), item, None, 'cpu')


def test_native_x0_gradients_and_shared_decode():
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    trainer = SDTrainer.__new__(SDTrainer)
    trainer.sd = FakeNativeModel()
    trainer.step_num = 1
    trainer._video_x0_frames_cache = None
    trainer._ensure_wan_depth_decoder = lambda: pytest.fail('native path loaded tiny decoder')
    velocity = torch.full((1, 24, 2, 4, 4), 0.1, requires_grad=True)
    noisy = torch.full_like(velocity, 0.2)
    t = torch.tensor([250.])
    preview = trainer._get_video_x0_frames(velocity, noisy, t, needs_grad=False)
    assert not preview.requires_grad
    frames = trainer._get_video_x0_frames(velocity, noisy, t, needs_grad=True)
    assert frames.requires_grad
    assert torch.allclose(frames, torch.full_like(frames, (0.2 - 0.25 * 0.1 + 1) / 2))
    assert trainer._get_video_x0_frames(velocity, noisy, t, True) is frames
    frames.square().mean().backward()
    assert torch.isfinite(velocity.grad).all() and velocity.grad.abs().sum() > 0


def test_krea_tiny_decode_preserves_gradient(monkeypatch):
    import importlib
    module = importlib.import_module('extensions_built_in.sd_trainer.SDTrainer')
    trainer = module.SDTrainer.__new__(module.SDTrainer)
    trainer.sd = SimpleNamespace(arch='krea2')
    trainer.train_config = SimpleNamespace(gradient_checkpointing=True)
    trainer._wan_depth_decoder = object()
    trainer._ensure_wan_depth_decoder = lambda: None
    monkeypatch.setattr(module, 'decode_wan_x0_to_frames', lambda x, _: x[:, :3].sigmoid())
    latent = torch.randn(1, 16, 4, 4, requires_grad=True)
    pixels = trainer._decode_krea_x0(latent)
    assert pixels.shape == (1, 3, 4, 4)
    pixels.square().mean().backward()
    assert latent.grad.abs().sum() > 0


def test_native_x0_cache_isolated_between_predictions_in_one_optimizer_step():
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    trainer = SDTrainer.__new__(SDTrainer)
    trainer.sd = FakeNativeModel()
    trainer.step_num = 1
    trainer._video_x0_frames_cache = None
    for value in [0.1, 0.3, -0.2, 0.05]:
        velocity = torch.full((1, 24, 1, 4, 4), value, requires_grad=True)
        noisy = torch.full_like(velocity, 0.2)
        t = torch.tensor([250.])
        frames = trainer._get_video_x0_frames(velocity, noisy, t, True)
        torch.testing.assert_close(frames, torch.full_like(frames, (0.2 - 0.25 * value + 1) / 2))
        frames.square().mean().backward()
        assert torch.isfinite(velocity.grad).all() and velocity.grad.abs().sum() > 0


def test_h3_tiled_decode_checkpoint_preserves_output_and_gradient():
    from extensions_built_in.diffusion_models.minimax_h3.src.vae import MiniMaxH3VideoVAE
    model = MiniMaxH3VideoVAE.__new__(MiniMaxH3VideoVAE)
    torch.nn.Module.__init__(model)
    model.use_tiling = True
    model.tile_size = 4
    model.tile_overlap_min = 2
    model.spatial_compression = 1
    model.post_quant_conv = torch.nn.Conv3d(2, 2, 1)
    model.decoder = torch.nn.Sequential(torch.nn.Conv3d(2, 3, 1), torch.nn.Tanh())
    model.requires_grad_(False)
    z = torch.randn(1, 2, 2, 8, 8, requires_grad=True)
    model.decoder.gradient_checkpointing = False
    reference = model._decode_clip(z)
    expected_grad, = torch.autograd.grad(reference.square().mean(), z)
    model.decoder.gradient_checkpointing = True
    checked = model._decode_clip(z)
    actual_grad, = torch.autograd.grad(checked.square().mean(), z)
    torch.testing.assert_close(checked, reference)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert torch.isfinite(actual_grad).all() and actual_grad.abs().sum() > 0


def test_cache_namespace_isolates_codec_flips_settings_and_source(tmp_path):
    path = tmp_path / 'image.jpg'
    path.write_bytes(b'fixture')
    item = FakeItem(path)
    a = perceptor_cache_suffix(item, 'h3_native_v1', input_size=252)
    assert a != perceptor_cache_suffix(item, 'wan_tiny', input_size=252)
    assert a != perceptor_cache_suffix(item, 'h3_native_v1', input_size=518)
    item.flip_x = True
    assert a != perceptor_cache_suffix(item, 'h3_native_v1', input_size=252)
    item.flip_x = False
    path.write_bytes(b'new fixture')
    assert a != perceptor_cache_suffix(item, 'h3_native_v1', input_size=252)


def test_native_depth_cache_handles_still_and_decoded_frame_count(tmp_path, monkeypatch):
    from toolkit import depth_consistency as depth
    from toolkit.config_modules import DepthConsistencyConfig
    path = tmp_path / 'image.jpg'
    path.write_bytes(b'fixture')
    item = FakeItem(path)
    monkeypatch.setattr(depth, 'DifferentiableDepthEncoder', lambda **_: lambda x: x.mean(1))
    calls = []
    def decode(fi):
        calls.append(fi)
        return torch.rand(5, 3, 8, 8)
    kwargs = dict(device='cpu', num_frames=22, decode_file_fn=decode,
                  cache_namespace='h3_native_v1', include_images=True)
    cfg = DepthConsistencyConfig(input_size=28)
    depth.cache_video_depth_gt_embeddings([item], cfg, **kwargs)
    assert item.is_depth_video_cached and len(calls) == 1
    depth.cache_video_depth_gt_embeddings([item], cfg, **kwargs)
    assert len(calls) == 1  # decoder, not requested input T, defines target T
    item.flip_x = True
    depth.cache_video_depth_gt_embeddings([item], cfg, **kwargs)
    assert len(calls) == 2


def test_native_identity_cache_never_loads_wan_decoder(tmp_path, monkeypatch):
    from toolkit import face_id, depth_consistency
    from toolkit.config_modules import FaceIDConfig
    path = tmp_path / 'image.jpg'
    path.write_bytes(b'fixture')
    item = FakeItem(path)
    fake_extractor = SimpleNamespace(_detect=lambda _: ([], None))
    monkeypatch.setattr(face_id, 'FaceIDExtractor', lambda **_: fake_extractor)
    monkeypatch.setattr(face_id, 'DifferentiableFaceEncoder', lambda: torch.nn.Identity())
    monkeypatch.setattr(depth_consistency, 'load_taehv_wan21', lambda **_: pytest.fail('loaded Wan'))
    calls = []
    def decode(fi):
        calls.append(fi)
        return torch.zeros(1, 3, 8, 8)
    kwargs = dict(device='cpu', arch='minimax_h3', num_frames=1, include_images=True,
                  decode_file_fn=decode, cache_namespace='h3_native_v1')
    face_id.cache_video_identity_embeddings([item], FaceIDConfig(), **kwargs)
    assert item.identity_gt_video.shape == (1, 512)
    assert item.identity_gt_video_valid.sum() == 0
    face_id.cache_video_identity_embeddings([item], FaceIDConfig(), **kwargs)
    assert len(calls) == 1


@pytest.mark.parametrize('source', ['subject', 'body'])
def test_native_depth_mask_selects_resizes_and_repeats_spatial_mask(source):
    from toolkit.depth_consistency import depth_mask_for_frames
    subject = torch.tensor([[[[1., 0.], [0., 0.]]]])
    body = 1 - subject
    batch = SimpleNamespace(subject_masks=subject, body_masks=body)
    frames = torch.zeros(3, 4, 4)
    mask = depth_mask_for_frames(batch, source, 0, frames)
    selected = subject if source == 'subject' else body
    expected = torch.nn.functional.interpolate(
        selected, size=(4, 4), mode='bilinear', align_corners=False,
    )[0].expand(3, -1, -1)
    torch.testing.assert_close(mask, expected)
    assert depth_mask_for_frames(batch, 'none', 0, frames) is None
    assert depth_mask_for_frames(SimpleNamespace(), source, 0, frames) is None


def test_native_image_depth_mask_excludes_background_loss_and_gradient():
    from toolkit.depth_consistency import depth_mask_for_frames, ssi_l1, multiscale_grad_loss
    target = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64
    mask = torch.zeros(1, 1, 8, 8)
    mask[:, :, :4, :4] = 1
    pred = target.clone()
    pred[4:, 4:] += torch.arange(16).reshape(4, 4) * 10
    pred = pred.unsqueeze(0).requires_grad_()
    frame_mask = depth_mask_for_frames(SimpleNamespace(subject_masks=mask), 'subject', 0, pred)[0]
    ssi, scale, shift = ssi_l1(pred[0], target, frame_mask)
    grad = multiscale_grad_loss(scale[0] * pred[0] + shift[0], target, frame_mask, scales=2)
    loss = ssi + grad
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-5, rtol=0)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[0, 4:, 4:].abs().sum() == 0
    assert ssi_l1(pred.detach()[0], target)[0] > 0
