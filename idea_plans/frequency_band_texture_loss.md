# Frequency Band Texture Loss

Preserve texture characteristics (skin quality, hair density, fabric weave) without memorizing pixel positions. Compare FFT power spectra of frequency bands between reference and x0 prediction.

## Motivation

Current identity losses operate on semantic features (ArcFace embeddings, skeleton ratios). There's no signal for texture fidelity — the model can generate the right face shape but with wrong skin quality, hair texture, or fabric detail. A pixel-level MSE loss would memorize; a frequency-domain loss preserves the statistical character of textures without spatial correspondence.

## Core Idea

Extract high-frequency band from images, compare their power spectra. The power spectrum captures "what frequencies are present" (texture character) but discards phase (spatial arrangement). This is the "this person has smooth skin and fine hair" signal without "this exact mole is at pixel (234, 567)".

```
ref_high = ref_image - gaussian_blur(ref_image, sigma)
ref_spectrum = log(|FFT(ref_high)|^2 + eps)      # cached per image

pred_high = x0_pred - gaussian_blur(x0_pred, sigma)
pred_spectrum = log(|FFT(pred_high)|^2 + eps)     # computed each step

loss = mse(pred_spectrum, ref_spectrum)
```

## Design

### Frequency Bands

Three possible granularities (start with single band, extend later):

1. **Single band (simplest):** high-freq = image - blur(sigma=3). Captures all texture above ~3px scale.
2. **Two bands:** mid-freq (blur(sigma=1) - blur(sigma=4)) for feature-scale texture (eyelashes, eyebrows, lip texture), high-freq (image - blur(sigma=1)) for fine texture (skin pores, fabric weave).
3. **Laplacian pyramid (full):** Multiple octaves, each weighted independently.

### Power Spectrum Computation

```python
def compute_texture_spectrum(image, blur_sigma=3.0):
    """Extract texture power spectrum from an image.
    
    Args:
        image: (B, 3, H, W) in [0, 1]
        blur_sigma: Gaussian blur sigma for low-freq extraction
    Returns:
        spectrum: (B, 3, H, W) log power spectrum of high-freq band
    """
    low = gaussian_blur(image, sigma=blur_sigma)
    high = image - low
    # 2D FFT per channel, shift zero-freq to center
    fft = torch.fft.fft2(high)
    fft = torch.fft.fftshift(fft)
    power = (fft.real ** 2 + fft.imag ** 2).clamp(min=1e-10)
    return torch.log(power)
```

Key properties:
- Fully differentiable (torch.fft.fft2, gaussian blur via conv2d)
- ~2ms for 512x512 on GPU
- Phase discarded -> no spatial memorization
- Log scale -> balances low and high magnitude frequencies

### Caching

Per-image, store in existing `_face_id_cache/` safetensors:
- `texture_spectrum_{sigma}`: (3, H, W) log power spectrum at given blur sigma
- Need to handle variable resolutions: either resize spectra to fixed size, or compute at multiple resolutions and match at training time

Resolution handling options:
- **Option A:** Cache at original resolution, resize spectrum to match x0_pred at training time. Spectra are smooth so resizing is fine.
- **Option B:** Cache at a fixed resolution (e.g., 256x256). Simpler but loses resolution-dependent texture info.
- **Option C:** Cache at multiple resolutions. Most accurate but more storage.

Recommend Option A — cache at original res, resize at training time.

### Training Integration

#### Config (FaceIDConfig)
```python
self.texture_loss_weight: float = 0.0        # 0 = disabled
self.texture_loss_min_t: float = 0.3         # only apply at medium noise
self.texture_loss_max_t: float = 0.5         # where texture decisions happen
self.texture_loss_blur_sigma: float = 3.0    # freq separation point
self.texture_loss_high_sigma: float = 0.0    # if >0, also add fine-texture band
```

#### Per-dataset overrides (DatasetConfig)
```python
self.texture_loss_weight: Union[float, None] = None  # inherit global
```

#### Loss computation (SDTrainer)
```python
if _need_texture_loss:
    # Only at medium timesteps where texture is being decided
    if texture_noise_mask.any():
        ref_spectrum = batch.texture_spectrum.to(x0_pixels.device)
        pred_spectrum = compute_texture_spectrum(x0_pixels, sigma=blur_sigma)
        
        # Resize ref spectrum to match pred if needed
        if ref_spectrum.shape != pred_spectrum.shape:
            ref_spectrum = F.interpolate(ref_spectrum, size=pred_spectrum.shape[-2:])
        
        texture_loss = F.mse_loss(pred_spectrum, ref_spectrum, reduction='none')
        texture_loss = texture_loss.mean(dim=(1, 2, 3))  # per-sample
        texture_loss = texture_loss * t_weight * texture_valid_mask.float()
```

### Timestep Window Rationale

- **t > 0.5:** x0_pred is too noisy for meaningful texture extraction. High-freq band is dominated by prediction noise, not texture.
- **t = 0.3-0.5:** Model has resolved structure but texture details are still being decided. This is where the signal is most useful — it guides the model's texture choices.
- **t < 0.3:** x0_pred texture is mostly determined. Signal is redundant with the diffusion MSE loss at this point.
- Could extend lower (0.1-0.5) if the loss is gentle enough. The power spectrum is robust to mild noise.

### What This Preserves vs. Doesn't

| Preserved | Not Preserved |
|-----------|---------------|
| Skin texture quality (smooth vs rough) | Exact freckle/mole positions |
| Hair strand density and character | Individual hair strand placement |
| Fabric weave pattern type | Exact fabric fold positions |
| Overall sharpness/softness | Specific edge locations |
| Frequency distribution of details | Spatial arrangement of details |

### Interaction with Existing Losses

- **Diffusion MSE:** Texture loss is complementary. MSE preserves global structure; texture loss preserves frequency statistics. They operate on orthogonal aspects.
- **Identity loss (ArcFace):** Identity is semantic (face shape, eye color); texture is sub-semantic (skin quality, hair density). No overlap.
- **Body proportion loss:** Operates on skeleton ratios, completely orthogonal to texture.
- **Face suppression weight:** If face region has suppressed diffusion loss, texture loss could fill the gap by preserving face texture without pixel-level correspondence.

### Extension: Per-Region Texture Loss

Could combine with face bboxes / body segmentation to apply different texture weights to different regions:
- Face region: higher weight (skin texture is identity-specific)
- Hair region: medium weight (hair texture varies with styling)
- Clothing region: lower weight (clothing varies between images)
- Background: zero weight

### Extension: Multi-Band

Two-band version with independent controls:

```yaml
face_id:
  texture_loss_weight: 0.01
  texture_loss_min_t: 0.3
  texture_loss_max_t: 0.5
  texture_loss_bands:
    - sigma: 1.0    # fine texture (pores, fabric weave)
      weight: 0.5
    - sigma: 4.0    # mid texture (eyelashes, hair strands)
      weight: 1.0
```

### Files to Modify

1. **`toolkit/texture_loss.py`** (new) — `compute_texture_spectrum()`, `TextureSpectrumEncoder` class
2. **`toolkit/config_modules.py`** — Add texture loss config fields to FaceIDConfig and DatasetConfig
3. **`toolkit/data_transfer_object/data_loader.py`** — Add `texture_spectrum` field to FileItemDTO and DataLoaderBatchDTO
4. **`extensions_built_in/sd_trainer/SDTrainer.py`** — Cache texture spectra at startup, compute loss in training loop
5. **`ui/src/types.ts`** + **`ui/src/app/jobs/new/jobConfig.ts`** + **`ui/src/app/jobs/new/SimpleJob.tsx`** — UI controls

### Open Questions

1. **Optimal blur sigma:** 3.0 is a starting point. Might need tuning per resolution — sigma=3 at 512px captures different texture scales than at 1024px. Could normalize by resolution.
2. **Per-channel vs grayscale spectrum:** Color textures (skin hue variation, hair color variation) might be useful to preserve. Start with per-channel (3 separate spectra), could try grayscale if too noisy.
3. **Spectrum comparison metric:** MSE on log-spectra is the simplest. Could also try cosine similarity on flattened spectra, or Earth Mover's Distance on radially-averaged spectra. Start with MSE.
4. **Interaction with VAE:** x0_pred goes through TAESD decode, which has its own frequency characteristics. The texture spectrum comparison should be robust to this since both ref and pred go through similar decode paths (though ref is cached from clean images, not decoded).
5. **Average mode:** Should we compare against the average texture spectrum (like average identity), or always per-image? Per-image makes more sense since texture varies less than identity across images of the same person.
