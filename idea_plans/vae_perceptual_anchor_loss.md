# VAE Perceptual Anchor Loss

Use VAE encoder/decoder features as perceptual loss anchors during diffusion training. Three tiers from simplest to most novel.

## Motivation

Current auxiliary losses operate on semantic features (ArcFace → identity, ViTPose → skeleton). There's no signal for general structural/perceptual quality. Traditional VGG perceptual loss requires decoding to pixels (expensive) and uses features trained for classification (not reconstruction). VAE features are trained to capture what matters for image reconstruction — exactly the right objective for a perceptual quality anchor.

## Prior Art

- **E-LatentLPIPS** (ECCV 2024, arXiv 2405.05967): LPIPS adapted for latent space. 9.7x faster, 25x less memory. Pretrained weights for SDXL 4-ch, SD3 16-ch, FLUX 16-ch. Uses augmentation ensemble to handle "blind spots." GitHub: `mingukkang/elatentlpips`
- **LPL / Latent Perceptual Loss** (ICLR 2025, arXiv 2411.04873): Uses VAE decoder intermediate features. Layers 3-4 most impactful. 6-20% FID improvement. Zero extra model loading.
- **REPA-E** (ICCV 2025, arXiv 2504.10483): End-to-end VAE+diffusion co-training with representation alignment. SOTA FID 1.12 on ImageNet 256.

## Three Tiers

### Tier 1: Same-Model Latent Comparison (Cheapest)

Compare x0_pred latents against target latents in the training model's own latent space. No decode needed, no extra model.

```
x0_pred_latents ──→ E-LatentLPIPS ──→ loss
target_latents  ──→      ↑
```

**What it provides:**
- Perceptually-weighted latent comparison (learned per-channel importance)
- Much better than raw MSE on latents (which treats all channels/positions equally)
- Augmentation ensemble handles blind spots (rotation, flip, color jitter on latents)

**Implementation:**
```python
# x0_pred and target are already available as latents during training
# No decode step needed
from elatentlpips import ELatentLPIPS

lpips_model = ELatentLPIPS(model_type='flux')  # pretrained for 16-ch
lpips_model.eval()
lpips_model.requires_grad_(False)

# During training:
perceptual_loss = lpips_model(x0_pred_latents, target_latents)  # (B,) or scalar
```

**Cost:** ~12ms per batch (vs ~117ms for pixel-space LPIPS). 0.6GB extra memory.
**Applicability:** Any model — just pick the right E-LatentLPIPS variant for the latent channel count.
**Limitation:** Can't provide cross-model signal (both latents are in the same space).

### Tier 2: Decoder Feature Matching (Free Infrastructure)

Compare intermediate features from the VAE decoder when decoding x0_pred vs target. The decoder is already loaded — just register forward hooks.

```
x0_pred_latents ──→ VAE Decoder ──→ layer3_features, layer4_features ──→ MSE ──→ loss
target_latents  ──→ VAE Decoder ──→ layer3_features, layer4_features ──→  ↑
```

**What it provides:**
- Multi-scale perceptual comparison at different abstraction levels
- Layer 3: mid-level features (edges, textures, spatial structure)
- Layer 4: high-level features (object parts, composition)
- Captures what the decoder "thinks is important" for reconstruction

**Implementation:**
```python
class DecoderFeatureLoss:
    def __init__(self, vae_decoder):
        self.features = {}
        # Register hooks on decoder's upsampling blocks
        for i, block in enumerate(vae_decoder.up_blocks):
            block.register_forward_hook(
                lambda mod, inp, out, idx=i: self.features.update({idx: out})
            )
    
    def compute(self, x0_pred_latents, target_latents):
        # Forward both through decoder, extract features
        self.features.clear()
        _ = self.decoder(x0_pred_latents)
        pred_features = dict(self.features)
        
        self.features.clear()
        with torch.no_grad():
            _ = self.decoder(target_latents)
        target_features = dict(self.features)
        
        # Compare at layers 3 and 4 (most impactful per LPL paper)
        loss = 0
        for layer_idx in [2, 3]:  # 0-indexed
            pred_f = F.normalize(pred_features[layer_idx], dim=1)
            tgt_f = F.normalize(target_features[layer_idx], dim=1)
            loss += F.mse_loss(pred_f, tgt_f)
        return loss
```

**Cost:** One extra decoder forward pass per step (~50-200ms depending on VAE). But decoder is already loaded.
**Applicability:** Any model with a loaded VAE decoder. Best when decoder is already in VRAM.
**Limitation:** Requires full VAE decode (not just TAESD). More expensive than Tier 1.

**Optimization:** Only decode through the first few layers (stop before final upsample). This captures the perceptual features without the full decode cost.

### Tier 3: Cross-Model VAE Anchor (Most Novel)

Use a superior model's VAE encoder as a frozen perceptual anchor. The canonical example: Flux 2 VAE (32ch) as anchor for SDXL training (4ch).

```
x0_pred (pixels, from SDXL) ──→ Flux2_VAE.encode() ──→ latent_pred (32ch) ──→ loss
reference (pixels)           ──→ Flux2_VAE.encode() ──→ latent_ref  (32ch) ──→  ↑  [cached]
```

**What it provides:**
- A richer perceptual space (32ch captures more than 4ch)
- Artifacts that SDXL's own VAE tolerates (because it can't represent them) show up in Flux 2's latent space
- Cross-model quality gradient — "how does a better model perceive this output?"
- Independent of the training model's own latent space

**Implementation:**

#### Caching (once per reference image):
```python
# Load Flux 2 VAE encoder
from extensions_built_in.diffusion_models.flux2.src.autoencoder import AutoEncoder, AutoEncoderParams
anchor_vae = AutoEncoder(AutoEncoderParams())
anchor_vae.load_state_dict(load_file("flux2_vae.safetensors"))
anchor_vae.eval()
anchor_vae.requires_grad_(False)
anchor_vae.to(device)

# Cache reference latents (per image, stored in _face_id_cache/)
ref_pixels = load_and_preprocess(image)  # (1, 3, H, W) in [0, 1]
ref_latent = anchor_vae.encode(ref_pixels * 2 - 1)  # encode expects [-1, 1]
# Store as 'anchor_vae_latent' in safetensors cache
```

#### Training:
```python
# x0_pixels already decoded by TAESD/TAEF2 for other losses
# Now encode through anchor VAE
with torch.no_grad():  # anchor VAE is frozen, but we need grad through x0_pixels
    pass  # Actually, we DO want grad through x0_pixels → anchor_vae.encode
    
anchor_pred = anchor_vae.encode(x0_pixels * 2 - 1)  # (B, 32, H/8, W/8)

# Load cached reference (from batch)
anchor_ref = batch.anchor_vae_embedding  # (B, 32, H/8, W/8)

# Resize if needed (x0_pixels may be different resolution than cached ref)
if anchor_pred.shape != anchor_ref.shape:
    anchor_ref = F.interpolate(anchor_ref, size=anchor_pred.shape[-2:], mode='bilinear')

# Comparison: per-channel normalized MSE
anchor_pred_n = F.normalize(anchor_pred.flatten(2), dim=-1)  # (B, 32, H*W/64) → norm per spatial
anchor_ref_n = F.normalize(anchor_ref.flatten(2), dim=-1)
anchor_loss = 1.0 - F.cosine_similarity(anchor_pred_n, anchor_ref_n, dim=-1).mean(dim=1)  # (B,)
```

**Cost:** ~50-100ms for Flux 2 VAE encode at 512px. ~160MB extra VRAM for encoder.
**Applicability:** Most useful when training a model with a weaker VAE (SDXL 4ch) using a stronger VAE (Flux 32ch) as anchor.
**Limitation:** Requires decoding to pixels first (using TAESD), then re-encoding through anchor VAE. Two-step process.

#### Differentiability concern:
The anchor VAE encode IS differentiable (standard conv network). But x0_pixels comes from TAESD decode, which is also differentiable. So the gradient chain is:

```
loss → anchor_vae.encode(x0_pixels) → x0_pixels = TAESD(x0_pred_latents) → x0_pred_latents
```

Full gradient flow from anchor loss back to the diffusion model's prediction. Same chain as ArcFace identity loss.

#### Multi-scale anchor features:
Instead of comparing just the final encoder output, extract intermediate features using hooks:

```python
# Register hooks on anchor VAE encoder's downsampling blocks
# Compare at multiple scales for richer signal
anchor_features_pred = extract_multi_scale(anchor_vae.encoder, x0_pixels)
anchor_features_ref = extract_multi_scale(anchor_vae.encoder, ref_pixels)  # cached

multi_scale_loss = sum(
    F.mse_loss(F.normalize(pred, dim=1), F.normalize(ref, dim=1))
    for pred, ref in zip(anchor_features_pred, anchor_features_ref)
)
```

## Recommended Approach

### Start with Tier 1 (E-LatentLPIPS)
- Drop-in, pretrained, validated
- Provides immediate perceptual quality improvement over raw MSE
- No decode needed, minimal compute overhead
- Already handles the "blind spot" problem with augmentation ensemble

### Add Tier 3 for cross-model training
- When training SDXL or older models, Flux 2 VAE anchor provides quality gradient from a superior model
- The signal is "does this look good to a model that understands images better than yours?"
- Most impactful when there's a large quality gap between training model's VAE and anchor VAE

### Tier 2 is a fallback
- Useful when you can't load an extra model (VRAM constrained)
- The decoder features are "free" but the extra decode pass isn't
- Best for fine-tuning runs where full VAE is already in memory

## Config Design

```yaml
face_id:
  # Tier 1: E-LatentLPIPS (same-model latent comparison)
  latent_perceptual_loss_weight: 0.0      # 0 = disabled
  latent_perceptual_loss_min_t: 0.0
  latent_perceptual_loss_max_t: 0.5       # only useful at lower noise

  # Tier 3: Cross-model VAE anchor
  anchor_vae_loss_weight: 0.0             # 0 = disabled
  anchor_vae_path: null                   # path to anchor VAE weights
  anchor_vae_type: "flux2"                # architecture type
  anchor_vae_loss_min_t: 0.0
  anchor_vae_loss_max_t: 0.5
  anchor_vae_multi_scale: true            # use intermediate encoder features
```

Per-dataset overrides follow the existing pattern:
```yaml
datasets:
  - folder_path: /data/photos
    anchor_vae_loss_weight: 0.01          # override for this dataset
```

## Timestep Window

Like identity loss, the perceptual anchor is most useful at **low-to-medium timesteps** where the x0 prediction has meaningful structure:

- **t > 0.7:** x0_pred is too noisy for meaningful perceptual comparison. Skip.
- **t = 0.3-0.7:** Structure is forming. Anchor loss guides structural quality.
- **t < 0.3:** Fine details being decided. Anchor loss guides detail quality.
- **t < 0.1:** Nearly clean. Anchor loss mostly redundant with MSE.

Default window: t=0.0 to t=0.5 (same range where ArcFace and texture loss are useful).

## Interaction with Existing Losses

| Loss | What it preserves | Latent space |
|---|---|---|
| Diffusion MSE | Overall reconstruction | Training model's own latent |
| E-LatentLPIPS (Tier 1) | Perceptual reconstruction quality | Training model's own latent (weighted) |
| Anchor VAE (Tier 3) | Structural quality as perceived by better model | Anchor model's latent |
| ArcFace | Face identity | ArcFace embedding (512-d) |
| ViTPose | Body proportions | Skeleton ratios (8-10 values) |
| Texture spectrum | Frequency characteristics | FFT power spectrum |

These are all orthogonal — each operates in a different representation space measuring a different property. No redundancy.

## Files to Modify

### Tier 1 (E-LatentLPIPS):
1. **`requirements.txt`** — add `elatentlpips` dependency
2. **`toolkit/config_modules.py`** — add `latent_perceptual_loss_weight`, `_min_t`, `_max_t` to FaceIDConfig
3. **`extensions_built_in/sd_trainer/SDTrainer.py`** — load E-LatentLPIPS model, add loss computation block
4. **`toolkit/data_transfer_object/data_loader.py`** — no changes needed (operates on latents already in batch)

### Tier 3 (Cross-model anchor):
1. **`toolkit/vae_anchor.py`** (new) — `AnchorVAEEncoder` class, multi-scale feature extraction, caching
2. **`toolkit/config_modules.py`** — add anchor VAE config fields
3. **`toolkit/data_transfer_object/data_loader.py`** — add `anchor_vae_embedding` field
4. **`extensions_built_in/sd_trainer/SDTrainer.py`** — cache anchor embeddings at startup, compute loss in training loop

## Open Questions

1. **E-LatentLPIPS augmentation ensemble:** The paper uses 64 augmentations per comparison to handle blind spots. Is that too expensive for per-step training? Could reduce to 4-8 augmentations.

2. **Anchor VAE resolution mismatch:** If training at 512px but anchor VAE was designed for 1024px, do the latent features transfer? VAEs are generally resolution-flexible (fully convolutional), but quality may degrade.

3. **Anchor VAE for video:** For Wan 2.2 training, could use Flux 2's image VAE on individual frames as a per-frame quality anchor. The 3D temporal VAE handles temporal coherence; the 2D anchor handles per-frame quality.

4. **Gradient through TAESD for Tier 3:** x0_pixels from TAESD is approximate. Errors in TAESD decode propagate to the anchor VAE encode. Is this a problem? Probably minor since both x0_pred and reference go through the same pipeline.

5. **Which is better for Flux training: Tier 1 or Tier 3?** For Flux training with its own 32ch VAE, E-LatentLPIPS (Tier 1) operates in the native latent space — no decode needed. Using Flux's own VAE as a Tier 3 anchor would require decode → re-encode (wasteful). Tier 1 is clearly better for same-model training. Tier 3 is for cross-model (training SDXL with Flux anchor).

6. **Codebase already has DiffusionFeatureExtractor2:** This existing module extracts multi-scale features from 32-ch latents. Could be adapted for the anchor VAE feature extraction instead of building from scratch.
