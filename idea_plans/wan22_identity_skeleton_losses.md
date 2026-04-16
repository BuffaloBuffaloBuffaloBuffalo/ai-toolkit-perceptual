# Wan 2.2 Identity + Skeleton Loss Integration

Extend the existing identity (ArcFace) and body proportion (ViTPose) losses to work with Wan 2.2 video models, leveraging the dual-transformer MoE architecture to apply the right loss to the right expert.

## Current State

### What already works
- Wan 2.2 LoRA training: 14B T2V, 14B I2V, 5B TI2V
- Dual-transformer MoE with boundary switching (`switch_boundary_every`)
- Split LoRA saves (`_high_noise.safetensors`, `_low_noise.safetensors`)
- Video dataset loading with frame extraction
- All identity/skeleton losses work on single images

### What's missing
- x0 decoding for video latents (5D → per-frame pixel-space)
- Per-frame face detection and ArcFace embedding
- Per-frame skeleton extraction and ratio comparison
- Temporal consistency enforcement
- Memory-efficient frame subsampling for discriminator losses
- Expert-aware loss routing (skeleton → high-noise, identity → low-noise)

## Architecture

### Wan 2.2 MoE Structure

```
Timestep > boundary (0.875):  High-noise expert (14B transformer_1)
                              → Handles layout, structure, body pose, composition
                              → Skeleton/proportion loss most useful here

Timestep < boundary (0.875):  Low-noise expert (14B transformer_2)
                              → Handles detail refinement, texture, identity features
                              → ArcFace identity loss most useful here
```

### Video x0 Recovery

For flow matching: `x0 = noisy_latents - sigma * v_pred` where latents are `(B, C, T, H, W)`.

The 3D VAE decodes to `(B, 3, T_out, H_out, W_out)` where:
- T_out = (T-1)*4 + 1 (temporal upscale 4x)
- H_out = H * 8 (spatial upscale 8x for 14B VAE)
- W_out = W * 8

For discriminator losses, we need individual frames: `(T_out, 3, H_out, W_out)`.

### Memory Constraints

At 24GB VRAM with 4-bit quantized 14B model:
- Model: ~7GB quantized
- Latents + gradients: ~4-6GB for 40 frames at 480P
- VAE decode of x0: ~2-3GB for full video
- ArcFace + ViTPose per frame: ~0.5GB
- Headroom: ~8-10GB available

**Key insight:** We cannot decode ALL frames. Must subsample.

## Design

### Frame Subsampling Strategy

Instead of processing all T frames through discriminator models, subsample K frames (configurable, default 4):

```python
# Select frames to evaluate: first, last, and evenly spaced middle frames
n_frames = x0_pixels.shape[2]  # temporal dim
if n_frames <= max_eval_frames:
    eval_indices = list(range(n_frames))
else:
    eval_indices = np.linspace(0, n_frames - 1, max_eval_frames, dtype=int).tolist()

# Extract individual frames for discriminator losses
eval_frames = x0_pixels[:, :, eval_indices]  # (B, 3, K, H, W)
eval_frames = rearrange(eval_frames, 'b c k h w -> (b k) c h w')  # (B*K, 3, H, W)
```

This keeps memory bounded regardless of video length.

### Config Extensions

```yaml
face_id:
  # Existing image-mode settings (all still work)
  identity_loss_weight: 0.01
  body_proportion_loss_weight: 0.01
  
  # New video-specific settings
  video_eval_frames: 4              # max frames to evaluate per step
  video_identity_loss_weight: null  # if set, overrides for video (null = use global)
  video_skeleton_loss_weight: null  # if set, overrides for video
  
  # Expert routing (Wan 2.2 dual-transformer only)
  skeleton_expert: "high"           # which expert gets skeleton loss: "high", "low", "both"
  identity_expert: "low"            # which expert gets identity loss: "high", "low", "both"
  
  # Temporal consistency
  temporal_identity_weight: 0.0     # cross-frame identity consistency loss
  temporal_skeleton_weight: 0.0     # cross-frame skeleton smoothness loss
```

### Expert-Aware Loss Routing

During training, the trainer already knows which expert is active (via `timestep > boundary`). The loss routing just gates on this:

```python
# In SDTrainer, during loss computation:
if is_video and self.sd.is_dual_transformer:
    active_expert = "high" if t_ratio.mean() > boundary else "low"
    
    _apply_skeleton = (
        skeleton_expert == "both" or skeleton_expert == active_expert
    )
    _apply_identity = (
        identity_expert == "both" or identity_expert == active_expert
    )
```

This requires no architecture changes — just conditional gating in the loss block.

### Video x0 Decode Path

The existing TAESD/TAEF2 decoder handles 2D images. For video, we need the full 3D VAE decode or a frame-wise approach:

**Option A: Full 3D VAE decode** (accurate but expensive)
```python
x0_pixels = self.sd.vae.decode(x0_latents).sample  # (B, 3, T, H, W)
```
- Pro: preserves temporal coherence from VAE
- Con: ~2-3GB VRAM, slow for 40+ frames
- Would need chunked decode (VAE supports this via causal temporal convolutions)

**Option B: Frame-wise TAESD decode** (fast, approximate)
```python
# Reshape to treat frames as batch
x0_2d = rearrange(x0_latents[:, :, eval_indices], 'b c k h w -> (b k) c h w')
x0_pixels = self.taesd.decode(x0_2d)  # (B*K, 3, H, W)
```
- Pro: fast, low memory, reuses existing TAESD
- Con: ignores temporal latent correlations, may have artifacts
- For discriminator losses (not pixel-level), this is probably fine

**Recommendation:** Option B for training (fast, good enough for discriminator losses). Option A only for preview generation.

### Identity Loss on Video Frames

Extends existing identity loss to work on decoded video frames:

```python
if _need_id_loss and is_video:
    # Decode subsampled frames
    eval_frames = decode_eval_frames(x0_latents, eval_indices)  # (B*K, 3, H, W)
    
    # Expand face bboxes to match (repeat per frame)
    frame_bboxes = [bbox for bbox in scaled_bboxes for _ in eval_indices]
    
    # Run ArcFace on all eval frames
    gen_embeddings, crops = self.id_loss_model(eval_frames, bboxes=frame_bboxes, return_crops=True)
    
    # Run face detector gate on crops
    face_detected = detect_faces_batch(crops)
    
    # Compare against reference (same ref for all frames)
    ref_expanded = ref_embedding.repeat_interleave(len(eval_indices), dim=0)
    
    # Bias correction + cos_sim (existing code, just on more samples)
    cos_sim = compute_bias_corrected_cos_sim(gen_embeddings, ref_expanded)
    
    # Average across frames for this sample's loss
    cos_sim_per_video = rearrange(cos_sim, '(b k) -> b k', k=len(eval_indices))
    # ... existing loss computation on the per-video average
```

### Skeleton Loss on Video Frames

The skeleton loss is particularly valuable for video because body proportions should be CONSISTENT across frames:

```python
if _need_body_proportion_loss and is_video:
    eval_frames = decode_eval_frames(x0_latents, eval_indices)
    
    # Run ViTPose on all eval frames
    gen_ratios, gen_vis = self.body_proportion_model(
        eval_frames, ref_ratios=ref_ratios_expanded, include_head=include_head)
    
    # Per-frame loss against cached reference ratios
    frame_loss = compute_bp_loss(gen_ratios, ref_ratios_expanded, gen_vis, ref_vis_expanded)
    
    # Reshape to (B, K) and average
    frame_loss = rearrange(frame_loss, '(b k) -> b k', k=len(eval_indices))
    bp_loss = frame_loss.mean(dim=1)  # (B,)
```

### Temporal Consistency Losses

These are NEW losses specific to video — they don't have image-mode equivalents.

**Cross-frame identity consistency:**
```python
if temporal_identity_weight > 0:
    # cos_sim between consecutive frame embeddings (not vs reference)
    frame_embs = rearrange(gen_embeddings, '(b k) d -> b k d', k=K)
    consecutive_cos = F.cosine_similarity(frame_embs[:, :-1], frame_embs[:, 1:], dim=-1)
    # Penalize inconsistency: loss = (1 - mean_consecutive_cos)
    temporal_id_loss = (1.0 - consecutive_cos).mean()
```

**Cross-frame skeleton smoothness:**
```python
if temporal_skeleton_weight > 0:
    # L1 between consecutive frame ratios (body proportions shouldn't jump)
    frame_ratios = rearrange(gen_ratios, '(b k) d -> b k d', k=K)
    ratio_jitter = (frame_ratios[:, :-1] - frame_ratios[:, 1:]).abs().mean()
    temporal_bp_loss = ratio_jitter
```

These are cheap (no model inference, just comparing already-computed embeddings/ratios across frames) and directly address the temporal coherence problem.

## Implementation Plan

### Phase 1: Video x0 decode + frame subsampling
**Files:** `SDTrainer.py`
- Add `video_eval_frames` config
- Implement frame-wise TAESD decode for video latents
- Gate by `is_video` (latent has 5 dims)
- Test with Wan 2.1 1.3B (cheapest model)

### Phase 2: Per-frame identity loss
**Files:** `SDTrainer.py`, `face_id.py`
- Expand existing identity loss block to handle video frames
- Repeat face bboxes and reference embeddings per frame
- Face detector gate runs per-frame
- Average cos_sim across frames for the sample's loss
- All existing features work: bias correction, clean targets, per-dataset weights

### Phase 3: Per-frame skeleton loss
**Files:** `SDTrainer.py`, `body_id.py`
- Same pattern as identity: expand, run ViTPose per frame, average
- Head keypoints included when configured

### Phase 4: Expert routing for Wan 2.2
**Files:** `SDTrainer.py`, `config_modules.py`
- Add `skeleton_expert` and `identity_expert` config
- Gate loss computation by active expert
- Default: skeleton → high-noise, identity → low-noise

### Phase 5: Temporal consistency losses
**Files:** `SDTrainer.py`, `config_modules.py`
- Add cross-frame identity and skeleton consistency losses
- These are video-only (gated by `is_video`)
- Lightweight — just comparisons between already-computed frame embeddings

### Phase 6: Preview generation
**Files:** `SDTrainer.py`
- Video previews with skeleton overlays per frame
- Side-by-side: reference skeleton vs predicted skeleton
- Identity crops shown for each evaluated frame

## Low-Resolution Motion Training (256px)

For skeleton/motion training, 256px is sufficient:
- ViTPose input is 256x256 anyway — no downscale needed
- ArcFace crops to 112x112 — works fine from 256px source
- Skeleton ratios are resolution-invariant (normalized bone lengths)
- Latent tokens drop ~16x vs 512px → much more frames fit in VRAM
- Could train ALL frames instead of subsampling 4

This enables a two-stage or mixed training approach:
- **Low-res dataset (256px):** Focus on motion, body proportion, temporal consistency. Use skeleton loss + temporal losses. Cheap enough to use many frames.
- **High-res dataset (512-1024px):** Focus on identity, texture, detail. Use ArcFace identity loss + texture loss. Fewer frames, higher resolution.

Or combine in one run using per-dataset resolution overrides (already supported).

## Memory Budget (24GB, Wan 2.2 14B quantized, 40 frames at 480P)

| Component | VRAM | Notes |
|-----------|------|-------|
| Model (4-bit) | ~7 GB | Quantized dual-transformer |
| Latents + grad | ~5 GB | 40 frames |
| TAESD decode (4 frames) | ~0.3 GB | Frame-wise, not full video |
| ArcFace (4 frames) | ~0.2 GB | Small model, 112x112 crops |
| ViTPose (4 frames) | ~0.3 GB | 256x256 input |
| Face detector (4 frames) | ~0.1 GB | SCRFD, 160x160 |
| **Total discriminator overhead** | **~0.9 GB** | For 4 eval frames |

This fits comfortably within the ~8-10GB headroom. Could increase `video_eval_frames` to 8 with minimal impact.

## Open Questions

1. **Face bboxes for video:** The current caching extracts face bboxes from single images. For video datasets, should we extract per-frame bboxes or use the first frame's bbox for all frames? Faces move — per-frame is more accurate but requires video-aware caching.

2. **Reference embedding for video:** In average mode, the reference is the dataset average. For video, should each frame use the same average, or should we compute per-frame references (e.g., if the video shows different angles)?

3. **I2V conditioning frame:** For Wan 2.2 I2V, the first frame is conditioned. Should identity loss weight the first frame differently (it should match the conditioning image exactly)?

4. **Temporal loss weight schedule:** Should temporal consistency loss increase over training? Early on, the model is learning basic structure; late in training, temporal coherence matters more.

5. **Frame selection strategy:** Uniform spacing vs keyframe-based (frames with most motion)? Uniform is simpler and probably sufficient.

6. **Wan 2.2 5B (TI2V-5B):** This model has a different VAE (16x16 spatial compression, 48 latent channels). The TAESD frame-wise decode trick may not work — may need the full 3D VAE. Need to verify.
