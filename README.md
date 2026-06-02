# Perceptual LoRA Toolkit

An extension of [AI Toolkit by Ostris](https://github.com/ostris/ai-toolkit) that adds two layers of regularization to LoRA training:

1. **Perceptual anchoring**: train against frozen vision models (depth, identity, body proportions) instead of only per-pixel loss, so the LoRA picks up shape and identity without baking in source artifacts. The depth anchor is the most useful one in practice; it lets the LoRA pick up the shapes in your dataset without locking in the colors, textures, or lighting.
2. **Weight noising**: inject small Gaussian noise into LoRA parameter values at each optimizer step. Biases training toward flat loss minima, spreads learning across more singular directions of the LoRA factorization (measured **+20% stable rank** on Flux 2 Klein 9B at matched training settings), and reliably reduces memorization on small / single-image datasets where standard training overcooks or diverges.

These can be used independently or together. Weight noising is the bigger practical win for subject-likeness LoRAs; perceptual anchoring is the bigger win when you need geometric/structural control.

## Contents

- [Supported and Experimental Models](#supported-and-experimental-models): which models are validated, which need more testing
- [Perceptual Anchoring](#perceptual-anchoring): depth, identity, body, face suppression
- [Weight Noising](#weight-noising): per-step Gaussian perturbation of LoRA weights
- [Auto-Masking](#auto-masking): body / clothing / subject masks for region-weighted loss
- [Reg Dataset Semantics](#reg-dataset-semantics): how reg samples are treated in this extension
- [Training Metrics](#training-metrics): what gets logged each step
- [Training Previews](#training-previews): what each anchor saves to disk
- [Dataset-Tools UI](#dataset-tools-ui): preflight passes for masks, depth, faces
- [Quickstart Templates](#quickstart-templates): UI presets for validated configs
- [Z-Image Turbo](#z-image-turbo): model-specific notes, current best-guess settings, ComfyUI inference tip
- [LTX-2.3 Video (experimental)](#ltx-23-video-experimental): early video training support, example configs
- [Tips and Tricks](#tips-and-tricks): empirical patterns from training runs
- [Examples](#examples)
  - [Yoshitaka Amano Style (small-dataset style LoRA)](#example-yoshitaka-amano-style-small-dataset-style-lora)
- [Configuration Reference](#configuration-reference): every extension-specific config option
- [Upstream: AI Toolkit by Ostris](#upstream-ai-toolkit-by-ostris)
- [Installation](#installation): local setup, or one-click RunPod template
- [Running the Web UI](#running-the-web-ui): start the job builder, dataset tools, and training monitors

## Supported and Experimental Models

The model dropdown in the web UI is split into two groups, and the same split applies when you train from a config file.

**Supported:** SDXL, FLUX.2 Klein 9B, and Z-Image Turbo. These are known to work well with both weight noising and perceptual anchors, so they're the safest place to start. FLUX.2 Klein 9B and Z-Image Turbo each have a ready-made [Quickstart Template](#quickstart-templates) you can apply in one click.

**Experimental:** everything else in the selector, including Chroma 1 Base, Chroma 1 HD, Chroma Radiance, Z-Image (base), and LTX-2.3, plus the other architectures you can load from a config. These may work, but they haven't had enough testing to call them validated. When you train one:

- Start from values close to the [Quickstart Templates](#quickstart-templates) and tweak from there.
- Turn on weight noising first, before you add a perceptual anchor. Weight noising is generally safe and improves results in most cases. Perceptual anchors are stronger but can destabilize training if the strength is wrong, so add one only after the plain weight-noise run looks healthy.

If you get an experimental model working well, please [open an issue](https://github.com/BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual/issues) with your config and samples so we can move it into the supported set.

## Perceptual Anchoring

The standard LoRA training loss is per-pixel MSE in latent space. It tells the model "match this exact image." On small datasets that turns into a strong instruction to memorize, which is why you often see washed-out colors, baked-in lighting, and "burn-in" (stippling, JPEG ghosts) showing up in every generation.

Perceptual anchors give the LoRA more targeted guidance. Each one is a frozen vision model that scores a single property of the generated image, like its depth or its facial identity, and the LoRA gets rewarded for matching the training images on that property alone. You pick which properties matter for what you're training.

```mermaid
flowchart TD
    LegendNote["∇ = gradients flow back<br/>along this edge during backprop"]
    GT([Training image])
    LegendNote ~~~ GT
    GT --> Encode[VAE encode]
    Encode --> Z0[Clean latent z₀]
    Z0 --> Noise[Add noise at step t]
    Noise --> Zt[Noisy latent z_t]
    Zt --> Model[/LoRA model/]
    Model <-->|∇| Zhat[Predicted z₀']

    Z0 -.-> Diff["Diffusion loss<br/>(MSE in latent space)"]
    Zhat <-.->|∇| Diff

    subgraph Perceptual["Perceptual anchor path (this extension)"]
        Decode[VAE decode]
        RGBp[Predicted RGB]
        Pp["Frozen perceptor<br/>(DA2 / ArcFace / ViTPose)"]
        Pg[Same frozen perceptor]
        Anchor["Perceptual anchor loss<br/>(compares predicted vs. clean ground truth perceptor outputs,<br/>not pixels)"]
    end

    Zhat <-->|∇| Decode
    Decode <-->|∇| RGBp
    RGBp <-->|∇| Pp
    GT --> Pg
    Pp <-.->|∇| Anchor
    Pg -.-> Anchor

    Diff --> Total((Total loss))
    Anchor --> Total

    classDef frozen fill:#e8eaf6,stroke:#3949ab,color:#1a237e
    classDef trainable fill:#fff8e1,stroke:#f57c00,color:#e65100
    classDef loss fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    classDef anchor fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
    classDef legendNode fill:#fafafa,stroke:#bbb,color:#555,stroke-dasharray:3 3

    class Encode frozen
    class Model trainable
    class Diff,Total loss
    class Decode,RGBp,Pp,Pg,Anchor anchor
    class LegendNote legendNode
    style Perceptual fill:#faf5fc,stroke:#6a1b9a,stroke-dasharray:5 4,color:#4a148c
```

The anchor path (purple) is what this extension adds. Both the GT image and the LoRA's prediction go through the **same frozen perceptor**, and the loss is computed on its outputs (a depth map for DA2, a face embedding for ArcFace, a keypoint heatmap for ViTPose). Gradients flow back through the perceptor and VAE decoder, translating the perceptual loss into a **latent-space update** for the LoRA. The weights most strongly nudged are the ones whose latents most affected the property the perceptor measures (depth, identity, pose); others barely move. Loss splitting (described below) takes this further by running the diffusion-loss step and the anchor-loss step alternately rather than summing them every step.

### Depth-Consistency Anchor

Tells the LoRA to keep the geometric structure of the training images while ignoring everything else. It separates "what's in the scene" (which is the LoRA's job) from "how it looks in this particular photo" (which can be left to the model's prior). Useful for:

- **Subject LoRAs that generalize.** The model learns the subject's shape and pose without baking in the outfit, lighting, or backdrop of each training photo.
- **Style transfer** that keeps scene composition but changes appearance.
- **Reducing texture burn-in and stippling** on small datasets. Depth doesn't reinforce per-pixel artifacts, so fine-detail memorization slows down a lot.

Powered by Depth-Anything-V2 (Small by default; Base or Large can be selected for stronger geometry).

**Quick start:**

```yaml
depth_consistency:
  loss_weight: 0.1                       # default; 0 disables
  model_id: depth-anything/Depth-Anything-V2-Small-hf
  mask_source: subject                   # 'none' | 'subject' | 'body'
  loss_min_t: 0.0
  loss_max_t: 1.0
  preview_every: 100
```

The default of `0.1` is calibrated for **DA2-Small** (the default perceptor). If you switch to **DA2-Large**, drop the weight to around `0.001`, since the larger model produces much higher-magnitude gradients and `0.1` will overpower the diffusion loss. **DA2-Base** sits between the two; start at `0.01` and tune from there. If outputs look washed-out, over-smoothed, or the LoRA seems to be ignoring color and texture, the depth weight is too high. Halve it and retry.

**Per-dataset overrides** (handy when different folders need different strengths):

```yaml
datasets:
  - folder_path: /path/to/portraits
    depth_loss_weight: 0.2               # stronger structure on portraits
    depth_loss_min_t: 0.5                # only fire on noisy timesteps for this set
```

Ground-truth depth maps are cached automatically at job start, so the anchor adds no per-step preprocessing cost once training begins.

**Loss splitting (on by default whenever depth anchoring is active).** When the diffusion loss and depth anchor pull in different directions, having them fire on alternating optimizer steps instead of competing every step turns out to work better than running them together for almost every workflow we've tested. As of this version, the trainer turns this on automatically for every dataset whose effective depth-consistency weight is > 0, so you usually don't need to set anything. If you want to be explicit, you can flip it on or off globally:

```yaml
train:
  loss_split: diffusion_depth   # force on for all datasets
  # loss_split: null            # force off everywhere
  # (omit the key entirely for autodetect, which is the default)
```

Or override per dataset, which always wins over the global setting:

```yaml
datasets:
  - folder_path: /path/to/data
    loss_split: diffusion_depth
```

This separates structure-learning (depth) from appearance-learning (diffusion) into distinct optimizer steps. In practice it acts as a strong implicit regularizer against burn-in: fine-texture parameters update much more slowly than coarse-structure parameters, since the two losses only really agree on the latter. The autodetect default means turning on the depth anchor is enough to get the splitting behavior; you only need to touch this if you want the old summed-every-step behavior back.

### Identity Anchor (ArcFace)
Keeps the trained subject's face recognizable across poses, expressions, and lighting. Useful when you're training on diverse appearances of the same person and the diffusion loss alone isn't enough to lock in identity. Recommended weight: **0.01 to 0.1**.

### Body Proportion Anchor (ViTPose)
Keeps body proportions (limb lengths, torso ratio) consistent with the training images. Useful for full-body subject LoRAs where the body shape should stay recognizable across generated poses. Recommended weight: **0.1 to 0.2**.

### Face Suppression
The inverse of the identity anchor: it tells the LoRA to *ignore* faces. The diffusion loss is downweighted (or zeroed) inside detected face regions, so the model doesn't learn to reproduce the faces in your dataset. Use this when training a style or clothing LoRA on a dataset that happens to contain people, and you want the style or outfit but not the faces.

Set `face_id.face_suppression_weight` between `0` (off) and `1` (full suppression). Per-dataset overrides are supported.

### Quick-start config

```yaml
depth_consistency:
  loss_weight: 0.1                       # primary anchor (DA2-Small default; use 0.001 for DA2-Large)
face_id:
  identity_loss_weight: 0.1              # secondary
  body_proportion_loss_weight: 0.1       # secondary
  face_suppression_weight: 0.5           # optional
  identity_metrics: true                 # log id_sim without applying loss
```

Per-dataset overrides let you tune each anchor for the dataset's content:

```yaml
datasets:
  - folder_path: /path/to/portraits
    depth_loss_weight: 0.2               # stronger structure on portraits
    identity_loss_weight: 0.1            # stronger face on close-ups
  - folder_path: /path/to/fullbody
    body_proportion_loss_weight: 0.15    # preserve body shape
    face_suppression_weight: 1.0         # full suppression, don't learn faces
```

See `config/examples/train_lora_flux_identity_24gb.yaml` for a complete example.

## Weight Noising

A small Gaussian perturbation is added to LoRA parameter **values** after each optimizer step (`p.data += σ · randn`, filtered to LoRA-tagged params only). This is **not** gradient noise; the noise hits the weights themselves, not the gradient.

The technique sits between classical weight noise (Graves 2011) and the SAM/SGLD family, but is gradient-free and runs as a few lines in the optimizer loop. It pairs particularly well with the depth anchor on small datasets, but works on its own too. Even with diffusion loss alone, weight noise reliably produces better subject likeness and resists the overcooking failure mode that standard LoRA training falls into on tiny datasets.

### What it does

- **Pushes training toward flat loss minima.** Standard expected-loss expansion adds a `½ σ² · Tr(H)` flat-minima penalty (Camuto et al. NeurIPS 2020). Verified empirically via the `grad/fisher` metric (diagonal-Fisher trace, which proxies `Tr(H)`); the curve trends down vs the unperturbed baseline.
- **Spreads learning across the LoRA's rank budget.** Measured **+20% stable rank**, **+12% participation ratio** on a 112-module LoKr trained on Flux 2 Klein 9B at matched step count. Frobenius norm barely moves (+2%), ruling out weight inflation; the same energy is just distributed across more singular directions. This is the LoRA-specific manifestation of the flat-minima bias and is probably the biggest reason the technique works.
- **Resists memorization on small datasets.** Single-image runs converge reliably where the same config without noise overcooks or diverges. The model exhibits a "self-healing" property in this regime: weights can briefly diverge into a worse basin and recover, because no single singular direction is load-bearing.

### Recommended starting config

```yaml
train:
  weight_noise:
    enabled: true
    mode: relative
    sigma: 0.0125      # 0.01 – 0.017 is the typical useful range
    log_every: 50
```

### Modes

- **`relative`** (default): `σ_per_param = sigma × ‖w‖_RMS`. Adapts automatically to per-layer scale, which matters because LoRA layers can have very different gradient/weight magnitudes (LoRA-up vs LoRA-down, attention vs MLP). LoRA-up params (init=0) get no noise until they learn something, so early training is safe by construction.
- **`absolute`**: fixed σ everywhere. Use when you've calibrated a specific magnitude target across all layers.

### Metrics

- **`weight_noise_norm`**: Frobenius norm of the injected noise per logged step. In relative mode this grows with the LoRA's weight magnitude during training; that's normal and expected.
- **`grad/fisher`**: diagonal Fisher trace (sum of Adam's `exp_avg_sq` over LoRA params). Should trend downward over training in a noised run vs flat-or-rising in a baseline.

### Notes

- LoRA / LoKr params only. The implementation filters to network-tagged adapter params, not the base model.
- Distributed training (DDP/FSDP) not yet tested.
- The current implementation does noise + Adam; a custom SGD+OU optimizer would save ~25% LoRA-state VRAM but isn't shipped.
- There's also a parallel `train.gradient_noise.*` block (noise on gradients before optimizer.step rather than on weights after). Same Neelakantan-style modes; weight noise is the empirically stronger of the two.

## Auto-Masking

Splits each training image into regions (body, clothing, and subject = body ∪ clothing) so different parts of the image can be weighted differently in the loss. Useful for:

- **Subject LoRAs** that should focus on the person, not the background. Set the background weight low and the body weight high.
- **Clothing LoRAs** that should learn outfit details while ignoring face and body.
- **Letting the depth anchor focus on the subject** instead of computing depth-consistency over the whole image, where most of the frame is usually background you don't care about.

Masks are generated per image at job start and cached.

```yaml
subject_mask:
  enabled: true
  body_close_radius: 5
```

```yaml
datasets:
  - folder_path: /path/to/data
    background_loss_weight: 0.3
    clothing_loss_weight: 0.7
    body_loss_weight: 1.0
    perceptual_restrict_to_body: true    # restrict perceptual anchors to body region
```

The depth anchor picks which mask it uses via `depth_consistency.mask_source` (`subject`, `body`, or `none`).

QC tiles for visual inspection are saved at job start and can be regenerated from the dataset-tools UI.

## Reg Dataset Semantics

Reg datasets (`is_reg: true`) work the classic Dreambooth way: they're prior-preservation samples that train the model on generic non-subject images alongside your subject samples, so it doesn't forget how to make non-subject content while it's learning the subject. In this extension, reg semantics are tightened up:

- **All perceptual anchors are turned off on reg samples.** Only the diffusion loss fires, scaled by `train.reg_weight`.
- **Subject conditioning is stripped.** No clip-image or trigger-word injection.

The effect is that reg samples teach the model "produce sharp prior-distribution images" without contaminating any of the subject-specific anchors. The 50/50 reg/train alternation runs at the optimizer-step level, so the gradient stays clean under any accumulation setting. `train.reg_weight` (default `1.0`) controls how strongly reg pulls vs. train.

## Training Metrics

Every active loss is logged so you can see during a run whether each anchor is doing its job. The training UI shows live charts plus per-sample tooltips on each point (which images drove the loss this step).

| Metric | What it tells you |
|---|---|
| `diffusion_loss` | How well the model is matching training images per-pixel. Watch for it bottoming out, which usually means memorization. |
| `diffusion_loss_tNN` | Diffusion loss broken down by timestep band (`t00` through `t90`). Useful for spotting whether low-noise or high-noise timesteps are dominating. |
| `depth_consistency_loss` | How well the predicted geometry matches the training images. Should fall steadily; if it goes flat, the depth anchor isn't converging. |
| `depth_loss_tNN` | Depth loss per timestep band. |
| `id_sim` | Face cosine similarity (higher is better). Set `face_id.identity_metrics: true` to log this without applying the loss. |
| `id_sim_tNN` | Per-timestep face similarity. |
| `body_proportion_loss` | Pose-proportion error. |
| `grad_norm` | Total gradient magnitude post-clip. Spikes usually mean a loss explosion. |
| `grad_norm_diffusion`, `grad_norm_depth`, `grad_cos_diff_depth` | Optional gradient-cosine diagnostic. See below. |
| `weight_noise_norm` | Frobenius norm of injected weight noise. Only logged when `train.weight_noise.enabled`; cadence set by `log_every`. |
| `grad/fisher` | Sum of Adam's `exp_avg_sq` across LoRA params; diagonal-Fisher proxy for `Tr(Hessian)`. Drops over training when weight noise biases toward flat minima. Free to compute (just sums optimizer state); always on. |

**Gradient-cosine diagnostic.** When you suspect two anchors are pulling in opposite directions, this measures how aligned their gradients are. Cosine near +1 means they reinforce each other, near 0 means they're independent, negative means they're fighting. Off by default; enable with `train.gradient_cosine_log_every: 50`.

## Training Previews

Visual previews are saved during training so you can see at a glance what each anchor is responding to.

| Directory | What you see |
|---|---|
| `depth_previews/` | Side-by-side comparison of GT image, GT depth, predicted image, and predicted depth. Annotated with timestep and depth-loss value so you can scroll through training and watch the geometry converge. |
| `id_previews/` | What the identity anchor is seeing: the face crop being scored, alongside the noisy input and the model's x0 prediction, with the cosine similarity overlaid. |
| `body_previews/` | Skeleton overlays for reference vs. predicted poses. |
| `subject_mask_previews/` | Mask QC: each image with its body, clothing, and subject masks overlaid, generated once at job start. |

## Dataset-Tools UI

Before training, the web UI provides preflight passes that prepare the cached data each anchor needs:

- **Depth preflight.** Runs depth estimation across the dataset and shows visual QC tiles so you can spot bad masks or odd crops before they cost you a training run.
- **Subject-mask preflight.** Generates and caches the body, clothing, and subject masks with overlays for review.
- **Face-detection preflight.** Caches face bounding boxes and identity embeddings.

All three run as non-blocking background jobs. Start them and come back when they're done.

The `scripts/sample_dataset.py` utility builds a smaller dataset directory by sampling N random images (with their captions) from a larger source. Useful for building reg sets, running ablations, or making smoke-test datasets without copying everything.

## Quickstart Templates

The new-job form in the web UI has a **Quickstart Template** selector at the top of the Job card. Picking a template overwrites the current form with a validated config, preserving your training name and dataset folder path so you can apply mid-flow without losing what you've already filled in. Each template also has a matching YAML file under `config/examples/` for CLI use (`python run.py <yaml>`).

Current templates:

- **Subject Likeness (Flux 2 Klein 9B + Weight Noise)**: the full empirically-validated recipe. LoKr (linear/alpha 32, conv/alpha 16, full-rank, factor 8) + weight noise (relative, σ=0.0125) + full-image depth-consistency + multi-bucket (`resolution: [512, 768, 1024]`, `num_repeats: [16, 4, 1]`). AdamW8bit @ lr=5e-5, batch=4, 1200 steps. Defaults `model.name_or_path` to the HuggingFace release (`black-forest-labs/FLUX.2-klein-base-9B`) so the template runs without any local checkpoint. Use this when captions describe the full image.
  - YAML: [`config/examples/subject_likeness_flux2_klein9b.yaml`](config/examples/subject_likeness_flux2_klein9b.yaml)

- **Subject Likeness, Masked (Flux 2 Klein 9B + Weight Noise)**: same recipe plus subject masking with per-region weights (`background:0`, `clothing:1`, `body:1`) and depth-consistency restricted to the subject mask. Use this when you can be disciplined about captioning only the changeable parts of the character and skipping the background/setting. See the [Tips and Tricks](#tips-and-tricks) section for the rationale.
  - YAML: [`config/examples/subject_likeness_masked_flux2_klein9b.yaml`](config/examples/subject_likeness_masked_flux2_klein9b.yaml)

- **Style LoRA (Flux 2 Klein 9B + Depth Anchor)**: small-dataset style recipe, from the validated [Yoshitaka Amano](#example-yoshitaka-amano-style-small-dataset-style-lora) run. LoKr (linear/alpha 32, conv/alpha 16, full-rank, factor 8) + full-image depth-consistency (`0.005`, DA2-Large at `input_size: 1400`) with diffusion/depth loss splitting (`loss_split: diffusion_depth`). Single 768 bucket, AdamW8bit @ lr=5e-5, batch=2, grad-accum=1, 4000 steps. No weight noise (the block is there, disabled) and no subject mask. On a tiny style set the depth anchor does the work: it pushes the LoRA toward what's invariant across the artist's images (linework, color, paper) and away from memorizing specific compositions.
  - YAML: [`config/examples/style_lora_flux2_klein9b.yaml`](config/examples/style_lora_flux2_klein9b.yaml)

- **Subject Likeness (Z-Image Turbo + Weight Noise)**: see [Z-Image Turbo](#z-image-turbo) below. LoKr + weight noise (relative, σ=0.0125) via the de-distill training adapter. Single 1024 bucket, subject-masked depth-consistency (`background:0`, `clothing:1`), AdamW8bit @ lr=2.5e-4, batch=4, 3000 steps. Custom timestep distribution and curve to front-load high-t and low-mid-t training. Transformer in bf16 (Tongyi-MAI warns against FP8 on Turbo), text encoder quantized to qfloat8.
  - YAML: [`config/examples/subject_likeness_zimage_turbo.yaml`](config/examples/subject_likeness_zimage_turbo.yaml)

Templates live in `ui/src/app/jobs/new/quickstarts.ts`; the YAML files under `config/examples/` mirror them and stay in sync. Adding a new template is a one-export change on the TS side plus a YAML mirror. The chosen template name shows in the dropdown label and stays there until you pick another. It's not saved to the config; the form *is* the template after apply.

## Z-Image Turbo

Z-Image Turbo works well with weight noising and the depth anchor, so it's in the supported set, but the quickstart is still a best guess from a few production runs rather than a fully tuned recipe. If you find settings that work better, please [open an issue](https://github.com/BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual/issues) with your config and samples. Step/LR/sigma sweeps, multi-bucket evidence, and reports on Z-Image base are all useful.

A few notes:

- The de-distill adapter (`ostris/zimage_turbo_training_adapter_v2`) is merged at load with `+1.0` and runtime-inverted to `-1.0` at sample time. The optimizer sees a base-like model; the LoRA inverts back to 8-step turbo at inference.
- Don't quantize the transformer. Tongyi-MAI warns FP8 on Turbo causes noticeable quality degradation, so the quickstart keeps `model.quantize: false`. Text encoder quantization is fine.
- Z-Image reuses the Flux VAE byte-for-byte, so Flux caption and cropping habits transfer.
- ZiT can take up to **200 steps per training image** to converge, noticeably more than Flux 2 Klein typically needs. The quickstart defaults to 3000 steps so a 10 to 15 image dataset has room to settle. Optimizations to bring this down are in progress.

For inference, [RES_2S](https://github.com/ClownsharkBatwing/RES4LYF) at 8 steps tends to look noticeably cleaner than the default Euler/DPM samplers most ComfyUI workflows use. Fine eye and iris detail holds up better.

The trainer uses a fixed `shift=3.0` rather than the dynamic shifting `diffusers.ZImagePipeline` defaults to. That matches the ComfyUI workflow, so trained LoRAs look right there; the mismatch is preview-only.

## LTX-2.3 Video (experimental)

Early support for training LoRAs on LTX-2.3 (22B), the video model, including the depth-consistency anchor applied across frames. It's fresh and only lightly tested, so treat the example configs as starting points rather than tuned recipes. Grab one from `config/examples/` to start: `train_lora_ltx23_24gb_smoke.yaml` for a quick smoke test, `train_lora_ltx23_80gb.yaml` for a real run, or `train_ltx23_cartwheel_lokr.yaml` for a worked example. If you find settings that hold up, [open an issue](https://github.com/BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual/issues) with your config and samples.

## Tips and Tricks

A few empirically-useful patterns picked up across training runs.

### Subject masking + targeted captions

If you can be disciplined about captioning, combining subject masking with captions that describe **only the changeable parts of the character** (clothing, expression, pose) and skip the background/setting entirely will give noticeably better results. The combination tells the LoRA two things at once:

- *Spatial*: only the subject region carries diffusion gradient (via the mask).
- *Semantic*: only the captioned attributes are promptable; everything else becomes part of the subject's identity.

Suggested per-region weights when subject masking is on:

```yaml
datasets:
  - folder_path: /path/to/subject
    background_loss_weight: 0     # don't learn the background at all
    clothing_loss_weight: 1       # full diffusion loss on clothing
    body_loss_weight: 1           # full diffusion loss on body
```

The Subject Likeness quickstart template ships with subject masking **off** by default since the "caption everything" workflow is more common. Flip it on in the UI and use these weights when you have the caption discipline to make it count.

### Bucket repeat ratios at scales of 4

When training across multiple resolution buckets, biasing toward lower-res buckets with descending num_repeats in **scales of 4** (e.g. `16:4:1` for 512:768:1024) trains the structural features faster while still anchoring fine detail at the higher-res buckets. The lower-res buckets:

- See each image more often per epoch, pushing coarse structure into the weights early.
- Are cheaper per step, so the extra repeats are inexpensive.

The higher-res buckets train less frequently but their presence prevents the LoRA from collapsing into "low-res only" generations.

Set the per-resolution `num_repeats` as a list aligned 1:1 with the resolution list:

```yaml
datasets:
  - folder_path: /path/to/data
    resolution: [512, 768, 1024]
    num_repeats: [16, 4, 1]
```

The Subject Likeness quickstart uses exactly this ratio.

### Start with a small dataset

With these methods **you need far fewer images** than you'd think to get good, generalizable results. Start with **10 to 15 images** picked for quality and diversity. If that converges well, try adding more images to the same run or restart with the larger dataset. Going big from the start tends to be wasted effort.

### Body horror mid-training is usually fine

Anecdotally, you may see more body horror and extra limbs partway through training when weight noising is on. This is normal. The noise pushes the weights around more between optimizer steps, so some checkpoints may diverge pretty badly before the run converges.

Rough heuristic: budget around **80 to 100 steps per training image**. If you're sampling every 25 steps and see continuous body horror for more than 20% of the run, the noise sigma is probably too high. Lower it in increments of 0.0025 until it resolves. We're still figuring out the training dynamics across different datasets, so reports of what worked (or didn't) are welcome.

I recommend saving checkpoints every 25 steps. The optimal checkpoint is often in a narrow window.

## Examples

> **Note on inference target.** All example configs in this README and under `config/examples/` are tuned for inference against the **distilled** model (Flux 2 Klein). If you plan to apply the trained LoRAs against the **base (non-distilled)** model instead, checkpoints in the **500–800 step range** are usually closer to optimal than the **1000–1200 step range** the configs save out.

### Example: Yoshitaka Amano Style (small-dataset style LoRA)

Training a style LoRA from a small dataset of 14 illustrations. With depth anchoring the LoRA learns enough of the artist's visual language to carry it onto subjects nowhere in the dataset.

Yoshitaka Amano is the illustrator behind the original Final Fantasy character art and a long-running body of solo watercolor portrait work. Flux 2 Klein 9B doesn't reproduce his look from a prompt alone; it defaults to generic anime or oil-paint stylings.

One illustration from the dataset is shown below to give a feel for what the LoRA is asked to learn: loose ink linework, watercolor washes, ornate costuming, hair drawn as long flowing tendrils.

| ![Reference](https://github.com/BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual/releases/download/examples-amano-v1/dataset_reference.jpeg) |
|:---:|
| One of the 14 training illustrations. |

Key bits (full config at `output/amano/config.yaml` after a run):

- LoKr, linear/alpha 32, conv/alpha 16, full-rank, factor 8.
- 4000 steps, batch size 1, gradient accumulation 2.
- Resolution 768.
- Depth anchor: weight `0.005`, DA2-Large at `input_size: 1400`, `mask_source: none`.
- Loss splitting on the dataset (`loss_split: diffusion_depth`).

You can watch the depth anchor converge across training. Ground-truth pair (RGB | depth) first, then predicted pairs from an early step and a late step at a comparable noise level. Early on the predicted depth has heavy halo artifacts and doesn't track the figure cleanly; by the end it's a much closer match.

Ground truth (RGB | depth):

![Ground truth](https://github.com/BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual/releases/download/examples-amano-v1/preview_gt.jpg)

Early prediction (step 383, t=0.82), `depth_consistency_loss: 17.17`:

![Early prediction](https://github.com/BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual/releases/download/examples-amano-v1/preview_pred_early.jpg)

Late prediction (step 3941, t=0.81), `depth_consistency_loss: 6.67`:

![Late prediction](https://github.com/BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual/releases/download/examples-amano-v1/preview_pred_late.jpg)

None of these subjects appear in the training set. The LoRA carries Amano's linework, color treatment, and composition language onto subjects from very different IPs:

| Cloud (FF7) | Snow White (Disney) | Ziggy Stardust (Bowie) |
|:---:|:---:|:---:|
| ![Cloud](https://github.com/BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual/releases/download/examples-amano-v1/output_cloud.png) | ![Snow White](https://github.com/BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual/releases/download/examples-amano-v1/output_snow.png) | ![Ziggy](https://github.com/BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual/releases/download/examples-amano-v1/output_ziggy.png) |

With a style dataset this small, the diffusion loss alone tends to overfit on the specific compositions of the training images; every output starts looking like a slight variation on the same handful of poses and figures. The depth anchor pushes the LoRA toward what's invariant across the artist's work (linework, paper texture, color treatment) and away from what's incidental (this exact figure, in this exact pose, against this exact background). Loss splitting reinforces the separation: the diffusion-step focuses on appearance, the depth-step on structure, and they only really agree on the high-level "this looks like Amano" signal.

## Configuration Reference

Every extension-specific config option, grouped by the YAML block it lives in. Defaults shown match what you get if you omit the option entirely.

### `depth_consistency.*`

The depth anchor.

| Option | Default | What it does, when to use it |
|---|---|---|
| `loss_weight` | `0.1` | Master switch. `0` disables. `0.1` is calibrated for DA2-Small (the default perceptor). Drop to ~`0.001` if you switch to DA2-Large, since it produces much higher-magnitude gradients. If outputs look washed-out or over-smoothed, halve the weight and retry. |
| `model_id` | `Depth-Anything-V2-Small-hf` | Which DA2 variant. Small is fast and adequate for most subjects. Base or Large give cleaner depth on cluttered scenes at higher VRAM cost. **Lower the loss weight when using a larger model** (Base ~`0.01`, Large ~`0.001`). |
| `mask_source` | `subject` | Which mask the loss applies through. `subject` is recommended for subject LoRAs (loss restricted to the person). `body` excludes clothing. `none` uses the full image. |
| `loss_min_t` | `0.0` | Lower edge of the timestep window where the depth anchor fires. |
| `loss_max_t` | `1.0` | Upper edge of the timestep window. Narrow to mid-high (e.g. `0.5` to `0.9`) to focus the anchor on the identity-encoding noise band. |
| `ssi_weight` | `1.0` | Scale-and-shift-invariant L1 term weight. Rarely needs tuning. |
| `grad_weight` | `0.5` | Multi-scale gradient term weight. Increase for more sensitivity to fine geometric structure. |
| `grad_scales` | `4` | Number of pyramid scales for the gradient term. Rarely needs tuning. |
| `input_size` | `518` | DA2 input resolution. Must be a multiple of 14. Can go up to `1400` for the clearest depth maps, at proportionally higher VRAM and compute cost. The default `518` is a good balance for most setups; bump to `714` or `980` if you want sharper depth on detailed scenes, or `1400` for the maximum the perceptor will accept. |
| `grad_checkpoint` | `true` | Gradient checkpointing through the perceptor for memory savings. Leave on unless you're on huge VRAM. |
| `preview_every` | `100` | Save a depth preview tile every N steps to `depth_previews/`. Set to 0 to disable. |
| `preview_min_t` | `0.0` | Only save previews at timesteps at or above this. |

### `face_id.*`

The identity-related anchor losses.

| Option | Default | What it does, when to use it |
|---|---|---|
| `face_model` | `buffalo_l` | InsightFace model used for face detection and embeddings. Don't change unless you have a specific reason. |

**Identity anchor** (the loss that keeps face recognizable):

| Option | Default | What it does, when to use it |
|---|---|---|
| `identity_loss_weight` | `0.0` | Master switch. Typical 0.01 to 0.1. Higher locks in face shape harder, but too high constrains expressions. |
| `identity_loss_min_t` | `0.0` | Timestep window lower edge. |
| `identity_loss_max_t` | `1.0` | Timestep window upper edge. |
| `identity_loss_min_cos` | `0.2` | Minimum face similarity for the loss to fire on a sample. Below this, the predicted x0 likely doesn't contain a recognizable face yet, so the loss is skipped to avoid hallucinating one. |
| `identity_metrics` | `false` | Log `id_sim` without applying the loss. Useful for measuring identity drift in vanilla runs as a baseline. |
| `identity_loss_use_average` | `true` | Compare against the dataset's average face embedding instead of per-image. More robust on diverse training sets. |
| `identity_loss_average_blend` | `0.0` | Blend per-image with dataset average. 0 = per-image only, 0.5 = midpoint, 1.0 = pure average. |
| `identity_loss_use_random` | `false` | Compare against a random embedding from the dataset each step. Useful for mixed-identity training. |
| `identity_loss_num_refs` | `0` | If > 0, compare against K random embeddings and use best match. |

**Body proportion anchor** (ViTPose bone-length ratios):

| Option | Default | What it does, when to use it |
|---|---|---|
| `body_proportion_loss_weight` | `0.0` | Master switch. Typical 0.1 to 0.2. Use on full-body subject LoRAs where body shape should stay recognizable across poses. |
| `body_proportion_loss_min_t` | `0.0` | Timestep window lower edge. |
| `body_proportion_loss_max_t` | `1.0` | Timestep window upper edge. |
| `body_proportion_include_head` | `false` | Include head-related ratios. Off by default since identity anchor handles the head better. |

**Face suppression** (the inverse anchor; tells the LoRA to *not* learn faces):

| Option | Default | What it does, when to use it |
|---|---|---|
| `face_suppression_weight` | `null` | Master switch. `null` = no suppression. `0.0` = zero face loss (don't learn faces at all). `0.5` = half. `1.0` = normal. Use `0.0` for style or clothing LoRAs trained on photos with people. |
| `face_suppression_expand` | `2.0` | Multiplier on the face bounding box. `1.0` = tight face box, `1.8-2.0` = full head coverage. |
| `face_suppression_soft` | `false` | Gaussian falloff at the box edges instead of a hard rectangle. Smoother but slightly more invasive. |

### `subject_mask.*`

Auto-masking pipeline.

| Option | Default | What it does, when to use it |
|---|---|---|
| `enabled` | `false` | Master switch. Set true to extract per-image masks at job start. |
| `body_close_radius` | `2` | Morphological closing on the body mask. Higher values fill gaps in limbs and hair (e.g. `5` for blurry photos) at the cost of boundary precision. Changing this invalidates cached masks. |
| `mask_dilate_radius` | `0` | Outer dilation on the subject mask. Useful when you want a padding margin around the subject. |
| `skin_bias` | `0.0` | Bias added to body-class logits where skin tone is detected. Set to 1-3 if your dataset has lots of exposed skin and SegFormer is mislabeling it as clothing. |
| `save_debug_previews` | `false` | Save preview tiles per image. The dataset-tools UI preflight does the same thing on demand. |
| `segformer_res` | `768` | SegFormer input resolution. Don't change unless you know what you're doing. |
| `cache_resolution` | `256` | Cached mask resolution. Higher = sharper at training time, more disk. |
| `yolo_ckpt`, `yolo_conf`, `sam_size`, `dtype`, `primary_only` | (defaults) | Detection / segmentation backend knobs. The defaults work for almost everyone. |

### `train.*` (extension-specific additions)

| Option | Default | What it does, when to use it |
|---|---|---|
| `reg_weight` | `1.0` | Multiplier on diffusion loss for reg samples. `1.0` (equal pull) is the sane default. Increase to 1.5-2.0 if reg isn't preserving the prior strongly enough. |
| `loss_split` | autodetect | Global default for the per-dataset `loss_split` knob. When the key is omitted, the trainer turns on `diffusion_depth` automatically for every dataset whose effective depth-consistency loss weight is > 0. Set explicitly to `diffusion_depth` to force on everywhere, or to `null` to force off (back to summed-every-step). Per-dataset `loss_split` always wins. |
| `gradient_cosine_log_every` | `0` | Diagnostic. Every N optimizer steps, measure `cos(g_diffusion, g_depth)` and log the per-loss gradient norms. `0` disables. Use 50-100 to diagnose anchor conflicts without much overhead. |
| `diffusion_loss_min_t` / `diffusion_loss_max_t` | `0.0` / `1.0` | Global timestep window for the diffusion loss. Samples outside the window are zeroed. Per-dataset overrides supported (see below). |
| `min_denoising_steps` | `0` | Lower bound on the timestep sampler (0-999). The training loop only samples from `[min, max]`. Useful for focused training, e.g. `min=700, max=700` to train at one specific noise level. |
| `max_denoising_steps` | `999` | Upper bound on the timestep sampler. |

### `train.weight_noise.*`

Per-step Gaussian perturbation of LoRA weights (see [Weight Noising](#weight-noising)).

| Option | Default | What it does, when to use it |
|---|---|---|
| `enabled` | `false` | Master switch. When `false` the injector is a no-op regardless of other fields. |
| `mode` | `relative` | `relative` (σ × per-param weight RMS, adapts per-layer) or `absolute` (fixed σ everywhere). |
| `sigma` | `0.0125` | Noise scale. In `relative` mode, a multiplier on each tensor's weight RMS. Typical useful range **0.01 – 0.017**. Lower values barely do anything; higher risk noise overpowering the gradient. |
| `log_every` | `50` | Cadence for emitting `weight_noise_norm`. `0` disables logging (still injects). |

### `train.gradient_noise.*`

Per-step Gaussian noise injected into LoRA **gradients** before `optimizer.step()`. Closely related to weight noise but acts on the gradient side; empirically the weaker of the two for LoRA fine-tuning. Includes the SGLD-style Neelakantan annealed mode.

| Option | Default | What it does, when to use it |
|---|---|---|
| `enabled` | `false` | Master switch. |
| `mode` | `neelakantan` | `absolute` (fixed σ), `relative` (σ × per-param grad RMS), or `neelakantan` (`σ_t = eta / (1 + step)^gamma`, annealed). |
| `sigma` | `1e-3` | Noise scale for `absolute` and `relative` modes. |
| `eta` | `0.01` | Initial noise scale for `neelakantan` (paper default). |
| `gamma` | `0.55` | Anneal exponent for `neelakantan` (paper default). |
| `log_every` | `50` | Cadence for emitting `grad_noise_snr` and `grad_noise_norm`. |

### Per-dataset overrides (`datasets[].*`)

Every entry in `datasets:` accepts these extension-specific overrides. `null` or omitted = inherit the global value.

| Option | What it does, when to use it |
|---|---|
| `is_reg` | Mark this dataset as a regularization set. Strips subject conditioning and turns off all perceptual anchors on its samples. |
| `loss_split` | Per-dataset override of the global `train.loss_split`. Omit (or set `null`) to inherit. Set to `diffusion_depth` to force on for this dataset (alternates diffusion and depth-anchor per optimizer step). Set to `sum` to force off for this dataset (losses sum every step). Per-dataset always wins over the global. |
| `resolution` | Single int (`512`) or a list (`[256, 512, 768, 1024]`). A list expands into one internal dataset per resolution at load time. |
| `num_repeats` | Scalar (broadcast to every resolution) or a list aligned 1:1 with `resolution` for per-bucket repeat counts. E.g. `resolution: [256, 512, 768, 1024]` paired with `num_repeats: [64, 16, 4, 1]` biases sampling toward the lower-res buckets while still anchoring higher-res quality. Mismatched list lengths raise a clear error. |
| `diffusion_loss_weight` | Per-dataset multiplier on the diffusion loss for this dataset. Set to 0 to fully suppress diffusion loss on this set (useful for anchor-only training). |
| `diffusion_loss_min_t` / `diffusion_loss_max_t` | Per-dataset timestep window for the diffusion loss. Inclusive bounds. Inherits global `train.diffusion_loss_min_t/max_t` when omitted. |
| `depth_loss_weight` | Per-dataset override of the depth anchor's `loss_weight`. Set to `0` to fully disable the depth anchor for this dataset (skips perceptor compute on its samples). |
| `depth_loss_min_t` / `depth_loss_max_t` | Per-dataset depth-anchor timestep window. |
| `depth_model_id` | Per-dataset DA2 variant. Useful if one dataset has unusual geometry that benefits from Large while others stay on Small. |
| `identity_loss_weight` / `_min_t` / `_max_t` / `_min_cos` | Per-dataset identity-anchor controls. Stronger weights on portrait crops, weaker on full-body. |
| `body_proportion_loss_weight` / `_min_t` / `_max_t` | Per-dataset body-proportion controls. Useful for full-body shots where pose proportions matter. |
| `face_suppression_weight` | Per-dataset face suppression. Per-dataset takes priority over global. |
| `background_loss_weight` / `clothing_loss_weight` / `body_loss_weight` | Per-region diffusion weight scaling. Used when `subject_mask.enabled` is true. Set background low (e.g. 0.3) and body high (1.0) to tell the LoRA to focus on the subject. |
| `perceptual_restrict_to_body` | Restrict perceptual-anchor losses to the body mask region for this dataset. |

---

## Upstream: AI Toolkit by Ostris

This extension is based on [AI Toolkit](https://github.com/ostris/ai-toolkit), an all-in-one training suite for diffusion models on consumer hardware.

### Support the Original Author

[Sponsor on GitHub](https://github.com/orgs/ostris) | [Support on Patreon](https://www.patreon.com/ostris) | [Donate on PayPal](https://www.paypal.com/donate/?hosted_button_id=9GEFUKC8T9R9W)


---




## Installation

### Run on RunPod

No local GPU? [**Deploy the prebuilt RunPod template →**](https://console.runpod.io/deploy?template=bn89xhug15&ref=hfo6q4j3). It runs the same Docker image this repo builds, with the web UI already up on port 8675.

### Local install

Requirements:
- Python 3.12 (3.10 and 3.11 are fine too). Don't use 3.13 or newer yet, see the note below.
- Nvidia GPU with enough VRAM for what you're training
- Python venv
- git

**Heads up on Python 3.13:** a few of the native dependencies (insightface, onnxruntime, torchcodec) don't ship wheels for 3.13 yet, especially on Windows. If you build your venv with 3.13 the install ends up broken and the trainer dies during startup before it can even write a log file, so from the UI it just looks like the job is stuck on "starting" forever with no error in the log. Use 3.12 and you'll be fine.

If your system Python is already 3.13 or newer, the easy fix is to make a 3.12 environment with conda and run everything inside it:
```bash
conda create -n ai-toolkit python=3.12
conda activate ai-toolkit
# then run the torch + requirements install steps below
```

Linux:
```bash
git clone https://github.com/BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual.git
cd ai-toolkit-perceptual
python3 -m venv venv
source venv/bin/activate
# install torch first
pip3 install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip3 install -r requirements.txt
```

Windows:
```bash
git clone https://github.com/BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual.git
cd ai-toolkit-perceptual
python -m venv venv
.\venv\Scripts\activate
pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

**RTX 50-series (Blackwell) GPUs:** the `cu126` build pinned above has no kernels for your card, so training dies with `CUDA error: no kernel image is available for execution on the device`. Install torch from the `cu128` channel instead (run this in place of the torch line above):
```bash
pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

For devices running **DGX OS** (including DGX Spark), follow [these](dgx_instructions.md) instructions.

## Running the Web UI

Job creation, the dataset-tools preflights, and live training monitors all live in the web UI. Since this repo only documents what it adds on top of upstream AI Toolkit, it's easy to miss how to start the UI itself.

After a local install, launch it from the `ui/` directory:

```bash
cd ui
npm run build_and_start
```

This installs the UI's Node dependencies, initializes its database, builds, and serves the app at **http://localhost:8675** (requires Node.js 18+). The first run does the full build; afterwards `npm run start` restarts it without rebuilding.

> Using the RunPod template? The UI is already running, so just open port **8675** from the pod's **Connect** menu.

