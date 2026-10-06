# Upstream sync and perceptual-loss validation

## Scope

Synced through upstream `ecee894ed2b1f3716d9d7326693061ec1a3105bb` (2026-09-27).
The fork's original source snapshot is recorded as upstream `06ef3d34`; the
history bridge preserves the fork tree and gives subsequent merges a common
ancestor. Existing fork-specific losses, metrics, caches, UI controls and base
LoRA merging were reconciled with the upstream v2 loaders and new model families.

Availability of an upstream architecture does not validate every custom loss on
that architecture. These are short integration tests, not image-quality or
convergence benchmarks. Krea Turbo/edit and H3 Ref2VA/Fast variants were not run.

## Real training, 2026-10-05

RTX 4090 (24 GB), Python 3.12, torch 2.7.0+cu126, transformers 5.5.3,
huggingface_hub 1.23.0, peft 0.18.1, and diffusers commit
`c943837899b16cbae2f619b8dd4f7bb6f07dd81a`. Runs used seed 42, rank-4 LoRA,
batch size 1, checkpointing, cached text/latents and disabled sampling.

| Model | Standard data / resolution | Steps | Applied depth loss | Applied identity loss | Depth gradient norm |
|---|---|---:|---:|---:|---:|
| Krea2 Raw | 42 scarlett_full images / 384 | 8 | 0.00717–0.04542 | 0.01906–0.04819 | 0.000865–0.004971 |
| MiniMax H3 FL2VA | 42 scarlett_full images / 256 | 8 | 0.00348–0.00910 | 0.01070–0.03663 | 0.000537–0.010081 |
| MiniMax H3 FL2VA video | standard dance clip / 384, 5 frames | 3 | 0.01546–0.01864 | 0.02666–0.03194 | 0.001327–0.001622 |

All three jobs exited successfully and saved final LoRA and optimizer checkpoints.
All logged losses and gradient norms were finite. Numeric ranges above come
from the SQLite records for steps 1–7; step 0 is also present in the console log.
The video row uses SQLite steps 1–2 (step 0: depth 0.01809, identity 0.03217,
depth gradient norm 0.001692).
Krea used the fork's automatic diffusion/depth alternation; H3 explicitly summed
the losses (`train.loss_split: null`). Face gating can intentionally skip samples.

Reproduce from the repository root (standard test fixtures must be present):

```bash
SEED=42 python run.py config/examples/train_krea2_perceptor_smoke.yaml
SEED=42 python run.py config/examples/train_minimax_h3_perceptor_smoke.yaml
mkdir -p test_data/minimax_video_smoke
cp test_data/dance_clips/man_dancing_square_16fps_4s.mp4 test_data/minimax_video_smoke/
SEED=42 python run.py config/examples/train_minimax_h3_video_perceptor_smoke.yaml
```

The H3 example contains local ComfyUI weight paths; adjust them to your storage.
Keep `quantize: true, qtype: convrot8` and `quantize_te: true, qtype_te: nvfp4`
to preserve the shipped checkpoint formats. Omitting these dequantizes them.
The examples use block offloading to leave room for the perceptors/native VAE.

Artifacts reside under `output/<config.name>/`: `loss_log.db`, checkpoints,
optimizer state, saved config, and depth previews. Krea also writes face
previews. Full logs for this validation are `output/krea2_perceptor_smoke.log`
and `output/minimax_h3_perceptor_smoke_kernel_fix.log` in the integration worktree.

## Codec and regression checks

- Krea2 uses normalized Qwen/Wan latents with TAEW2.1, including full-bucket
  depth targets. Its tiny decoder remains differentiable.
- H3 uses the native 24-channel video VAE for both targets and predictions.
  Ground truth prefers the exact cached training latent; uncached inputs use
  the dataset transform and model encoder. Corrupt cached latents are errors,
  not silently replaced targets. Image inputs use the single-frame 5D path.
- Native target caches include codec version, source metadata, bucket/crop,
  flips, clip-selection settings and perceptor settings. Existing other-model
  targets are neither reused nor overwritten.
- A real H3 forward exposed an upstream Triton free-variable lookup failure.
  Publishing `libdevice` alongside `tl`/`triton` fixes older Triton runtimes;
  a real CUDA test matches reference int8 quantization, including zero rows
  and padding.
- Combined selected regression suite (including CUDA and clean-target checks):
  **127 passed, 2 skipped**, rerun in the primary checkout after integration.
  Trainer plus all 38 registered model classes import.
- Fixed upstream Next.js dynamic-route/page parameter types and removed the
  build-time type-error bypass. Full generated-route typechecking is required,
  not just a fresh-checkout check without `.next/types`.
- Production UI build passes with typechecking enabled. The optional macOS
  temperature-sensor module still emits an upstream warning on Linux.

H3 body-proportion loss is explicitly rejected by the native decoder path.
These tests do not establish full-finetuning, distributed training, long-video
memory limits, or generation quality after extended training.
