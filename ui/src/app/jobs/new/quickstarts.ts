// Quickstart templates pre-load a known-good job config into the new-job form.
// Each template's `apply()` returns a fresh JobConfig, preserving a few
// already-filled user-side fields (training name, dataset folder_path) so the
// template can be applied mid-flow without nuking work the user has already done.

import { JobConfig } from '@/types';

export interface QuickstartTemplate {
  id: string;
  label: string;
  description: string;
  /** Produce a new JobConfig from the template. Receives the current form
   *  state so it can preserve user-entered fields like name and folder_path. */
  apply: (current: JobConfig) => JobConfig;
}

// ---------------------------------------------------------------------------
// Subject Likeness — Flux 2 Klein 9B + Weight Noise
// ---------------------------------------------------------------------------
// Abstracted from a validated production run that empirically produced strong
// subject likeness on Flux 2 Klein 9B using weight noise + LoKr + multi-bucket
// resolutions + depth-consistency. Two variants:
//   - Default: full-image diffusion loss, "caption everything" workflow.
//   - Masked: subject masking + per-region weights, "caption only changeable
//     parts" workflow. Better results when the user can be disciplined about
//     what they caption.
// Specifics that vary per user (dataset path, training name, sample prompt,
// model checkpoint) are abstracted; everything else encodes the technique.

function buildSubjectLikenessConfig(
  current: JobConfig,
  opts: { subjectMask: boolean },
): JobConfig {
  // Preserve user-entered fields if they exist; otherwise fall back to
  // sensible blanks so the form clearly shows what still needs filling in.
  const existingName = current.config.name || '';
  const existingDatasetPath =
    current.config.process[0]?.datasets?.[0]?.folder_path || '';
  const existingGpuIds = (current.config.process[0] as any)?.gpu_ids;

  // Single dataset entry covering all three buckets via the per-resolution
  // num_repeats list form (preprocess_dataset_raw_config expands this into
  // one internal dataset per resolution). 16:4:1 ratio biases learning
  // toward 512 while still seeing 768 and 1024 to anchor higher-res quality.
  const dataset: any = {
    folder_path: existingDatasetPath,
    mask_path: null,
    mask_min_value: 0.1,
    default_caption: '',
    caption_ext: 'txt',
    caption_dropout_rate: 0.05,
    cache_latents_to_disk: true,
    is_reg: false,
    network_weight: 1,
    resolution: [512, 768, 1024],
    controls: [],
    shrink_video_to_frames: true,
    num_frames: 1,
    flip_x: false,
    flip_y: false,
    num_repeats: [16, 4, 1],
    diffusion_loss_weight: 1,
    depth_loss_weight: 0.005,
    loss_split: 'sum',
  };

  if (opts.subjectMask) {
    // Per-region diffusion weights — only applied when subject_mask is on.
    // Captions should describe ONLY the changeable parts (clothing, expression,
    // pose) and skip background/setting. The mask makes "ignore background"
    // spatial; the captions make "everything captioned is promptable" semantic.
    dataset.background_loss_weight = 0;
    dataset.clothing_loss_weight = 1;
    dataset.body_loss_weight = 1;
  }

  return {
    job: 'extension',
    config: {
      name: existingName,
      process: [
        {
          type: 'diffusion_trainer',
          // Relative path resolves to <CWD>/output which is the same place
          // the UI's defaultTrainFolder points. On Runpod this is symlinked
          // to /workspace/output for persistence (see docker/start.sh).
          training_folder: 'output',
          sqlite_db_path: './aitk_db.db',
          device: 'cuda',
          trigger_word: null,
          performance_log_every: 10,
          network: {
            type: 'lokr',
            linear: 32,
            linear_alpha: 32,
            conv: 16,
            conv_alpha: 16,
            lokr_full_rank: true,
            lokr_factor: 8,
            network_kwargs: { ignore_if_contains: [] },
          },
          save: {
            dtype: 'bf16',
            save_every: 25,
            max_step_saves_to_keep: 120,
            save_format: 'diffusers',
            push_to_hub: false,
          },
          datasets: [dataset],
          train: {
            weight_noise: {
              enabled: true,
              mode: 'relative',
              sigma: 0.0125,
              log_every: 1,
            },
            max_grad_norm: 1,
            batch_size: 4,
            bypass_guidance_embedding: false,
            steps: 1200,
            gradient_accumulation: 1,
            train_unet: true,
            train_text_encoder: false,
            gradient_checkpointing: true,
            noise_scheduler: 'flowmatch',
            optimizer: 'adamw8bit',
            timestep_type: 'linear',
            content_or_style: 'balanced',
            optimizer_params: { weight_decay: 0.0001 },
            unload_text_encoder: false,
            cache_text_embeddings: true,
            lr: 5e-5,
            ema_config: { use_ema: false, ema_decay: 0.99 },
            skip_first_sample: false,
            force_first_sample: true,
            disable_sampling: false,
            dtype: 'bf16',
            diff_output_preservation: false,
            diff_output_preservation_multiplier: 1,
            diff_output_preservation_class: 'person',
            switch_boundary_every: 1,
            loss_type: 'mse',
            diffusion_loss_weight: 1,
            diffusion_loss_max_t: 1,
            diffusion_loss_min_t: 0,
            custom_timestep_distribution: null,
            custom_timestep_curve: null,
            max_denoising_steps: 999,
            min_denoising_steps: 0,
            loss_split: null,
          },
          logging: { log_every: 1, use_ui_logger: true },
          model: {
            // Default to the HuggingFace Flux 2 Klein 9B release; user can
            // swap this for a local checkpoint after applying the template.
            name_or_path: 'black-forest-labs/FLUX.2-klein-base-9B',
            quantize: true,
            qtype: 'qfloat8',
            quantize_te: true,
            qtype_te: 'qfloat8',
            arch: 'flux2_klein_9b',
            low_vram: false,
            model_kwargs: { match_target_res: false },
            layer_offloading: false,
            layer_offloading_text_encoder_percent: 1,
            layer_offloading_transformer_percent: 1,
          },
          sample: {
            sampler: 'flowmatch',
            sample_every: 100,
            width: 640,
            height: 960,
            samples: [
              { prompt: 'a photo of a person' },
            ],
            neg: '',
            seed: 42,
            walk_seed: false,
            guidance_scale: 4,
            sample_steps: 25,
            num_frames: 1,
            fps: 1,
          },
          face_id: {
            enabled: false,
            init_scale: 0.3,
            identity_loss_weight: 0,
            landmark_loss_weight: 0,
            body_proportion_loss_weight: 0,
            body_proportion_loss_min_t: 0.8,
            body_proportion_loss_max_t: 1,
            body_shape_loss_weight: 0,
            body_shape_loss_max_t: 1,
            body_shape_loss_min_t: 0.8,
            identity_loss_min_t: 0,
            identity_loss_max_t: 0.9,
            identity_metrics: false,
            identity_loss_use_average: true,
            identity_loss_min_cos: 0.4,
          },
          depth_consistency: {
            loss_weight: 0.005,
            input_size: 518,
            preview_every: 1,
            // 'subject' restricts the depth-loss perceptor to the cached
            // subject mask; 'none' applies it full-image. Tied to the
            // subject_mask.enabled toggle below.
            mask_source: opts.subjectMask ? 'subject' : 'none',
            loss_max_t: 1,
            model_id: 'depth-anything/Depth-Anything-V2-Large-hf',
            loss_min_t: 0,
            grad_checkpoint: true,
            ssi_weight: 0,
            grad_weight: 1,
            grad_scales: 6,
          },
          subject_mask: {
            enabled: opts.subjectMask,
            background_loss_weight: 0,
            clothing_loss_weight: 1,
            save_debug_previews: true,
            cache_resolution: 768,
            segformer_res: 768,
          },
          ...(existingGpuIds !== undefined ? { gpu_ids: existingGpuIds } : {}),
        } as any,
      ],
    },
    meta: { name: '[name]', version: '1.0' },
  } as unknown as JobConfig;
}

const subjectLikenessFlux2Klein9b: QuickstartTemplate = {
  id: 'subject_likeness_flux2_klein9b',
  label: 'Subject Likeness (Flux 2 Klein 9B + Weight Noise)',
  description:
    'LoKr + weight noise (relative σ=0.0125) on Flux 2 Klein 9B. Single dataset ' +
    'with 512/768/1024 buckets at 16:4:1 num_repeats, full-image depth-consistency, ' +
    'AdamW8bit @ lr=5e-5, batch=4, 1200 steps. Captions describe everything in ' +
    'the image.',
  apply: (current) => buildSubjectLikenessConfig(current, { subjectMask: false }),
};

const subjectLikenessMaskedFlux2Klein9b: QuickstartTemplate = {
  id: 'subject_likeness_masked_flux2_klein9b',
  label: 'Subject Likeness, Masked (Flux 2 Klein 9B + Weight Noise)',
  description:
    'Same as the default Subject Likeness template, plus subject masking with ' +
    'background:0 / clothing:1 / body:1 region weights and depth-consistency ' +
    'restricted to the subject mask. Captions should describe only the ' +
    'changeable parts of the character (clothing, expression, pose) and skip ' +
    'the background / setting. Better results when you have caption discipline.',
  apply: (current) => buildSubjectLikenessConfig(current, { subjectMask: true }),
};

// ---------------------------------------------------------------------------
// Style LoRA (Flux 2 Klein 9B + Depth Anchor)
// ---------------------------------------------------------------------------
// Small-dataset STYLE recipe, abstracted from the validated Yoshitaka Amano
// run (see the "Yoshitaka Amano Style" example in the README). On a tiny style
// set the diffusion loss alone overfits to the specific training compositions;
// the full-image depth anchor plus diffusion/depth loss splitting pushes the
// LoRA toward what is invariant across the artist's work (linework, color
// treatment, paper) and away from memorizing individual images. No weight
// noise (the Amano run didn't use it); the block is left disabled so it is one
// toggle away. No subject mask, since this trains a style and not a subject.

function buildStyleDepthAnchorConfig(current: JobConfig): JobConfig {
  const existingName = current.config.name || '';
  const existingDatasetPath =
    current.config.process[0]?.datasets?.[0]?.folder_path || '';
  const existingGpuIds = (current.config.process[0] as any)?.gpu_ids;

  // Single 768 bucket. loss_split: diffusion_depth runs the diffusion step on
  // appearance and the depth step on structure; they only really agree on the
  // high-level "this looks like the style" signal.
  const dataset: any = {
    folder_path: existingDatasetPath,
    mask_path: null,
    mask_min_value: 0.1,
    default_caption: '',
    caption_ext: 'txt',
    caption_dropout_rate: 0.05,
    cache_latents_to_disk: true,
    is_reg: false,
    network_weight: 1,
    resolution: [768],
    controls: [],
    shrink_video_to_frames: true,
    num_frames: 1,
    flip_x: false,
    flip_y: false,
    num_repeats: 1,
    diffusion_loss_weight: 1,
    depth_loss_weight: 0.005,
    identity_loss_weight: 0,
    loss_split: 'diffusion_depth',
  };

  return {
    job: 'extension',
    config: {
      name: existingName,
      process: [
        {
          type: 'diffusion_trainer',
          training_folder: 'output',
          sqlite_db_path: './aitk_db.db',
          device: 'cuda',
          trigger_word: null,
          performance_log_every: 10,
          network: {
            type: 'lokr',
            linear: 32,
            linear_alpha: 32,
            conv: 16,
            conv_alpha: 16,
            lokr_full_rank: true,
            lokr_factor: 8,
            network_kwargs: { ignore_if_contains: [] },
          },
          save: {
            dtype: 'bf16',
            save_every: 50,
            max_step_saves_to_keep: 120,
            save_format: 'diffusers',
            push_to_hub: false,
          },
          datasets: [dataset],
          train: {
            // Disabled to match the Amano run. Flip enabled: true (sigma stays
            // a sane 0.0125) to stack weight noise on top of the depth anchor.
            weight_noise: {
              enabled: false,
              mode: 'relative',
              sigma: 0.0125,
              log_every: 1,
            },
            max_grad_norm: 1,
            batch_size: 2,
            bypass_guidance_embedding: false,
            steps: 4000,
            gradient_accumulation: 1,
            train_unet: true,
            train_text_encoder: false,
            gradient_checkpointing: true,
            noise_scheduler: 'flowmatch',
            optimizer: 'adamw8bit',
            timestep_type: 'weighted',
            content_or_style: 'balanced',
            optimizer_params: { weight_decay: 0.0001 },
            unload_text_encoder: false,
            cache_text_embeddings: true,
            lr: 5e-5,
            ema_config: { use_ema: false, ema_decay: 0.99 },
            skip_first_sample: false,
            force_first_sample: true,
            disable_sampling: false,
            dtype: 'bf16',
            diff_output_preservation: false,
            diff_output_preservation_multiplier: 1,
            diff_output_preservation_class: 'person',
            switch_boundary_every: 1,
            loss_type: 'mse',
            diffusion_loss_weight: 1,
            diffusion_loss_max_t: 1,
            diffusion_loss_min_t: 0,
            custom_timestep_distribution: null,
            custom_timestep_curve: null,
            max_denoising_steps: 999,
            min_denoising_steps: 0,
            loss_split: null,
          },
          logging: { log_every: 1, use_ui_logger: true },
          model: {
            name_or_path: 'black-forest-labs/FLUX.2-klein-base-9B',
            quantize: true,
            qtype: 'qfloat8',
            quantize_te: true,
            qtype_te: 'qfloat8',
            arch: 'flux2_klein_9b',
            low_vram: false,
            model_kwargs: { match_target_res: false },
            layer_offloading: false,
            layer_offloading_text_encoder_percent: 1,
            layer_offloading_transformer_percent: 1,
          },
          sample: {
            sampler: 'flowmatch',
            sample_every: 100,
            width: 640,
            height: 960,
            samples: [
              { prompt: 'a portrait of a person' },
            ],
            neg: '',
            seed: 42,
            walk_seed: false,
            guidance_scale: 4,
            sample_steps: 25,
            num_frames: 1,
            fps: 1,
          },
          face_id: {
            enabled: false,
            init_scale: 0.3,
            identity_loss_weight: 0,
            landmark_loss_weight: 0,
            body_proportion_loss_weight: 0,
            body_proportion_loss_min_t: 0.8,
            body_proportion_loss_max_t: 1,
            body_shape_loss_weight: 0,
            body_shape_loss_max_t: 1,
            body_shape_loss_min_t: 0.8,
            identity_loss_min_t: 0,
            identity_loss_max_t: 0.9,
            identity_metrics: false,
            identity_loss_use_average: true,
            identity_loss_min_cos: 0.4,
          },
          depth_consistency: {
            loss_weight: 0.005,
            input_size: 1400,
            preview_every: 1,
            mask_source: 'none',
            loss_max_t: 0.9,
            model_id: 'depth-anything/Depth-Anything-V2-Large-hf',
            loss_min_t: 0,
            grad_checkpoint: true,
            ssi_weight: 0,
            grad_weight: 1,
            grad_scales: 6,
          },
          subject_mask: {
            enabled: false,
            background_loss_weight: 0,
            clothing_loss_weight: 0,
            save_debug_previews: true,
            cache_resolution: 768,
            segformer_res: 768,
          },
          ...(existingGpuIds !== undefined ? { gpu_ids: existingGpuIds } : {}),
        } as any,
      ],
    },
    meta: { name: '[name]', version: '1.0' },
  } as unknown as JobConfig;
}

const styleDepthAnchorFlux2Klein9b: QuickstartTemplate = {
  id: 'style_lora_flux2_klein9b',
  label: 'Style LoRA (Flux 2 Klein 9B + Depth Anchor)',
  description:
    'Small-dataset style recipe (from the validated Amano run). LoKr + ' +
    'full-image depth-consistency (0.005, DA2-Large @ input_size 1400) with ' +
    'diffusion/depth loss splitting. Single 768 bucket, AdamW8bit @ lr=5e-5, ' +
    'batch=2, grad-accum=1, 4000 steps. No weight noise (block left disabled) ' +
    'and no subject mask: the depth anchor keeps a tiny style set from ' +
    'overfitting to specific training compositions.',
  apply: (current) => buildStyleDepthAnchorConfig(current),
};

// ---------------------------------------------------------------------------
// Subject Likeness — Z-Image Turbo + Weight Noise
// ---------------------------------------------------------------------------
// Z-Image Turbo recipe abstracted from a validated production run. Trains
// through the de-distill assistant LoRA (so the optimizer sees a "base-like"
// model and the trained LoRA inverts cleanly back to 8-step turbo at sample
// time). No transformer quantization — Tongyi-MAI explicitly warns against
// FP8 on Turbo. Custom timestep distribution + curve front-load training on
// the timesteps where Turbo's distillation needs the most correction.

function buildSubjectLikenessZImageTurboConfig(current: JobConfig): JobConfig {
  const existingName = current.config.name || '';
  const existingDatasetPath =
    current.config.process[0]?.datasets?.[0]?.folder_path || '';
  const existingGpuIds = (current.config.process[0] as any)?.gpu_ids;

  // Single 1024 bucket — matches the validated production run. Multi-bucket
  // (1024/768/512) was tried and was empirically worse for ZiT than holding
  // the bucket at native resolution.
  const dataset: any = {
    folder_path: existingDatasetPath,
    mask_path: null,
    mask_min_value: 0.1,
    default_caption: '',
    caption_ext: 'txt',
    caption_dropout_rate: 0.05,
    cache_latents_to_disk: true,
    is_reg: false,
    network_weight: 1,
    resolution: [1024],
    controls: [],
    shrink_video_to_frames: true,
    num_frames: 1,
    flip_x: false,
    flip_y: false,
    num_repeats: 1,
    diffusion_loss_weight: 1,
    depth_loss_weight: 0.005,
    loss_split: 'sum',
  };

  return {
    job: 'extension',
    config: {
      name: existingName,
      process: [
        {
          type: 'diffusion_trainer',
          training_folder: 'output',
          sqlite_db_path: './aitk_db.db',
          device: 'cuda',
          trigger_word: null,
          performance_log_every: 10,
          network: {
            type: 'lokr',
            linear: 32,
            linear_alpha: 32,
            conv: 16,
            conv_alpha: 16,
            lokr_full_rank: true,
            lokr_factor: 8,
            network_kwargs: { ignore_if_contains: [] },
          },
          save: {
            dtype: 'bf16',
            save_every: 100,
            max_step_saves_to_keep: 120,
            save_format: 'diffusers',
            push_to_hub: false,
          },
          datasets: [dataset],
          train: {
            weight_noise: {
              enabled: true,
              mode: 'relative',
              sigma: 0.0125,
              log_every: 1,
            },
            max_grad_norm: 1,
            batch_size: 4,
            bypass_guidance_embedding: false,
            // ZiT can take ~200 steps per training image to converge, so a
            // 10 to 15 image dataset wants something in the 2000 to 3000 range.
            steps: 3000,
            gradient_accumulation: 1,
            train_unet: true,
            train_text_encoder: false,
            gradient_checkpointing: true,
            noise_scheduler: 'flowmatch',
            optimizer: 'adamw8bit',
            timestep_type: 'linear',
            content_or_style: 'balanced',
            optimizer_params: { weight_decay: 0.0001 },
            unload_text_encoder: false,
            cache_text_embeddings: true,
            lr: 2.5e-4,
            ema_config: { use_ema: false, ema_decay: 0.99 },
            skip_first_sample: false,
            force_first_sample: true,
            disable_sampling: false,
            dtype: 'bf16',
            diff_output_preservation: false,
            diff_output_preservation_multiplier: 1,
            diff_output_preservation_class: 'person',
            switch_boundary_every: 1,
            loss_type: 'mse',
            diffusion_loss_weight: 1,
            diffusion_loss_max_t: 1,
            diffusion_loss_min_t: 0,
            // Front-load training on the steps Turbo's distillation needs
            // the most correction at — high t (early denoising) for the
            // distribution and low-mid t for the curve. Trainer evaluates
            // these via PCHIP at run time.
            custom_timestep_distribution: {
              points: [
                { x: 0, y: 1 },
                { x: 0.048801606351679024, y: 3 },
                { x: 0.12443620508367365, y: 1.1754955492521588 },
                { x: 1, y: 1.124230987147281 },
              ],
              normalize: false,
              sourceName: 'high_90s',
            },
            custom_timestep_curve: {
              points: [
                { x: 0, y: 1 },
                { x: 0.6226329803466797, y: 2.0341763245432 },
                { x: 0.6226329803466797, y: 2.0341763245432 },
                { x: 1, y: 1 },
              ],
              normalize: true,
              sourceName: 'boost_low_mid_t',
            },
            max_denoising_steps: 999,
            min_denoising_steps: 0,
            loss_split: null,
          },
          logging: { log_every: 1, use_ui_logger: true },
          model: {
            name_or_path: 'Tongyi-MAI/Z-Image-Turbo',
            // Tongyi-MAI explicitly warns that FP8 on Z-Image Turbo causes
            // noticeable quality degradation — keep the transformer in bf16.
            // Text encoder quantization is fine.
            quantize: false,
            qtype: 'qfloat8',
            quantize_te: true,
            qtype_te: 'qfloat8',
            arch: 'zimage:turbo',
            low_vram: false,
            model_kwargs: {},
            layer_offloading: false,
            layer_offloading_text_encoder_percent: 1,
            layer_offloading_transformer_percent: 1,
            // De-distill adapter — merged at load with +1.0, runtime
            // inverted to -1.0 at sample time. See ZImageModel.load_training_adapter.
            assistant_lora_path:
              'ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors',
          },
          sample: {
            sampler: 'flowmatch',
            sample_every: 100,
            width: 1024,
            height: 1024,
            samples: [{ prompt: 'a photo of a person' }],
            neg: '',
            seed: 42,
            walk_seed: false,
            // Turbo expects guidance_scale=1 and 8 steps. Higher CFG
            // produces washed-out output.
            guidance_scale: 1,
            sample_steps: 8,
            num_frames: 1,
            fps: 1,
          },
          face_id: {
            enabled: false,
            init_scale: 0.3,
            identity_loss_weight: 0,
            landmark_loss_weight: 0,
            body_proportion_loss_weight: 0,
            body_proportion_loss_min_t: 0.8,
            body_proportion_loss_max_t: 1,
            body_shape_loss_weight: 0,
            body_shape_loss_max_t: 1,
            body_shape_loss_min_t: 0.8,
            identity_loss_min_t: 0,
            identity_loss_max_t: 0.9,
            identity_metrics: false,
            identity_loss_use_average: true,
            identity_loss_min_cos: 0.4,
          },
          depth_consistency: {
            loss_weight: 0.005,
            input_size: 518,
            preview_every: 1,
            mask_source: 'subject',
            loss_max_t: 1,
            model_id: 'depth-anything/Depth-Anything-V2-Large-hf',
            loss_min_t: 0,
            grad_checkpoint: true,
            ssi_weight: 0,
            grad_weight: 1,
            grad_scales: 6,
          },
          subject_mask: {
            enabled: true,
            background_loss_weight: 0,
            clothing_loss_weight: 1,
            save_debug_previews: true,
            cache_resolution: 768,
            segformer_res: 768,
          },
          ...(existingGpuIds !== undefined ? { gpu_ids: existingGpuIds } : {}),
        } as any,
      ],
    },
    meta: { name: '[name]', version: '1.0' },
  } as unknown as JobConfig;
}

const subjectLikenessZImageTurbo: QuickstartTemplate = {
  id: 'subject_likeness_zimage_turbo',
  label: 'Subject Likeness (Z-Image Turbo + Weight Noise)',
  description:
    'LoKr + weight noise (relative σ=0.0125) on Z-Image Turbo via the ' +
    'de-distill training adapter. Single 1024 bucket (multi-bucket was tried ' +
    'and empirically worse), subject-masked depth-consistency (background:0 ' +
    '/ clothing:1), AdamW8bit @ lr=2.5e-4, batch=4, 3000 steps. Custom ' +
    'timestep distribution + curve front-load high-t and low-mid-t training. ' +
    'Transformer kept in bf16 (Tongyi-MAI warns against FP8 on Turbo); text ' +
    'encoder quantized.',
  apply: (current) => buildSubjectLikenessZImageTurboConfig(current),
};

export const QUICKSTARTS: QuickstartTemplate[] = [
  subjectLikenessFlux2Klein9b,
  subjectLikenessMaskedFlux2Klein9b,
  styleDepthAnchorFlux2Klein9b,
  subjectLikenessZImageTurbo,
];
