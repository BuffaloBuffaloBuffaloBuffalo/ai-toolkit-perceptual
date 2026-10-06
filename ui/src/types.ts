/**
 * GPU API response
 */

export interface GpuUtilization {
  gpu: number;
  memory: number;
}

export interface GpuMemory {
  total: number;
  free: number;
  used: number;
  // unified memory (e.g. GB10): figures are the system RAM pool the GPU shares
  shared?: boolean;
}

export interface GpuPower {
  draw: number;
  limit: number;
}

export interface GpuClocks {
  graphics: number;
  memory: number;
}

export interface GpuFan {
  speed: number;
}

export interface GpuInfo {
  index: number;
  name: string;
  driverVersion: string;
  temperature: number;
  utilization: GpuUtilization;
  memory: GpuMemory;
  power: GpuPower;
  clocks: GpuClocks;
  fan: GpuFan;
}

export interface CpuInfo {
  name: string;
  cores: number;
  temperature: number;
  totalMemory: number;
  freeMemory: number;
  availableMemory: number;
  currentLoad: number;
}

export interface GPUApiResponse {
  hasNvidiaSmi: boolean;
  isMac: boolean;
  gpus: GpuInfo[];
  error?: string;
}

/**
 * System monitor stream (SSE at /api/monitor)
 */

// Rolling history only logs load + memory; everything else (temps, fans,
// power, clocks) is instantaneous-only via MonitorSample.
export interface MonitorHistoryPoint {
  t: number; // epoch ms
  cpu: { load: number; memUsedMb: number };
  // one entry per GPU, same order as MonitorSample.gpu.gpus (sorted by index)
  gpus: { load: number; memUsedMb: number }[];
}

export interface MonitorSample {
  t: number;
  cpu: CpuInfo | null;
  gpu: GPUApiResponse;
}

export interface MonitorInit extends MonitorSample {
  history: MonitorHistoryPoint[];
}

/**
 * Training configuration
 */

export interface NetworkConfig {
  type: string;
  linear: number;
  linear_alpha: number;
  conv: number;
  conv_alpha: number;
  lokr_full_rank: boolean;
  lokr_factor: number;
  network_kwargs: {
    ignore_if_contains: string[];
  };
  transformer_only?: boolean;
}

export interface SaveConfig {
  dtype: string;
  save_every: number;
  max_step_saves_to_keep: number;
  save_format: string;
  push_to_hub: boolean;
  /** Write a stepped copy of the optimizer state alongside each checkpoint
   *  (e.g. ``{job}_000000250_optimizer.pt``) so stateful optimizers can be
   *  rolled back to a specific checkpoint. The canonical ``optimizer.pt`` is
   *  still written for resume-from-latest. Stepped copies are pruned by
   *  ``max_step_saves_to_keep``. Defaults to false. */
  save_optimizer_per_checkpoint?: boolean;
}

export interface DatasetConfig {
  batch_size?: number;
  folder_path: string;
  mask_path: string | null;
  mask_min_value: number;
  default_caption: string;
  caption_ext: string;
  caption_dropout_rate: number;
  shuffle_tokens?: boolean;
  is_reg: boolean;
  network_weight: number;
  cache_latents_to_disk?: boolean;
  resolution: number[];
  controls: string[];
  depth_as_control?: boolean;
  control_path?: string | null;
  num_frames: number;
  shrink_video_to_frames: boolean;
  do_i2v?: boolean;
  do_audio?: boolean;
  audio_normalize?: boolean;
  audio_preserve_pitch?: boolean;
  fps?: number;
  flip_x: boolean;
  flip_y: boolean;
  num_repeats?: number | number[];
  control_path_1?: string | null;
  control_path_2?: string | null;
  control_path_3?: string | null;
  // Per-dataset loss overrides (undefined = inherit global face_id config)
  identity_loss_weight?: number;
  depth_loss_weight?: number;
  depth_loss_min_t?: number;
  depth_loss_max_t?: number;
  loss_split?: 'diffusion_depth' | 'sum' | null;
  identity_loss_min_t?: number;
  identity_loss_max_t?: number;
  identity_loss_min_cos?: number;
  landmark_loss_weight?: number;
  body_proportion_loss_weight?: number;
  body_proportion_loss_min_t?: number;
  body_proportion_loss_max_t?: number;
  body_proportion_include_head?: boolean;
  body_shape_loss_weight?: number;
  body_shape_loss_min_t?: number;
  body_shape_loss_max_t?: number;
  body_shape_loss_min_cos?: number;
  normal_loss_weight?: number;
  normal_loss_min_t?: number;
  normal_loss_max_t?: number;
  vae_anchor_loss_weight?: number;
  vae_anchor_loss_min_t?: number;
  vae_anchor_loss_max_t?: number;
  diffusion_loss_weight?: number;
  diffusion_loss_min_t?: number;
  diffusion_loss_max_t?: number;
  face_suppression_weight?: number;
  face_suppression_expand?: number;
  face_suppression_soft?: boolean;
  latent_perceptual_loss_weight?: number;
  latent_perceptual_loss_min_t?: number;
  latent_perceptual_loss_max_t?: number;
  // Subject mask per-dataset overrides (undefined = inherit global subject_mask config)
  background_loss_weight?: number;
  clothing_loss_weight?: number;
  body_loss_weight?: number;
  perceptual_restrict_to_body?: boolean;
  auto_frame_count?: boolean;
}

export interface EMAConfig {
  use_ema: boolean;
  ema_decay: number;
}

export interface WeightNoiseConfig {
  /** Master toggle. When false the injector is a no-op regardless of other fields. */
  enabled: boolean;
  /** 'relative' (σ × per-param weight RMS) | 'absolute' (fixed σ everywhere). */
  mode: 'relative' | 'absolute';
  /** σ multiplier for 'relative', or fixed σ for 'absolute'. */
  sigma: number;
  /** Cadence for emitting the weight_noise_norm metric. 0 disables logging. */
  log_every: number;
}

export interface ValidationItem {
  image_path: string;
  prompt: string;
}

export interface ValidationConfig {
  validation_items: ValidationItem[];
  resolution: number;
  validate_every_n_steps: number;
  validation_sigmas?: number[];
}

export interface TrainConfig {
  min_denoising_steps?: number;
  max_denoising_steps?: number;
  loss_split?: 'diffusion_depth' | null;
  batch_size: number;
  bypass_guidance_embedding?: boolean;
  steps: number;
  gradient_accumulation: number;
  train_unet: boolean;
  train_text_encoder: boolean;
  gradient_checkpointing: boolean;
  noise_scheduler: string;
  timestep_type: string;
  /** Inlined curve dict when timestep_type === 'custom'. Populated by the
   *  CustomTimestepCurvePicker on job creation; the trainer evaluates it via
   *  PCHIP at run time. See toolkit/timestep_weighing/custom_curve.py. */
  custom_timestep_curve?: { points: { x: number; y: number }[]; normalize?: boolean; sourceName?: string } | null;
  /** Inlined distribution curve when timestep_type === 'custom_distribution'.
   *  Same shape as custom_timestep_curve but interpreted as an unnormalized
   *  PDF — the trainer renormalizes and samples timesteps via inverse CDF. */
  custom_timestep_distribution?: { points: { x: number; y: number }[]; normalize?: boolean; sourceName?: string } | null;
  content_or_style: string;
  optimizer: string;
  lr: number;
  ema_config?: EMAConfig;
  weight_noise?: WeightNoiseConfig;
  dtype: string;
  unload_text_encoder: boolean;
  cache_text_embeddings: boolean;
  optimizer_params: {
    weight_decay: number;
  };
  skip_first_sample: boolean;
  force_first_sample: boolean;
  disable_sampling: boolean;
  diff_output_preservation: boolean;
  diff_output_preservation_multiplier: number;
  diff_output_preservation_class: string;
  blank_prompt_preservation?: boolean;
  blank_prompt_preservation_multiplier?: number;
  switch_boundary_every: number;
  loss_type: 'mse' | 'mae' | 'wavelet' | 'stepped';
  diffusion_loss_weight?: number;
  diffusion_loss_min_t?: number;
  diffusion_loss_max_t?: number;
  latent_perceptual_loss_weight?: number;
  latent_perceptual_loss_min_t?: number;
  latent_perceptual_loss_max_t?: number;
  latent_perceptual_encoder?: string;
  latent_perceptual_preview_every?: number;
  do_differential_guidance?: boolean;
  differential_guidance_scale?: number;
  audio_loss_multiplier?: number;
  max_loss?: number | null;
  validation_config?: ValidationConfig;
  do_guidance_loss?: boolean;
  guidance_loss_target?: number;
}

export interface QuantizeKwargsConfig {
  exclude: string[];
}

export interface MergeLoraConfig {
  path: string;
  weight: number;
  lokr_factor?: number | null;
}

export interface ModelConfig {
  name_or_path: string;
  quantize: boolean;
  quantize_te: boolean;
  qtype: string;
  qtype_te: string;
  quantize_kwargs?: QuantizeKwargsConfig;
  arch: string;
  low_vram: boolean;
  model_kwargs: { [key: string]: any };
  layer_offloading?: boolean;
  layer_offloading_transformer_percent?: number;
  layer_offloading_text_encoder_percent?: number;
  assistant_lora_path?: string;
  merge_loras?: MergeLoraConfig[];
  unconditional_lora_path?: string;
  compile?: boolean;
  block_compile?: boolean;
  compile_mode?: 'default' | 'max-autotune' | 'fastest';
  compile_fullgraph?: boolean;
  compile_dynamic?: boolean;
  cache_size_limit?: number;
}

export interface SampleItem {
  prompt: string;
  width?: number;
  height?: number;
  neg?: string;
  seed?: number;
  guidance_scale?: number;
  sample_steps?: number;
  fps?: number;
  num_frames?: number;
  duration?: number;
  ctrl_img?: string | null;
  ctrl_idx?: number;
  network_multiplier?: number;
  ctrl_img_1?: string | null;
  ctrl_img_2?: string | null;
  ctrl_img_3?: string | null;
}

export interface SampleConfig {
  sampler: string;
  sample_every: number;
  sample_start_step: number;
  width: number;
  height: number;
  prompts?: string[];
  samples: SampleItem[];
  neg: string;
  seed: number;
  walk_seed: boolean;
  guidance_scale: number;
  sample_steps: number;
  num_frames: number;
  fps: number;
  duration?: number;
}

export interface LoggingConfig {
  log_every: number;
  use_ui_logger: boolean;
}

export interface SliderConfig {
  guidance_strength?: number;
  anchor_strength?: number;
  positive_prompt?: string;
  negative_prompt?: string;
  target_class?: string;
  anchor_class?: string | null;
}

export interface FaceIDConfig {
  enabled: boolean;
  num_tokens: number;
  dropout_prob: number;
  face_model: string;
  scale_lr_multiplier: number;
  init_scale: number;
  vision_enabled?: boolean;
  vision_model?: string;
  vision_num_tokens?: number;
  vision_crop_padding?: number;
  identity_loss_weight?: number;
  identity_loss_min_t?: number;
  identity_loss_max_t?: number;
  identity_loss_min_cos?: number;
  identity_loss_use_average?: boolean;
  identity_loss_average_blend?: number;
  identity_loss_use_random?: boolean;
  identity_loss_num_refs?: number;
  // Keep only the newest N face preview images in id_previews/ (<=0 = keep all)
  identity_loss_preview_max_keep?: number;
  identity_metrics?: boolean;
  landmark_loss_weight?: number;
  body_proportion_loss_weight?: number;
  body_proportion_loss_min_t?: number;
  body_proportion_loss_max_t?: number;
  body_proportion_include_head?: boolean;
  body_shape_loss_weight?: number;
  body_shape_loss_min_t?: number;
  body_shape_loss_max_t?: number;
  normal_loss_weight?: number;
  normal_loss_min_t?: number;
  normal_loss_max_t?: number;
  vae_anchor_loss_weight?: number;
  vae_anchor_loss_min_t?: number;
  vae_anchor_loss_max_t?: number;
  vae_anchor_model_path?: string;
  face_suppression_weight?: number;
  face_suppression_expand?: number;
  face_suppression_soft?: boolean;
}

export interface BodyIDConfig {
  enabled: boolean;
  num_tokens: number;
  dropout_prob: number;
  detection_threshold: number;
  scale_lr_multiplier: number;
  init_scale: number;
}

export interface SubjectMaskConfig {
  body_close_radius?: number;
  enabled: boolean;
  yolo_ckpt?: string;
  yolo_conf?: number;
  primary_only?: boolean;
  sam_size?: 'tiny' | 'small' | 'base_plus' | 'large';
  segformer_res?: number;
  cache_resolution?: number;
  dtype?: 'fp16' | 'bf16' | 'fp32';
  // Region loss-weight knobs — undefined = no-op (no weighting applied)
  background_loss_weight?: number;
  clothing_loss_weight?: number;
  body_loss_weight?: number;
  perceptual_restrict_to_body?: boolean;
  // Debug: when true, cache_subject_masks writes a 5-panel tile.png per image
  // to _face_id_cache/_previews/ for visual inspection
  save_debug_previews?: boolean;
}

export interface DepthConsistencyConfig {
  // Enable by setting loss_weight > 0
  loss_weight?: number;
  loss_min_t?: number;
  loss_max_t?: number;
  // Frozen Depth-Anything-V2 perceptor
  model_id?: string;
  input_size?: number;
  // Loss composition (MiDaS formulation)
  ssi_weight?: number;
  grad_weight?: number;
  grad_scales?: number;
  // Spatial mask source
  mask_source?: 'none' | 'subject' | 'body';
  // Memory controls
  grad_checkpoint?: boolean;
  // Preview cadence (steps); 0 disables
  preview_every?: number;
  // Keep only the newest N preview files in depth_previews/ (<=0 = keep all)
  preview_max_keep?: number;
}

export interface ProcessConfig {
  type: string;
  sqlite_db_path?: string;
  training_folder: string;
  performance_log_every: number;
  trigger_word: string | null;
  device: string;
  network?: NetworkConfig;
  slider?: SliderConfig;
  face_id?: FaceIDConfig;
  body_id?: BodyIDConfig;
  subject_mask?: SubjectMaskConfig;
  depth_consistency?: DepthConsistencyConfig;
  save: SaveConfig;
  datasets: DatasetConfig[];
  train: TrainConfig;
  logging: LoggingConfig;
  model: ModelConfig;
  sample: SampleConfig;
}

export interface ConfigObject {
  name: string;
  process: ProcessConfig[];
}

export interface MetaConfig {
  name: string;
  version: string;
}

export interface JobConfig {
  job: string;
  config: ConfigObject;
  meta: MetaConfig;
}

// A LoRA published on the hub, offered for a specific model option. `path` is a
// 'org/repo/path_to/file.safetensors' reference; the backend looks for it under
// the models folder first and downloads it into MODELS_PATH/loras if missing.
export interface CloudLora {
  path: string;
  name: string;
  description?: string;
}

export interface CaptionLora {
  path: string;
  name: string;
  strength: number;
}

export interface CaptionProcessConfig {
  type: string;
  sqlite_db_path?: string;
  device: string;
  caption: {
    model_name_or_path: string;
    model_name_or_path2?: string;
    dtype: string;
    quantize: boolean;
    qtype: string;
    low_vram: boolean;
    extensions: string[];
    path_to_caption: string;
    recaption: boolean;
    compile?: boolean;
    caption_prompt?: string;
    max_res?: number;
    max_new_tokens?: number;
    fixed_caption?: string;
    caption_format?: string;
    extract_vocals_before_transcribe?: boolean;
    keep_timestamps?: boolean;
    caption_extension?: string;
    thinking?: boolean;
    batch_size?: number;
    layer_offloading?: boolean;
    layer_offloading_percent?: number;
    loras?: CaptionLora[];
  }
}

export interface CaptionConfigObject {
  name: string;
  process: CaptionProcessConfig[];
}

export interface CaptionJobConfig {
  job: string;
  config: CaptionConfigObject;
}

export interface ConfigDoc {
  title: string | React.ReactNode;
  description: React.ReactNode;
}

export interface SelectOption {
  readonly value: string;
  readonly label: string;
}
export interface GroupedSelectOption {
  readonly label: string;
  readonly options: SelectOption[];
}

export type JobStatus = 'queued' | 'running' | 'stopping' | 'stopped' | 'completed' | 'error';
