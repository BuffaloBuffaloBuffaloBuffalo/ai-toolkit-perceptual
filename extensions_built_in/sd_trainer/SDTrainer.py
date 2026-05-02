import os
import random
import contextlib
from collections import OrderedDict
from typing import Union, Literal, List, Optional

import numpy as np
from einops import rearrange
from diffusers import T2IAdapter, AutoencoderTiny, ControlNetModel

import torch.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader, ConcatDataset

from toolkit import train_tools
from toolkit.basic import value_map, adain, get_mean_std
from toolkit.clip_vision_adapter import ClipVisionAdapter
from toolkit.config_modules import GenerateImageConfig
from toolkit.data_loader import get_dataloader_datasets
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO, FileItemDTO
from toolkit.guidance import get_targeted_guidance_loss, get_guidance_loss, GuidanceType
from toolkit.image_utils import show_tensors, show_latents
from toolkit.ip_adapter import IPAdapter
from toolkit.custom_adapter import CustomAdapter
from toolkit.print import print_acc
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
from toolkit.reference_adapter import ReferenceAdapter
from toolkit.stable_diffusion_model import StableDiffusion, BlankNetwork
from toolkit.train_tools import get_torch_dtype, apply_snr_weight, add_all_snr_to_noise_scheduler, \
    apply_learnable_snr_gos, LearnableSNRGamma
import gc
import torch
from jobs.process import BaseSDTrainProcess
from torchvision import transforms
from diffusers import EMAModel
import math
from toolkit.train_tools import precondition_model_outputs_flow_match
from toolkit.models.diffusion_feature_extraction import DiffusionFeatureExtractor, load_dfe
from toolkit.util.losses import wavelet_loss, stepped_loss
import torch.nn.functional as F
from toolkit.unloader import unload_text_encoder
from toolkit.config_modules import FaceIDConfig, BodyIDConfig, SubjectMaskConfig, DepthConsistencyConfig
from toolkit.face_id import FaceIDProjector, VisionFaceProjector, DifferentiableFaceEncoder, DifferentiableLandmarkEncoder, cache_face_embeddings
from toolkit.body_id import BodyIDProjector, DifferentiableBodyProportionEncoder, cache_body_embeddings, cache_body_proportion_embeddings
from toolkit.body_shape import DifferentiableBodyShapeEncoder, cache_body_shape_embeddings
from toolkit.normal_id import DifferentiableNormalEncoder, cache_normal_embeddings
from toolkit.vae_anchor import VAEAnchorEncoder, cache_vae_anchor_features
from toolkit.subject_mask import cache_subject_masks
from toolkit.depth_consistency import (
    DifferentiableDepthEncoder,
    compute_depth_consistency_loss,
    cache_depth_gt_embeddings,
    cache_video_depth_gt_embeddings,
    load_taehv_wan21,
    decode_wan_x0_to_frames,
    save_video_depth_preview,
)
from PIL import Image
from torchvision.transforms import functional as TF


def flush():
    torch.cuda.empty_cache()
    gc.collect()


adapter_transforms = transforms.Compose([
    transforms.ToTensor(),
])


class SDTrainer(BaseSDTrainProcess):

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        self.assistant_adapter: Union['T2IAdapter', 'ControlNetModel', None]
        self.do_prior_prediction = False
        self.do_long_prompts = False
        self.do_guided_loss = False
        self.taesd: Optional[AutoencoderTiny] = None

        self._clip_image_embeds_unconditional: Union[List[str], None] = None
        self.negative_prompt_pool: Union[List[str], None] = None
        self.batch_negative_prompt: Union[List[str], None] = None

        self.is_bfloat = self.train_config.dtype == "bfloat16" or self.train_config.dtype == "bf16"

        self.do_grad_scale = True
        if self.is_fine_tuning and self.is_bfloat:
            self.do_grad_scale = False
        if self.adapter_config is not None:
            if self.adapter_config.train:
                self.do_grad_scale = False

        # if self.train_config.dtype in ["fp16", "float16"]:
        #     # patch the scaler to allow fp16 training
        #     org_unscale_grads = self.scaler._unscale_grads_
        #     def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
        #         return org_unscale_grads(optimizer, inv_scale, found_inf, True)
        #     self.scaler._unscale_grads_ = _unscale_grads_replacer

        self.cached_blank_embeds: Optional[PromptEmbeds] = None
        self.cached_trigger_embeds: Optional[PromptEmbeds] = None
        self.diff_output_preservation_embeds: Optional[PromptEmbeds] = None
        
        self.dfe: Optional[DiffusionFeatureExtractor] = None
        self.unconditional_embeds = None
        
        if self.train_config.diff_output_preservation:
            if self.trigger_word is None:
                raise ValueError("diff_output_preservation requires a trigger_word to be set")
            if self.network_config is None:
                raise ValueError("diff_output_preservation requires a network to be set")
            if self.train_config.train_text_encoder:
                raise ValueError("diff_output_preservation is not supported with train_text_encoder")
        
        if self.train_config.blank_prompt_preservation:
            if self.network_config is None:
                raise ValueError("blank_prompt_preservation requires a network to be set")
        
        if self.train_config.blank_prompt_preservation or self.train_config.diff_output_preservation:
            # always do a prior prediction when doing output preservation
            self.do_prior_prediction = True
        
        # LoRA+ID face conditioning
        face_id_raw = self.get_conf('face_id', None)
        if face_id_raw is not None:
            self.face_id_config = FaceIDConfig(**face_id_raw)
        else:
            self.face_id_config = None

        # Auto-masking (Phase 1: caching only — not yet consumed by loss)
        subject_mask_raw = self.get_conf('subject_mask', None)
        if subject_mask_raw is not None:
            self.subject_mask_config = SubjectMaskConfig(**subject_mask_raw)
        else:
            self.subject_mask_config = None

        # Depth-consistency loss (MiDaS SSI + multi-scale gradient via DA2)
        depth_consistency_raw = self.get_conf('depth_consistency', None)
        if depth_consistency_raw is not None:
            self.depth_consistency_config = DepthConsistencyConfig(**depth_consistency_raw)
        else:
            self.depth_consistency_config = None

        self.face_id_projector: Optional[FaceIDProjector] = None
        self.vision_face_projector: Optional[VisionFaceProjector] = None
        self.id_loss_model: Optional[DifferentiableFaceEncoder] = None
        self.landmark_loss_model: Optional[DifferentiableLandmarkEncoder] = None
        self.body_proportion_model: Optional[DifferentiableBodyProportionEncoder] = None
        self.body_shape_model: Optional[DifferentiableBodyShapeEncoder] = None
        self.normal_model: Optional[DifferentiableNormalEncoder] = None
        self.vae_anchor_encoder: Optional[VAEAnchorEncoder] = None
        self.vae_anchor_projector = None
        self.depth_encoder: Optional[DifferentiableDepthEncoder] = None
        # Wan 2.1 video path: TAEHV tiny decoder for x0 → frames. Lazy-loaded
        # the first time a 5D noise_pred arrives through the depth block.
        self._wan_depth_decoder = None
        self._last_depth_consistency_loss: Optional[float] = None
        self._last_depth_consistency_loss_applied: Optional[float] = None
        self._last_depth_consistency_ssi: Optional[float] = None
        self._last_depth_consistency_grad: Optional[float] = None
        self._last_depth_loss_bins: Optional[dict] = None
        self._last_identity_loss: Optional[float] = None
        self._last_landmark_loss: Optional[float] = None
        self._last_body_proportion_loss: Optional[float] = None
        self._last_body_proportion_loss_applied: Optional[float] = None
        self._last_bp_sim_bins: Optional[dict] = None
        self._last_body_shape_loss: Optional[float] = None
        self._last_body_shape_loss_applied: Optional[float] = None
        self._last_body_shape_cos: Optional[float] = None
        self._last_body_shape_l1: Optional[float] = None
        self._last_body_shape_gated_pct: Optional[float] = None
        self._last_bsh_sim_bins: Optional[dict] = None
        self._last_normal_loss: Optional[float] = None
        self._last_normal_loss_applied: Optional[float] = None
        self._last_normal_cos: Optional[float] = None
        self._last_vae_anchor_loss: Optional[float] = None
        self._last_vae_anchor_loss_applied: Optional[float] = None
        self._last_vae_anchor_per_level: Optional[dict] = None
        self._last_diffusion_loss: Optional[float] = None
        self._last_diffusion_loss_applied: Optional[float] = None
        self._last_diffusion_loss_bins: Optional[dict] = None
        # Gradient cosine diagnostic: norms of the depth-only and
        # everything-else-only gradients at the LoRA params, plus their
        # cosine. Populated by the dual-backward path in
        # `train_single_accumulation` only when
        # `train_config.gradient_cosine_log_every > 0` and the step matches.
        # Note: norms reported are AMP-scaled (matches what the grad scaler
        # produces); the cosine is scale-invariant and the ratio of the two
        # norms is preserved, so all three are still meaningful as diagnostics.
        self._last_grad_norm_diffusion: Optional[float] = None
        self._last_grad_norm_depth: Optional[float] = None
        self._last_grad_cos_diff_depth: Optional[float] = None
        # Set by `calculate_loss` to the depth-applied loss tensor (pre-detach)
        # whenever the depth-consistency loss contributes this microbatch.
        # Read by the dual-backward block; reset to None each microbatch.
        self._dc_applied_for_grad = None
        self._last_identity_loss_applied: Optional[float] = None
        self._last_landmark_loss_applied: Optional[float] = None
        self._last_timestep: Optional[float] = None
        self._last_id_sim: Optional[float] = None
        self._last_id_sim_bins: Optional[dict] = None
        self._last_id_clean_target: Optional[float] = None
        self._last_id_clean_delta: Optional[float] = None
        self._last_pure_noise_cos: Optional[float] = None
        self._id_face_detector = None  # SCRFD detector for x0 quality gating
        self._last_shape_sim_bins: Optional[dict] = None

        # Bin dicts above hold a running-mean accumulator per t-band:
        #   {bin_key: {'sum': float, 'count': int}}.
        # See `_bin_update` for the per-sample writer and `_bin_finalize` for
        # the flush-time mean. Bins are reset at the start of every optimizer
        # step in `hook_train_loop` so that:
        #   - same-bin samples within a microbatch don't overwrite each other,
        #   - cross-microbatch (gradient_accumulation_steps>1) samples merge,
        #   - bins from a prior optimizer step never bleed into the next one.

        # MetricBuffer accumulates *every* metric scalar across all
        # microbatches in one optimizer step (fixes the
        # gradient_accumulation_steps > 1 overwrite bug, where every metric
        # except `loss` reflected only the final microbatch). We snapshot
        # every existing `self._last_*` value into the buffer at the end of
        # `train_single_accumulation`; `get_loss_metrics` then prefers the
        # buffer's cross-microbatch mean over the single-microbatch shim.
        # The buffer is *display-only* — it never touches the loss tensor.
        from extensions_built_in.sd_trainer.metric_buffer import MetricBuffer
        self._metric_buffer: MetricBuffer = MetricBuffer(per_sample_cap=16)
        # Mirror map: `_last_<attr>` → metric name in `loss_dict`. Keep in
        # sync with the loss_dict construction at the bottom of
        # `hook_train_loop` so the buffer's keys align with what users see.
        self._last_to_metric: dict = {
            '_last_face_token_norm': 'face_token_norm',
            '_last_vision_token_norm': 'vision_token_norm',
            '_last_body_token_norm': 'body_token_norm',
            '_last_identity_loss': 'identity_loss',
            '_last_landmark_loss': 'landmark_loss',
            '_last_pure_noise_cos': 'pure_noise_cos',
            '_last_diffusion_loss': 'diffusion_loss',
            '_last_diffusion_loss_applied': 'diffusion_loss_applied',
            '_last_identity_loss_applied': 'identity_loss_applied',
            '_last_landmark_loss_applied': 'landmark_loss_applied',
            '_last_body_proportion_loss': 'body_proportion_loss',
            '_last_body_proportion_loss_applied': 'body_proportion_loss_applied',
            '_last_body_shape_loss': 'body_shape_loss',
            '_last_body_shape_loss_applied': 'body_shape_loss_applied',
            '_last_body_shape_cos': 'body_shape_cos',
            '_last_body_shape_l1': 'body_shape_l1',
            '_last_body_shape_gated_pct': 'body_shape_gated_pct',
            '_last_normal_loss': 'normal_loss',
            '_last_normal_loss_applied': 'normal_loss_applied',
            '_last_normal_cos': 'normal_cos',
            '_last_vae_anchor_loss': 'vae_anchor_loss',
            '_last_vae_anchor_loss_applied': 'vae_anchor_loss_applied',
            '_last_depth_consistency_loss': 'depth_consistency_loss',
            '_last_depth_consistency_loss_applied': 'depth_consistency_loss_applied',
            '_last_depth_consistency_ssi': 'depth_consistency_ssi',
            '_last_depth_consistency_grad': 'depth_consistency_grad',
            '_last_grad_norm_diffusion': 'grad_norm_diffusion',
            '_last_grad_norm_depth': 'grad_norm_depth',
            '_last_grad_cos_diff_depth': 'grad_cos_diff_depth',
            '_last_timestep': 'timestep',
            '_last_id_sim': 'id_sim',
            '_last_id_clean_target': 'id_clean_target',
            '_last_id_clean_delta': 'id_clean_delta',
        }

        # E-LatentLPIPS perceptual loss
        self.latent_perceptual_model = None
        self._latent_perceptual_loss_accumulator: float = 0.0
        self._latent_perceptual_loss_applied_accumulator: float = 0.0
        self._latent_perceptual_accumulation_count: int = 0
        self._lp_preview_cache: Optional[dict] = None

        # Body shape conditioning (SMPL betas)
        body_id_raw = self.get_conf('body_id', None)
        if body_id_raw is not None:
            self.body_id_config = BodyIDConfig(**body_id_raw)
        else:
            self.body_id_config = None
        self.body_id_projector: Optional[BodyIDProjector] = None

        # store the loss target for a batch so we can use it in a loss
        self._guidance_loss_target_batch: float = 0.0
        if isinstance(self.train_config.guidance_loss_target, (int, float)):
            self._guidance_loss_target_batch = float(self.train_config.guidance_loss_target)
        elif isinstance(self.train_config.guidance_loss_target, list):
            self._guidance_loss_target_batch = float(self.train_config.guidance_loss_target[0])
        else:
            raise ValueError(f"Unknown guidance loss target type {type(self.train_config.guidance_loss_target)}")


    # -------------------------------------------------------------
    # Per-t-band running mean helpers (display-only metrics)
    #
    # Bin storage shape: {bin_key: {'sum': float, 'count': int}}.
    # `_bin_update` writes; `_bin_finalize` collapses to {key: mean}.
    # These ONLY touch metric scalars; they never participate in the
    # loss tensor or its gradient.
    # -------------------------------------------------------------
    @staticmethod
    def _bin_update(bins: dict, bin_key: str, value: float) -> None:
        slot = bins.get(bin_key)
        if slot is None:
            bins[bin_key] = {'sum': float(value), 'count': 1}
        else:
            slot['sum'] += float(value)
            slot['count'] += 1

    @staticmethod
    def _bin_finalize(bins: Optional[dict]) -> dict:
        if not bins:
            return {}
        out: dict = {}
        for k, slot in bins.items():
            cnt = slot.get('count', 0) if isinstance(slot, dict) else 0
            if cnt > 0:
                out[k] = slot['sum'] / cnt
        return out

    def _reset_step_bins(self) -> None:
        """Reset all per-t-band bin accumulators at the start of a fresh
        optimizer step. Called from `hook_train_loop` so that bins span a
        full optimizer step (all gradient-accumulation microbatches), then
        flush cleanly via `get_loss_metrics`."""
        self._last_id_sim_bins = None
        self._last_shape_sim_bins = None
        self._last_bp_sim_bins = None
        self._last_bsh_sim_bins = None
        self._last_depth_loss_bins = None
        self._last_diffusion_loss_bins = None

    def _record_sample(
        self,
        metric_name: str,
        value: float,
        t: Optional[float] = None,
        idx: Optional[int] = None,
        batch=None,
    ) -> None:
        """Record one sample's value into the per-sample breakdown buffer.

        ``idx`` + ``batch`` are used to derive the human-readable sample tag
        (basename of the source image / video). Both are optional —
        callers without batch metadata can pass just ``metric_name``,
        ``value`` and ``t``.

        Display-only; never participates in the loss tensor.
        """
        if not hasattr(self, '_metric_buffer'):
            return
        sample_tag = None
        if batch is not None and idx is not None:
            try:
                file_items = getattr(batch, 'file_items', None)
                if file_items is not None and idx < len(file_items):
                    fi = file_items[idx]
                    path = getattr(fi, 'path', None)
                    if path:
                        # Use parent_dir/basename so identical filenames in
                        # different dataset folders don't collapse into one
                        # synthetic series in the by_sample chart.
                        parent = os.path.basename(os.path.dirname(str(path)))
                        base = os.path.basename(str(path))
                        sample_tag = f'{parent}/{base}' if parent else base
                        # Append bucket dims so the same file processed at
                        # two resolutions (multi-bucket runs) stays distinct.
                        ch = getattr(fi, 'crop_height', None)
                        cw = getattr(fi, 'crop_width', None)
                        if isinstance(ch, int) and isinstance(cw, int) and ch > 0 and cw > 0:
                            sample_tag = f'{sample_tag}@{ch}x{cw}'
            except Exception:
                sample_tag = None
        self._metric_buffer.add_per_sample(
            metric_name, value, t=t, sample_tag=sample_tag,
        )

    def _iter_trainable_params(self):
        """Yield the flat list of trainable parameter tensors.

        ``self.params`` is either a flat list of tensors or a list of
        param-group dicts (each with a ``'params'`` list). Yields only
        tensors so callers can call ``.grad`` / ``autograd.grad`` directly.
        """
        params = getattr(self, 'params', None)
        if not params:
            return
        for entry in params:
            if isinstance(entry, dict):
                for p in entry.get('params', []):
                    yield p
            else:
                yield entry

    def _record_grad_cosine(self, trainable_params, dc_grads, pre_existing_grads) -> None:
        """Compute per-loss gradient norms + cosine and stash on `_last_*`.

        Computes the gradient contribution of THIS microbatch by
        subtracting any grad accumulated by earlier microbatches in the
        current optimizer step:
            g_full_this_mb = p.grad - pre_existing
            g_diff_this_mb = g_full_this_mb - g_dc_this_mb
        Display-only; never modifies ``p.grad``.
        """
        norm_dc_sq = 0.0
        norm_diff_sq = 0.0
        dot = 0.0
        if pre_existing_grads is None:
            pre_existing_grads = [None] * len(trainable_params)
        for p, g_dc, g_pre in zip(trainable_params, dc_grads, pre_existing_grads):
            g_now = p.grad
            # Per-microbatch full grad = current p.grad minus what was
            # already there from prior microbatches.
            if g_now is None:
                g_full = None
            elif g_pre is None:
                g_full = g_now.detach().float()
            else:
                g_full = g_now.detach().float() - g_pre.float()
            g_dc_f = g_dc.detach().float() if g_dc is not None else None

            if g_full is None and g_dc_f is None:
                continue
            if g_full is None:
                _ndc = float((g_dc_f * g_dc_f).sum())
                norm_dc_sq += _ndc
                norm_diff_sq += _ndc
                dot += -_ndc
                continue
            if g_dc_f is None:
                norm_diff_sq += float((g_full * g_full).sum())
                continue
            gdiff = g_full - g_dc_f
            norm_dc_sq += float((g_dc_f * g_dc_f).sum())
            norm_diff_sq += float((gdiff * gdiff).sum())
            dot += float((gdiff * g_dc_f).sum())

        norm_dc = norm_dc_sq ** 0.5
        norm_diff = norm_diff_sq ** 0.5
        denom = norm_dc * norm_diff
        cos = (dot / denom) if denom > 1e-12 else 0.0
        self._last_grad_norm_diffusion = norm_diff
        self._last_grad_norm_depth = norm_dc
        self._last_grad_cos_diff_depth = cos

    def _snapshot_metrics_to_buffer(self) -> None:
        """Mirror every freshly-written ``self._last_<attr>`` scalar into
        ``self._metric_buffer`` so cross-microbatch means are correct under
        ``gradient_accumulation_steps > 1``.

        Called once at the end of every ``train_single_accumulation`` call.
        Each mapped ``_last_<attr>`` is read, mirrored into the buffer with
        weight=1.0 (one observation per microbatch), and then **reset to
        None**. The reset prevents a stale value from microbatch N from
        being double-counted in microbatch N+1 when the metric's gating
        conditions don't fire.

        The trade-off: the existing flush block at the bottom of
        ``hook_train_loop`` will see ``None`` for these attrs and skip
        them; the buffer-flush merge a few lines later then writes the
        cross-microbatch mean back into ``loss_dict`` under the same key.

        Display-only; never touches loss tensors.
        """
        if not hasattr(self, '_metric_buffer'):
            return
        buf = self._metric_buffer
        for attr, metric_name in self._last_to_metric.items():
            val = getattr(self, attr, None)
            if val is None:
                continue
            buf.add_scalar(metric_name, val, weight=1.0)
            setattr(self, attr, None)

    def before_model_load(self):
        pass
    
    def cache_sample_prompts(self):
        if self.train_config.disable_sampling:
            return
        if self.sample_config is not None and self.sample_config.samples is not None and len(self.sample_config.samples) > 0:
            # cache all the samples
            self.sd.sample_prompts_cache = []
            sample_folder = os.path.join(self.save_root, 'samples')
            output_path = os.path.join(sample_folder, 'test.jpg')
            for i in range(len(self.sample_config.prompts)):
                sample_item = self.sample_config.samples[i]
                prompt = self.sample_config.prompts[i]

                # needed so we can autoparse the prompt to handle flags
                gen_img_config = GenerateImageConfig(
                    prompt=prompt, # it will autoparse the prompt
                    negative_prompt=sample_item.neg,
                    output_path=output_path,
                    ctrl_img=sample_item.ctrl_img,
                    ctrl_img_1=sample_item.ctrl_img_1,
                    ctrl_img_2=sample_item.ctrl_img_2,
                    ctrl_img_3=sample_item.ctrl_img_3,
                )
                
                has_control_images = False
                if gen_img_config.ctrl_img is not None or gen_img_config.ctrl_img_1 is not None or gen_img_config.ctrl_img_2 is not None or gen_img_config.ctrl_img_3 is not None:
                    has_control_images = True
                # see if we need to encode the control images
                if self.sd.encode_control_in_text_embeddings and has_control_images:
                    
                    ctrl_img_list = []
                    
                    if gen_img_config.ctrl_img is not None:
                        ctrl_img = Image.open(gen_img_config.ctrl_img).convert("RGB")
                        # convert to 0 to 1 tensor
                        ctrl_img = (
                            TF.to_tensor(ctrl_img)
                            .unsqueeze(0)
                            .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                        )
                        ctrl_img_list.append(ctrl_img)
                    
                    if gen_img_config.ctrl_img_1 is not None:
                        ctrl_img_1 = Image.open(gen_img_config.ctrl_img_1).convert("RGB")
                        # convert to 0 to 1 tensor
                        ctrl_img_1 = (
                            TF.to_tensor(ctrl_img_1)
                            .unsqueeze(0)
                            .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                        )
                        ctrl_img_list.append(ctrl_img_1)
                    if gen_img_config.ctrl_img_2 is not None:
                        ctrl_img_2 = Image.open(gen_img_config.ctrl_img_2).convert("RGB")
                        # convert to 0 to 1 tensor
                        ctrl_img_2 = (
                            TF.to_tensor(ctrl_img_2)
                            .unsqueeze(0)
                            .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                        )
                        ctrl_img_list.append(ctrl_img_2)
                    if gen_img_config.ctrl_img_3 is not None:
                        ctrl_img_3 = Image.open(gen_img_config.ctrl_img_3).convert("RGB")
                        # convert to 0 to 1 tensor
                        ctrl_img_3 = (
                            TF.to_tensor(ctrl_img_3)
                            .unsqueeze(0)
                            .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                        )
                        ctrl_img_list.append(ctrl_img_3)
                    
                    if self.sd.has_multiple_control_images:
                        ctrl_img = ctrl_img_list
                    else:
                        ctrl_img = ctrl_img_list[0] if len(ctrl_img_list) > 0 else None
                    
                    
                    positive = self.sd.encode_prompt(
                        gen_img_config.prompt,
                        control_images=ctrl_img
                    ).to('cpu')
                    negative = self.sd.encode_prompt(
                        gen_img_config.negative_prompt,
                        control_images=ctrl_img
                    ).to('cpu')
                else:
                    positive = self.sd.encode_prompt(gen_img_config.prompt).to('cpu')
                    negative = self.sd.encode_prompt(gen_img_config.negative_prompt).to('cpu')
                
                self.sd.sample_prompts_cache.append({
                    'conditional': positive,
                    'unconditional': negative
                })
        

    def before_dataset_load(self):
        # Auto-mirror depth conditioning onto reg datasets when the train
        # side uses it. Without this, reg samples train without depth input
        # while train samples have it — defeating the prior-preservation
        # role of reg under depth-conditioned LoRA. Skipped when the user
        # explicitly configured `controls` or `control_path` on the reg
        # dataset (their setting wins).
        if self.datasets is not None and self.datasets_reg is not None:
            train_has_depth = any(
                'depth' in (ds.controls or []) for ds in self.datasets
            )
            if train_has_depth:
                # Inherit the loss perceptor's model_id so reg conditioning
                # depth comes from the same model the user picked in the UI.
                _depth_model_id = None
                if self.depth_consistency_config is not None:
                    _depth_model_id = getattr(
                        self.depth_consistency_config, 'model_id', None,
                    )
                for reg_ds in self.datasets_reg:
                    if reg_ds.controls or reg_ds.control_path is not None:
                        continue
                    reg_ds.controls = ['depth']
                    if _depth_model_id and reg_ds.depth_model_id is None:
                        reg_ds.depth_model_id = _depth_model_id
                    print_acc(
                        f"  reg dataset '{reg_ds.folder_path}': "
                        f"auto-enabling controls=['depth'] (depth_model_id={reg_ds.depth_model_id or 'default'}) "
                        "to mirror train-side depth conditioning"
                    )

        self.assistant_adapter = None
        # get adapter assistant if one is set
        if self.train_config.adapter_assist_name_or_path is not None:
            adapter_path = self.train_config.adapter_assist_name_or_path

            if self.train_config.adapter_assist_type == "t2i":
                # dont name this adapter since we are not training it
                self.assistant_adapter = T2IAdapter.from_pretrained(
                    adapter_path, torch_dtype=get_torch_dtype(self.train_config.dtype)
                ).to(self.device_torch)
            elif self.train_config.adapter_assist_type == "control_net":
                self.assistant_adapter = ControlNetModel.from_pretrained(
                    adapter_path, torch_dtype=get_torch_dtype(self.train_config.dtype)
                ).to(self.device_torch, dtype=get_torch_dtype(self.train_config.dtype))
            else:
                raise ValueError(f"Unknown adapter assist type {self.train_config.adapter_assist_type}")

            self.assistant_adapter.eval()
            self.assistant_adapter.requires_grad_(False)
            flush()
        if self.train_config.train_turbo and self.train_config.show_turbo_outputs:
            if self.model_config.is_xl:
                self.taesd = AutoencoderTiny.from_pretrained("madebyollin/taesdxl",
                                                             torch_dtype=get_torch_dtype(self.train_config.dtype))
            else:
                self.taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd",
                                                             torch_dtype=get_torch_dtype(self.train_config.dtype))
            self.taesd.to(dtype=get_torch_dtype(self.train_config.dtype), device=self.device_torch)
            self.taesd.eval()
            self.taesd.requires_grad_(False)

    def hook_add_extra_train_params(self, params):
        params = super().hook_add_extra_train_params(params)
        if self.face_id_config is not None and self.face_id_config.enabled:
            hidden_size = self.sd.unet.hidden_size if hasattr(self.sd.unet, 'hidden_size') else 4096
            self.face_id_projector = FaceIDProjector(
                id_dim=512,
                hidden_size=hidden_size,
                num_tokens=self.face_id_config.num_tokens,
            ).to(self.device_torch)
            # Set initial output_scale from config
            with torch.no_grad():
                self.face_id_projector.output_scale.fill_(self.face_id_config.init_scale)
            # Separate param groups: output_scale gets higher LR so it can
            # ramp up from near-zero without being bottlenecked
            scale_mult = self.face_id_config.scale_lr_multiplier
            proj_params = [p for n, p in self.face_id_projector.named_parameters() if n != 'output_scale']
            scale_params = [self.face_id_projector.output_scale]
            params.append({
                'params': proj_params,
                'lr': self.train_config.lr,
            })
            params.append({
                'params': scale_params,
                'lr': self.train_config.lr * scale_mult,
            })
            total_params = sum(p.numel() for p in self.face_id_projector.parameters())
            print_acc(f"LoRA+ID: FaceIDProjector added ({total_params} params, output_scale LR={self.train_config.lr * scale_mult:.1e})")

            # Vision face projector (CLIP/DINOv2 spatial tokens → face tokens)
            if self.face_id_config.vision_enabled:
                # Detect hidden size from the vision model config
                from transformers import AutoConfig
                vision_config = AutoConfig.from_pretrained(self.face_id_config.vision_model)
                # CLIP returns a parent CLIPConfig — get the vision sub-config
                if hasattr(vision_config, 'vision_config'):
                    vision_dim = vision_config.vision_config.hidden_size
                else:
                    vision_dim = vision_config.hidden_size

                self.vision_face_projector = VisionFaceProjector(
                    vision_dim=vision_dim,
                    hidden_size=hidden_size,
                    num_tokens=self.face_id_config.vision_num_tokens,
                ).to(self.device_torch)
                vp_params = [p for n, p in self.vision_face_projector.named_parameters() if n != 'output_scale']
                vp_scale = [self.vision_face_projector.output_scale]
                params.append({
                    'params': vp_params,
                    'lr': self.train_config.lr,
                })
                params.append({
                    'params': vp_scale,
                    'lr': self.train_config.lr * scale_mult,
                })
                vp_total = sum(p.numel() for p in self.vision_face_projector.parameters())
                print_acc(f"LoRA+ID: VisionFaceProjector added ({vp_total} params, vision_dim={vision_dim}, tokens={self.face_id_config.vision_num_tokens})")

        # Body shape projector (SMPL betas → body tokens)
        if self.body_id_config is not None and self.body_id_config.enabled:
            hidden_size_body = self.sd.unet.hidden_size if hasattr(self.sd.unet, 'hidden_size') else 4096
            self.body_id_projector = BodyIDProjector(
                id_dim=10,
                hidden_size=hidden_size_body,
                num_tokens=self.body_id_config.num_tokens,
            ).to(self.device_torch)
            with torch.no_grad():
                self.body_id_projector.output_scale.fill_(self.body_id_config.init_scale)
            body_scale_mult = self.body_id_config.scale_lr_multiplier
            bp_params = [p for n, p in self.body_id_projector.named_parameters() if n != 'output_scale']
            bp_scale = [self.body_id_projector.output_scale]
            params.append({
                'params': bp_params,
                'lr': self.train_config.lr,
            })
            params.append({
                'params': bp_scale,
                'lr': self.train_config.lr * body_scale_mult,
            })
            bp_total = sum(p.numel() for p in self.body_id_projector.parameters())
            print_acc(f"LoRA+ID: BodyIDProjector added ({bp_total} params, tokens={self.body_id_config.num_tokens}, scale_lr={body_scale_mult}x)")

        return params

    def _get_identity_state_dict(self) -> Optional[OrderedDict]:
        """Collect projector weights and averaged identity embeddings for saving in LoRA."""
        extra = OrderedDict()

        # Projector weights (prefixed)
        if self.face_id_projector is not None:
            for k, v in self.face_id_projector.state_dict().items():
                extra[f'face_id_projector.{k}'] = v
        if self.vision_face_projector is not None:
            for k, v in self.vision_face_projector.state_dict().items():
                extra[f'vision_face_projector.{k}'] = v
        if self.body_id_projector is not None:
            for k, v in self.body_id_projector.state_dict().items():
                extra[f'body_id_projector.{k}'] = v

        # Averaged identity embeddings from training data
        if getattr(self, '_avg_face_embedding', None) is not None:
            extra['identity.face_embedding'] = self._avg_face_embedding
        if getattr(self, '_avg_vision_face_embedding', None) is not None:
            extra['identity.vision_face_embedding'] = self._avg_vision_face_embedding
        if getattr(self, '_avg_body_embedding', None) is not None:
            extra['identity.body_embedding'] = self._avg_body_embedding

        return extra if extra else None

    def hook_before_train_loop(self):
        super().hook_before_train_loop()

        # Check if face_suppression_weight is active globally or on any dataset
        _any_face_suppression = False
        if self.face_id_config is not None:
            _global_fsw = getattr(self.face_id_config, 'face_suppression_weight', None)
            if _global_fsw is not None and _global_fsw > 0.0:
                _any_face_suppression = True
        if not _any_face_suppression:
            for dl in [self.data_loader, self.data_loader_reg]:
                if dl is not None:
                    for ds in get_dataloader_datasets(dl):
                        fsw = getattr(ds.dataset_config, 'face_suppression_weight', None)
                        if fsw is not None and fsw > 0.0:
                            _any_face_suppression = True
                            break

        # Scan all datasets for per-dataset loss weight overrides that might enable
        # models/caching even when the global weight is 0
        def _any_dataset_overrides(field_name):
            for dl in [self.data_loader, self.data_loader_reg]:
                if dl is not None:
                    for ds in get_dataloader_datasets(dl):
                        val = getattr(ds.dataset_config, field_name, None)
                        if val is not None and val > 0:
                            return True
            return False

        _ds_identity = _any_dataset_overrides('identity_loss_weight')
        _ds_landmark = _any_dataset_overrides('landmark_loss_weight')
        _ds_body_prop = _any_dataset_overrides('body_proportion_loss_weight')
        _ds_body_shape = _any_dataset_overrides('body_shape_loss_weight')
        _ds_normal = _any_dataset_overrides('normal_loss_weight')
        _ds_vae_anchor = _any_dataset_overrides('vae_anchor_loss_weight')
        _ds_depth = _any_dataset_overrides('depth_loss_weight')
        _vae_anchor_enabled = (self.face_id_config is not None
                               and (self.face_id_config.vae_anchor_loss_weight > 0 or _ds_vae_anchor))

        # LoRA+ID: cache face embeddings for all datasets
        # Run if face conditioning is enabled OR identity loss is enabled OR landmark loss is enabled OR face suppression is active
        _any_face = self.face_id_config is not None and (self.face_id_config.enabled or self.face_id_config.identity_loss_weight > 0 or self.face_id_config.landmark_loss_weight > 0 or self.face_id_config.body_proportion_loss_weight > 0 or self.face_id_config.body_shape_loss_weight > 0 or self.face_id_config.normal_loss_weight > 0 or _vae_anchor_enabled or self.face_id_config.identity_metrics or _ds_identity or _ds_landmark or _ds_body_prop or _ds_body_shape or _ds_normal)
        _any_face = _any_face or _any_face_suppression
        if _any_face:
            # If face_suppression_weight needs bboxes but no face_id config exists, create a minimal one
            _face_cache_config = self.face_id_config
            if _face_cache_config is None and _any_face_suppression:
                _face_cache_config = FaceIDConfig()
            print_acc("LoRA+ID: Extracting and caching face embeddings...")
            if self.data_loader is not None:
                datasets = get_dataloader_datasets(self.data_loader)
                for dataset in datasets:
                    cache_face_embeddings(dataset.file_list, _face_cache_config)
            if self.data_loader_reg is not None:
                datasets = get_dataloader_datasets(self.data_loader_reg)
                for dataset in datasets:
                    cache_face_embeddings(dataset.file_list, _face_cache_config)

        # Compute per-dataset average identity embeddings
        self._identity_mean_embed = None  # (512,) ArcFace bias direction, set after model load

        self._identity_avg_embeds = {}  # dataset key -> (512,) unit-normalized average
        self._identity_original_embeds = {}  # dataset key -> [(file_item, original_embed)] for clean_cos computation
        _need_avg = (_any_face and self.face_id_config is not None and
                     (self.face_id_config.identity_loss_use_average or self.face_id_config.identity_loss_average_blend > 0))
        if _need_avg:
            for dl in [self.data_loader, self.data_loader_reg]:
                if dl is None:
                    continue
                for dataset in get_dataloader_datasets(dl):
                    valid_embeds = []
                    for fi in dataset.file_list:
                        emb = getattr(fi, 'identity_embedding', None)
                        if emb is not None and emb.abs().sum() > 0:
                            valid_embeds.append(emb)
                    if valid_embeds:
                        avg_embed = torch.stack(valid_embeds).mean(dim=0)
                        avg_embed = avg_embed / (avg_embed.norm() + 1e-8)
                        key = dataset.dataset_config.folder_path or id(dataset)
                        self._identity_avg_embeds[key] = avg_embed
                        cos_spread = torch.stack(valid_embeds).std(dim=0).mean()
                        if self.face_id_config.identity_loss_use_average:
                            # Store originals before replacement (for clean_cos targets)
                            original_pairs = []
                            for fi in dataset.file_list:
                                emb = getattr(fi, 'identity_embedding', None)
                                if emb is not None and emb.abs().sum() > 0:
                                    original_pairs.append((fi, emb.clone()))
                            self._identity_original_embeds[key] = original_pairs
                            # Pure average: replace per-image embeddings (only where face was detected)
                            for fi in dataset.file_list:
                                emb = getattr(fi, 'identity_embedding', None)
                                if emb is not None and emb.abs().sum() > 0:
                                    fi.identity_embedding = avg_embed.clone()
                            print_acc(f"  identity_loss_use_average: replaced {len(valid_embeds)} embeddings with dataset mean (cos_spread={cos_spread:.4f})")
                        else:
                            print_acc(f"  identity_loss_average_blend={self.face_id_config.identity_loss_average_blend}: {len(valid_embeds)} embeddings, cos_spread={cos_spread:.4f}")

        # Build per-dataset embedding pools for random / multi-ref modes
        self._identity_embed_pools = {}  # dataset folder_path -> stacked (N, 512) tensor
        _need_pools = (self.face_id_config is not None and
                       (self.face_id_config.identity_loss_use_random or self.face_id_config.identity_loss_num_refs > 0))
        if _any_face and _need_pools:
            for dl in [self.data_loader, self.data_loader_reg]:
                if dl is None:
                    continue
                for dataset in get_dataloader_datasets(dl):
                    valid_embeds = []
                    for fi in dataset.file_list:
                        emb = getattr(fi, 'identity_embedding', None)
                        if emb is not None and emb.abs().sum() > 0:
                            valid_embeds.append(emb)
                    if valid_embeds:
                        pool = torch.stack(valid_embeds)  # (N, 512)
                        key = dataset.dataset_config.folder_path or id(dataset)
                        self._identity_embed_pools[key] = pool
                        print_acc(f"  identity embedding pool: {len(valid_embeds)} embeddings for {key}")

        # Body proportions: cache ViTPose bone-length ratios for all datasets
        if self.face_id_config is not None and (self.face_id_config.body_proportion_loss_weight > 0 or _ds_body_prop):
            print_acc("LoRA+ID: Extracting and caching body proportion embeddings...")
            if self.data_loader is not None:
                datasets = get_dataloader_datasets(self.data_loader)
                for dataset in datasets:
                    cache_body_proportion_embeddings(dataset.file_list, self.face_id_config)
            if self.data_loader_reg is not None:
                datasets = get_dataloader_datasets(self.data_loader_reg)
                for dataset in datasets:
                    cache_body_proportion_embeddings(dataset.file_list, self.face_id_config)

        # Body shape (HybrIK): cache SMPL betas for body shape loss
        if self.face_id_config is not None and (self.face_id_config.body_shape_loss_weight > 0 or _ds_body_shape):
            print_acc("LoRA+ID: Extracting and caching body shape embeddings (HybrIK)...")
            if self.data_loader is not None:
                datasets = get_dataloader_datasets(self.data_loader)
                for dataset in datasets:
                    cache_body_shape_embeddings(dataset.file_list, self.face_id_config)
            if self.data_loader_reg is not None:
                datasets = get_dataloader_datasets(self.data_loader_reg)
                for dataset in datasets:
                    cache_body_shape_embeddings(dataset.file_list, self.face_id_config)

        # Normal maps (Sapiens): cache surface normals for normal loss
        if self.face_id_config is not None and (self.face_id_config.normal_loss_weight > 0 or _ds_normal):
            print_acc("LoRA+ID: Extracting and caching normal maps (Sapiens)...")
            if self.data_loader is not None:
                datasets = get_dataloader_datasets(self.data_loader)
                for dataset in datasets:
                    cache_normal_embeddings(dataset.file_list, self.face_id_config)
            if self.data_loader_reg is not None:
                datasets = get_dataloader_datasets(self.data_loader_reg)
                for dataset in datasets:
                    cache_normal_embeddings(dataset.file_list, self.face_id_config)

        # NOTE: depth consistency GT caching is deferred until after the TAEF2
        # decoder is loaded (further below) so the cache can run pixels through
        # the same encode + decode chain training will use, giving the loss a
        # true zero-floor target.

        # Auto-masking (YOLO + SAM 2 + SegFormer-clothes): cache person/body/clothing
        # masks for region-aware loss weighting. Debug preview tiles (when enabled)
        # are written to save_root/subject_mask_previews/ — NOT inside the dataset
        # folder — so they can't accidentally be picked up as training images.
        if self.subject_mask_config is not None and self.subject_mask_config.enabled:
            print_acc("Auto-masking: Extracting and caching subject masks...")
            _sm_preview_dir = os.path.join(self.save_root, 'subject_mask_previews')
            if self.data_loader is not None:
                datasets = get_dataloader_datasets(self.data_loader)
                for dataset in datasets:
                    cache_subject_masks(
                        dataset.file_list, self.subject_mask_config,
                        preview_dir=_sm_preview_dir,
                    )
            if self.data_loader_reg is not None:
                datasets = get_dataloader_datasets(self.data_loader_reg)
                for dataset in datasets:
                    cache_subject_masks(
                        dataset.file_list, self.subject_mask_config,
                        preview_dir=_sm_preview_dir,
                    )

        # VAE anchor features: cache multi-scale VAE encoder features for perceptual anchor loss
        if _vae_anchor_enabled:
            print_acc("LoRA+ID: Extracting and caching VAE anchor features...")
            if self.data_loader is not None:
                datasets = get_dataloader_datasets(self.data_loader)
                for dataset in datasets:
                    cache_vae_anchor_features(dataset.file_list, self.face_id_config)
            if self.data_loader_reg is not None:
                datasets = get_dataloader_datasets(self.data_loader_reg)
                for dataset in datasets:
                    cache_vae_anchor_features(dataset.file_list, self.face_id_config)

        # Body shape (conditioning): cache SMPL betas for all datasets
        if self.body_id_config is not None and self.body_id_config.enabled:
            print_acc("LoRA+ID: Extracting and caching body shape embeddings...")
            if self.data_loader is not None:
                datasets = get_dataloader_datasets(self.data_loader)
                for dataset in datasets:
                    cache_body_embeddings(dataset.file_list, self.body_id_config)
            if self.data_loader_reg is not None:
                datasets = get_dataloader_datasets(self.data_loader_reg)
                for dataset in datasets:
                    cache_body_embeddings(dataset.file_list, self.body_id_config)

        # Compute averaged identity embeddings for saving in LoRA
        self._avg_face_embedding = None
        self._avg_vision_face_embedding = None
        self._avg_body_embedding = None
        if self.data_loader is not None:
            all_datasets = get_dataloader_datasets(self.data_loader)
            all_files = [fi for ds in all_datasets for fi in ds.file_list]

            face_embs = [fi.face_embedding for fi in all_files if getattr(fi, 'face_embedding', None) is not None]
            if face_embs:
                # Filter out zero vectors (no face detected)
                nonzero = [e for e in face_embs if e.abs().sum() > 0]
                if nonzero:
                    self._avg_face_embedding = F.normalize(torch.stack(nonzero).mean(dim=0), dim=-1)

            vision_embs = [fi.vision_face_embedding for fi in all_files if getattr(fi, 'vision_face_embedding', None) is not None]
            if vision_embs:
                nonzero = [e for e in vision_embs if e.abs().sum() > 0]
                if nonzero:
                    self._avg_vision_face_embedding = torch.stack(nonzero).mean(dim=0)

            body_embs = [fi.body_embedding for fi in all_files if getattr(fi, 'body_embedding', None) is not None]
            if body_embs:
                nonzero = [e for e in body_embs if e.abs().sum() > 0]
                if nonzero:
                    self._avg_body_embedding = torch.stack(nonzero).mean(dim=0)

        # Determine if any loss needs a latent decoder (TAESD / TAEF2).
        # Depth-consistency also decodes x0 to pixels, so it shares the same
        # decoder path — critical for Flux-2 where the base VAE has no
        # diffusers-style `.config` / `.scaling_factor` attributes.
        _need_face_decoder = (
            self.face_id_config is not None
            and (self.face_id_config.identity_loss_weight > 0
                 or self.face_id_config.landmark_loss_weight > 0
                 or self.face_id_config.body_proportion_loss_weight > 0
                 or self.face_id_config.body_shape_loss_weight > 0
                 or self.face_id_config.normal_loss_weight > 0
                 or _vae_anchor_enabled
                 or self.face_id_config.identity_metrics
                 or _ds_identity or _ds_landmark or _ds_body_prop or _ds_body_shape or _ds_normal)
        ) or (
            self.depth_consistency_config is not None
            and (self.depth_consistency_config.loss_weight > 0 or _ds_depth)
        )

        # Load identity loss model (ArcFace + TAESD) if enabled
        # Works with or without face token conditioning (face_id.enabled)
        if (self.face_id_config is not None
                and (self.face_id_config.identity_loss_weight > 0 or self.face_id_config.identity_metrics or _ds_identity)):
            print_acc("LoRA+ID: Loading identity loss model (ArcFace)...")
            self.id_loss_model = DifferentiableFaceEncoder()
            self.id_loss_model.to(self.device_torch)

            # Compute ArcFace bias direction for cos_sim correction.
            # ArcFace maps all non-face inputs to a tight cluster (~0.5 cos_sim
            # vs faces). Subtracting this bias direction makes non-face inputs
            # score ~0 while preserving face identity discrimination.
            print_acc("  Computing ArcFace bias direction from noise...")
            with torch.no_grad():
                noise_embeds = []
                for _ in range(200):
                    noise_img = torch.randn(1, 3, 112, 112, device=self.device_torch) * 0.3 + 0.5
                    noise_img = noise_img.clamp(0, 1)
                    emb = self.id_loss_model(noise_img)
                    noise_embeds.append(emb)
                noise_embeds = torch.cat(noise_embeds, dim=0)
                self._identity_mean_embed = noise_embeds.mean(dim=0).cpu()
            print_acc(f"  ArcFace bias correction: noise mean norm={self._identity_mean_embed.norm():.4f}")

            # Load lightweight face detector to gate identity loss on x0 predictions.
            # Blobs that ArcFace scores ~0.4 against faces should be filtered out
            # BEFORE cosine similarity — if no face is detected in the crop, skip it.
            print_acc("  Loading SCRFD face detector for x0 quality gate...")
            from insightface.app import FaceAnalysis
            self._id_face_detector = FaceAnalysis(
                name='buffalo_l', allowed_modules=['detection'],
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self._id_face_detector.prepare(ctx_id=0, det_size=(160, 160))
            print_acc("  SCRFD face detector loaded")

            # Compute per-image clean_cos targets for average mode.
            # Each image's clean ArcFace similarity to the dataset average becomes
            # its loss target — a profile shot scoring 0.7 won't be pushed beyond 0.7.
            if self._identity_original_embeds:
                mean_emb = self._identity_mean_embed
                for key, pairs in self._identity_original_embeds.items():
                    avg = self._identity_avg_embeds[key]
                    avg_c = F.normalize(avg - mean_emb, p=2, dim=-1)
                    clean_vals = []
                    for fi, orig_emb in pairs:
                        orig_c = F.normalize(orig_emb - mean_emb, p=2, dim=-1)
                        clean_cos = F.cosine_similarity(orig_c.unsqueeze(0), avg_c.unsqueeze(0), dim=-1).item()
                        fi.identity_clean_cos = max(clean_cos, 0.1)  # clamp floor to prevent div instability
                        clean_vals.append(fi.identity_clean_cos)
                    import numpy as _np
                    print_acc(f"  Clean cos targets for {key}: "
                              f"min={min(clean_vals):.3f} max={max(clean_vals):.3f} "
                              f"mean={_np.mean(clean_vals):.3f} std={_np.std(clean_vals):.3f}")
                del self._identity_original_embeds  # free memory

        # Load landmark loss model (MediaPipe FaceMesh) if enabled
        if (self.face_id_config is not None
                and (self.face_id_config.landmark_loss_weight > 0 or _ds_landmark)):
            print_acc("LoRA+ID: Loading landmark loss model (MediaPipe FaceMesh)...")
            self.landmark_loss_model = DifferentiableLandmarkEncoder()
            self.landmark_loss_model.to(self.device_torch)

        # Load body proportion loss model (ViTPose) if enabled
        if (self.face_id_config is not None
                and (self.face_id_config.body_proportion_loss_weight > 0 or _ds_body_prop)):
            print_acc("LoRA+ID: Loading body proportion loss model (ViTPose)...")
            self.body_proportion_model = DifferentiableBodyProportionEncoder()
            self.body_proportion_model.to(self.device_torch)

        # Load body shape loss model (HybrIK) if enabled
        if (self.face_id_config is not None
                and (self.face_id_config.body_shape_loss_weight > 0 or _ds_body_shape)):
            print_acc("LoRA+ID: Loading body shape loss model (HybrIK)...")
            self.body_shape_model = DifferentiableBodyShapeEncoder()
            self.body_shape_model.to(self.device_torch)

        # Load normal loss model (Sapiens) if enabled
        if (self.face_id_config is not None
                and (self.face_id_config.normal_loss_weight > 0 or _ds_normal)):
            print_acc("LoRA+ID: Loading normal loss model (Sapiens 0.3B)...")
            self.normal_model = DifferentiableNormalEncoder()
            self.normal_model.to(self.device_torch)

        # Load Depth-Anything-V2 perceptor for depth consistency loss if enabled
        if (self.depth_consistency_config is not None
                and (self.depth_consistency_config.loss_weight > 0 or _ds_depth)):
            print_acc("DepthConsistency: Loading Depth-Anything-V2 perceptor...")
            self.depth_encoder = DifferentiableDepthEncoder(
                model_id=self.depth_consistency_config.model_id,
                input_size=self.depth_consistency_config.input_size,
                grad_checkpoint=self.depth_consistency_config.grad_checkpoint,
                device=self.device_torch,
            )

        # Load E-LatentLPIPS model if latent perceptual loss is enabled
        if self.train_config.latent_perceptual_loss_weight > 0:
            from elatentlpips import ELatentLPIPS
            encoder = self.train_config.latent_perceptual_encoder
            # Auto-detect encoder from model architecture if set to 'auto'
            if encoder == 'auto':
                arch = self.sd.model_config.arch if hasattr(self.sd, 'model_config') else None
                if arch in ('sdxl', 'sd1', 'sd2', 'ssd'):
                    encoder = 'sdxl' if arch == 'sdxl' else 'sd15'
                elif arch in ('sd3',):
                    encoder = 'sd3'
                elif arch in ('flux', 'flex1', 'flex2'):
                    encoder = 'flux'
                else:
                    encoder = 'sdxl'  # safe default (4ch)
                print_acc(f"  Auto-detected E-LatentLPIPS encoder: {encoder} (arch={arch})")
            print_acc(f"Loading E-LatentLPIPS model (encoder={encoder}) for latent perceptual loss")
            self.latent_perceptual_model = ELatentLPIPS(
                pretrained=True, encoder=encoder, verbose=True, augment=None,
            )
            self.latent_perceptual_model.eval()
            self.latent_perceptual_model.requires_grad_(False)
            self.latent_perceptual_model.to(self.device_torch)
            num_params = sum(p.numel() for p in self.latent_perceptual_model.parameters())
            print_acc(f"E-LatentLPIPS loaded: {num_params:,} parameters on {self.device_torch}")

        # Load VAE anchor encoder if enabled
        if _vae_anchor_enabled:
            self.vae_anchor_encoder = VAEAnchorEncoder(
                vae_path=self.face_id_config.vae_anchor_model_path
            )
            self.vae_anchor_encoder.load(
                device=self.device_torch, dtype=torch.float32
            )
            print_acc("  VAE anchor encoder loaded")

            # Enable fine-grained gradient checkpointing on SDXL VAE decoder.
            # The diffusers built-in only checkpoints per UpDecoderBlock (4 segments).
            # We patch to checkpoint each ResNet + upsampler individually (~15 segments),
            # reducing decode activations from ~6.7GB to ~2.8GB at 768x768.
            if self.sd.vae is not None and hasattr(self.sd.vae, 'decoder'):
                _decoder = self.sd.vae.decoder
                def _fine_ckpt_forward(sample, latent_embeds=None):
                    from torch.utils.checkpoint import checkpoint as ckpt
                    sample = _decoder.conv_in(sample)
                    upscale_dtype = next(iter(_decoder.up_blocks.parameters())).dtype
                    sample = ckpt(_decoder.mid_block, sample, latent_embeds, use_reentrant=False)
                    sample = sample.to(upscale_dtype)
                    for up_block in _decoder.up_blocks:
                        for resnet in up_block.resnets:
                            sample = ckpt(resnet, sample, None, use_reentrant=False)
                        if up_block.upsamplers is not None:
                            for upsampler in up_block.upsamplers:
                                sample = ckpt(upsampler, sample, use_reentrant=False)
                    if latent_embeds is None:
                        sample = _decoder.conv_norm_out(sample)
                    else:
                        sample = _decoder.conv_norm_out(sample, latent_embeds)
                    sample = _decoder.conv_act(sample)
                    sample = _decoder.conv_out(sample)
                    return sample

                _decoder.forward = _fine_ckpt_forward
                print_acc("  VAE decoder: fine-grained gradient checkpointing enabled")

        # Load lightweight decoder for face losses (identity)
        if _need_face_decoder and self.taesd is None:
            if hasattr(self.sd.vae, 'config') and self.sd.vae.config is not None:
                vae_channels = self.sd.vae.config.get('latent_channels', 4)
            elif hasattr(self.sd.vae, 'params'):
                vae_channels = getattr(self.sd.vae.params, 'z_channels', 4)
            else:
                vae_channels = 4
            if vae_channels == 32:
                # Flux 2 VAE (32-ch latents packed to 128-ch) — use TAEF2
                from toolkit.taesd import Decoder as TAESDDecoder
                from huggingface_hub import hf_hub_download
                from safetensors.torch import load_file as load_sf
                print_acc("  Loading TAEF2 decoder for face losses...")
                weights_path = hf_hub_download("madebyollin/taef2", "taef2.safetensors")
                sd = load_sf(weights_path)
                # extract decoder weights: remap "decoder.layers.N" -> "N+1" (Clamp at idx 0 has no params)
                decoder_sd = {}
                for k, v in sd.items():
                    if k.startswith("decoder.layers."):
                        rest = k.replace("decoder.layers.", "")
                        parts = rest.split(".", 1)
                        idx = int(parts[0]) + 1
                        remainder = "." + parts[1] if len(parts) > 1 else ""
                        decoder_sd[str(idx) + remainder] = v
                decoder = TAESDDecoder(latent_channels=32, use_midblock_gn=True)
                decoder.load_state_dict(decoder_sd)
                decoder.to(dtype=get_torch_dtype(self.train_config.dtype), device=self.device_torch)
                decoder.eval()
                decoder.requires_grad_(False)
                self._taef2_decoder = decoder
            elif self.sd.is_flux or 'flex' in getattr(self.sd, 'arch', ''):
                taesd_name = "madebyollin/taef1"
                print_acc(f"  Loading TAESD ({taesd_name}) for face losses...")
                self.taesd = AutoencoderTiny.from_pretrained(
                    taesd_name, torch_dtype=get_torch_dtype(self.train_config.dtype))
                self.taesd.to(dtype=get_torch_dtype(self.train_config.dtype), device=self.device_torch)
                self.taesd.eval()
                self.taesd.requires_grad_(False)
            elif self.model_config.is_xl:
                print_acc("  Loading TAESD (madebyollin/taesdxl) for face losses...")
                self.taesd = AutoencoderTiny.from_pretrained(
                    "madebyollin/taesdxl", torch_dtype=get_torch_dtype(self.train_config.dtype))
                self.taesd.to(dtype=get_torch_dtype(self.train_config.dtype), device=self.device_torch)
                self.taesd.eval()
                self.taesd.requires_grad_(False)
            else:
                print_acc("  Loading TAESD (madebyollin/taesd) for face losses...")
                self.taesd = AutoencoderTiny.from_pretrained(
                    "madebyollin/taesd", torch_dtype=get_torch_dtype(self.train_config.dtype))
                self.taesd.to(dtype=get_torch_dtype(self.train_config.dtype), device=self.device_torch)
                self.taesd.eval()
                self.taesd.requires_grad_(False)

        # Depth consistency: cache GT depth maps (DA2 output) for all datasets.
        # v3 caches GT from VAE-encode → trainer-decoder pixels so the live
        # depth loss has a true zero-floor target (the cleanest the trainer
        # can produce). Mirrors SDTrainer.py:2865-2890 decode dispatch.
        if (self.depth_consistency_config is not None
                and (self.depth_consistency_config.loss_weight > 0 or _ds_depth)):
            print_acc("DepthConsistency: Extracting and caching GT depth maps...")

            # Pull VAE scaling once; all VAEs we hit (Flux 2, SD, SDXL) expose these.
            _vae_cfg = getattr(self.sd.vae, 'config', {}) or {}
            _vae_scale = float(_vae_cfg.get('scaling_factor', 1.0)) if hasattr(_vae_cfg, 'get') else 1.0
            _vae_shift = float(_vae_cfg.get('shift_factor', 0.0) or 0.0) if hasattr(_vae_cfg, 'get') else 0.0
            _vae_dtype = self.sd.vae.dtype

            def _vae_roundtrip_for_depth(arr: torch.Tensor) -> torch.Tensor:
                """[0,1] pixels → VAE encode → trainer decoder → [0,1] pixels."""
                if next(self.sd.vae.parameters()).device != self.device_torch:
                    self.sd.vae.to(self.device_torch)
                arr_norm = (arr * 2.0 - 1.0).to(_vae_dtype)
                posterior = self.sd.vae.encode(arr_norm)
                # Flux 2's VAE.encode returns a Tensor directly; standard
                # diffusers VAEs return an object with `.latent_dist`. Cover both.
                if hasattr(posterior, 'latent_dist'):
                    raw_latent = posterior.latent_dist.mode()  # deterministic, batched
                elif isinstance(posterior, torch.Tensor):
                    raw_latent = posterior
                elif hasattr(posterior, 'sample') and callable(getattr(posterior, 'sample')):
                    raw_latent = posterior.sample()
                else:
                    raw_latent = posterior[0]
                # Match dataloader's encode_images: scaling_factor * (latent - shift_factor)
                scaled = _vae_scale * (raw_latent - _vae_shift)
                if getattr(self, '_taef2_decoder', None) is not None:
                    # TAEF2 expects 32-ch unpacked latents; Flux 2 VAE encoder
                    # outputs 128-ch packed (same as model output). Mirror the
                    # rearrange used in the depth loss decode path.
                    if scaled.shape[1] != 32:
                        scaled = rearrange(
                            scaled,
                            "b (c p1 p2) h w -> b c (h p1) (w p2)",
                            c=32, p1=2, p2=2,
                        )
                    dec_dtype = next(self._taef2_decoder.parameters()).dtype
                    pixels = self._taef2_decoder(scaled.to(dec_dtype)).float()
                elif self.taesd is not None:
                    td = next(self.taesd.parameters()).dtype
                    pixels = self.taesd.decode(scaled.to(td)).sample.float()
                    pixels = (pixels + 1.0) * 0.5
                else:
                    unscaled = scaled / _vae_scale
                    if _vae_shift:
                        unscaled = unscaled + _vae_shift
                    pixels = self.sd.vae.decode(unscaled.to(_vae_dtype)).sample.float()
                    pixels = (pixels + 1.0) * 0.5
                return pixels.clamp(0, 1)

            def _cache_dataset_depth(ds):
                cache_depth_gt_embeddings(
                    ds.file_list, self.depth_consistency_config,
                    device=self.device_torch,
                    vae_roundtrip_fn=_vae_roundtrip_for_depth,
                )
                if getattr(ds, 'num_frames', 1) > 1:
                    cache_video_depth_gt_embeddings(
                        ds.file_list, self.depth_consistency_config,
                        device=self.device_torch,
                        num_frames=ds.num_frames,
                    )

            if self.data_loader is not None:
                for dataset in get_dataloader_datasets(self.data_loader):
                    _cache_dataset_depth(dataset)
            if self.data_loader_reg is not None:
                for dataset in get_dataloader_datasets(self.data_loader_reg):
                    _cache_dataset_depth(dataset)

        if self.is_caching_text_embeddings:
            # make sure model is on cpu for this part so we don't oom.
            self.sd.unet.to('cpu')

        # cache unconditional embeds (blank prompt)
        with torch.no_grad():
            kwargs = {}
            if self.sd.encode_control_in_text_embeddings:
                # just do a blank image for unconditionals
                control_image = torch.zeros((1, 3, 224, 224), device=self.sd.device_torch, dtype=self.sd.torch_dtype)
                if self.sd.has_multiple_control_images:
                    control_image = [control_image]
                
                kwargs['control_images'] = control_image
            self.unconditional_embeds = self.sd.encode_prompt(
                [self.train_config.unconditional_prompt],
                long_prompts=self.do_long_prompts,
                **kwargs
            ).to(
                self.device_torch,
                dtype=self.sd.torch_dtype
            ).detach()
        
        if self.train_config.do_prior_divergence:
            self.do_prior_prediction = True
        # move vae to device if we did not cache latents
        if not self.is_latents_cached:
            self.sd.vae.eval()
            self.sd.vae.to(self.device_torch)
        else:
            # offload it. Already cached
            self.sd.vae.to('cpu')
            flush()
        add_all_snr_to_noise_scheduler(self.sd.noise_scheduler, self.device_torch)
        if self.adapter is not None:
            self.adapter.to(self.device_torch)

            # check if we have regs and using adapter and caching clip embeddings
            has_reg = self.datasets_reg is not None and len(self.datasets_reg) > 0
            is_caching_clip_embeddings = self.datasets is not None and any([self.datasets[i].cache_clip_vision_to_disk for i in range(len(self.datasets))])

            if has_reg and is_caching_clip_embeddings:
                # we need a list of unconditional clip image embeds from other datasets to handle regs
                unconditional_clip_image_embeds = []
                datasets = get_dataloader_datasets(self.data_loader)
                for i in range(len(datasets)):
                    unconditional_clip_image_embeds += datasets[i].clip_vision_unconditional_cache

                if len(unconditional_clip_image_embeds) == 0:
                    raise ValueError("No unconditional clip image embeds found. This should not happen")

                self._clip_image_embeds_unconditional = unconditional_clip_image_embeds

        if self.train_config.negative_prompt is not None:
            if os.path.exists(self.train_config.negative_prompt):
                with open(self.train_config.negative_prompt, 'r') as f:
                    self.negative_prompt_pool = f.readlines()
                    # remove empty
                    self.negative_prompt_pool = [x.strip() for x in self.negative_prompt_pool if x.strip() != ""]
            else:
                # single prompt
                self.negative_prompt_pool = [self.train_config.negative_prompt]

        # handle unload text encoder
        if self.train_config.unload_text_encoder or self.is_caching_text_embeddings:
            print_acc("Caching embeddings and unloading text encoder")
            with torch.no_grad():
                if self.train_config.train_text_encoder:
                    raise ValueError("Cannot unload text encoder if training text encoder")
                # cache embeddings
                self.sd.text_encoder_to(self.device_torch)
                encode_kwargs = {}
                if self.sd.encode_control_in_text_embeddings:
                    # just do a blank image for unconditionals
                    control_image = torch.zeros((1, 3, 224, 224), device=self.sd.device_torch, dtype=self.sd.torch_dtype)
                    if self.sd.has_multiple_control_images:
                        control_image = [control_image]
                    encode_kwargs['control_images'] = control_image
                self.cached_blank_embeds = self.sd.encode_prompt("", **encode_kwargs)
                if self.trigger_word is not None:
                    self.cached_trigger_embeds = self.sd.encode_prompt(self.trigger_word, **encode_kwargs)
                if self.train_config.diff_output_preservation:
                    self.diff_output_preservation_embeds = self.sd.encode_prompt(self.train_config.diff_output_preservation_class)
                
                self.cache_sample_prompts()
                
                print_acc("\n***** UNLOADING TEXT ENCODER *****")
                if self.is_caching_text_embeddings:
                    print_acc("Embeddings cached to disk. We dont need the text encoder anymore")
                else:
                    print_acc("This will train only with a blank prompt or trigger word, if set")
                    print_acc("If this is not what you want, remove the unload_text_encoder flag")
                print_acc("***********************************")
                print_acc("")

                # unload the text encoder
                if self.is_caching_text_embeddings:
                    unload_text_encoder(self.sd)
                else:
                    # todo once every model is tested to work, unload properly. Though, this will all be merged into one thing.
                    # keep legacy usage for now. 
                    self.sd.text_encoder_to("cpu")
                flush()
        
        if self.train_config.blank_prompt_preservation and self.cached_blank_embeds is None:
            # make sure we have this if not unloading
            self.cached_blank_embeds = self.sd.encode_prompt("").to(
                self.device_torch,
                dtype=self.sd.torch_dtype
            ).detach()
        
        if self.train_config.diffusion_feature_extractor_path is not None:
            vae = self.sd.vae
            # if not (self.model_config.arch in ["flux"]) or self.sd.vae.__class__.__name__ == "AutoencoderPixelMixer":
            #     vae = self.sd.vae
            self.dfe = load_dfe(self.train_config.diffusion_feature_extractor_path, vae=vae)
            self.dfe.to(self.device_torch)
            if hasattr(self.dfe, 'vision_encoder') and self.train_config.gradient_checkpointing:
                # must be set to train for gradient checkpointing to work
                self.dfe.vision_encoder.train()
                self.dfe.vision_encoder.gradient_checkpointing = True
            else:
                self.dfe.eval()
                
            # enable gradient checkpointing on the vae
            if vae is not None and self.train_config.gradient_checkpointing:
                try:
                    vae.enable_gradient_checkpointing()
                    vae.train()
                except:
                    pass


    def process_output_for_turbo(self, pred, noisy_latents, timesteps, noise, batch):
        # to process turbo learning, we make one big step from our current timestep to the end
        # we then denoise the prediction on that remaining step and target our loss to our target latents
        # this currently only works on euler_a (that I know of). Would work on others, but needs to be coded to do so.
        # needs to be done on each item in batch as they may all have different timesteps
        batch_size = pred.shape[0]
        pred_chunks = torch.chunk(pred, batch_size, dim=0)
        noisy_latents_chunks = torch.chunk(noisy_latents, batch_size, dim=0)
        timesteps_chunks = torch.chunk(timesteps, batch_size, dim=0)
        latent_chunks = torch.chunk(batch.latents, batch_size, dim=0)
        noise_chunks = torch.chunk(noise, batch_size, dim=0)

        with torch.no_grad():
            # set the timesteps to 1000 so we can capture them to calculate the sigmas
            self.sd.noise_scheduler.set_timesteps(
                self.sd.noise_scheduler.config.num_train_timesteps,
                device=self.device_torch
            )
            train_timesteps = self.sd.noise_scheduler.timesteps.clone().detach()

            train_sigmas = self.sd.noise_scheduler.sigmas.clone().detach()

            # set the scheduler to one timestep, we build the step and sigmas for each item in batch for the partial step
            self.sd.noise_scheduler.set_timesteps(
                1,
                device=self.device_torch
            )

        denoised_pred_chunks = []
        target_pred_chunks = []

        for i in range(batch_size):
            pred_item = pred_chunks[i]
            noisy_latents_item = noisy_latents_chunks[i]
            timesteps_item = timesteps_chunks[i]
            latents_item = latent_chunks[i]
            noise_item = noise_chunks[i]
            with torch.no_grad():
                timestep_idx = [(train_timesteps == t).nonzero().item() for t in timesteps_item][0]
                single_step_timestep_schedule = [timesteps_item.squeeze().item()]
                # extract the sigma idx for our midpoint timestep
                sigmas = train_sigmas[timestep_idx:timestep_idx + 1].to(self.device_torch)

                end_sigma_idx = random.randint(timestep_idx, len(train_sigmas) - 1)
                end_sigma = train_sigmas[end_sigma_idx:end_sigma_idx + 1].to(self.device_torch)

                # add noise to our target

                # build the big sigma step. The to step will now be to 0 giving it a full remaining denoising half step
                # self.sd.noise_scheduler.sigmas = torch.cat([sigmas, torch.zeros_like(sigmas)]).detach()
                self.sd.noise_scheduler.sigmas = torch.cat([sigmas, end_sigma]).detach()
                # set our single timstep
                self.sd.noise_scheduler.timesteps = torch.from_numpy(
                    np.array(single_step_timestep_schedule, dtype=np.float32)
                ).to(device=self.device_torch)

                # set the step index to None so it will be recalculated on first step
                self.sd.noise_scheduler._step_index = None

            denoised_latent = self.sd.noise_scheduler.step(
                pred_item, timesteps_item, noisy_latents_item.detach(), return_dict=False
            )[0]

            residual_noise = (noise_item * end_sigma.flatten()).detach().to(self.device_torch, dtype=get_torch_dtype(
                self.train_config.dtype))
            # remove the residual noise from the denoised latents. Output should be a clean prediction (theoretically)
            denoised_latent = denoised_latent - residual_noise

            denoised_pred_chunks.append(denoised_latent)

        denoised_latents = torch.cat(denoised_pred_chunks, dim=0)
        # set the scheduler back to the original timesteps
        self.sd.noise_scheduler.set_timesteps(
            self.sd.noise_scheduler.config.num_train_timesteps,
            device=self.device_torch
        )

        output = denoised_latents / self.sd.vae.config['scaling_factor']
        output = self.sd.vae.decode(output).sample

        if self.train_config.show_turbo_outputs:
            # since we are completely denoising, we can show them here
            with torch.no_grad():
                show_tensors(output)

        # we return our big partial step denoised latents as our pred and our untouched latents as our target.
        # you can do mse against the two here  or run the denoised through the vae for pixel space loss against the
        # input tensor images.

        return output, batch.tensor.to(self.device_torch, dtype=get_torch_dtype(self.train_config.dtype))

    def _build_subject_mask_weight(
        self,
        batch: 'DataLoaderBatchDTO',
        noisy_latents_shape,
        dtype=torch.float32,
    ):
        """Build the per-region loss weight map from cached subject masks.

        Returns a float tensor of shape (B, C_latent, lat_h, lat_w) that can be
        multiplied into ``mask_multiplier`` (composes multiplicatively with face
        suppression; does not replace it), or ``None`` when no weighting applies.

        Composition rules (order matters; each default is no-op):
          weight_map = ones
          if bg_w is set:        weight_map *= where(person, 1, bg_w)
          if clothing_w is set:  weight_map *= where(clothing, clothing_w, 1)
          if body_w is set:      weight_map *= where(body, body_w, 1)

        Each of bg_w / clothing_w / body_w is resolved per-item:
          non-None per-dataset override > non-None global > None (no-op).
        """
        smc = getattr(self, 'subject_mask_config', None)
        if smc is None or not smc.enabled:
            return None
        have_masks = (
            getattr(batch, 'subject_masks', None) is not None
            or getattr(batch, 'body_masks', None) is not None
            or getattr(batch, 'clothing_masks', None) is not None
        )
        if not have_masks:
            return None

        g_bg = smc.background_loss_weight
        g_cl = smc.clothing_loss_weight
        g_bd = smc.body_loss_weight

        bg_list = getattr(batch, 'background_loss_weight_list', None) or []
        cl_list = getattr(batch, 'clothing_loss_weight_list', None) or []
        bd_list = getattr(batch, 'body_loss_weight_list', None) or []
        bs = len(batch.file_items)

        def _pad(lst):
            out = list(lst)
            if len(out) < bs:
                out = out + [None] * (bs - len(out))
            return out

        bg_ws = [(w if w is not None else g_bg) for w in _pad(bg_list)]
        cl_ws = [(w if w is not None else g_cl) for w in _pad(cl_list)]
        bd_ws = [(w if w is not None else g_bd) for w in _pad(bd_list)]

        any_bg = any(w is not None for w in bg_ws)
        any_cl = any(w is not None for w in cl_ws)
        any_bd = any(w is not None for w in bd_ws)
        if not (any_bg or any_cl or any_bd):
            return None

        if len(noisy_latents_shape) == 5:
            # Video B,C,T,H,W
            lat_h, lat_w = noisy_latents_shape[3], noisy_latents_shape[4]
        else:
            lat_h, lat_w = noisy_latents_shape[2], noisy_latents_shape[3]
        device = getattr(self, 'device_torch', torch.device('cpu'))

        def _resize_mask(stacked):
            # stacked: (B, 1, H_c, W_c) bool → float on device (B, 1, lat_h, lat_w)
            if stacked is None:
                return None
            m = stacked.to(device=device, dtype=torch.float32)
            m = torch.nn.functional.interpolate(
                m, size=(lat_h, lat_w), mode='nearest'
            )
            return m

        person_lat = _resize_mask(getattr(batch, 'subject_masks', None))
        body_lat = _resize_mask(getattr(batch, 'body_masks', None))
        clothing_lat = _resize_mask(getattr(batch, 'clothing_masks', None))

        def _per_item(weights):
            ws = [float(w) if w is not None else 1.0 for w in weights]
            return torch.tensor(ws, device=device, dtype=dtype).view(bs, 1, 1, 1)

        def _active(weights):
            return torch.tensor(
                [1.0 if w is not None else 0.0 for w in weights],
                device=device, dtype=dtype,
            ).view(bs, 1, 1, 1)

        bg_scalar = _per_item(bg_ws)
        cl_scalar = _per_item(cl_ws)
        bd_scalar = _per_item(bd_ws)
        bg_active = _active(bg_ws)
        cl_active = _active(cl_ws)
        bd_active = _active(bd_ws)

        subject_weight = torch.ones(
            (bs, 1, lat_h, lat_w), device=device, dtype=dtype,
        )

        if any_bg and person_lat is not None:
            person = person_lat.to(dtype=dtype)
            term_on = torch.ones_like(person)
            term_off = bg_scalar.expand_as(person)
            layer = person * term_on + (1.0 - person) * term_off
            layer = bg_active * layer + (1.0 - bg_active) * torch.ones_like(layer)
            subject_weight = subject_weight * layer

        if any_cl and clothing_lat is not None:
            clothing = clothing_lat.to(dtype=dtype)
            term_on = cl_scalar.expand_as(clothing)
            term_off = torch.ones_like(clothing)
            layer = clothing * term_on + (1.0 - clothing) * term_off
            layer = cl_active * layer + (1.0 - cl_active) * torch.ones_like(layer)
            subject_weight = subject_weight * layer

        if any_bd and body_lat is not None:
            body = body_lat.to(dtype=dtype)
            term_on = bd_scalar.expand_as(body)
            term_off = torch.ones_like(body)
            layer = body * term_on + (1.0 - body) * term_off
            layer = bd_active * layer + (1.0 - bd_active) * torch.ones_like(layer)
            subject_weight = subject_weight * layer

        # Broadcast over latent channels for downstream multiplication
        subject_weight = subject_weight.expand(
            -1, noisy_latents_shape[1], -1, -1,
        )
        return subject_weight

    def _build_body_restrict_mask(self, batch: 'DataLoaderBatchDTO', spatial_shape):
        """Return a float (B, H, W) body-region mask resized to ``spatial_shape`` or None.

        None = perceptual_restrict_to_body disabled (global and all per-item) or
        body masks not cached. When returned, the mask has 1.0 inside the body
        region and 0.0 outside. Items that have not opted in are set to 1.0
        everywhere so their perceptual loss is unchanged.

        Used by per-pixel perceptual losses (currently normal_loss) so they can
        focus on identity-relevant regions (hair/face/limbs) when enabled.
        """
        smc = getattr(self, 'subject_mask_config', None)
        if smc is None or not smc.enabled:
            return None
        if getattr(batch, 'body_masks', None) is None:
            return None
        g_restrict = bool(getattr(smc, 'perceptual_restrict_to_body', False))
        per_item = [
            (v if v is not None else g_restrict)
            for v in getattr(batch, 'perceptual_restrict_to_body_list', [])
        ]
        if not any(per_item):
            return None
        _, H, W = spatial_shape
        B = len(per_item)
        body = batch.body_masks.to(torch.float32)  # (B, 1, H_c, W_c)
        body = torch.nn.functional.interpolate(body, size=(H, W), mode='nearest')
        body = body.squeeze(1)  # (B, H, W)
        restrict_vec = torch.tensor(
            [1.0 if v else 0.0 for v in per_item], dtype=torch.float32
        ).view(B, 1, 1)
        # items that opted in: body mask; items that didn't: all-ones
        out = restrict_vec * body + (1.0 - restrict_vec) * torch.ones_like(body)
        return out

    # you can expand these in a child class to make customization easier
    def calculate_loss(
            self,
            noise_pred: torch.Tensor,
            noise: torch.Tensor,
            noisy_latents: torch.Tensor,
            timesteps: torch.Tensor,
            batch: 'DataLoaderBatchDTO',
            mask_multiplier: Union[torch.Tensor, float] = 1.0,
            prior_pred: Union[torch.Tensor, None] = None,
            **kwargs
    ):
        loss_target = self.train_config.loss_target
        is_reg_list = batch.get_is_reg_list()
        is_reg = any(is_reg_list)
        # Per-sample reg flag tensor (B,), used to gate every auxiliary
        # (non-diffusion) loss off for reg samples. Diffusion loss has its
        # own `loss_multiplier * reg_weight` scaling and is unaffected.
        is_reg_per_sample = torch.tensor(
            [bool(v) for v in is_reg_list],
            device=self.device_torch, dtype=torch.bool,
        )
        # Per-sample loss-split flag — see DatasetConfig.loss_split. When
        # set to 'diffusion_depth', the dataset alternates which big loss
        # fires per optimizer step (not per microbatch — gating keys on
        # self.step_num, which only advances after the full accumulation
        # window completes, so all microbatches in one optimizer step see
        # the same active loss). Even step_num: diffusion only; odd: depth
        # only. Other auxiliaries fire as their own gating allows on both.
        _split_list = getattr(batch, 'loss_split_list', None) or [None] * is_reg_per_sample.shape[0]
        loss_split_diff_depth = torch.tensor(
            [s == 'diffusion_depth' for s in _split_list],
            device=self.device_torch, dtype=torch.bool,
        )
        _step_is_diffusion = (self.step_num % 2 == 0)
        additional_loss = 0.0
        # Reset every microbatch so a step where depth doesn't contribute
        # leaves the dual-backward block with no tensor to act on.
        self._dc_applied_for_grad = None

        # log timestep ratio for distribution visualization
        with torch.no_grad():
            num_train_ts = float(self.sd.noise_scheduler.config.num_train_timesteps)
            self._last_timestep = (timesteps.float() / num_train_ts).mean().item()

        prior_mask_multiplier = None
        target_mask_multiplier = None
        dtype = get_torch_dtype(self.train_config.dtype)

        has_mask = batch.mask_tensor is not None

        with torch.no_grad():
            loss_multiplier = torch.tensor(batch.loss_multiplier_list).to(self.device_torch, dtype=torch.float32)

        if self.train_config.match_noise_norm:
            # match the norm of the noise
            noise_norm = torch.linalg.vector_norm(noise, ord=2, dim=(1, 2, 3), keepdim=True)
            noise_pred_norm = torch.linalg.vector_norm(noise_pred, ord=2, dim=(1, 2, 3), keepdim=True)
            noise_pred = noise_pred * (noise_norm / noise_pred_norm)

        if self.train_config.pred_scaler != 1.0:
            noise_pred = noise_pred * self.train_config.pred_scaler

        target = None

        if self.train_config.target_noise_multiplier != 1.0:
            noise = noise * self.train_config.target_noise_multiplier

        if self.train_config.correct_pred_norm or (self.train_config.inverted_mask_prior and prior_pred is not None and has_mask):
            if self.train_config.correct_pred_norm and not is_reg:
                with torch.no_grad():
                    # this only works if doing a prior pred
                    if prior_pred is not None:
                        prior_mean = prior_pred.mean([2,3], keepdim=True)
                        prior_std = prior_pred.std([2,3], keepdim=True)
                        noise_mean = noise_pred.mean([2,3], keepdim=True)
                        noise_std = noise_pred.std([2,3], keepdim=True)

                        mean_adjust = prior_mean - noise_mean
                        std_adjust = prior_std - noise_std

                        mean_adjust = mean_adjust * self.train_config.correct_pred_norm_multiplier
                        std_adjust = std_adjust * self.train_config.correct_pred_norm_multiplier

                        target_mean = noise_mean + mean_adjust
                        target_std = noise_std + std_adjust

                        eps = 1e-5
                        # match the noise to the prior
                        noise = (noise - noise_mean) / (noise_std + eps)
                        noise = noise * (target_std + eps) + target_mean
                        noise = noise.detach()

            if self.train_config.inverted_mask_prior and prior_pred is not None and has_mask:
                assert not self.train_config.train_turbo
                with torch.no_grad():
                    prior_mask = batch.mask_tensor.to(self.device_torch, dtype=dtype)
                    if len(noise_pred.shape) == 5:
                        # video B,C,T,H,W
                        lat_height = batch.latents.shape[3]
                        lat_width = batch.latents.shape[4]
                    else: 
                        lat_height = batch.latents.shape[2]
                        lat_width = batch.latents.shape[3]
                    # resize to size of noise_pred
                    prior_mask = torch.nn.functional.interpolate(prior_mask, size=(lat_height, lat_width), mode='bicubic')
                    # stack first channel to match channels of noise_pred
                    prior_mask = torch.cat([prior_mask[:1]] * noise_pred.shape[1], dim=1)
                    
                    if len(noise_pred.shape) == 5:
                        prior_mask = prior_mask.unsqueeze(2)  # add time dimension back for video
                        prior_mask = prior_mask.repeat(1, 1, noise_pred.shape[2], 1, 1) 

                    prior_mask_multiplier = 1.0 - prior_mask
                    
                    # scale so it is a mean of 1
                    prior_mask_multiplier = prior_mask_multiplier / prior_mask_multiplier.mean()
                if hasattr(self.sd, 'get_loss_target'):
                    target = self.sd.get_loss_target(
                        noise=noise, 
                        batch=batch, 
                        timesteps=timesteps,
                    ).detach()
                elif self.sd.is_flow_matching:
                    target = (noise - batch.latents).detach()
                else:
                    target = noise
        elif prior_pred is not None and not self.train_config.do_prior_divergence:
            assert not self.train_config.train_turbo
            # matching adapter prediction
            target = prior_pred
        elif self.sd.prediction_type == 'v_prediction':
            # v-parameterization training
            target = self.sd.noise_scheduler.get_velocity(batch.tensor, noise, timesteps)
        
        elif hasattr(self.sd, 'get_loss_target'):
            target = self.sd.get_loss_target(
                noise=noise, 
                batch=batch, 
                timesteps=timesteps,
            ).detach()
            
        elif self.sd.is_flow_matching:
            # forward ODE
            target = (noise - batch.latents).detach()
            # reverse ODE
            # target = (batch.latents - noise).detach()
        else:
            target = noise
            
        if self.dfe is not None:
            if self.dfe.version == 1:
                model = self.sd
                if model is not None and hasattr(model, 'get_stepped_pred'):
                    stepped_latents = model.get_stepped_pred(noise_pred, noise)
                else:
                    # stepped_latents = noise - noise_pred
                    # first we step the scheduler from current timestep to the very end for a full denoise
                    bs = noise_pred.shape[0]
                    noise_pred_chunks = torch.chunk(noise_pred, bs)
                    timestep_chunks = torch.chunk(timesteps, bs)
                    noisy_latent_chunks = torch.chunk(noisy_latents, bs)
                    stepped_chunks = []
                    for idx in range(bs):
                        model_output = noise_pred_chunks[idx]
                        timestep = timestep_chunks[idx]
                        self.sd.noise_scheduler._step_index = None
                        self.sd.noise_scheduler._init_step_index(timestep)
                        sample = noisy_latent_chunks[idx].to(torch.float32)
                        
                        sigma = self.sd.noise_scheduler.sigmas[self.sd.noise_scheduler.step_index]
                        sigma_next = self.sd.noise_scheduler.sigmas[-1] # use last sigma for final step
                        prev_sample = sample + (sigma_next - sigma) * model_output
                        stepped_chunks.append(prev_sample)
                    
                    stepped_latents = torch.cat(stepped_chunks, dim=0)
                    
                stepped_latents = stepped_latents.to(self.sd.vae.device, dtype=self.sd.vae.dtype)
                sl = stepped_latents
                if len(sl.shape) == 5:
                    # video B,C,T,H,W
                    sl = sl.permute(0, 2, 1, 3, 4)  # B,T,C,H,W
                    b, t, c, h, w = sl.shape
                    sl = sl.reshape(b * t, c, h, w)
                pred_features = self.dfe(sl.float())
                with torch.no_grad():
                    bl = batch.latents
                    bl = bl.to(self.sd.vae.device)
                    if len(bl.shape) == 5:
                        # video B,C,T,H,W
                        bl = bl.permute(0, 2, 1, 3, 4)  # B,T,C,H,W
                        b, t, c, h, w = bl.shape
                        bl = bl.reshape(b * t, c, h, w)
                    target_features = self.dfe(bl.float())
                    # scale dfe so it is weaker at higher noise levels
                    dfe_scaler = 1 - (timesteps.float() / 1000.0).view(-1, 1, 1, 1).to(self.device_torch)
                
                dfe_loss = torch.nn.functional.mse_loss(pred_features, target_features, reduction="none") * \
                    self.train_config.diffusion_feature_extractor_weight * dfe_scaler
                additional_loss += dfe_loss.mean()
            elif self.dfe.version == 2:
                # version 2
                # do diffusion feature extraction on target
                with torch.no_grad():
                    rectified_flow_target = noise.float() - batch.latents.float()
                    target_feature_list = self.dfe(torch.cat([rectified_flow_target, noise.float()], dim=1))
                
                # do diffusion feature extraction on prediction
                pred_feature_list = self.dfe(torch.cat([noise_pred.float(), noise.float()], dim=1))
                
                dfe_loss = 0.0
                for i in range(len(target_feature_list)):
                    dfe_loss += torch.nn.functional.mse_loss(pred_feature_list[i], target_feature_list[i], reduction="mean")
                
                additional_loss += dfe_loss * self.train_config.diffusion_feature_extractor_weight * 100.0
            elif self.dfe.version in [3, 4, 5, 6]:
                dfe_loss = self.dfe(
                    noise=noise,
                    noise_pred=noise_pred,
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    batch=batch,
                    scheduler=self.sd.noise_scheduler
                )
                additional_loss += dfe_loss * self.train_config.diffusion_feature_extractor_weight 
            else:
                raise ValueError(f"Unknown diffusion feature extractor version {self.dfe.version}")
        
        if self.train_config.do_guidance_loss:
            with torch.no_grad():
                # we make cached blank prompt embeds that match the batch size
                unconditional_embeds = concat_prompt_embeds(
                    [self.unconditional_embeds] * noisy_latents.shape[0],
                )
                unconditional_target = self.predict_noise(
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    conditional_embeds=unconditional_embeds,
                    unconditional_embeds=None,
                    batch=batch,
                )
                is_video = len(target.shape) == 5
                
                if self.train_config.do_guidance_loss_cfg_zero:
                    # zero cfg
                    # ref https://github.com/WeichenFan/CFG-Zero-star/blob/cdac25559e3f16cb95f0016c04c709ea1ab9452b/wan_pipeline.py#L557
                    batch_size = target.shape[0]
                    positive_flat = target.view(batch_size, -1)
                    negative_flat = unconditional_target.view(batch_size, -1)
                    # Calculate dot production
                    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
                    # Squared norm of uncondition
                    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
                    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
                    st_star = dot_product / squared_norm

                    alpha = st_star
                    
                    alpha = alpha.view(batch_size, 1, 1, 1) if not is_video else alpha.view(batch_size, 1, 1, 1, 1)
                else:
                    alpha = 1.0

                guidance_scale = self._guidance_loss_target_batch
                if isinstance(guidance_scale, list):
                    guidance_scale = torch.tensor(guidance_scale).to(target.device, dtype=target.dtype)
                    guidance_scale = guidance_scale.view(-1, 1, 1, 1) if not is_video else guidance_scale.view(-1, 1, 1, 1, 1)
                
                unconditional_target = unconditional_target * alpha
                target = unconditional_target + guidance_scale * (target - unconditional_target)

            if self.train_config.do_differential_guidance:
                with torch.no_grad():
                    guidance_scale = self.train_config.differential_guidance_scale
                    target = noise_pred + guidance_scale * (target - noise_pred)
            
        if target is None:
            target = noise

        pred = noise_pred

        if self.train_config.train_turbo:
            pred, target = self.process_output_for_turbo(pred, noisy_latents, timesteps, noise, batch)

        ignore_snr = False

        if loss_target == 'source' or loss_target == 'unaugmented':
            assert not self.train_config.train_turbo
            # ignore_snr = True
            if batch.sigmas is None:
                raise ValueError("Batch sigmas is None. This should not happen")

            # src https://github.com/huggingface/diffusers/blob/324d18fba23f6c9d7475b0ff7c777685f7128d40/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L1190
            denoised_latents = noise_pred * (-batch.sigmas) + noisy_latents
            weighing = batch.sigmas ** -2.0
            if loss_target == 'source':
                # denoise the latent and compare to the latent in the batch
                target = batch.latents
            elif loss_target == 'unaugmented':
                # we have to encode images into latents for now
                # we also denoise as the unaugmented tensor is not a noisy diffirental
                with torch.no_grad():
                    unaugmented_latents = self.sd.encode_images(batch.unaugmented_tensor).to(self.device_torch, dtype=dtype)
                    unaugmented_latents = unaugmented_latents * self.train_config.latent_multiplier
                    target = unaugmented_latents.detach()

                # Get the target for loss depending on the prediction type
                if self.sd.noise_scheduler.config.prediction_type == "epsilon":
                    target = target  # we are computing loss against denoise latents
                elif self.sd.noise_scheduler.config.prediction_type == "v_prediction":
                    target = self.sd.noise_scheduler.get_velocity(target, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {self.sd.noise_scheduler.config.prediction_type}")

            # mse loss without reduction
            loss_per_element = (weighing.float() * (denoised_latents.float() - target.float()) ** 2)
            loss = loss_per_element
        else:

            if self.train_config.loss_type == "mae":
                loss = torch.nn.functional.l1_loss(pred.float(), target.float(), reduction="none")
            elif self.train_config.loss_type == "wavelet":
                loss = wavelet_loss(pred, batch.latents, noise)
            elif self.train_config.loss_type == "stepped":
                loss = stepped_loss(pred, batch.latents, noise, noisy_latents, timesteps, self.sd.noise_scheduler)
                # the way this loss works, it is low, increase it to match predictable LR effects
                loss = loss * 10.0
            else:
                loss = torch.nn.functional.mse_loss(pred.float(), target.float(), reduction="none")
                
            do_weighted_timesteps = False
            if self.sd.is_flow_matching:
                if self.train_config.linear_timesteps or self.train_config.linear_timesteps2:
                    do_weighted_timesteps = True
                if self.train_config.timestep_type == "weighted":
                    # use the noise scheduler to get the weights for the timesteps
                    do_weighted_timesteps = True

            # handle linear timesteps and only adjust the weight of the timesteps
            if do_weighted_timesteps:
                # calculate the weights for the timesteps
                timestep_weight = self.sd.noise_scheduler.get_weights_for_timesteps(
                    timesteps,
                    v2=self.train_config.linear_timesteps2,
                    timestep_type=self.train_config.timestep_type
                ).to(loss.device, dtype=loss.dtype)
                if len(loss.shape) == 4:
                    timestep_weight = timestep_weight.view(-1, 1, 1, 1).detach()
                elif len(loss.shape) == 5:
                    timestep_weight = timestep_weight.view(-1, 1, 1, 1, 1).detach()
                loss = loss * timestep_weight

        if self.train_config.do_prior_divergence and prior_pred is not None:
            loss = loss + (torch.nn.functional.mse_loss(pred.float(), prior_pred.float(), reduction="none") * -1.0)

        if self.train_config.train_turbo:
            mask_multiplier = mask_multiplier[:, 3:, :, :]
            # resize to the size of the loss
            mask_multiplier = torch.nn.functional.interpolate(mask_multiplier, size=(pred.shape[2], pred.shape[3]), mode='nearest')

        # multiply by our mask
        try:
            if len(noise_pred.shape) == 5:
                # video B,C,T,H,W
                mask_multiplier = mask_multiplier.unsqueeze(2)  # add time dimension back for video
                mask_multiplier = mask_multiplier.repeat(1, 1, noise_pred.shape[2], 1, 1)
            loss = loss * mask_multiplier
        except Exception as e:
            # todo handle mask with video models
            print("Could not apply mask multiplier to loss")
            print(e)
            pass

        prior_loss = None
        if self.train_config.inverted_mask_prior and prior_pred is not None and prior_mask_multiplier is not None:
            assert not self.train_config.train_turbo
            if self.train_config.loss_type == "mae":
                prior_loss = torch.nn.functional.l1_loss(pred.float(), prior_pred.float(), reduction="none")
            else:
                prior_loss = torch.nn.functional.mse_loss(pred.float(), prior_pred.float(), reduction="none")

            prior_loss = prior_loss * prior_mask_multiplier * self.train_config.inverted_mask_prior_multiplier
            if torch.isnan(prior_loss).any():
                print_acc("Prior loss is nan")
                prior_loss = None
            else:
                if len(noise_pred.shape) == 5:
                    # video B,C,T,H,W
                    prior_loss = prior_loss.mean([1, 2, 3, 4])
                else:
                    prior_loss = prior_loss.mean([1, 2, 3])
                # loss = loss + prior_loss
                # loss = loss + prior_loss
            # loss = loss + prior_loss
        if len(noise_pred.shape) == 5:
            loss = loss.mean([1, 2, 3, 4])
        else:
            loss = loss.mean([1, 2, 3])
        # apply loss multiplier before prior loss
        # multiply by our mask
        try:
            loss = loss * loss_multiplier
        except:
            # todo handle mask with video models
            pass

        if prior_loss is not None:
            loss = loss + prior_loss

        if not self.train_config.train_turbo:
            if self.train_config.learnable_snr_gos:
                # add snr_gamma
                loss = apply_learnable_snr_gos(loss, timesteps, self.snr_gos)
            elif self.train_config.snr_gamma is not None and self.train_config.snr_gamma > 0.000001 and not ignore_snr:
                # add snr_gamma
                loss = apply_snr_weight(loss, timesteps, self.sd.noise_scheduler, self.train_config.snr_gamma,
                                        fixed=True)
            elif self.train_config.min_snr_gamma is not None and self.train_config.min_snr_gamma > 0.000001 and not ignore_snr:
                # add min_snr_gamma
                loss = apply_snr_weight(loss, timesteps, self.sd.noise_scheduler, self.train_config.min_snr_gamma)

        # snapshot raw diffusion loss before timestep gating (for logging)
        with torch.no_grad():
            raw_diffusion_loss = loss.mean().detach()

            # Per-sample diffusion loss binned by timestep (0.1 bands) —
            # mirrors depth_loss_t00..t90 so the dashboard shows diffusion
            # loss evolution per noise level. Display-only; never feeds back
            # into the loss tensor.
            if loss.dim() == 1 and loss.shape[0] > 0:
                if self._last_diffusion_loss_bins is None:
                    self._last_diffusion_loss_bins = {}
                _diff_per_sample = loss.detach().float().cpu()
                _diff_t = (timesteps.float() / num_train_ts).detach()
                if _diff_t.dim() == 0:
                    _diff_t = _diff_t.expand(_diff_per_sample.shape[0])
                _diff_t_cpu = _diff_t.float().cpu().flatten()
                for _diff_i in range(_diff_per_sample.shape[0]):
                    _t_i = float(
                        _diff_t_cpu[_diff_i]
                        if _diff_i < _diff_t_cpu.numel()
                        else _diff_t_cpu[0]
                    )
                    _bin_start = int(_t_i * 10) / 10.0
                    _bin_key = f'diffusion_loss_t{int(_bin_start*100):02d}'
                    self._bin_update(
                        self._last_diffusion_loss_bins,
                        _bin_key,
                        float(_diff_per_sample[_diff_i]),
                    )
                    self._record_sample(
                        'diffusion_loss', float(_diff_per_sample[_diff_i]),
                        t=_t_i, idx=_diff_i, batch=batch,
                    )

        # apply diffusion loss timestep gating (per-sample, before .mean())
        if self.train_config.diffusion_loss_min_t > 0.0 or self.train_config.diffusion_loss_max_t < 1.0:
            t_ratio = timesteps.float() / num_train_ts
            diff_mask = ((t_ratio >= self.train_config.diffusion_loss_min_t) &
                         (t_ratio <= self.train_config.diffusion_loss_max_t)).float()
            loss = loss * diff_mask

        # apply per-sample diffusion loss weight overrides
        per_sample_diff_w = batch.diffusion_loss_weight_list
        # Loss-split gate: on depth steps, force per-sample diff weight to 0
        # for samples whose dataset has loss_split='diffusion_depth'. The
        # zero-weight samples are then excluded from active_count below
        # (mirrors the depth_loss_weight=0 short-circuit at the depth path).
        if loss_split_diff_depth.any() and not _step_is_diffusion:
            per_sample_diff_w = list(per_sample_diff_w)
            for _i in range(len(per_sample_diff_w)):
                if loss_split_diff_depth[_i]:
                    per_sample_diff_w[_i] = 0.0
        if any(w is not None for w in per_sample_diff_w):
            global_diff_w = self.train_config.diffusion_loss_weight
            diff_weights = torch.tensor(
                [w if w is not None else global_diff_w for w in per_sample_diff_w],
                device=loss.device, dtype=loss.dtype
            )
            # Exclude zero-weight samples from denominator to prevent dilution
            active_count = (diff_weights > 0).sum().clamp(min=1)
            loss = (loss * diff_weights).sum() / active_count
        else:
            loss = loss.mean()
            # scale diffusion loss
            if self.train_config.diffusion_loss_weight != 1.0:
                loss = loss * self.train_config.diffusion_loss_weight
        self._last_diffusion_loss_applied = loss.detach().item()

        # check for audio loss
        if batch.audio_pred is not None and batch.audio_target is not None:
            audio_loss = torch.nn.functional.mse_loss(batch.audio_pred.float(), batch.audio_target.float(), reduction="mean")
            audio_loss = audio_loss * self.train_config.audio_loss_multiplier
            loss = loss + audio_loss

        # check for additional losses
        if self.adapter is not None and hasattr(self.adapter, "additional_loss") and self.adapter.additional_loss is not None:

            loss = loss + self.adapter.additional_loss.mean()
            self.adapter.additional_loss = None

        if self.train_config.target_norm_std:
            # seperate out the batch and channels
            pred_std = noise_pred.std([2, 3], keepdim=True)
            norm_std_loss = torch.abs(self.train_config.target_norm_std_value - pred_std).mean()
            loss = loss + norm_std_loss

        # snapshot diffusion loss before identity loss is added
        # uses raw (un-gated) loss so the metric shows what diffusion loss
        # would have been even when timestep-gated off
        self._last_diffusion_loss = raw_diffusion_loss.item()

        # Auxiliary face losses (identity, landmark) via decoded x0 predictions
        # Evaluated at HIGH noise where x0_pred is a genuine generation.
        _need_id_loss = (self.id_loss_model is not None
                         and batch.identity_embedding is not None)
        _need_landmark_loss = (self.landmark_loss_model is not None
                               and batch.landmark_embedding is not None)
        _need_body_proportion_loss = (self.body_proportion_model is not None
                                      and batch.body_proportion_embedding is not None)
        _need_body_shape_loss = (self.body_shape_model is not None
                                 and batch.body_shape_embedding is not None)
        _need_normal_loss = (self.normal_model is not None
                             and batch.normal_embedding is not None)
        _need_vae_anchor_loss = (self.vae_anchor_encoder is not None
                                 and batch.vae_anchor_features is not None)

        if _need_id_loss or _need_landmark_loss or _need_body_proportion_loss or _need_body_shape_loss or _need_normal_loss or _need_vae_anchor_loss:
            num_train_timesteps = float(self.sd.noise_scheduler.config.num_train_timesteps)
            t_ratio = timesteps.float() / num_train_timesteps

            # Build per-sample timestep masks (per-dataset overrides fall back to global)
            def _per_sample_mask(batch_min_list, batch_max_list, global_min, global_max):
                """Build (B,) bool mask using per-sample min/max_t with global fallback.

                Reg samples are unconditionally excluded — auxiliary losses
                shouldn't fire on samples used for prior preservation. The
                diffusion loss handles reg via `loss_multiplier * reg_weight`
                separately and is unaffected by this gate.
                """
                min_vals = torch.tensor(
                    [v if v is not None else global_min for v in batch_min_list],
                    device=t_ratio.device, dtype=t_ratio.dtype,
                )
                max_vals = torch.tensor(
                    [v if v is not None else global_max for v in batch_max_list],
                    device=t_ratio.device, dtype=t_ratio.dtype,
                )
                t_mask = (t_ratio > min_vals) & (t_ratio < max_vals)
                return t_mask & (~is_reg_per_sample.to(t_mask.device))

            # Face losses (identity + landmark) use their own timestep window
            high_noise_mask = _per_sample_mask(
                batch.identity_loss_min_t_list, batch.identity_loss_max_t_list,
                self.face_id_config.identity_loss_min_t, self.face_id_config.identity_loss_max_t,
            )

            # Body proportion loss has its own timestep window
            bp_noise_mask = _per_sample_mask(
                batch.body_proportion_loss_min_t_list, batch.body_proportion_loss_max_t_list,
                self.face_id_config.body_proportion_loss_min_t, self.face_id_config.body_proportion_loss_max_t,
            )

            # Body shape loss has its own timestep window
            bsh_noise_mask = _per_sample_mask(
                batch.body_shape_loss_min_t_list, batch.body_shape_loss_max_t_list,
                self.face_id_config.body_shape_loss_min_t, self.face_id_config.body_shape_loss_max_t,
            )

            # Normal loss has its own timestep window
            nrm_noise_mask = _per_sample_mask(
                batch.normal_loss_min_t_list, batch.normal_loss_max_t_list,
                self.face_id_config.normal_loss_min_t, self.face_id_config.normal_loss_max_t,
            )

            # VAE anchor loss has its own timestep window
            va_noise_mask = _per_sample_mask(
                batch.vae_anchor_loss_min_t_list, batch.vae_anchor_loss_max_t_list,
                self.face_id_config.vae_anchor_loss_min_t, self.face_id_config.vae_anchor_loss_max_t,
            )

            cos_sim = None  # set by identity loss, reused by landmark loss for gating

            # identity_metrics: always decode x0 so we can track similarity at all timesteps
            _id_metrics_always = (self.face_id_config is not None
                                  and self.face_id_config.identity_metrics
                                  and _need_id_loss)

            # Decode x0 if any loss needs it
            any_active = (high_noise_mask.any()
                          or (_need_body_proportion_loss and bp_noise_mask.any())
                          or (_need_body_shape_loss and bsh_noise_mask.any())
                          or (_need_normal_loss and nrm_noise_mask.any())
                          or (_need_vae_anchor_loss and va_noise_mask.any())
                          or _id_metrics_always)
            if any_active:
                # Recover x0 prediction from model output
                if self.sd.is_flow_matching:
                    # Flow matching: x0 = noisy_latents - sigma * v_pred
                    sigma = t_ratio.view(-1, 1, 1, 1)
                    x0_pred = noisy_latents - sigma * noise_pred
                else:
                    # DDPM: recover x0 from epsilon or v prediction
                    alphas_cumprod = self.sd.noise_scheduler.alphas_cumprod.to(
                        device=timesteps.device, dtype=noisy_latents.dtype
                    )
                    alpha_bar = alphas_cumprod[timesteps.long()].view(-1, 1, 1, 1)
                    sqrt_alpha_bar = alpha_bar.sqrt()
                    sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()
                    if self.sd.prediction_type == 'v_prediction':
                        x0_pred = sqrt_alpha_bar * noisy_latents - sqrt_one_minus_alpha_bar * noise_pred
                    else:
                        # epsilon prediction
                        x0_pred = (noisy_latents - sqrt_one_minus_alpha_bar * noise_pred) / sqrt_alpha_bar.clamp(min=1e-8)

                # Decode x0 prediction to pixel space for face recognition
                if hasattr(self, '_taef2_decoder') and self._taef2_decoder is not None:
                    x0_for_decode = x0_pred
                    if x0_for_decode.shape[1] != 32:
                        x0_for_decode = rearrange(
                            x0_for_decode,
                            "b (c p1 p2) h w -> b c (h p1) (w p2)",
                            c=32, p1=2, p2=2,
                        )
                    dec_dtype = next(self._taef2_decoder.parameters()).dtype
                    x0_pixels = self._taef2_decoder(x0_for_decode.to(dec_dtype)).float().clamp(0, 1)
                elif self.taesd is not None:
                    taesd_dtype = next(self.taesd.parameters()).dtype
                    x0_pixels = self.taesd.decode(x0_pred.to(taesd_dtype)).sample.float()
                    x0_pixels = (x0_pixels + 1.0) * 0.5  # [-1,1] -> [0,1]
                else:
                    vae_dtype = self.sd.vae.dtype
                    x0_unscaled = x0_pred / self.sd.vae.config['scaling_factor']
                    if 'shift_factor' in self.sd.vae.config and self.sd.vae.config['shift_factor']:
                        x0_unscaled = x0_unscaled + self.sd.vae.config['shift_factor']
                    x0_decoded = self.sd.vae.decode(x0_unscaled.to(vae_dtype)).sample
                    x0_pixels = (x0_decoded.float() + 1.0) * 0.5  # [-1,1] -> [0,1]
                x0_pixels = x0_pixels.clamp(0, 1)

                # Scale face bboxes from original image coords to x0_pixels coords
                # Pipeline: orig -> scale -> crop -> latent -> decode to x0_pixels
                scaled_bboxes = None
                if batch.face_bboxes is not None:
                    _, _, px_h, px_w = x0_pixels.shape
                    scaled_bboxes = []
                    for idx in range(x0_pixels.shape[0]):
                        raw_bbox = batch.face_bboxes[idx] if idx < len(batch.face_bboxes) else None
                        if raw_bbox is not None:
                            fi = batch.file_items[idx]
                            orig_w = float(fi.width)
                            orig_h = float(fi.height)
                            bx1, by1, bx2, by2 = raw_bbox.float()

                            # Dataloader applies deterministic flips before
                            # scale+crop; mirror that here in raw coords so the
                            # bbox lands in the correct half of the training tensor.
                            if getattr(fi, 'flip_x', False):
                                bx1, bx2 = orig_w - bx2, orig_w - bx1
                            if getattr(fi, 'flip_y', False):
                                by1, by2 = orig_h - by2, orig_h - by1

                            stw = float(getattr(fi, 'scale_to_width', None) or orig_w)
                            sth = float(getattr(fi, 'scale_to_height', None) or orig_h)
                            bx1 = bx1 * (stw / orig_w)
                            by1 = by1 * (sth / orig_h)
                            bx2 = bx2 * (stw / orig_w)
                            by2 = by2 * (sth / orig_h)

                            cx = float(getattr(fi, 'crop_x', None) or 0)
                            cy = float(getattr(fi, 'crop_y', None) or 0)
                            cw = float(getattr(fi, 'crop_width', None) or stw)
                            ch = float(getattr(fi, 'crop_height', None) or sth)
                            bx1 = bx1 - cx
                            by1 = by1 - cy
                            bx2 = bx2 - cx
                            by2 = by2 - cy

                            if bx2 <= 0 or by2 <= 0 or bx1 >= cw or by1 >= ch:
                                scaled_bboxes.append(None)
                                continue

                            bx1 = bx1 * (px_w / cw)
                            by1 = by1 * (px_h / ch)
                            bx2 = bx2 * (px_w / cw)
                            by2 = by2 * (px_h / ch)

                            bx1 = max(0.0, min(bx1, float(px_w)))
                            by1 = max(0.0, min(by1, float(px_h)))
                            bx2 = max(0.0, min(bx2, float(px_w)))
                            by2 = max(0.0, min(by2, float(px_h)))

                            scaled_bboxes.append([bx1, by1, bx2, by2])
                        else:
                            scaled_bboxes.append(None)

                # --- Identity loss (ArcFace cosine similarity) ---
                if _need_id_loss:
                    _has_per_sample_id_w = any(w is not None and w > 0 for w in batch.identity_loss_weight_list)
                    _id_metrics_only = self.face_id_config.identity_loss_weight == 0 and not _has_per_sample_id_w
                    _id_ctx = torch.no_grad() if _id_metrics_only else contextlib.nullcontext()
                    with _id_ctx:
                        id_weight = t_ratio
                        gen_embedding, _arcface_crops = self.id_loss_model(
                            x0_pixels, bboxes=scaled_bboxes, return_crops=True)  # (B, 512), (B, 3, 112, 112)

                        # Gate: run face detector on x0 crops — skip blobs that
                        # ArcFace would score ~0.4 against faces.  ~1.5ms per crop.
                        _face_detected = torch.ones(gen_embedding.shape[0], dtype=torch.bool,
                                                     device=gen_embedding.device)
                        if self._id_face_detector is not None and _arcface_crops is not None:
                            with torch.no_grad():
                                for _ci in range(_arcface_crops.shape[0]):
                                    crop_np = (_arcface_crops[_ci].clamp(0, 1) * 255).byte()
                                    crop_np = crop_np.permute(1, 2, 0).cpu().numpy()  # (112,112,3) RGB
                                    crop_bgr = crop_np[:, :, ::-1].copy()  # BGR for insightface
                                    faces = self._id_face_detector.get(crop_bgr)
                                    if len(faces) == 0:
                                        _face_detected[_ci] = False

                        ref_embedding = batch.identity_embedding.to(
                            gen_embedding.device, dtype=gen_embedding.dtype)

                        # Blend mode: interpolate between per-image and dataset average
                        _blend = self.face_id_config.identity_loss_average_blend
                        if _blend > 0 and not self.face_id_config.identity_loss_use_average and self._identity_avg_embeds:
                            for idx in range(ref_embedding.shape[0]):
                                fi = batch.file_items[idx]
                                key = fi.dataset_config.folder_path or id(fi.dataset_config)
                                avg = self._identity_avg_embeds.get(key)
                                if avg is not None:
                                    avg_dev = avg.to(ref_embedding.device, dtype=ref_embedding.dtype)
                                    blended = (1.0 - _blend) * ref_embedding[idx] + _blend * avg_dev
                                    ref_embedding[idx] = blended / (blended.norm() + 1e-8)

                        # Random mode: replace ref with random embedding from same dataset's pool
                        if self.face_id_config.identity_loss_use_random and self._identity_embed_pools:
                            import random as _rand
                            for idx in range(ref_embedding.shape[0]):
                                fi = batch.file_items[idx]
                                key = fi.dataset_config.folder_path or id(fi.dataset_config)
                                pool = self._identity_embed_pools.get(key)
                                if pool is not None:
                                    rand_idx = _rand.randint(0, pool.shape[0] - 1)
                                    ref_embedding[idx] = pool[rand_idx].to(ref_embedding.device, dtype=ref_embedding.dtype)

                        # ArcFace bias correction: subtract mean embedding direction
                        # so non-face inputs score ~0 instead of ~0.5
                        if self._identity_mean_embed is not None:
                            mean_emb = self._identity_mean_embed.to(gen_embedding.device, dtype=gen_embedding.dtype)
                            gen_centered = gen_embedding - mean_emb.unsqueeze(0)
                            gen_centered = F.normalize(gen_centered, p=2, dim=-1)
                            ref_centered = ref_embedding - mean_emb.unsqueeze(0)
                            ref_centered = F.normalize(ref_centered, p=2, dim=-1)
                        else:
                            gen_centered = gen_embedding
                            ref_centered = ref_embedding

                        # Multi-ref mode: compare against K random refs, use best match
                        _num_refs = self.face_id_config.identity_loss_num_refs
                        if _num_refs > 0 and self._identity_embed_pools:
                            import random as _rand
                            all_cos = [F.cosine_similarity(gen_centered, ref_centered, dim=-1)]
                            for _ in range(_num_refs - 1):
                                rand_ref = ref_centered.clone()
                                for idx in range(rand_ref.shape[0]):
                                    fi = batch.file_items[idx]
                                    key = fi.dataset_config.folder_path or id(fi.dataset_config)
                                    pool = self._identity_embed_pools.get(key)
                                    if pool is not None:
                                        rand_idx = _rand.randint(0, pool.shape[0] - 1)
                                        rand_emb = pool[rand_idx].to(rand_ref.device, dtype=rand_ref.dtype)
                                        if self._identity_mean_embed is not None:
                                            rand_emb = F.normalize(rand_emb - mean_emb, p=2, dim=-1)
                                        rand_ref[idx] = rand_emb
                                all_cos.append(F.cosine_similarity(gen_centered, rand_ref, dim=-1))
                            cos_sim = torch.stack(all_cos).max(dim=0).values  # (B,) best match
                        else:
                            cos_sim = F.cosine_similarity(gen_centered, ref_centered, dim=-1)  # (B,)

                        # Metric mask: valid face, gated by timestep window unless identity_metrics
                        # is on (then all timesteps log so id_sim_tNN bins fill across t).
                        ref_valid = ref_embedding.abs().sum(dim=-1) > 0
                        if _id_metrics_always:
                            metric_mask = ref_valid  # all timesteps for logging
                        else:
                            metric_mask = ref_valid & high_noise_mask
                        # Loss mask: ALWAYS respects identity_loss_min_t/max_t even when
                        # identity_metrics drops the t-window from metric_mask, plus the
                        # per-sample cosine threshold to prevent pushing on hallucinated faces.
                        cos_threshold = torch.tensor(
                            [v if v is not None else self.face_id_config.identity_loss_min_cos
                             for v in batch.identity_loss_min_cos_list],
                            device=cos_sim.device, dtype=cos_sim.dtype,
                        )
                        loss_mask = (ref_valid & high_noise_mask
                                     & (cos_sim.detach() > cos_threshold) & _face_detected)

                        # Build per-sample identity weights early — gate loss_mask so
                        # zero-weight samples are fully excluded from loss computation
                        per_sample_id_w = batch.identity_loss_weight_list
                        _has_per_ds_id_w = any(w is not None for w in per_sample_id_w)
                        if _has_per_ds_id_w:
                            global_id_w = self.face_id_config.identity_loss_weight
                            id_weights = torch.tensor(
                                [w if w is not None else global_id_w for w in per_sample_id_w],
                                device=cos_sim.device, dtype=cos_sim.dtype,
                            )
                            loss_mask = loss_mask & (id_weights > 0)

                        # Clean similarity targets: in average mode, each sample's target
                        # is its clean ArcFace score vs the average, not 1.0.
                        # Loss = max(0, 1 - cos_sim/clean_cos) — normalized so no sample
                        # is over-weighted regardless of its absolute target.
                        _has_clean_targets = (
                            self.face_id_config.identity_loss_use_average
                            and any(c is not None for c in batch.identity_clean_cos_list)
                        )
                        if _has_clean_targets:
                            clean_cos = torch.tensor(
                                [c if c is not None else 1.0 for c in batch.identity_clean_cos_list],
                                device=cos_sim.device, dtype=cos_sim.dtype,
                            )
                            # Normalized shortfall: 0 when at target, 1 when cos_sim=0
                            id_loss_per_sample = torch.clamp(1.0 - cos_sim / clean_cos, min=0.0) * id_weight * loss_mask.float()
                        else:
                            id_loss_per_sample = (1.0 - cos_sim) * id_weight * loss_mask.float()
                        id_loss = id_loss_per_sample.sum() / max(loss_mask.sum().item(), 1.0)

                    if not _id_metrics_only and loss_mask.any():
                        if _has_per_ds_id_w:
                            weighted_id = (id_loss_per_sample * id_weights)
                            id_applied = weighted_id.sum() / max(loss_mask.sum().item(), 1.0)
                        else:
                            id_applied = self.face_id_config.identity_loss_weight * id_loss
                        loss = loss + id_applied
                        self._last_identity_loss_applied = id_applied.detach().item()
                    # Gate metrics by per-sample weights so zero-weight datasets
                    # don't bleed into identity metrics
                    valid_mask = metric_mask
                    if _has_per_ds_id_w:
                        valid_mask = valid_mask & (id_weights > 0)
                    with torch.no_grad():
                        if valid_mask.any():
                            raw_cos_dist = (1.0 - cos_sim) * valid_mask.float()
                            raw_count = valid_mask.sum().item()
                            self._last_identity_loss = (raw_cos_dist.sum() / raw_count).item()
                            raw_cos_sim = (cos_sim * valid_mask.float()).sum() / raw_count
                            self._last_id_sim = raw_cos_sim.item()
                            # Clean target metrics (average mode only)
                            if _has_clean_targets:
                                valid_clean = clean_cos * valid_mask.float()
                                self._last_id_clean_target = (valid_clean.sum() / raw_count).item()
                                raw_delta = (cos_sim - clean_cos) * valid_mask.float()
                                self._last_id_clean_delta = (raw_delta.sum() / raw_count).item()
                            if self._last_id_sim_bins is None:
                                self._last_id_sim_bins = {}
                            for idx in range(cos_sim.shape[0]):
                                if valid_mask[idx]:
                                    t_val = t_ratio[idx].item()
                                    bin_start = int(t_val * 10) / 10.0
                                    bin_key = f'id_sim_t{int(bin_start*100):02d}'
                                    cs_val = cos_sim[idx].item()
                                    self._bin_update(
                                        self._last_id_sim_bins,
                                        bin_key,
                                        cs_val,
                                    )
                                    # Per-sample breakdown for the tooltip.
                                    self._record_sample(
                                        'id_sim', cs_val,
                                        t=t_val, idx=idx, batch=batch,
                                    )
                        # save decoded x0 predictions + ArcFace crops to visualize identity pipeline
                        if valid_mask.any():
                            id_preview_dir = os.path.join(self.save_root, 'id_previews')
                            os.makedirs(id_preview_dir, exist_ok=True)
                            noisy_for_decode = noisy_latents
                            if hasattr(self, '_taef2_decoder') and self._taef2_decoder is not None:
                                nfd = noisy_for_decode
                                if nfd.shape[1] != 32:
                                    nfd = rearrange(nfd, "b (c p1 p2) h w -> b c (h p1) (w p2)", c=32, p1=2, p2=2)
                                noisy_pixels = self._taef2_decoder(nfd.to(dec_dtype)).float().clamp(0, 1)
                            elif self.taesd is not None:
                                noisy_pixels = self.taesd.decode(noisy_for_decode.to(taesd_dtype)).sample.float()
                                noisy_pixels = ((noisy_pixels + 1.0) * 0.5).clamp(0, 1)
                            else:
                                noisy_pixels = None
                            for idx in range(x0_pixels.shape[0]):
                                if valid_mask[idx]:
                                    pred_pix = x0_pixels[idx].clamp(0, 1).cpu()
                                    pred_img = TF.to_pil_image(pred_pix)
                                    cos_val = cos_sim[idx].item()
                                    t_val = t_ratio[idx].item()
                                    src_name = os.path.splitext(os.path.basename(batch.file_items[idx].path))[0] if idx < len(batch.file_items) else 'unknown'
                                    ref_emb_norm = ref_embedding[idx].norm().item()
                                    gen_emb_norm = gen_embedding[idx].norm().item()
                                    has_bbox = scaled_bboxes is not None and idx < len(scaled_bboxes) and scaled_bboxes[idx] is not None

                                    # Use the actual ArcFace input crop (not a reconstruction)
                                    crop_img = TF.to_pil_image(_arcface_crops[idx].clamp(0, 1).cpu())

                                    # Build combined: [noisy | x0 | crop_112x112]
                                    from PIL import Image as PILImage, ImageDraw
                                    h = pred_img.height
                                    crop_resized = crop_img.resize((h, h))  # scale crop to match height
                                    panels = [pred_img, crop_resized]
                                    if noisy_pixels is not None:
                                        noisy_img = TF.to_pil_image(noisy_pixels[idx].clamp(0, 1).cpu())
                                        panels.insert(0, noisy_img)
                                    total_w = sum(p.width for p in panels)
                                    combined = PILImage.new('RGB', (total_w, h + 20), color=(0, 0, 0))
                                    x_off = 0
                                    for p in panels:
                                        combined.paste(p, (x_off, 0))
                                        x_off += p.width
                                    # Draw bbox on x0 prediction panel
                                    if has_bbox:
                                        draw = ImageDraw.Draw(combined)
                                        bx_off = panels[0].width if noisy_pixels is not None else 0
                                        bx1c, by1c, bx2c, by2c = [float(v) for v in scaled_bboxes[idx]]
                                        draw.rectangle([bx_off + bx1c, by1c, bx_off + bx2c, by2c], outline='lime', width=2)
                                    # Annotate bottom strip
                                    draw = ImageDraw.Draw(combined)
                                    label = f"cos={cos_val:.3f} t={t_val:.2f} bbox={'Y' if has_bbox else 'N'} ref_n={ref_emb_norm:.3f} gen_n={gen_emb_norm:.3f}"
                                    draw.text((4, h + 2), label, fill='white')
                                    combined.save(os.path.join(id_preview_dir, f'{src_name}_step{self.step_num:06d}_t{t_val:.2f}_cos{cos_val:.3f}.jpg'))

                            # Console log per-sample breakdown every 50 steps
                            if self.step_num % 50 == 0:
                                for idx in range(x0_pixels.shape[0]):
                                    src = os.path.basename(batch.file_items[idx].path) if idx < len(batch.file_items) else '?'
                                    cs = cos_sim[idx].item()
                                    tr = t_ratio[idx].item()
                                    rv = ref_valid[idx].item() if ref_valid is not None else '?'
                                    hm = high_noise_mask[idx].item()
                                    mm = metric_mask[idx].item()
                                    lm = loss_mask[idx].item()
                                    hb = scaled_bboxes is not None and idx < len(scaled_bboxes) and scaled_bboxes[idx] is not None
                                    re_n = ref_embedding[idx].norm().item()
                                    ge_n = gen_embedding[idx].norm().item()
                                    fd = _face_detected[idx].item()
                                    print(f"  ID[{idx}] {src}: cos={cs:.3f} t={tr:.2f} ref_valid={rv} hnm={hm} mm={mm} lm={lm} face={fd} bbox={hb} ref_norm={re_n:.3f} gen_norm={ge_n:.3f}")

                # --- Landmark shape loss (MediaPipe FaceMesh weighted region MSE) ---
                if _need_landmark_loss:
                    lm_weight = t_ratio
                    gen_landmarks = self.landmark_loss_model(x0_pixels, bboxes=scaled_bboxes)  # (B, 478, 2)

                    ref_landmarks = batch.landmark_embedding.to(
                        gen_landmarks.device, dtype=gen_landmarks.dtype)

                    # Valid mask: reference landmarks are non-zero, timestep in window,
                    # and face is recognizable (reuse identity cos_sim if available)
                    lm_valid_mask = (ref_landmarks.abs().sum(dim=(-1, -2)) > 0) & high_noise_mask
                    if cos_sim is not None:
                        lm_cos_threshold = torch.tensor(
                            [v if v is not None else self.face_id_config.identity_loss_min_cos
                             for v in batch.identity_loss_min_cos_list],
                            device=cos_sim.device, dtype=cos_sim.dtype,
                        )
                        lm_valid_mask = lm_valid_mask & (cos_sim.detach() > lm_cos_threshold)

                    # MediaPipe FaceMesh region indices
                    jaw_idx = DifferentiableLandmarkEncoder.FACE_OVAL    # weight 3x
                    mouth_idx = DifferentiableLandmarkEncoder.LIPS       # weight 2x
                    mid_idx = DifferentiableLandmarkEncoder.MIDFACE      # weight 1x

                    lm_loss_per_sample = torch.zeros(gen_landmarks.shape[0],
                                                     device=gen_landmarks.device)
                    _lm_eps = 1e-6  # prevent NaN grad from sqrt(0)
                    for b_idx in range(gen_landmarks.shape[0]):
                        if not lm_valid_mask[b_idx]:
                            continue
                        gen = gen_landmarks[b_idx]   # (478, 2)
                        ref = ref_landmarks[b_idx]   # (478, 2)
                        jaw_loss = (gen[jaw_idx] - ref[jaw_idx]).pow(2).sum(-1).clamp(min=_lm_eps).sqrt().mean()
                        mouth_loss = (gen[mouth_idx] - ref[mouth_idx]).pow(2).sum(-1).clamp(min=_lm_eps).sqrt().mean()
                        mid_loss = (gen[mid_idx] - ref[mid_idx]).pow(2).sum(-1).clamp(min=_lm_eps).sqrt().mean()
                        lm_loss_per_sample[b_idx] = (jaw_loss * 3 + mouth_loss * 2 + mid_loss * 1) / 6.0

                    # Gate by per-sample landmark weights
                    per_sample_lm_w = batch.landmark_loss_weight_list
                    _has_per_ds_lm_w = any(w is not None for w in per_sample_lm_w)
                    if _has_per_ds_lm_w:
                        global_lm_w = self.face_id_config.landmark_loss_weight
                        lm_weights = torch.tensor(
                            [w if w is not None else global_lm_w for w in per_sample_lm_w],
                            device=lm_valid_mask.device, dtype=torch.float32,
                        )
                        lm_valid_mask = lm_valid_mask & (lm_weights > 0)

                    lm_loss_per_sample = lm_loss_per_sample * lm_weight * lm_valid_mask.float()
                    lm_loss = lm_loss_per_sample.sum() / max(lm_valid_mask.sum().item(), 1.0)

                    if lm_valid_mask.any():
                        if _has_per_ds_lm_w:
                            weighted_lm = (lm_loss_per_sample * lm_weights)
                            lm_applied = weighted_lm.sum() / max(lm_valid_mask.sum().item(), 1.0)
                        else:
                            lm_applied = self.face_id_config.landmark_loss_weight * lm_loss
                        loss = loss + lm_applied
                        self._last_landmark_loss_applied = lm_applied.detach().item()
                    with torch.no_grad():
                        if lm_valid_mask.any():
                            raw_lm = lm_loss_per_sample.sum() / lm_valid_mask.sum().item()
                            self._last_landmark_loss = raw_lm.item()
                            # binned shape similarity by timestep (0.1 bands)
                            if self._last_shape_sim_bins is None:
                                self._last_shape_sim_bins = {}
                            for idx in range(lm_loss_per_sample.shape[0]):
                                if lm_valid_mask[idx]:
                                    t_val = t_ratio[idx].item()
                                    bin_start = int(t_val * 10) / 10.0
                                    bin_key = f'shape_sim_t{int(bin_start*100):02d}'
                                    lm_val = lm_loss_per_sample[idx].item()
                                    self._bin_update(
                                        self._last_shape_sim_bins,
                                        bin_key,
                                        lm_val,
                                    )
                                    self._record_sample(
                                        'landmark_loss', lm_val,
                                        t=t_val, idx=idx, batch=batch,
                                    )

                # --- Body proportion loss (ViTPose bone-length ratio matching) ---
                if _need_body_proportion_loss:
                    bp_weight = t_ratio
                    _bp_include_head = getattr(self.face_id_config, 'body_proportion_include_head', False)
                    ref_bp = batch.body_proportion_embedding.to(
                        x0_pixels.device, dtype=x0_pixels.dtype)
                    _bp_n = ref_bp.shape[-1] // 2  # first half = ratios, second half = vis
                    ref_ratios = ref_bp[:, :_bp_n]

                    # Scale person bboxes from original image coords to x0_pixels coords
                    # Same pipeline as face bboxes: orig -> scale -> crop -> x0_pixels
                    scaled_person_bboxes = None
                    if batch.person_bboxes is not None:
                        _, _, px_h, px_w = x0_pixels.shape
                        scaled_person_bboxes = []
                        for idx in range(x0_pixels.shape[0]):
                            raw_bbox = batch.person_bboxes[idx] if idx < len(batch.person_bboxes) else None
                            if raw_bbox is not None and raw_bbox.abs().sum() > 0:
                                fi = batch.file_items[idx]
                                orig_w = float(fi.width)
                                orig_h = float(fi.height)
                                pbx1, pby1, pbx2, pby2 = raw_bbox.float()

                                if getattr(fi, 'flip_x', False):
                                    pbx1, pbx2 = orig_w - pbx2, orig_w - pbx1
                                if getattr(fi, 'flip_y', False):
                                    pby1, pby2 = orig_h - pby2, orig_h - pby1

                                stw = float(getattr(fi, 'scale_to_width', None) or orig_w)
                                sth = float(getattr(fi, 'scale_to_height', None) or orig_h)
                                pbx1 = pbx1 * (stw / orig_w)
                                pby1 = pby1 * (sth / orig_h)
                                pbx2 = pbx2 * (stw / orig_w)
                                pby2 = pby2 * (sth / orig_h)

                                cx = float(getattr(fi, 'crop_x', None) or 0)
                                cy = float(getattr(fi, 'crop_y', None) or 0)
                                cw = float(getattr(fi, 'crop_width', None) or stw)
                                ch = float(getattr(fi, 'crop_height', None) or sth)
                                pbx1 = pbx1 - cx
                                pby1 = pby1 - cy
                                pbx2 = pbx2 - cx
                                pby2 = pby2 - cy

                                if pbx2 <= 0 or pby2 <= 0 or pbx1 >= cw or pby1 >= ch:
                                    scaled_person_bboxes.append(None)
                                    continue

                                pbx1 = pbx1 * (px_w / cw)
                                pby1 = pby1 * (px_h / ch)
                                pbx2 = pbx2 * (px_w / cw)
                                pby2 = pby2 * (px_h / ch)

                                pbx1 = max(0.0, min(pbx1, float(px_w)))
                                pby1 = max(0.0, min(pby1, float(px_h)))
                                pbx2 = max(0.0, min(pbx2, float(px_w)))
                                pby2 = max(0.0, min(pby2, float(px_h)))

                                scaled_person_bboxes.append([pbx1, pby1, pbx2, pby2])
                            else:
                                scaled_person_bboxes.append(None)

                    # ViTPose works best with full images — no person cropping needed
                    gen_ratios, gen_vis = self.body_proportion_model(
                        x0_pixels, ref_ratios=ref_ratios, include_head=_bp_include_head)  # (B, N), (B, N)
                    ref_vis = ref_bp[:, _bp_n:]

                    # Valid mask: reference embedding is non-zero AND timestep in body's own window
                    bp_valid_mask = (ref_bp.abs().sum(dim=-1) > 0) & bp_noise_mask

                    # Build per-sample body proportion weights early — gate valid mask
                    per_sample_bp_w = batch.body_proportion_loss_weight_list
                    _has_per_ds_bp_w = any(w is not None for w in per_sample_bp_w)
                    if _has_per_ds_bp_w:
                        global_bp_w = self.face_id_config.body_proportion_loss_weight
                        bp_weights = torch.tensor(
                            [w if w is not None else global_bp_w for w in per_sample_bp_w],
                            device=bp_valid_mask.device, dtype=torch.float32,
                        )
                        bp_valid_mask = bp_valid_mask & (bp_weights > 0)

                    # Use minimum of reference and generated visibility as weight
                    combined_vis = torch.min(ref_vis, gen_vis)
                    weighted_diff = (gen_ratios - ref_ratios).abs() * combined_vis
                    bp_loss_per_sample = weighted_diff.sum(dim=-1) / combined_vis.sum(dim=-1).clamp(min=1e-6)

                    # Penalize missing keypoints: ref has high confidence (>=0.5) but gen dropped below threshold
                    _bp_vis_threshold = 0.2
                    missing_mask = (ref_vis >= 0.5) & (gen_vis < _bp_vis_threshold)
                    missing_count = missing_mask.float().sum(dim=-1)  # (B,) how many ratios were dropped
                    ref_high_count = (ref_vis >= 0.5).float().sum(dim=-1).clamp(min=1.0)
                    # Penalty = fraction of high-confidence reference ratios that prediction dropped
                    visibility_penalty = missing_count / ref_high_count  # (B,) in [0, 1]
                    bp_loss_per_sample = bp_loss_per_sample + visibility_penalty
                    bp_loss_per_sample = bp_loss_per_sample * bp_weight * bp_valid_mask.float()
                    bp_loss = bp_loss_per_sample.sum() / max(bp_valid_mask.sum().item(), 1.0)

                    if bp_valid_mask.any():
                        if _has_per_ds_bp_w:
                            weighted_bp = (bp_loss_per_sample * bp_weights)
                            bp_applied = weighted_bp.sum() / max(bp_valid_mask.sum().item(), 1.0)
                        else:
                            bp_applied = self.face_id_config.body_proportion_loss_weight * bp_loss
                        loss = loss + bp_applied
                        self._last_body_proportion_loss_applied = bp_applied.detach().item()
                    with torch.no_grad():
                        if bp_valid_mask.any():
                            raw_bp = bp_loss_per_sample.sum() / bp_valid_mask.sum().item()
                            self._last_body_proportion_loss = raw_bp.item()
                            # binned by timestep (0.1 bands)
                            if self._last_bp_sim_bins is None:
                                self._last_bp_sim_bins = {}
                            for idx in range(bp_loss_per_sample.shape[0]):
                                if bp_valid_mask[idx]:
                                    t_val = t_ratio[idx].item()
                                    bin_start = int(t_val * 10) / 10.0
                                    bin_key = f'bp_sim_t{int(bin_start*100):02d}'
                                    bp_val = bp_loss_per_sample[idx].item()
                                    self._bin_update(
                                        self._last_bp_sim_bins,
                                        bin_key,
                                        1.0 - bp_val,
                                    )
                                    self._record_sample(
                                        'body_proportion_loss', bp_val,
                                        t=t_val, idx=idx, batch=batch,
                                    )

                            # Save skeleton preview images (prediction + reference side-by-side)
                            from toolkit.body_id import draw_skeleton_overlay
                            from PIL import Image as PILImage
                            from PIL.ImageOps import exif_transpose
                            bp_preview_dir = os.path.join(self.save_root, 'body_previews')
                            os.makedirs(bp_preview_dir, exist_ok=True)
                            for idx in range(x0_pixels.shape[0]):
                                if bp_valid_mask[idx]:
                                    try:
                                        import dsntnn as _dsntnn
                                        pred_pil = TF.to_pil_image(x0_pixels[idx].clamp(0, 1).cpu())
                                        pw, ph = pred_pil.size
                                        pred_bbox = [0, 0, pw, ph]

                                        # Run ViTPose via HF processor on prediction (correct coords)
                                        pred_inputs = self.body_proportion_model.processor(
                                            images=pred_pil, boxes=[[[0, 0, pw, ph]]], return_tensors="pt"
                                        )
                                        pred_pv = pred_inputs['pixel_values'].to(
                                            device=x0_pixels.device,
                                            dtype=next(self.body_proportion_model.model.parameters()).dtype,
                                        )
                                        pred_out = self.body_proportion_model.model(pred_pv, dataset_index=torch.tensor([0], device=pred_pv.device))
                                        pred_out.heatmaps = pred_out.heatmaps.float()
                                        pred_results = self.body_proportion_model.processor.post_process_pose_estimation(
                                            pred_out, boxes=[[[0, 0, pw, ph]]]
                                        )[0]
                                        if pred_results:
                                            pkp = pred_results[0]['keypoints']
                                            pscores = pred_results[0]['scores']
                                            pkp_norm = torch.zeros(17, 2)
                                            pkp_norm[:, 0] = pkp[:, 0] / pw
                                            pkp_norm[:, 1] = pkp[:, 1] / ph
                                            pred_skeleton = draw_skeleton_overlay(pred_pil, pkp_norm, pscores)
                                        else:
                                            pred_skeleton = pred_pil

                                        # Run ViTPose via HF processor on reference
                                        ref_path = batch.file_items[idx].path
                                        ref_pil = exif_transpose(PILImage.open(ref_path)).convert('RGB')
                                        rw, rh = ref_pil.size
                                        ref_inputs = self.body_proportion_model.processor(
                                            images=ref_pil, boxes=[[[0, 0, rw, rh]]], return_tensors="pt"
                                        )
                                        ref_pv = ref_inputs['pixel_values'].to(
                                            device=x0_pixels.device,
                                            dtype=next(self.body_proportion_model.model.parameters()).dtype,
                                        )
                                        ref_out = self.body_proportion_model.model(ref_pv, dataset_index=torch.tensor([0], device=ref_pv.device))
                                        ref_out.heatmaps = ref_out.heatmaps.float()
                                        ref_results = self.body_proportion_model.processor.post_process_pose_estimation(
                                            ref_out, boxes=[[[0, 0, rw, rh]]]
                                        )[0]
                                        if ref_results:
                                            rkp = ref_results[0]['keypoints']
                                            rscores = ref_results[0]['scores']
                                            rkp_norm = torch.zeros(17, 2)
                                            rkp_norm[:, 0] = rkp[:, 0] / rw
                                            rkp_norm[:, 1] = rkp[:, 1] / rh
                                            ref_pil = ref_pil.resize(pred_pil.size)
                                            ref_skeleton = draw_skeleton_overlay(ref_pil, rkp_norm, rscores)
                                        else:
                                            ref_skeleton = ref_pil.resize(pred_pil.size)

                                        # Side-by-side: reference | prediction
                                        combined = PILImage.new('RGB', (ref_skeleton.width + pred_skeleton.width, max(ref_skeleton.height, pred_skeleton.height)))
                                        combined.paste(ref_skeleton, (0, 0))
                                        combined.paste(pred_skeleton, (ref_skeleton.width, 0))
                                    except Exception as e:
                                        print_acc(f"  body preview failed: {e}")
                                        combined = TF.to_pil_image(x0_pixels[idx].clamp(0, 1).cpu())

                                    t_val = t_ratio[idx].item()
                                    bp_val = bp_loss_per_sample[idx].item()
                                    src_name = os.path.splitext(os.path.basename(batch.file_items[idx].path))[0] if idx < len(batch.file_items) else 'unknown'
                                    combined.save(os.path.join(
                                        bp_preview_dir,
                                        f'{src_name}_step{self.step_num:06d}_t{t_val:.2f}_bp{bp_val:.4f}.jpg'
                                    ))

                # --- Body shape loss (HybrIK SMPL beta matching) ---
                if _need_body_shape_loss and bsh_noise_mask.any():
                    bsh_weight = t_ratio  # timestep weighting (higher noise → more weight)

                    ref_betas = batch.body_shape_embedding.to(
                        x0_pixels.device, dtype=x0_pixels.dtype)

                    # Scale person bboxes (reuse same pattern as body proportion)
                    scaled_person_bboxes_bsh = None
                    if batch.person_bboxes is not None:
                        _, _, px_h, px_w = x0_pixels.shape
                        scaled_person_bboxes_bsh = []
                        for idx in range(x0_pixels.shape[0]):
                            raw_pb = batch.person_bboxes[idx] if idx < len(batch.person_bboxes) else None
                            if raw_pb is not None:
                                fi = batch.file_items[idx]
                                orig_w, orig_h = float(fi.width), float(fi.height)
                                pbx1, pby1, pbx2, pby2 = raw_pb.float()
                                if getattr(fi, 'flip_x', False):
                                    pbx1, pbx2 = orig_w - pbx2, orig_w - pbx1
                                if getattr(fi, 'flip_y', False):
                                    pby1, pby2 = orig_h - pby2, orig_h - pby1
                                stw = float(getattr(fi, 'scale_to_width', None) or orig_w)
                                sth = float(getattr(fi, 'scale_to_height', None) or orig_h)
                                pbx1 = pbx1 * (stw / orig_w)
                                pby1 = pby1 * (sth / orig_h)
                                pbx2 = pbx2 * (stw / orig_w)
                                pby2 = pby2 * (sth / orig_h)
                                cx = float(getattr(fi, 'crop_x', None) or 0)
                                cy = float(getattr(fi, 'crop_y', None) or 0)
                                cw = float(getattr(fi, 'crop_width', None) or stw)
                                ch = float(getattr(fi, 'crop_height', None) or sth)
                                pbx1 -= cx; pby1 -= cy; pbx2 -= cx; pby2 -= cy
                                if pbx2 <= 0 or pby2 <= 0 or pbx1 >= cw or pby1 >= ch:
                                    scaled_person_bboxes_bsh.append(None)
                                    continue
                                pbx1 = max(0.0, pbx1 * (px_w / cw))
                                pby1 = max(0.0, pby1 * (px_h / ch))
                                pbx2 = min(float(px_w), pbx2 * (px_w / cw))
                                pby2 = min(float(px_h), pby2 * (px_h / ch))
                                scaled_person_bboxes_bsh.append([pbx1, pby1, pbx2, pby2])
                            else:
                                scaled_person_bboxes_bsh.append(None)

                    # Run HybrIK on decoded x0
                    gen_betas = self.body_shape_model(
                        x0_pixels, person_bboxes=scaled_person_bboxes_bsh
                    )  # (B, 10)

                    # Metric mask: reference non-zero AND timestep in window
                    bsh_metric_mask = (ref_betas.abs().sum(dim=-1) > 0) & bsh_noise_mask

                    # Cosine similarity for gating and logging
                    bsh_cos = F.cosine_similarity(gen_betas, ref_betas, dim=-1)  # (B,)

                    # Loss mask: also gate by per-sample cosine threshold
                    bsh_cos_threshold = torch.tensor(
                        [v if v is not None else self.face_id_config.body_shape_loss_min_cos
                         for v in batch.body_shape_loss_min_cos_list],
                        device=bsh_cos.device, dtype=bsh_cos.dtype,
                    )
                    bsh_valid_mask = bsh_metric_mask & (bsh_cos.detach() > bsh_cos_threshold)

                    # Gate by per-sample body shape weights
                    per_sample_bsh_w = batch.body_shape_loss_weight_list
                    _has_per_ds_bsh_w = any(w is not None for w in per_sample_bsh_w)
                    if _has_per_ds_bsh_w:
                        global_bsh_w = self.face_id_config.body_shape_loss_weight
                        bsh_weights = torch.tensor(
                            [w if w is not None else global_bsh_w for w in per_sample_bsh_w],
                            device=bsh_valid_mask.device, dtype=torch.float32,
                        )
                        bsh_valid_mask = bsh_valid_mask & (bsh_weights > 0)

                    # L1 loss on 10-dim betas
                    bsh_loss_per_sample = (gen_betas - ref_betas).abs().mean(dim=-1)
                    bsh_loss_per_sample = bsh_loss_per_sample * bsh_weight * bsh_valid_mask.float()
                    bsh_loss = bsh_loss_per_sample.sum() / max(bsh_valid_mask.sum().item(), 1.0)

                    if bsh_valid_mask.any():
                        if _has_per_ds_bsh_w:
                            weighted_bsh = bsh_loss_per_sample * bsh_weights
                            bsh_applied = weighted_bsh.sum() / max(bsh_valid_mask.sum().item(), 1.0)
                        else:
                            bsh_applied = self.face_id_config.body_shape_loss_weight * bsh_loss
                        loss = loss + bsh_applied
                        self._last_body_shape_loss_applied = bsh_applied.detach().item()
                    with torch.no_grad():
                        # Log using metric_mask (no cosine gate) so metrics work even when gated
                        if bsh_metric_mask.any():
                            n_metric = bsh_metric_mask.sum().item()
                            n_gated = bsh_valid_mask.sum().item()

                            # Raw L1 (unweighted, always computed)
                            raw_l1 = (gen_betas - ref_betas).abs().mean(dim=-1)
                            raw_l1_mean = (raw_l1 * bsh_metric_mask.float()).sum() / n_metric
                            self._last_body_shape_l1 = raw_l1_mean.item()

                            # Applied loss
                            raw_bsh = (bsh_loss_per_sample.sum() / n_metric
                                       if n_gated > 0 else 0.0)
                            self._last_body_shape_loss = raw_bsh if isinstance(raw_bsh, float) else raw_bsh.item()

                            # Cosine similarity
                            bsh_cos_mean = (bsh_cos * bsh_metric_mask.float()).sum() / n_metric
                            self._last_body_shape_cos = bsh_cos_mean.item()

                            # Gated percentage
                            self._last_body_shape_gated_pct = n_gated / n_metric

                            # Cosine binned by timestep
                            if self._last_bsh_sim_bins is None:
                                self._last_bsh_sim_bins = {}
                            for idx in range(bsh_cos.shape[0]):
                                if bsh_metric_mask[idx]:
                                    t_val = t_ratio[idx].item()
                                    bin_start = int(t_val * 10) / 10.0
                                    bin_key = f'bsh_sim_t{int(bin_start*100):02d}'
                                    bsh_val = bsh_cos[idx].item()
                                    self._bin_update(
                                        self._last_bsh_sim_bins,
                                        bin_key,
                                        bsh_val,
                                    )
                                    self._record_sample(
                                        'body_shape_cos', bsh_val,
                                        t=t_val, idx=idx, batch=batch,
                                    )

                # --- Normal map loss (Sapiens surface normal matching) ---
                if _need_normal_loss and nrm_noise_mask.any():
                    nrm_weight = t_ratio  # timestep weighting

                    ref_normals = batch.normal_embedding.to(
                        x0_pixels.device, dtype=x0_pixels.dtype)  # (B, 3, 256, 256)

                    # Run Sapiens on decoded x0
                    gen_normals = self.normal_model(x0_pixels)  # (B, 3, 256, 256)

                    # Valid mask: reference non-zero AND timestep in window
                    ref_valid = (ref_normals.abs().sum(dim=1).sum(dim=(1, 2)) > 0)  # (B,)
                    nrm_valid_mask = ref_valid & nrm_noise_mask

                    # Gate by per-sample normal weights
                    per_sample_nrm_w = batch.normal_loss_weight_list
                    _has_per_ds_nrm_w = any(w is not None for w in per_sample_nrm_w)
                    if _has_per_ds_nrm_w:
                        global_nrm_w = self.face_id_config.normal_loss_weight
                        nrm_weights = torch.tensor(
                            [w if w is not None else global_nrm_w for w in per_sample_nrm_w],
                            device=nrm_valid_mask.device, dtype=torch.float32,
                        )
                        nrm_valid_mask = nrm_valid_mask & (nrm_weights > 0)

                    if nrm_valid_mask.any():
                        # Cosine dissimilarity per pixel, averaged spatially
                        cos_per_pixel = (ref_normals * gen_normals).sum(dim=1)  # (B, H, W)

                        # L1 per pixel, averaged spatially
                        l1_per_pixel = (ref_normals - gen_normals).abs().mean(dim=1)  # (B, H, W)

                        # Perceptual restriction to body mask (Phase 2 subject_mask).
                        # If any item requests it, multiply per-pixel error by the body
                        # mask at x0_pixels resolution before spatial averaging.
                        _body_restrict_mask = self._build_body_restrict_mask(
                            batch, cos_per_pixel.shape
                        )
                        if _body_restrict_mask is not None:
                            # normalize per-sample so items not restricted keep ~same magnitude
                            _brm = _body_restrict_mask.to(
                                cos_per_pixel.device, dtype=cos_per_pixel.dtype
                            )
                            # avoid div-by-zero for items with empty body mask
                            _brm_sum = _brm.sum(dim=(1, 2)).clamp(min=1.0)
                            cos_mean = (cos_per_pixel * _brm).sum(dim=(1, 2)) / _brm_sum
                            l1_mean = (l1_per_pixel * _brm).sum(dim=(1, 2)) / _brm_sum
                        else:
                            cos_mean = cos_per_pixel.mean(dim=(1, 2))  # (B,)
                            l1_mean = l1_per_pixel.mean(dim=(1, 2))  # (B,)
                        cos_loss = 1.0 - cos_mean  # (B,)

                        # Combined loss: L1 + cosine dissimilarity (same as Sapiens training)
                        nrm_loss_per_sample = (cos_loss + l1_mean) * nrm_weight * nrm_valid_mask.float()
                        nrm_loss = nrm_loss_per_sample.sum() / max(nrm_valid_mask.sum().item(), 1.0)

                        if _has_per_ds_nrm_w:
                            weighted_nrm = nrm_loss_per_sample * nrm_weights
                            nrm_applied = weighted_nrm.sum() / max(nrm_valid_mask.sum().item(), 1.0)
                        else:
                            nrm_applied = self.face_id_config.normal_loss_weight * nrm_loss
                        loss = loss + nrm_applied
                        self._last_normal_loss_applied = nrm_applied.detach().item()

                    with torch.no_grad():
                        if nrm_valid_mask.any():
                            n_v = nrm_valid_mask.sum().item()
                            cos_per_px = (ref_normals * gen_normals).sum(dim=1)
                            raw_cos = (cos_per_px.mean(dim=(1, 2)) * nrm_valid_mask.float()).sum() / n_v
                            self._last_normal_cos = raw_cos.item()

                            raw_l1 = (ref_normals - gen_normals).abs().mean(dim=(1, 2, 3))
                            raw_l1_mean = (raw_l1 * nrm_valid_mask.float()).sum() / n_v
                            self._last_normal_loss = raw_l1_mean.item()

                            # Save normal preview: ref | gen | x0_pred side by side
                            nrm_preview_dir = os.path.join(self.save_root, 'normal_previews')
                            os.makedirs(nrm_preview_dir, exist_ok=True)
                            for idx in range(gen_normals.shape[0]):
                                if nrm_valid_mask[idx]:
                                    # Normal maps to RGB: (n+1)/2 maps [-1,1] to [0,1]
                                    ref_rgb = ((ref_normals[idx] + 1) * 0.5).clamp(0, 1).cpu()
                                    gen_rgb = ((gen_normals[idx] + 1) * 0.5).clamp(0, 1).cpu()
                                    pred_rgb = x0_pixels[idx].clamp(0, 1).cpu()

                                    ref_img = TF.to_pil_image(ref_rgb)
                                    gen_img = TF.to_pil_image(gen_rgb)
                                    pred_img = TF.to_pil_image(pred_rgb)

                                    # Resize all to same height for comparison
                                    h = ref_img.height
                                    pred_img = pred_img.resize((int(pred_img.width * h / pred_img.height), h))

                                    from PIL import Image as PILImage
                                    total_w = ref_img.width + gen_img.width + pred_img.width + 8
                                    combined = PILImage.new('RGB', (total_w, h))
                                    x_off = 0
                                    combined.paste(ref_img, (x_off, 0)); x_off += ref_img.width + 4
                                    combined.paste(gen_img, (x_off, 0)); x_off += gen_img.width + 4
                                    combined.paste(pred_img, (x_off, 0))

                                    cos_val = cos_per_px[idx].mean().item()
                                    t_val = t_ratio[idx].item()
                                    src_name = os.path.splitext(os.path.basename(
                                        batch.file_items[idx].path))[0] if idx < len(batch.file_items) else 'unknown'
                                    combined.save(os.path.join(
                                        nrm_preview_dir,
                                        f'{src_name}_step{self.step_num:06d}_t{t_val:.2f}_cos{cos_val:.3f}.jpg'
                                    ))

                # --- VAE anchor loss (perceptual feature matching) ---
                if _need_vae_anchor_loss and va_noise_mask.any():
                    va_weight = torch.ones_like(t_ratio)  # uniform — gating window handles timestep filtering

                    # Reference features from batch (cached)
                    ref_features = batch.vae_anchor_features

                    # Check ref features are valid (not zero-padded from failed caching)
                    from toolkit.vae_anchor import FEATURE_LEVELS as _VA_LEVELS
                    ref_valid = torch.stack([
                        ref_features[level].abs().sum(dim=(1, 2, 3)) > 0
                        for level in _VA_LEVELS if level in ref_features
                    ], dim=0).all(dim=0).to(va_noise_mask.device)  # (B,)

                    # Build valid mask: timestep in window AND ref features non-zero
                    va_valid_mask = va_noise_mask & ref_valid

                    # Gate by per-dataset weights
                    per_sample_va_w = batch.vae_anchor_loss_weight_list
                    _has_per_ds_va_w = any(w is not None for w in per_sample_va_w)
                    if _has_per_ds_va_w:
                        global_va_w = self.face_id_config.vae_anchor_loss_weight
                        va_weights = torch.tensor(
                            [w if w is not None else global_va_w for w in per_sample_va_w],
                            device=va_valid_mask.device, dtype=torch.float32,
                        )
                        va_valid_mask = va_valid_mask & (va_weights > 0)

                    if va_valid_mask.any():
                        # SDXL VAE decode (bf16, differentiable) → clean pixels
                        # Flux 2 encoder (f32) → perceptual features for loss
                        # Free cached CUDA blocks first to defragment before large allocations
                        torch.cuda.empty_cache()
                        if next(self.sd.vae.parameters()).device != self.device_torch:
                            self.sd.vae.to(self.device_torch)
                        vae_dtype = self.sd.vae.dtype
                        x0_unscaled = x0_pred / self.sd.vae.config['scaling_factor']
                        if 'shift_factor' in self.sd.vae.config and self.sd.vae.config['shift_factor']:
                            x0_unscaled = x0_unscaled + self.sd.vae.config['shift_factor']
                        x0_vae_pixels = self.sd.vae.decode(x0_unscaled.to(vae_dtype)).sample.clamp(-1, 1)
                        with torch.amp.autocast('cuda', enabled=False):
                            _, pred_features = self.vae_anchor_encoder.encode_with_features(x0_vae_pixels.float())

                        # Compute cosine loss per feature level — returns (B,) per-sample losses
                        va_loss_per_sample, va_per_level = VAEAnchorEncoder.compute_loss(
                            pred_features, ref_features
                        )

                        # Apply timestep weighting and valid mask per-sample
                        va_loss_per_sample = va_loss_per_sample * va_weight * va_valid_mask.float()
                        va_active_count = max(va_valid_mask.sum().item(), 1.0)

                        if _has_per_ds_va_w:
                            weighted_va = va_loss_per_sample * va_weights
                            va_applied = weighted_va.sum() / va_active_count
                        else:
                            va_loss = va_loss_per_sample.sum() / va_active_count
                            va_applied = self.face_id_config.vae_anchor_loss_weight * va_loss
                        loss = loss + va_applied
                        self._last_vae_anchor_loss_applied = va_applied.detach().item()

                        with torch.no_grad():
                            self._last_vae_anchor_loss = (va_loss_per_sample.sum() / va_active_count).item()
                            self._last_vae_anchor_per_level = va_per_level

                            # Save VAE anchor previews (every N steps)
                            _va_preview_every = 500
                            if self.step_num % _va_preview_every == 0:
                                try:
                                    va_preview_dir = os.path.join(self.save_root, 'vae_anchor_previews')
                                    os.makedirs(va_preview_dir, exist_ok=True)
                                    va_preview = (x0_vae_pixels.detach().float() + 1.0) * 0.5
                                    for idx in range(va_preview.shape[0]):
                                        if va_valid_mask[idx]:
                                            pred_rgb = va_preview[idx].clamp(0, 1).cpu()
                                            pred_img = TF.to_pil_image(pred_rgb)

                                            heatmap_imgs = []
                                            for level_name in ['level_1', 'level_2', 'level_3', 'mid']:
                                                if level_name in pred_features and level_name in ref_features:
                                                    pf = pred_features[level_name][idx]
                                                    rf = ref_features[level_name][idx].to(pf.device, dtype=pf.dtype)
                                                    if pf.shape[1:] != rf.shape[1:]:
                                                        rf = F.interpolate(
                                                            rf.unsqueeze(0), size=pf.shape[1:],
                                                            mode='bilinear', align_corners=False
                                                        ).squeeze(0)
                                                    diff = (pf - rf).pow(2).mean(dim=0)
                                                    diff_min = diff.min()
                                                    diff_max = diff.max()
                                                    if diff_max > diff_min:
                                                        diff_norm = (diff - diff_min) / (diff_max - diff_min)
                                                    else:
                                                        diff_norm = torch.zeros_like(diff)
                                                    h_img = diff_norm.cpu().unsqueeze(0).repeat(3, 1, 1)
                                                    h_img[1] = 1.0 - h_img[1]
                                                    h_img[2] = 0.0
                                                    heatmap_pil = TF.to_pil_image(h_img.clamp(0, 1))
                                                    heatmap_pil = heatmap_pil.resize(
                                                        (pred_img.height, pred_img.height)
                                                    )
                                                    heatmap_imgs.append(heatmap_pil)

                                            from PIL import Image as PILImage
                                            panels = [pred_img] + heatmap_imgs
                                            total_w = sum(p.width for p in panels)
                                            h = pred_img.height
                                            combined = PILImage.new('RGB', (total_w, h + 20), color=(0, 0, 0))
                                            x_off = 0
                                            for p in panels:
                                                combined.paste(p, (x_off, 0))
                                                x_off += p.width

                                            from PIL import ImageDraw
                                            draw = ImageDraw.Draw(combined)
                                            t_val = t_ratio[idx].item()
                                            level_str = ' '.join(f'{k}={v:.4f}' for k, v in va_per_level.items())
                                            draw.text((4, h + 2), f't={t_val:.2f} total={va_loss_per_sample[idx].item():.4f} {level_str}', fill='white')
                                            src_name = os.path.splitext(os.path.basename(
                                                batch.file_items[idx].path))[0] if idx < len(batch.file_items) else 'unknown'
                                            combined.save(os.path.join(
                                                va_preview_dir,
                                                f'{src_name}_step{self.step_num:06d}_t{t_val:.2f}_va{va_loss_per_sample[idx].item():.4f}.jpg'
                                            ))
                                except Exception as e:
                                    print(f"Warning: VAE anchor preview failed: {e}")

                        if self.step_num % 50 == 0:
                            level_str = ' '.join(f'{k}={v:.4f}' for k, v in va_per_level.items())
                            print(f"  [VAE anchor] step={self.step_num} total={(va_loss_per_sample.sum() / va_active_count).item():.4f} {level_str}")

        # === Depth consistency loss (MiDaS SSI + multi-scale gradient via DA2) ===
        # Independent of face_id_config; does its own x0 decode and per-sample loop.
        # 4D (image) path — 5D (video) is handled by a separate block below.
        if (self.depth_encoder is not None
                and len(noise_pred.shape) == 4
                and getattr(batch, 'depth_gt_list', None) is not None):
            _dc_cfg = self.depth_consistency_config
            _dc_nt = float(self.sd.noise_scheduler.config.num_train_timesteps)
            _dc_t = timesteps.float() / _dc_nt
            # Per-sample t-band — matches identity-loss pattern: per-item
            # overrides from the batch fall back to the global config.
            _dc_min_list = getattr(batch, 'depth_loss_min_t_list', None) or [None] * _dc_t.shape[0]
            _dc_max_list = getattr(batch, 'depth_loss_max_t_list', None) or [None] * _dc_t.shape[0]
            _dc_min = torch.tensor(
                [v if v is not None else _dc_cfg.loss_min_t for v in _dc_min_list],
                device=_dc_t.device, dtype=_dc_t.dtype,
            )
            _dc_max = torch.tensor(
                [v if v is not None else _dc_cfg.loss_max_t for v in _dc_max_list],
                device=_dc_t.device, dtype=_dc_t.dtype,
            )
            _dc_active = (_dc_t > _dc_min) & (_dc_t < _dc_max)
            # Reg samples don't contribute to depth-consistency loss
            # (they're for prior preservation; their conditioning is
            # stripped, so structural matching to GT is meaningless).
            _dc_active = _dc_active & (~is_reg_per_sample.to(_dc_active.device))
            # Per-sample loss-weight override (None inherits global). Samples
            # with effective weight <= 0 are gated out of the active set so
            # they don't waste compute on the depth perceptor.
            _dc_w_list = getattr(batch, 'depth_loss_weight_list', None) or [None] * _dc_t.shape[0]
            _dc_eff_w = torch.tensor(
                [w if w is not None else _dc_cfg.loss_weight for w in _dc_w_list],
                device=_dc_t.device, dtype=_dc_t.dtype,
            )
            # Loss-split gate: zero depth weight for split samples on
            # diffusion steps. The _dc_active filter below then drops them
            # so the depth perceptor forward is also skipped.
            if loss_split_diff_depth.any() and _step_is_diffusion:
                _split_mask = loss_split_diff_depth.to(_dc_eff_w.device, dtype=_dc_eff_w.dtype)
                _dc_eff_w = _dc_eff_w * (1.0 - _split_mask)
            _dc_active = _dc_active & (_dc_eff_w > 0)

            if _dc_active.any():
                # Recover x0 prediction from model output (same math as the
                # face-losses block — we duplicate it so depth can run with
                # no face_id config present).
                if self.sd.is_flow_matching:
                    _dc_sigma = _dc_t.view(-1, 1, 1, 1)
                    _dc_x0_pred = noisy_latents - _dc_sigma * noise_pred
                else:
                    _dc_ab = self.sd.noise_scheduler.alphas_cumprod.to(
                        device=timesteps.device, dtype=noisy_latents.dtype
                    )[timesteps.long()].view(-1, 1, 1, 1)
                    _dc_sa = _dc_ab.sqrt()
                    _dc_s1ma = (1.0 - _dc_ab).sqrt()
                    if self.sd.prediction_type == 'v_prediction':
                        _dc_x0_pred = _dc_sa * noisy_latents - _dc_s1ma * noise_pred
                    else:
                        _dc_x0_pred = (noisy_latents - _dc_s1ma * noise_pred) / _dc_sa.clamp(min=1e-8)

                # Decode x0 to pixel space (same path the face-losses block uses).
                if hasattr(self, '_taef2_decoder') and self._taef2_decoder is not None:
                    _dc_for_dec = _dc_x0_pred
                    if _dc_for_dec.shape[1] != 32:
                        _dc_for_dec = rearrange(
                            _dc_for_dec,
                            "b (c p1 p2) h w -> b c (h p1) (w p2)",
                            c=32, p1=2, p2=2,
                        )
                    _dc_dd = next(self._taef2_decoder.parameters()).dtype
                    _dc_pixels = self._taef2_decoder(_dc_for_dec.to(_dc_dd)).float().clamp(0, 1)
                elif self.taesd is not None:
                    _dc_td = next(self.taesd.parameters()).dtype
                    _dc_pixels = self.taesd.decode(_dc_x0_pred.to(_dc_td)).sample.float()
                    _dc_pixels = (_dc_pixels + 1.0) * 0.5
                else:
                    # VAE may be offloaded to CPU when latents are cached
                    # (see hook_before_train_loop). Ensure it is on the
                    # training device before decoding.
                    if next(self.sd.vae.parameters()).device != self.device_torch:
                        self.sd.vae.to(self.device_torch)
                    _dc_vd = self.sd.vae.dtype
                    _dc_us = _dc_x0_pred / self.sd.vae.config['scaling_factor']
                    if 'shift_factor' in self.sd.vae.config and self.sd.vae.config['shift_factor']:
                        _dc_us = _dc_us + self.sd.vae.config['shift_factor']
                    _dc_pixels = self.sd.vae.decode(_dc_us.to(_dc_vd)).sample
                    _dc_pixels = (_dc_pixels.float() + 1.0) * 0.5
                _dc_pixels = _dc_pixels.clamp(0, 1)

                # Spatial mask source
                _dc_masks = None
                if _dc_cfg.mask_source == 'subject':
                    _dc_masks = getattr(batch, 'subject_masks', None)
                elif _dc_cfg.mask_source == 'body':
                    _dc_masks = getattr(batch, 'body_masks', None)

                # Iterate per-sample (cached GT depths have per-image shapes).
                # _dc_total is the raw (unweighted) sum used for the metric.
                # _dc_weighted_total carries the per-sample-weighted sum used
                # for the actual gradient contribution (back-prop target).
                _dc_total = _dc_pixels.new_zeros(())
                _dc_weighted_total = _dc_pixels.new_zeros(())
                _dc_ssi_sum = 0.0
                _dc_grad_sum = 0.0
                _dc_n = 0
                for _dc_i in range(_dc_pixels.shape[0]):
                    if not _dc_active[_dc_i]:
                        continue
                    _dc_gt_i = batch.depth_gt_list[_dc_i] if _dc_i < len(batch.depth_gt_list) else None
                    if _dc_gt_i is None:
                        continue
                    _dc_gt_t = _dc_gt_i.to(_dc_pixels.device, dtype=torch.float32)
                    if _dc_gt_t.dim() == 2:
                        _dc_gt_t = _dc_gt_t.unsqueeze(0)
                    _dc_mask_t = None
                    if _dc_masks is not None and _dc_i < _dc_masks.shape[0]:
                        _dc_mask_t = _dc_masks[_dc_i].float().to(_dc_pixels.device)
                        if _dc_mask_t.dim() == 3:
                            _dc_mask_t = _dc_mask_t.squeeze(0)
                    _dc_loss_i, _dc_ssi_i, _dc_grad_i, _dc_dpred_i, _dc_dgt_i = compute_depth_consistency_loss(
                        self.depth_encoder,
                        _dc_pixels[_dc_i:_dc_i + 1],
                        _dc_gt_t,
                        _dc_mask_t,
                        ssi_weight=_dc_cfg.ssi_weight,
                        grad_weight=_dc_cfg.grad_weight,
                        grad_scales=_dc_cfg.grad_scales,
                    )
                    _dc_total = _dc_total + _dc_loss_i
                    _dc_weighted_total = _dc_weighted_total + _dc_loss_i * _dc_eff_w[_dc_i]
                    _dc_ssi_sum += float(_dc_ssi_i)
                    _dc_grad_sum += float(_dc_grad_i)
                    _dc_n += 1

                    # Per-sample depth loss binned by timestep (0.1 bands) —
                    # mirrors identity's id_sim_t00..t90 so the dashboard shows
                    # depth loss evolution per noise level.
                    if self._last_depth_loss_bins is None:
                        self._last_depth_loss_bins = {}
                    _dc_t_val = _dc_t[_dc_i].item()
                    _dc_bin_start = int(_dc_t_val * 10) / 10.0
                    _dc_bin_key = f'depth_loss_t{int(_dc_bin_start*100):02d}'
                    self._bin_update(
                        self._last_depth_loss_bins,
                        _dc_bin_key,
                        float(_dc_loss_i),
                    )
                    self._record_sample(
                        'depth_consistency_loss', float(_dc_loss_i),
                        t=_dc_t_val, idx=_dc_i, batch=batch,
                    )

                    # Preview: save [GT RGB | GT depth | Pred RGB | Pred depth] every N steps.
                    if (_dc_cfg.preview_every > 0
                            and self.step_num % _dc_cfg.preview_every == 0
                            and _dc_i < len(batch.file_items)):
                        try:
                            from toolkit.depth_consistency import render_depth_preview
                            from PIL import Image as _PILImage
                            from PIL.ImageOps import exif_transpose as _exif_transpose
                            pred_rgb = _dc_pixels[_dc_i].detach().clamp(0, 1).cpu()
                            pred_pil = TF.to_pil_image(pred_rgb)
                            ref_path = batch.file_items[_dc_i].path
                            ref_pil = _exif_transpose(_PILImage.open(ref_path)).convert('RGB')
                            combo = render_depth_preview(
                                pred_pil, ref_pil,
                                _dc_dpred_i.squeeze(0) if _dc_dpred_i.dim() == 3 else _dc_dpred_i,
                                _dc_dgt_i.squeeze(0) if _dc_dgt_i.dim() == 3 else _dc_dgt_i,
                                mask=_dc_mask_t,
                            )
                            dc_preview_dir = os.path.join(self.save_root, 'depth_previews')
                            os.makedirs(dc_preview_dir, exist_ok=True)
                            _t_val = _dc_t[_dc_i].item()
                            _dc_val = float(_dc_loss_i)
                            src_name = os.path.splitext(os.path.basename(ref_path))[0]
                            combo.save(os.path.join(
                                dc_preview_dir,
                                f'{src_name}_step{self.step_num:06d}_t{_t_val:.2f}_dc{_dc_val:.4f}.jpg'
                            ))
                        except Exception as e:  # noqa: BLE001
                            print_acc(f"  depth preview failed: {e}")

                if _dc_n > 0:
                    _dc_avg = _dc_total / _dc_n
                    # Per-sample weights are already baked into _dc_weighted_total,
                    # so the applied loss is just its mean — no further global
                    # multiply. When all per-sample weights inherit the global
                    # (the default), this is identical to `global * mean(loss)`.
                    _dc_applied = _dc_weighted_total / _dc_n
                    # Stash the depth-applied tensor (pre-detach) so the
                    # dual-backward block can split the combined loss into
                    # (depth-only) and (everything-else) components.
                    self._dc_applied_for_grad = _dc_applied
                    loss = loss + _dc_applied
                    self._last_depth_consistency_loss = _dc_avg.detach().item()
                    self._last_depth_consistency_loss_applied = _dc_applied.detach().item()
                    self._last_depth_consistency_ssi = _dc_ssi_sum / _dc_n
                    self._last_depth_consistency_grad = _dc_grad_sum / _dc_n

        # === Depth consistency loss — 5D video path (Wan 2.1) ===
        # Decodes x0 with TAEHV (tiny, 11M params) for the full frame count,
        # runs the DA2 perceptor on flattened (B*T, 3, H, W) frames in chunks
        # under gradient checkpointing, then computes per-frame SSI + multi-scale
        # gradient loss against a cached per-frame GT cube.
        if (self.depth_encoder is not None
                and len(noise_pred.shape) == 5
                and getattr(batch, 'depth_gt_video_list', None) is not None):
            _dc_cfg = self.depth_consistency_config
            _dc_nt = float(self.sd.noise_scheduler.config.num_train_timesteps)
            _dc_t = timesteps.float() / _dc_nt
            _dc_min_list = getattr(batch, 'depth_loss_min_t_list', None) or [None] * _dc_t.shape[0]
            _dc_max_list = getattr(batch, 'depth_loss_max_t_list', None) or [None] * _dc_t.shape[0]
            _dc_min = torch.tensor(
                [v if v is not None else _dc_cfg.loss_min_t for v in _dc_min_list],
                device=_dc_t.device, dtype=_dc_t.dtype,
            )
            _dc_max = torch.tensor(
                [v if v is not None else _dc_cfg.loss_max_t for v in _dc_max_list],
                device=_dc_t.device, dtype=_dc_t.dtype,
            )
            _dc_active = (_dc_t > _dc_min) & (_dc_t < _dc_max)
            # Reg samples are excluded from the depth loss (see image path).
            _dc_active = _dc_active & (~is_reg_per_sample.to(_dc_active.device))
            # Per-sample loss-weight override (see image path).
            _dc_w_list = getattr(batch, 'depth_loss_weight_list', None) or [None] * _dc_t.shape[0]
            _dc_eff_w = torch.tensor(
                [w if w is not None else _dc_cfg.loss_weight for w in _dc_w_list],
                device=_dc_t.device, dtype=_dc_t.dtype,
            )
            # Loss-split gate (see image path).
            if loss_split_diff_depth.any() and _step_is_diffusion:
                _split_mask = loss_split_diff_depth.to(_dc_eff_w.device, dtype=_dc_eff_w.dtype)
                _dc_eff_w = _dc_eff_w * (1.0 - _split_mask)
            _dc_active = _dc_active & (_dc_eff_w > 0)

            if _dc_active.any():
                # Lazy-load TAEHV on first video step — keeps image-only runs lean.
                if self._wan_depth_decoder is None:
                    print_acc("DepthConsistency (video): loading TAEHV tiny decoder...")
                    self._wan_depth_decoder = load_taehv_wan21(
                        device=self.device_torch,
                        dtype=get_torch_dtype(self.train_config.dtype),
                    )

                # x0 recovery (flow-matching only — Wan 2.1 uses flowmatch).
                if self.sd.is_flow_matching:
                    _dc_sigma = _dc_t.view(-1, 1, 1, 1, 1)
                    _dc_x0_pred = noisy_latents - _dc_sigma * noise_pred
                else:
                    # Non-flow video is unusual; skip cleanly if it ever happens.
                    _dc_x0_pred = None

                if _dc_x0_pred is not None:
                    # Decode to (B, 3, T, H, W) in [0, 1].
                    _dc_frames = decode_wan_x0_to_frames(
                        _dc_x0_pred, self._wan_depth_decoder
                    )
                    B_vid, _, T_out, H_out, W_out = _dc_frames.shape
                    # (B, 3, T, H, W) → (B, T, 3, H, W) → (B*T, 3, H, W).
                    _dc_flat = _dc_frames.permute(0, 2, 1, 3, 4).reshape(
                        B_vid * T_out, 3, H_out, W_out
                    )

                    # Run DA2 on chunks with gradient checkpointing to cap peak
                    # memory. Chunk size keeps activations bounded regardless of
                    # video length; grad-ckpt frees them across the wrapping
                    # VAE+transformer backward.
                    from torch.utils.checkpoint import checkpoint as _ckpt
                    _dc_chunk = int(getattr(_dc_cfg, 'frames_per_chunk', 8))

                    def _enc_fn(x):
                        return self.depth_encoder(x)

                    _dc_depth_chunks = []
                    for _c in _dc_flat.split(_dc_chunk, dim=0):
                        _dc_depth_chunks.append(
                            _ckpt(_enc_fn, _c, use_reentrant=False)
                        )
                    _dc_depth_flat = torch.cat(_dc_depth_chunks, dim=0)
                    # (B*T, H, W) or (B*T, 1, H, W) — normalize to (B*T, H, W).
                    if _dc_depth_flat.dim() == 4:
                        _dc_depth_flat = _dc_depth_flat.squeeze(1)
                    _dc_depth = _dc_depth_flat.reshape(B_vid, T_out, *_dc_depth_flat.shape[1:])

                    _dc_total = _dc_frames.new_zeros(())
                    _dc_weighted_total = _dc_frames.new_zeros(())
                    _dc_ssi_sum = 0.0
                    _dc_grad_sum = 0.0
                    _dc_n = 0
                    _dc_preview_b = None

                    for _b in range(B_vid):
                        if not _dc_active[_b]:
                            continue
                        _gt_cube = (batch.depth_gt_video_list[_b]
                                    if _b < len(batch.depth_gt_video_list) else None)
                        if _gt_cube is None:
                            continue
                        _gt_cube = _gt_cube.to(_dc_frames.device, dtype=torch.float32)
                        # Align GT T to generated T (should already match when the
                        # cache was built with the dataset's num_frames).
                        T_g = _gt_cube.shape[0]
                        if T_g != T_out:
                            ix = torch.linspace(0, T_g - 1, T_out, device=_gt_cube.device).long()
                            _gt_cube = _gt_cube[ix]

                        # Resize GT depth map to match generated depth spatial size
                        # (DA2 output size may differ if source video had a
                        # different aspect than x0 decode output).
                        if _gt_cube.shape[-2:] != _dc_depth.shape[-2:]:
                            _gt_cube = F.interpolate(
                                _gt_cube.unsqueeze(1),  # (T, 1, H, W)
                                size=_dc_depth.shape[-2:],
                                mode='bilinear',
                                align_corners=False,
                            ).squeeze(1)

                        # Per-frame SSI + multi-scale gradient loss. We feed the
                        # already-extracted generated depth directly via a small
                        # helper below; compute_depth_consistency_loss re-runs
                        # the encoder which would double-compute.
                        from toolkit.depth_consistency import ssi_l1, multiscale_grad_loss
                        gen_d = _dc_depth[_b]  # (T, H, W)
                        # Per-frame scalar loss, then mean over T.
                        ssi_per = []
                        grad_per = []
                        for _t in range(T_out):
                            _s, _, _ = ssi_l1(gen_d[_t], _gt_cube[_t])
                            ssi_per.append(_s)
                            grad_per.append(multiscale_grad_loss(
                                gen_d[_t], _gt_cube[_t], scales=_dc_cfg.grad_scales
                            ))
                        ssi_mean = torch.stack(ssi_per).mean()
                        grad_mean = torch.stack(grad_per).mean()
                        loss_b = _dc_cfg.ssi_weight * ssi_mean + _dc_cfg.grad_weight * grad_mean

                        _dc_total = _dc_total + loss_b
                        _dc_weighted_total = _dc_weighted_total + loss_b * _dc_eff_w[_b]
                        _dc_ssi_sum += float(ssi_mean.detach())
                        _dc_grad_sum += float(grad_mean.detach())
                        _dc_n += 1

                        # Per-sample depth loss binned by timestep (0.1 bands).
                        if self._last_depth_loss_bins is None:
                            self._last_depth_loss_bins = {}
                        _dc_b_t_val = _dc_t[_b].item()
                        _dc_bin_start = int(_dc_b_t_val * 10) / 10.0
                        _dc_bin_key = f'depth_loss_t{int(_dc_bin_start*100):02d}'
                        self._bin_update(
                            self._last_depth_loss_bins,
                            _dc_bin_key,
                            float(loss_b.detach()),
                        )
                        self._record_sample(
                            'depth_consistency_loss', float(loss_b.detach()),
                            t=_dc_b_t_val, idx=_b, batch=batch,
                        )
                        if _dc_preview_b is None:
                            _dc_preview_b = _b

                    if _dc_n > 0:
                        _dc_avg = _dc_total / _dc_n
                        # Per-sample weights are baked into _dc_weighted_total.
                        _dc_applied = _dc_weighted_total / _dc_n
                        # Stash for the dual-backward gradient cosine block.
                        self._dc_applied_for_grad = _dc_applied
                        loss = loss + _dc_applied
                        self._last_depth_consistency_loss = _dc_avg.detach().item()
                        self._last_depth_consistency_loss_applied = _dc_applied.detach().item()
                        self._last_depth_consistency_ssi = _dc_ssi_sum / _dc_n
                        self._last_depth_consistency_grad = _dc_grad_sum / _dc_n

                        # Preview: animated webp [gen_rgb | gen_depth | gt_depth].
                        _dc_preview_min_t = float(getattr(_dc_cfg, 'preview_min_t', 0.0))
                        if (_dc_cfg.preview_every > 0
                                and self.step_num % _dc_cfg.preview_every == 0
                                and _dc_preview_b is not None
                                and _dc_t[_dc_preview_b].item() >= _dc_preview_min_t):
                            try:
                                _gt_cube_p = batch.depth_gt_video_list[_dc_preview_b].to(
                                    dtype=torch.float32
                                )
                                dc_preview_dir = os.path.join(self.save_root, 'depth_previews')
                                os.makedirs(dc_preview_dir, exist_ok=True)
                                _t_val = _dc_t[_dc_preview_b].item()
                                out_path = os.path.join(
                                    dc_preview_dir,
                                    f'step{self.step_num:06d}_t{_t_val:.2f}.webp'
                                )
                                save_video_depth_preview(
                                    out_path,
                                    gen_rgb=_dc_frames[_dc_preview_b].permute(1, 0, 2, 3).detach(),
                                    gen_depth=_dc_depth[_dc_preview_b].detach(),
                                    gt_depth=_gt_cube_p,
                                    fps=16,
                                )
                            except Exception as e:  # noqa: BLE001
                                print_acc(f"  video depth preview failed: {e}")

        # E-LatentLPIPS perceptual loss in latent space
        if self.latent_perceptual_model is not None:
            try:
                lp_loss_raw, lp_loss_applied = self._compute_latent_perceptual_loss(
                    noise_pred=noise_pred, noisy_latents=noisy_latents,
                    timesteps=timesteps, batch=batch,
                )
                if lp_loss_applied is not None:
                    loss = loss + lp_loss_applied
                self._latent_perceptual_loss_accumulator += lp_loss_raw
                self._latent_perceptual_loss_applied_accumulator += (lp_loss_applied.item() if lp_loss_applied is not None else 0.0)
                self._latent_perceptual_accumulation_count += 1
                self._lp_preview_cache = {
                    'noise_pred': noise_pred.detach(), 'noisy_latents': noisy_latents.detach(),
                    'timesteps': timesteps.detach(), 'batch_latents': batch.latents.detach(),
                }
            except Exception as e:
                if self.step_num <= 2:
                    print(f"WARNING: E-LatentLPIPS failed: {e}")
                    print(f"  noise_pred shape: {noise_pred.shape}, noisy_latents shape: {noisy_latents.shape}")
                    print(f"  timesteps: {timesteps}, is_flow_matching: {self.sd.is_flow_matching}")
                    import traceback; traceback.print_exc()

        return loss + additional_loss

    def _compute_latent_perceptual_loss(self, noise_pred, noisy_latents, timesteps, batch):
        """Compute E-LatentLPIPS between x0_pred and target latents."""
        bs = noise_pred.shape[0]
        num_train_ts = float(self.sd.noise_scheduler.config.num_train_timesteps)
        t_01 = (timesteps.float() / num_train_ts).to(noise_pred.device)

        # Per-sample timestep gating (per-dataset overrides fall back to global)
        global_min_t = self.train_config.latent_perceptual_loss_min_t
        global_max_t = self.train_config.latent_perceptual_loss_max_t
        min_vals = torch.tensor(
            [v if v is not None else global_min_t for v in batch.latent_perceptual_loss_min_t_list],
            device=t_01.device, dtype=t_01.dtype,
        )
        max_vals = torch.tensor(
            [v if v is not None else global_max_t for v in batch.latent_perceptual_loss_max_t_list],
            device=t_01.device, dtype=t_01.dtype,
        )
        t_mask = ((t_01 >= min_vals) & (t_01 <= max_vals)).float()
        # Reg samples don't contribute (auxiliary loss; conditioning is
        # stripped, so a perceptual match is meaningless).
        _is_reg_lp = torch.tensor(
            [bool(v) for v in batch.get_is_reg_list()],
            device=t_01.device, dtype=torch.bool,
        )
        t_mask = t_mask * (~_is_reg_lp).float()

        # Per-dataset weight overrides
        global_weight = self.train_config.latent_perceptual_loss_weight
        per_sample_weights = torch.full((bs,), global_weight, device=noise_pred.device)
        for idx, w in enumerate(batch.latent_perceptual_loss_weight_list):
            if w is not None:
                per_sample_weights[idx] = w

        sample_weights = t_mask * per_sample_weights
        if sample_weights.sum().item() < 1e-8:
            return 0.0, None

        # Recover x0_pred from model output
        if self.sd.is_flow_matching:
            # Flow matching: x0 = noisy - t * v_pred
            t_expand = t_01.view(-1, 1, 1, 1)
            if len(noise_pred.shape) == 5:
                t_expand = t_01.view(-1, 1, 1, 1, 1)
            x0_pred = noisy_latents - t_expand * noise_pred
        else:
            # DDPM: recover x0 from epsilon or v prediction
            alphas_cumprod = self.sd.noise_scheduler.alphas_cumprod.to(
                device=timesteps.device, dtype=noisy_latents.dtype
            )
            alpha_bar = alphas_cumprod[timesteps.long()].view(-1, 1, 1, 1)
            sqrt_alpha_bar = alpha_bar.sqrt()
            sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()
            if hasattr(self.sd, 'prediction_type') and self.sd.prediction_type == 'v_prediction':
                x0_pred = sqrt_alpha_bar * noisy_latents - sqrt_one_minus_alpha_bar * noise_pred
            else:
                x0_pred = (noisy_latents - sqrt_one_minus_alpha_bar * noise_pred) / sqrt_alpha_bar.clamp(min=1e-8)

        target_latents = batch.latents.to(noise_pred.device, dtype=noise_pred.dtype).detach()

        # E-LatentLPIPS 'flux' encoder expects 16ch. If latents are packed
        # (e.g. 128ch from Flux 2's 32ch VAE), unpack to spatial format first,
        # then take the first 16 channels as an approximation.
        x0_for_lp = x0_pred
        tgt_for_lp = target_latents
        ch = x0_for_lp.shape[1]
        if ch > 16 and ch % 4 == 0:
            # Unpack pixel-shuffle: (B, C*4, H, W) -> (B, C, H*2, W*2)
            from einops import rearrange
            z_ch = ch // 4  # 128 -> 32, or 64 -> 16
            x0_for_lp = rearrange(x0_for_lp, "b (c p1 p2) h w -> b c (h p1) (w p2)", c=z_ch, p1=2, p2=2)
            tgt_for_lp = rearrange(tgt_for_lp, "b (c p1 p2) h w -> b c (h p1) (w p2)", c=z_ch, p1=2, p2=2)
        # If still > 16ch (e.g. 32ch Flux 2), take first 16
        if x0_for_lp.shape[1] > 16:
            x0_for_lp = x0_for_lp[:, :16]
            tgt_for_lp = tgt_for_lp[:, :16]

        lp_loss = self.latent_perceptual_model(
            x0_for_lp.float(), tgt_for_lp.float(),
            normalize=False, add_l1_loss=True, ensembling=False,
        ).view(bs)

        raw_loss = lp_loss.detach().mean().item()
        weighted_loss = (lp_loss * sample_weights).mean()
        return raw_loss, weighted_loss

    def _save_latent_perceptual_preview(self, noise_pred, noisy_latents, timesteps, batch_latents):
        """Save diagnostic preview: heatmap + decoded x0_pred + decoded target."""
        preview_dir = os.path.join(self.save_root, 'latent_perceptual_previews')
        os.makedirs(preview_dir, exist_ok=True)

        num_train_ts = float(self.sd.noise_scheduler.config.num_train_timesteps)
        t_01 = (timesteps.float() / num_train_ts).to(noise_pred.device)
        t_expand = t_01.view(-1, 1, 1, 1)
        if len(noise_pred.shape) == 5:
            t_expand = t_01.view(-1, 1, 1, 1, 1)

        with torch.no_grad():
            x0_pred = noisy_latents - t_expand * noise_pred
            target_latents = batch_latents.to(noise_pred.device, dtype=noise_pred.dtype)

            # Per-spatial L2 diff heatmap
            diff = (x0_pred[:1].float() - target_latents[:1].float())
            spatial_diff = diff.pow(2).mean(dim=1)[0]  # (H, W)
            t_val = t_01[0].item()

            diff_np = spatial_diff.sqrt().cpu().numpy()
            diff_max = diff_np.max() if diff_np.max() > 0 else 1.0
            heatmap = Image.fromarray((diff_np / diff_max * 255).astype(np.uint8), mode='L')

            step_str = f'{self.step_num:09d}'
            heatmap.save(os.path.join(preview_dir, f'step_{step_str}_t{t_val:.3f}_heatmap.png'))

            # Per-channel diagnostics
            print(f"[Latent Perceptual Preview] step={self.step_num}, t={t_val:.3f}")
            for ch in range(min(diff.shape[1], 16)):
                ch_diff = diff[0, ch]
                print(f"  ch{ch:2d}: mean_diff={ch_diff.mean().item():+.4f}, "
                      f"std_diff={ch_diff.std().item():.4f}, "
                      f"max_abs={ch_diff.abs().max().item():.4f}")

    def preprocess_batch(self, batch: 'DataLoaderBatchDTO'):
        return batch

    def get_guided_loss(
            self,
            noisy_latents: torch.Tensor,
            conditional_embeds: PromptEmbeds,
            match_adapter_assist: bool,
            network_weight_list: list,
            timesteps: torch.Tensor,
            pred_kwargs: dict,
            batch: 'DataLoaderBatchDTO',
            noise: torch.Tensor,
            unconditional_embeds: Optional[PromptEmbeds] = None,
            **kwargs
    ):
        loss = get_guidance_loss(
            noisy_latents=noisy_latents,
            conditional_embeds=conditional_embeds,
            match_adapter_assist=match_adapter_assist,
            network_weight_list=network_weight_list,
            timesteps=timesteps,
            pred_kwargs=pred_kwargs,
            batch=batch,
            noise=noise,
            sd=self.sd,
            unconditional_embeds=unconditional_embeds,
            train_config=self.train_config,
            **kwargs
        )

        return loss
    
    
    # ------------------------------------------------------------------
    #  Mean-Flow loss (Geng et al., “Mean Flows for One-step Generative
    #  Modelling”, 2025 – see Alg. 1 + Eq. (6) of the paper)
    # This version avoids jvp / double-back-prop issues with Flash-Attention
    # adapted from the work of lodestonerock
    # ------------------------------------------------------------------
    def get_mean_flow_loss(
            self,
            noisy_latents: torch.Tensor,
            conditional_embeds: PromptEmbeds,
            match_adapter_assist: bool,
            network_weight_list: list,
            timesteps: torch.Tensor,
            pred_kwargs: dict,
            batch: 'DataLoaderBatchDTO',
            noise: torch.Tensor,
            unconditional_embeds: Optional[PromptEmbeds] = None,
            **kwargs
    ):
        dtype = get_torch_dtype(self.train_config.dtype)
        total_steps = float(self.sd.noise_scheduler.config.num_train_timesteps)  # e.g. 1000
        base_eps = 1e-3
        min_time_gap = 1e-2
        
        with torch.no_grad():
            num_train_timesteps = self.sd.noise_scheduler.config.num_train_timesteps
            batch_size = batch.latents.shape[0]
            timestep_t_list = []
            timestep_r_list = []

            for i in range(batch_size):
                t1 = random.randint(0, num_train_timesteps - 1)
                t2 = random.randint(0, num_train_timesteps - 1)
                t_t = self.sd.noise_scheduler.timesteps[min(t1, t2)]
                t_r = self.sd.noise_scheduler.timesteps[max(t1, t2)]
                if (t_t - t_r).item() < min_time_gap * 1000:
                    scaled_time_gap = min_time_gap * 1000
                    if t_t.item() + scaled_time_gap > 1000:
                        t_r = t_r - scaled_time_gap
                    else:
                        t_t = t_t + scaled_time_gap
                timestep_t_list.append(t_t)
                timestep_r_list.append(t_r)

            timesteps_t = torch.stack(timestep_t_list, dim=0).float()
            timesteps_r = torch.stack(timestep_r_list, dim=0).float()

            t_frac = timesteps_t / total_steps  # [0,1]
            r_frac = timesteps_r / total_steps  # [0,1]

            latents_clean = batch.latents.to(dtype)
            noise_sample = noise.to(dtype)

            lerp_vector = latents_clean * (1.0 - t_frac[:, None, None, None]) + noise_sample * t_frac[:, None, None, None]

            eps = base_eps

            # concatenate timesteps as input for u(z, r, t)
            timesteps_cat = torch.cat([t_frac, r_frac], dim=0) * total_steps

        # model predicts u(z, r, t)
        u_pred = self.predict_noise(
            noisy_latents=lerp_vector.to(dtype),
            timesteps=timesteps_cat.to(dtype),
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            batch=batch,
            **pred_kwargs
        )

        with torch.no_grad():
            t_frac_plus_eps = (t_frac + eps).clamp(0.0, 1.0)
            lerp_perturbed = latents_clean * (1.0 - t_frac_plus_eps[:, None, None, None]) + noise_sample * t_frac_plus_eps[:, None, None, None]
            timesteps_cat_perturbed = torch.cat([t_frac_plus_eps, r_frac], dim=0) * total_steps

            u_perturbed = self.predict_noise(
                noisy_latents=lerp_perturbed.to(dtype),
                timesteps=timesteps_cat_perturbed.to(dtype),
                conditional_embeds=conditional_embeds,
                unconditional_embeds=unconditional_embeds,
                batch=batch,
                **pred_kwargs
            )

        # compute du/dt via finite difference (detached)
        du_dt = (u_perturbed - u_pred).detach() / eps
        # du_dt = (u_perturbed - u_pred).detach()
        du_dt = du_dt.to(dtype)
        
        
        time_gap = (t_frac - r_frac)[:, None, None, None].to(dtype)
        time_gap.clamp(min=1e-4)
        u_shifted = u_pred + time_gap * du_dt
        # u_shifted = u_pred + du_dt / time_gap
        # u_shifted = u_pred

        # a step is done like this:
        # stepped_latent = model_input + (timestep_next - timestep) * model_output
        
        # flow target velocity
        # v_target = (noise_sample - latents_clean) / time_gap
        # flux predicts opposite of velocity, so we need to invert it
        v_target = (latents_clean - noise_sample) / time_gap

        # compute loss
        loss = torch.nn.functional.mse_loss(
            u_shifted.float(),
            v_target.float(),
            reduction='none'
        )

        with torch.no_grad():
            pure_loss = loss.mean().detach()
            pure_loss.requires_grad_(True)

        loss = loss.mean()
        if loss.item() > 1e3:
            pass
        self.accelerator.backward(loss)
        return pure_loss



    def get_prior_prediction(
            self,
            noisy_latents: torch.Tensor,
            conditional_embeds: PromptEmbeds,
            match_adapter_assist: bool,
            network_weight_list: list,
            timesteps: torch.Tensor,
            pred_kwargs: dict,
            batch: 'DataLoaderBatchDTO',
            noise: torch.Tensor,
            unconditional_embeds: Optional[PromptEmbeds] = None,
            conditioned_prompts=None,
            **kwargs
    ):
        # todo for embeddings, we need to run without trigger words
        was_unet_training = self.sd.unet.training
        was_network_active = False
        if self.network is not None:
            was_network_active = self.network.is_active
            self.network.is_active = False
        can_disable_adapter = False
        was_adapter_active = False
        if self.adapter is not None and (isinstance(self.adapter, IPAdapter) or
                                         isinstance(self.adapter, ReferenceAdapter) or
                                         (isinstance(self.adapter, CustomAdapter))
        ):
            can_disable_adapter = True
            was_adapter_active = self.adapter.is_active
            self.adapter.is_active = False

        if self.train_config.unload_text_encoder and self.adapter is not None and not isinstance(self.adapter, CustomAdapter):
            raise ValueError("Prior predictions currently do not support unloading text encoder with adapter")
        # do a prediction here so we can match its output with network multiplier set to 0.0
        with torch.no_grad():
            dtype = get_torch_dtype(self.train_config.dtype)

            embeds_to_use = conditional_embeds.clone().detach()
            # handle clip vision adapter by removing triggers from prompt and replacing with the class name
            if (self.adapter is not None and isinstance(self.adapter, ClipVisionAdapter)) or self.embedding is not None:
                prompt_list = batch.get_caption_list()
                class_name = ''

                triggers = ['[trigger]', '[name]']
                remove_tokens = []

                if self.embed_config is not None:
                    triggers.append(self.embed_config.trigger)
                    for i in range(1, self.embed_config.tokens):
                        remove_tokens.append(f"{self.embed_config.trigger}_{i}")
                    if self.embed_config.trigger_class_name is not None:
                        class_name = self.embed_config.trigger_class_name

                if self.adapter is not None:
                    triggers.append(self.adapter_config.trigger)
                    for i in range(1, self.adapter_config.num_tokens):
                        remove_tokens.append(f"{self.adapter_config.trigger}_{i}")
                    if self.adapter_config.trigger_class_name is not None:
                        class_name = self.adapter_config.trigger_class_name

                for idx, prompt in enumerate(prompt_list):
                    for remove_token in remove_tokens:
                        prompt = prompt.replace(remove_token, '')
                    for trigger in triggers:
                        prompt = prompt.replace(trigger, class_name)
                    prompt_list[idx] = prompt

                if batch.prompt_embeds is not None:
                    embeds_to_use = batch.prompt_embeds.clone().to(self.device_torch, dtype=dtype)
                else:
                    prompt_kwargs = {}
                    if self.sd.encode_control_in_text_embeddings and batch.control_tensor is not None:
                        prompt_kwargs['control_images'] = batch.control_tensor.to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                    embeds_to_use = self.sd.encode_prompt(
                        prompt_list,
                        long_prompts=self.do_long_prompts).to(
                        self.device_torch,
                        dtype=dtype,
                        **prompt_kwargs
                    ).detach()

            # dont use network on this
            # self.network.multiplier = 0.0
            self.sd.unet.eval()

            if self.adapter is not None and isinstance(self.adapter, IPAdapter) and not self.sd.is_flux and not self.sd.is_lumina2:
                # we need to remove the image embeds from the prompt except for flux
                embeds_to_use: PromptEmbeds = embeds_to_use.clone().detach()
                end_pos = embeds_to_use.text_embeds.shape[1] - self.adapter_config.num_tokens
                embeds_to_use.text_embeds = embeds_to_use.text_embeds[:, :end_pos, :]
                if unconditional_embeds is not None:
                    unconditional_embeds = unconditional_embeds.clone().detach()
                    unconditional_embeds.text_embeds = unconditional_embeds.text_embeds[:, :end_pos]

            if unconditional_embeds is not None:
                unconditional_embeds = unconditional_embeds.to(self.device_torch, dtype=dtype).detach()
            
            guidance_embedding_scale = self.train_config.cfg_scale
            if self.train_config.do_guidance_loss:
                guidance_embedding_scale = self._guidance_loss_target_batch

            prior_pred = self.sd.predict_noise(
                latents=noisy_latents.to(self.device_torch, dtype=dtype).detach(),
                conditional_embeddings=embeds_to_use.to(self.device_torch, dtype=dtype).detach(),
                unconditional_embeddings=unconditional_embeds,
                timestep=timesteps,
                guidance_scale=self.train_config.cfg_scale,
                guidance_embedding_scale=guidance_embedding_scale,
                rescale_cfg=self.train_config.cfg_rescale,
                batch=batch,
                **pred_kwargs  # adapter residuals in here
            )
            if was_unet_training:
                self.sd.unet.train()
            prior_pred = prior_pred.detach()
            # remove the residuals as we wont use them on prediction when matching control
            if match_adapter_assist and 'down_intrablock_additional_residuals' in pred_kwargs:
                del pred_kwargs['down_intrablock_additional_residuals']
            if match_adapter_assist and 'down_block_additional_residuals' in pred_kwargs:
                del pred_kwargs['down_block_additional_residuals']
            if match_adapter_assist and 'mid_block_additional_residual' in pred_kwargs:
                del pred_kwargs['mid_block_additional_residual']

            if can_disable_adapter:
                self.adapter.is_active = was_adapter_active
            # restore network
            # self.network.multiplier = network_weight_list
            if self.network is not None:
                self.network.is_active = was_network_active
        return prior_pred

    def before_unet_predict(self):
        pass

    def after_unet_predict(self):
        pass

    def end_of_training_loop(self):
        pass

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timesteps: Union[int, torch.Tensor] = 1,
        conditional_embeds: Union[PromptEmbeds, None] = None,
        unconditional_embeds: Union[PromptEmbeds, None] = None,
        batch: Optional['DataLoaderBatchDTO'] = None,
        is_primary_pred: bool = False,
        **kwargs,
    ):
        dtype = get_torch_dtype(self.train_config.dtype)
        guidance_embedding_scale = self.train_config.cfg_scale
        if self.train_config.do_guidance_loss:
            guidance_embedding_scale = self._guidance_loss_target_batch
        return self.sd.predict_noise(
            latents=noisy_latents.to(self.device_torch, dtype=dtype),
            conditional_embeddings=conditional_embeds.to(self.device_torch, dtype=dtype),
            unconditional_embeddings=unconditional_embeds,
            timestep=timesteps,
            guidance_scale=self.train_config.cfg_scale,
            guidance_embedding_scale=guidance_embedding_scale,
            detach_unconditional=False,
            rescale_cfg=self.train_config.cfg_rescale,
            bypass_guidance_embedding=self.train_config.bypass_guidance_embedding,
            batch=batch,
            **kwargs
        )
    

    def train_single_accumulation(self, batch: DataLoaderBatchDTO):
        with torch.no_grad():
            self.timer.start('preprocess_batch')
            if isinstance(self.adapter, CustomAdapter):
                batch = self.adapter.edit_batch_raw(batch)
            batch = self.preprocess_batch(batch)
            if isinstance(self.adapter, CustomAdapter):
                batch = self.adapter.edit_batch_processed(batch)
            dtype = get_torch_dtype(self.train_config.dtype)
            # sanity check
            if self.sd.vae.dtype != self.sd.vae_torch_dtype:
                self.sd.vae = self.sd.vae.to(self.sd.vae_torch_dtype)
            if isinstance(self.sd.text_encoder, list):
                for encoder in self.sd.text_encoder:
                    if encoder.dtype != self.sd.te_torch_dtype:
                        encoder.to(self.sd.te_torch_dtype)
            else:
                if self.sd.text_encoder.dtype != self.sd.te_torch_dtype:
                    self.sd.text_encoder.to(self.sd.te_torch_dtype)

            noisy_latents, noise, timesteps, conditioned_prompts, imgs = self.process_general_training_batch(batch)
            if self.train_config.do_cfg or self.train_config.do_random_cfg:
                # pick random negative prompts
                if self.negative_prompt_pool is not None:
                    negative_prompts = []
                    for i in range(noisy_latents.shape[0]):
                        num_neg = random.randint(1, self.train_config.max_negative_prompts)
                        this_neg_prompts = [random.choice(self.negative_prompt_pool) for _ in range(num_neg)]
                        this_neg_prompt = ', '.join(this_neg_prompts)
                        negative_prompts.append(this_neg_prompt)
                    self.batch_negative_prompt = negative_prompts
                else:
                    self.batch_negative_prompt = ['' for _ in range(batch.latents.shape[0])]

            if self.adapter and isinstance(self.adapter, CustomAdapter):
                # condition the prompt
                # todo handle more than one adapter image
                conditioned_prompts = self.adapter.condition_prompt(conditioned_prompts)

            network_weight_list = batch.get_network_weight_list()
            if self.train_config.single_item_batching:
                network_weight_list = network_weight_list + network_weight_list

            has_adapter_img = batch.control_tensor is not None
            has_clip_image = batch.clip_image_tensor is not None
            has_clip_image_embeds = batch.clip_image_embeds is not None
            # force it to be true if doing regs as we handle those differently
            if any([batch.file_items[idx].is_reg for idx in range(len(batch.file_items))]):
                has_clip_image = True
                if self._clip_image_embeds_unconditional is not None:
                    has_clip_image_embeds = True  # we are caching embeds, handle that differently
                    has_clip_image = False

            # do prior pred if prior regularization batch
            do_reg_prior = False
            if any([batch.file_items[idx].prior_reg for idx in range(len(batch.file_items))]):
                do_reg_prior = True

            if self.adapter is not None and isinstance(self.adapter, IPAdapter) and not has_clip_image and has_adapter_img:
                raise ValueError(
                    "IPAdapter control image is now 'clip_image_path' instead of 'control_path'. Please update your dataset config ")

            match_adapter_assist = False

            # check if we are matching the adapter assistant
            if self.assistant_adapter:
                if self.train_config.match_adapter_chance == 1.0:
                    match_adapter_assist = True
                elif self.train_config.match_adapter_chance > 0.0:
                    match_adapter_assist = torch.rand(
                        (1,), device=self.device_torch, dtype=dtype
                    ) < self.train_config.match_adapter_chance

            self.timer.stop('preprocess_batch')

            is_reg = False
            loss_multiplier = torch.ones((noisy_latents.shape[0], 1, 1, 1), device=self.device_torch, dtype=dtype)
            for idx, file_item in enumerate(batch.file_items):
                if file_item.is_reg:
                    loss_multiplier[idx] = loss_multiplier[idx] * self.train_config.reg_weight
                    is_reg = True

            adapter_images = None
            sigmas = None
            if has_adapter_img and (self.adapter or self.assistant_adapter):
                with self.timer('get_adapter_images'):
                    # todo move this to data loader
                    if batch.control_tensor is not None:
                        adapter_images = batch.control_tensor.to(self.device_torch, dtype=dtype).detach()
                        # match in channels
                        if self.assistant_adapter is not None:
                            in_channels = self.assistant_adapter.config.in_channels
                            if adapter_images.shape[1] != in_channels:
                                # we need to match the channels
                                adapter_images = adapter_images[:, :in_channels, :, :]
                    else:
                        raise NotImplementedError("Adapter images now must be loaded with dataloader")

            clip_images = None
            if has_clip_image:
                with self.timer('get_clip_images'):
                    # todo move this to data loader
                    if batch.clip_image_tensor is not None:
                        clip_images = batch.clip_image_tensor.to(self.device_torch, dtype=dtype).detach()

            mask_multiplier = torch.ones((noisy_latents.shape[0], 1, 1, 1), device=self.device_torch, dtype=dtype)
            if batch.mask_tensor is not None:
                with self.timer('get_mask_multiplier'):
                    # upsampling no supported for bfloat16
                    mask_multiplier = batch.mask_tensor.to(self.device_torch, dtype=torch.float16).detach()
                    # scale down to the size of the latents, mask multiplier shape(bs, 1, width, height), noisy_latents shape(bs, channels, width, height)
                    if len(noisy_latents.shape) == 5:
                        # video B,C,T,H,W
                        h = noisy_latents.shape[3]
                        w = noisy_latents.shape[4]
                    else:
                        h = noisy_latents.shape[2]
                        w = noisy_latents.shape[3]
                    mask_multiplier = torch.nn.functional.interpolate(
                        mask_multiplier, size=(h, w)
                    )
                    # expand to match latents
                    mask_multiplier = mask_multiplier.expand(-1, noisy_latents.shape[1], -1, -1)
                    mask_multiplier = mask_multiplier.to(self.device_torch, dtype=dtype).detach()
                    # make avg 1.0
                    mask_multiplier = mask_multiplier / mask_multiplier.mean()

            # Face suppression mask: downweight loss in detected face bounding boxes
            # Resolve per-sample None → global face_id_config fallback
            _global_fsw = getattr(self.face_id_config, 'face_suppression_weight', None) if self.face_id_config else None
            _global_fse = getattr(self.face_id_config, 'face_suppression_expand', 2.0) if self.face_id_config else 2.0
            _global_fss = getattr(self.face_id_config, 'face_suppression_soft', False) if self.face_id_config else False
            _resolved_fsw_list = [
                (w if w is not None else _global_fsw) for w in batch.face_suppression_weight_list
            ]
            _resolved_fse_list = [
                (e if e is not None else _global_fse) for e in batch.face_suppression_expand_list
            ]
            _resolved_fss_list = [
                (s if s is not None else _global_fss) for s in batch.face_suppression_soft_list
            ]
            if any(w is not None and w > 0.0 for w in _resolved_fsw_list):
                if batch.face_bboxes is not None:
                    if len(noisy_latents.shape) == 5:
                        lat_h, lat_w = noisy_latents.shape[3], noisy_latents.shape[4]
                    else:
                        lat_h, lat_w = noisy_latents.shape[2], noisy_latents.shape[3]
                    face_supp_mask = torch.ones(
                        (noisy_latents.shape[0], 1, lat_h, lat_w),
                        device=self.device_torch, dtype=dtype,
                    )
                    for idx in range(noisy_latents.shape[0]):
                        fsw = _resolved_fsw_list[idx]
                        if fsw is None or fsw <= 0.0:
                            continue
                        # Invert: user-facing 0=no suppression, 1=full suppression
                        # Internal mask needs 0=zero loss, 1=normal
                        supp_val = 1.0 - fsw
                        expand = _resolved_fse_list[idx]
                        soft = _resolved_fss_list[idx]
                        raw_bbox = batch.face_bboxes[idx] if idx < len(batch.face_bboxes) else None
                        if raw_bbox is None:
                            continue
                        fi = batch.file_items[idx]
                        orig_w = float(fi.width)
                        orig_h = float(fi.height)
                        bx1, by1, bx2, by2 = [float(v) for v in raw_bbox]
                        # Apply dataloader flips in raw coords before scale+crop.
                        if getattr(fi, 'flip_x', False):
                            bx1, bx2 = orig_w - bx2, orig_w - bx1
                        if getattr(fi, 'flip_y', False):
                            by1, by2 = orig_h - by2, orig_h - by1
                        # scale to resized coords
                        stw = float(getattr(fi, 'scale_to_width', None) or orig_w)
                        sth = float(getattr(fi, 'scale_to_height', None) or orig_h)
                        bx1 *= stw / orig_w; by1 *= sth / orig_h
                        bx2 *= stw / orig_w; by2 *= sth / orig_h
                        # crop offset
                        cx = float(getattr(fi, 'crop_x', None) or 0)
                        cy = float(getattr(fi, 'crop_y', None) or 0)
                        cw = float(getattr(fi, 'crop_width', None) or stw)
                        ch = float(getattr(fi, 'crop_height', None) or sth)
                        bx1 -= cx; by1 -= cy; bx2 -= cx; by2 -= cy
                        # skip if face is outside crop
                        if bx2 <= 0 or by2 <= 0 or bx1 >= cw or by1 >= ch:
                            continue

                        # Asymmetric bbox expansion for head coverage
                        face_w = bx2 - bx1
                        face_h = by2 - by1
                        if expand > 1.0:
                            # Up: full expansion (crown/hair extend far above face box)
                            exp_up = (expand - 1.0) * face_h * 1.0
                            # Sides: moderate (hair volume)
                            exp_side = (expand - 1.0) * face_w * 0.5
                            # Down: less (chin to neck)
                            exp_down = (expand - 1.0) * face_h * 0.3
                            ebx1 = bx1 - exp_side
                            eby1 = by1 - exp_up
                            ebx2 = bx2 + exp_side
                            eby2 = by2 + exp_down
                            # clamp to crop bounds
                            ebx1 = max(0.0, ebx1); eby1 = max(0.0, eby1)
                            ebx2 = min(cw, ebx2);   eby2 = min(ch, eby2)
                        else:
                            ebx1, eby1, ebx2, eby2 = bx1, by1, bx2, by2

                        # to latent coords (expanded bbox)
                        lbx1 = ebx1 * lat_w / cw; lbx2 = ebx2 * lat_w / cw
                        lby1 = eby1 * lat_h / ch; lby2 = eby2 * lat_h / ch

                        if soft and expand > 1.0:
                            # Gaussian falloff mask centered on original face bbox center
                            center_x = (bx1 + bx2) * 0.5 * lat_w / cw
                            center_y = (by1 + by2) * 0.5 * lat_h / ch
                            expanded_w_lat = lbx2 - lbx1
                            expanded_h_lat = lby2 - lby1
                            sigma_x = max(expanded_w_lat / 4.0, 0.5)
                            sigma_y = max(expanded_h_lat / 4.0, 0.5)
                            # Build 2D Gaussian on the latent grid
                            yy = torch.arange(lat_h, device=self.device_torch, dtype=dtype) + 0.5
                            xx = torch.arange(lat_w, device=self.device_torch, dtype=dtype) + 0.5
                            gy = torch.exp(-0.5 * ((yy - center_y) / sigma_y) ** 2)
                            gx = torch.exp(-0.5 * ((xx - center_x) / sigma_x) ** 2)
                            gaussian_2d = gy.unsqueeze(1) * gx.unsqueeze(0)  # (lat_h, lat_w)
                            # gaussian_2d is 1.0 at center, falls off to ~0
                            # mask_value = 1.0 - fsw * gaussian_value
                            face_supp_mask[idx, 0] = 1.0 - fsw * gaussian_2d
                        else:
                            # Hard rectangle (original behavior or expanded hard rect)
                            x1 = max(0, int(lbx1)); y1 = max(0, int(lby1))
                            x2 = min(lat_w, int(lbx2) + 1); y2 = min(lat_h, int(lby2) + 1)
                            if x2 > x1 and y2 > y1:
                                face_supp_mask[idx, :, y1:y2, x1:x2] = supp_val
                    # Save suppression mask overlay on reference image every 50 steps
                    if self.step_num % 50 == 0:
                        try:
                            from PIL import Image
                            import numpy as np
                            supp_preview_dir = os.path.join(self.save_root, 'suppression_previews')
                            os.makedirs(supp_preview_dir, exist_ok=True)
                            for idx in range(face_supp_mask.shape[0]):
                                fi = batch.file_items[idx]
                                ref_img = Image.open(fi.path).convert('RGB')
                                ref_w, ref_h = ref_img.size
                                # face_supp_mask is (B, 1, lat_h, lat_w), values: 1.0=normal, low=suppressed
                                mask_np = face_supp_mask[idx, 0].detach().float().cpu().numpy()
                                # Resize mask to match reference image
                                mask_pil = Image.fromarray((mask_np * 255).clip(0, 255).astype(np.uint8))
                                mask_resized = mask_pil.resize((ref_w, ref_h), Image.NEAREST)
                                mask_arr = np.array(mask_resized).astype(np.float32) / 255.0
                                # Overlay: red tint where suppressed (mask < 1)
                                ref_arr = np.array(ref_img).astype(np.float32)
                                suppression = 1.0 - mask_arr  # 1.0 = fully suppressed, 0 = normal
                                # Blend: original * mask + red * (1 - mask)
                                ref_arr[..., 0] = ref_arr[..., 0] * (1.0 - suppression * 0.6) + 255 * suppression * 0.6
                                ref_arr[..., 1] = ref_arr[..., 1] * (1.0 - suppression * 0.6)
                                ref_arr[..., 2] = ref_arr[..., 2] * (1.0 - suppression * 0.6)
                                src_name = os.path.splitext(os.path.basename(fi.path))[0]
                                Image.fromarray(ref_arr.clip(0, 255).astype(np.uint8)).save(
                                    os.path.join(supp_preview_dir, f'{src_name}_step{self.step_num:06d}.jpg')
                                )
                        except Exception as e:
                            print(f"WARNING: suppression preview save failed: {e}")

                    # expand to match latent channels for multiplication
                    face_supp_mask = face_supp_mask.expand(-1, noisy_latents.shape[1], -1, -1)
                    # normalize to mean=1 so total loss magnitude is preserved
                    face_supp_mask = face_supp_mask / face_supp_mask.mean()
                    mask_multiplier = mask_multiplier * face_supp_mask
                else:
                    print("WARNING: face_suppression_weight is set but no face bboxes available. "
                          "Enable face_id or ensure InsightFace face detection is running.")

            # -------- Subject-mask region loss weighting (Phase 2) --------
            # Delegates to _build_subject_mask_weight. Composes multiplicatively
            # with face_suppression above; does NOT replace it.
            subject_weight = self._build_subject_mask_weight(
                batch, noisy_latents.shape, dtype=dtype,
            )
            if subject_weight is not None:
                mask_multiplier = mask_multiplier * subject_weight

        def get_adapter_multiplier():
            if self.adapter and isinstance(self.adapter, T2IAdapter):
                # training a t2i adapter, not using as assistant.
                return 1.0
            elif match_adapter_assist:
                # training a texture. We want it high
                adapter_strength_min = 0.9
                adapter_strength_max = 1.0
            else:
                # training with assistance, we want it low
                # adapter_strength_min = 0.4
                # adapter_strength_max = 0.7
                adapter_strength_min = 0.5
                adapter_strength_max = 1.1

            adapter_conditioning_scale = torch.rand(
                (1,), device=self.device_torch, dtype=dtype
            )

            adapter_conditioning_scale = value_map(
                adapter_conditioning_scale,
                0.0,
                1.0,
                adapter_strength_min,
                adapter_strength_max
            )
            return adapter_conditioning_scale

        # flush()
        with self.timer('grad_setup'):

            # text encoding
            grad_on_text_encoder = False
            if self.train_config.train_text_encoder:
                grad_on_text_encoder = True

            if self.embedding is not None:
                grad_on_text_encoder = True

            if self.adapter and isinstance(self.adapter, ClipVisionAdapter):
                grad_on_text_encoder = True

            if self.adapter_config and self.adapter_config.type == 'te_augmenter':
                grad_on_text_encoder = True

            # have a blank network so we can wrap it in a context and set multipliers without checking every time
            if self.network is not None:
                network = self.network
            else:
                network = BlankNetwork()

            # set the weights
            network.multiplier = network_weight_list

        # activate network if it exits

        prompts_1 = conditioned_prompts
        prompts_2 = None
        if self.train_config.short_and_long_captions_encoder_split and self.sd.is_xl:
            prompts_1 = batch.get_caption_short_list()
            prompts_2 = conditioned_prompts

            # make the batch splits
        if self.train_config.single_item_batching:
            if self.model_config.refiner_name_or_path is not None:
                raise ValueError("Single item batching is not supported when training the refiner")
            batch_size = noisy_latents.shape[0]
            # chunk/split everything
            noisy_latents_list = torch.chunk(noisy_latents, batch_size, dim=0)
            noise_list = torch.chunk(noise, batch_size, dim=0)
            timesteps_list = torch.chunk(timesteps, batch_size, dim=0)
            conditioned_prompts_list = [[prompt] for prompt in prompts_1]
            if imgs is not None:
                imgs_list = torch.chunk(imgs, batch_size, dim=0)
            else:
                imgs_list = [None for _ in range(batch_size)]
            if adapter_images is not None:
                adapter_images_list = torch.chunk(adapter_images, batch_size, dim=0)
            else:
                adapter_images_list = [None for _ in range(batch_size)]
            if clip_images is not None:
                clip_images_list = torch.chunk(clip_images, batch_size, dim=0)
            else:
                clip_images_list = [None for _ in range(batch_size)]
            mask_multiplier_list = torch.chunk(mask_multiplier, batch_size, dim=0)
            if prompts_2 is None:
                prompt_2_list = [None for _ in range(batch_size)]
            else:
                prompt_2_list = [[prompt] for prompt in prompts_2]

        else:
            noisy_latents_list = [noisy_latents]
            noise_list = [noise]
            timesteps_list = [timesteps]
            conditioned_prompts_list = [prompts_1]
            imgs_list = [imgs]
            adapter_images_list = [adapter_images]
            clip_images_list = [clip_images]
            mask_multiplier_list = [mask_multiplier]
            if prompts_2 is None:
                prompt_2_list = [None]
            else:
                prompt_2_list = [prompts_2]

        for noisy_latents, noise, timesteps, conditioned_prompts, imgs, adapter_images, clip_images, mask_multiplier, prompt_2 in zip(
                noisy_latents_list,
                noise_list,
                timesteps_list,
                conditioned_prompts_list,
                imgs_list,
                adapter_images_list,
                clip_images_list,
                mask_multiplier_list,
                prompt_2_list
        ):

            # if self.train_config.negative_prompt is not None:
            #     # add negative prompt
            #     conditioned_prompts = conditioned_prompts + [self.train_config.negative_prompt for x in
            #                                                  range(len(conditioned_prompts))]
            #     if prompt_2 is not None:
            #         prompt_2 = prompt_2 + [self.train_config.negative_prompt for x in range(len(prompt_2))]

            with (network):
                # encode clip adapter here so embeds are active for tokenizer
                if self.adapter and isinstance(self.adapter, ClipVisionAdapter):
                    with self.timer('encode_clip_vision_embeds'):
                        if has_clip_image:
                            conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                clip_images.detach().to(self.device_torch, dtype=dtype),
                                is_training=True,
                                has_been_preprocessed=True
                            )
                        else:
                            # just do a blank one
                            conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                torch.zeros(
                                    (noisy_latents.shape[0], 3, 512, 512),
                                    device=self.device_torch, dtype=dtype
                                ),
                                is_training=True,
                                has_been_preprocessed=True,
                                drop=True
                            )
                        # it will be injected into the tokenizer when called
                        self.adapter(conditional_clip_embeds)

                # do the custom adapter after the prior prediction
                if self.adapter and isinstance(self.adapter, CustomAdapter) and (has_clip_image or is_reg):
                    quad_count = random.randint(1, 4)
                    self.adapter.train()
                    self.adapter.trigger_pre_te(
                        tensors_preprocessed=clip_images if not is_reg else None,  # on regs we send none to get random noise
                        is_training=True,
                        has_been_preprocessed=True,
                        quad_count=quad_count,
                        batch_tensor=batch.tensor if not is_reg else None,
                        batch_size=noisy_latents.shape[0]
                    )

                with self.timer('encode_prompt'):
                    unconditional_embeds = None
                    prompt_kwargs = {}
                    if self.sd.encode_control_in_text_embeddings and batch.control_tensor is not None:
                        prompt_kwargs['control_images'] = batch.control_tensor.to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                    if self.train_config.unload_text_encoder or self.is_caching_text_embeddings:
                        with torch.set_grad_enabled(False):
                            if batch.prompt_embeds is not None:
                                # use the cached embeds
                                conditional_embeds = batch.prompt_embeds.clone().detach().to(
                                    self.device_torch, dtype=dtype
                                )
                            else:
                                embeds_to_use = self.cached_blank_embeds.clone().detach().to(
                                    self.device_torch, dtype=dtype
                                )
                                if self.cached_trigger_embeds is not None and not is_reg:
                                    embeds_to_use = self.cached_trigger_embeds.clone().detach().to(
                                        self.device_torch, dtype=dtype
                                    )
                                conditional_embeds = concat_prompt_embeds(
                                    [embeds_to_use] * noisy_latents.shape[0]
                                )
                            if self.train_config.do_cfg:
                                unconditional_embeds = self.cached_blank_embeds.clone().detach().to(
                                    self.device_torch, dtype=dtype
                                )
                                unconditional_embeds = concat_prompt_embeds(
                                    [unconditional_embeds] * noisy_latents.shape[0]
                                )

                            if isinstance(self.adapter, CustomAdapter):
                                self.adapter.is_unconditional_run = False

                    elif grad_on_text_encoder:
                        with torch.set_grad_enabled(True):
                            if isinstance(self.adapter, CustomAdapter):
                                self.adapter.is_unconditional_run = False
                            conditional_embeds = self.sd.encode_prompt(
                                conditioned_prompts, prompt_2,
                                dropout_prob=self.train_config.prompt_dropout_prob,
                                long_prompts=self.do_long_prompts,
                                **prompt_kwargs
                            ).to(
                                self.device_torch,
                                dtype=dtype)

                            if self.train_config.do_cfg:
                                if isinstance(self.adapter, CustomAdapter):
                                    self.adapter.is_unconditional_run = True
                                # todo only do one and repeat it
                                unconditional_embeds = self.sd.encode_prompt(
                                    self.batch_negative_prompt,
                                    self.batch_negative_prompt,
                                    dropout_prob=self.train_config.prompt_dropout_prob,
                                    long_prompts=self.do_long_prompts,
                                    **prompt_kwargs
                                ).to(
                                    self.device_torch,
                                    dtype=dtype)
                                if isinstance(self.adapter, CustomAdapter):
                                    self.adapter.is_unconditional_run = False
                    else:
                        with torch.set_grad_enabled(False):
                            # make sure it is in eval mode
                            if isinstance(self.sd.text_encoder, list):
                                for te in self.sd.text_encoder:
                                    te.eval()
                            else:
                                self.sd.text_encoder.eval()
                            if isinstance(self.adapter, CustomAdapter):
                                self.adapter.is_unconditional_run = False
                            conditional_embeds = self.sd.encode_prompt(
                                conditioned_prompts, prompt_2,
                                dropout_prob=self.train_config.prompt_dropout_prob,
                                long_prompts=self.do_long_prompts,
                                **prompt_kwargs
                            ).to(
                                self.device_torch,
                                dtype=dtype)
                            if self.train_config.do_cfg:
                                if isinstance(self.adapter, CustomAdapter):
                                    self.adapter.is_unconditional_run = True
                                unconditional_embeds = self.sd.encode_prompt(
                                    self.batch_negative_prompt,
                                    dropout_prob=self.train_config.prompt_dropout_prob,
                                    long_prompts=self.do_long_prompts,
                                    **prompt_kwargs
                                ).to(
                                    self.device_torch,
                                    dtype=dtype)
                                if isinstance(self.adapter, CustomAdapter):
                                    self.adapter.is_unconditional_run = False
                            
                            if self.train_config.diff_output_preservation:
                                dop_prompts = [p.replace(self.trigger_word, self.train_config.diff_output_preservation_class) for p in conditioned_prompts]
                                dop_prompts_2 = None
                                if prompt_2 is not None:
                                    dop_prompts_2 = [p.replace(self.trigger_word, self.train_config.diff_output_preservation_class) for p in prompt_2]
                                self.diff_output_preservation_embeds = self.sd.encode_prompt(
                                    dop_prompts, dop_prompts_2,
                                    dropout_prob=self.train_config.prompt_dropout_prob,
                                    long_prompts=self.do_long_prompts,
                                    **prompt_kwargs
                                ).to(
                                    self.device_torch,
                                    dtype=dtype)
                        # detach the embeddings
                        conditional_embeds = conditional_embeds.detach()
                        if self.train_config.do_cfg:
                            unconditional_embeds = unconditional_embeds.detach()
                    
                    if self.decorator:
                        conditional_embeds.text_embeds = self.decorator(
                            conditional_embeds.text_embeds
                        )
                        if self.train_config.do_cfg:
                            unconditional_embeds.text_embeds = self.decorator(
                                unconditional_embeds.text_embeds, 
                                is_unconditional=True
                            )

                # flush()
                pred_kwargs = {}

                # LoRA+ID: project face embeddings and add to prediction kwargs
                if self.face_id_projector is not None and batch.face_embedding is not None:
                    face_emb = batch.face_embedding.to(self.device_torch, dtype=self.face_id_projector.norm.weight.dtype)
                    face_tokens = self.face_id_projector(face_emb)

                    # Vision face projection (CLIP/DINOv2 spatial tokens)
                    vision_tokens = None
                    if self.vision_face_projector is not None and batch.vision_face_embedding is not None:
                        vision_emb = batch.vision_face_embedding.to(
                            self.device_torch,
                            dtype=self.vision_face_projector.resampler.proj_in.weight.dtype,
                        )
                        vision_tokens = self.vision_face_projector(vision_emb)
                        # Log vision token norm before dropout
                        self._last_vision_token_norm = vision_tokens.detach().float().norm(dim=-1).mean().item()

                    # Body shape projection (SMPL betas)
                    body_tokens = None
                    if self.body_id_projector is not None and batch.body_embedding is not None:
                        body_emb = batch.body_embedding.to(
                            self.device_torch,
                            dtype=self.body_id_projector.norm.weight.dtype,
                        )
                        body_tokens = self.body_id_projector(body_emb)
                        self._last_body_token_norm = body_tokens.detach().float().norm(dim=-1).mean().item()

                    # Synchronized dropout (same mask for all identity tokens)
                    if self.face_id_config.dropout_prob > 0:
                        drop_mask = (torch.rand(face_tokens.shape[0], 1, 1, device=face_tokens.device) > self.face_id_config.dropout_prob).float()
                        face_tokens = face_tokens * drop_mask
                        if vision_tokens is not None:
                            vision_tokens = vision_tokens * drop_mask
                        if body_tokens is not None:
                            body_tokens = body_tokens * drop_mask

                    # Concatenate all identity tokens
                    token_parts = [face_tokens]
                    if vision_tokens is not None:
                        token_parts.append(vision_tokens)
                    if body_tokens is not None:
                        token_parts.append(body_tokens)
                    all_face_tokens = torch.cat(token_parts, dim=1) if len(token_parts) > 1 else face_tokens

                    pred_kwargs['face_tokens'] = all_face_tokens
                    # mean per-token L2 norm (batch-size invariant)
                    per_token_norms = face_tokens.detach().float().norm(dim=-1)  # (B, num_tokens)
                    self._last_face_token_norm = per_token_norms.mean().item()

                if has_adapter_img:
                    if (self.adapter and isinstance(self.adapter, T2IAdapter)) or (
                            self.assistant_adapter and isinstance(self.assistant_adapter, T2IAdapter)):
                        with torch.set_grad_enabled(self.adapter is not None):
                            adapter = self.assistant_adapter if self.assistant_adapter is not None else self.adapter
                            adapter_multiplier = get_adapter_multiplier()
                            with self.timer('encode_adapter'):
                                down_block_additional_residuals = adapter(adapter_images)
                                if self.assistant_adapter:
                                    # not training. detach
                                    down_block_additional_residuals = [
                                        sample.to(dtype=dtype).detach() * adapter_multiplier for sample in
                                        down_block_additional_residuals
                                    ]
                                else:
                                    down_block_additional_residuals = [
                                        sample.to(dtype=dtype) * adapter_multiplier for sample in
                                        down_block_additional_residuals
                                    ]

                                pred_kwargs['down_intrablock_additional_residuals'] = down_block_additional_residuals

                if self.adapter and isinstance(self.adapter, IPAdapter):
                    with self.timer('encode_adapter_embeds'):
                        # number of images to do if doing a quad image
                        quad_count = random.randint(1, 4)
                        image_size = self.adapter.input_size
                        if has_clip_image_embeds:
                            # todo handle reg images better than this
                            if is_reg:
                                # get unconditional image embeds from cache
                                embeds = [
                                    load_file(random.choice(batch.clip_image_embeds_unconditional)) for i in
                                    range(noisy_latents.shape[0])
                                ]
                                conditional_clip_embeds = self.adapter.parse_clip_image_embeds_from_cache(
                                    embeds,
                                    quad_count=quad_count
                                )

                                if self.train_config.do_cfg:
                                    embeds = [
                                        load_file(random.choice(batch.clip_image_embeds_unconditional)) for i in
                                        range(noisy_latents.shape[0])
                                    ]
                                    unconditional_clip_embeds = self.adapter.parse_clip_image_embeds_from_cache(
                                        embeds,
                                        quad_count=quad_count
                                    )

                            else:
                                conditional_clip_embeds = self.adapter.parse_clip_image_embeds_from_cache(
                                    batch.clip_image_embeds,
                                    quad_count=quad_count
                                )
                                if self.train_config.do_cfg:
                                    unconditional_clip_embeds = self.adapter.parse_clip_image_embeds_from_cache(
                                        batch.clip_image_embeds_unconditional,
                                        quad_count=quad_count
                                    )
                        elif is_reg:
                            # we will zero it out in the img embedder
                            clip_images = torch.zeros(
                                (noisy_latents.shape[0], 3, image_size, image_size),
                                device=self.device_torch, dtype=dtype
                            ).detach()
                            # drop will zero it out
                            conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                clip_images,
                                drop=True,
                                is_training=True,
                                has_been_preprocessed=False,
                                quad_count=quad_count
                            )
                            if self.train_config.do_cfg:
                                unconditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                    torch.zeros(
                                        (noisy_latents.shape[0], 3, image_size, image_size),
                                        device=self.device_torch, dtype=dtype
                                    ).detach(),
                                    is_training=True,
                                    drop=True,
                                    has_been_preprocessed=False,
                                    quad_count=quad_count
                                )
                        elif has_clip_image:
                            conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                clip_images.detach().to(self.device_torch, dtype=dtype),
                                is_training=True,
                                has_been_preprocessed=True,
                                quad_count=quad_count,
                                # do cfg on clip embeds to normalize the embeddings for when doing cfg
                                # cfg_embed_strength=3.0 if not self.train_config.do_cfg else None
                                # cfg_embed_strength=3.0 if not self.train_config.do_cfg else None
                            )
                            if self.train_config.do_cfg:
                                unconditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                    clip_images.detach().to(self.device_torch, dtype=dtype),
                                    is_training=True,
                                    drop=True,
                                    has_been_preprocessed=True,
                                    quad_count=quad_count
                                )
                        else:
                            print_acc("No Clip Image")
                            print_acc([file_item.path for file_item in batch.file_items])
                            raise ValueError("Could not find clip image")

                    if not self.adapter_config.train_image_encoder:
                        # we are not training the image encoder, so we need to detach the embeds
                        conditional_clip_embeds = conditional_clip_embeds.detach()
                        if self.train_config.do_cfg:
                            unconditional_clip_embeds = unconditional_clip_embeds.detach()

                    with self.timer('encode_adapter'):
                        self.adapter.train()
                        conditional_embeds = self.adapter(
                            conditional_embeds.detach(),
                            conditional_clip_embeds,
                            is_unconditional=False
                        )
                        if self.train_config.do_cfg:
                            unconditional_embeds = self.adapter(
                                unconditional_embeds.detach(),
                                unconditional_clip_embeds,
                                is_unconditional=True
                            )
                        else:
                            # wipe out unconsitional
                            self.adapter.last_unconditional = None

                if self.adapter and isinstance(self.adapter, ReferenceAdapter):
                    # pass in our scheduler
                    self.adapter.noise_scheduler = self.lr_scheduler
                    if has_clip_image or has_adapter_img:
                        img_to_use = clip_images if has_clip_image else adapter_images
                        # currently 0-1 needs to be -1 to 1
                        reference_images = ((img_to_use - 0.5) * 2).detach().to(self.device_torch, dtype=dtype)
                        self.adapter.set_reference_images(reference_images)
                        self.adapter.noise_scheduler = self.sd.noise_scheduler
                    elif is_reg:
                        self.adapter.set_blank_reference_images(noisy_latents.shape[0])
                    else:
                        self.adapter.set_reference_images(None)

                prior_pred = None

                do_inverted_masked_prior = False
                if self.train_config.inverted_mask_prior and batch.mask_tensor is not None:
                    do_inverted_masked_prior = True

                do_correct_pred_norm_prior = self.train_config.correct_pred_norm

                do_guidance_prior = False

                if batch.unconditional_latents is not None:
                    # for this not that, we need a prior pred to normalize
                    guidance_type: GuidanceType = batch.file_items[0].dataset_config.guidance_type
                    if guidance_type == 'tnt':
                        do_guidance_prior = True

                if ((
                        has_adapter_img and self.assistant_adapter and match_adapter_assist) or self.do_prior_prediction or do_guidance_prior or do_reg_prior or do_inverted_masked_prior or self.train_config.correct_pred_norm):
                    with self.timer('prior predict'):
                        prior_embeds_to_use = conditional_embeds
                        # use diff_output_preservation embeds if doing dfe
                        if self.train_config.diff_output_preservation:
                            prior_embeds_to_use = self.diff_output_preservation_embeds.expand_to_batch(noisy_latents.shape[0])
                        
                        if self.train_config.blank_prompt_preservation:
                            blank_embeds = self.cached_blank_embeds.clone().detach().to(
                                self.device_torch, dtype=dtype
                            )
                            prior_embeds_to_use = concat_prompt_embeds(
                                [blank_embeds] * noisy_latents.shape[0]
                            )
                        
                        prior_pred = self.get_prior_prediction(
                            noisy_latents=noisy_latents,
                            conditional_embeds=prior_embeds_to_use,
                            match_adapter_assist=match_adapter_assist,
                            network_weight_list=network_weight_list,
                            timesteps=timesteps,
                            pred_kwargs=pred_kwargs,
                            noise=noise,
                            batch=batch,
                            unconditional_embeds=unconditional_embeds,
                            conditioned_prompts=conditioned_prompts
                        )
                        if prior_pred is not None:
                            prior_pred = prior_pred.detach()

                # do the custom adapter after the prior prediction
                if self.adapter and isinstance(self.adapter, CustomAdapter) and (has_clip_image or self.adapter_config.type in ['llm_adapter', 'text_encoder']):
                    quad_count = random.randint(1, 4)
                    self.adapter.train()
                    conditional_embeds = self.adapter.condition_encoded_embeds(
                        tensors_0_1=clip_images,
                        prompt_embeds=conditional_embeds,
                        is_training=True,
                        has_been_preprocessed=True,
                        quad_count=quad_count
                    )
                    if self.train_config.do_cfg and unconditional_embeds is not None:
                        unconditional_embeds = self.adapter.condition_encoded_embeds(
                            tensors_0_1=clip_images,
                            prompt_embeds=unconditional_embeds,
                            is_training=True,
                            has_been_preprocessed=True,
                            is_unconditional=True,
                            quad_count=quad_count
                        )

                if self.adapter and isinstance(self.adapter, CustomAdapter) and batch.extra_values is not None:
                    self.adapter.add_extra_values(batch.extra_values.detach())

                    if self.train_config.do_cfg:
                        self.adapter.add_extra_values(torch.zeros_like(batch.extra_values.detach()),
                                                      is_unconditional=True)

                if has_adapter_img:
                    if (self.adapter and isinstance(self.adapter, ControlNetModel)) or (
                            self.assistant_adapter and isinstance(self.assistant_adapter, ControlNetModel)):
                        if self.train_config.do_cfg:
                            raise ValueError("ControlNetModel is not supported with CFG")
                        with torch.set_grad_enabled(self.adapter is not None):
                            adapter: ControlNetModel = self.assistant_adapter if self.assistant_adapter is not None else self.adapter
                            adapter_multiplier = get_adapter_multiplier()
                            with self.timer('encode_adapter'):
                                # add_text_embeds is pooled_prompt_embeds for sdxl
                                added_cond_kwargs = {}
                                if self.sd.is_xl:
                                    added_cond_kwargs["text_embeds"] = conditional_embeds.pooled_embeds
                                    added_cond_kwargs['time_ids'] = self.sd.get_time_ids_from_latents(noisy_latents)
                                down_block_res_samples, mid_block_res_sample = adapter(
                                    noisy_latents,
                                    timesteps,
                                    encoder_hidden_states=conditional_embeds.text_embeds,
                                    controlnet_cond=adapter_images,
                                    conditioning_scale=1.0,
                                    guess_mode=False,
                                    added_cond_kwargs=added_cond_kwargs,
                                    return_dict=False,
                                )
                                pred_kwargs['down_block_additional_residuals'] = down_block_res_samples
                                pred_kwargs['mid_block_additional_residual'] = mid_block_res_sample
                
                if self.train_config.do_guidance_loss and isinstance(self.train_config.guidance_loss_target, list):
                    batch_size = noisy_latents.shape[0]
                    # update the guidance value, random float between guidance_loss_target[0] and guidance_loss_target[1]
                    self._guidance_loss_target_batch = [
                        random.uniform(
                            self.train_config.guidance_loss_target[0],
                            self.train_config.guidance_loss_target[1]
                        ) for _ in range(batch_size)
                    ]

                self.before_unet_predict()
                
                if unconditional_embeds is not None:
                    unconditional_embeds = unconditional_embeds.to(self.device_torch, dtype=dtype).detach()
                with self.timer('condition_noisy_latents'):
                    # do it for the model
                    noisy_latents = self.sd.condition_noisy_latents(noisy_latents, batch)
                    if self.adapter and isinstance(self.adapter, CustomAdapter):
                        noisy_latents = self.adapter.condition_noisy_latents(noisy_latents, batch)
                
                if self.train_config.timestep_type == 'next_sample':
                    with self.timer('next_sample_step'):
                        with torch.no_grad():
                            
                            stepped_timestep_indicies = [self.sd.noise_scheduler.index_for_timestep(t) + 1 for t in timesteps]
                            stepped_timesteps = [self.sd.noise_scheduler.timesteps[x] for x in stepped_timestep_indicies]
                            stepped_timesteps = torch.stack(stepped_timesteps, dim=0)
                            
                            # do a sample at the current timestep and step it, then determine new noise
                            next_sample_pred = self.predict_noise(
                                noisy_latents=noisy_latents.to(self.device_torch, dtype=dtype),
                                timesteps=timesteps,
                                conditional_embeds=conditional_embeds.to(self.device_torch, dtype=dtype),
                                unconditional_embeds=unconditional_embeds,
                                batch=batch,
                                **pred_kwargs
                            )
                            stepped_latents = self.sd.step_scheduler(
                                next_sample_pred,
                                noisy_latents,
                                timesteps,
                                self.sd.noise_scheduler
                            )
                            # stepped latents is our new noisy latents. Now we need to determine noise in the current sample
                            noisy_latents = stepped_latents
                            original_samples = batch.latents.to(self.device_torch, dtype=dtype)
                            # todo calc next timestep, for now this may work as it
                            t_01 = (stepped_timesteps / 1000).to(original_samples.device)
                            if len(stepped_latents.shape) == 4:
                                t_01 = t_01.view(-1, 1, 1, 1)
                            elif len(stepped_latents.shape) == 5:
                                t_01 = t_01.view(-1, 1, 1, 1, 1)
                            else:
                                raise ValueError("Unknown stepped latents shape", stepped_latents.shape)
                            next_sample_noise = (stepped_latents - (1.0 - t_01) * original_samples) / t_01
                            noise = next_sample_noise
                            timesteps = stepped_timesteps
                # do a prior pred if we have an unconditional image, we will swap out the giadance later
                if batch.unconditional_latents is not None or self.do_guided_loss:
                    # do guided loss
                    loss = self.get_guided_loss(
                        noisy_latents=noisy_latents,
                        conditional_embeds=conditional_embeds,
                        match_adapter_assist=match_adapter_assist,
                        network_weight_list=network_weight_list,
                        timesteps=timesteps,
                        pred_kwargs=pred_kwargs,
                        batch=batch,
                        noise=noise,
                        unconditional_embeds=unconditional_embeds,
                        mask_multiplier=mask_multiplier,
                        prior_pred=prior_pred,
                    )
                    
                elif self.train_config.loss_type == 'mean_flow':
                    loss = self.get_mean_flow_loss(
                        noisy_latents=noisy_latents,
                        conditional_embeds=conditional_embeds,
                        match_adapter_assist=match_adapter_assist,
                        network_weight_list=network_weight_list,
                        timesteps=timesteps,
                        pred_kwargs=pred_kwargs,
                        batch=batch,
                        noise=noise,
                        unconditional_embeds=unconditional_embeds,
                        prior_pred=prior_pred,
                    )
                else:
                    with self.timer('predict_unet'):
                        noise_pred = self.predict_noise(
                            noisy_latents=noisy_latents.to(self.device_torch, dtype=dtype),
                            timesteps=timesteps,
                            conditional_embeds=conditional_embeds.to(self.device_torch, dtype=dtype),
                            unconditional_embeds=unconditional_embeds,
                            batch=batch,
                            is_primary_pred=True,
                            **pred_kwargs
                        )
                    self.after_unet_predict()

                    with self.timer('calculate_loss'):
                        noise = noise.to(self.device_torch, dtype=dtype).detach()
                        prior_to_calculate_loss = prior_pred
                        # if we are doing diff_output_preservation and not noing inverted masked prior
                        # then we need to send none here so it will not target the prior
                        doing_preservation = self.train_config.diff_output_preservation or self.train_config.blank_prompt_preservation
                        if doing_preservation and not do_inverted_masked_prior:
                            prior_to_calculate_loss = None
                        
                        loss = self.calculate_loss(
                            noise_pred=noise_pred,
                            noise=noise,
                            noisy_latents=noisy_latents,
                            timesteps=timesteps,
                            batch=batch,
                            mask_multiplier=mask_multiplier,
                            prior_pred=prior_to_calculate_loss,
                        )
                    
                    # --- Pure-noise identity monitoring (no gradients, just tracking) ---
                    if (self.id_loss_model is not None
                            and batch.identity_embedding is not None
                            and self.step_num % 10 == 0):
                        with torch.no_grad():
                            pure_z = torch.randn_like(noisy_latents)
                            pure_t = torch.tensor([990.0], device=self.device_torch).expand(pure_z.shape[0])
                            # predict with text only, NO face tokens
                            pure_kwargs = {k: v for k, v in pred_kwargs.items() if k != 'face_tokens'}
                            pure_v = self.predict_noise(
                                noisy_latents=pure_z.to(self.device_torch, dtype=dtype),
                                timesteps=pure_t,
                                conditional_embeds=conditional_embeds.to(self.device_torch, dtype=dtype),
                                unconditional_embeds=unconditional_embeds,
                                batch=batch,
                                **pure_kwargs
                            )
                            if self.sd.is_flow_matching:
                                pure_x0 = pure_z - 0.99 * pure_v
                            else:
                                # DDPM: recover x0 at t=990
                                _pn_acp = self.sd.noise_scheduler.alphas_cumprod.to(
                                    device=pure_z.device, dtype=pure_z.dtype
                                )
                                _pn_ab = _pn_acp[990]
                                _pn_sab = _pn_ab.sqrt()
                                _pn_s1mab = (1.0 - _pn_ab).sqrt()
                                if self.sd.prediction_type == 'v_prediction':
                                    pure_x0 = _pn_sab * pure_z - _pn_s1mab * pure_v
                                else:
                                    pure_x0 = (pure_z - _pn_s1mab * pure_v) / _pn_sab.clamp(min=1e-8)
                            # decode through TAEF2/TAESD
                            if hasattr(self, '_taef2_decoder') and self._taef2_decoder is not None:
                                pn_decode = pure_x0
                                if pn_decode.shape[1] != 32:
                                    from einops import rearrange
                                    pn_decode = rearrange(pn_decode, "b (c p1 p2) h w -> b c (h p1) (w p2)", c=32, p1=2, p2=2)
                                dec_dtype = next(self._taef2_decoder.parameters()).dtype
                                pn_pixels = self._taef2_decoder(pn_decode.to(dec_dtype)).float().clamp(0, 1)
                            elif self.taesd is not None:
                                taesd_dtype = next(self.taesd.parameters()).dtype
                                pn_pixels = self.taesd.decode(pure_x0.to(taesd_dtype)).sample.float()
                                pn_pixels = ((pn_pixels + 1.0) * 0.5).clamp(0, 1)
                            else:
                                pn_pixels = None
                            if pn_pixels is not None:
                                pn_emb = self.id_loss_model(pn_pixels)
                                ref_emb = batch.identity_embedding.to(pn_emb.device, dtype=pn_emb.dtype)
                                pn_cos = F.cosine_similarity(pn_emb, ref_emb, dim=-1).mean().item()
                                self._last_pure_noise_cos = pn_cos
                                # save preview
                                pn_preview_dir = os.path.join(self.save_root, 'pure_noise_previews')
                                os.makedirs(pn_preview_dir, exist_ok=True)
                                img = TF.to_pil_image(pn_pixels[0].clamp(0, 1).cpu())
                                img.save(os.path.join(pn_preview_dir, f'step{self.step_num:06d}_cos{pn_cos:.3f}.jpg'))

                    if self.train_config.diff_output_preservation or self.train_config.blank_prompt_preservation:
                        # send the loss backwards otherwise checkpointing will fail
                        self.accelerator.backward(loss)
                        normal_loss = loss.detach() # dont send backward again
                        
                        with torch.no_grad():
                            if self.train_config.diff_output_preservation:
                                preservation_embeds = self.diff_output_preservation_embeds.expand_to_batch(noisy_latents.shape[0])
                            elif self.train_config.blank_prompt_preservation:
                                blank_embeds = self.cached_blank_embeds.clone().detach().to(
                                    self.device_torch, dtype=dtype
                                )
                                preservation_embeds = concat_prompt_embeds(
                                    [blank_embeds] * noisy_latents.shape[0]
                                )
                        preservation_pred = self.predict_noise(
                            noisy_latents=noisy_latents.to(self.device_torch, dtype=dtype),
                            timesteps=timesteps,
                            conditional_embeds=preservation_embeds.to(self.device_torch, dtype=dtype),
                            unconditional_embeds=unconditional_embeds,
                            batch=batch,
                            **pred_kwargs
                        )
                        multiplier = self.train_config.diff_output_preservation_multiplier if self.train_config.diff_output_preservation else self.train_config.blank_prompt_preservation_multiplier
                        preservation_loss = torch.nn.functional.mse_loss(preservation_pred, prior_pred) * multiplier
                        self.accelerator.backward(preservation_loss)

                        loss = normal_loss + preservation_loss
                        loss = loss.clone().detach()
                        # require grad again so the backward wont fail
                        loss.requires_grad_(True)
                        
                # check if nan
                if torch.isnan(loss):
                    print_acc("loss is nan")
                    loss = torch.zeros_like(loss).requires_grad_(True)

                with self.timer('backward'):
                    # todo we have multiplier seperated. works for now as res are not in same batch, but need to change
                    lm_scalar = loss_multiplier.mean()
                    loss = loss * lm_scalar
                    # IMPORTANT if gradient checkpointing do not leave with network when doing backward
                    # it will destroy the gradients. This is because the network is a context manager
                    # and will change the multipliers back to 0.0 when exiting. They will be
                    # 0.0 for the backward pass and the gradients will be 0.0
                    # I spent weeks on fighting this. DON'T DO IT
                    # with fsdp_overlap_step_with_backward():
                    # if self.is_bfloat:
                    # loss.backward()
                    # else:

                    # Gradient cosine diagnostic (display-only). When fired,
                    # we use `autograd.grad` for the depth-only gradient
                    # (doesn't touch p.grad), then the standard backward
                    # populates p.grad with the full gradient. The diffusion
                    # contribution is recovered by linearity:
                    #   g_diff = g_full - g_dc.
                    # Cost: one extra backward pass via the retained graph.
                    _grad_cos_every = int(getattr(
                        self.train_config, 'gradient_cosine_log_every', 0,
                    ) or 0)
                    _do_grad_cos = (
                        _grad_cos_every > 0
                        and self._dc_applied_for_grad is not None
                        and (self.step_num % _grad_cos_every == 0)
                    )

                    _dc_grads_snapshot = None
                    _pre_existing_grads = None
                    _trainable_params_snapshot = None
                    if _do_grad_cos:
                        try:
                            _trainable_params_snapshot = [
                                p for p in self._iter_trainable_params()
                                if p.requires_grad
                            ]
                            if _trainable_params_snapshot:
                                # Snapshot any grad accumulated by earlier
                                # microbatches in this optimizer step. We
                                # subtract this off after the new backward
                                # so the cosine measures THIS microbatch's
                                # contribution alone (not the running sum).
                                _pre_existing_grads = [
                                    p.grad.detach().clone() if p.grad is not None else None
                                    for p in _trainable_params_snapshot
                                ]
                                _dc_for_grad = self._dc_applied_for_grad * lm_scalar
                                _dc_grads_snapshot = torch.autograd.grad(
                                    _dc_for_grad,
                                    _trainable_params_snapshot,
                                    retain_graph=True,
                                    allow_unused=True,
                                )
                        except Exception as _gc_err:  # noqa: BLE001
                            print_acc(
                                f"  gradient cosine: depth-only autograd.grad failed: {_gc_err}"
                            )
                            _dc_grads_snapshot = None
                            _pre_existing_grads = None
                            _trainable_params_snapshot = None

                    self.accelerator.backward(loss)

                    if _dc_grads_snapshot is not None and _trainable_params_snapshot is not None:
                        try:
                            self._record_grad_cosine(
                                _trainable_params_snapshot,
                                _dc_grads_snapshot,
                                _pre_existing_grads,
                            )
                        except Exception as _gc_err:  # noqa: BLE001
                            print_acc(
                                f"  gradient cosine: norm/cosine compute failed: {_gc_err}"
                            )

        # Mirror this microbatch's metric scalars into the buffer so that
        # `hook_train_loop` can flush a real cross-microbatch mean. Pure
        # bookkeeping — no loss tensor reference is read here, the loss has
        # already been backpropped above.
        self._snapshot_metrics_to_buffer()
        return loss.detach()
        # flush()

    def hook_train_loop(self, batch: Union[DataLoaderBatchDTO, List[DataLoaderBatchDTO]]):
        # Reset display-only per-t-band bin accumulators at the start of every
        # optimizer step so that:
        #   (a) same-bin samples within a microbatch accumulate (running mean)
        #       instead of overwriting each other,
        #   (b) cross-microbatch (gradient_accumulation_steps > 1) samples
        #       merge into one mean per bin,
        #   (c) bins from prior optimizer steps don't leak into the next step.
        # This is purely metric bookkeeping — no loss tensor is touched.
        self._reset_step_bins()
        # Same reasoning for the cross-microbatch metric buffer: clear it at
        # the very top of every optimizer step so the next step's
        # `_snapshot_metrics_to_buffer` calls accumulate from zero.
        if hasattr(self, '_metric_buffer'):
            self._metric_buffer.reset()
        if isinstance(batch, list):
            batch_list = batch
        else:
            batch_list = [batch]
        total_loss = None
        self.optimizer.zero_grad()
        for batch in batch_list:
            if self.sd.is_multistage:
                # handle multistage switching
                if self.steps_this_boundary >= self.train_config.switch_boundary_every or self.current_boundary_index not in self.sd.trainable_multistage_boundaries:
                    # iterate to make sure we only train trainable_multistage_boundaries
                    while True:
                        self.steps_this_boundary = 0
                        self.current_boundary_index += 1
                        if self.current_boundary_index >= len(self.sd.multistage_boundaries):
                            self.current_boundary_index = 0
                        if self.current_boundary_index in self.sd.trainable_multistage_boundaries:
                            # if this boundary is trainable, we can stop looking
                            break
            loss = self.train_single_accumulation(batch)
            self.steps_this_boundary += 1
            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss
            if len(batch_list) > 1 and self.model_config.low_vram:
                torch.cuda.empty_cache()


        grad_norm = None
        if not self.is_grad_accumulation_step:
            # fix this for multi params
            if self.train_config.optimizer != 'adafactor':
                if isinstance(self.params[0], dict):
                    total_norm_sq = 0.0
                    for i in range(len(self.params)):
                        norm = self.accelerator.clip_grad_norm_(self.params[i]['params'], self.train_config.max_grad_norm)
                        total_norm_sq += norm.item() ** 2
                    grad_norm = total_norm_sq ** 0.5
                else:
                    grad_norm = self.accelerator.clip_grad_norm_(self.params, self.train_config.max_grad_norm).item()
            # only step if we are not accumulating
            with self.timer('optimizer_step'):
                self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                if self.adapter and isinstance(self.adapter, CustomAdapter):
                    self.adapter.post_weight_update()
            if self.ema is not None:
                with self.timer('ema_update'):
                    self.ema.update()
        else:
            # gradient accumulation. Just a place for breakpoint
            pass

        # TODO Should we only step scheduler on grad step? If so, need to recalculate last step
        with self.timer('scheduler_step'):
            self.lr_scheduler.step()

        if self.embedding is not None:
            with self.timer('restore_embeddings'):
                # Let's make sure we don't update any embedding weights besides the newly added token
                self.embedding.restore_embeddings()
        if self.adapter is not None and isinstance(self.adapter, ClipVisionAdapter):
            with self.timer('restore_adapter'):
                # Let's make sure we don't update any embedding weights besides the newly added token
                self.adapter.restore_embeddings()

        loss_dict = OrderedDict(
            {'loss': (total_loss / len(batch_list)).item()}
        )
        if grad_norm is not None:
            loss_dict['grad_norm'] = grad_norm

        # LoRA+ID: log face token norm
        if hasattr(self, '_last_face_token_norm') and self._last_face_token_norm is not None:
            loss_dict['face_token_norm'] = self._last_face_token_norm
            self._last_face_token_norm = None

        # Vision token norm (CLIP/DINOv2 face crop tokens)
        if hasattr(self, '_last_vision_token_norm') and self._last_vision_token_norm is not None:
            loss_dict['vision_token_norm'] = self._last_vision_token_norm
            self._last_vision_token_norm = None

        # Body token norm (SMPL betas)
        if hasattr(self, '_last_body_token_norm') and self._last_body_token_norm is not None:
            loss_dict['body_token_norm'] = self._last_body_token_norm
            self._last_body_token_norm = None

        # Identity loss (auxiliary face similarity loss)
        if self._last_identity_loss is not None:
            loss_dict['identity_loss'] = self._last_identity_loss
            self._last_identity_loss = None
        # Landmark loss (auxiliary face shape loss)
        if self._last_landmark_loss is not None:
            loss_dict['landmark_loss'] = self._last_landmark_loss
            self._last_landmark_loss = None
        if self._last_pure_noise_cos is not None:
            loss_dict['pure_noise_cos'] = self._last_pure_noise_cos
            self._last_pure_noise_cos = None
        if self._last_diffusion_loss is not None:
            loss_dict['diffusion_loss'] = self._last_diffusion_loss
            self._last_diffusion_loss = None
        if self._last_diffusion_loss_applied is not None:
            loss_dict['diffusion_loss_applied'] = self._last_diffusion_loss_applied
            self._last_diffusion_loss_applied = None
        # Gradient cosine diagnostic (only populated on firing steps when
        # `gradient_cosine_log_every > 0`).
        if self._last_grad_norm_diffusion is not None:
            loss_dict['grad_norm_diffusion'] = self._last_grad_norm_diffusion
            self._last_grad_norm_diffusion = None
        if self._last_grad_norm_depth is not None:
            loss_dict['grad_norm_depth'] = self._last_grad_norm_depth
            self._last_grad_norm_depth = None
        if self._last_grad_cos_diff_depth is not None:
            loss_dict['grad_cos_diff_depth'] = self._last_grad_cos_diff_depth
            self._last_grad_cos_diff_depth = None
        if self._last_identity_loss_applied is not None:
            loss_dict['identity_loss_applied'] = self._last_identity_loss_applied
            self._last_identity_loss_applied = None
        if self._last_landmark_loss_applied is not None:
            loss_dict['landmark_loss_applied'] = self._last_landmark_loss_applied
            self._last_landmark_loss_applied = None
        # Body proportion loss (auxiliary body shape loss)
        if self._last_body_proportion_loss is not None:
            loss_dict['body_proportion_loss'] = self._last_body_proportion_loss
            self._last_body_proportion_loss = None
        if self._last_body_proportion_loss_applied is not None:
            loss_dict['body_proportion_loss_applied'] = self._last_body_proportion_loss_applied
            self._last_body_proportion_loss_applied = None
        if self._last_bp_sim_bins is not None:
            for bin_key, sim_val in self._bin_finalize(self._last_bp_sim_bins).items():
                loss_dict[bin_key] = sim_val
            self._last_bp_sim_bins = None
        # Body shape loss (HybrIK SMPL betas)
        if self._last_body_shape_loss is not None:
            loss_dict['body_shape_loss'] = self._last_body_shape_loss
            self._last_body_shape_loss = None
        if self._last_body_shape_loss_applied is not None:
            loss_dict['body_shape_loss_applied'] = self._last_body_shape_loss_applied
            self._last_body_shape_loss_applied = None
        if self._last_body_shape_cos is not None:
            loss_dict['body_shape_cos'] = self._last_body_shape_cos
            self._last_body_shape_cos = None
        if self._last_body_shape_l1 is not None:
            loss_dict['body_shape_l1'] = self._last_body_shape_l1
            self._last_body_shape_l1 = None
        if self._last_body_shape_gated_pct is not None:
            loss_dict['body_shape_gated_pct'] = self._last_body_shape_gated_pct
            self._last_body_shape_gated_pct = None
        if self._last_bsh_sim_bins is not None:
            for bin_key, sim_val in self._bin_finalize(self._last_bsh_sim_bins).items():
                loss_dict[bin_key] = sim_val
            self._last_bsh_sim_bins = None
        # Normal loss (Sapiens surface normals)
        if self._last_normal_loss is not None:
            loss_dict['normal_loss'] = self._last_normal_loss
            self._last_normal_loss = None
        if self._last_normal_loss_applied is not None:
            loss_dict['normal_loss_applied'] = self._last_normal_loss_applied
            self._last_normal_loss_applied = None
        if self._last_normal_cos is not None:
            loss_dict['normal_cos'] = self._last_normal_cos
            self._last_normal_cos = None
        # VAE anchor loss (perceptual feature matching)
        if self._last_vae_anchor_loss is not None:
            loss_dict['vae_anchor_loss'] = self._last_vae_anchor_loss
            self._last_vae_anchor_loss = None
        if self._last_vae_anchor_loss_applied is not None:
            loss_dict['vae_anchor_loss_applied'] = self._last_vae_anchor_loss_applied
            self._last_vae_anchor_loss_applied = None
        if self._last_vae_anchor_per_level is not None:
            for level_name, level_val in self._last_vae_anchor_per_level.items():
                loss_dict[f'va_{level_name}'] = level_val
            self._last_vae_anchor_per_level = None
        # Depth consistency loss (MiDaS SSI + multi-scale gradient via DA2)
        if self._last_depth_consistency_loss is not None:
            loss_dict['depth_consistency_loss'] = self._last_depth_consistency_loss
            self._last_depth_consistency_loss = None
        if self._last_depth_consistency_loss_applied is not None:
            loss_dict['depth_consistency_loss_applied'] = self._last_depth_consistency_loss_applied
            self._last_depth_consistency_loss_applied = None
        if self._last_depth_consistency_ssi is not None:
            loss_dict['depth_consistency_ssi'] = self._last_depth_consistency_ssi
            self._last_depth_consistency_ssi = None
        if self._last_depth_consistency_grad is not None:
            loss_dict['depth_consistency_grad'] = self._last_depth_consistency_grad
            self._last_depth_consistency_grad = None
        if self._last_depth_loss_bins is not None:
            for bin_key, loss_val in self._bin_finalize(self._last_depth_loss_bins).items():
                loss_dict[bin_key] = loss_val
            self._last_depth_loss_bins = None
        if self._last_diffusion_loss_bins is not None:
            for bin_key, loss_val in self._bin_finalize(self._last_diffusion_loss_bins).items():
                loss_dict[bin_key] = loss_val
            self._last_diffusion_loss_bins = None
        if self._last_timestep is not None:
            loss_dict['timestep'] = self._last_timestep
            self._last_timestep = None
        if self._last_id_sim is not None:
            loss_dict['id_sim'] = self._last_id_sim
            self._last_id_sim = None
        if self._last_id_clean_target is not None:
            loss_dict['id_clean_target'] = self._last_id_clean_target
            self._last_id_clean_target = None
        if self._last_id_clean_delta is not None:
            loss_dict['id_clean_delta'] = self._last_id_clean_delta
            self._last_id_clean_delta = None
        if self._last_id_sim_bins is not None:
            for bin_key, sim_val in self._bin_finalize(self._last_id_sim_bins).items():
                loss_dict[bin_key] = sim_val
            self._last_id_sim_bins = None
        if self._last_shape_sim_bins is not None:
            for bin_key, sim_val in self._bin_finalize(self._last_shape_sim_bins).items():
                loss_dict[bin_key] = sim_val
            self._last_shape_sim_bins = None

        # E-LatentLPIPS perceptual loss metrics
        if self.latent_perceptual_model is not None and self._latent_perceptual_accumulation_count > 0:
            n = self._latent_perceptual_accumulation_count
            loss_dict['latent_perceptual_loss'] = self._latent_perceptual_loss_accumulator / n
            loss_dict['latent_perceptual_loss_applied'] = self._latent_perceptual_loss_applied_accumulator / n

            preview_every = self.train_config.latent_perceptual_preview_every
            if preview_every > 0 and self.step_num % preview_every == 0 and self._lp_preview_cache is not None:
                try:
                    self._save_latent_perceptual_preview(
                        noise_pred=self._lp_preview_cache['noise_pred'],
                        noisy_latents=self._lp_preview_cache['noisy_latents'],
                        timesteps=self._lp_preview_cache['timesteps'],
                        batch_latents=self._lp_preview_cache['batch_latents'],
                    )
                except Exception as e:
                    print(f"Warning: failed to save latent perceptual preview: {e}")

            self._latent_perceptual_loss_accumulator = 0.0
            self._latent_perceptual_loss_applied_accumulator = 0.0
            self._latent_perceptual_accumulation_count = 0
            self._lp_preview_cache = None

        # Text token norm (reference for face_token_norm comparison)
        if hasattr(self.sd, 'unet') and hasattr(self.sd.unet, '_last_txt_token_norm'):
            loss_dict['txt_token_norm'] = self.sd.unet._last_txt_token_norm
            self.sd.unet._last_txt_token_norm = None

        # Cross-microbatch correction: every metric mirrored into the
        # MetricBuffer during this optimizer step gets its weighted mean
        # over all microbatches written back into loss_dict, overwriting
        # the single-microbatch shim that the legacy `_last_*` flush block
        # produced above. With `gradient_accumulation_steps == 1` this is a
        # no-op (the buffer's mean equals the single observation); with > 1
        # it fixes the "last microbatch wins" bug.
        if hasattr(self, '_metric_buffer'):
            buffered_scalars = self._metric_buffer.flush_scalars()
            for k, v in buffered_scalars.items():
                loss_dict[k] = v

            # Per-sample breakdowns: for metrics where SDTrainer called
            # `_record_sample(...)`, wrap the scalar in a `MetricValue`
            # that *behaves like a float* for every downstream consumer
            # (arithmetic, format strings, epoch accumulators, prog-bar
            # printf) but carries the JSON breakdown payload as an
            # attribute. The logger picks up the breakdown via
            # `_coerce_value` and writes it into `value_text`. Metrics
            # without per-sample collection keep their plain scalar form;
            # this is purely additive.
            from extensions_built_in.sd_trainer.metric_buffer import MetricValue
            buffered_per_sample = self._metric_buffer.flush_per_sample()
            for k, payload in buffered_per_sample.items():
                scalar = loss_dict.get(k)
                if scalar is None:
                    scalar = payload.get('mean')
                if scalar is None:
                    continue
                try:
                    loss_dict[k] = MetricValue(float(scalar), payload)
                except (TypeError, ValueError):
                    loss_dict[k] = scalar

        # Canonical naming dual-write: every legacy key gets a sibling
        # under the new `subsystem/kind/variant` namespace (see
        # `extensions_built_in.sd_trainer.metric_naming.CANONICAL_RENAMES`).
        # Existing dashboards keep reading the legacy key for one release;
        # the new metrics tab consumes the canonical key directly.
        from extensions_built_in.sd_trainer.metric_naming import apply_dual_write
        loss_dict = apply_dual_write(loss_dict)

        self.end_of_training_loop()

        return loss_dict
