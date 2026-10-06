import os
from collections import OrderedDict
from typing import List, Optional

import huggingface_hub
import torch
from safetensors.torch import load_file

from toolkit.config_modules import ModelMergeLoraConfig, NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.models.lokr import factorization
from toolkit.print import print_acc

HF_TOKEN = os.getenv("HF_TOKEN", None)


def _resolve_lora_path(path: str) -> str:
    if os.path.exists(path):
        return path

    parts = path.split("/")
    if len(parts) != 3:
        raise ValueError(
            f"Merge LoRA path {path!r} is not a local file or hub path. "
            "Use a local path or 'owner/repo/filename.safetensors'."
        )

    repo_id = "/".join(parts[:2])
    filename = parts[2]
    try:
        return huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=HF_TOKEN,
        )
    except Exception as e:
        raise ValueError(f"Failed to download merge LoRA {path!r}: {e}") from e


def _detect_lora_dim(state_dict: dict) -> int:
    for key, value in state_dict.items():
        if key.endswith("lora_down.weight") or key.endswith("lora_A.weight"):
            return int(value.shape[0])
    raise ValueError(
        "Could not infer LoRA rank. Base LoRA merges currently support standard "
        "LoRA files with lora_down/lora_up or PEFT lora_A/lora_B weights."
    )


def _is_full_rank_lokr(state_dict: dict) -> bool:
    return any(key.endswith(".lokr_w1") or key.endswith(".lokr_w2") for key in state_dict.keys())


def _detect_lokr_dim(state_dict: dict) -> int:
    for key, value in state_dict.items():
        if key.endswith(".lokr_w1_a"):
            return int(value.shape[1])
        if key.endswith(".lokr_w1_b"):
            return int(value.shape[0])
        if key.endswith(".lokr_w2_a"):
            return int(value.shape[1])
        if key.endswith(".lokr_w2_b"):
            return int(value.shape[0])
    raise ValueError(
        "Could not infer LoKr rank. Set model.merge_loras[].lokr_factor for "
        "non-standard LoKr files or use a full-rank LoKr with lokr_w1/lokr_w2 weights."
    )


def _module_dims(module) -> Optional[tuple[int, int]]:
    if isinstance(module, torch.nn.Linear):
        return module.out_features, module.in_features
    if isinstance(module, torch.nn.Conv2d):
        return module.out_channels, module.in_channels
    return None


def _infer_lokr_factor_from_targets(
    state_dict: dict,
    model_to_train,
) -> Optional[int]:
    module_map = dict(model_to_train.named_modules())
    candidates = set()

    for key, value in state_dict.items():
        if not key.endswith(".lokr_w1") or len(value.shape) < 2:
            continue

        module_name = key[: -len(".lokr_w1")]
        for prefix in ("transformer.", "diffusion_model.", "unet."):
            if module_name.startswith(prefix):
                module_name = module_name[len(prefix):]
                break

        module = module_map.get(module_name)
        dims = _module_dims(module) if module is not None else None
        if dims is None:
            continue

        out_dim, in_dim = dims
        out_l, in_m = int(value.shape[0]), int(value.shape[1])
        max_factor = max(out_l, in_m, 128)
        for factor in range(1, max_factor + 1):
            if (
                factorization(out_dim, factor)[0] == out_l
                and factorization(in_dim, factor)[0] == in_m
            ):
                candidates.add(factor)
                break

    if len(candidates) == 1:
        return candidates.pop()
    return None


def _get_network_config_for_merge(
    sd,
    merge_lora: ModelMergeLoraConfig,
    state_dict: dict,
    model_to_train,
) -> NetworkConfig:
    if any("lokr_" in key for key in state_dict.keys()):
        full_rank = _is_full_rank_lokr(state_dict)
        dim = 4 if full_rank else _detect_lokr_dim(state_dict)
        lokr_factor = merge_lora.lokr_factor
        if lokr_factor is None:
            converted_state_dict = sd.convert_lora_weights_before_load(state_dict)
            lokr_factor = _infer_lokr_factor_from_targets(
                converted_state_dict,
                model_to_train,
            )
        if lokr_factor is None:
            lokr_factor = 8

        return NetworkConfig(
            type="lokr",
            linear=dim,
            linear_alpha=dim,
            lokr_full_rank=full_rank,
            lokr_factor=lokr_factor,
            transformer_only=True,
        )

    dim = _detect_lora_dim(state_dict)
    return NetworkConfig(
        type="lora",
        linear=dim,
        linear_alpha=dim,
        transformer_only=True,
    )


def _has_quantized_target(network: LoRASpecialNetwork) -> bool:
    for module in network.get_all_modules():
        org_module = module.org_module[0]
        state_dict = org_module.state_dict()
        if "weight._data" in state_dict:
            return True
    return False


def _meaningful_extra_keys(extra_weights) -> List[str]:
    if not extra_weights:
        return []
    keys = []
    for key in extra_weights.keys():
        if any(
            marker in key
            for marker in (
                "lora_down",
                "lora_up",
                "lora_A",
                "lora_B",
                "lokr_",
                "lycoris_",
            )
        ):
            keys.append(key)
    return keys


def merge_loras_into_base_model(
    sd,
    merge_loras: List[ModelMergeLoraConfig],
    model_to_train=None,
):
    if not merge_loras:
        return []

    merged = []
    text_encoder = getattr(sd, "text_encoder", None)
    model_to_train = model_to_train if model_to_train is not None else sd.get_model_to_train()
    network_kwargs = {}
    if hasattr(sd, "target_lora_modules"):
        network_kwargs["target_lin_modules"] = sd.target_lora_modules

    for merge_lora in merge_loras:
        resolved_path = _resolve_lora_path(merge_lora.path)
        state_dict = load_file(resolved_path)
        network_config = _get_network_config_for_merge(
            sd,
            merge_lora,
            state_dict,
            model_to_train,
        )
        network = LoRASpecialNetwork(
            text_encoder=text_encoder,
            unet=model_to_train,
            lora_dim=network_config.linear,
            multiplier=1.0,
            alpha=network_config.linear_alpha,
            train_unet=True,
            train_text_encoder=False,
            is_sdxl=sd.model_config.is_xl or sd.model_config.is_ssd,
            is_v2=sd.model_config.is_v2,
            is_v3=sd.model_config.is_v3,
            is_pixart=sd.model_config.is_pixart,
            is_auraflow=sd.model_config.is_auraflow,
            is_flux=sd.model_config.is_flux,
            is_lumina2=sd.model_config.is_lumina2,
            is_ssd=sd.model_config.is_ssd,
            is_vega=sd.model_config.is_vega,
            network_config=network_config,
            network_type=network_config.type,
            transformer_only=network_config.transformer_only,
            is_transformer=sd.is_transformer,
            base_model=sd,
            **network_kwargs,
        )
        network.apply_to(
            text_encoder,
            model_to_train,
            apply_text_encoder=False,
            apply_unet=True,
        )
        network.force_to(sd.device_torch, dtype=sd.torch_dtype)
        network.eval()
        network._update_torch_multiplier()

        extra_weights = network.load_weights(state_dict)
        extra_lora_keys = _meaningful_extra_keys(extra_weights)
        if extra_lora_keys:
            preview = ", ".join(extra_lora_keys[:8])
            raise ValueError(
                f"Merge LoRA {merge_lora.path!r} contains adapter keys that did not "
                f"match the loaded model: {preview}"
            )

        if _has_quantized_target(network):
            raise ValueError(
                f"Cannot merge LoRA {merge_lora.path!r} into quantized target modules. "
                "Disable model quantization for this job, or use an already merged base model."
            )

        print_acc(
            f"Merging base LoRA {merge_lora.path} at weight {merge_lora.weight:g}"
        )
        network.merge_in(merge_weight=merge_lora.weight)
        network.is_active = False
        if not hasattr(sd, "base_lora_merge_networks"):
            sd.base_lora_merge_networks = []
        sd.base_lora_merge_networks.append(network)

        merged.append(
            OrderedDict(
                [
                    ("path", merge_lora.path),
                    ("resolved_path", resolved_path),
                    ("weight", merge_lora.weight),
                    ("type", network_config.type),
                ]
            )
        )

    sd.base_lora_merges_metadata = merged
    return merged
