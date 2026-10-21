"""Loading Qwen3-VL on 24 GB and turning on LoRA.

Presets are deliberately conservative: a first run that fits is worth more than
a run that is 10% faster and OOMs.

    preset        model                       quant   image tokens   max_len
    4b_bf16       Qwen3-VL-4B-Instruct        -       512            1536
    8b_4bit       Qwen3-VL-8B-Instruct        4-bit   512            1536
    8b_4bit_768   Qwen3-VL-8B-Instruct        4-bit   768            1536
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

PRESETS: Dict[str, Dict[str, Any]] = {
    "4b_bf16": {
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "load_in_4bit": False,
        "image_tokens": 512,
        "max_length": 1536,
    },
    "8b_4bit": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "load_in_4bit": True,
        "image_tokens": 512,
        "max_length": 1536,
    },
    "8b_4bit_768": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "load_in_4bit": True,
        "image_tokens": 768,
        "max_length": 1536,
    },
}


@dataclass
class ModelConfig:
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    load_in_4bit: bool = False
    image_tokens: int = 512
    max_length: int = 1536
    attn_implementation: str = "sdpa"   # flash-attn on 4090 needs a source build
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = True

    @classmethod
    def from_preset(cls, name: str) -> "ModelConfig":
        if name not in PRESETS:
            raise KeyError(f"unknown preset {name}; choose from {sorted(PRESETS)}")
        return cls(**PRESETS[name])

    @property
    def max_pixels(self) -> int:
        # Qwen-VL tokens are 28x28 pixels after the 2x2 merge
        return int(28 * 28 * self.image_tokens)


def _has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


def load_model(cfg: ModelConfig):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    kwargs: Dict[str, Any] = {"device_map": {"": 0}, "low_cpu_mem_usage": True}
    attn = cfg.attn_implementation
    if attn == "flash_attention_2" and not _has_flash_attn():
        print("[warn] flash-attn missing, falling back to sdpa")
        attn = "sdpa"
    kwargs["attn_implementation"] = attn

    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            cfg.model_id, dtype=torch.bfloat16, **kwargs
        )
    except TypeError:  # transformers < 4.56 spells it torch_dtype
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            cfg.model_id, torch_dtype=torch.bfloat16, **kwargs
        )
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    return model, processor


def attach_lora(model, cfg: ModelConfig):
    from peft import LoraConfig, get_peft_model

    if cfg.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
    )
    return model


def prepare_for_training(model, cfg: ModelConfig):
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # without this the LoRA gradients do not reach the checkpointed blocks
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.config.use_cache = False
    return model


def image_token_id(model) -> int:
    cfg = model.config
    for attr in ("image_token_id",):
        value = getattr(cfg, attr, None)
        if value is not None:
            return int(value)
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None and getattr(text_cfg, "image_token_id", None) is not None:
        return int(text_cfg.image_token_id)
    raise ValueError("cannot find image_token_id on the model config")


def gpu_report() -> str:
    import torch

    if not torch.cuda.is_available():
        return "CUDA unavailable"
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"{name} | {total:.1f} GB | torch {torch.__version__}"


def set_cache_env(cache_dir: str) -> None:
    for key in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE"):
        os.environ.setdefault(key, cache_dir)

