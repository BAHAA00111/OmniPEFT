"""
OmniPEFT QLoRA 4-bit Quantized Model & Trainer Engine.

Loads open-weights Causal LMs in 4-bit NormalFloat (NF4) quantization via bitsandbytes,
prepares model weights for k-bit training, attaches low-rank adaptation (PEFT LoRA) matrices,
and enables gradient checkpointing to constrain peak VRAM consumption below 8GB.
"""

from dataclasses import dataclass, field
import logging
from typing import List, Literal, Optional, Tuple, Union

from peft import (
    LoraConfig,
    PeftMixedModel,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


@dataclass
class QLoRAConfig:
    """Dataclass holding QLoRA quantization and PEFT adapter hyper-parameters."""

    # Base Model Configuration
    model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct"
    torch_dtype: torch.dtype = torch.bfloat16
    device_map: str = "auto"
    trust_remote_code: bool = True

    # 4-bit BitsAndBytes Configuration
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16
    bnb_4bit_use_double_quant: bool = True

    # LoRA Adapter Configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: Union[TaskType, str] = TaskType.CAUSAL_LM

    # Memory Optimizations
    use_gradient_checkpointing: bool = True


class QLoRAModelBuilder:
    """Production builder for 4-bit NF4 quantized base models with PEFT LoRA adapters."""

    def __init__(self, config: Optional[QLoRAConfig] = None) -> None:
        """Initialize QLoRA model builder with configuration settings.

        Args:
            config: Optional QLoRAConfig instance; defaults to standard QLoRA hyper-parameters.
        """
        self.config = config or QLoRAConfig()

    def build_bitsandbytes_config(self) -> BitsAndBytesConfig:
        """Construct Hugging Face BitsAndBytesConfig for NF4 double quantization.

        Returns:
            BitsAndBytesConfig configured for sub-8GB VRAM footprint.
        """
        return BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=self.config.bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
        )

    def build_lora_config(self) -> LoraConfig:
        """Construct PEFT LoraConfig for target linear projection layers.

        Returns:
            LoraConfig targeting specified linear projection modules.
        """
        return LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.bias,
            task_type=self.config.task_type,
        )

    def print_trainable_parameters(self, model: nn.Module) -> Tuple[int, int, float]:
        """Calculate and log total vs. trainable parameter counts and percentage.

        Args:
            model: PyTorch/PEFT model instance.

        Returns:
            Tuple of (trainable_params, total_params, trainable_percentage).
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        percentage = (trainable_params / all_param) * 100 if all_param > 0 else 0.0
        logger.info(
            "Model Parameter Profile | Trainable: %s | Total: %s | Trainable %%: %.4f%%",
            f"{trainable_params:,}",
            f"{all_param:,}",
            percentage,
        )
        return trainable_params, all_param, percentage

    def load_model_and_tokenizer(
        self,
    ) -> Tuple[Union[PeftModel, PeftMixedModel], PreTrainedTokenizer]:
        """Load 4-bit NF4 quantized base model, apply PEFT LoRA adapter, and configure tokenizer.

        Returns:
            Tuple of (PeftModel | PeftMixedModel, PreTrainedTokenizer) configured for training.
        """
        logger.info("Initializing 4-bit NF4 Quantization Config...")
        bnb_config = self.build_bitsandbytes_config()

        logger.info(
            "Loading base model: %s in 4-bit...", self.config.model_name_or_path
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            quantization_config=bnb_config,
            device_map=self.config.device_map,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=self.config.torch_dtype,
        )

        logger.info(
            "Loading tokenizer matching architecture: %s...",
            self.config.model_name_or_path,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Prepare model for 4-bit quantization and freeze non-adapter parameters
        logger.info("Preparing model for k-bit training...")
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
        )

        # Explicitly enable gradient checkpointing on base model if requested
        if self.config.use_gradient_checkpointing:
            logger.info("Enabling gradient checkpointing for VRAM optimization...")
            base_model.gradient_checkpointing_enable()

        # Attach LoRA adapters to linear target projection layers
        logger.info(
            "Attaching LoRA adapters (r=%d, alpha=%d, targets=%s)...",
            self.config.lora_r,
            self.config.lora_alpha,
            self.config.target_modules,
        )
        lora_config = self.build_lora_config()
        model = get_peft_model(base_model, lora_config)

        # Log parameter profile
        self.print_trainable_parameters(model)

        return model, tokenizer
