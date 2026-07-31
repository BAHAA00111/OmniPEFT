"""
OmniPEFT Dynamic Linear Weight Fusion Engine.

Eliminates inference latency overhead by mathematically merging low-rank LoRA adapter
matrices (B @ A) directly into base model projection parameters (W_fused = W_base + (alpha / r) * B @ A)
and replacing adapter/quantized layers with standard PyTorch nn.Linear modules for zero-overhead deployment.
"""

import logging
from typing import Union, cast

import torch
from torch import nn
from peft import PeftMixedModel, PeftModel
from peft.tuners.lora import LoraLayer
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

ModelType = Union[PeftModel, PeftMixedModel, PreTrainedModel, nn.Module]


class WeightFusionEngine:
    def __init__(self, target_dtype: torch.dtype = torch.bfloat16) -> None:
        self.target_dtype = target_dtype
        logger.info(
            "Initialized WeightFusionEngine | Target Precision: %s", self.target_dtype
        )

    @staticmethod
    def compute_delta_weight(
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        r: int,
        alpha: float,
    ) -> torch.Tensor:
        """Compute Delta W matrix update: Delta W = (alpha / r) * (B @ A).

        Args:
            lora_a: Low-rank weight matrix A with shape (r, in_features).
            lora_b: Low-rank weight matrix B with shape (out_features, r).
            r: Rank dimension of LoRA adapter.
            alpha: Scaling hyperparameter alpha.

        Returns:
            Delta W weight update matrix with shape (out_features, in_features).
        """
        scaling = alpha / r if r > 0 else 1.0

        # Ensure matrix multiplication precision alignment
        b_float = lora_b.to(dtype=torch.float32)
        a_float = lora_a.to(dtype=torch.float32)

        # Compute delta W = (alpha / r) * (B @ A)
        delta_w: torch.Tensor = scaling * torch.matmul(b_float, a_float)
        return delta_w

    def fuse_peft_model(
        self,
        model: ModelType,
        safe_merge: bool = True,
    ) -> nn.Module:
        """Fuse PEFT LoRA adapters into base model weights and return an unwrapped model.

        Args:
            model: PeftModel or nn.Module containing trained LoRA adapters.
            safe_merge: Validate merged parameter norms and avoid NaN/Inf corruptions.

        Returns:
            Unwrapped PyTorch model with standard nn.Linear layers for zero-overhead deployment.
        """
        logger.info("Initiating zero-latency weight fusion protocol...")

        # 1. Primary Pathway: Native PEFT Merge & Unload
        if isinstance(model, (PeftModel, PeftMixedModel)):
            try:
                logger.info("Executing native PEFT model.merge_and_unload()...")
                merge_fn = getattr(model, "merge_and_unload", None)
                if callable(merge_fn):
                    merged = merge_fn(safe_merge=safe_merge)
                    if isinstance(merged, nn.Module):
                        fused_module = cast(nn.Module, merged)
                        fused_module.to(dtype=self.target_dtype)
                        logger.info("Native PEFT weight fusion executed successfully.")
                        return fused_module
            except (AttributeError, RuntimeError, ValueError) as e:
                logger.warning("Native merge_and_unload failed (%s)...", e)

        # 2. Fallback Pathway: Manual Layer-by-Layer Matrix Fusion
        unwrapped: nn.Module
        if hasattr(model, "module") and isinstance(model.module, nn.Module):
            unwrapped = cast(nn.Module, model.module)
        else:
            unwrapped = cast(nn.Module, model)

        fused_count = self._fuse_module_recursive(unwrapped)

        logger.info(
            "Layer-by-layer fusion completed | Total linear projections replaced: %d",
            fused_count,
        )
        return unwrapped

    def _fuse_module_recursive(self, parent_module: nn.Module) -> int:
        """Recursively scan and replace LoraLayer targets with fused nn.Linear instances.

        Args:
            parent_module: Parent PyTorch module to inspect.

        Returns:
            Count of fused linear layers replaced.
        """
        fused_count = 0

        for name, child in list(parent_module.named_children()):
            if isinstance(child, LoraLayer) and hasattr(child, "lora_A"):
                adapter_name = getattr(child, "active_adapter", "default")
                if isinstance(adapter_name, (list, tuple)):
                    adapter_name = adapter_name[0]

                lora_a_dict = getattr(child, "lora_A", {})
                lora_b_dict = getattr(child, "lora_B", {})

                if adapter_name in lora_a_dict and adapter_name in lora_b_dict:
                    lora_a_mod = lora_a_dict[adapter_name]
                    lora_b_mod = lora_b_dict[adapter_name]

                    # Safely extract weight Tensors
                    a_weight = getattr(lora_a_mod, "weight", None)
                    b_weight = getattr(lora_b_mod, "weight", None)

                    if isinstance(a_weight, torch.Tensor) and isinstance(
                        b_weight, torch.Tensor
                    ):
                        r_dict = getattr(child, "r", {})
                        r = (
                            r_dict.get(adapter_name, 16)
                            if isinstance(r_dict, dict)
                            else 16
                        )

                        alpha_dict = getattr(child, "lora_alpha", {})
                        alpha = (
                            alpha_dict.get(adapter_name, 32.0)
                            if isinstance(alpha_dict, dict)
                            else 32.0
                        )

                        # Calculate Delta W
                        delta_w = self.compute_delta_weight(
                            a_weight, b_weight, r, float(alpha)
                        )

                        # Extract base layer and base weight tensor
                        base_layer = (
                            child.get_base_layer()
                            if hasattr(child, "get_base_layer")
                            else child
                        )
                        base_w_raw = getattr(base_layer, "weight", None)

                        if isinstance(base_w_raw, torch.Tensor):
                            base_w = base_w_raw.to(
                                dtype=torch.float32, device=delta_w.device
                            )
                            fused_w = (base_w + delta_w).to(dtype=self.target_dtype)

                            out_features, in_features = fused_w.shape

                            base_bias = getattr(base_layer, "bias", None)
                            has_bias = isinstance(base_bias, torch.Tensor)

                            # Construct standard zero-overhead nn.Linear replacement
                            fused_linear = nn.Linear(
                                in_features=in_features,
                                out_features=out_features,
                                bias=has_bias,
                                device=fused_w.device,
                                dtype=self.target_dtype,
                            )

                            fused_linear.weight.data.copy_(fused_w)
                            if has_bias and isinstance(base_bias, torch.Tensor):
                                fused_linear.bias.data.copy_(
                                    base_bias.to(dtype=self.target_dtype)
                                )

                            # Replace LoraLayer with standard nn.Linear
                            setattr(parent_module, name, fused_linear)
                            fused_count += 1
            else:
                # Recurse through sub-modules
                fused_count += self._fuse_module_recursive(child)

        return fused_count
