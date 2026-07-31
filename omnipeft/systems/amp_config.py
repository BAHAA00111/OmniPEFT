"""
OmniPEFT Automatic Mixed Precision (AMP) Configuration & Execution Engine.

Manages dynamic precision context generation (bfloat16 / float16 / float32)
optimized for NVIDIA Ampere Tensor Cores (e.g., RTX 3080). Handles gradient scaling
and unscaling abstractions for fp16 backpropagation and native bfloat16 execution.
"""

from dataclasses import dataclass
import logging
from typing import Any, Optional

import torch
from torch.amp import GradScaler, autocast

logger = logging.getLogger(__name__)


@dataclass
class AMPConfig:
    """Dataclass encapsulating hardware-aware mixed precision settings."""

    enabled: bool = True
    device_type: str = "cuda"
    dtype_str: str = "auto"  # Options: 'auto', 'bfloat16', 'float16', 'float32'
    use_grad_scaler: bool = False
    autocast_dtype: torch.dtype = torch.bfloat16


class AMPManager:
    """Production-grade AMP Manager for model forward passes and gradient scaling."""

    def __init__(
        self,
        enabled: bool = True,
        preferred_dtype: str = "auto",
        device_type: str = "cuda",
    ) -> None:
        """Initialize AMP Manager and inspect GPU compute capability.

        Args:
            enabled: Master toggle for automatic mixed precision.
            preferred_dtype: Precision string ('auto', 'bfloat16', 'float16', 'float32').
            device_type: Target device backend (default: 'cuda').
        """
        self.device_type = device_type
        self.enabled = enabled and torch.cuda.is_available() and device_type == "cuda"

        # Determine exact target torch.dtype based on hardware and request
        self.autocast_dtype = self._resolve_precision(preferred_dtype)

        # bfloat16 has dynamic range identical to fp32 (8 exponent bits), so it does
        # NOT require gradient scaling. float16 has only 5 exponent bits, requiring GradScaler.
        self.use_grad_scaler = self.enabled and self.autocast_dtype == torch.float16

        self.config = AMPConfig(
            enabled=self.enabled,
            device_type=self.device_type,
            dtype_str=preferred_dtype,
            use_grad_scaler=self.use_grad_scaler,
            autocast_dtype=self.autocast_dtype,
        )

        # Initialize PyTorch GradScaler (PyTorch 2.0+ torch.amp.GradScaler)
        self.scaler: Optional[GradScaler] = (
            GradScaler(device=self.device_type, enabled=self.use_grad_scaler)
            if self.enabled
            else None
        )

        logger.info(
            "Initialized AMP Engine | Enabled: %s | DType: %s | GradScaler: %s | Device: %s",
            self.enabled,
            self.autocast_dtype,
            self.use_grad_scaler,
            self.device_type,
        )

    def _resolve_precision(self, preferred_dtype: str) -> torch.dtype:
        if not self.enabled:
            return torch.float32

        bf16_supported = torch.cuda.is_bf16_supported()

        if preferred_dtype == "auto":
            if bf16_supported:
                logger.info(
                    "NVIDIA Ampere/Ada/Hopper architecture detected. Selected native torch.bfloat16."
                )
                return torch.bfloat16
            else:
                logger.info(
                    "Legacy CUDA architecture detected (no native bf16). Fallback to torch.float16."
                )
                return torch.float16

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }

        selected = dtype_map.get(preferred_dtype.lower(), torch.bfloat16)

        if selected == torch.bfloat16 and not bf16_supported:
            logger.warning(
                "bfloat16 requested but not supported on current CUDA device. Falling back to float16."
            )
            return torch.float16

        return selected

    def autocast_context(self) -> Any:
        """Return a PyTorch autocast context manager for forward pass wrapping."""
        return autocast(
            device_type=self.device_type,
            dtype=self.autocast_dtype,
            enabled=self.enabled,
        )

    def backward(self, loss: torch.Tensor) -> None:
        """Execute backward pass on loss, handling scaling automatically if fp16 is active.

        Args:
            loss: Computed loss tensor.
        """
        if self.scaler is not None and self.use_grad_scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        model_parameters: Optional[Any] = None,
        max_grad_norm: Optional[float] = None,
    ) -> None:
        """Unscale gradients, optionally clip gradient norms, and step the optimizer safely.

        Args:
            optimizer: PyTorch optimizer instance.
            model_parameters: Model parameters iterable (required if clipping).
            max_grad_norm: Maximum gradient threshold for norm clipping.
        """
        if self.scaler is not None and self.use_grad_scaler:
            if max_grad_norm is not None and model_parameters is not None:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_parameters, max_grad_norm)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            if max_grad_norm is not None and model_parameters is not None:
                torch.nn.utils.clip_grad_norm_(model_parameters, max_grad_norm)
            optimizer.step()
