"""
OmniPEFT Engine - High-Performance Inference Runner & Benchmarking Pipeline

Provides production-grade generation utilities, batched streaming inference,
CUDA precise-event timing, and memory profiling across Base, Dynamic PEFT,
and Fused weight execution targets.
"""

from dataclasses import dataclass
import gc
import logging
import time
from typing import Any, Dict, List, Union

import torch
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Dataclass holding precise benchmarking and profiling metrics."""

    model_variant: str
    total_samples: int
    total_tokens_generated: int
    total_latency_seconds: float
    latency_per_token_ms: float
    tokens_per_second: float
    peak_vram_mb: float
    allocated_vram_mb: float


class InferenceRunner:
    """Production Inference Engine and Benchmarking Suite for OmniPEFT."""

    def __init__(
        self,
        model: Any,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Initialize the InferenceRunner with model, tokenizer, and system settings."""
        self.device = torch.device(device)
        self.model = model
        self.tokenizer = tokenizer
        self.torch_dtype = torch_dtype

        # Ensure tokenizer padding strategy is configured for batched inference
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Set model to strict evaluation mode
        if hasattr(self.model, "eval"):
            self.model.eval()

    @staticmethod
    def _clear_vram() -> None:
        """Purges VRAM and forces Python garbage collection."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> List[str]:
        """Executes single or batched text generation."""
        if isinstance(prompts, str):
            prompts = [prompts]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample and temperature > 0.0:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )
        else:
            gen_kwargs["do_sample"] = False

        # Use torch.amp.autocast with type assertion for Pyright/Mypy
        with torch.amp.autocast(device_type=self.device.type, dtype=self.torch_dtype):
            outputs = self.model.generate(**inputs, **gen_kwargs)  # type: ignore[operator]

        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded_outputs

    @torch.inference_mode()
    def benchmark_throughput(
        self,
        eval_prompts: List[str],
        variant_name: str = "fused_linear",
        max_new_tokens: int = 128,
        warmup_steps: int = 2,
    ) -> BenchmarkResult:
        """Executes high-precision CUDA event benchmarking."""
        logger.info(f"Starting benchmark suite for variant: '{variant_name}'...")

        self._clear_vram()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        # Warmup phase
        if warmup_steps > 0 and len(eval_prompts) > 0:
            warmup_batch = eval_prompts[: min(warmup_steps, len(eval_prompts))]
            _ = self.generate(warmup_batch, max_new_tokens=16, do_sample=False)
            if self.device.type == "cuda":
                torch.cuda.synchronize()

        # Initialize CUDA timer events if CUDA is active
        use_cuda_events = self.device.type == "cuda"
        if use_cuda_events:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        start_wall_time = time.perf_counter()
        total_tokens_generated = 0

        # Execute generation loop
        for prompt in eval_prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_length = inputs.input_ids.shape[1]

            with torch.amp.autocast(
                device_type=self.device.type, dtype=self.torch_dtype
            ):
                outputs = self.model.generate(  # type: ignore[operator]
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            generated_tokens = outputs.shape[1] - input_length
            total_tokens_generated += generated_tokens

        if use_cuda_events:
            end_event.record()
            torch.cuda.synchronize()
            cuda_latency_ms = start_event.elapsed_time(end_event)
            total_latency_sec = cuda_latency_ms / 1000.0
        else:
            end_wall_time = time.perf_counter()
            total_latency_sec = end_wall_time - start_wall_time

        tokens_per_second = (
            total_tokens_generated / total_latency_sec if total_latency_sec > 0 else 0.0
        )
        latency_per_token_ms = (
            (total_latency_sec / total_tokens_generated) * 1000.0
            if total_tokens_generated > 0
            else 0.0
        )

        # VRAM statistics
        if use_cuda_events:
            peak_vram_mb = round(
                torch.cuda.max_memory_allocated(self.device) / (1024**2), 2
            )
            allocated_vram_mb = round(
                torch.cuda.memory_allocated(self.device) / (1024**2), 2
            )
        else:
            peak_vram_mb = 0.0
            allocated_vram_mb = 0.0

        result = BenchmarkResult(
            model_variant=variant_name,
            total_samples=len(eval_prompts),
            total_tokens_generated=total_tokens_generated,
            total_latency_seconds=round(total_latency_sec, 4),
            latency_per_token_ms=round(latency_per_token_ms, 2),
            tokens_per_second=round(tokens_per_second, 2),
            peak_vram_mb=peak_vram_mb,
            allocated_vram_mb=allocated_vram_mb,
        )

        logger.info(
            f"[{variant_name.upper()}] Benchmarking Complete: "
            f"{result.tokens_per_second} tok/s | "
            f"{result.latency_per_token_ms} ms/tok | "
            f"Peak VRAM: {result.peak_vram_mb} MB"
        )

        return result
