"""
Measures forward-pass latency, memory allocation, and token throughput
comparing standard dynamic PEFT adapter models against fused zero-latency models.
"""

import gc
import logging
import time
from typing import Dict, List, Optional, Union

import torch
from torch import nn

logger = logging.getLogger(__name__)


class LatencyBenchmarkEngine:
    def __init__(
        self,
        warmup_steps: int = 10,
        benchmark_steps: int = 50,
        device: Optional[str] = None,
    ) -> None:
        """Initialize benchmark engine configuration.

        Args:
            warmup_steps: Iterations to warm up CUDA kernels and caching allocators.
            benchmark_steps: Measurement iterations for percentile statistical calculation.
            device: Target device string ('cuda', 'cpu', 'cuda:0'). Defaults to CUDA if available.
        """
        self.warmup_steps = warmup_steps
        self.benchmark_steps = benchmark_steps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def measure_latency(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Measure precise execution latency and peak VRAM allocation for a given module.

        Args:
            model: PyTorch module (fused or unfused PEFT model) to evaluate.
            input_ids: Input token tensor of shape (batch_size, seq_len).
            attention_mask: Optional attention mask tensor of shape (batch_size, seq_len).

        Returns:
            Dictionary containing latency percentiles (p50, p90, p99, mean) and peak VRAM allocation.
        """
        model.eval()
        model.to(self.device)
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Force garbage collection & reset CUDA memory stats baseline
        gc.collect()
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Warmup execution loop
        with torch.no_grad():
            for _ in range(self.warmup_steps):
                if attention_mask is not None:
                    _ = model(input_ids, attention_mask=attention_mask)
                else:
                    _ = model(input_ids)
                if torch.cuda.is_available() and self.device.startswith("cuda"):
                    torch.cuda.synchronize()

        timings: List[float] = []

        # Precision benchmark timing loop
        with torch.no_grad():
            for _ in range(self.benchmark_steps):
                if torch.cuda.is_available() and self.device.startswith("cuda"):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                    start_event.record()
                    if attention_mask is not None:
                        _ = model(input_ids, attention_mask=attention_mask)
                    else:
                        _ = model(input_ids)
                    end_event.record()

                    torch.cuda.synchronize()
                    elapsed_ms = start_event.elapsed_time(end_event)
                else:
                    t0 = time.perf_counter()
                    if attention_mask is not None:
                        _ = model(input_ids, attention_mask=attention_mask)
                    else:
                        _ = model(input_ids)
                    t1 = time.perf_counter()
                    elapsed_ms = (t1 - t0) * 1000.0

                timings.append(elapsed_ms)

        # Statistical aggregation
        timings_tensor = torch.tensor(timings, dtype=torch.float32)
        p50 = float(torch.quantile(timings_tensor, 0.50).item())
        p90 = float(torch.quantile(timings_tensor, 0.90).item())
        p99 = float(torch.quantile(timings_tensor, 0.99).item())
        mean_ms = float(torch.mean(timings_tensor).item())

        peak_vram_mb = 0.0
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

        return {
            "mean_latency_ms": mean_ms,
            "p50_latency_ms": p50,
            "p90_latency_ms": p90,
            "p99_latency_ms": p99,
            "peak_vram_mb": peak_vram_mb,
        }

    def compare_unfused_vs_fused(
        self,
        unfused_model: nn.Module,
        fused_model: nn.Module,
        batch_size: int = 1,
        seq_len: int = 512,
        vocab_size: int = 32000,
    ) -> Dict[str, Union[Dict[str, float], float]]:
        dummy_input = torch.randint(
            0, vocab_size, (batch_size, seq_len), dtype=torch.long
        )

        logger.info("Executing benchmark for Unfused Adapter Model...")
        unfused_metrics = self.measure_latency(unfused_model, dummy_input)

        logger.info("Executing benchmark for Fused Standalone Model...")
        fused_metrics = self.measure_latency(fused_model, dummy_input)

        speedup = unfused_metrics["mean_latency_ms"] / max(
            fused_metrics["mean_latency_ms"], 1e-6
        )
        vram_reduction = unfused_metrics["peak_vram_mb"] - fused_metrics["peak_vram_mb"]

        results: Dict[str, Union[Dict[str, float], float]] = {
            "unfused": unfused_metrics,
            "fused": fused_metrics,
            "speedup_factor": speedup,
            "vram_saved_mb": vram_reduction,
        }

        logger.info(
            "Benchmark Execution Complete | Speedup: %.2fx | Memory Reduction: %.2f MB",
            speedup,
            vram_reduction,
        )
        return results
