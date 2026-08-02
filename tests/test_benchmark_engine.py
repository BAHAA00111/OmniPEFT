"""
OmniPEFT Unit Tests for Latency Benchmarking Engine.

Verifies statistical latency aggregation, metric structure integrity,
and execution speedup comparisons on synthetic PyTorch modules.
"""

from typing import cast

import torch
from torch import nn

from omnipeft.systems.benchmark_engine import LatencyBenchmarkEngine


class MockLinearLM(nn.Module):
    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.proj(x)
        out = self.head(x)
        return out


def test_measure_latency_structure() -> None:
    model = MockLinearLM()
    engine = LatencyBenchmarkEngine(warmup_steps=2, benchmark_steps=5, device="cpu")

    dummy_ids = torch.randint(0, 1000, (2, 16), dtype=torch.long)
    metrics = engine.measure_latency(model, dummy_ids)

    assert "mean_latency_ms" in metrics
    assert "p50_latency_ms" in metrics
    assert "p90_latency_ms" in metrics
    assert "p99_latency_ms" in metrics
    assert "peak_vram_mb" in metrics

    assert metrics["mean_latency_ms"] > 0.0
    assert metrics["p50_latency_ms"] <= metrics["p90_latency_ms"]
    assert metrics["p90_latency_ms"] <= metrics["p99_latency_ms"]


def test_compare_unfused_vs_fused_metrics() -> None:
    unfused_model = MockLinearLM()
    fused_model = MockLinearLM()

    engine = LatencyBenchmarkEngine(warmup_steps=2, benchmark_steps=5, device="cpu")

    results = engine.compare_unfused_vs_fused(
        unfused_model=unfused_model,
        fused_model=fused_model,
        batch_size=1,
        seq_len=16,
        vocab_size=1000,
    )

    assert "unfused" in results
    assert "fused" in results
    assert "speedup_factor" in results
    assert "vram_saved_mb" in results

    speedup = cast(float, results["speedup_factor"])
    assert speedup > 0.0
