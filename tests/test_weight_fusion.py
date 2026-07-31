"""
OmniPEFT Unit Tests for Weight Fusion Engine.

Verifies numerical parity between dynamic LoRA adapter forward passes
and fused static linear layers: ||Model_adapter(x) - Model_fused(x)|| < epsilon
"""

from typing import Generator, Tuple, cast

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import PreTrainedModel

from omnipeft.systems.weight_fusion import WeightFusionEngine


class ToyCausalLM(nn.Module):
    """Simple synthetic model with linear layers for fast deterministic testing."""

    def __init__(self, in_features: int = 64, hidden_dim: int = 128, out_features: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        return out


@pytest.fixture
def dummy_model_and_adapter() -> Generator[Tuple[nn.Module, torch.Tensor], None, None]:
    """Fixture supplying a synthetic model with attached LoRA adapter and random inputs."""
    torch.manual_seed(42)
    base_model = ToyCausalLM(in_features=32, hidden_dim=64, out_features=32)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,  # Passed as int to satisfy LoraConfig signature
        target_modules=["fc1", "fc2"],
        lora_dropout=0.0,  # Zero dropout essential for deterministic parity verification
        bias="none",
    )

    # Cast base_model to PreTrainedModel to satisfy PEFT's strict type stubs
    raw_peft = get_peft_model(cast(PreTrainedModel, base_model), lora_config)
    peft_model = cast(nn.Module, raw_peft)
    peft_model.eval()

    # Generate synthetic input batch: shape (batch_size=2, seq_len=4, dim=32)
    dummy_input = torch.randn(2, 4, 32, dtype=torch.float32)

    yield peft_model, dummy_input


def test_compute_delta_weight_math() -> None:
    """Test standalone Delta W matrix math: Delta W = (alpha / r) * (B @ A)."""
    r = 4
    alpha = 16.0
    in_features = 8
    out_features = 12

    # Deterministic Low-Rank matrices
    lora_a = torch.randn(r, in_features, dtype=torch.float32)
    lora_b = torch.randn(out_features, r, dtype=torch.float32)

    delta_w = WeightFusionEngine.compute_delta_weight(
        lora_a=lora_a,
        lora_b=lora_b,
        r=r,
        alpha=alpha,
    )

    # Expected Delta W computation
    expected_scaling = alpha / r
    expected_delta_w = expected_scaling * torch.matmul(lora_b, lora_a)

    assert delta_w.shape == (out_features, in_features)
    torch.testing.assert_close(delta_w, expected_delta_w, rtol=1e-5, atol=1e-5)


def test_numerical_parity_adapter_vs_fused(
    dummy_model_and_adapter: Tuple[nn.Module, torch.Tensor]
) -> None:
    """Verify numerical output parity: ||Model_adapter(x) - Model_fused(x)|| < epsilon."""
    peft_model, x = dummy_model_and_adapter

    # 1. Forward pass prior to weight fusion
    with torch.no_grad():
        logits_adapter = peft_model(x)

    # 2. Perform zero-latency weight fusion
    fusion_engine = WeightFusionEngine(target_dtype=torch.float32)
    fused_model = fusion_engine.fuse_peft_model(peft_model, safe_merge=True)
    fused_model.eval()

    # 3. Forward pass on fused linear architecture
    with torch.no_grad():
        logits_fused = fused_model(x)

    # 4. Assert strict numerical parity threshold (epsilon = 1e-4)
    abs_diff = torch.abs(logits_adapter - logits_fused)
    max_diff = torch.max(abs_diff).item()

    assert max_diff < 1e-4, f"Parity violation: Max absolute error {max_diff} exceeded tolerance 1e-4"
    torch.testing.assert_close(logits_adapter, logits_fused, rtol=1e-4, atol=1e-4)


def test_structure_replacement_contains_no_lora_layers(
    dummy_model_and_adapter: Tuple[nn.Module, torch.Tensor]
) -> None:
    """Assert all LoRA dynamic adapter structures are eliminated after fusion."""
    peft_model, _ = dummy_model_and_adapter

    fusion_engine = WeightFusionEngine(target_dtype=torch.float32)
    fused_model = fusion_engine.fuse_peft_model(peft_model, safe_merge=True)

    # Validate module tree contains zero remaining adapter layers
    for name, module in fused_model.named_modules():
        module_type_str = str(type(module)).lower()
        assert "lora" not in module_type_str, f"Found un-fused LoRA layer at: {name}"