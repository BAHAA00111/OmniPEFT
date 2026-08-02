"""
OmniPEFT CLI Script for Fusion Latency & Memory Profiling.

Usage:
    python omnipeft/scripts/benchmark_fusion.py
        --base-model-path Qwen/Qwen2.5-3B-Instruct
        --adapter-path ./checkpoints/checkpoint-final
        --batch-size 1
        --seq-len 512
        --warmup-steps 10
        --benchmark-steps 50
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, cast

import torch
from torch import nn
from peft import PeftModel
from transformers import AutoModelForCausalLM

from omnipeft.systems.benchmark_engine import LatencyBenchmarkEngine
from omnipeft.systems.weight_fusion import WeightFusionEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("omnipeft.scripts.benchmark_fusion")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for benchmarking fusion performance."""
    parser = argparse.ArgumentParser(
        description="OmniPEFT: Benchmark Latency & Memory Improvements of Fused Models."
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        required=True,
        help="HuggingFace hub identifier or local disk path to base model.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to adapter checkpoint directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for benchmark evaluation tensors.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Sequence length for benchmark evaluation tensors.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Number of initial CUDA warmup iterations.",
    )
    parser.add_argument(
        "--benchmark-steps",
        type=int,
        default=50,
        help="Number of benchmark evaluation iterations.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional destination path to export JSON report.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Target execution device ('cuda' or 'cpu').",
    )
    return parser.parse_args()


def run_fusion_benchmark(
    base_model_path: str,
    adapter_path: str,
    batch_size: int = 1,
    seq_len: int = 512,
    warmup_steps: int = 10,
    benchmark_steps: int = 50,
    output_json: str | None = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load model architecture, generate fused variant, and execute full latency evaluation."""
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    logger.info("Loading Base Model from %s...", base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    logger.info("Attaching Adapter Checkpoint from %s...", adapter_path)
    unfused_peft = PeftModel.from_pretrained(
        model=base_model,
        model_id=adapter_path,
        is_trainable=False,
    )
    unfused_peft.to(device=device)
    unfused_peft.eval()

    logger.info("Constructing Fused Model Variant...")
    fusion_engine = WeightFusionEngine(target_dtype=dtype)

    # Cast unfused model to nn.Module to bypass Pylance union ambiguities
    unfused_module = cast(nn.Module, unfused_peft)
    fused_model = fusion_engine.fuse_peft_model(unfused_module, safe_merge=True)
    fused_model.eval()

    vocab_size = getattr(fused_model.config, "vocab_size", 32000)

    # Initialize Benchmark Suite
    benchmarker = LatencyBenchmarkEngine(
        warmup_steps=warmup_steps,
        benchmark_steps=benchmark_steps,
        device=device,
    )

    results = benchmarker.compare_unfused_vs_fused(
        unfused_model=unfused_module,
        fused_model=fused_model,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
    )

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Benchmark report exported to %s", out_path.resolve())

    return results


def main() -> None:
    args = parse_args()
    try:
        run_fusion_benchmark(
            base_model_path=args.base_model_path,
            adapter_path=args.adapter_path,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            warmup_steps=args.warmup_steps,
            benchmark_steps=args.benchmark_steps,
            output_json=args.output_json,
            device=args.device,
        )
    except Exception:
        logger.exception("Benchmark CLI Execution Failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
