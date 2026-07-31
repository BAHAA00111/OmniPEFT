"""
OmniPEFT Standalone Weight Fusion CLI Script.

Converts adapter checkpoints and base models into unified, zero-overhead standalone
deployable model artifacts.

Usage:
    python omnipeft/scripts/merge_weights.py 
        --base-model-path Qwen/Qwen2.5-3B-Instruct 
        --adapter-path ./checkpoints/checkpoint-final 
        --output-dir ./fused_models/qwen2.5-3b-fused 
        --precision bfloat16
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import  Dict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from omnipeft.systems.weight_fusion import WeightFusionEngine

# Configure production logging layout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("omnipeft.scripts.merge_weights")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OmniPEFT: Merge LoRA Adapter Weights into Base Model for Zero-Overhead Deployment."
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
        help="Path to directory containing LoRA adapter weights (adapter_model.bin / adapter_config.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Target export directory for fused standalone model artifacts.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Floating point precision for saved model weights.",
    )
    parser.add_argument(
        "--safe-serialization",
        action="store_true",
        default=True,
        help="Export checkpoint using Safetensors format.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to perform matrix fusion computations on.",
    )
    return parser.parse_args()


def resolve_dtype(precision_str: str) -> torch.dtype:
    mapping: Dict[str, torch.dtype] = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(precision_str, torch.bfloat16)


def run_weight_fusion(
    base_model_path: str,
    adapter_path: str,
    output_dir: str,
    precision: str = "bfloat16",
    safe_serialization: bool = True,
    device: str = "cpu",
) -> None:
    """Execute complete end-to-end weight fusion flow and export unified weights."""
    target_dtype = resolve_dtype(precision)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Weight Fusion Execution Engine")
    logger.info("Base Model Path: %s", base_model_path)
    logger.info("Adapter Path:    %s", adapter_path)
    logger.info("Output Dir:      %s", output_path.resolve())
    logger.info("Target Precision: %s | Target Device: %s", precision, device)

    # 1. Load Tokenizer & Save to Output Directory
    logger.info("Loading tokenizer from %s...", base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(str(output_path))
    logger.info("Tokenizer saved to export directory.")

    # 2. Load Base Model
    logger.info("Loading Base Model into memory...")
    raw_base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=target_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # 3. Attach LoRA Adapter
    logger.info("Attaching trained LoRA Adapter from %s...", adapter_path)
    peft_model = PeftModel.from_pretrained(
        model=raw_base_model,
        model_id=adapter_path,
        is_trainable=False,
    )
    peft_model.to(device=device)
    peft_model.eval()

    # 4. Perform Weight Fusion
    fusion_engine = WeightFusionEngine(target_dtype=target_dtype)
    fused_model = fusion_engine.fuse_peft_model(model=peft_model, safe_merge=True)

    # 5. Export Unified Standalone Model
    logger.info("Saving unified fused model parameters to %s...", output_path)
    save_fn = getattr(fused_model, "save_pretrained", None)
    if callable(save_fn):
        save_fn(
            save_directory=str(output_path),
            safe_serialization=safe_serialization,
        )
    else:
        # Fallback for standard torch.nn.Module instances
        torch.save(fused_model.state_dict(), output_path / "model.pt")

    logger.info("Zero-latency weight fusion successfully completed and exported!")

    logger.info("Zero-latency weight fusion successfully completed and exported!")


def main() -> None:
    args = parse_args()
    try:
        run_weight_fusion(
            base_model_path=args.base_model_path,
            adapter_path=args.adapter_path,
            output_dir=args.output_dir,
            precision=args.precision,
            safe_serialization=args.safe_serialization,
            device=args.device,
        )
    except Exception as e:
        logger.error("Weight Fusion Pipeline failed with error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()