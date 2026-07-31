"""
OmniPEFT QLoRA Training Executable Pipeline.

Combines streaming data loaders, 4-bit NF4 quantized LLMs, automatic mixed precision (AMP),
gradient accumulation, and lightweight adapter checkpointing for fine-tuning under 8GB VRAM.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from transformers import get_cosine_schedule_with_warmup

# OmniPEFT Core Modules
from omnipeft.data.loader import create_streaming_dataloader
from omnipeft.engine.checkpoint_manager import CheckpointManager
from omnipeft.engine.qlora_trainer import QLoRAConfig, QLoRAModelBuilder
from omnipeft.systems.amp_config import AMPManager

# Configure Production Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("omnipeft.train")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="OmniPEFT QLoRA High-Efficiency Fine-Tuning Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model & Architecture
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="HuggingFaceH4/ultrafeedback_binarized",
        help="HuggingFace dataset ID or local path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/checkpoints",
        help="Output directory for adapter checkpoints.",
    )

    # Hyperparameters
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (tokens).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device micro batch size.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=8,
        help="Number of update steps to accumulate before stepping optimizer.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Total training optimization steps.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Ratio of total steps for learning rate warmup.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm threshold for clipping.",
    )

    # LoRA Architecture
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA attention rank.")
    parser.add_argument(
        "--lora-alpha", type=int, default=32, help="LoRA alpha scaling factor."
    )
    parser.add_argument(
        "--lora-dropout", type=float, default=0.05, help="LoRA dropout rate."
    )

    # Infrastructure & Monitoring
    parser.add_argument(
        "--logging-steps", type=int, default=10, help="Log metrics every N steps."
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Save adapter checkpoint every N steps.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="AMP execution precision.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )

    return parser.parse_args()


def get_gpu_memory_usage_mb() -> float:
    """Query current peak allocated CUDA VRAM in Megabytes."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def main() -> None:
    """Execute main QLoRA fine-tuning loop."""
    args = parse_args()

    # 1. Reproducibility & Initialization
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    logger.info("Starting OmniPEFT QLoRA Training Pipeline...")
    logger.info("Target Model: %s", args.model_name)
    logger.info(
        "Effective Global Batch Size: %d (Micro-Batch: %d, Accumulation: %d)",
        args.batch_size * args.grad_accum_steps,
        args.batch_size,
        args.grad_accum_steps,
    )

    # 2. Instantiate Automatic Mixed Precision (AMP) Engine
    amp_manager = AMPManager(enabled=True, preferred_dtype=args.precision)

    # 3. Build 4-Bit Quantized QLoRA Model & Tokenizer
    qlora_cfg = QLoRAConfig(
        model_name_or_path=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_gradient_checkpointing=True,
    )
    builder = QLoRAModelBuilder(config=qlora_cfg)
    model, tokenizer = builder.load_model_and_tokenizer()

    # 4. Create Streaming DataLoader
    logger.info("Initializing Streaming Dataset Loader...")
    train_dataloader = create_streaming_dataloader(
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        split="train_sft",
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )

    # 5. Initialize Optimizer & Learning Rate Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Attempt 8-bit AdamW if bitsandbytes is available, otherwise fallback to PyTorch AdamW
    try:
        import bitsandbytes as bnb

        optimizer: torch.optim.Optimizer = bnb.optim.AdamW8bit(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        logger.info("Initialized BitsAndBytes 8-bit AdamW Optimizer.")
    except (ImportError, RuntimeError, AttributeError) as e:
        logger.warning("Could not load 8-bit AdamW (%s)...", e)

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    warmup_steps = int(args.max_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.max_steps,
    )

    # 6. Initialize Checkpoint Manager
    ckpt_manager = CheckpointManager(
        output_dir=args.output_dir,
        max_to_keep=3,
        metric_name="train_loss",
        mode="min",
    )

    # 7. Training Loop Execution
    logger.info("=== Starting Training Loop ===")
    model.train()

    global_step = 0
    accumulated_loss = 0.0
    start_time = time.time()
    data_iter = iter(train_dataloader)

    while global_step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        micro_step_loss = 0.0

        for micro_step in range(args.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                batch = next(data_iter)

            # Move tensors to GPU
            input_ids = batch["input_ids"].to("cuda", non_blocking=True)
            attention_mask = batch["attention_mask"].to("cuda", non_blocking=True)
            labels = batch["labels"].to("cuda", non_blocking=True)

            # Mixed-Precision Forward Pass
            with amp_manager.autocast_context():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / args.grad_accum_steps

            # Scale and Accumulate Backward Pass
            amp_manager.backward(loss)
            micro_step_loss += loss.item() * args.grad_accum_steps

        # Unscale, Clip Norm, and Step Optimizer
        amp_manager.step_optimizer(
            optimizer=optimizer,
            model_parameters=trainable_params,
            max_grad_norm=args.max_grad_norm,
        )
        scheduler.step()

        global_step += 1
        accumulated_loss += micro_step_loss

        # Logging Metrics
        if global_step % args.logging_steps == 0 or global_step == args.max_steps:
            avg_loss = accumulated_loss / args.logging_steps
            accumulated_loss = 0.0
            elapsed = time.time() - start_time
            throughput = (
                global_step * args.batch_size * args.grad_accum_steps
            ) / elapsed
            current_lr = scheduler.get_last_lr()[0]
            peak_vram = get_gpu_memory_usage_mb()

            logger.info(
                "Step %d/%d | Loss: %.4f | LR: %.2e | Peak VRAM: %.1f MB | Speed: %.2f samples/sec",
                global_step,
                args.max_steps,
                avg_loss,
                current_lr,
                peak_vram,
                throughput,
            )

        # Saving Lightweight Checkpoint
        if global_step % args.save_steps == 0 or global_step == args.max_steps:
            metrics = {
                "train_loss": micro_step_loss,
                "peak_vram_mb": get_gpu_memory_usage_mb(),
            }
            ckpt_manager.save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                step=global_step,
                metrics=metrics,
                optimizer=optimizer,
            )

    logger.info("=== Training Complete ===")
    logger.info("Final adapter saved to: %s", Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()
