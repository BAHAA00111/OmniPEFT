"""
OmniPEFT Checkpoint Management Engine.

Handles saving and managing lightweight PEFT adapter checkpoints (~50MB - 100MB)
instead of multi-gigabyte full model weights. Features automatic top-K retention,
training state serialization, and safe filesystem directory cleanups.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn
from peft import PeftMixedModel, PeftModel
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
ModelType = Union[PeftModel, PeftMixedModel, nn.Module]


class CheckpointManager:
    def __init__(
        self,
        output_dir: Union[str, Path] = "artifacts/checkpoints",
        max_to_keep: int = 3,
        save_on_best_metric: bool = True,
        metric_name: str = "eval_loss",
        mode: str = "min",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self.save_on_best_metric = save_on_best_metric
        self.metric_name = metric_name
        self.mode = mode.lower()

        if self.mode not in ("min", "max"):
            raise ValueError(f"Invalid mode '{mode}'. Expected 'min' or 'max'.")

        self.best_metric_value: float = (
            float("inf") if self.mode == "min" else float("-inf")
        )
        self.saved_checkpoints: List[Path] = []

        logger.info(
            "Initialized CheckpointManager | Output Dir: %s | Max Retention: %d | Track Metric: %s (%s)",
            self.output_dir.resolve(),
            self.max_to_keep,
            self.metric_name,
            self.mode,
        )

    def _is_better_metric(self, current_val: float) -> bool:
        """Compare current metric against best recorded value.

        Args:
            current_val: Metric value of the current checkpoint.

        Returns:
            True if current_val improves upon best_metric_value, False otherwise.
        """
        if self.mode == "min":
            return current_val < self.best_metric_value
        return current_val > self.best_metric_value

    def save_checkpoint(
        self,
        model: ModelType,
        tokenizer: Optional[TokenizerType] = None,
        step: int = 0,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save PEFT adapter weights, tokenizer, and metadata to disk.

        Args:
            model: PeftModel or PyTorch module holding LoRA adapters.
            tokenizer: Optional HF Tokenizer to save alongside adapter.
            step: Current global training step.
            epoch: Current training epoch.
            metrics: Optional dictionary of evaluation metrics.
            optimizer: Optional PyTorch optimizer (saved separately if provided).
            extra_state: Optional dictionary containing additional trainer state.

        Returns:
            Path object pointing to the newly created checkpoint directory.
        """
        metrics = metrics or {}
        checkpoint_dir = self.output_dir / f"checkpoint-step-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Saving lightweight adapter checkpoint to: %s", checkpoint_dir)

        # 1. Save PEFT Adapter Weights (Only ~50MB - 100MB)
        if isinstance(model, (PeftModel, PeftMixedModel)):
            model.save_pretrained(str(checkpoint_dir))
        elif hasattr(model, "save_pretrained"):
            # Handle unwrapped models exposing save_pretrained
            save_fn = model.save_pretrained
            if callable(save_fn):
                save_fn(str(checkpoint_dir))
        else:
            # Fallback: Filter and save trainable parameters (adapter layers only)
            adapter_state_dict = {
                k: v.cpu()
                for k, v in model.state_dict().items()
                if "lora_" in k or v.requires_grad
            }
            torch.save(adapter_state_dict, checkpoint_dir / "adapter_model.bin")

        # 2. Save Tokenizer Configuration
        if tokenizer is not None:
            tokenizer.save_pretrained(str(checkpoint_dir))
        # 2. Save Tokenizer Configuration
        if tokenizer is not None:
            tokenizer.save_pretrained(checkpoint_dir)

        # 3. Save Optimizer State (Optional, saved inside checkpoint for resume support)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer_state.pt")

        # 4. Save Metadata & Training State JSON
        is_best = False
        if self.metric_name in metrics:
            current_metric = metrics[self.metric_name]
            if self._is_better_metric(current_metric):
                self.best_metric_value = current_metric
                is_best = True

        metadata: Dict[str, Any] = {
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
            "is_best": is_best,
            "best_metric_value": self.best_metric_value,
            "extra_state": extra_state or {},
        }

        with open(checkpoint_dir / "trainer_state.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # 5. Handle Best Model Link/Symlink Copy
        if is_best and self.save_on_best_metric:
            best_dir = self.output_dir / "checkpoint-best"
            logger.info(
                "New best %s achieved (%.4f)! Updating %s",
                self.metric_name,
                self.best_metric_value,
                best_dir,
            )
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(checkpoint_dir, best_dir)

        # 6. Update tracking queue and execute retention cleanup
        self.saved_checkpoints.append(checkpoint_dir)
        self._rotate_checkpoints()

        return checkpoint_dir

    def _rotate_checkpoints(self) -> None:
        """Purge older checkpoints exceeding max_to_keep limit."""
        while len(self.saved_checkpoints) > self.max_to_keep:
            oldest_ckpt = self.saved_checkpoints.pop(0)
            if oldest_ckpt.exists() and oldest_ckpt.name != "checkpoint-best":
                logger.info("Purging old checkpoint: %s", oldest_ckpt)
                try:
                    shutil.rmtree(oldest_ckpt)
                except (OSError, PermissionError) as e:
                    logger.warning(
                        "Failed to remove checkpoint directory %s: %s", oldest_ckpt, e
                    )

    @staticmethod
    def load_trainer_state(checkpoint_dir: Union[str, Path]) -> Dict[str, Any]:
        """Load trainer metadata JSON from specified checkpoint directory.

        Args:
            checkpoint_dir: Path to directory containing trainer_state.json.

        Returns:
            Dictionary containing step, epoch, metrics, and custom state.
        """
        state_file = Path(checkpoint_dir) / "trainer_state.json"
        if not state_file.exists():
            logger.warning("No trainer_state.json found in %s", checkpoint_dir)
            return {}

        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
