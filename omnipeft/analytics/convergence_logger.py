"""
OmniPEFT Dual-Backend Telemetry & Convergence Logger.

Manages synchronized streaming log updates to TensorBoard and Weights & Biases (wandb).
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ConvergenceLogger:
    """Unified telemetry engine for TensorBoard and Weights & Biases tracking."""

    def __init__(
        self,
        log_dir: str = "./logs",
        experiment_name: str = "omnipeft_experiment",
        use_wandb: bool = False,
        wandb_project: str = "omnipeft",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize telemetry backends.

        Args:
            log_dir: Disk directory path for saving TensorBoard event logs.
            experiment_name: Unique run name for experiment identification.
            use_wandb: Enable Weights & Biases telemetry backend.
            wandb_project: W&B project dashboard name.
            config: Optional hyperparameters dictionary for logging metadata.
        """
        self.log_path = Path(log_dir) / experiment_name
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb

        # 1. Initialize TensorBoard Writer
        self.tb_writer: Optional[Any] = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb_writer = SummaryWriter(log_dir=str(self.log_path))
            logger.info("TensorBoard telemetry initialized at %s", self.log_path)
        except (ImportError, RuntimeError, OSError) as e:
            logger.warning(
                "TensorBoard telemetry initialization skipped (%s). TensorBoard logging disabled.",
                e,
            )

        # 2. Initialize Weights & Biases Run
        self.wandb_run: Optional[Any] = None
        if self.use_wandb:
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=experiment_name,
                    config=config or {},
                    reinit=True,
                )
                logger.info(
                    "Weights & Biases telemetry initialized | Project: %s",
                    wandb_project,
                )
            except (ImportError, RuntimeError, AttributeError, OSError) as e:
                logger.warning(
                    "Failed to initialize Weights & Biases: %s. W&B logging disabled.",
                    e,
                )

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metric key-value pairs at a specified training/evaluation step.

        Args:
            metrics: Dictionary of numerical metric values (e.g., {"loss": 1.5, "ppl": 4.48}).
            step: Global step integer index.
        """
        # Log to TensorBoard
        if self.tb_writer is not None:
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    self.tb_writer.add_scalar(key, float(val), global_step=step)

        # Log to Weights & Biases
        if self.wandb_run is not None:
            try:
                import wandb

                wandb.log(metrics, step=step)
            except (ImportError, RuntimeError, AttributeError, OSError) as e:
                logger.warning("Error writing to W&B at step %d: %s", step, e)

    def close(self) -> None:
        """Flush buffers and finalize logger instances."""
        if self.tb_writer is not None:
            try:
                self.tb_writer.flush()
                self.tb_writer.close()
                logger.info("TensorBoard session closed.")
            except (RuntimeError, OSError) as e:
                logger.debug("Silently handling TensorBoard close error: %s", e)

        if self.wandb_run is not None:
            try:
                import wandb

                wandb.finish()
                logger.info("Weights & Biases session closed.")
            except (ImportError, RuntimeError, AttributeError, OSError) as e:
                logger.debug("Silently handling W&B cleanup error on shutdown: %s", e)
