"""OmniPEFT CUDA Memory Allocator & OOM Protection Suite.

Handles environment configuration for PyTorch memory allocators, active VRAM
profiling, memory-fragmentation mitigation, and dynamic dynamic memory cleanup during
long-context LLM backpropagation.
"""

from collections import namedtuple
import contextlib
import gc
import logging
import os
from typing import Generator

# Set PyTorch allocator environment variables BEFORE importing torch
# to guarantee allocator memory split thresholds take effect.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

import torch

logger = logging.getLogger(__name__)

VRAMStats = namedtuple(
    "VRAMStats",
    ["allocated_gb", "reserved_gb", "max_allocated_gb", "free_gb", "total_gb"],
)


class CUDAMemoryGuard:
    """Production CUDA Allocator & VRAM Optimizer for RTX 3080 10GB limits."""

    def __init__(self, device_id: int = 0, target_vram_gb: float = 8.0) -> None:
        # Initialize memory guard configuration.
        self.device_id = device_id
        self.target_vram_gb = target_vram_gb

        if not torch.cuda.is_available():
            logger.warning(
                "CUDA is not available in current environment. MemoryGuard running in CPU mock mode."
            )
            self.device = torch.device("cpu")
            self.total_vram_gb = 0.0
        else:
            self.device = torch.device(f"cuda:{self.device_id}")
            self.total_vram_gb = torch.cuda.get_device_properties(
                self.device
            ).total_memory / (1024**3)
            self._verify_allocator_config()

    def _verify_allocator_config(self) -> None:
        # Assert PyTorch allocator configuration flags are set correctly.
        alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "max_split_size_mb:128" not in alloc_conf:
            logger.warning(
                "PYTORCH_CUDA_ALLOC_CONF not initialized before PyTorch load! "
                "Current conf: %s",
                alloc_conf,
            )
        else:
            logger.info("CUDA Allocator configured successfully: %s", alloc_conf)

    def get_vram_stats(self) -> VRAMStats:
        """Query precise CUDA memory stats in Gigabytes.

        Returns:
            VRAMStats namedtuple containing allocated, reserved, peak, and free
            memory metrics.
        """
        if not torch.cuda.is_available():
            return VRAMStats(0.0, 0.0, 0.0, 0.0, 0.0)

        allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
        reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated(self.device) / (1024**3)
        free = self.total_vram_gb - reserved

        return VRAMStats(
            allocated_gb=round(allocated, 3),
            reserved_gb=round(reserved, 3),
            max_allocated_gb=round(max_allocated, 3),
            free_gb=round(free, 3),
            total_gb=round(self.total_vram_gb, 3),
        )

    def purge_vram(self) -> VRAMStats:
        """Perform multi-stage garbage collection and reset CUDA allocator cache.

        Executes python GC, empties PyTorch allocator cache, and resets peak stats.
        """
        if not torch.cuda.is_available():
            return self.get_vram_stats()

        initial_stats = self.get_vram_stats()

        # Step 1: Force Python reference cleanup
        gc.collect()

        # Step 2: Empty PyTorch allocator cache back to OS/driver
        torch.cuda.empty_cache()

        # Step 3: Reset peak allocation trackers
        torch.cuda.reset_peak_memory_stats(self.device)

        cleared_stats = self.get_vram_stats()
        freed_gb = round(initial_stats.reserved_gb - cleared_stats.reserved_gb, 3)

        logger.debug(
            "Purged VRAM: Freed %s GB | Reserved: %s GB -> %s GB",
            freed_gb,
            initial_stats.reserved_gb,
            cleared_stats.reserved_gb,
        )
        return cleared_stats

    @contextlib.contextmanager
    def zero_oom_context(self) -> Generator[None, None, None]:
        """Context manager to intercept CUDA Out-Of-Memory (OOM) exceptions dynamically.

        If an OOM occurs, it flushes VRAM caches, runs garbage collection, and raises a cleanly
        formatted RuntimeError with active memory diagnostics.
        """
        try:
            yield
        except torch.cuda.OutOfMemoryError as oom_err:
            logger.error("CUDA OOM Fault Detected! Initiating Emergency VRAM Purge...")
            stats_before = self.get_vram_stats()
            self.purge_vram()
            stats_after = self.get_vram_stats()

            error_msg = (
                f"\n[OmniPEFT OOM Interceptor Report]\n"
                f"  Target Device: CUDA:{self.device_id} ({self.total_vram_gb:.2f} GB Total)\n"
                f"  Allocated Memory Pre-OOM: {stats_before.allocated_gb:.2f} GB\n"
                f"  Reserved Memory Pre-OOM:  {stats_before.reserved_gb:.2f} GB\n"
                f"  Emergency Cleanup Recovered: {stats_before.reserved_gb - stats_after.reserved_gb:.2f} GB\n"
                f"  Recommendation: Decrease per_device_train_batch_size or enable gradient_checkpointing."
            )
            raise RuntimeError(error_msg) from oom_err

    def log_memory_summary(self) -> None:
        """Log a formatted snapshot of current VRAM fragmentation and allocation."""
        if not torch.cuda.is_available():
            logger.info("CUDA not available. Skipping memory summary.")
            return

        stats = self.get_vram_stats()
        fragmentation = round(
            ((stats.reserved_gb - stats.allocated_gb) / (stats.reserved_gb + 1e-6))
            * 100,
            2,
        )

        logger.info(
            f"VRAM Summary [CUDA:{self.device_id}] | "
            f"Allocated: {stats.allocated_gb}GB | "
            f"Reserved: {stats.reserved_gb}GB | "
            f"Free: {stats.free_gb}GB | "
            f"Fragmentation: {fragmentation}%"
        )


# Global Singleton Instance for quick initialization across submodules
memory_guard = CUDAMemoryGuard()
