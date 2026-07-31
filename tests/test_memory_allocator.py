"""
Verifies CUDA allocator environment setup, VRAM profiling accuracy,
garbage collection purging, and dynamic OOM context interception.
"""
import os
import pytest
import torch

from omnipeft.systems.memory_allocator import CUDAMemoryGuard, VRAMStats, memory_guard


@pytest.fixture(scope="module")
def guard_instance() -> CUDAMemoryGuard:
    # Fixture providing a initialized CUDAMemoryGuard instance.
    return CUDAMemoryGuard(device_id=0, target_vram_gb=8.0)


class TestCUDAMemoryGuardEnvironment:
    # Tests verifying CUDA environment flags and allocator initialization.

    def test_cuda_allocator_env_variable(self) -> None:
        # Verify PYTORCH_CUDA_ALLOC_CONF is configured with split size limits.
        alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        assert "max_split_size_mb:128" in alloc_conf, (
            f"Expected 'max_split_size_mb:128' in PYTORCH_CUDA_ALLOC_CONF, got: {alloc_conf}"
        )
        assert "expandable_segments:True" in alloc_conf, (
            f"Expected 'expandable_segments:True' in PYTORCH_CUDA_ALLOC_CONF, got: {alloc_conf}"
        )

    def test_singleton_instance_available(self) -> None:
        # Verify the global singleton memory_guard is instantiated properly.
        assert isinstance(memory_guard, CUDAMemoryGuard)


class TestVRAMStatsAndProfiling:
    # Tests for VRAM metric tracking and memory profiling calculations.

    def test_vram_stats_structure(self, guard_instance: CUDAMemoryGuard) -> None:
        """Verify get_vram_stats returns a valid VRAMStats tuple with positive values."""
        stats = guard_instance.get_vram_stats()
        assert isinstance(stats, VRAMStats)

        if torch.cuda.is_available():
            assert stats.total_gb > 0.0
            assert stats.allocated_gb >= 0.0
            assert stats.reserved_gb >= 0.0
            assert stats.free_gb >= 0.0
            assert stats.max_allocated_gb >= 0.0
            assert stats.reserved_gb <= stats.total_gb
        else:
            pytest.skip("CUDA device not available in current execution runner.")

    def test_vram_allocation_tracking(self, guard_instance: CUDAMemoryGuard) -> None:
        # erify that allocating a CUDA tensor accurately updates VRAM stats.
        if not torch.cuda.is_available():
            pytest.skip("CUDA device not available.")

        guard_instance.purge_vram()
        initial_stats = guard_instance.get_vram_stats()

        # Allocate a ~500MB Float32 Tensor on GPU (125,000,000 elements * 4 bytes ≈ 500 MB)
        num_elements = 125_000_000
        dummy_tensor = torch.empty((num_elements,), dtype=torch.float32, device="cuda:0")

        post_alloc_stats = guard_instance.get_vram_stats()
        assert post_alloc_stats.allocated_gb > initial_stats.allocated_gb, (
            "VRAM allocated_gb should increase after tensor creation."
        )

        # Cleanup dummy tensor
        del dummy_tensor
        guard_instance.purge_vram()


class TestMemoryPurgingAndOOMGuard:
    # Tests verifying garbage collection and dynamic OOM recovery routines.

    def test_purge_vram_frees_memory(self, guard_instance: CUDAMemoryGuard) -> None:
        """Verify purge_vram clears unreferenced PyTorch CUDA allocator cache."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA device not available.")

        # Ensure VRAM is clean before starting test
        guard_instance.purge_vram()

        # Allocate ~1.49 GiB temporary tensor (20,000 x 20,000 * 4 bytes)
        # This fits comfortably inside 10GB VRAM to test allocation reservation
        temp_tensor = torch.ones((20_000, 20_000), dtype=torch.float32, device="cuda:0")
        
        # Verify allocation happened
        stats_during = guard_instance.get_vram_stats()
        assert stats_during.reserved_gb > 0.0

        # Delete Python reference and purge PyTorch CUDA cache
        del temp_tensor
        stats_after_purge = guard_instance.purge_vram()

        assert stats_after_purge.reserved_gb < stats_during.reserved_gb, (
            f"purge_vram should release reserved cache. "
            f"Before purge: {stats_during.reserved_gb} GB, After purge: {stats_after_purge.reserved_gb} GB"
        )

    def test_zero_oom_context_normal_execution(self, guard_instance: CUDAMemoryGuard) -> None:
        # Verify zero_oom_context allows normal execution without raising exceptions.
        executed = False
        with guard_instance.zero_oom_context():
            x = torch.tensor([1.0, 2.0, 3.0])
            executed = (x.sum().item() == 6.0)

        assert executed is True

    def test_zero_oom_context_intercepts_oom(self, guard_instance: CUDAMemoryGuard) -> None:
        # Verify zero_oom_context catches torch.cuda.OutOfMemoryError and raises formatted RuntimeError.
        if not torch.cuda.is_available():
            pytest.skip("CUDA device not available.")

        with pytest.raises(RuntimeError) as exc_info:
            with guard_instance.zero_oom_context():
                # Intentionally trigger OOM by asking for an impossible allocation (e.g. 1 Terabyte)
                huge_elements = 1024 * 1024 * 1024 * 256  # 256 Billion Float32s = 1 TB
                _ = torch.empty((huge_elements,), dtype=torch.float32, device="cuda:0")

        assert "[OmniPEFT OOM Interceptor Report]" in str(exc_info.value)
        assert "Allocated Memory Pre-OOM" in str(exc_info.value)

    def test_log_memory_summary_execution(self, guard_instance: CUDAMemoryGuard) -> None:
        # Verify log_memory_summary executes cleanly without errors.
        try:
            guard_instance.log_memory_summary()
        except Exception as err:
            pytest.fail(f"log_memory_summary raised an unexpected exception: {err}")