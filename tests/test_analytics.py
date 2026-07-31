"""OmniPEFT Unit Tests for Metrics & Convergence Logger."""

import math
import tempfile

from omnipeft.analytics.convergence_logger import ConvergenceLogger
from omnipeft.analytics.metrics import EvaluationMetricsEngine, PerplexityTracker


def test_perplexity_tracker_math() -> None:
    """Verify loss accumulation and perplexity exp(loss) formula."""
    tracker = PerplexityTracker()

    # Step 1: loss = 2.0, tokens = 10 -> total_loss = 20.0
    tracker.update(loss=2.0, num_tokens=10)
    # Step 2: loss = 1.0, tokens = 10 -> total_loss = 30.0, total_tokens = 20
    tracker.update(loss=1.0, num_tokens=10)

    # Mean loss = 30.0 / 20 = 1.5
    assert abs(tracker.current_loss - 1.5) < 1e-5
    # Perplexity = exp(1.5) ≈ 4.481689
    expected_ppl = math.exp(1.5)
    assert abs(tracker.perplexity - expected_ppl) < 1e-4


def test_perplexity_overflow_safety() -> None:
    """Ensure perplexity calculation gracefully caps massive losses without throwing OverflowError."""
    tracker = PerplexityTracker()
    tracker.update(loss=500.0, num_tokens=1)

    ppl = tracker.perplexity
    assert ppl > 0.0  # Safe cap handling


def test_evaluation_metrics_fallback() -> None:
    """Verify ROUGE/BLEU computation handles sample predictions cleanly."""
    preds = ["The quick brown fox jumps over the lazy dog"]
    refs = ["The quick brown fox jumped over a lazy dog"]

    metrics = EvaluationMetricsEngine.compute_generation_metrics(preds, refs)

    assert "rouge1" in metrics
    assert "rouge2" in metrics
    assert "rougeL" in metrics
    assert "bleu" in metrics
    assert metrics["rougeL"] > 0.0


def test_convergence_logger_lifecycle() -> None:
    """Test logger initialization and file flushing in temporary directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger_engine = ConvergenceLogger(
            log_dir=tmp_dir,
            experiment_name="test_run",
            use_wandb=False,
        )

        logger_engine.log_metrics({"train/loss": 1.25, "train/ppl": 3.49}, step=1)
        logger_engine.close()