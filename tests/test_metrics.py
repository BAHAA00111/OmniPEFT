"""
OmniPEFT Analytics Test Suite - Metric Engine & Perplexity Verification.
Comprehensive unit tests verifying streaming cross-entropy accumulation,
mathematical bounds on perplexity, custom LCS ROUGE-L computation, fallback
evaluation paths, and edge-case resilience.
"""

import math
from unittest.mock import patch
import pytest

from omnipeft.analytics.metrics import EvaluationMetricsEngine, PerplexityTracker



# 1. Tests for PerplexityTracker
class TestPerplexityTracker:
    """Tests covering online loss aggregation and exponentiation logic."""

    def test_initial_state(self) -> None:
        """Verify tracker starts with clean initial state zero accumulators."""
        tracker = PerplexityTracker()
        assert tracker.current_loss == 0.0
        assert tracker.perplexity == 1.0

    def test_single_update_correctness(self) -> None:
        """Verify single update math matches PPL = exp(loss)."""
        tracker = PerplexityTracker()
        loss_val = 2.0
        tokens = 10
        tracker.update(loss=loss_val, num_tokens=tokens)

        assert tracker.current_loss == pytest.approx(loss_val)
        assert tracker.perplexity == pytest.approx(math.exp(loss_val))

    def test_weighted_token_accumulation(self) -> None:
        """Verify weighted average calculation across batches of varying sequence lengths."""
        tracker = PerplexityTracker()
        # Batch 1: Loss 2.0 over 10 tokens
        tracker.update(loss=2.0, num_tokens=10)
        # Batch 2: Loss 1.0 over 30 tokens
        tracker.update(loss=1.0, num_tokens=30)

        # Expected mean loss: (2.0 * 10 + 1.0 * 30) / (10 + 30) = 50.0 / 40 = 1.25
        expected_mean_loss = 1.25
        expected_ppl = math.exp(expected_mean_loss)

        assert tracker.current_loss == pytest.approx(expected_mean_loss)
        assert tracker.perplexity == pytest.approx(expected_ppl)

    def test_invalid_and_nan_loss_handling(self) -> None:
        """Ensure NaN and Inf values are safely filtered without corrupting state."""
        tracker = PerplexityTracker()
        tracker.update(loss=1.5, num_tokens=10)

        # Inject invalid loss values
        tracker.update(loss=float("nan"), num_tokens=10)
        tracker.update(loss=float("inf"), num_tokens=10)
        tracker.update(loss=float("-inf"), num_tokens=10)

        # State should remain untouched
        assert tracker.current_loss == pytest.approx(1.5)
        assert tracker.perplexity == pytest.approx(math.exp(1.5))

    def test_perplexity_overflow_capping(self) -> None:
        """Verify extreme high loss caps exponent at 100.0 without raising OverflowError."""
        tracker = PerplexityTracker()
        tracker.update(loss=200.0, num_tokens=1)

        # Should return exp(100.0) safely without throwing
        assert tracker.perplexity == pytest.approx(math.exp(100.0))

    def test_reset_behavior(self) -> None:
        """Verify tracker state fully clears when reset() is called."""
        tracker = PerplexityTracker()
        tracker.update(loss=3.5, num_tokens=100)
        tracker.reset()

        assert tracker.current_loss == 0.0
        assert tracker.perplexity == 1.0



# 2. Tests for EvaluationMetricsEngine
class TestEvaluationMetricsEngine:
    """Tests covering LCS ROUGE-L math, Hugging Face metric integration, and Python fallbacks."""

    def test_compute_lcs_length(self) -> None:
        """Verify exact dynamic programming LCS matrix computation."""
        # LCS between ['the', 'quick', 'brown', 'fox'] and ['the', 'fast', 'brown', 'dog'] -> ['the', 'brown'] = 2
        seq1 = ["the", "quick", "brown", "fox"]
        seq2 = ["the", "fast", "brown", "dog"]

        lcs_len = EvaluationMetricsEngine._compute_lcs_length(seq1, seq2)
        assert lcs_len == 2

    @pytest.mark.parametrize(
        "pred, ref, expected_score",
        [
            ("the cat sat on the mat", "the cat sat on the mat", 1.0),
            ("", "the cat sat on the mat", 0.0),
            ("the cat sat on the mat", "", 0.0),
            ("completely unrelated text", "fox leaps over fence", 0.0),
            ("the cat mat", "the cat sat on the mat", 2 * (3 / 3) * (3 / 6) / ((3 / 3) + (3 / 6))),
        ],
    )
    def test_compute_rouge_l(self, pred: str, ref: str, expected_score: float) -> None:
        """Verify custom ROUGE-L F1 score against exact mathematical targets."""
        score = EvaluationMetricsEngine.compute_rouge_l(pred, ref)
        assert score == pytest.approx(expected_score, abs=1e-4)

    def test_compute_generation_metrics_empty_or_mismatched(self) -> None:
        """Verify graceful return of zero dict when inputs are empty or mismatched in length."""
        # Empty inputs
        res_empty = EvaluationMetricsEngine.compute_generation_metrics([], [])
        assert res_empty == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "bleu": 0.0}

        # Mismatched lengths
        res_mismatch = EvaluationMetricsEngine.compute_generation_metrics(
            predictions=["pred1", "pred2"],
            references=["ref1"],
        )
        assert res_mismatch == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "bleu": 0.0}

    def test_fallback_computation_path(self) -> None:
        """Verify smooth fallback execution when external metric packages raise an Exception."""
        preds = ["the cat sat on the mat", "a small dog barked"]
        refs = ["the cat sat on the mat", "the dog barked loudly"]

        # Intercept import of 'evaluate' to trigger the fallback logic cleanly
        with patch("builtins.__import__", side_effect=ImportError("HF Evaluate missing")):
            metrics = EvaluationMetricsEngine.compute_generation_metrics(preds, refs)

            # Ensure fallback returns valid ROUGE dictionary structure
            assert "rouge1" in metrics
            assert "rouge2" in metrics
            assert "rougeL" in metrics
            assert "bleu" in metrics
            assert metrics["bleu"] == 0.0  # Fallback algorithm sets BLEU to 0.0
            assert metrics["rougeL"] > 0.0

    def test_successful_metric_computation_structure(self) -> None:
        """Verify metric engine output keys and scaling range [0, 100]."""
        preds = ["the quick brown fox jumps over the lazy dog"]
        refs = ["the quick brown fox jumped over a lazy dog"]

        metrics = EvaluationMetricsEngine.compute_generation_metrics(preds, refs)

        for key in ["rouge1", "rouge2", "rougeL", "bleu"]:
            assert key in metrics
            assert 0.0 <= metrics[key] <= 100.0