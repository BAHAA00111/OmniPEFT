"""OmniPEFT Unit Tests for Hypothesis Testing Engine."""

import pytest
import numpy as np

from omnipeft.analytics.hypothesis_testing import (
    HypothesisTestingEngine,
    SignificanceTestResult,
)


def test_hypothesis_testing_significant_difference() -> None:
    """Verify hypothesis engine detects clear statistical superiority (p < 0.01)."""
    np.random.seed(42)
    # Generate 50 paired samples where fused is consistently higher by ~0.15
    x_base = np.random.normal(loc=0.50, scale=0.05, size=50).tolist()
    x_fused = [x + np.random.normal(loc=0.15, scale=0.02) for x in x_base]

    engine = HypothesisTestingEngine(alpha_threshold=0.01)
    result = engine.evaluate_significance(x_base, x_fused)

    assert isinstance(result, SignificanceTestResult)
    assert result.mean_delta > 0.0
    assert result.t_test_p_value < 0.01
    assert result.wilcoxon_p_value < 0.01
    assert result.is_statistically_significant is True


def test_hypothesis_testing_insignificant_difference() -> None:
    """Verify hypothesis engine flags noisy differences as statistically insignificant."""
    np.random.seed(42)
    # Generate 30 paired samples with identical distribution
    x_base = np.random.normal(loc=0.70, scale=0.05, size=30).tolist()
    x_fused = np.random.normal(loc=0.701, scale=0.05, size=30).tolist()

    engine = HypothesisTestingEngine(alpha_threshold=0.01)
    result = engine.evaluate_significance(x_base, x_fused)

    assert result.is_statistically_significant is False
    assert result.t_test_p_value > 0.01


def test_hypothesis_testing_validation_checks() -> None:
    """Verify error raises on vector length mismatch or insufficient sample count."""
    engine = HypothesisTestingEngine()

    with pytest.raises(ValueError, match="Vector dimension mismatch"):
        engine.evaluate_significance([0.5, 0.6], [0.5])

    with pytest.raises(ValueError, match="at least 2 paired observations"):
        engine.evaluate_significance([0.5], [0.6])
