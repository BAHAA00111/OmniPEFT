"""
OmniPEFT Statistical Significance & Hypothesis Testing Engine (Phase 5 Step 2).

Provides paired statistical tests (Paired Student's t-test & Wilcoxon Signed-Rank test)
to formally verify whether model performance improvements are statistically significant.
"""

import dataclasses
import logging
from typing import Dict, List, Union

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SignificanceTestResult:
    """Immutable data model holding statistical hypothesis test results."""

    base_mean: float
    fused_mean: float
    mean_delta: float
    t_statistic: float
    t_test_p_value: float
    wilcoxon_statistic: float
    wilcoxon_p_value: float
    cohens_d: float
    alpha_threshold: float
    is_statistically_significant: bool

    def to_dict(self) -> Dict[str, Union[float, bool]]:
        """Export result dataclass as a clean primitive dictionary."""
        return dataclasses.asdict(self)


class HypothesisTestingEngine:
    def __init__(self, alpha_threshold: float = 0.01) -> None:
        """Initialize engine parameters.

        Args:
            alpha_threshold: Significance level boundary (default: p < 0.01).
        """
        self.alpha_threshold = alpha_threshold

    @staticmethod
    def _compute_cohens_d(x_base: np.ndarray, x_fused: np.ndarray) -> float:
        """Calculate Cohen's d effect size for paired samples."""
        diff = x_fused - x_base
        std_diff = float(np.std(diff, ddof=1))
        if std_diff == 0.0:
            return 0.0
        return float(np.mean(diff) / std_diff)

    def evaluate_significance(
        self,
        scores_base: List[float],
        scores_fused: List[float],
    ) -> SignificanceTestResult:
        """Execute paired two-tailed t-test and Wilcoxon signed-rank test over evaluation scores.

        Args:
            scores_base: Sample-by-sample score vector for baseline model (X_base).
            scores_fused: Sample-by-sample score vector for fine-tuned/fused model (X_fused).

        Returns:
            SignificanceTestResult object containing p-values, statistics, and significance assertion.

        Raises:
            ValueError: If sample vector lengths do not match or contain fewer than 2 observations.
        """
        if len(scores_base) != len(scores_fused):
            raise ValueError(
                f"Vector dimension mismatch: len(scores_base)={len(scores_base)} != "
                f"len(scores_fused)={len(scores_fused)}"
            )

        if len(scores_base) < 2:
            raise ValueError(
                "Hypothesis testing requires at least 2 paired observations."
            )

        arr_base = np.array(scores_base, dtype=np.float64)
        arr_fused = np.array(scores_fused, dtype=np.float64)

        base_mean = float(np.mean(arr_base))
        fused_mean = float(np.mean(arr_fused))
        mean_delta = fused_mean - base_mean

        # 1. Paired Two-Tailed Student's t-test (Unpack tuple to bypass SciPy stub type ambiguity)
        t_stat_raw, t_p_val_raw = stats.ttest_rel(arr_fused, arr_base)
        t_stat = float(t_stat_raw)
        t_p_val = float(t_p_val_raw)

        # 2. Wilcoxon Signed-Rank Test (Non-parametric)
        diff = arr_fused - arr_base
        if np.all(diff == 0.0):
            # Zero variance difference fallback
            w_stat, w_p_val = 0.0, 1.0
        else:
            w_res = stats.wilcoxon(diff, alternative="two-sided")
            w_stat = float(getattr(w_res, "statistic", 0.0))
            w_p_val = float(getattr(w_res, "pvalue", 1.0))

        # Effect Size
        cohens_d = self._compute_cohens_d(arr_base, arr_fused)

        # Formally verify assertion: p < alpha_threshold AND mean delta > 0
        is_significant = bool((t_p_val < self.alpha_threshold) and (mean_delta > 0.0))

        logger.info(
            "Hypothesis Test Executed | Delta: +%.4f | t-pVal: %.2e | Wilcox-pVal: %.2e | Sig (p < %.2f): %s",
            mean_delta,
            t_p_val,
            w_p_val,
            self.alpha_threshold,
            is_significant,
        )

        return SignificanceTestResult(
            base_mean=round(base_mean, 6),
            fused_mean=round(fused_mean, 6),
            mean_delta=round(mean_delta, 6),
            t_statistic=round(t_stat, 6),
            t_test_p_value=t_p_val,
            wilcoxon_statistic=round(w_stat, 6),
            wilcoxon_p_value=w_p_val,
            cohens_d=round(cohens_d, 6),
            alpha_threshold=self.alpha_threshold,
            is_statistically_significant=is_significant,
        )
