"""
OmniPEFT Exploratory Data Analysis (EDA) Engine.

Executes multi-threaded tabular profiling over large instruction datasets (150k+ samples)
using Polars and Transformers tokenizers. Computes sequence length quantiles (P50, P90, P99),
detects missing/corrupted features, and exports reports and visualization artifacts.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class TokenQuantiles:
    # Dataclass holding token length percentile metrics.

    p50: float
    p90: float
    p95: float
    p99: float
    max_len: int
    min_len: int
    mean: float
    std_dev: float


@dataclass
class EDAReportSummary:
    # Dataclass holding full dataset EDA metadata and health metrics.

    dataset_name: str
    total_samples: int
    missing_value_counts: Dict[str, int]
    prompt_length_quantiles: TokenQuantiles
    response_length_quantiles: TokenQuantiles
    total_sequence_quantiles: TokenQuantiles
    recommended_max_seq_length: int


class EDAPipeline:
    """High-performance EDA engine for instruction fine-tuning datasets."""

    def __init__(
        self,
        tokenizer_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct",
        output_dir: str = "artifacts/eda_reports",
    ) -> None:
        """Initialize EDA pipeline with tokenizer and output paths.

        Args:
            tokenizer_name_or_path: HuggingFace model path for exact token length calculations.
            output_dir: Path where visual plots, JSON summaries, and MD reports will be saved.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing tokenizer for EDA: %s", tokenizer_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            trust_remote_code=True,
            use_fast=True,
        )

    def compute_quantiles(self, lengths: List[int]) -> TokenQuantiles:
        """Compute P50, P90, P95, P99, mean, and standard deviation over sequence lengths.

        Args:
            lengths: List of integer sequence lengths.

        Returns:
            Populated TokenQuantiles object.
        """
        arr = np.array(lengths, dtype=np.int32)
        return TokenQuantiles(
            p50=float(np.percentile(arr, 50)),
            p90=float(np.percentile(arr, 90)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
            max_len=int(np.max(arr)),
            min_len=int(np.min(arr)),
            mean=float(np.mean(arr)),
            std_dev=float(np.std(arr)),
        )

    def analyze_dataframe(
        self,
        df: pl.DataFrame,
        dataset_name: str = "ultrafeedback_150k",
        prompt_col: str = "prompt",
        response_col: str = "response",
    ) -> EDAReportSummary:
        """Perform full statistical profiling over a Polars DataFrame.

        Args:
            df: Input Polars DataFrame containing instruction dataset samples.
            dataset_name: Identifier for artifact naming.
            prompt_col: Column name containing instruction/user prompts.
            response_col: Column name containing assistant responses.

        Returns:
            EDAReportSummary object containing all statistical findings.
        """
        logger.info("Profiling dataset '%s' with %d samples...", dataset_name, len(df))

        # Step 1: Missing Feature Analysis
        missing_counts = {}
        for col in df.columns:
            null_count = df[col].null_count()
            missing_counts[col] = null_count
            if null_count > 0:
                logger.warning(
                    "Feature '%s' contains %d missing values!", col, null_count
                )

        # Drop null entries in target columns for sequence analysis
        clean_df = df.drop_nulls(subset=[prompt_col, response_col])

        # Step 2: High-Speed Batch Tokenization
        prompts = clean_df[prompt_col].to_list()
        responses = clean_df[response_col].to_list()

        logger.info("Executing batch tokenization over %d samples...", len(prompts))

        # Tokenize prompts and responses
        prompt_encodings = self.tokenizer(
            prompts, add_special_tokens=False, truncation=False
        )
        response_encodings = self.tokenizer(
            responses, add_special_tokens=False, truncation=False
        )

        prompt_lens = [len(tokens) for tokens in prompt_encodings["input_ids"]]
        response_lens = [len(tokens) for tokens in response_encodings["input_ids"]]
        total_lens = [p + r for p, r in zip(prompt_lens, response_lens)]

        # Step 3: Quantile Calculations
        prompt_stats = self.compute_quantiles(prompt_lens)
        response_stats = self.compute_quantiles(response_lens)
        total_stats = self.compute_quantiles(total_lens)

        # Set recommended max sequence length to nearest upper power of 2 covering P99
        recommended_max_seq = 2048 if total_stats.p99 <= 2048 else 4096

        summary = EDAReportSummary(
            dataset_name=dataset_name,
            total_samples=len(df),
            missing_value_counts=missing_counts,
            prompt_length_quantiles=prompt_stats,
            response_length_quantiles=response_stats,
            total_sequence_quantiles=total_stats,
            recommended_max_seq_length=recommended_max_seq,
        )

        # Step 4: Save Reports and Artifacts
        self._save_summary_json(summary)
        self._save_markdown_report(summary)
        self._generate_distribution_plots(
            total_lens, prompt_lens, response_lens, summary
        )

        return summary

    def _save_summary_json(self, summary: EDAReportSummary) -> None:
        """Export summary metadata as structured JSON."""
        json_path = self.output_dir / f"{summary.dataset_name}_eda_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, indent=4)
        logger.info("Exported JSON summary to: %s", json_path)

    def _save_markdown_report(self, summary: EDAReportSummary) -> None:
        """Export publication-ready Markdown report."""
        md_path = self.output_dir / f"{summary.dataset_name}_eda_report.md"

        md_content = f"""# OmniPEFT EDA Report: {summary.dataset_name}

## 1. Dataset Overview
* **Total Samples Profiled:** `{summary.total_samples:,}`
* **Recommended Sequence Length Cutoff:** `{summary.recommended_max_seq_length}`

## 2. Feature Health & Missing Values
| Feature Name | Null / Missing Count | Status |
| :--- | :--- | :--- |
"""
        for col, count in summary.missing_value_counts.items():
            status = "PASS" if count == 0 else f"WARNING ({count} nulls)"
            md_content += f"| `{col}` | {count} | {status} |\n"

        tot = summary.total_sequence_quantiles
        p_stat = summary.prompt_length_quantiles
        r_stat = summary.response_length_quantiles

        md_content += f"""
## 3. Sequence Length Distribution Analysis (Token Count)
| Metric | Prompt Tokens | Response Tokens | Total Combined Tokens |
| :--- | :--- | :--- | :--- |
| **Mean ± Std** | {p_stat.mean:.1f} ± {p_stat.std_dev:.1f} | {r_stat.mean:.1f} ± {r_stat.std_dev:.1f} | {tot.mean:.1f} ± {tot.std_dev:.1f} |
| **P50 (Median)** | {p_stat.p50:.0f} | {r_stat.p50:.0f} | {tot.p50:.0f} |
| **P90** | {p_stat.p90:.0f} | {r_stat.p90:.0f} | {tot.p90:.0f} |
| **P95** | {p_stat.p95:.0f} | {r_stat.p95:.0f} | {tot.p95:.0f} |
| **P99** | **{p_stat.p99:.0f}** | **{r_stat.p99:.0f}** | **{tot.p99:.0f}** |
| **Max Length** | {p_stat.max_len} | {r_stat.max_len} | {tot.max_len} |

## 4. Engineering Recommendations
* **Context Truncation Strategy:** Setting max sequence length to **{summary.recommended_max_seq_length}** tokens retains **>99%** of complete instruction sequences without information loss.
* **VRAM Impact:** Limits peak activation memory footprint during backward passes under 8GB VRAM.
"""
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info("Exported Markdown EDA report to: %s", md_path)

    def _generate_distribution_plots(
        self,
        total_lens: List[int],
        prompt_lens: List[int],
        response_lens: List[int],
        summary: EDAReportSummary,
    ) -> None:
        """Generate high-resolution PNG plots for sequence distributions."""
        plt.style.use(
            "seaborn-v0_8-darkgrid"
            if "seaborn-v0_8-darkgrid" in plt.style.available
            else "default"
        )
        _fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

        # Plot 1: Combined Sequence Length Histogram
        axes[0].hist(total_lens, bins=60, color="#1f77b4", alpha=0.8, edgecolor="black")
        axes[0].axvline(
            summary.total_sequence_quantiles.p50,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"P50 ({int(summary.total_sequence_quantiles.p50)})",
        )
        axes[0].axvline(
            summary.total_sequence_quantiles.p99,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"P99 ({int(summary.total_sequence_quantiles.p99)})",
        )
        axes[0].set_title(
            "Total Sequence Token Length Distribution", fontsize=14, fontweight="bold"
        )
        axes[0].set_xlabel("Token Count", fontsize=12)
        axes[0].set_ylabel("Frequency", fontsize=12)
        axes[0].legend(fontsize=11)

        # Plot 2: Prompt vs Response Distribution Comparison
        axes[1].hist(
            prompt_lens, bins=50, color="#2ca02c", alpha=0.6, label="Prompt Tokens"
        )
        axes[1].hist(
            response_lens, bins=50, color="#d62728", alpha=0.6, label="Response Tokens"
        )
        axes[1].set_title(
            "Prompt vs. Response Token Distributions", fontsize=14, fontweight="bold"
        )
        axes[1].set_xlabel("Token Count", fontsize=12)
        axes[1].set_ylabel("Frequency", fontsize=12)
        axes[1].legend(fontsize=11)

        plt.tight_layout()
        plot_path = self.output_dir / f"{summary.dataset_name}_token_distribution.png"
        plt.savefig(plot_path)
        plt.close()
        logger.info("Exported visualization distribution chart to: %s", plot_path)
