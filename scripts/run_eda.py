"""
OmniPEFT Executable Exploratory Data Analysis (EDA) CLI Script.

Loads target instruction datasets, executes multi-threaded sequence profiling,
and writes analytical summaries (quantiles, missing features, token distributions)
to specified report artifacts.
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

from datasets import load_dataset
import polars as pl

from omnipeft.data.eda_pipeline import EDAPipeline

# Setup CLI Logging Format
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("omnipeft.scripts.run_eda")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the EDA pipeline execution."""
    parser = argparse.ArgumentParser(
        description="OmniPEFT High-Performance Dataset EDA Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="HuggingFaceH4/ultrafeedback_binarized",
        help="Hugging Face dataset identifier or path to local parquet/jsonl file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train_sft",
        help="Dataset split to analyze (e.g., 'train_sft', 'train', 'test').",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace tokenizer model ID for precise token count profiling.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/eda_reports",
        help="Directory where output JSON, MD reports, and PNG plots will be saved.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=150000,
        help="Maximum number of samples to process from dataset stream.",
    )
    parser.add_argument(
        "--prompt-col",
        type=str,
        default="prompt",
        help="DataFrame column containing user instruction text.",
    )
    parser.add_argument(
        "--response-col",
        type=str,
        default="response",
        help="DataFrame column containing target assistant response text.",
    )
    return parser.parse_args()


def extract_text_pairs(
    samples: List[Dict[str, Any]], prompt_col: str, response_col: str
) -> List[Dict[str, str]]:
    """Normalize raw dataset messages/columns into prompt-response dictionaries."""
    extracted = []
    for sample in samples:
        prompt_text = ""
        response_text = ""

        # ChatML / UltraFeedback messages format
        if "messages" in sample and isinstance(sample["messages"], list):
            for msg in sample["messages"]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    prompt_text = content
                elif role == "assistant":
                    response_text = content
        else:
            prompt_text = str(sample.get(prompt_col, sample.get("instruction", "")))
            response_text = str(sample.get(response_col, sample.get("output", "")))

        if prompt_text and response_text:
            extracted.append({"prompt": prompt_text, "response": response_text})

    return extracted


def main() -> None:
    """Execute EDA workflow."""
    args = parse_args()

    logger.info("Starting OmniPEFT EDA Workflow...")
    logger.info("Target Dataset: %s (Split: %s)", args.dataset_name, args.split)
    logger.info("Output Directory: %s", args.output_dir)

    # Instantiate EDA Pipeline
    pipeline = EDAPipeline(
        tokenizer_name_or_path=args.tokenizer_name,
        output_dir=args.output_dir,
    )

    # Ingest Dataset
    logger.info("Fetching dataset stream...")
    try:
        raw_ds = load_dataset(args.dataset_name, split=args.split, streaming=True)

        sample_buffer: List[Dict[str, Any]] = []
        for i, item in enumerate(raw_ds):
            if i >= args.max_samples:
                break
            sample_buffer.append(dict(item))

        logger.info("Successfully ingested %d raw samples.", len(sample_buffer))
    except Exception as e: # noqa: BLE001
        logger.error("Failed to load dataset '%s': %s", args.dataset_name, e)
        sys.exit(1)

    # Normalize Schema into Polars DataFrame
    logger.info("Normalizing instruction fields into Polars DataFrame...")
    clean_records = extract_text_pairs(
        sample_buffer,
        prompt_col=args.prompt_col,
        response_col=args.response_col,
    )

    if not clean_records:
        logger.error("No valid prompt/response pairs extracted! Aborting.")
        sys.exit(1)

    df = pl.DataFrame(clean_records)

    # Clean dataset slug for artifact naming
    dataset_slug = Path(args.dataset_name).stem.replace("/", "_")

    # Run Analysis and Generate Artifacts
    summary = pipeline.analyze_dataframe(
        df=df,
        dataset_name=dataset_slug,
        prompt_col="prompt",
        response_col="response",
    )

    logger.info("=== EDA Profiling Complete ===")
    logger.info("Total Samples Profiled: %d", summary.total_samples)
    logger.info(
        "Recommended Max Sequence Length: %d tokens", summary.recommended_max_seq_length
    )
    logger.info(
        "Token Length Quantiles (P50 / P90 / P99): %d / %d / %d",
        int(summary.total_sequence_quantiles.p50),
        int(summary.total_sequence_quantiles.p90),
        int(summary.total_sequence_quantiles.p99),
    )
    logger.info("Artifacts saved to: %s", Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()
