"""
Evaluates baseline vs. fine-tuned/fused models over target test datasets, calculates streaming
perplexity and text generation metrics, and exports sample-paired score vectors.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from omnipeft.analytics.metrics import EvaluationMetricsEngine, PerplexityTracker

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("omnipeft.evaluate")


class ModelEvaluator:
    """Execution engine for batch inference and evaluation metric extraction."""

    def __init__(self, model_name_or_path: str, device: str = "cuda") -> None:
        """Initialize model and tokenizer for evaluation."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(
            "Loading evaluation model from: %s on %s", model_name_or_path, self.device
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        loaded_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True,
        )

        self.model: PreTrainedModel = cast(PreTrainedModel, loaded_model).to(self.device)  # type: ignore[assignment]
        self.model.eval()

    @torch.no_grad()
    def evaluate_dataset(
        self,
        prompts: List[str],
        references: List[str],
        max_new_tokens: int = 128,
    ) -> Dict[str, Any]:
        """Perform evaluation pass computing aggregate metrics and sample score vectors."""
        ppl_tracker = PerplexityTracker()
        predictions: List[str] = []
        sample_rouge_l: List[float] = []

        logger.info("Executing evaluation inference over %d samples...", len(prompts))

        for idx, (prompt, ref) in enumerate(zip(prompts, references)):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # 1. Forward Pass for Loss / Perplexity
            labels = self.tokenizer(ref, return_tensors="pt")["input_ids"].to(
                self.device
            )

            loss_outputs: Any = self.model(input_ids=labels, labels=labels)
            loss_val = float(loss_outputs.loss.item())
            ppl_tracker.update(loss=loss_val, num_tokens=labels.shape[1])

            # 2. Text Generation Pass
            generated_ids: torch.Tensor = self.model.generate(  # type: ignore[attr-defined]
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            input_length = inputs["input_ids"].shape[1]
            pred_tokens = generated_ids[0][input_length:]
            pred_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
            predictions.append(pred_text)

            # Compute per-sample LCS ROUGE-L score vector
            s_rouge = EvaluationMetricsEngine.compute_rouge_l(pred_text, ref)
            sample_rouge_l.append(round(s_rouge * 100.0, 4))

        # Global aggregate evaluation
        aggregate_metrics = EvaluationMetricsEngine.compute_generation_metrics(
            predictions, references
        )
        aggregate_metrics["perplexity"] = round(ppl_tracker.perplexity, 4)

        return {
            "aggregate": aggregate_metrics,
            "sample_scores": sample_rouge_l,
            "predictions": predictions,
        }


def run_evaluation_entrypoint(
    base_model_path: str,
    fused_model_path: str,
    test_dataset_path: str,
    output_dir: str = "./artifacts/evaluation_results",
) -> None:
    """Run pipeline to evaluate baseline and fused models, exporting raw evaluations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    test_file = Path(test_dataset_path)
    if not test_file.exists():
        logger.warning(
            "Test dataset file '%s' not found. Creating dynamic evaluation sample...",
            test_dataset_path,
        )
        test_data = [
            {
                "prompt": "Summarize the key benefits of Parameter-Efficient Fine-Tuning (PEFT):",
                "reference": "PEFT reduces VRAM footprint, speeds up training, and preserves model performance.",
            },
            {
                "prompt": "What is the formula for linear weight fusion?",
                "reference": "W_fused = W_base + (alpha / r) * (B @ A)",
            },
        ]
    else:
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)

    prompts = [item["prompt"] for item in test_data]
    references = [item["reference"] for item in test_data]

    # Evaluate Baseline
    evaluator_base = ModelEvaluator(base_model_path)
    base_results = evaluator_base.evaluate_dataset(prompts, references)

    # Evaluate Fused/Fine-tuned Model
    evaluator_fused = ModelEvaluator(fused_model_path)
    fused_results = evaluator_fused.evaluate_dataset(prompts, references)

    payload = {
        "base_model": base_model_path,
        "fused_model": fused_model_path,
        "base_results": base_results["aggregate"],
        "fused_results": fused_results["aggregate"],
        "scores_base": base_results["sample_scores"],
        "scores_fused": fused_results["sample_scores"],
    }

    raw_eval_file = output_path / "raw_evaluation_vectors.json"
    with open(raw_eval_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Evaluation complete. Results saved to: %s", raw_eval_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniPEFT Unified Evaluation Engine")
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Baseline model path or HF identifier",
    )
    parser.add_argument(
        "--fused_model",
        type=str,
        default="./artifacts/fused_model",
        help="Fused/fine-tuned model path",
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="./data_storage/test_split.json",
        help="Path to test JSON dataset",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./artifacts/evaluation_results"
    )

    args = parser.parse_args()
    run_evaluation_entrypoint(
        base_model_path=args.base_model,
        fused_model_path=args.fused_model,
        test_dataset_path=args.test_dataset,
        output_dir=args.output_dir,
    )
