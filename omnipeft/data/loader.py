"""
OmniPEFT Memory-Mapped Streaming Data Loader Engine.

Streams 150k+ instruction samples on-the-fly using PyTorch IterableDataset
and Hugging Face Streaming Datasets. Formats prompt-response templates and applies
selective label masking (-100) to ensure backpropagation gradients are computed
exclusively on target generation tokens.
"""

import logging
from typing import Any, Dict, Iterator, List, Optional, Union, cast

import torch
from datasets import IterableDataset as HFIterableDataset, load_dataset
from torch.utils.data import DataLoader, IterableDataset as PyTorchIterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


class StreamingInstructionDataset(PyTorchIterableDataset):
    """Memory-efficient streaming dataset for instruction fine-tuning."""

    DEFAULT_PROMPT_TEMPLATE = (
        "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    )

    def __init__(
        self,
        dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized",
        split: str = "train_sft",
        tokenizer_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct",
        tokenizer: Optional[TokenizerType] = None,
        max_seq_length: int = 2048,
        prompt_template: Optional[str] = None,
        streaming: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.max_seq_length = max_seq_length
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.streaming = streaming
        self.seed = seed

        logger.info(
            "Initializing StreamingInstructionDataset | Dataset: %s | Split: %s | Max Len: %d",
            self.dataset_name,
            self.split,
            self.max_seq_length,
        )

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path,
                trust_remote_code=True,
                use_fast=True,
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _extract_prompt_and_response(self, sample: Dict[str, Any]) -> tuple[str, str]:
        # Schema 1: UltraFeedback / ChatML messages format
        if "messages" in sample:
            messages = sample["messages"]
            prompt_text = ""
            response_text = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    prompt_text = content
                elif role == "assistant":
                    response_text = content
            return prompt_text, response_text

        # Schema 2: Standard instruction/prompt/response format
        prompt_text = str(sample.get("prompt", sample.get("instruction", "")))
        response_text = str(sample.get("response", sample.get("output", "")))
        return prompt_text, response_text

    def _process_sample(
        self, sample: Dict[str, Any]
    ) -> Optional[Dict[str, torch.Tensor]]:
        prompt_raw, response_raw = self._extract_prompt_and_response(sample)

        if not prompt_raw or not response_raw:
            return None

        formatted_prompt = self.prompt_template.format(instruction=prompt_raw)
        formatted_response = f"{response_raw}<|im_end|>"

        # Tokenize prompt portion separately to measure precise prompt token length
        prompt_ids = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(
            formatted_response, add_special_tokens=False
        )

        # Concatenate full sequence
        input_ids = prompt_ids + response_ids

        # Truncate to max_seq_length if required
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[: self.max_seq_length]

        # Construct Labels with Loss Masking (-100 for prompt tokens)
        prompt_len = len(prompt_ids)
        if prompt_len >= len(input_ids):
            # Sequence truncated before response began
            labels = [-100] * len(input_ids)
        else:
            labels = ([-100] * prompt_len) + input_ids[prompt_len:]

        # Attention mask (1 for valid tokens)
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Stream and yield processed training samples indefinitely/per split."""
        raw_dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=self.streaming,
        )

        if self.streaming:
            # Cast raw_dataset so Pylance knows it's an HFIterableDataset
            hf_dataset = cast(HFIterableDataset, raw_dataset)
            hf_dataset = hf_dataset.shuffle(seed=self.seed, buffer_size=10_000)
        else:
            hf_dataset = raw_dataset

        for raw_sample in hf_dataset:
            sample_dict: Dict[str, Any] = dict(raw_sample)
            processed = self._process_sample(sample_dict)
            if processed is not None:
                yield processed


class DynamicDataCollator:
    """Pad batch tensors dynamically to the longest sequence in the micro-batch."""

    def __init__(self, pad_token_id: int, ignore_index: int = -100) -> None:
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = [b for b in batch if b is not None]
        if not batch:
            return {}

        batch_max_len = max(len(s["input_ids"]) for s in batch)

        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []

        for sample in batch:
            input_ids = sample["input_ids"]
            attention_mask = sample["attention_mask"]
            labels = sample["labels"]

            pad_len = batch_max_len - len(input_ids)

            # Pad input_ids with pad_token_id
            padded_input_ids.append(
                torch.cat(
                    [
                        input_ids,
                        torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                    ]
                )
            )

            # Pad attention_mask with 0
            padded_attention_masks.append(
                torch.cat(
                    [
                        attention_mask,
                        torch.zeros(pad_len, dtype=torch.long),
                    ]
                )
            )

            # Pad labels with ignore_index (-100)
            padded_labels.append(
                torch.cat(
                    [
                        labels,
                        torch.full((pad_len,), self.ignore_index, dtype=torch.long),
                    ]
                )
            )

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_masks),
            "labels": torch.stack(padded_labels),
        }


def create_streaming_dataloader(
    dataset_name: str,
    tokenizer: TokenizerType,
    split: str = "train_sft",
    batch_size: int = 2,
    max_seq_length: int = 2048,
    seed: int = 42,
) -> DataLoader[Dict[str, torch.Tensor]]:
    dataset = StreamingInstructionDataset(
        dataset_name=dataset_name,
        split=split,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        streaming=True,
        seed=seed,
    )

    # Safely resolve and cast pad_token_id to int to satisfy Pylance
    raw_pad_id = tokenizer.pad_token_id
    if raw_pad_id is None:
        raw_pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    if isinstance(raw_pad_id, list):
        pad_token_id = int(raw_pad_id[0])
    else:
        pad_token_id = int(raw_pad_id)

    collator = DynamicDataCollator(pad_token_id=pad_token_id)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collator,
        pin_memory=True,
    )
