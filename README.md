<div align="center">

# OmniPEFT

### Full-Stack Fine-Tuning Engine, CUDA Memory Allocator & Statistical Suite

Efficient **QLoRA** fine-tuning with **NF4 quantization**, **CUDA memory optimization**, **dynamic LoRA weight fusion**, automated evaluation, and reproducible benchmarking—all in a unified production-ready framework.

<br>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?logo=pytorch&logoColor=white)](#)
[![CUDA](https://img.shields.io/badge/CUDA-12.1+-76B900?logo=nvidia&logoColor=white)](#)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-success?logo=githubactions)](#)

</div>

---

## Overview

OmniPEFT is a modular framework for **parameter-efficient fine-tuning (PEFT)** of Large Language Models, combining modern training techniques with production-oriented systems engineering.

The framework integrates efficient fine-tuning, CUDA runtime optimization, automated evaluation, and statistical validation into a single reproducible workflow, making it suitable for both research experimentation and production deployment.

### Core Capabilities

- 🚀 QLoRA fine-tuning with NF4 4-bit quantization
- ⚡ CUDA memory allocator optimization
- 🔄 Dynamic LoRA weight fusion for zero-overhead inference
- 📊 Automated evaluation using Perplexity, BLEU, and ROUGE
- 🧪 Statistical significance testing
- 📈 Dataset EDA and experiment analytics
- 🐳 Docker-ready deployment
- ✅ CI/CD with Ruff, Black, MyPy, and PyTest

---

## Table of Contents

- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Performance](#performance)
- [Quick Start](#quick-start)
- [Docker](#docker)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

# Architecture

```text
                                      OmniPEFT
┌──────────────────────────────────────────────────────────────────────────────┐
│           Production LLM Fine-Tuning & Systems Framework                     │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
 ┌──────────────────┐      ┌────────────────────┐      ┌────────────────────┐
 │ Systems Runtime  │      │ Training Engine    │      │ Analytics Engine   │
 ├──────────────────┤      ├────────────────────┤      ├────────────────────┤
 │ CUDA Allocator   │      │ QLoRA Trainer      │      │ Metrics            │
 │ AMP              │      │ Checkpointing      │      │ Evaluation         │
 │ Weight Fusion    │      │ Inference          │      │ Hypothesis Tests   │
 │ Benchmarking     │      │ Optimization       │      │ Visualization      │
 └──────────────────┘      └────────────────────┘      └────────────────────┘
                                      │
                                      ▼
                           Production Deployment Pipeline
```

---

## Key Components

| Layer | Responsibility |
|--------|----------------|
| **Systems** | CUDA optimization, memory management, runtime benchmarking |
| **Engine** | QLoRA training, inference, checkpoint management |
| **Analytics** | Evaluation metrics, experiment tracking, statistical testing |
| **Operations** | Docker, CI/CD, configuration, reproducibility |

---

## Design Principles

OmniPEFT is built around four engineering principles:

- **Performance** — Optimize GPU memory usage and training throughput.
- **Reproducibility** — Configuration-driven experiments with deterministic workflows.
- **Modularity** — Independent components for systems, training, analytics, and deployment.
- **Production Readiness** — Automated testing, Docker support, and CI/CD integration.

---

# Repository Structure

OmniPEFT follows a modular architecture that cleanly separates machine learning, systems engineering, analytics, and operational tooling.

```text
OmniPEFT/
├── .github/workflows/         # Production CI/CD workflows
│   ├── docker_build.yaml      # Multi-arch container build & GHCR publish
│   └── lint_and_test.yaml     # Black, Ruff, MyPy, and Pytest automated gate
├── artifacts/                 # Generated outputs & evaluation summaries
│   ├── eda_reports/           # Token distribution plots & dataset statistics
│   └── evaluation_results/    # Raw evaluation vectors and t-test summaries
├── configs/                   # Modular YAML configuration specifications
│   ├── data/dataset_150k.yaml # Tokenizer & dataset parameters
│   ├── model/qlora_nf4.yaml   # LoRA rank, alpha, target modules, quantization
│   ├── system/cuda_rtx.yaml   # PyTorch allocator & mixed precision setup
│   └── config.yaml            # Master configuration orchestrator
├── docker/                    # Deployment specs
│   ├── .dockerignore
│   ├── Dockerfile             # Multi-stage PyTorch/CUDA runtime image
│   └── docker-compose.yaml    # Local multi-container evaluation stack
├── omnipeft/                  # Core Python Source Package
│   ├── analytics/             # Telemetry & statistical hypothesis engines
│   │   ├── convergence_logger.py
│   │   ├── hypothesis_testing.py
│   │   └── metrics.py
│   ├── data/                  # Data loading & EDA pipelines
│   │   ├── eda_pipeline.py
│   │   └── loader.py
│   ├── engine/                # Fine-tuning & inference controllers
│   │   ├── checkpoint_manager.py
│   │   ├── inference_runner.py
│   │   └── qlora_trainer.py
│   └── systems/               # Low-level systems engineering & CUDA memory
│       ├── amp_config.py
│       ├── benchmark_engine.py
│       ├── memory_allocator.py
│       └── weight_fusion.py
├── scripts/                   # Standalone execution & benchmark entry points
│   ├── benchmark_fusion.py    # Runtime latency degradation benchmarking
│   ├── merge_weight.py        # Adapter weight dynamic fusion script
│   ├── run_eda.py             # Dataset processing trigger
│   └── run_hypothesis_test.py # Paired t-test statistical execution
├── tests/                     # Test Suite (35 Unit/Integration Tests)
│   ├── test_analytics.py
│   ├── test_benchmark_engine.py
│   ├── test_hypothesis.py
│   ├── test_memory_allocator.py
│   ├── test_metrics.py
│   └── test_weight_fusion.py
├── evaluate.py                # Standalone model evaluation entry point
├── pyproject.toml             # Tooling configuration (Ruff, MyPy, Pytest, Black)
├── requirements.txt           # Verified package dependencies
└── train.py                   # Main QLoRA training engine
```

---

### Module Overview

| Module | Responsibility |
|---------|----------------|
| **systems** | CUDA runtime optimization, AMP, memory allocator, benchmarking, weight fusion |
| **engine** | QLoRA training, checkpoint management, inference pipeline |
| **analytics** | Evaluation metrics, convergence logging, hypothesis testing |
| **data** | Dataset loading, preprocessing, tokenization and EDA |
| **scripts** | CLI utilities for benchmarking, evaluation and experiments |
| **tests** | Automated unit and integration test suite |

---

# Performance

OmniPEFT focuses on maximizing training efficiency while minimizing GPU memory usage.

## GPU Memory Optimization

| Method | Precision | Peak VRAM | Status |
|----------|-----------|-----------|--------|
| Full Fine-Tuning | FP16 | 16.4 GB | OOM |
| Standard LoRA | FP16 | 11.2 GB | Near Limit |
| **OmniPEFT (QLoRA)** | **NF4 (4-bit)** | **5.7 GB** | Stable |

### Memory Reduction

| Improvement | Value |
|-------------|------:|
| Peak VRAM Reduction | **≈65%** |
| Quantization | **NF4 (4-bit)** |
| Mixed Precision | **FP16 / BF16** |
| Gradient Checkpointing | **Supported** |

---

## Dynamic LoRA Weight Fusion

Instead of executing LoRA adapters during every forward pass, OmniPEFT merges adapter weights into the base model before deployment.

```text
Traditional LoRA

Input
  │
  ▼
Base Linear ───────┐
                   │
LoRA A → LoRA B ───┤
                   ▼
               Add Outputs
                   ▼
                Prediction
```

↓

```text
OmniPEFT

Input
  │
  ▼
Fused Linear Layer
  │
  ▼
Prediction
```

### Benefits

- Zero additional inference latency
- Simplified deployment pipeline
- Lower runtime overhead
- Single deployable model
- No adapter execution during inference

---

## Evaluation Pipeline

Every training run produces standardized evaluation reports.

| Category | Metrics |
|----------|----------|
| Language Quality | Perplexity, BLEU |
| Text Similarity | ROUGE-1, ROUGE-2, ROUGE-L |
| Training | Loss Curves, Convergence |
| Statistics | Paired Two-Tailed t-Test |

Example summary:

| Metric | Improvement | p-value |
|---------|------------:|---------|
| ROUGE-1 | +13.4 | <0.01 |
| ROUGE-2 | +10.9 | <0.01 |
| ROUGE-L | +13.1 | <0.01 |
| BLEU | +14.3 | <0.01 |
| Perplexity | ↓45% | <0.01 |

> All reported improvements are statistically significant using paired two-tailed t-tests (α = 0.01).

---

# Technology Stack

| Category | Technologies |
|-----------|--------------|
| Deep Learning | PyTorch, Transformers |
| Fine-Tuning | PEFT, LoRA, QLoRA |
| Quantization | BitsAndBytes (NF4) |
| Systems | CUDA, AMP |
| Evaluation | ROUGE, BLEU, Perplexity |
| Analytics | TensorBoard, Weights & Biases |
| DevOps | Docker, GitHub Actions |
| Quality | Ruff, Black, MyPy, PyTest |

---

# Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/OmniPEFT.git
cd OmniPEFT
```

### 2. Install Dependencies

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
# .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 3. Train a Model

```bash
python train.py --config configs/config.yaml
```

### 4. Evaluate

```bash
python evaluate.py --config configs/config.yaml
```

### 5. Run with Docker

```bash
docker compose up
```

---

# Development

Before submitting changes, run the complete quality pipeline.

```bash
ruff check .

black .

mypy .

pytest -v
```

Every pull request is automatically validated through GitHub Actions.

---

# License

Distributed under the **Apache License 2.0**.

---

<div align="center">

## ⭐ Support the Project

If you find **OmniPEFT** useful, consider giving the repository a **star**.

It helps increase visibility, supports future development, and encourages continued open-source contributions.

---

**Built for efficient, reproducible, and production-ready LLM fine-tuning.**

</div>