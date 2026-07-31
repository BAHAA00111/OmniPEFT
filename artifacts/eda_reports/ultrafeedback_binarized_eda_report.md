# OmniPEFT EDA Report: ultrafeedback_binarized

## 1. Dataset Overview

* **Total Samples Profiled:** `61,115`
* **Recommended Sequence Length Cutoff:** `2048`

## 2. Feature Health & Missing Values

| Feature Name | Null / Missing Count | Status |
| :--- | :--- | :--- |
| `prompt` | 0 | PASS |
| `response` | 0 | PASS |

## 3. Sequence Length Distribution Analysis (Token Count)

| Metric | Prompt Tokens | Response Tokens | Total Combined Tokens |
| :--- | :--- | :--- | :--- |
| **Mean ± Std** | 159.1 ± 230.7 | 275.2 ± 238.2 | 434.3 ± 308.2 |
| **P50 (Median)** | 77 | 222 | 378 |
| **P90** | 417 | 618 | 833 |
| **P95** | 621 | 738 | 989 |
| **P99** | **1131** | **905** | **1428** |
| **Max Length** | 5629 | 1482 | 5818 |

## 4. Engineering Recommendations

* **Context Truncation Strategy:** Setting max sequence length to **2048** tokens retains **>99%** of complete instruction sequences without information loss.
* **VRAM Impact:** Limits peak activation memory footprint during backward passes under 8GB VRAM.
