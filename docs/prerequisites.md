# Prerequisites

## Hardware

| Target model | Params | VRAM required | Minimum GPU |
|---|---|---|---|
| Gemma 3 27B-IT | 27B | ~55 GB | 1×A100 80 GB |
| Mistral Small 3.1 24B | 24B | ~48 GB | 1×A100 80 GB |
| Qwen 2.5 32B-IT | 32B | ~64 GB | 1×A100 80 GB |
| Llama 3.1 70B-IT | 70B | ~140 GB | **2×H100 80 GB** |
| Qwen3-235B-A22B *(data gen only)* | 235B MoE | ~240 GB (fp8) | **2×H200** |

Probe training and all evaluation run on **CPU only** once activations are cached.
The paper used [RunPod](https://runpod.io) H200 SXM pods. An A100 80 GB or H200 pod covers Gemma/Mistral/Qwen extraction; rent a 2×H100 pod for Llama; rent a 2×H200 pod for synthetic data generation. Storage budget: ~70 GB per 27–32B model, ~140 GB for Llama 70B.
Download models to an NVMe volume attached to your cloud instance to avoid re-downloading.

## Software

- [Python 3.10+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/#installation)

## Accounts

- HuggingFace account with **approved access** to:
  - [`google/gemma-3-27b-it`](https://huggingface.co/google/gemma-3-27b-it)
  - [`meta-llama/Llama-3.1-70B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)
  - [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) *(low-resource path)*
  - [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) *(low-resource path)*
  - [`google/gemma-2-2b-it`](https://huggingface.co/google/gemma-2-2b-it) *(low-resource path)*
- `HF_TOKEN` environment variable set to your token
- `OPENAI_API_KEY` environment variable set to your key (required for three-phase labeling in Phase 3)
