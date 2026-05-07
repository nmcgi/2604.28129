# Latent Adversarial Detection (LAD)

> Original paper: https://arxiv.org/abs/2604.28129

(Unfamiliar with a term? Check out [VOCAB.md](VOCAB.md))

Reproduction of **arXiv:2604.28129v1** — *Latent Adversarial Detection: Adaptive Probing
of LLM Activations for Multi-Turn Attack Detection* (Kulkarni, 2026).

LAD monitors an LLM's residual stream in real time and flags multi-turn prompt-injection
attacks before the first overtly harmful turn. It works by tracking *adversarial
restlessness* — the elevated cumulative activation drift that phase-shifted attacks
produce — using five scalar trajectory features and raw activation vectors fed into an
XGBoost probe (primary, Sections 6–7). An optional contrastive MLP encoder stage
(Section 3.4 / Appendix C) can replace the raw activations with 128-dim embeddings;
pass `--variant contrastive` to the training and inference scripts to use it.

## Included Starter Dataset

40 training + 10 eval synthetic conversations ship with this repo under
`data/synthetic/`. Each conversation carries three-phase turn-level labels
(benign / pivoting / adversarial) across the six attack categories from the
paper: gradual escalation, trust building, context poisoning, role accumulation,
instruction fragmentation, and tool-use exploitation. This is enough to run the
full smoke-check pipeline on a consumer GPU without any API calls or data
generation — see [Quick Start](#quick-start--smoke-test) below.

---

## Prerequisites

### Hardware

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

### Software

- [Python 3.10+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/#installation)

### Accounts

- HuggingFace account with **approved access** to:
  - [`google/gemma-3-27b-it`](https://huggingface.co/google/gemma-3-27b-it)
  - [`meta-llama/Llama-3.1-70B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)
  - [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) *(low-resource path)*
  - [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) *(low-resource path)*
  - [`google/gemma-2-2b-it`](https://huggingface.co/google/gemma-2-2b-it) *(low-resource path)*
- `HF_TOKEN` environment variable set to your token
- `OPENAI_API_KEY` environment variable set to your key (required for three-phase labeling in Phase 3)

## Quick Start — Smoke Test

### Consumer GPU (≥3 GB VRAM) — no API key, no HF token required

Uses the included 50-conversation starter dataset and Qwen 2.5 1.5B (ungated,
≈3 GB VRAM in bf16). Complete [Phase 1](#phase-1--environment-setup) first.

```bash
# 1. Install deps
uv sync --extra quantize

# 2. Download model (no HuggingFace access approval needed)
uvx hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./models/qwen-1.5b-it

# 3. Extract activations from the included starter dataset (~10–20 min)
uv run extract_activations.py --model qwen1.5b --source synthetic --split train
uv run extract_activations.py --model qwen1.5b --source synthetic --split eval

# 4. Train probe (CPU, ~2 min)
uv run train_probe.py --model qwen1.5b

# 5. Evaluate
uv run eval_probe.py --model qwen1.5b
```

Expected: ~85–90% detection, high FPR due to the small 50-conversation training
set. This validates that *adversarial restlessness* is detectable even on a 1.5B
model; it is not representative of the paper's results (2,625 conversations,
24–70B models, 2–4% FPR).

### A100 80 GB — closer to paper conditions

Uses the included starter dataset with Qwen 2.5 32B. Same commands as above but
substitute `--model qwen` (and download `Qwen/Qwen2.5-32B-Instruct` in Phase 2).
Takes ~30 minutes end-to-end. To generate a larger dataset instead of using the
included one, see [Phase 3](#phase-3--synthetic-dataset-generation-2-4-h-2h200).

Expected on 50-conversation smoke test: ~89% detection rate but high FPR (~57–74%)
due to limited training data. With the full dataset (2,625 conversations) and
`--variant contrastive`, FPR drops to 2–4% while detection stays at 85–89% (paper
Sections 6–7). The scalar-only ablation (Section 5) also achieves ~89.6% detection
but retains the high 57–74% FPR regardless of dataset size.

## Phase 1 — Environment Setup

```bash
git clone https://github.com/nmcgi/2604.28129.git
cd 2604.28129

uv sync --extra vllm --extra quantize

mkdir -p data/lmsys/train data/lmsys/eval \
         data/safedial/train data/safedial/eval \
         data/activations \
         models
```

## Phase 2 — Model Download

```bash
export HF_TOKEN="hf_..."

# Gemma 3 27B instruction-tuned  (layer ℓ=32, d=5376)
uvx hf download google/gemma-3-27b-it \
  --local-dir ./models/gemma-3-27b-it --token $HF_TOKEN

# Mistral Small 3.1 24B instruction-tuned  (layer ℓ=24, d=5120)
uvx hf download mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
  --local-dir ./models/mistral-24b-it --token $HF_TOKEN

# Qwen 2.5 32B instruction-tuned  (layer ℓ=32, d=5120)
uvx hf download Qwen/Qwen2.5-32B-Instruct \
  --local-dir ./models/qwen-32b-it --token $HF_TOKEN

# Llama 3.1 70B instruction-tuned  (layer ℓ=40, d=8192)
uvx hf download meta-llama/Llama-3.1-70B-Instruct \
  --local-dir ./models/llama-70b-it --token $HF_TOKEN

# Qwen3-235B-A22B — for synthetic dataset generation only (Phase 3)
uvx hf download Qwen/Qwen3-235B-A22B \
  --local-dir ./models/qwen3-235b --token $HF_TOKEN
```

Model configuration (layer index and hidden dimension used during extraction):

| Model key | Path | Layer | d |
|---|---|---|---|
| `gemma` | `models/gemma-3-27b-it` | 32 | 5376 |
| `mistral` | `models/mistral-24b-it` | 24 | 5120 |
| `qwen` | `models/qwen-32b-it` | 32 | 5120 |
| `llama` | `models/llama-70b-it` | 40 | 8192 |

## Phase 3 — Synthetic Dataset Generation *(2–4 h, 2×H200)*

Start the vLLM server on 2×H200:

```bash
vllm serve ./models/qwen3-235b \
  --tensor-parallel-size 2 \
  --quantization fp8 \
  --max-model-len 8192 \
  --port 8000
```

In a second terminal, run generation:

```bash
uv run generate_synthetic.py --n-train 1125 --n-eval 797
```

Output: `data/synthetic/{train,eval}/conv_{i}.json` — 1,125 training + 797 eval conversations.

Also ingest the real-world sources (requires `OPENAI_API_KEY` for three-phase labeling):

```bash
uv run ingest_lmsys.py    # 1,200 train + 800 eval from lmsys/lmsys-chat-1m
uv run ingest_safedial.py # 300 train + 200 eval from Hongyu-Cao/SafeDialBench
```

Both scripts default to `--label-mode threephase`: each user turn is labeled
benign / pivoting / adversarial by `gpt-4o-mini` using conversation context,
matching the paper's three-phase scheme (Section 3.3). Pass `--label-mode moderation`
(LMSYS) or `--label-mode all_adversarial` (SafeDialBench) for the cheaper binary
fallback, but expect higher false-positive rates at evaluation time.

> **Alternative if Qwen3-235B-A22B is unavailable**: pass `--provider anthropic` or
> `--provider openai` with an appropriate `--gen-model` and `--base-url` instead of
> running a local vLLM server. No code changes required.

## Phase 4 — Activation Extraction *(1–8 h per model, GPU)*

Run for each model × source × split combination:

```bash
for MODEL in gemma mistral qwen llama; do
  for SOURCE in synthetic lmsys safedial; do
    for SPLIT in train eval; do
      uv run extract_activations.py --model $MODEL --source $SOURCE --split $SPLIT
    done
  done
done
```

Expected output per model (~48 MB per 1,000 conversations):

```
data/activations/<model>/train_synthetic.npz
data/activations/<model>/eval_synthetic.npz
data/activations/<model>/train_lmsys.npz
data/activations/<model>/eval_lmsys.npz
data/activations/<model>/train_safedial.npz
data/activations/<model>/eval_safedial.npz
```

## Phase 5 — Probe Training *(~12 min per model, CPU)*

```bash
for MODEL in gemma mistral qwen llama; do
  uv run train_probe.py --model $MODEL
done
```

Default (`--variant standard`): `models/<model>/xgb.json`, `models/<model>/scaler.pkl`.
With `--variant contrastive`: additionally saves `models/<model>/encoder.pt`
(and uses `xgb_contrastive.json` / `scaler_contrastive.pkl`).

## Phase 6 — Evaluation *(~1 min per model, CPU)*

```bash
uv run eval_probe.py --model all
```

Expected output (combined held-out set, n=1,797):

| Model | Detection | False Positive Rate |
|---|---|---|
| Gemma 3 27B | 85.3% | 4.0% |
| Mistral Small 3.1 24B | 87.7% | 2.8% |
| **Qwen 2.5 32B** | **89.4%** | **2.4%** |
| Llama 3.1 70B | 87.3% | 2.9% |

Synthetic-only held-out (Gemma 3 27B): **93.8% detection / 3.5% FP** (paper Section 5).

## Phase 7 — Inference Demo

```bash
uv run lad_infer.py --model qwen
```

Expected: turns 1–6 show P(adv) < 0.5; turn 9+ ("avoid triggering detection mechanisms")
exceeds θ=0.5 and prints `*** FLAGGED ***`. Latency ~100 ms/turn.

## Low-Resource Path — GTX 1650 (4 GB VRAM)

This path runs the full pipeline on consumer hardware using small models.
Detection performance will differ from the paper; the goal is to validate that the
*approach* (trajectory scalars + probe) transfers to smaller models.

### Hardware fit

| Model key | Params | VRAM (bf16) | VRAM (4-bit) | Fits GTX 1650? | Gated? |
|---|---|---|---|---|---|
| `qwen1.5b` | 1.5 B | ~3 GB | ~1 GB | **Yes (bf16)** | No |
| `llama1b`  | 1 B   | ~2 GB | ~0.7 GB | **Yes (bf16)** | **Yes** |
| `llama3b`  | 3 B   | ~6 GB | ~2 GB | Yes (4-bit only) | **Yes** |
| `gemma2b`  | 2 B   | ~4.5 GB | ~1.5 GB | Yes (4-bit only) | **Yes** |
| `phi3.5`   | 3.8 B | ~7.6 GB | ~2.5 GB | Yes (4-bit only) | No |

### Step 1 — Model Download

```bash
export HF_TOKEN="hf_..."

# Qwen 2.5 1.5B instruction-tuned  (layer ℓ=14, d=1536)
uvx hf download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir ./models/qwen-1.5b-it --token $HF_TOKEN

# Llama 3.2 1B instruction-tuned  (layer ℓ=8, d=2048)
uvx hf download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir ./models/llama-1b-it --token $HF_TOKEN

# Llama 3.2 3B instruction-tuned  (layer ℓ=14, d=3072)  — needs --quantize
uvx hf download meta-llama/Llama-3.2-3B-Instruct \
  --local-dir ./models/llama-3b-it --token $HF_TOKEN

# Gemma 2 2B instruction-tuned  (layer ℓ=13, d=2304)  — needs --quantize
uvx hf download google/gemma-2-2b-it \
  --local-dir ./models/gemma-2b-it --token $HF_TOKEN

# Phi-3.5 Mini instruction-tuned  (layer ℓ=16, d=3072)  — needs --quantize
uvx hf download microsoft/Phi-3.5-mini-instruct \
  --local-dir ./models/phi-3.5-mini --token $HF_TOKEN
```

### Step 2 — Dataset

The included starter dataset (40 train + 10 eval) is ready to use — skip to
Step 3. To generate more data, load any instruction-tuned model in LM Studio
(e.g. `Qwen 2.5 7B Q4`, `Llama 3.1 8B Q4`), enable the local server (default
port 1234), then:

```bash
uv run generate_synthetic.py \
  --base-url http://localhost:1234/v1 \
  --gen-model "your-model-name-as-shown-in-lmstudio" \
  --n-train 100 --n-eval 40
```

> **Note:** LM Studio is only used for *generating* the conversation dataset.
> Activation extraction still requires loading model weights locally via
> `transformers` — the probe hooks directly into the model's layers, which an
> API cannot expose.

### Step 3 — Extract activations (GPU, ~10–30 min)

```bash
# Fits without quantization (recommended for GTX 1650):
uv run extract_activations.py --model qwen1.5b --source synthetic --split train
uv run extract_activations.py --model qwen1.5b --source synthetic --split eval

# Larger small models need --quantize:
uv run extract_activations.py --model llama3b --quantize --source synthetic --split train
uv run extract_activations.py --model llama3b --quantize --source synthetic --split eval
```

### Step 4 — Train probe (CPU, ~2 min)

```bash
uv run train_probe.py --model qwen1.5b
```

### Step 5 — Evaluate and run demo

```bash
uv run eval_probe.py --model qwen1.5b
uv run lad_infer.py  --model qwen1.5b
# or with 4-bit:
uv run lad_infer.py  --model llama3b --quantize
```

## Quick Reference

| Script | Runs on | Est. time |
|---|---|---|
| `generate_synthetic.py` | 2×H200 (vLLM) | 2–4 h |
| `extract_activations.py` | 1–2×H100/H200 | 1–8 h/model |
| `train_probe.py` | CPU | ~12 min/model |
| `eval_probe.py` | CPU | ~1 min/model |
| `lad_infer.py` | GPU (target model) + CPU | ~100 ms/turn |

Total GPU-hours for full 4-model reproduction: ~3 h generation + ~20 h extraction.
Total CPU-hours: ~1 h training + evaluation.

## Repository Structure

```
.
├── README.md                  — this file
├── VOCAB.md                   — plain-English glossary of all jargon terms
├── pyproject.toml             — dependency spec (uv)
├── generate_synthetic.py      — synthetic dataset generation (Phase 3)
├── ingest_lmsys.py            — LMSYS-Chat-1M ingestion with 3-phase labeling
├── ingest_safedial.py         — SafeDialBench ingestion with 3-phase labeling
├── extract_activations.py     — activation extraction + trajectory scalars (Phase 4)
├── train_probe.py             — contrastive encoder + XGBoost probe (Phase 5)
├── eval_probe.py              — held-out evaluation + early detection (Phase 6)
├── lad_infer.py               — real-time inference demo (Phase 7)
├── eval_adversarial_robustness.py  — α-sweep robustness simulation (Appendix M)
├── eval_cross_model_transfer.py    — cross-model F1 matrix (Appendix F)
├── eval_label_validation.py        — LLM-as-judge label validation (Appendix D.2)
├── eval_layer_sensitivity.py       — layer sensitivity sweep (Appendix G)
├── eval_sae_ablation.py            — GemmaScope SAE ablation (Appendix O)
├── data/synthetic/            — starter dataset (40 train + 10 eval); tracked in git
│   ├── train/conv_*.json
│   └── eval/conv_*.json
├── data/lmsys/                — not tracked; ingested by ingest_lmsys.py
├── data/safedial/             — not tracked; ingested by ingest_safedial.py
└── data/activations/          — not tracked; generated by extract_activations.py
    └── <model>/<split>_<source>.npz
└── models/                    — saved by train_probe.py; not tracked in git
    └── <model>/{xgb.json, scaler.pkl}          — standard variant
              {encoder.pt, xgb_contrastive.json} — contrastive variant
```

## Citation

```bibtex
@article{kulkarni2026lad,
  title   = {Latent Adversarial Detection: Adaptive Probing of {LLM} Activations
             for Multi-Turn Attack Detection},
  author  = {Kulkarni, Prashant},
  journal = {arXiv preprint arXiv:2604.28129},
  year    = {2026}
}
```
