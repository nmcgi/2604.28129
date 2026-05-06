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

## Prerequisites

### Hardware

| Target model | Params | VRAM required | Minimum GPU |
|---|---|---|---|
| Gemma 3 27B-IT | 27B | ~55 GB | 1×H200 (141 GB) |
| Mistral Small 3.1 24B | 24B | ~48 GB | 1×H200 (141 GB) |
| Qwen 2.5 32B-IT | 32B | ~64 GB | 1×H200 (141 GB) |
| Llama 3.1 70B-IT | 70B | ~140 GB | **2×H100 80 GB** |
| Qwen3-235B-A22B *(data gen only)* | 235B MoE | ~480 GB | **2×H200** |

Probe training and all evaluation run on **CPU only** once activations are cached.
The paper used [RunPod](https://runpod.io) H200 SXM pods. A single H200 pod covers
Gemma/Mistral/Qwen extraction; rent a 2×H100 pod for Llama; rent a 2×H200 pod for
synthetic data generation. Storage budget: ~70 GB per 27–32B model, ~140 GB for Llama 70B.
Download models to an NVMe volume attached to your cloud instance to avoid re-downloading.

### Software

- Python 3.10+
- CUDA 12.x + PyTorch 2.5+
- Ubuntu 22.04+ (or WSL2)

### Accounts

- HuggingFace account with **approved access** to:
  - [`google/gemma-3-27b-it`](https://huggingface.co/google/gemma-3-27b-it)
  - [`meta-llama/Llama-3.1-70B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)
- `HF_TOKEN` environment variable set to your token

## Phase 1 — Environment Setup

```bash
git clone https://github.com/nmcgi/2604.28129.git
cd 2604.28129

uv sync --extra vllm
source .venv/bin/activate

mkdir -p data/synthetic/train data/synthetic/eval \
         data/lmsys/train data/lmsys/eval \
         data/safedial/train data/safedial/eval \
         data/activations \
         models
```

## Phase 2 — Model Download

```bash
export HF_TOKEN="hf_..."

# Gemma 3 27B instruction-tuned  (layer ℓ=32, d=5376)
huggingface-cli download google/gemma-3-27b-it \
  --local-dir ./models/gemma-3-27b-it --token $HF_TOKEN

# Mistral Small 3.1 24B instruction-tuned  (layer ℓ=24, d=5120)
huggingface-cli download mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
  --local-dir ./models/mistral-24b-it --token $HF_TOKEN

# Qwen 2.5 32B instruction-tuned  (layer ℓ=32, d=5120)
huggingface-cli download Qwen/Qwen2.5-32B-Instruct \
  --local-dir ./models/qwen-32b-it --token $HF_TOKEN

# Llama 3.1 70B instruction-tuned  (layer ℓ=40, d=8192)
huggingface-cli download meta-llama/Llama-3.1-70B-Instruct \
  --local-dir ./models/llama-70b-it --token $HF_TOKEN

# Qwen3-235B-A22B — for synthetic dataset generation only (Phase 3)
huggingface-cli download Qwen/Qwen3-235B-A22B \
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
vllm serve Qwen/Qwen3-235B-A22B \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --port 8000
```

In a second terminal, run generation:

```bash
python generate_synthetic.py --n-train 1125 --n-eval 797
```

Output: `data/synthetic/{train,eval}/conv_{i}.json` — 1,125 training + 797 eval conversations.

Also ingest the real-world sources (requires `OPENAI_API_KEY` for three-phase labeling):

```bash
python ingest_lmsys.py    # 1,200 train + 800 eval from lmsys/lmsys-chat-1m
python ingest_safedial.py # 300 train + 200 eval from Hongyu-Cao/SafeDialBench
```

Both scripts default to `--label-mode threephase`: each user turn is labeled
benign / pivoting / adversarial by `gpt-4o-mini` using conversation context,
matching the paper's three-phase scheme (Section 3.3). Pass `--label-mode moderation`
(LMSYS) or `--label-mode all_adversarial` (SafeDialBench) for the cheaper binary
fallback, but expect higher false-positive rates at evaluation time.

> **Alternative if Qwen3-235B-A22B is unavailable**: replace the vLLM server with the
> Anthropic or OpenAI API. Swap `client` in `generate_synthetic.py` for
> `anthropic.Anthropic()` and use `claude-sonnet-4-6` or `gpt-4o`.

## Phase 4 — Activation Extraction *(1–8 h per model, GPU)*

Run for each model × source × split combination:

```bash
for MODEL in gemma mistral qwen llama; do
  for SOURCE in synthetic lmsys safedial; do
    for SPLIT in train eval; do
      python extract_activations.py --model $MODEL --source $SOURCE --split $SPLIT
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
  python train_probe.py --model $MODEL
done
```

Default (`--variant standard`): `models/<model>/xgb.json`, `models/<model>/scaler.pkl`.
With `--variant contrastive`: additionally saves `models/<model>/encoder.pt`
(and uses `xgb_contrastive.json` / `scaler_contrastive.pkl`).

## Phase 6 — Evaluation *(~1 min per model, CPU)*

```bash
python eval_probe.py --model all
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
python lad_infer.py --model qwen
```

Expected: turns 1–6 show P(adv) < 0.5; turn 9+ ("avoid triggering detection mechanisms")
exceeds θ=0.5 and prints `*** FLAGGED ***`. Latency ~100 ms/turn.

## Quick Start — Smoke Test (no H100 required)

Run the standard variant on Qwen 2.5 32B with 50 conversations. Requires only
~64 GB VRAM (fits on an A100 80 GB pod) and takes ~30 minutes end-to-end.

```bash
# 1. Generate 50 conversations via Anthropic/OpenAI API (no local Qwen3-235B needed)
#    Configure the client in generate_synthetic.py before running.
python generate_synthetic.py --n-train 40 --n-eval 10

# 2. Extract activations — Qwen 32B on A100 80 GB
python extract_activations.py --model qwen --source synthetic --split train
python extract_activations.py --model qwen --source synthetic --split eval

# 3. Train probe — standard variant (raw activations + 5 scalars, no contrastive encoder)
python train_probe.py --model qwen

# 4. Evaluate
python eval_probe.py --model qwen
```

Expected on 50-conversation smoke test: ~89% detection (the paper's scalars-only
baseline achieves 89.6%, Section 5; the standard variant matches or exceeds this),
at higher FP (~57–74%) with only 50 training conversations and no contrastive encoder.
Add `--variant contrastive` and more data to bring FP down to 2–4%.

## Low-Resource Path — GTX 1650 (4 GB VRAM)

This path runs the full pipeline on consumer hardware using small models.
Detection performance will differ from the paper; the goal is to validate that the
*approach* (trajectory scalars + probe) transfers to smaller models.

### Hardware fit

| Model key | Params | VRAM (bf16) | VRAM (4-bit) | Fits GTX 1650? |
|---|---|---|---|---|
| `qwen1.5b` | 1.5 B | ~3 GB | ~1 GB | **Yes (bf16)** |
| `llama1b`  | 1 B   | ~2 GB | ~0.7 GB | **Yes (bf16)** |
| `llama3b`  | 3 B   | ~6 GB | ~2 GB | Yes (4-bit only) |
| `gemma2b`  | 2 B   | ~4 GB | ~1.5 GB | Yes (4-bit only) |
| `phi3.5`   | 3.8 B | ~7.6 GB | ~2.5 GB | Yes (4-bit only) |

Install 4-bit quantization support once (CPU-only wheels also work):
```bash
uv sync --extra quantize
```

### Step 1 — Generate synthetic data via LM Studio

Load any instruction-tuned model in LM Studio (e.g. `Qwen 2.5 7B Q4`, `Llama 3.1 8B Q4`),
enable the local server (default port 1234), then:

```bash
python generate_synthetic.py \
  --base-url http://localhost:1234/v1 \
  --gen-model "your-model-name-as-shown-in-lmstudio" \
  --n-train 100 --n-eval 40
```

> **Note:** LM Studio is only used for *generating* the conversation dataset. Activation
> extraction still requires loading model weights locally via `transformers` — the
> probe hooks directly into the model's layers, which an API cannot expose.

### Step 2 — Extract activations (GPU, ~10–30 min)

```bash
# Fits without quantization (recommended for GTX 1650):
python extract_activations.py --model qwen1.5b --source synthetic --split train
python extract_activations.py --model qwen1.5b --source synthetic --split eval

# Larger small models need --quantize:
python extract_activations.py --model llama3b --quantize --source synthetic --split train
python extract_activations.py --model llama3b --quantize --source synthetic --split eval
```

### Step 3 — Train probe (CPU, ~2 min)

```bash
python train_probe.py --model qwen1.5b
```

### Step 4 — Evaluate and run demo

```bash
python eval_probe.py --model qwen1.5b
python lad_infer.py  --model qwen1.5b
# or with 4-bit:
python lad_infer.py  --model llama3b --quantize
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
├── data/                      — generated by pipeline; not tracked in git
│   ├── synthetic/{train,eval}/conv_*.json
│   ├── lmsys/{train,eval}/
│   ├── safedial/{train,eval}/
│   └── activations/<model>/<split>_<source>.npz
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
