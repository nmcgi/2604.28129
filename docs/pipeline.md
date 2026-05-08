# Pipeline

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

## Phase 2 — Download Models

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
