# Low-Resource Path — GTX 1650 (4 GB VRAM)

This path runs the full pipeline on consumer hardware using small models.
Detection performance will differ from the paper; the goal is to validate that the
*approach* (trajectory scalars + probe) transfers to smaller models.

## Hardware fit

| Model key | Params | VRAM (bf16) | VRAM (4-bit) | Fits GTX 1650? | Gated? |
|---|---|---|---|---|---|
| `qwen1.5b` | 1.5 B | ~3 GB | ~1 GB | **Yes (bf16)** | No |
| `llama1b`  | 1 B   | ~2 GB | ~0.7 GB | **Yes (bf16)** | **Yes** |
| `llama3b`  | 3 B   | ~6 GB | ~2 GB | Yes (4-bit only) | **Yes** |
| `gemma2b`  | 2 B   | ~4.5 GB | ~1.5 GB | Yes (4-bit only) | **Yes** |
| `phi3.5`   | 3.8 B | ~7.6 GB | ~2.5 GB | Yes (4-bit only) | No |

## Step 1 — Model Download

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

## Step 2 — Dataset

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

## Step 3 — Extract activations (GPU, ~10–30 min)

```bash
# Fits without quantization (recommended for GTX 1650):
uv run extract_activations.py --model qwen1.5b --source synthetic --split train
uv run extract_activations.py --model qwen1.5b --source synthetic --split eval

# Larger small models need --quantize:
uv run extract_activations.py --model llama3b --quantize --source synthetic --split train
uv run extract_activations.py --model llama3b --quantize --source synthetic --split eval
```

## Step 4 — Train probe (CPU, ~2 min)

```bash
uv run train_probe.py --model qwen1.5b
```

## Step 5 — Evaluate and run demo

```bash
uv run eval_probe.py --model qwen1.5b
uv run lad_infer.py  --model qwen1.5b
# or with 4-bit:
uv run lad_infer.py  --model llama3b --quantize
```
