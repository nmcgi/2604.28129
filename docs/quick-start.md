# Quick Start — Smoke Test

40 training + 10 eval synthetic conversations ship with this repo under
`data/synthetic/`. Each conversation carries three-phase turn-level labels
(benign / pivoting / adversarial) across the six attack categories from the
paper: gradual escalation, trust building, context poisoning, role accumulation,
instruction fragmentation, and tool-use exploitation. This is enough to run the
full smoke-check pipeline on a consumer GPU without any API calls or data
generation.

## Consumer GPU (≥3 GB VRAM) — no API key, no HF token required

Uses the included 50-conversation starter dataset and Qwen 2.5 1.5B (ungated,
≈3 GB VRAM in bf16). Complete [Phase 1](pipeline.md#phase-1--environment-setup) first.

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

## A100 80 GB — closer to paper conditions

Uses the included starter dataset with Qwen 2.5 32B. Same commands as above but
substitute `--model qwen` (and download `Qwen/Qwen2.5-32B-Instruct` in Phase 2).
Takes ~30 minutes end-to-end. To generate a larger dataset instead of using the
included one, see [Phase 3](pipeline.md#phase-3--synthetic-dataset-generation-2-4-h-2h200).

Expected on 50-conversation smoke test: ~89% detection rate but high FPR (~57–74%)
due to limited training data. With the full dataset (2,625 conversations) and
`--variant contrastive`, FPR drops to 2–4% while detection stays at 85–89% (paper
Sections 6–7). The scalar-only ablation (Section 5) also achieves ~89.6% detection
but retains the high 57–74% FPR regardless of dataset size.
