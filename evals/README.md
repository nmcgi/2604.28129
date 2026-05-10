# Evals

Measurement and comparison experiments around the trained LAD probe.
Where `ablations/` answers *which components matter*, `evals/` answers
*how well does it work, against what, and where does it fail*.

All scripts assume cached activations under `data/activations/<model>/`
and a trained probe under `probes/<model>/`. Run from the repo root.

## Scripts

| Script | What it measures | Paper |
|---|---|---|
| `eval_baselines.py`                  | LAD vs. off-the-shelf safety tools (Prompt-Guard, LLM Guard, Lakera Guard) — turn/conv detection, FP, phase selectivity, McNemar tests | §7.3 #6 / Fig 9 / Table 9 |
| `eval_baselines_text.py`             | LAD vs. trained-on-our-data text baselines — TF-IDF + LogReg, TF-IDF + scalars, unsupervised cumulative-drift threshold | Appendix, Table 9 |
| `eval_cross_model_transfer.py`       | Off-diagonal F1 matrix: train scalar-only probe on model A, score on model B — confirms probes are model-specific | §6 / Appendix F |
| `eval_roc_pr.py`                     | Per-source AUROC and PR-AUC at conversation level (synth / LMSYS / SafeDial); writes ROC+PR curves to `figures/` | Appendix H, J / Fig 19 |
| `eval_feature_importance.py`         | Top-K XGBoost gain importance with the 5 trajectory scalars named explicitly; prints + plots | Appendix I / Fig 20 |
| `eval_early_detection_per_category.py` | Stratifies early-detection rate, mean lead time, and overall detection by attack category (gradual escalation, role accumulation, …) on the synthetic eval split | Appendix L, Table 13 |
| `eval_lmsys_length_stratify.py`      | LMSYS detection/FP bucketed by user-turn count (≤10 / 11–20 / 21+); also identifies LMSYS errors common to all models | Appendix J |
| `eval_label_validation.py`           | Frontier-LLM-as-judge agreement check on three-phase labels — pairwise + Fleiss' κ across Anthropic / OpenAI / Google judges | Appendix D.2 |

## Usage

Each script takes `--model <key>` (e.g. `qwen1.5b`, `qwen`, `gemma`,
`llama`, `mistral`); most also accept `--variant {standard,contrastive}`.

```bash
python evals/eval_baselines.py --model qwen1.5b
python evals/eval_roc_pr.py --model qwen1.5b
python evals/eval_lmsys_length_stratify.py --cross-model qwen1.5b llama1b gemma2b
```

`eval_cross_model_transfer.py` iterates over every model in `MODEL_D` and
takes no `--model` flag. `eval_label_validation.py` is the only script
here that hits external APIs — it needs `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, and `GOOGLE_API_KEY`, plus `uv sync --extra label-val`.
The rest run on CPU from cached artifacts.
