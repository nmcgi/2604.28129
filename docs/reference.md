# Reference

## Quick Reference

| Script | Paper section | Runs on | Est. time |
|---|---|---|---|
| `generate_synthetic.py` | §3.1, Dataset | 2×H200 (vLLM) | 2–4 h |
| `extract_activations.py` | §3.2 | 1–2×H100/H200 | 1–8 h/model |
| `train_probe.py` | §3.3–3.4 | CPU | ~12 min/model |
| `eval_probe.py` | §5–7 | CPU | ~1 min/model |
| `lad_infer.py` | §8 | GPU (target model) + CPU | ~100 ms/turn |
| `ablations/ablation_adversarial_robustness.py` | Appendix M | CPU | — |
| `ablations/ablation_feature.py` | §7.3 #3 / Fig 8 left | CPU | — |
| `ablations/ablation_label.py` | Appendix K, Table 11 | CPU | — |
| `ablations/ablation_layer_sensitivity.py` | Appendix G | GPU | — |
| `ablations/ablation_loso.py` | Appendix K, Table 10 | CPU | — |
| `ablations/ablation_sae.py` | Appendix O | GPU | — |
| `ablations/ablation_six_feature.py` | §3.3 / §5.1 | CPU | — |
| `evals/eval_baselines.py` | Table 9 (off-the-shelf) | CPU | — |
| `evals/eval_baselines_text.py` | Table 9 (TF-IDF, cumdrift) | CPU | — |
| `evals/eval_cross_model_transfer.py` | Appendix F | CPU | — |
| `evals/eval_early_detection_per_category.py` | Appendix L, Table 13 | CPU | — |
| `evals/eval_feature_importance.py` | Appendix I, Fig 20 | CPU | — |
| `evals/eval_label_validation.py` | Appendix D.2 | API | — |
| `evals/eval_lmsys_length_stratify.py` | Appendix J | CPU | — |
| `evals/eval_roc_pr.py` | Appendix H, J / Fig 19 | CPU | — |
| `deploy/monitor.py` | Appendix N.3 | CPU (daemon) | — |
| `deploy/review.py` | Appendix N.1 step 3 | CPU | — |
| `deploy/adapt.py` | Appendix N.1 step 4 / N.2 | CPU | — |

Total GPU-hours for full 4-model reproduction: ~3 h generation + ~20 h extraction.
Total CPU-hours: ~1 h training + evaluation.

## Repository Structure

```
.
├── README.md                  — project overview and links to docs/
├── VOCAB.md                   — plain-English glossary of all jargon terms
├── pyproject.toml             — dependency spec (uv)
├── docs/
│   ├── prerequisites.md       — hardware, software, and account requirements
│   ├── quick-start.md         — smoke-test instructions (consumer GPU + A100)
│   ├── pipeline.md            — full pipeline, Phases 1–7
│   ├── pipeline-small.md      — GTX 1650 / small-model path
│   └── reference.md           — this file: script table + repo structure
├── generate_synthetic.py      — synthetic dataset generation (Phase 3)
├── ingest_lmsys.py            — LMSYS-Chat-1M ingestion with 3-phase labeling
├── ingest_safedial.py         — SafeDialBench ingestion with 3-phase labeling
├── extract_activations.py     — activation extraction + trajectory scalars (Phase 4)
├── train_probe.py             — contrastive encoder + XGBoost probe (Phase 5)
├── eval_probe.py              — held-out evaluation + early detection (Phase 6)
├── lad_infer.py               — real-time inference demo (Phase 7)
├── ablations/                 — component-removal experiments
│   ├── ablation_adversarial_robustness.py  — α-sweep drift suppression (Appendix M)
│   ├── ablation_feature.py                 — per-scalar leave-one-out (§7.3 #3 / Fig 8)
│   ├── ablation_label.py                   — three-phase vs binary labels (Table 11)
│   ├── ablation_layer_sensitivity.py       — extraction-layer sweep (Appendix G)
│   ├── ablation_loso.py                    — leave-one-source-out (Table 10)
│   ├── ablation_sae.py                     — GemmaScope SAE ablation (Appendix O)
│   └── ablation_six_feature.py             — 5- vs 6-feature variant (§3.3 / §5.1)
├── evals/                     — measurement / comparison experiments
│   ├── eval_baselines.py                   — PromptGuard / LLM Guard / Lakera (Table 9)
│   ├── eval_baselines_text.py              — TF-IDF / cumdrift threshold (Table 9)
│   ├── eval_cross_model_transfer.py        — cross-model F1 matrix (Appendix F)
│   ├── eval_early_detection_per_category.py — per-category early detection (Table 13)
│   ├── eval_feature_importance.py          — XGBoost top-K features (Appendix I, Fig 20)
│   ├── eval_label_validation.py            — LLM-as-judge label validation (Appendix D.2)
│   ├── eval_lmsys_length_stratify.py       — length buckets + cross-model agreement (Appendix J)
│   └── eval_roc_pr.py                      — per-source AUROC / PR-AUC (Fig 19)
├── deploy/                    — production adaptation loop (Appendix N)
│   ├── README.md                           — Mermaid diagram of the loop
│   ├── monitor.py                          — sliding-window FP trigger (N.3)
│   ├── review.py                           — second-opinion + human review (N.1)
│   └── adapt.py                            — atomic retrain + probe swap (N.2)
├── data/synthetic/            — starter dataset (40 train + 10 eval); tracked in git
│   ├── train/conv_*.json
│   └── eval/conv_*.json
├── data/lmsys/                — not tracked; ingested by ingest_lmsys.py
├── data/safedial/             — not tracked; ingested by ingest_safedial.py
├── data/activations/          — not tracked; generated by extract_activations.py
│   └── <model>/<split>_<source>.npz
├── data/labeled/              — not tracked; populated by deploy/review.py
│   ├── auto/                  — second-opinion-agreed conversations
│   └── review/                — human-review queue
├── logs/                      — not tracked; lad_infer.py --log destination
│   ├── preds.jsonl            — per-turn predictions
│   └── labels.jsonl           — review.py decisions
├── models/                    — downloaded LLM weights (not tracked in git)
│   └── <model_name>/          — e.g. gemma-3-27b-it, llama-70b-it
└── probes/                    — trained probe artifacts (not tracked in git)
    └── <model_key>/           — e.g. gemma, llama, qwen1.5b
        ├── xgb.json           — standard variant
        ├── scaler.pkl
        ├── encoder.pt         — contrastive variant only
        ├── xgb_contrastive.json
        └── scaler_contrastive.pkl
```
