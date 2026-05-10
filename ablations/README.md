# Ablations

Component-removal experiments that probe *why* LAD works. Each script holds
the rest of the pipeline fixed and varies one knob — features, labels,
sources, layers, attacker strength — to isolate its contribution.

All scripts assume cached activations under `data/activations/<model>/` and
(except for layer/SAE/robustness) train a fresh probe on a subset of the
features. Run from the repo root so the `train_probe` / `eval_probe`
imports resolve.

## Scripts

| Script | What it measures | Paper |
|---|---|---|
| `ablation_six_feature.py`        | 5 trajectory scalars vs. 5 + absolute turn position `t`; expects 5-feature wins by ~1.2pp because `t` reintroduces a length confound | §3.3 / §5.1 |
| `ablation_feature.py`            | Per-scalar leave-one-out across `{drift_mag, cosine, cum_drift, accel, mean_drift}` to show no single scalar dominates (<4pp drop each) | §7.3 #3 / Fig 8 |
| `ablation_loso.py`               | Leave-one-source-out: drop synthetic / LMSYS / SafeDial in turn, evaluate on each held-out source to confirm sources are non-redundant | Appendix K, Table 10 |
| `ablation_label.py`              | Three-phase per-turn labels vs. binary conversation-level labels; binary hits 100% detection but 50–59% FP (over-fits early benign turns) | Appendix K, Table 11 |
| `ablation_layer_sensitivity.py`  | Re-extracts activations from every Nth decoder layer and reports 5-fold CV — confirms the signal is not layer-specific (<1.2pp spread) | Appendix G |
| `ablation_adversarial_robustness.py` | Probe-aware attacker that interpolates each turn toward the previous one (`v' = (1-α)v_t + α v_{t-1}`); sweeps α and reports the break point under three attacker models | Appendix M |
| `ablation_sae.py`                | Zeros top-K / random-K / bottom-K GemmaScope-2 SAE latents in each activation and re-evaluates; expects ≤0.4pp degradation, showing trajectory scalars carry the signal | Appendix O |

## Usage

Each script takes `--model <key>` matching `MODEL_D` in `train_probe.py`
(e.g. `qwen1.5b`, `qwen`, `gemma`, `llama`, `mistral`). Most also accept
`--variant {standard,contrastive}` to switch between the raw-activation
XGBoost probe and the contrastive-encoder variant.

```bash
python ablations/ablation_feature.py --model qwen1.5b
python ablations/ablation_loso.py --model qwen1.5b
python ablations/ablation_layer_sensitivity.py --model gemma --step 4
```

`ablation_layer_sensitivity.py` and `ablation_sae.py` need a GPU and the
target model's weights under `models/`. The other five run on CPU from
cached activations.
