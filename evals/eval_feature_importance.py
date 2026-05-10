"""
eval_feature_importance.py
Top-K XGBoost feature importance per model (Appendix I, Figure 20).

Reads `feature_importances_` (gain-based) from a trained standard XGBoost
probe, names the 5 trajectory scalars explicitly, and prints + plots the
top-K features. Confirms the paper's claim that trajectory scalars
consistently rank among the top features while specific activation
dimensions are model-specific.

The 5 trajectory scalars (last 5 columns of the feature vector,
matching extract_activations.py:90-97 and Algorithm 3) are:
    drift_mag    = ||v_t - v_{t-1}||_2
    cos_sim      = cos(v_t, v_{t-1})
    cum_drift    = sum_{i<=t} ||delta_i||
    drift_accel  = ||delta_t|| - ||delta_{t-1}||
    mean_drift   = cum_drift / (t-1)

For the contrastive variant, the first 128 columns are encoder embeddings
(unnamed) and the last 5 are the same scalars.

Usage:
  python eval_feature_importance.py --model qwen1.5b
  python eval_feature_importance.py --model qwen1.5b --variant contrastive
  python eval_feature_importance.py --model qwen1.5b --top 20
"""
import argparse
import os
import pickle
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import xgboost as xgb
from train_probe import MODEL_D

SCALAR_NAMES = ["drift_mag", "cos_sim", "cum_drift", "drift_accel", "mean_drift"]


def feature_name(idx: int, d: int, n_total: int, variant: str) -> str:
    """Map a flat feature index to a human-readable name.

    standard:    [act_0 .. act_{d-1}, drift_mag, cos_sim, cum_drift,
                  drift_accel, mean_drift]
    scalar:      [drift_mag, cos_sim, cum_drift, drift_accel, mean_drift]
    contrastive: [emb_0 .. emb_127, drift_mag, cos_sim, cum_drift,
                  drift_accel, mean_drift]
    """
    if variant == "scalar":
        return SCALAR_NAMES[idx]
    n_pre = n_total - 5
    if idx >= n_pre:
        return SCALAR_NAMES[idx - n_pre]
    if variant == "contrastive":
        return f"emb_{idx}"
    return f"act_{idx}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   required=True, choices=list(MODEL_D))
    ap.add_argument("--variant", default="standard",
                    choices=["standard", "scalar", "contrastive"])
    ap.add_argument("--top",     type=int, default=10)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    d         = MODEL_D[args.model]
    model_dir = f"probes/{args.model}"

    xgb_path = {
        "standard":    "xgb.json",
        "scalar":      "xgb_scalar.json",
        "contrastive": "xgb_contrastive.json",
    }[args.variant]
    if not os.path.exists(f"{model_dir}/{xgb_path}"):
        sys.exit(f"Missing {model_dir}/{xgb_path} - run train_probe.py first")

    clf = xgb.XGBClassifier()
    clf.load_model(f"{model_dir}/{xgb_path}")
    importances = clf.feature_importances_  # gain-based, sums to 1.0
    n_total = len(importances)

    expected = {"standard": d + 5, "scalar": 5, "contrastive": 128 + 5}[args.variant]
    if n_total != expected:
        print(f"  [warn] feature count {n_total} != expected {expected} for variant {args.variant}")

    sorted_idx = np.argsort(-importances)
    top = sorted_idx[: args.top]

    print(f"\n{args.model.upper()} ({args.variant}) - top-{args.top} XGBoost features by gain")
    print(f"{'Rank':>4s}  {'Feature':<14s}  {'Importance':>10s}")
    print("-" * 36)
    for rank, idx in enumerate(top, 1):
        name = feature_name(int(idx), d, n_total, args.variant)
        print(f"{rank:>4d}  {name:<14s}  {importances[idx] * 100:9.2f}%")

    # Aggregate share captured by the 5 trajectory scalars
    n_pre = n_total - 5
    scalar_share = float(importances[n_pre:].sum()) * 100
    other_share  = float(importances[:n_pre].sum()) * 100
    print(f"\nTrajectory scalars total: {scalar_share:.2f}%   "
          f"(remaining {other_share:.2f}% across {n_pre} other features)")
    print("Paper Figure 20: trajectory scalars consistently dominate the top features.")

    if args.no_plots:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [warn] matplotlib not installed - skipping plot")
        return

    names = [feature_name(int(i), d, n_total, args.variant) for i in top]
    vals  = [importances[i] * 100 for i in top]
    colors = ["tab:red" if n in SCALAR_NAMES else "tab:blue" for n in names]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(range(len(top)), vals[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("Importance (%)")
    ax.set_title(f"Top-{args.top} features ({args.model}, {args.variant})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = f"figures/{args.model}_{args.variant}_feature_importance.png"
    os.makedirs("figures", exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nPlot written to {out}")


if __name__ == "__main__":
    main()
