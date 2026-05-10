"""
eval_roc_pr.py
Per-source ROC and PR analysis (Appendix H, J - Table 8, Figure 19).

Loads a trained probe, computes per-source AUROC and PR-AUC at the conversation
level, and writes ROC + PR curves to figures/<model>_<source>_roc_pr.png.

Conversation-level scoring: P(adv) for a conversation = max P(adv) over its
turns (matches the paper's flag-if-any-turn-exceeds-theta rule).

Paper Table 8 (Qwen 2.5 32B):
    Source     AUROC   PR-AUC   Conv Det / FP
    Synth      0.979   0.920    95.5 / 2.0
    LMSYS      0.907   0.467    55.3 / 5.4
    SafeDial   --      --       100  / -- (no benigns)

The LMSYS PR-AUC drop (0.38-0.44 across models) reflects a 5% adversarial
class imbalance combined with surface-similar benign tech chat.

Figure 19: synthetic-only probe (red dashed) vs expanded 3-source probe
(blue solid) - adding real-world data substantially improves both ROC
discrimination and PR precision on LMSYS.

Usage:
  python eval_roc_pr.py --model qwen1.5b
  python eval_roc_pr.py --model qwen1.5b --variant contrastive
"""
import argparse
import glob
import os
import pickle
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import xgboost as xgb
from sklearn.metrics import (roc_curve, precision_recall_curve,
                             roc_auc_score, average_precision_score)

from train_probe import MODEL_D, ContrastiveEncoder, load_npz
from eval_probe import score_conversations


def load_probe(model_dir: str, variant: str, d: int):
    clf = xgb.XGBClassifier()
    if variant == "standard":
        clf.load_model(f"{model_dir}/xgb.json")
        with open(f"{model_dir}/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return clf, scaler, None
    if variant == "scalar":
        clf.load_model(f"{model_dir}/xgb_scalar.json")
        with open(f"{model_dir}/scaler_scalar.pkl", "rb") as f:
            scaler = pickle.load(f)
        return clf, scaler, None
    enc = ContrastiveEncoder(d_in=d)
    enc.load_state_dict(torch.load(f"{model_dir}/encoder.pt", map_location="cpu"))
    enc.eval()
    clf.load_model(f"{model_dir}/xgb_contrastive.json")
    with open(f"{model_dir}/scaler_contrastive.pkl", "rb") as f:
        scaler = pickle.load(f)
    return clf, scaler, enc


def featurize(X: np.ndarray, d: int, variant: str, enc) -> np.ndarray:
    if enc is not None:
        with torch.no_grad():
            embs = enc(torch.tensor(X[:, :d], dtype=torch.float32)).numpy()
        return np.hstack([embs, X[:, d:]])
    if variant == "scalar":
        return X[:, d:]
    return X


def conv_level_scores(y, conv_ids, probs):
    """Aggregate turn-level (y, prob) into conversation-level (y_conv, p_conv).

    y_conv: 1 if any turn has y > 0, else 0.
    p_conv: max prob across all turns of the conversation (matches the
            theta-flagging rule in eval_probe.score_conversations).
    """
    unique = np.unique(conv_ids)
    y_conv, p_conv = [], []
    for cid in unique:
        mask = conv_ids == cid
        y_conv.append(int(y[mask].max() > 0))
        p_conv.append(float(probs[mask].max()))
    return np.array(y_conv), np.array(p_conv)


def evaluate_source(name: str, npz_path: str, clf, scaler, enc, d: int, variant: str,
                    fig_path: str | None):
    if not os.path.exists(npz_path):
        return None
    X, y, ids = load_npz([npz_path])
    feat  = featurize(X, d, variant, enc)
    probs = clf.predict_proba(scaler.transform(feat))[:, 1]

    y_conv, p_conv = conv_level_scores(y, ids, probs)
    if y_conv.sum() == 0 or (y_conv == 0).sum() == 0:
        # Need both classes for AUROC / PR-AUC; SafeDial has no benigns.
        det, fp, _, n_adv, n_ben = score_conversations(y, ids, probs)
        return {"source": name, "n_adv": n_adv, "n_ben": n_ben,
                "auroc": float("nan"), "pr_auc": float("nan"),
                "det": det, "fp": fp}

    auroc  = roc_auc_score(y_conv, p_conv)
    pr_auc = average_precision_score(y_conv, p_conv)
    det, fp, _, n_adv, n_ben = score_conversations(y, ids, probs)

    plotted = False
    if fig_path:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [warn] matplotlib not installed - skipping plot")
        else:
            plotted = True
            fpr, tpr, _ = roc_curve(y_conv, p_conv)
            prec, rec, _ = precision_recall_curve(y_conv, p_conv)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].plot(fpr, tpr, label=f"AUROC={auroc:.3f}")
            axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
            axes[0].set(xlabel="FPR", ylabel="TPR", title=f"ROC - {name}")
            axes[0].legend(); axes[0].grid(alpha=0.3)
            axes[1].plot(rec, prec, label=f"PR-AUC={pr_auc:.3f}")
            axes[1].set(xlabel="Recall", ylabel="Precision", title=f"PR - {name}")
            axes[1].legend(); axes[1].grid(alpha=0.3)
            fig.tight_layout()
            os.makedirs(os.path.dirname(fig_path), exist_ok=True)
            fig.savefig(fig_path, dpi=120)
            plt.close(fig)

    return {"source": name, "n_adv": n_adv, "n_ben": n_ben,
            "auroc": auroc, "pr_auc": pr_auc, "det": det, "fp": fp,
            "plotted": plotted}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   required=True, choices=list(MODEL_D))
    ap.add_argument("--variant", default="standard",
                    choices=["standard", "scalar", "contrastive"])
    ap.add_argument("--no-plots", action="store_true",
                    help="Skip matplotlib output (text metrics only)")
    args = ap.parse_args()

    d         = MODEL_D[args.model]
    act_dir   = f"data/activations/{args.model}"
    model_dir = f"probes/{args.model}"

    clf, scaler, enc = load_probe(model_dir, args.variant, d)

    print(f"\n{args.model.upper()}  (variant={args.variant})\n")
    print(f"{'Source':<14s}  {'n_adv':>6s}  {'n_ben':>6s}  "
          f"{'AUROC':>7s}  {'PR-AUC':>7s}  {'Det':>7s}  {'FP':>7s}")
    print("-" * 64)

    results = []
    for src in ["synthetic", "lmsys", "safedial"]:
        npz   = f"{act_dir}/eval_{src}.npz"
        figpath = (None if args.no_plots
                   else f"figures/{args.model}_{src}_{args.variant}_roc_pr.png")
        res = evaluate_source(src, npz, clf, scaler, enc, d, args.variant, figpath)
        if res is None:
            continue
        results.append(res)
        auroc_s  = f"{res['auroc']:.3f}" if not np.isnan(res['auroc']) else "  n/a"
        pr_s     = f"{res['pr_auc']:.3f}" if not np.isnan(res['pr_auc']) else "  n/a"
        fp_s     = f"{res['fp']:5.1f}%" if not np.isnan(res['fp']) else "  n/a"
        print(f"{res['source']:<14s}  {res['n_adv']:>6d}  {res['n_ben']:>6d}  "
              f"{auroc_s:>7s}  {pr_s:>7s}  {res['det']:6.1f}%  {fp_s:>7s}")

    if not results:
        print("No eval activations found.")
        return

    if not args.no_plots and any(r.get("plotted") for r in results):
        print(f"\nPlots written to figures/{args.model}_*_{args.variant}_roc_pr.png")
    print("\nPaper Table 8 reference (Qwen 32B): Synth 0.979 / 0.920, LMSYS 0.907 / 0.467")


if __name__ == "__main__":
    main()
