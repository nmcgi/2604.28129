"""
ablation_loso.py
Leave-one-source-out evaluation (Section 7.3 ablation #1).

For each source S in {synthetic, lmsys, safedial}:
  - Train a probe on the remaining two sources (filter out train_<S>.npz)
  - Evaluate per-source on the combined held-out set

Reproduces the paper's claim that each training source is non-redundant: removing
any source causes catastrophic failure on the held-out source from that distribution
(e.g. 0% SafeDialBench detection without SafeDialBench training, 100% FP without LMSYS).

Usage:
  python ablation_loso.py --model qwen1.5b
"""
import argparse
import glob
import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import xgboost as xgb

from train_probe import MODEL_D, load_npz, train_xgb
from eval_probe import score_conversations

SOURCES = ["synthetic", "lmsys", "safedial"]


def per_source_scores(model_name, clf, scaler):
    """Return dict {source: (det, fp, n_adv, n_ben)}."""
    act_dir = f"data/activations/{model_name}"
    out = {}
    for src in SOURCES:
        files = glob.glob(f"{act_dir}/eval_{src}.npz")
        if not files:
            continue
        X, y, ids = load_npz(files)
        probs = clf.predict_proba(scaler.transform(X))[:, 1]
        det, fp, _, n_adv, n_ben = score_conversations(y, ids, probs)
        out[src] = (det, fp, n_adv, n_ben)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_D))
    args = ap.parse_args()

    act_dir = f"data/activations/{args.model}"

    # --- Baseline: all sources in training ---
    train_files = glob.glob(f"{act_dir}/train_*.npz")
    if not train_files:
        print(f"No training activations under {act_dir}/")
        sys.exit(1)
    print(f"Baseline (all sources):  training on {[os.path.basename(f) for f in train_files]}")
    X_tr, y_tr, _ = load_npz(train_files)
    y_bin = (y_tr >= 1).astype(np.int8)
    tmp = tempfile.mkdtemp(prefix="lad_loso_")
    clf, scaler = train_xgb(X_tr, y_bin, tmp, xgb_name="loso_baseline.json",
                            scaler_name="loso_baseline.pkl")
    baseline = per_source_scores(args.model, clf, scaler)

    print(f"\n  {'eval source':>12s}  {'det':>7s}  {'fp':>7s}  {'n_adv':>6s}  {'n_ben':>6s}")
    print(f"  {'-'*46}")
    for src, (det, fp, n_adv, n_ben) in baseline.items():
        print(f"  {src:>12s}  {det:6.1f}%  {fp:6.1f}%  {n_adv:6d}  {n_ben:6d}")

    # --- Leave-one-out runs ---
    results = {}  # dropped -> per_source_scores
    for dropped in SOURCES:
        kept = [f for f in train_files if f"train_{dropped}.npz" not in f]
        if len(kept) == len(train_files):
            print(f"\n[skip] no train_{dropped}.npz to drop")
            continue
        if not kept:
            print(f"\n[skip] dropping {dropped} would leave nothing to train on")
            continue
        print(f"\nDrop '{dropped}':  training on {[os.path.basename(f) for f in kept]}")
        X_tr, y_tr, _ = load_npz(kept)
        y_bin = (y_tr >= 1).astype(np.int8)
        clf, scaler = train_xgb(X_tr, y_bin, tmp,
                                xgb_name=f"loso_drop_{dropped}.json",
                                scaler_name=f"loso_drop_{dropped}.pkl")
        results[dropped] = per_source_scores(args.model, clf, scaler)
        print(f"  {'eval source':>12s}  {'det':>7s}  {'fp':>7s}  {'n_adv':>6s}  {'n_ben':>6s}")
        print(f"  {'-'*46}")
        for src, (det, fp, n_adv, n_ben) in results[dropped].items():
            marker = "  <-- held out from training" if src == dropped else ""
            print(f"  {src:>12s}  {det:6.1f}%  {fp:6.1f}%  {n_adv:6d}  {n_ben:6d}{marker}")

    # --- Summary table ---
    print("\n=== Summary: detection rate per (dropped, eval) source ===")
    print(f"  {'dropped':>10s} | " + " ".join(f"{s:>10s}" for s in SOURCES))
    print(f"  {'-'*10}-+-" + "-".join(["-"*10]*len(SOURCES)))
    for dropped, scores in results.items():
        row = f"  {dropped:>10s} | "
        for src in SOURCES:
            if src in scores:
                det = scores[src][0]
                row += f"{det:9.1f}% "
            else:
                row += f"{'n/a':>10s} "
        print(row)
    print("\nPaper: dropping any source causes catastrophic failure on the held-out source.")
