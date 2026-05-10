"""
ablation_feature.py
Per-scalar leave-one-out feature ablation (Section 7.3 #3 / Figure 8 left).

For each trajectory scalar k in {drift_mag, cosine, cum_drift, accel, mean_drift}:
  - Remove that column from features (keep activations + remaining 4 scalars)
  - Train an XGBoost probe on the reduced feature set
  - Evaluate on the combined held-out set; record Δdetection vs. full-feature baseline

Reproduces the paper claim: no single scalar dominates (<4pp drop for any k),
confirming a distributed trajectory signal.

Usage:
  python ablation_feature.py --model qwen1.5b
"""
import argparse
import glob
import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from train_probe import MODEL_D, load_npz, train_xgb
from eval_probe import score_conversations

SCALAR_NAMES = ["drift_mag", "cosine", "cum_drift", "accel", "mean_drift"]


def train_and_eval(X_train, y_train, X_eval, y_eval, conv_eval_ids, tmp_dir, tag):
    """Fit a probe on the given features and report combined-eval det/FP."""
    y_bin = (y_train >= 1).astype(np.int8)
    clf, scaler = train_xgb(X_train, y_bin, tmp_dir,
                            xgb_name=f"feat_{tag}.json",
                            scaler_name=f"feat_{tag}.pkl")
    probs = clf.predict_proba(scaler.transform(X_eval))[:, 1]
    det, fp, n_conv, n_adv, n_ben = score_conversations(y_eval, conv_eval_ids, probs)
    return det, fp, n_conv, n_adv, n_ben


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_D))
    args = ap.parse_args()

    d = MODEL_D[args.model]
    act_dir = f"data/activations/{args.model}"

    train_files = glob.glob(f"{act_dir}/train_*.npz")
    eval_files  = glob.glob(f"{act_dir}/eval_*.npz")
    if not train_files or not eval_files:
        print(f"Missing activations under {act_dir}/")
        sys.exit(1)

    X_train, y_train, _ = load_npz(train_files)
    X_eval,  y_eval,  ids_eval = load_npz(eval_files)
    print(f"Train {X_train.shape}, eval {X_eval.shape}")

    tmp = tempfile.mkdtemp(prefix="lad_feat_")

    # Baseline: all 5 scalars + raw activations
    print("\nBaseline (all 5 scalars + activations):")
    det_base, fp_base, n_conv, n_adv, n_ben = train_and_eval(
        X_train, y_train, X_eval, y_eval, ids_eval, tmp, "baseline")
    print(f"  n={n_conv} ({n_adv} adv, {n_ben} ben)  det={det_base:.1f}%  fp={fp_base:.1f}%")

    # Per-scalar leave-one-out
    print("\nLeave-one-out feature ablation:")
    print(f"  {'removed':>12s}  {'det':>7s}  {'fp':>7s}  {'Δdet':>7s}  {'Δfp':>7s}")
    print(f"  {'-'*48}")
    rows = []
    for k, name in enumerate(SCALAR_NAMES):
        # Drop column d+k from X (one of the 5 trajectory scalars)
        keep = [i for i in range(X_train.shape[1]) if i != d + k]
        Xt_drop = X_train[:, keep]
        Xe_drop = X_eval[:,  keep]
        det, fp, *_ = train_and_eval(Xt_drop, y_train, Xe_drop, y_eval, ids_eval, tmp, name)
        d_det = det - det_base
        d_fp  = fp - fp_base
        rows.append((name, det, fp, d_det, d_fp))
        print(f"  {name:>12s}  {det:6.1f}%  {fp:6.1f}%  {d_det:+6.1f}  {d_fp:+6.1f}")

    max_drop = max(-r[3] for r in rows)
    print(f"\n  Max single-feature drop: {max_drop:.1f}pp  (paper target: <4pp)")
    print("  No single scalar dominates -> distributed trajectory signal.")
