"""
ablation_six_feature.py
Six-feature variant comparison (Section 3.3 / Section 5.1).

The paper deliberately uses 5 trajectory scalars and reports that adding absolute
turn position t as a sixth feature *hurts* detection by 1.2pp -- t introduces a
residual length confound even when indexed absolutely.

This script reconstructs t at eval time (no re-extraction needed) by counting user
turns per conversation, then trains and evaluates two probes on combined activations:
  - 5 scalars (paper baseline)
  - 5 scalars + t (sixth-feature variant)

Reports combined-eval det/FP for each. Paper expectation: 5-feature wins by ~1.2pp.

Usage:
  python ablation_six_feature.py --model qwen1.5b
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


def add_turn_position(X, conv_ids):
    """Append turn-position column t (0-indexed user turn within each conv) to X."""
    t = np.zeros(len(X), dtype=np.float32)
    for cid in np.unique(conv_ids):
        mask = conv_ids == cid
        t[mask] = np.arange(int(mask.sum()), dtype=np.float32)
    return np.hstack([X, t.reshape(-1, 1)])


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

    X_train, y_train, ids_train = load_npz(train_files)
    X_eval,  y_eval,  ids_eval  = load_npz(eval_files)
    y_train_bin = (y_train >= 1).astype(np.int8)

    tmp = tempfile.mkdtemp(prefix="lad_6feat_")

    # --- 5-feature baseline: paper's choice ---
    print("\n5-feature baseline (paper):")
    clf5, sc5 = train_xgb(X_train, y_train_bin, tmp,
                          xgb_name="five_feat.json", scaler_name="five_feat.pkl")
    probs5 = clf5.predict_proba(sc5.transform(X_eval))[:, 1]
    det5, fp5, n_conv, n_adv, n_ben = score_conversations(y_eval, ids_eval, probs5)
    print(f"  n={n_conv} ({n_adv} adv, {n_ben} ben)  det={det5:.1f}%  fp={fp5:.1f}%")

    # --- 6-feature variant: + absolute turn position ---
    print("\n6-feature variant (+ turn position t):")
    X_train_6 = add_turn_position(X_train, ids_train)
    X_eval_6  = add_turn_position(X_eval,  ids_eval)
    clf6, sc6 = train_xgb(X_train_6, y_train_bin, tmp,
                          xgb_name="six_feat.json", scaler_name="six_feat.pkl")
    probs6 = clf6.predict_proba(sc6.transform(X_eval_6))[:, 1]
    det6, fp6, *_ = score_conversations(y_eval, ids_eval, probs6)
    print(f"  n={n_conv} ({n_adv} adv, {n_ben} ben)  det={det6:.1f}%  fp={fp6:.1f}%")

    print(f"\nDelta (5 - 6):  det={det5-det6:+.1f}pp  fp={fp5-fp6:+.1f}pp")
    print("Paper: removing t (5 features) improves detection by +1.2pp -> length confound.")
