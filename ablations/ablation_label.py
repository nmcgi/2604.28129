"""
ablation_label.py
Label-granularity ablation (Appendix K, Table 11).

Compares two label regimes on the same training data:

  three-phase (paper, primary)
    Each user turn carries an individual label. y_bin = (y >= 1) at the turn
    level: only pivoting + adversarial turns are positives. Early benign turns
    of an adversarial conversation stay benign during training.

  binary (ablation)
    Conversation-level label propagated to every turn. Every turn in an
    adversarial conversation becomes a positive, including the early benign
    setup turns. The probe loses the pivoting-phase signal and over-fires on
    benign conversations whose surface form resembles the propagated convs.

Paper Table 11 (across 4 models):
    Labels        Det.       FP
    Three-phase   96-98%     0.5-2%
    Binary        100%       50-59%

The 100% binary detection is an over-fit: the classifier learns to fire on
any conversation whose first few turns look like the early turns of training-
set adversarial conversations -- which is most benign technical chat.

Usage:
  python ablation_label.py --model qwen1.5b
"""
import argparse
import glob
import os
import pickle
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import xgboost as xgb
from train_probe import MODEL_D, load_npz, train_xgb
from eval_probe import score_conversations


def synthesize_binary_labels(y_3class: np.ndarray, conv_ids: np.ndarray) -> np.ndarray:
    """Propagate conversation-level adv/ben label to every turn.

    For each conv_id, if any turn has y > 0 (pivoting OR adversarial) the
    entire conversation is treated as positive; every turn gets y_bin = 1.
    Otherwise every turn gets y_bin = 0. This is the binary-label regime
    against which the paper compares the three-phase scheme.
    """
    y_bin = np.zeros_like(y_3class, dtype=np.int8)
    for cid in np.unique(conv_ids):
        mask = conv_ids == cid
        if y_3class[mask].max() > 0:
            y_bin[mask] = 1
    return y_bin


def train_and_eval(X_train, y_train, X_eval, y_eval, ids_eval):
    """Train standard XGBoost on (X, y), eval on (X_eval, y_eval). Returns (det, fp)."""
    with tempfile.TemporaryDirectory(prefix="lblab_") as tmp:
        train_xgb(X_train, y_train, tmp)
        clf = xgb.XGBClassifier()
        clf.load_model(f"{tmp}/xgb.json")
        with open(f"{tmp}/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    probs = clf.predict_proba(scaler.transform(X_eval))[:, 1]
    det, fp, _, _, _ = score_conversations(y_eval, ids_eval, probs)
    return det, fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_D))
    args = ap.parse_args()

    act_dir = f"data/activations/{args.model}"
    train_files = sorted(glob.glob(f"{act_dir}/train_*.npz"))
    eval_files  = sorted(glob.glob(f"{act_dir}/eval_*.npz"))
    if not train_files or not eval_files:
        sys.exit(f"Need cached activations in {act_dir} (run extract_activations.py first)")

    X_train, y_train, ids_train = load_npz(train_files)
    X_eval,  y_eval,  ids_eval  = load_npz(eval_files)

    print(f"Train: {X_train.shape}, Eval: {X_eval.shape}")
    print(f"Train turn labels: {(y_train==0).sum()} benign, "
          f"{(y_train==1).sum()} pivoting, {(y_train==2).sum()} adversarial")

    # --- Three-phase: turn-level binarization (paper primary) ---
    y_threephase = (y_train >= 1).astype(np.int8)
    print(f"\n[three-phase]  positives = {y_threephase.sum()} / {len(y_threephase)} turns")
    det_tp, fp_tp = train_and_eval(X_train, y_threephase, X_eval, y_eval, ids_eval)

    # --- Binary: conversation-level label propagated to all turns ---
    y_binary = synthesize_binary_labels(y_train, ids_train)
    print(f"[binary]       positives = {y_binary.sum()} / {len(y_binary)} turns")
    det_bn, fp_bn = train_and_eval(X_train, y_binary, X_eval, y_eval, ids_eval)

    print()
    print(f"{'Labels':<14s}  {'Det':>7s}  {'FP':>7s}")
    print("-" * 32)
    print(f"{'three-phase':<14s}  {det_tp:6.1f}%  {fp_tp:6.1f}%   (paper: 96-98% / 0.5-2%)")
    print(f"{'binary':<14s}  {det_bn:6.1f}%  {fp_bn:6.1f}%   (paper: 100%   / 50-59%)")

    if not np.isnan(fp_bn) and not np.isnan(fp_tp):
        print(f"\nFP delta: binary - three-phase = +{fp_bn - fp_tp:.1f}pp")


if __name__ == "__main__":
    main()
