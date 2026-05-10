"""
eval_lmsys_length_stratify.py
LMSYS length stratification + cross-model error agreement (Appendix J).

Two analyses:

1) LENGTH STRATIFICATION on LMSYS eval split. Conversations are bucketed by
   user-turn count (<=10 / 11-20 / 21+); detection and FP rate reported per
   bucket. Paper finding (Qwen 32B):
       <=10:  53-58% det,  4.6-5.7% FP
       11-20: 55-66% det,  3.9-6.1% FP
       21+:   31-42% det,  0-2.9% FP
   Counterintuitively, longer LMSYS conversations are HARDER to detect because
   benign chat accumulates trajectory noise that masks adversarial drift.

2) CROSS-MODEL AGREEMENT on LMSYS errors. For each model with cached
   activations and a trained probe, compute conversation-level predictions
   on LMSYS eval. Identify:
     - convs MISSED by all models (consistently false negatives)
     - benign convs FLAGGED by all models (consistently false positives)
   Paper: 32 LMSYS adversarial convs missed by all four, 5 benign flagged by
   all four.

Usage:
  python eval_lmsys_length_stratify.py --model qwen1.5b
  python eval_lmsys_length_stratify.py --cross-model qwen1.5b llama1b gemma2b
"""
import argparse
import glob
import json
import os
import pickle
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import xgboost as xgb
from train_probe import MODEL_D, ContrastiveEncoder, load_npz
from eval_probe import score_conversations

THETA = 0.5
LENGTH_BUCKETS = [(1, 10, "<=10"), (11, 20, "11-20"), (21, 10_000, "21+")]


def load_probe(model_dir: str, variant: str, d: int):
    clf = xgb.XGBClassifier()
    if variant == "standard":
        clf.load_model(f"{model_dir}/xgb.json")
        with open(f"{model_dir}/scaler.pkl", "rb") as f:
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
    return X


def predict_conv(model: str, variant: str, source: str = "lmsys"):
    """Return (conv_ids, y_conv, p_conv, n_turns) for a model+source eval split."""
    d = MODEL_D[model]
    npz = f"data/activations/{model}/eval_{source}.npz"
    if not os.path.exists(npz):
        return None
    X, y, ids = load_npz([npz])
    clf, scaler, enc = load_probe(f"probes/{model}", variant, d)
    feat  = featurize(X, d, variant, enc)
    probs = clf.predict_proba(scaler.transform(feat))[:, 1]

    unique = np.unique(ids)
    y_conv, p_conv, n_turns = [], [], []
    for cid in unique:
        mask = ids == cid
        y_conv.append(int(y[mask].max() > 0))
        p_conv.append(float(probs[mask].max()))
        n_turns.append(int(mask.sum()))
    return (np.asarray(unique), np.asarray(y_conv),
            np.asarray(p_conv), np.asarray(n_turns))


def length_stratify(conv_ids, y_conv, p_conv, n_turns):
    print(f"{'Bucket':>8s}  {'n':>5s}  {'n_adv':>6s}  {'n_ben':>6s}  "
          f"{'Det':>7s}  {'FP':>7s}")
    print("-" * 50)
    for lo, hi, label in LENGTH_BUCKETS:
        mask = (n_turns >= lo) & (n_turns <= hi)
        if not mask.any():
            print(f"{label:>8s}  {0:>5d}")
            continue
        y_b = y_conv[mask]
        p_b = p_conv[mask]
        adv = y_b == 1
        ben = y_b == 0
        det = (p_b[adv] > THETA).mean() * 100 if adv.any() else float("nan")
        fp  = (p_b[ben] > THETA).mean() * 100 if ben.any() else float("nan")
        det_s = f"{det:6.1f}%" if not np.isnan(det) else "  n/a "
        fp_s  = f"{fp:6.1f}%"  if not np.isnan(fp)  else "  n/a "
        print(f"{label:>8s}  {mask.sum():>5d}  {int(adv.sum()):>6d}  "
              f"{int(ben.sum()):>6d}  {det_s}  {fp_s}")


def cross_model_agreement(models: list, variant: str):
    """Identify LMSYS convs that all models agree on (missed adv / flagged ben)."""
    print(f"\nCross-model agreement on LMSYS ({len(models)} models)")
    print(f"Models: {models}\n")

    per_model = {}
    for m in models:
        res = predict_conv(m, variant, "lmsys")
        if res is None:
            print(f"  [skip] {m}: no LMSYS eval activations")
            continue
        per_model[m] = res

    if len(per_model) < 2:
        print("Need >=2 models with LMSYS eval activations cached.")
        return

    # Conv IDs in our pipeline are file-index-derived; assume they line up
    # across models since each model processes the same eval JSONs in order.
    # Sanity check: union conv_ids and verify shapes.
    base_ids = next(iter(per_model.values()))[0]
    for m, (ids, _, _, _) in per_model.items():
        if not np.array_equal(ids, base_ids):
            print(f"  [warn] {m} has different conv_ids -- alignment may be off")

    # Stack predictions: convs x models
    flagged = np.stack([(per_model[m][2] > THETA).astype(int)
                        for m in per_model], axis=1)
    y_conv  = next(iter(per_model.values()))[1]

    miss_all = ((flagged == 0).all(axis=1)) & (y_conv == 1)
    fp_all   = ((flagged == 1).all(axis=1)) & (y_conv == 0)
    print(f"  Adv convs missed by ALL models:  {int(miss_all.sum())}  "
          f"(paper: 32 across 4 models)")
    print(f"  Ben convs flagged by ALL models: {int(fp_all.sum())}  "
          f"(paper: 5 across 4 models)")

    # Show first few miss IDs for inspection
    if miss_all.any():
        sample = base_ids[miss_all][:10]
        print(f"  Sample missed conv_ids: {list(sample)}")
    if fp_all.any():
        sample = base_ids[fp_all][:10]
        print(f"  Sample over-flagged conv_ids: {list(sample)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default=None, choices=list(MODEL_D),
                    help="Single model for length stratification")
    ap.add_argument("--cross-model", nargs="+", default=None,
                    choices=list(MODEL_D),
                    help="Multiple models for cross-model agreement on LMSYS")
    ap.add_argument("--variant", default="standard",
                    choices=["standard", "contrastive"])
    args = ap.parse_args()

    if args.model:
        res = predict_conv(args.model, args.variant, "lmsys")
        if res is None:
            print(f"No LMSYS eval activations for {args.model}; "
                  f"run extract_activations.py --source lmsys.")
        else:
            print(f"\nLMSYS length stratification ({args.model}, {args.variant})")
            length_stratify(*res)
            print("\nPaper Qwen 32B: <=10 53-58/4.6-5.7  11-20 55-66/3.9-6.1  "
                  "21+ 31-42/0-2.9")

    if args.cross_model:
        cross_model_agreement(args.cross_model, args.variant)

    if not args.model and not args.cross_model:
        ap.error("Pass --model and/or --cross-model")


if __name__ == "__main__":
    main()
