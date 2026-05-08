"""
eval_cross_model_transfer.py
Scalar-only cross-model transfer test (Section 6 / Appendix F).

Trains a 5-feature (trajectory scalars only) XGBoost probe on each model's
activations and evaluates it on every other model's combined eval set.
Reports an off-diagonal F1 matrix to confirm probes are model-specific.

Usage:
  python eval_cross_model_transfer.py
"""
import glob
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from train_probe import MODEL_D, load_npz

MODELS = list(MODEL_D)
THETA  = 0.5


def load_scalars_only(act_dir, split):
    """Load only the 5 trajectory scalar features (last 5 columns of X)."""
    files = glob.glob(f"{act_dir}/{split}_*.npz")
    if not files:
        return None, None, None
    X, y, ids = load_npz(files)
    return X[:, -5:], y, ids   # drop raw activations, keep scalars


def train_scalar_probe(X_scalars, y):
    scaler = StandardScaler().fit(X_scalars)
    X_norm = scaler.transform(X_scalars)
    pos = int(y.sum())
    neg = len(y) - pos
    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg / pos,
        eval_metric="logloss", random_state=42,
        tree_method="hist",
    )
    clf.fit(X_norm, y)
    return clf, scaler


if __name__ == "__main__":
    # Build per-model scalar probes
    probes = {}
    for m in MODELS:
        act_dir = f"data/activations/{m}"
        X_tr, y_tr, _ = load_scalars_only(act_dir, "train")
        if X_tr is None:
            print(f"[{m}] no training activations found — skipping")
            continue
        print(f"Training scalar probe for {m}  ({X_tr.shape[0]} turns)…")
        clf, scaler = train_scalar_probe(X_tr, y_tr)
        probes[m] = (clf, scaler)

    # Evaluate every (train_model, eval_model) combination
    trained_models = list(probes)
    print(f"\nF1 matrix (rows=trained on, cols=evaluated on), θ={THETA}")
    header = f"{'':12s}" + "".join(f"{m:10s}" for m in trained_models)
    print(header)
    print("-" * len(header))

    for train_m in trained_models:
        clf, scaler = probes[train_m]
        row = f"{train_m:12s}"
        for eval_m in trained_models:
            act_dir = f"data/activations/{eval_m}"
            X_ev, y_ev, _ = load_scalars_only(act_dir, "eval")
            if X_ev is None:
                row += f"{'n/a':10s}"
                continue
            probs = clf.predict_proba(scaler.transform(X_ev))[:, 1]
            preds = (probs > THETA).astype(int)
            y_ev_bin = (y_ev >= 1).astype(int)  # binarize 3-class labels
            f1 = f1_score(y_ev_bin, preds, zero_division=0)
            marker = " *" if train_m == eval_m else "  "
            row += f"{f1*100:7.1f}%{marker} "
        print(row)

    print("\n* = on-diagonal (same model). Off-diagonal F1 should average ~50% (near random).")
