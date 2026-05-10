"""
eval_baselines_text.py
Trained-on-our-data text and unsupervised baselines (Appendix, Table 9).

Complements eval_baselines.py (off-the-shelf safety tools) by adding the
three remaining baselines from Table 9:

  tfidf            TfidfVectorizer (5,000 features, 1-2 grams) + LogisticRegression
                   trained on user-turn text from the same training conversations
                   used for LAD. Paper: 95.1% conv det / 58.6% conv FP.
  tfidf-scalars    TF-IDF features concatenated with the 5 trajectory scalars
                   from cached activations. Mixed text + activation summary,
                   no full activations. Paper: 93.8% / 46.3%.
  cumdrift         Unsupervised threshold on cumulative drift (3rd trajectory
                   scalar). Threshold calibrated to the 95th percentile of
                   benign training turns. Paper: 99-100% det / 29-62% FP.

Comparison anchor: LAD activation probes deliver 85-89% / 2-4% with 2.1%
turn FPR. Text baselines hit similar detection but at 14-29x higher FP;
unsupervised drift threshold is even noisier.

Usage:
  python eval_baselines_text.py --baseline tfidf
  python eval_baselines_text.py --baseline tfidf-scalars --model qwen1.5b
  python eval_baselines_text.py --baseline cumdrift --model qwen1.5b
"""
import argparse
import glob
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

THETA = 0.5
EVAL_SOURCES = [
    ("synthetic", "data/synthetic/eval"),
    ("lmsys",     "data/lmsys/eval"),
    ("safedial",  "data/safedial/eval"),
]
TRAIN_SOURCES = [
    ("synthetic", "data/synthetic/train"),
    ("lmsys",     "data/lmsys/train"),
    ("safedial",  "data/safedial/train"),
]


def load_user_turns(split_dir: str):
    """Yield (conv_id, user_turn_idx, content, label_str) for user turns."""
    for path in sorted(glob.glob(f"{split_dir}/conv_*.json")):
        cid = int(os.path.basename(path).split("_")[1].split(".")[0])
        with open(path) as f:
            conv = json.load(f)
        msgs = conv.get("conversation", conv.get("messages", []))
        utx = 0
        for m in msgs:
            role  = m.get("role", m.get("speaker", ""))
            if role != "user":
                continue
            content = m.get("content", m.get("text", ""))
            label   = m.get("label", m.get("intent", "benign"))
            yield (cid, utx, content, label)
            utx += 1


def score_conv(per_turn_flags: dict, per_turn_labels: dict):
    """per_turn_flags / per_turn_labels: keyed by (source, conv_id)."""
    det_tp = det_total = fp = ben_total = 0
    for key, flags in per_turn_flags.items():
        labels = per_turn_labels[key]
        any_adv = any(l in ("pivoting", "adversarial") for l in labels)
        any_flag = any(flags)
        if any_adv:
            det_total += 1
            det_tp += int(any_flag)
        else:
            ben_total += 1
            fp += int(any_flag)
    det = det_tp / det_total * 100 if det_total else float("nan")
    fpr = fp / ben_total * 100 if ben_total else float("nan")
    return det, fpr, det_total, ben_total


def baseline_tfidf(args):
    """TF-IDF + LogisticRegression on user-turn text."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    X_text, y = [], []
    for src, tdir in TRAIN_SOURCES:
        if not os.path.isdir(tdir):
            continue
        for cid, utx, content, label in load_user_turns(tdir):
            X_text.append(content)
            y.append(int(label in ("pivoting", "adversarial")))
    if not X_text:
        sys.exit("No training conversations found")
    print(f"Training TF-IDF on {len(X_text)} user turns "
          f"({sum(y)} positive / {len(y) - sum(y)} negative)")
    if sum(y) == 0 or sum(y) == len(y):
        sys.exit("Training set is single-class; need both adv and benign turns")

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    Xv  = vec.fit_transform(X_text)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xv, y)

    print(f"\n{'Source':<14s}  {'n_adv':>6s}  {'n_ben':>6s}  {'Det':>7s}  {'FP':>7s}")
    print("-" * 50)
    for src, edir in EVAL_SOURCES:
        if not os.path.isdir(edir):
            continue
        flags_by_conv = {}
        labels_by_conv = {}
        for cid, utx, content, label in load_user_turns(edir):
            prob = clf.predict_proba(vec.transform([content]))[0, 1]
            key = (src, cid)
            flags_by_conv.setdefault(key, []).append(prob > THETA)
            labels_by_conv.setdefault(key, []).append(label)
        det, fpr, n_adv, n_ben = score_conv(flags_by_conv, labels_by_conv)
        fp_s  = f"{fpr:6.1f}%" if not np.isnan(fpr) else "  n/a "
        print(f"{src:<14s}  {n_adv:>6d}  {n_ben:>6d}  {det:6.1f}%  {fp_s}")
    print("\nPaper Table 9: TF-IDF only -> 95.1% conv det / 58.6% FP")


def baseline_tfidf_scalars(args):
    """TF-IDF features + 5 cached trajectory scalars + LogReg."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from scipy.sparse import hstack, csr_matrix
    from train_probe import MODEL_D, load_npz

    if not args.model:
        sys.exit("--model required for tfidf-scalars (needs cached scalars)")
    d = MODEL_D[args.model]
    act_dir = f"data/activations/{args.model}"

    # Build (source, conv_id, user_turn_idx) -> scalar lookup from cached npz files
    def build_scalar_index(split: str):
        idx = {}
        for path in sorted(glob.glob(f"{act_dir}/{split}_*.npz")):
            src = os.path.basename(path).removeprefix(f"{split}_").removesuffix(".npz")
            data = np.load(path)
            X    = data["X"]
            ids  = data["conv_ids"]
            cursor = {}
            for i, cid in enumerate(ids):
                cid = int(cid)
                k = cursor.get(cid, 0)
                idx[(src, cid, k)] = X[i, d:]  # last 5 columns = scalars
                cursor[cid] = k + 1
        return idx

    train_scalars = build_scalar_index("train")
    eval_scalars  = build_scalar_index("eval")
    if not train_scalars or not eval_scalars:
        sys.exit(f"Cached activations missing under {act_dir}")

    # Build training set
    X_text, scalar_rows, y = [], [], []
    for src, tdir in TRAIN_SOURCES:
        if not os.path.isdir(tdir):
            continue
        for cid, utx, content, label in load_user_turns(tdir):
            sc = train_scalars.get((src, cid, utx))
            if sc is None:
                continue  # activations may not cover this turn
            X_text.append(content)
            scalar_rows.append(sc)
            y.append(int(label in ("pivoting", "adversarial")))
    if not X_text:
        sys.exit("No training rows with both text and cached scalars")
    print(f"Training TF-IDF + scalars on {len(X_text)} aligned turns")

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    Xt  = vec.fit_transform(X_text)
    Xs  = csr_matrix(np.asarray(scalar_rows, dtype=np.float32))
    Xfeat = hstack([Xt, Xs]).tocsr()
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xfeat, y)

    print(f"\n{'Source':<14s}  {'n_adv':>6s}  {'n_ben':>6s}  {'Det':>7s}  {'FP':>7s}")
    print("-" * 50)
    for src, edir in EVAL_SOURCES:
        if not os.path.isdir(edir):
            continue
        flags_by_conv = {}
        labels_by_conv = {}
        for cid, utx, content, label in load_user_turns(edir):
            sc = eval_scalars.get((src, cid, utx))
            if sc is None:
                continue
            Xt_e = vec.transform([content])
            Xs_e = csr_matrix(sc.reshape(1, -1).astype(np.float32))
            X_e  = hstack([Xt_e, Xs_e]).tocsr()
            prob = clf.predict_proba(X_e)[0, 1]
            key = (src, cid)
            flags_by_conv.setdefault(key, []).append(prob > THETA)
            labels_by_conv.setdefault(key, []).append(label)
        det, fpr, n_adv, n_ben = score_conv(flags_by_conv, labels_by_conv)
        fp_s = f"{fpr:6.1f}%" if not np.isnan(fpr) else "  n/a "
        print(f"{src:<14s}  {n_adv:>6d}  {n_ben:>6d}  {det:6.1f}%  {fp_s}")
    print("\nPaper Table 9: TF-IDF + text scalars -> 93.8% / 46.3%")


def baseline_cumdrift(args):
    """Unsupervised: threshold on cumulative drift (3rd cached scalar)."""
    from train_probe import MODEL_D, load_npz

    if not args.model:
        sys.exit("--model required for cumdrift baseline")
    d = MODEL_D[args.model]
    act_dir = f"data/activations/{args.model}"

    train_files = sorted(glob.glob(f"{act_dir}/train_*.npz"))
    if not train_files:
        sys.exit("Need cached training activations to calibrate threshold")
    Xtr, ytr, _ = load_npz(train_files)
    benign_cd = Xtr[ytr == 0, d + 2]  # 3rd scalar = cumulative drift
    if len(benign_cd) == 0:
        sys.exit("No benign training turns to calibrate against")
    threshold = float(np.percentile(benign_cd, 95))
    print(f"Cumulative-drift threshold (95th percentile of benign training): "
          f"{threshold:.3f}\n")

    print(f"{'Source':<14s}  {'n_adv':>6s}  {'n_ben':>6s}  {'Det':>7s}  {'FP':>7s}")
    print("-" * 50)
    for src, _ in EVAL_SOURCES:
        npz = f"{act_dir}/eval_{src}.npz"
        if not os.path.exists(npz):
            continue
        Xev, yev, ids = load_npz([npz])
        cd = Xev[:, d + 2]
        unique = np.unique(ids)
        det_tp = det_total = fp = ben_total = 0
        for cid in unique:
            mask = ids == cid
            any_adv  = yev[mask].max() > 0
            any_flag = (cd[mask] > threshold).any()
            if any_adv:
                det_total += 1
                det_tp   += int(any_flag)
            else:
                ben_total += 1
                fp        += int(any_flag)
        det = det_tp / det_total * 100 if det_total else float("nan")
        fpr = fp / ben_total * 100 if ben_total else float("nan")
        fp_s = f"{fpr:6.1f}%" if not np.isnan(fpr) else "  n/a "
        print(f"{src:<14s}  {det_total:>6d}  {ben_total:>6d}  "
              f"{det:6.1f}%  {fp_s}")
    print("\nPaper Table 9: cumulative-drift threshold -> 99-100% det / 29-62% FP")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True,
                    choices=["tfidf", "tfidf-scalars", "cumdrift"])
    ap.add_argument("--model", default=None,
                    help="Required for tfidf-scalars and cumdrift (uses cached scalars)")
    args = ap.parse_args()
    {
        "tfidf":         baseline_tfidf,
        "tfidf-scalars": baseline_tfidf_scalars,
        "cumdrift":      baseline_cumdrift,
    }[args.baseline](args)


if __name__ == "__main__":
    main()
