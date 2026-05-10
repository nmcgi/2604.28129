"""
ablation_sae.py
GemmaScope 2 SAE feature ablation (Appendix O).

Loads GemmaScope 2 (layer 32, 65k width) SAE, identifies the top-K SAE latents
by XGBoost gain importance, zeros them out in each activation vector, then
re-evaluates the probe. Also evaluates random-K and bottom-K ablation as controls.

Reproduces Figure 23: top-1000 SAE ablation degrades accuracy by ≤0.4 pp,
confirming detection is driven by trajectory scalars rather than SAE content features.

Requires:
  huggingface-hub (already a main dependency — no extras needed)
  GemmaScope 2 SAE checkpoint: google/gemma-scope-27b-pt-res (layer 32, width 65k)

Usage:
  python ablation_sae.py [--k 10 50 100 200 500 1000]
"""
import argparse
import glob
import os
import pickle
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import xgboost as xgb

from train_probe import MODEL_D, load_npz, ContrastiveEncoder

SAE_REPO   = "google/gemma-scope-27b-pt-res"
SAE_LAYER  = 32  # match activation extraction layer for Gemma (layer 31 is one block earlier)
SAE_WIDTH  = 65536
MODEL_NAME = "gemma"


def load_sae():
    """Load GemmaScope 2 SAE weights (encoder matrix W_enc, bias b_enc)."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(SAE_REPO, filename=f"layer_{SAE_LAYER}/width_{SAE_WIDTH//1000}k/average_l0_130/params.npz")
    data = np.load(path)
    W_enc = torch.tensor(data["W_enc"], dtype=torch.float32)   # (d, width)
    b_enc = torch.tensor(data["b_enc"], dtype=torch.float32)   # (width,)
    W_dec = torch.tensor(data["W_dec"], dtype=torch.float32)   # (width, d)
    b_dec = torch.tensor(data["b_dec"], dtype=torch.float32)   # (d,)
    return W_enc, b_enc, W_dec, b_dec


def sae_encode(acts, W_enc, b_enc):
    """Return SAE latent activations (N, width) for input acts (N, d)."""
    x = torch.tensor(acts, dtype=torch.float32)
    return torch.relu(x @ W_enc + b_enc).numpy()


def sae_ablate(acts, latent_acts, ablate_indices, W_dec, b_dec):
    """Zero out selected SAE latents and reconstruct; return ablated activations."""
    lat = torch.tensor(latent_acts, dtype=torch.float32).clone()
    lat[:, ablate_indices] = 0.0
    x = torch.tensor(acts, dtype=torch.float32)
    recon = lat @ torch.tensor(W_dec, dtype=torch.float32) + torch.tensor(b_dec, dtype=torch.float32)
    # Add residual (original - full reconstruction) back
    full_recon = torch.tensor(latent_acts, dtype=torch.float32) @ torch.tensor(W_dec, dtype=torch.float32) + torch.tensor(b_dec, dtype=torch.float32)
    ablated = x - full_recon + recon
    return ablated.numpy()


def eval_accuracy(X_feat, y, clf, scaler):
    probs = clf.predict_proba(scaler.transform(X_feat))[:, 1]
    y_bin = (y >= 1).astype(int)  # binarize 3-class labels
    return ((probs > 0.5).astype(int) == y_bin).mean() * 100


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[10, 20, 50, 100, 200, 500, 1000],
                    help="Number of SAE features to ablate")
    ap.add_argument("--variant", default="standard", choices=["standard", "contrastive"])
    args = ap.parse_args()

    d         = MODEL_D[MODEL_NAME]
    act_dir   = f"data/activations/{MODEL_NAME}"
    model_dir = f"probes/{MODEL_NAME}"

    eval_files = glob.glob(f"{act_dir}/eval_*.npz")
    X_eval, y_eval, conv_ids = load_npz(eval_files)
    acts  = X_eval[:, :d]
    scalars = X_eval[:, d:]

    clf = xgb.XGBClassifier()
    if args.variant == "standard":
        clf.load_model(f"{model_dir}/xgb.json")
        with open(f"{model_dir}/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        baseline_feat = X_eval
    else:
        enc = ContrastiveEncoder(d_in=d)
        enc.load_state_dict(torch.load(f"{model_dir}/encoder.pt", map_location="cpu"))
        enc.eval()
        clf.load_model(f"{model_dir}/xgb_contrastive.json")
        with open(f"{model_dir}/scaler_contrastive.pkl", "rb") as f:
            scaler = pickle.load(f)
        with torch.no_grad():
            embs = enc(torch.tensor(acts, dtype=torch.float32)).numpy()
        baseline_feat = np.hstack([embs, scalars])

    baseline_acc = eval_accuracy(baseline_feat, y_eval, clf, scaler)
    print(f"Baseline accuracy: {baseline_acc:.1f}%")

    print("Loading GemmaScope 2 SAE…")
    W_enc, b_enc, W_dec, b_dec = load_sae()

    print("Computing SAE latent activations…")
    lat_acts = sae_encode(acts, W_enc, b_enc)   # (N, 65536)

    # Rank SAE latents by XGBoost gain importance: train a probe on SAE latents
    print("Ranking SAE latents by XGBoost gain importance…")
    from sklearn.preprocessing import StandardScaler as _SS
    rng = np.random.default_rng(42)
    y_bin_all = (y_eval >= 1).astype(int)
    n_sub = min(len(lat_acts), 10_000)
    sub_idx = rng.choice(len(lat_acts), n_sub, replace=False)
    lat_scaler = _SS().fit(lat_acts[sub_idx])
    lat_norm   = lat_scaler.transform(lat_acts[sub_idx])
    sae_clf    = xgb.XGBClassifier(n_estimators=100, max_depth=4,
                                    eval_metric="logloss", random_state=42,
                                    tree_method="hist")
    sae_clf.fit(lat_norm, y_bin_all[sub_idx])
    importance    = sae_clf.feature_importances_   # gain-based, shape (65536,)
    sorted_indices = np.argsort(-importance)

    print(f"\n{'K':>6s}  {'Top-K':>8s}  {'Random-K':>10s}  {'Bottom-K':>10s}")
    print("-" * 42)

    for K in args.k:
        results = {}
        for mode in ["top", "random", "bottom"]:
            if mode == "top":
                idx = sorted_indices[:K]
            elif mode == "bottom":
                idx = sorted_indices[-K:]
            else:
                idx = rng.choice(SAE_WIDTH, K, replace=False)

            ablated_acts = sae_ablate(acts, lat_acts, idx, W_dec, b_dec)

            if args.variant == "standard":
                feat = np.hstack([ablated_acts, scalars])
            else:
                with torch.no_grad():
                    embs_abl = enc(torch.tensor(ablated_acts, dtype=torch.float32)).numpy()
                feat = np.hstack([embs_abl, scalars])

            acc = eval_accuracy(feat, y_eval, clf, scaler)
            results[mode] = acc

        print(f"{K:6d}  {results['top']:7.1f}%  {results['random']:9.1f}%  {results['bottom']:9.1f}%")

    print("\nPaper: top-1000 ablation degrades accuracy by ≤0.4 pp (baseline ~94.8%).")
