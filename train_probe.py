"""
train_probe.py
Two probe variants (Section 3.4, Appendix C):

  --variant standard  (default)
    XGBoost directly on raw activations + 5 trajectory scalars (d+5 features).
    Matches the primary results in Sections 6–7 / Figure 7 (85–89% det, 2–4% FP).
    Saves: probes/<model_name>/xgb.json
           probes/<model_name>/scaler.pkl

  --variant scalar
    XGBoost on trajectory scalars only (5 features, no raw activations).
    Reproduces the scalars-only ablation from Section 5 (89.6% det, 57–74% FP).
    Saves: probes/<model_name>/xgb_scalar.json
           probes/<model_name>/scaler_scalar.pkl

  --variant contrastive
    Stage 3a: Contrastive MLP encoder (d → 512 → 128, CPU ~10 min)
    Stage 3b: XGBoost on 128-dim embeddings + 5 scalars (133 features, CPU ~2 min)
    Saves: probes/<model_name>/encoder.pt
           probes/<model_name>/xgb_contrastive.json
           probes/<model_name>/scaler_contrastive.pkl
"""
import argparse
import glob
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

MODEL_D = {
    # Paper models
    "gemma": 5376, "mistral": 5120, "qwen": 5120, "llama": 8192,
    # Small models (GTX 1650 / 4 GB VRAM)
    "qwen1.5b": 1536, "llama1b": 2048, "llama3b": 3072, "gemma2b": 2304, "phi3.5": 3072,
}

class ContrastiveEncoder(nn.Module):
    def __init__(self, d_in, hidden=512, out=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out),
        )
    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=-1)

def load_npz(paths):
    Xs, ys, ids = [], [], []
    offset = 0
    for p in paths:
        d = np.load(p)
        Xs.append(d["X"])
        ys.append(d["y"])
        conv_ids = d["conv_ids"] + offset
        ids.append(conv_ids)
        offset += int(d["conv_ids"].max()) + 1
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ids)

def build_pairs(X, y, conv_ids, n_pairs=50_000, seed=42):
    """Sample contrastive pairs enforcing cross-conversation membership for style invariance."""
    rng = np.random.default_rng(seed)
    y_bin = (y >= 1).astype(np.int8)  # collapse pivoting+adversarial → 1
    adv_idx = np.where(y_bin == 1)[0]
    ben_idx = np.where(y_bin == 0)[0]
    a_idx, b_idx, labels = [], [], []
    for _ in range(n_pairs // 2):
        # Same-intent positive pair — must come from different conversations
        pool = adv_idx if rng.random() > 0.5 else ben_idx
        for _ in range(20):  # retry until cross-conv pair found
            a, b = rng.choice(pool, 2, replace=False)
            if conv_ids[a] != conv_ids[b]:
                break
        a_idx.append(a)
        b_idx.append(b)
        labels.append(1)
        # Different-intent negative pair
        a = rng.choice(adv_idx)
        b = rng.choice(ben_idx)
        a_idx.append(a)
        b_idx.append(b)
        labels.append(0)
    return X[a_idx], X[b_idx], np.array(labels, dtype=np.float32)

def train_xgb(X_feat, y, model_dir, xgb_name="xgb.json", scaler_name="scaler.pkl"):
    """Fit StandardScaler + XGBoost and save to model_dir."""
    scaler = StandardScaler().fit(X_feat)
    X_norm = scaler.transform(X_feat)
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
    clf.save_model(f"{model_dir}/{xgb_name}")
    with open(f"{model_dir}/{scaler_name}", "wb") as f:
        pickle.dump(scaler, f)
    return clf, scaler

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   required=True, choices=list(MODEL_D))
    ap.add_argument("--variant", default="standard",
                    choices=["standard", "scalar", "contrastive"],
                    help="standard: raw activations+scalars → XGBoost (default, matches paper §6–7); "
                         "scalar: 5 trajectory scalars only → XGBoost (Section 5 ablation); "
                         "contrastive: MLP encoder → XGBoost")
    ap.add_argument("--epochs",  type=int, default=50)
    args = ap.parse_args()

    d = MODEL_D[args.model]
    act_dir   = f"data/activations/{args.model}"
    model_dir = f"probes/{args.model}"
    os.makedirs(model_dir, exist_ok=True)

    train_files = glob.glob(f"{act_dir}/train_*.npz")
    print(f"Loading training activations from: {train_files}")
    X_train, y_train, ids_train = load_npz(train_files)
    y_bin = (y_train >= 1).astype(np.int8)  # binarize 3-class labels for classification
    print(f"Train set: {X_train.shape}, {y_bin.sum()} adversarial / {(y_bin==0).sum()} benign")

    if args.variant == "standard":
        # Primary variant (Sections 6–7): raw activations + 5 scalars, no encoder
        print("\nTraining standard XGBoost (raw activations + 5 trajectory scalars)…")
        train_xgb(X_train, y_bin, model_dir)
        print(f"XGBoost + scaler saved → {model_dir}/")

    elif args.variant == "scalar":
        # Scalars-only ablation (Section 5): 5 trajectory features, no raw activations
        print("\nTraining scalar-only XGBoost (5 trajectory scalars, no activations)…")
        train_xgb(X_train[:, d:], y_bin, model_dir,
                  xgb_name="xgb_scalar.json", scaler_name="scaler_scalar.pkl")
        print(f"Scalar XGBoost + scaler saved → {model_dir}/")

    else:
        # --- Stage 3a: Contrastive Encoder ---
        print("\nStage 3a: Training contrastive encoder…")
        Pa, Pb, pair_labels = build_pairs(X_train[:, :d], y_bin, ids_train)
        Pa_t  = torch.tensor(Pa,          dtype=torch.float32)
        Pb_t  = torch.tensor(Pb,          dtype=torch.float32)
        lbl_t = torch.tensor(pair_labels, dtype=torch.float32)

        dataset = TensorDataset(Pa_t, Pb_t, lbl_t)
        loader  = DataLoader(dataset, batch_size=1024, shuffle=True)

        enc       = ContrastiveEncoder(d_in=d)
        optimizer = optim.AdamW(enc.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.CosineEmbeddingLoss(margin=0.2)

        for ep in range(args.epochs):
            total_loss = 0.0
            for a_b, b_b, y_b in loader:
                z_a, z_b = enc(a_b), enc(b_b)
                loss = criterion(z_a, z_b, y_b * 2 - 1)  # 0/1 → -1/+1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            if (ep + 1) % 10 == 0:
                print(f"  Epoch {ep+1}/{args.epochs}  loss={total_loss/len(loader):.4f}")

        torch.save(enc.state_dict(), f"{model_dir}/encoder.pt")
        print(f"Encoder saved → {model_dir}/encoder.pt")

        # --- Stage 3b: XGBoost on embeddings + scalars ---
        print("\nStage 3b: Training XGBoost classifier on contrastive embeddings…")
        enc.eval()
        with torch.no_grad():
            embs = enc(torch.tensor(X_train[:, :d], dtype=torch.float32)).numpy()  # (N, 128)
        X_feat = np.hstack([embs, X_train[:, d:]])  # (N, 133)
        train_xgb(X_feat, y_bin, model_dir,
                  xgb_name="xgb_contrastive.json", scaler_name="scaler_contrastive.pkl")
        print(f"Contrastive XGBoost + scaler saved → {model_dir}/")
