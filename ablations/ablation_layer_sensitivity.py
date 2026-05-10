"""
ablation_layer_sensitivity.py
Layer sensitivity sweep (Appendix G).

Re-extracts activations from every decoder layer (or a sampled subset) for one
model on the synthetic eval set and reports 5-fold CV accuracy for each layer.
Confirms that the trajectory-scalar signal is not layer-specific (<1.2 pp spread).

Because extracting all layers is expensive, use --step to sample every Nth layer.

Usage:
  python ablation_layer_sensitivity.py --model gemma [--step 4] [--n-convs 100]
"""
import argparse
import json
import glob
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from tqdm import tqdm

MODEL_CFG = {
    # Paper models
    "gemma":   {"path": "models/gemma-3-27b-it",                    "n_layers": 62},
    "mistral": {"path": "models/mistral-24b-it",                    "n_layers": 40},
    "qwen":    {"path": "models/qwen-32b-it",                       "n_layers": 64},
    "llama":   {"path": "models/llama-70b-it",                      "n_layers": 80},
    # Small models (GTX 1650 / 4 GB VRAM)
    "qwen1.5b": {"path": "Qwen/Qwen2.5-1.5B-Instruct",             "n_layers": 28},
    "llama1b":  {"path": "meta-llama/Llama-3.2-1B-Instruct",       "n_layers": 16},
    "llama3b":  {"path": "meta-llama/Llama-3.2-3B-Instruct",       "n_layers": 28},
    "gemma2b":  {"path": "google/gemma-2-2b-it",                   "n_layers": 26},
    "phi3.5":   {"path": "microsoft/Phi-3.5-mini-instruct",        "n_layers": 32},
}


def extract_one_layer(conv_json, model, tok, layer_idx):
    """Return (scalars, labels) for user turns in one conversation, scalar-only (5 features)."""
    hook_out = []

    def hook_fn(module, inp, out):
        hook_out.append(out[0].detach().cpu().float())

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    msgs_so_far, scalars_list, labels = [], [], []
    prev_act, cum_drift, prev_mag = None, 0.0, 0.0

    for msg in conv_json.get("messages", []):
        msgs_so_far.append({"role": msg["role"], "content": msg["content"]})
        if msg["role"] != "user":
            continue

        ctx = tok.apply_chat_template(msgs_so_far, tokenize=False, add_generation_prompt=False)
        ids = tok(ctx, return_tensors="pt",
                  truncation=True, max_length=4096).input_ids.to(model.device)
        with torch.no_grad():
            _ = model(ids)

        act = hook_out[-1][0, -1, :].numpy()

        if prev_act is None:
            s = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        else:
            delta     = act - prev_act
            mag       = float(np.linalg.norm(delta))
            cos       = float(np.dot(act, prev_act) / (np.linalg.norm(act) * np.linalg.norm(prev_act) + 1e-9))
            cum_drift += mag
            t         = len(scalars_list) + 1
            s = np.array([mag, cos, cum_drift, mag - prev_mag, cum_drift / (t - 1) if t > 1 else 0.0])
            prev_mag  = mag
        prev_act = act

        scalars_list.append(s)
        lbl_str = msg.get("label", "benign")
        labels.append(0 if lbl_str == "benign" else 1)

    handle.remove()
    return np.array(scalars_list, dtype=np.float32), np.array(labels, dtype=np.int8)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   required=True, choices=list(MODEL_CFG))
    ap.add_argument("--step",    type=int, default=4, help="Evaluate every Nth layer")
    ap.add_argument("--n-convs", type=int, default=100,
                    help="Number of eval conversations to use (subset for speed)")
    args = ap.parse_args()

    cfg = MODEL_CFG[args.model]
    print(f"Loading {args.model}…")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16,
        device_map="balanced", attn_implementation="eager",
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(cfg["path"])

    files = sorted(glob.glob("data/synthetic/eval/conv_*.json"))[: args.n_convs]
    convs = [json.load(open(f)) for f in files]

    layers = list(range(0, cfg["n_layers"], args.step))
    print(f"Sweeping {len(layers)} layers (step={args.step}) on {len(convs)} conversations\n")

    results = []
    for layer_idx in tqdm(layers, desc="Layers"):
        all_X, all_y = [], []
        for conv in convs:
            X, y = extract_one_layer(conv, model, tok, layer_idx)
            if len(X):
                all_X.append(X)
                all_y.append(y)
        if not all_X:
            continue
        X_cat = np.vstack(all_X)
        y_cat = np.concatenate(all_y)
        if y_cat.sum() == 0 or (y_cat == 0).sum() == 0:
            continue

        pipe = make_pipeline(
            StandardScaler(),
            xgb.XGBClassifier(n_estimators=100, max_depth=4, eval_metric="logloss",
                               random_state=42, tree_method="hist"),
        )
        scores = cross_val_score(pipe, X_cat, y_cat, cv=5, scoring="accuracy")
        acc = scores.mean() * 100
        results.append((layer_idx, acc))
        pct = layer_idx / cfg["n_layers"] * 100
        print(f"  Layer {layer_idx:3d} ({pct:4.0f}%)  acc={acc:.1f}%")

    if results:
        accs = [r[1] for r in results]
        print(f"\nSpread: {max(accs)-min(accs):.1f} pp  (paper target: <1.2 pp with scalar features)")
