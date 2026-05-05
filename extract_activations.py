"""
extract_activations.py
Extracts per-turn activation vectors + 5 trajectory scalars for one model.
Output: data/activations/<model_name>/<split>_<source>.npz
Each .npz: arrays X (N, d+5), y (N,), conv_ids (N,)
"""
import argparse, json, os, glob
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_CFG = {
    # --- Paper models (require H100/H200) ---
    "gemma":   {"path": "models/gemma-3-27b-it",  "layer": 32, "d": 5376},
    "mistral": {"path": "models/mistral-24b-it",  "layer": 24, "d": 5120},
    "qwen":    {"path": "models/qwen-32b-it",     "layer": 32, "d": 5120},
    "llama":   {"path": "models/llama-70b-it",    "layer": 40, "d": 8192},
    # --- Small models (GTX 1650 / 4 GB VRAM) ---
    # bfloat16: qwen1.5b (~3 GB), llama1b (~2 GB) fit without quantization.
    # Use --quantize for gemma2b (~2 GB q4), llama3b (~2 GB q4), phi3.5 (~2 GB q4).
    "qwen1.5b": {"path": "Qwen/Qwen2.5-1.5B-Instruct",      "layer": 14, "d": 1536},
    "llama1b":  {"path": "meta-llama/Llama-3.2-1B-Instruct", "layer":  8, "d": 2048},
    "llama3b":  {"path": "meta-llama/Llama-3.2-3B-Instruct", "layer": 14, "d": 3072},
    "gemma2b":  {"path": "google/gemma-2-2b-it",             "layer": 13, "d": 2304},
    "phi3.5":   {"path": "microsoft/Phi-3.5-mini-instruct",  "layer": 16, "d": 3072},
}

def load_model(cfg, quantize=False):
    if quantize:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], quantization_config=bnb, device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            attn_implementation="eager",   # required for forward-hook compatibility
        )
    model.eval()
    tok = AutoTokenizer.from_pretrained(cfg["path"])
    return model, tok

def extract_conv(conv_json, model, tok, layer_idx):
    """Returns (X_turns, y_turns) for all user turns in one conversation."""
    hook_out = []

    def hook_fn(module, inp, out):
        hook_out.append(out[0].detach().cpu().float())  # [batch, seq, d]; cast BF16→FP32

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

    msgs_so_far = []
    activations, labels = [], []
    prev_act = None

    for msg in conv_json.get("messages", []):
        msgs_so_far.append({"role": msg["role"], "content": msg["content"]})
        if msg["role"] != "user":
            continue

        # Cumulative context: all messages up to this user turn (Algorithm 3)
        ctx = tok.apply_chat_template(msgs_so_far, tokenize=False, add_generation_prompt=False)
        ids = tok(ctx, return_tensors="pt").input_ids.to(model.device)

        with torch.no_grad():
            _ = model(ids)

        act = hook_out[-1][0, -1, :].numpy()   # last-token, shape (d,)

        # 5 trajectory scalars (Algorithm 3, Appendix E)
        if prev_act is None:
            scalars = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
            cum_drift = 0.0
            prev_mag  = 0.0
        else:
            delta     = act - prev_act
            mag       = float(np.linalg.norm(delta))
            cos       = float(np.dot(act, prev_act) / (np.linalg.norm(act) * np.linalg.norm(prev_act) + 1e-9))
            cum_drift += mag
            accel     = mag - prev_mag
            t         = len(activations) + 1
            mean_d    = cum_drift / (t - 1)
            scalars   = np.array([mag, cos, cum_drift, accel, mean_d])
            prev_mag  = mag

        prev_act = act
        feature  = np.concatenate([act, scalars])   # (d+5,)

        # Map label string to int: benign=0, pivoting=1, adversarial=2
        lbl_str = msg.get("label", "benign")
        lbl = {"benign": 0, "pivoting": 1, "adversarial": 2}.get(lbl_str, 0)

        activations.append(feature)
        labels.append(lbl)

    handle.remove()
    return np.array(activations, dtype=np.float32), np.array(labels, dtype=np.int8)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    required=True, choices=list(MODEL_CFG))
    ap.add_argument("--source",   required=True,
                    choices=["synthetic", "lmsys", "safedial", "synthetic_extended"])
    ap.add_argument("--split",    default="train", choices=["train", "eval"],
                    help="Ignored for synthetic_extended (no train/eval subdirectory)")
    ap.add_argument("--quantize", action="store_true",
                    help="Load in 4-bit via bitsandbytes (pip install bitsandbytes). "
                         "Required for llama3b/gemma2b/phi3.5 on a 4 GB GPU.")
    args = ap.parse_args()

    cfg = MODEL_CFG[args.model]
    print(f"Loading {args.model} (layer={cfg['layer']}, d={cfg['d']}, quantize={args.quantize})…")
    model, tok = load_model(cfg, quantize=args.quantize)

    if args.source == "synthetic_extended":
        data_dir = "data/synthetic_extended"
    else:
        data_dir = f"data/{args.source}/{args.split}"
    files    = sorted(glob.glob(f"{data_dir}/conv_*.json"))
    print(f"Processing {len(files)} conversations from {data_dir}")

    all_X, all_y, all_ids = [], [], []
    for f in tqdm(files):
        conv = json.load(open(f))
        X, y = extract_conv(conv, model, tok, cfg["layer"])
        if len(X):
            all_X.append(X); all_y.append(y)
            conv_id = int(os.path.basename(f).split("_")[1].split(".")[0])
            all_ids.extend([conv_id] * len(y))

    out_dir = f"data/activations/{args.model}"
    os.makedirs(out_dir, exist_ok=True)
    if args.source == "synthetic_extended":
        out_path = f"{out_dir}/train_synthetic_extended.npz"
    else:
        out_path = f"{out_dir}/{args.split}_{args.source}.npz"
    np.savez_compressed(out_path,
                        X=np.vstack(all_X),
                        y=np.concatenate(all_y),
                        conv_ids=np.array(all_ids))
    print(f"Saved {out_path}  ({np.vstack(all_X).shape})")
