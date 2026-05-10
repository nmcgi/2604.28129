"""
deploy/review.py
Hybrid LLM + human review queue (Appendix N.1 step 3).

Reads flagged conversations from a queue directory, runs a text-level
second-opinion classifier (default: PromptGuard-86M), and routes:

  - LAD flagged + text classifier flagged   -> data/labeled/auto/   (auto-label adversarial)
  - LAD flagged + text classifier benign    -> data/labeled/review/ (human review queue)

The auto-labeled stream feeds adapt.py for retraining; the human queue
needs an operator to walk it via `--review-cli`.

Each input conversation file is a JSON document with the same schema as
data/synthetic/eval/conv_*.json, optionally augmented with the LAD turn
probabilities under a `lad_probs` key.

Paper N.2: separation of activation extraction (GPU, model-dependent)
from probe training (CPU, fast) is what makes this loop cheap; the same
holds for review -- text classification runs on CPU per turn.

Usage:
  # Route flagged conversations through the second-opinion classifier
  python deploy/review.py --queue queues/flagged --auto data/labeled/auto \
                          --review data/labeled/review

  # Walk the human-review queue interactively
  python deploy/review.py --review-cli data/labeled/review
"""
import argparse
import json
import os
import shutil
import sys
import time

THETA = 0.5


def load_promptguard():
    from transformers import pipeline
    return pipeline("text-classification",
                    model="meta-llama/Prompt-Guard-86M",
                    device=-1, top_k=None)


def text_classifier_score(pipe, text: str) -> float:
    """Returns P(non-benign) under the second-opinion text classifier."""
    scores = pipe(text[:2000], truncation=True)[0]
    return float(sum(s["score"] for s in scores
                     if s["label"].lower() != "benign"))


def conv_text_score(pipe, conv: dict) -> float:
    """Max non-benign score across user turns."""
    msgs = conv.get("conversation", conv.get("messages", []))
    user_texts = [m.get("content", m.get("text", "")) for m in msgs
                  if m.get("role", m.get("speaker", "")) == "user"]
    if not user_texts:
        return 0.0
    return max(text_classifier_score(pipe, t) for t in user_texts)


def route_one(path: str, pipe, auto_dir: str, review_dir: str) -> str:
    with open(path) as f:
        conv = json.load(f)
    text_p = conv_text_score(pipe, conv)
    flagged_by_text = text_p > THETA
    target = auto_dir if flagged_by_text else review_dir
    os.makedirs(target, exist_ok=True)
    out = os.path.join(target, os.path.basename(path))
    conv["text_classifier_p"] = text_p
    conv["text_classifier_flagged"] = flagged_by_text
    with open(out, "w") as f:
        json.dump(conv, f, indent=2)
    return out


def route_queue(queue_dir: str, auto_dir: str, review_dir: str):
    if not os.path.isdir(queue_dir):
        sys.exit(f"Queue {queue_dir} does not exist")
    print("Loading PromptGuard...")
    pipe = load_promptguard()
    files = sorted(f for f in os.listdir(queue_dir) if f.endswith(".json"))
    if not files:
        print(f"No flagged conversations in {queue_dir}")
        return
    print(f"Routing {len(files)} flagged conversations from {queue_dir}\n")
    for fname in files:
        src = os.path.join(queue_dir, fname)
        out = route_one(src, pipe, auto_dir, review_dir)
        bucket = "auto" if out.startswith(auto_dir) else "review"
        print(f"  {fname} -> {bucket}")
        os.remove(src)


def review_cli(review_dir: str, label_log: str = "logs/labels.jsonl"):
    """Interactively walk the human-review queue."""
    if not os.path.isdir(review_dir):
        sys.exit(f"Review directory {review_dir} does not exist")
    files = sorted(f for f in os.listdir(review_dir) if f.endswith(".json"))
    if not files:
        print(f"No conversations awaiting review in {review_dir}")
        return
    os.makedirs(os.path.dirname(label_log) or ".", exist_ok=True)
    log = open(label_log, "a")
    for fname in files:
        path = os.path.join(review_dir, fname)
        with open(path) as f:
            conv = json.load(f)
        print(f"\n----- {fname} -----")
        msgs = conv.get("conversation", conv.get("messages", []))
        for i, m in enumerate(msgs):
            role = m.get("role", m.get("speaker", ""))
            text = m.get("content", m.get("text", ""))[:200]
            label_hint = m.get("label", "")
            tag = f"[{label_hint}]" if label_hint else ""
            print(f"  T{i:02d} {role:9s} {tag} {text}")
        print(f"  text_classifier_p = {conv.get('text_classifier_p', 'n/a')}")

        choice = input("Label this conv [b]enign / [a]dversarial / [s]kip: ").strip().lower()
        if choice not in ("b", "a"):
            continue
        label = "benign" if choice == "b" else "adversarial"
        conv_id = conv.get("conv_id", os.path.splitext(fname)[0])
        log.write(json.dumps({"ts": time.time(), "conv_id": conv_id,
                              "label": label, "labeler": "human"}) + "\n")
        log.flush()
        # Move to a "completed" bucket so we don't re-prompt
        done_dir = os.path.join(review_dir, "_done")
        os.makedirs(done_dir, exist_ok=True)
        shutil.move(path, os.path.join(done_dir, fname))
    log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue",       default="queues/flagged",
                    help="Directory of flagged-conversation JSON files to route")
    ap.add_argument("--auto",        default="data/labeled/auto",
                    help="Output directory for auto-labeled (LAD + text agree)")
    ap.add_argument("--review",      default="data/labeled/review",
                    help="Output directory for human-review queue (LAD + text disagree)")
    ap.add_argument("--review-cli",  default=None, metavar="REVIEW_DIR",
                    help="Walk a human-review queue and write decisions to label log")
    ap.add_argument("--label-log",   default="logs/labels.jsonl",
                    help="Where review-cli writes label decisions")
    args = ap.parse_args()

    if args.review_cli:
        review_cli(args.review_cli, args.label_log)
    else:
        route_queue(args.queue, args.auto, args.review)


if __name__ == "__main__":
    main()
