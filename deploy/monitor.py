"""
deploy/monitor.py
Sliding-window FP-rate monitor + novel-trajectory detector (Appendix N.3).

Tails a prediction log written by lad_infer.py (--log) and a label log
written by deploy/review.py. Joins on conv_id, computes:

  - sliding-window FP rate over the last N labeled flagged convs
  - "novel trajectory" alerts: high cumulative drift + low P(adv)

When FP rate exceeds the configured threshold, fires a retrain by writing
a sentinel file (data/labeled/.retrain-trigger) that deploy/adapt.py picks
up. This keeps the monitor stateless and the trigger auditable.

Paper N.3:
  - Rising FP: if the fraction of flagged benign convs increases over a
    sliding window, the deployment distribution has shifted -- retrain.
  - Novel trajectory patterns: high cum_drift but low P(adv) suggests an
    attack pattern absent from training -- escalate for human review.

Usage:
  python deploy/monitor.py --pred-log logs/preds.jsonl \
                           --label-log logs/labels.jsonl \
                           --window 200 --fp-threshold 0.05
"""
import argparse
import json
import os
import sys
import time
from collections import deque
from typing import Iterable

DEFAULT_TRIGGER_PATH = "data/labeled/.retrain-trigger"


def tail_jsonl(path: str, poll_interval: float = 1.0) -> Iterable[dict]:
    """Yield each new JSON line appended to `path`. Blocks indefinitely.

    Files that don't exist yet are waited for; truncations are handled by
    re-opening when the file shrinks. This is a deliberately small tail
    implementation -- swap for a proper log shipper in production.
    """
    while not os.path.exists(path):
        time.sleep(poll_interval)
    f = open(path, "r")
    f.seek(0, os.SEEK_END)
    while True:
        line = f.readline()
        if not line:
            # Detect rotation: stat tells us if the file was truncated
            if os.path.getsize(path) < f.tell():
                f.close()
                f = open(path, "r")
            time.sleep(poll_interval)
            continue
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def fire_retrain(trigger_path: str, reason: dict) -> None:
    os.makedirs(os.path.dirname(trigger_path) or ".", exist_ok=True)
    with open(trigger_path, "w") as f:
        json.dump({"ts": time.time(), **reason}, f)
    print(f"  [TRIGGER] retrain requested -> {trigger_path}: {reason}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-log",      required=True,
                    help="JSONL prediction log written by lad_infer.py --log")
    ap.add_argument("--label-log",     required=True,
                    help="JSONL label log written by deploy/review.py")
    ap.add_argument("--window",        type=int, default=200,
                    help="Sliding window size in flagged-and-labeled convs")
    ap.add_argument("--fp-threshold",  type=float, default=0.05,
                    help="Trigger retrain if FP rate exceeds this (default: 0.05)")
    ap.add_argument("--novel-drift",   type=float, default=50.0,
                    help="Cumulative drift above this with P(adv)<novel-padv "
                         "is reported as a novel trajectory")
    ap.add_argument("--novel-padv",    type=float, default=0.3,
                    help="P(adv) ceiling for novel-trajectory alerts")
    ap.add_argument("--trigger-path",  default=DEFAULT_TRIGGER_PATH)
    ap.add_argument("--once",          action="store_true",
                    help="Replay logs once and exit (test mode)")
    args = ap.parse_args()

    # State: window of (conv_id, was_flagged, ground_truth_adv)
    window = deque(maxlen=args.window)
    # In-memory join state until each pred is matched with a label
    pending_preds = {}  # conv_id -> {p_adv, flagged, cum_drift}
    seen_labels   = {}  # conv_id -> label

    def consume_pred(rec: dict):
        cid = rec.get("conv_id")
        if cid is None:
            return
        pending_preds[cid] = {
            "p_adv":     float(rec.get("p_adv", 0.0)),
            "flagged":   bool(rec.get("flagged", False)),
            "cum_drift": float(rec.get("cum_drift", 0.0)),
        }
        # Novel-trajectory detector: high drift, low p_adv
        if (rec.get("cum_drift", 0.0) >= args.novel_drift and
                rec.get("p_adv", 0.0) < args.novel_padv):
            print(f"  [novel?] conv={cid} cum_drift={rec['cum_drift']:.1f} "
                  f"p_adv={rec['p_adv']:.2f}")
        # Match with any waiting label
        if cid in seen_labels:
            consume_join(cid, seen_labels.pop(cid))

    def consume_label(rec: dict):
        cid = rec.get("conv_id")
        if cid is None:
            return
        label = rec.get("label", "benign")
        if cid in pending_preds:
            consume_join(cid, label)
        else:
            seen_labels[cid] = label

    def consume_join(cid, label: str):
        pred = pending_preds.pop(cid, None)
        if pred is None or not pred["flagged"]:
            return  # only flagged convs contribute to FP rate
        was_adv = label in ("pivoting", "adversarial")
        window.append((cid, True, was_adv))
        # Compute sliding FP
        flagged = sum(1 for _, f, _ in window if f)
        false_p = sum(1 for _, f, t in window if f and not t)
        fp_rate = false_p / flagged if flagged else 0.0
        print(f"  [join] conv={cid} label={label} fp_rate={fp_rate:.3f} "
              f"({false_p}/{flagged} in window of {len(window)})")
        if flagged >= args.window // 2 and fp_rate > args.fp_threshold:
            fire_retrain(args.trigger_path,
                         {"fp_rate": fp_rate, "n_flagged": flagged,
                          "window": args.window})

    def replay_once(path: str, callback):
        if not os.path.exists(path):
            return
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    callback(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if args.once:
        replay_once(args.pred_log,  consume_pred)
        replay_once(args.label_log, consume_label)
        return

    print(f"Monitoring {args.pred_log} + {args.label_log} "
          f"(window={args.window}, fp_threshold={args.fp_threshold})")
    # Naive interleave: tail predictions, dip into labels each cycle.
    pred_iter  = tail_jsonl(args.pred_log)
    for rec in pred_iter:
        consume_pred(rec)
        # Drain any new labels that appeared since last cycle
        try:
            with open(args.label_log) as lf:
                lf.seek(getattr(consume_label, "_offset", 0))
                for line in lf:
                    line = line.strip()
                    if line:
                        try:
                            consume_label(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                consume_label._offset = lf.tell()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
