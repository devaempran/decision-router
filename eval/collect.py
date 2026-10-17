"""Send tickets straight to decisionLM's /api/decide and record every answer (resumable).

    python eval/collect.py --split fit  --theta 0 --out results/raw/tier1_fit.jsonl         # tier 1 only
    python eval/collect.py --split fit  --theta 2 --out results/raw/tier2_fit.jsonl \
        --only-below 0.95 --tier1 results/raw/tier1_fit.jsonl                               # force the reasoner

theta 0 never escalates (the fast path); theta 2 always does. Running the two lets report.py simulate any threshold
without re-running the model. `run_pipeline.py` is the end-to-end check through Redis and the deciders."""
import argparse
import json
import os
import time
import urllib.error
import urllib.request

from queues import QUESTION, TEAMS

HERE = os.path.dirname(os.path.abspath(__file__))
OPTIONS = list(TEAMS)


def load_tickets(split):
    with open(os.path.join(HERE, "data", "tickets.jsonl"), encoding="utf-8") as f:
        return [t for t in map(json.loads, f) if split in ("all", t["split"])]


def decide(url, text, theta, max_think_tokens, max_think_seconds, timeout):
    body = {"Context": text, "Question": QUESTION, "Options": OPTIONS, "theta": theta,
            "max_think_tokens": max_think_tokens, "max_think_seconds": max_think_seconds}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    out["wall_s"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:5001/api/decide")
    ap.add_argument("--split", choices=["fit", "test", "all"], required=True)
    ap.add_argument("--theta", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only-below", type=float, help="only tickets whose tier-1 confidence is below this (needs --tier1)")
    ap.add_argument("--tier1", help="a tier-1 results file, for --only-below")
    ap.add_argument("--max-think-tokens", type=int, default=6000)
    ap.add_argument("--max-think-seconds", type=float, default=150)
    args = ap.parse_args()

    out_path = os.path.join(HERE, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    tickets = load_tickets(args.split)
    if args.only_below is not None:
        with open(os.path.join(HERE, args.tier1), encoding="utf-8") as f:
            conf = {r["id"]: r["confidence"] for r in map(json.loads, f)}
        tickets = [t for t in tickets if conf.get(t["id"], 0) < args.only_below]
    todo = [t for t in tickets if t["id"] not in done]
    print(f"{len(tickets)} tickets, {len(done)} already done, {len(todo)} to run", flush=True)
    t_start = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for i, t in enumerate(todo, 1):
            try:
                r = decide(args.url, t["text"], args.theta, args.max_think_tokens, args.max_think_seconds, 600)
            except (urllib.error.URLError, OSError, ValueError) as e:
                print(f"  {t['id']}: {e!r}", flush=True)
                continue
            f.write(json.dumps({"id": t["id"], "gold": t["gold"], "queue": t["queue"], **r}) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}  {(time.time() - t_start) / i:.2f} s/ticket", flush=True)


if __name__ == "__main__":
    main()
