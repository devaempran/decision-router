"""The end-to-end check: the test tickets through Redis, the deciders and decisionLM, read back from the per-team streams.

    python eval/run_pipeline.py --theta 0.85            (decisionLM must already be answering; Redis on port 6380)

Writes a pipeline config for the eight teams, starts `pipeline/run.py start` with it, publishes every test ticket to the
ingestion stream, waits until each one is on a team stream or in the dead-letter stream, then writes
results/raw/pipeline_test.jsonl (what report.py reads). Only streams prefixed `eval:` are touched."""
import argparse
import json
import os
import subprocess
import sys
import time

import redis

from collect import HERE, load_tickets
from queues import QUESTION, TEAMS

ROOT = os.path.dirname(HERE)
PREFIX = "eval:"
STREAMS = {option: f"{PREFIX}team{i}" for i, option in enumerate(TEAMS, 1)}


def config_text(theta, port, service_url, dashboard_port):
    options = "\n".join(f'[[options]]\nname = "{option}"\nstream = "{stream}"\n' for option, stream in STREAMS.items())
    return f"""[redis]
host = "127.0.0.1"
port = {port}

[ingestion]
stream = "{PREFIX}tickets"
group = "eval-deciders"
dlq_stream = "{PREFIX}tickets:dlq"

[decision]
url = "{service_url}"
question = "{QUESTION}"
theta = {theta}
model_version = "decisionLM-qwen3-8b-q8"
review_stream = "{PREFIX}review"

{options}
[dashboard]
port = {dashboard_port}
label_field = "expected"

[service]
start = false

[run]
deciders = 1   # one GPU, and the service answers one request at a time

[reliability]
reclaim_idle_s = 900
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--theta", type=float, default=0.85)
    ap.add_argument("--redis-port", type=int, default=6380)
    ap.add_argument("--service-url", default="http://127.0.0.1:5001/api/decide")
    ap.add_argument("--dashboard-port", type=int, default=5056)
    ap.add_argument("--timeout-min", type=float, default=90)
    args = ap.parse_args()

    r = redis.Redis(port=args.redis_port, decode_responses=True)
    r.ping()
    for key in r.scan_iter(f"{PREFIX}*"):
        r.delete(key)
    cfg_path = os.path.join(HERE, "results", "pipeline.eval.toml")
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(config_text(args.theta, args.redis_port, args.service_url, args.dashboard_port))

    tickets = load_tickets("test")
    gold = {t["id"]: t["gold"] for t in tickets}
    env = dict(os.environ, REDIS_PORT=str(args.redis_port))
    proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "pipeline", "run.py"), "start", "--config", cfg_path], env=env)
    try:
        time.sleep(5)  # groups are created at start, but events written earlier are still decided (groups start at 0)
        for t in tickets:
            r.xadd(f"{PREFIX}tickets", {"id": t["id"], "payload": t["text"], "expected": t["gold"]})
        print(f"published {len(tickets)} tickets at theta {args.theta}", flush=True)
        deadline = time.time() + args.timeout_min * 60
        while time.time() < deadline:
            routed = sum(r.xlen(s) for s in STREAMS.values()) + r.xlen(f"{PREFIX}review") + r.xlen(f"{PREFIX}tickets:dlq")
            print(f"  {routed}/{len(tickets)} finished", flush=True)
            if routed >= len(tickets):
                break
            time.sleep(30)
    finally:
        proc.terminate()
        proc.wait(timeout=30)

    rows, seen = [], set()
    for option, stream in STREAMS.items():
        for _, fields in r.xrange(stream):
            if fields["id"] in seen:
                continue
            seen.add(fields["id"])
            rows.append({"id": fields["id"], "gold": gold[fields["id"]], "answer": fields.get("label", option),
                         "confidence": float(fields.get("confidence", 0)), "source": fields.get("source"), "theta": args.theta})
    dead = [fields["id"] for _, fields in r.xrange(f"{PREFIX}tickets:dlq")]
    out = os.path.join(HERE, "results", "raw", "pipeline_test.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    print(f"{len(rows)} routed, {len(dead)} dead-lettered, {len(tickets) - len(rows) - len(dead)} missing -> {out}")


if __name__ == "__main__":
    main()
