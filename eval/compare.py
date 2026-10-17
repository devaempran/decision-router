"""Local two-tier setup against hosted models, on the test split.

    python eval/compare.py       needs report.py's inputs plus results/raw/hosted_<model>_test.jsonl (hosted.py)
                                 writes results/COMPARISON.md

Local rows come from the recorded tier-1 / reasoner runs (report.cascade); hosted rows from hosted.py. Accuracy counts an
unusable hosted reply as wrong. Local cost is not a per-call fee: it is shown as GPU-seconds per 1,000 decisions."""
import json
import os

import numpy as np

from hosted import PRICE_DATE, PRICES
from report import HERE, RAW, boot_ci, cascade, ece, pct, split_data

REPORT = os.path.join(HERE, "results", "report.json")


def per_decision_latency(d, theta):
    esc = d["conf"] < theta
    return d["lat1"] + np.where(esc, np.nan_to_num(d["lat2"]), 0)


def local_rows(test, thetas):
    rows = []
    for label, theta in thetas:
        r = cascade(test, theta)
        if r is None:
            continue
        lat = per_decision_latency(test, theta)
        rows.append({"name": label, "accuracy": r["accuracy"], "ci95": boot_ci(r["ok"]), "p50_ms": float(np.median(lat) * 1000),
                     "p95_ms": float(np.percentile(lat, 95) * 1000),
                     "ece": ece(test["conf"], (test["pred1"] == test["gold"]).astype(float))[0] if theta == 0 else None,
                     "cost": f"{r['gpu_seconds_per_1000']:.0f} GPU-s per 1,000"})
    return rows


def hosted_row(model, test):
    path = os.path.join(RAW, f"hosted_{model}_test.jsonl")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        recs = {r["id"]: r for r in map(json.loads, f)}
    if set(recs) != set(test["ids"]):
        print(f"{model}: {len(recs)} of {len(test['ids'])} test tickets done; compared on those only")
    ids = [i for i in test["ids"] if i in recs]
    ok = np.array([recs[i]["answer"] == recs[i]["gold"] for i in ids], dtype=float)
    used = [recs[i] for i in ids if recs[i].get("answer")]
    conf = np.array([r["confidence"] for r in used])
    used_ok = np.array([r["answer"] == r["gold"] for r in used], dtype=float)
    wall = np.array([r["wall_s"] for r in used]) * 1000
    pin, pout = PRICES[model]
    dollars = np.mean([(r["input_tokens"] * pin + r["output_tokens"] * pout) / 1e6 for r in used]) * 1000
    return {"name": f"{model} (prompted)", "n": len(ids), "accuracy": float(ok.mean()), "ci95": boot_ci(ok),
            "p50_ms": float(np.median(wall)), "p95_ms": float(np.percentile(wall, 95)), "ece": ece(conf, used_ok)[0],
            "stated": float(conf.mean()), "cost": f"${dollars:.2f} per 1,000", "dollars_per_1000": float(dollars),
            "unusable": len(ids) - len(used)}


def main():
    test = split_data("test")
    with open(REPORT, encoding="utf-8") as f:
        chosen = json.load(f)["chosen_theta"]
    rows = local_rows(test, [("local, tier 1 only", 0.0), ("local, two-tier at 0.85", 0.85), (f"local, two-tier at {chosen}", chosen)])
    hosted = [r for r in (hosted_row(m, test) for m in PRICES) if r]
    seen = set()
    rows = [r for r in rows if not (r["name"] in seen or seen.add(r["name"]))] + hosted
    L = ["# Local two-tier setup vs hosted models", "",
         f"{len(test['ids'])} test complaints, 8 teams. Hosted rows: one plain prompted call per ticket (hosted.py), list prices of {PRICE_DATE}. "
         "Latency is wall-clock per decision on one RTX 5060 Ti (local) or over the network (hosted), one request at a time.", "",
         "| setup | accuracy (95% interval) | latency p50 | latency p95 | calibration error of its own stated confidence | cost |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        e = f"{r['ece']:.3f}" if r["ece"] is not None else "n/a (the reasoner states no confidence)"
        L.append(f"| {r['name']} | {pct(r['accuracy'])} ({pct(r['ci95'][0])} to {pct(r['ci95'][1])}) | {r['p50_ms']:.0f} ms | "
                 f"{r['p95_ms']:.0f} ms | {e} | {r['cost']} |")
    if not hosted:
        L += ["", "No hosted results yet: run `python eval/hosted.py --model claude-haiku-5-5` and `--model claude-sonnet-5-5` (needs ANTHROPIC_API_KEY)."]
    with open(os.path.join(HERE, "results", "COMPARISON.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
