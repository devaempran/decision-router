"""Local tier 1 decides, a hosted model (Claude Haiku 5.5) takes over below a confidence threshold.

    python eval/hybrid.py        needs results/raw/{tier1,hosted_claude-haiku-5-5}_{fit,test}.jsonl; writes results/HYBRID.md

Simulated exactly from the recorded runs: a ticket whose tier-1 confidence is below theta gets Haiku's answer, the rest keep tier 1's.
Every ticket pays tier 1 first (the decider), so an escalated ticket's latency is tier 1 plus the Haiku call. Theta is chosen on the
fit split (the least escalation within half a point of the best fit accuracy) and the test split is scored once."""
import json
import os

import numpy as np

from hosted import PRICES
from report import HERE, boot_ci, load, pct, split_data

MODEL = "claude-haiku-5-5"
THETAS = [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]  # 1.01: every ticket goes to Haiku


def data(split):
    d = split_data(split)
    h = load(f"hosted_{MODEL}_{split}.jsonl")
    pin, pout = PRICES[MODEL]
    d["pred_h"] = np.array([d["options"].index(h[i]["answer"]) if h[i].get("answer") in d["options"] else -1 for i in d["ids"]])
    d["lat_h"] = np.array([h[i]["wall_s"] for i in d["ids"]])
    d["cost_h"] = np.array([(h[i]["input_tokens"] * pin + h[i]["output_tokens"] * pout) / 1e6 for i in d["ids"]])
    return d


def hybrid(d, theta):
    esc = d["conf"] < theta
    ok = (np.where(esc, d["pred_h"], d["pred1"]) == d["gold"]).astype(float)
    overall = d["lat1"] + np.where(esc, d["lat_h"], 0)
    return {"theta": theta, "escalated": float(esc.mean()), "accuracy": float(ok.mean()), "ci95": boot_ci(ok),
            "decider_avg_ms": float(d["lat1"].mean() * 1000), "overall_avg_ms": float(overall.mean() * 1000),
            "overall_p95_ms": float(np.percentile(overall, 95) * 1000),
            "dollars_per_1000": float(np.where(esc, d["cost_h"], 0).mean() * 1000)}


def label(theta):
    return "tier 1 only" if theta == 0 else "Haiku only" if theta > 1 else f"hybrid at {theta}"


def main():
    fit, test = data("fit"), data("test")
    fit_rows = [hybrid(fit, t) for t in THETAS]
    best = max(r["accuracy"] for r in fit_rows)
    chosen = min((r for r in fit_rows if r["accuracy"] >= best - 0.005 and r["theta"] <= 1), key=lambda r: (r["escalated"], r["theta"]))["theta"]

    sweep = [(hybrid(fit, t), hybrid(test, t)) for t in THETAS]
    low = test["conf"] < 0.85
    low_stats = {"n": int(low.sum()), "tier1": float((test["pred1"] == test["gold"])[low].mean()),
                 "haiku": float((test["pred_h"] == test["gold"])[low].mean())}
    shown = [0.0, 0.85, 0.9, chosen, 1.01]
    final = {label(t): hybrid(test, t) for t in dict.fromkeys(shown)}

    L = ["# Hybrid: local tier 1 decides, Claude Haiku 5.5 takes the uncertain tickets", "",
         f"{len(test['ids'])} test complaints, 8 teams. Haiku is the same plain prompted call as in COMPARISON.md. Every ticket goes to tier 1 first;",
         "tickets below the threshold are then sent to Haiku too, so an escalated ticket's latency is tier 1 plus the Haiku call.", "",
         "## Threshold sweep", "",
         "| theta | fit: escalated | fit: accuracy | test: escalated | test: accuracy | test: overall avg latency | test: $ per 1,000 |",
         "|---|---|---|---|---|---|---|"]
    for f, t in sweep:
        th = "all to Haiku" if f["theta"] > 1 else f"{f['theta']:.2f}"
        L.append(f"| {th} | {pct(f['escalated'])} | {pct(f['accuracy'])} | {pct(t['escalated'])} | {pct(t['accuracy'])} | "
                 f"{t['overall_avg_ms']:.0f} ms | ${t['dollars_per_1000']:.3f} |")
    L += ["", f"Chosen on the fit split: theta = {chosen}.", "", "## Test split, scored once", "",
          "| setup | escalated | accuracy (95% interval) | decider avg latency | overall avg latency | overall p95 | $ per 1,000 decisions |",
          "|---|---|---|---|---|---|---|"]
    for name, r in final.items():
        L.append(f"| {name} | {pct(r['escalated'])} | {pct(r['accuracy'])} ({pct(r['ci95'][0])} to {pct(r['ci95'][1])}) | "
                 f"{r['decider_avg_ms']:.0f} ms | {r['overall_avg_ms']:.0f} ms | {r['overall_p95_ms']:.0f} ms | ${r['dollars_per_1000']:.3f} |")
    L += ["", f"On the {low_stats['n']} test tickets below 0.85 confidence: tier 1 scores {pct(low_stats['tier1'])}, Haiku {pct(low_stats['haiku'])}.", "",
          "Decider latency is the tier-1 average over every ticket (the cost every decision pays). Overall latency adds the Haiku call for escalated tickets.",
          "The Haiku-only row here still pays tier 1 first; Haiku called directly has a median of 927 ms (COMPARISON.md)."]
    text = "\n".join(L) + "\n"
    with open(os.path.join(HERE, "results", "HYBRID.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    with open(os.path.join(HERE, "results", "hybrid.json"), "w", encoding="utf-8") as f:
        json.dump({"chosen_theta": chosen, "test": final, "low_confidence_below_0.85": low_stats}, f, indent=1)
    print(text)


if __name__ == "__main__":
    main()
