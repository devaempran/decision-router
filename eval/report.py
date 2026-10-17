"""Accuracy, calibration, escalation and speed of the local two-tier setup on the ticket splits.

    python eval/report.py            reads results/raw/*.jsonl, writes results/report.json and results/REPORT.md

Method: tier 1 ran on every ticket (theta 0). The reasoner ran on every ticket whose tier-1 confidence was below
SWEEP_MAX (theta 2 forces it). Any threshold up to SWEEP_MAX is then simulated exactly: a ticket below the threshold takes
the reasoner's answer, the rest keep tier 1's. The threshold is chosen on the fit split and scored once on the test split."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "results", "raw")
SWEEP_MAX = 0.90
THETAS = [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
SHIPPED_THETA = 0.85
with open(os.path.join(HERE, "..", "service", "config", "calibration.json"), encoding="utf-8") as _f:
    SERVICE_T = json.load(_f)["temperature"]
rng = np.random.default_rng(0)


def load(name):
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f) if r.get("answer")}


def ece(conf, ok, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - ok[m].mean())
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "confidence": float(conf[m].mean()),
                         "accuracy": float(ok[m].mean())})
    return float(total), rows


def boot_ci(values, n=2000):
    values = np.asarray(values, dtype=float)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def rescale(prob_rows, k):
    """Re-temper calibrated probabilities: p' proportional to p**k, i.e. temperature T/k (k < 1 softens)."""
    logp = np.log(np.clip(prob_rows, 1e-12, 1)) * k
    logp -= logp.max(axis=1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(axis=1, keepdims=True)


def split_data(split):
    t1, t2 = load(f"tier1_{split}.jsonl"), load(f"tier2_{split}.jsonl")
    ids = sorted(t1)
    options = list(t1[ids[0]]["probabilities"])
    probs = np.array([[t1[i]["probabilities"][o] for o in options] for i in ids])
    gold = np.array([options.index(t1[i]["gold"]) for i in ids])
    return {
        "ids": ids, "options": options, "probs": probs, "gold": gold, "conf": probs.max(axis=1), "pred1": probs.argmax(axis=1),
        "pred2": np.array([options.index(t2[i]["answer"]) if i in t2 else -1 for i in ids]),
        "lat1": np.array([t1[i]["latency_s"] for i in ids]),
        "lat2": np.array([t2[i]["latency_s"] if i in t2 else np.nan for i in ids]),
        "forced": np.array([bool(t2[i].get("forced")) if i in t2 else False for i in ids]),
    }


def cascade(d, theta):
    """Accuracy and cost if everything below `theta` went to the reasoner; None if the reasoner did not run on all of them."""
    esc = d["conf"] < theta
    if (esc & (d["pred2"] < 0)).any():
        return None
    ok = (np.where(esc, d["pred2"], d["pred1"]) == d["gold"]).astype(float)
    lat = d["lat1"] + np.where(esc, np.nan_to_num(d["lat2"]), 0)
    return {"theta": theta, "escalated": float(esc.mean()), "accuracy": float(ok.mean()), "ok": ok,
            "mean_latency_s": float(lat.mean()), "gpu_seconds_per_1000": float(lat.mean() * 1000)}


def fit_scale(d):
    """k minimising NLL on the fit split, backed off until mean confidence is at most accuracy (prefer under-confidence)."""
    ks = np.round(np.arange(0.2, 1.6, 0.02), 2)
    acc = float((d["pred1"] == d["gold"]).mean())
    rows = np.arange(len(d["gold"]))
    k = min(ks, key=lambda k: -np.log(rescale(d["probs"], k)[rows, d["gold"]] + 1e-12).mean())
    while k > ks[0] and rescale(d["probs"], k).max(axis=1).mean() > acc:
        k = round(k - 0.02, 2)
    return float(k)


def calib(d, k=1.0):
    p = rescale(d["probs"], k) if k != 1.0 else d["probs"]
    conf, ok = p.max(axis=1), (p.argmax(axis=1) == d["gold"]).astype(float)
    e, bins = ece(conf, ok)
    brier = float(np.mean(np.sum((p - np.eye(p.shape[1])[d["gold"]]) ** 2, axis=1)))
    return {"accuracy": float(ok.mean()), "ece": e, "brier": brier, "mean_confidence": float(conf.mean()), "bins": bins}


def strip(r):
    return {k: v for k, v in r.items() if k != "ok"}


def main():
    fit, test = split_data("fit"), split_data("test")
    out = {"service_temperature": SERVICE_T, "n_fit": len(fit["ids"]), "n_test": len(test["ids"])}
    out["tier1_fit"], out["tier1_test"] = calib(fit), calib(test)
    k = fit_scale(fit)
    out["refit"] = {"scale_k": k, "temperature": SERVICE_T / k, "fit": calib(fit, k), "test": calib(test, k)}
    out["tier1_test"]["ci95"] = boot_ci((test["pred1"] == test["gold"]).astype(float))
    out["latency_tier1_ms"] = {"p50": float(np.median(test["lat1"]) * 1000), "p95": float(np.percentile(test["lat1"], 95) * 1000)}

    out["sweep"] = {name: [strip(r) for r in (cascade(d, t) for t in THETAS) if r] for name, d in (("fit", fit), ("test", test))}
    fit_rows = out["sweep"]["fit"]
    best = max(r["accuracy"] for r in fit_rows)
    chosen = min((r for r in fit_rows if r["accuracy"] >= best - 0.005), key=lambda r: (r["escalated"], r["theta"]))
    out["chosen_theta"] = chosen["theta"]

    final = {}
    for label, theta in (("never (tier 1 only)", 0.0), (f"{SHIPPED_THETA} (shipped default)", SHIPPED_THETA),
                         (f"{chosen['theta']} (chosen on the fit split)", chosen["theta"])):
        r = cascade(test, theta)
        if r:
            final[label] = strip(r) | {"ci95": boot_ci(r["ok"])}
    out["test_at"] = final

    low = (test["conf"] < SWEEP_MAX) & (test["pred2"] >= 0)
    if low.any():
        out["reasoner_on_low_confidence_test"] = {
            "n": int(low.sum()), "tier1_accuracy": float((test["pred1"] == test["gold"])[low].mean()),
            "reasoner_accuracy": float((test["pred2"] == test["gold"])[low].mean()),
            "reasoner_mean_s": float(np.nanmean(test["lat2"][low])), "forced_answer_share": float(test["forced"][low].mean())}
    pipe = load("pipeline_test.jsonl")
    if pipe:
        ok = np.array([r["answer"] == r["gold"] for r in pipe.values()])
        out["pipeline_test"] = {"theta": next(iter(pipe.values())).get("theta"), "n": len(pipe), "accuracy": float(ok.mean()), "ci95": boot_ci(ok),
                                "escalated": float(np.mean([r.get("source") == "reasoner" for r in pipe.values()]))}

    with open(os.path.join(HERE, "results", "report.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    write_markdown(out)
    with open(os.path.join(HERE, "results", "REPORT.md"), encoding="utf-8") as f:
        print(f.read())


def pct(x):
    return f"{100 * x:.1f}%"


def write_markdown(o):
    L = ["# Applied evaluation: consumer complaint routing", "",
         f"{o['n_fit']} fit tickets and {o['n_test']} test tickets, 8 teams, the same number per team (see DATASET.md). "
         f"Service temperature T = {o['service_temperature']:.2f} (fit on ARC).", "",
         "## Tier 1 (one forward pass)", "", "| | accuracy | ECE | Brier | mean confidence |", "|---|---|---|---|---|"]
    for name, key in (("fit", "tier1_fit"), ("test", "tier1_test")):
        c = o[key]
        L.append(f"| {name}, ARC temperature | {pct(c['accuracy'])} | {c['ece']:.3f} | {c['brier']:.3f} | {pct(c['mean_confidence'])} |")
    r = o["refit"]
    for name in ("fit", "test"):
        c = r[name]
        L.append(f"| {name}, refit T = {r['temperature']:.2f} | {pct(c['accuracy'])} | {c['ece']:.3f} | {c['brier']:.3f} | "
                 f"{pct(c['mean_confidence'])} |")
    lo, hi = o["tier1_test"]["ci95"]
    L += ["", f"Test accuracy 95% bootstrap interval: {pct(lo)} to {pct(hi)}. Tier 1 latency p50 {o['latency_tier1_ms']['p50']:.0f} ms, "
          f"p95 {o['latency_tier1_ms']['p95']:.0f} ms.", "", "Reliability (test, ARC temperature):", "",
          "| confidence bin | n | mean confidence | accuracy |", "|---|---|---|---|"]
    L += [f"| {b['bin']} | {b['n']} | {pct(b['confidence'])} | {pct(b['accuracy'])} |" for b in o["tier1_test"]["bins"]]
    L += ["", "## Escalation threshold", "",
          "| theta | fit: escalated | fit: accuracy | test: escalated | test: accuracy | test: mean s/decision |", "|---|---|---|---|---|---|"]
    test_by = {x["theta"]: x for x in o["sweep"]["test"]}
    for f in o["sweep"]["fit"]:
        t = test_by.get(f["theta"])
        if t:
            L.append(f"| {f['theta']:.2f} | {pct(f['escalated'])} | {pct(f['accuracy'])} | {pct(t['escalated'])} | {pct(t['accuracy'])} | "
                     f"{t['mean_latency_s']:.2f} |")
    L += ["", f"Chosen on the fit split: theta = {o['chosen_theta']} (the least escalation within half a point of the best fit accuracy).",
          "", "## Test split, scored once", "",
          "| threshold | escalated | accuracy (95% interval) | mean s/decision | GPU-seconds per 1,000 decisions |", "|---|---|---|---|---|"]
    for name, x in o["test_at"].items():
        L.append(f"| {name} | {pct(x['escalated'])} | {pct(x['accuracy'])} ({pct(x['ci95'][0])} to {pct(x['ci95'][1])}) | "
                 f"{x['mean_latency_s']:.2f} | {x['gpu_seconds_per_1000']:.0f} |")
    if "reasoner_on_low_confidence_test" in o:
        x = o["reasoner_on_low_confidence_test"]
        L += ["", f"On the {x['n']} test tickets with tier-1 confidence below {SWEEP_MAX}: tier 1 scores {pct(x['tier1_accuracy'])}, "
              f"the reasoner {pct(x['reasoner_accuracy'])} ({x['reasoner_mean_s']:.1f} s each; "
              f"{pct(x['forced_answer_share'])} needed a forced answer)."]
    if "pipeline_test" in o:
        x = o["pipeline_test"]
        L += ["", "## Through the full pipeline (Redis, deciders, per-team streams)", "",
              f"{x['n']} test tickets routed at theta {x['theta']}: accuracy {pct(x['accuracy'])} "
              f"({pct(x['ci95'][0])} to {pct(x['ci95'][1])}), {pct(x['escalated'])} escalated."]
    with open(os.path.join(HERE, "results", "REPORT.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
