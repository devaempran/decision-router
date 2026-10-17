# eval: the router on real complaints

Runs public, labelled consumer complaints through the real service and pipeline, then reports accuracy, calibration, escalation rate and
speed, refits the escalation threshold, and compares against a hosted model. Results: [`results/REPORT.md`](results/REPORT.md) and
[`results/COMPARISON.md`](results/COMPARISON.md). What the data is and what it is not: [`DATASET.md`](DATASET.md).

```
prepare.py        download the complaints, write data/tickets.jsonl (fit and test splits)
collect.py        tickets -> decisionLM /api/decide, one JSON line per answer (resumable)
report.py         accuracy, calibration, threshold sweep, test scored once -> results/report.json, REPORT.md
run_pipeline.py   the test split through Redis, the deciders and decisionLM (the end-to-end check)
hybrid.py         tier 1 decides, Haiku escalates: accuracy, decider and overall latency, cost -> results/HYBRID.md
hosted.py         the same tickets through a prompted Claude call (needs ANTHROPIC_API_KEY)
compare.py        local vs hosted: accuracy, latency, calibration, cost -> results/COMPARISON.md
tests/            the harness's own logic, with no model or network
```

## Findings

320 test complaints, 8 teams (full tables: [`results/REPORT.md`](results/REPORT.md), [`results/COMPARISON.md`](results/COMPARISON.md)).
The intervals are 95% bootstrap intervals; with 320 tickets they are about plus or minus 4.5 points, so read differences under 5 points as unproven.

| | accuracy | latency p50 | cost per 1,000 decisions |
|---|---|---|---|
| local tier 1 (Qwen3-8B Q8_0) | 75.0% (70.3 to 79.7) | 131 ms | 141 GPU-seconds, no per-call fee |
| local two-tier at the shipped 0.85 | 73.1% (68.4 to 78.1) | 149 ms (p95 31 s) | 5,236 GPU-seconds |
| Claude Haiku 5.5, prompted | 80.3% (75.9 to 84.7) | 927 ms | $0.10 |
| Claude Sonnet 5.5, prompted | 81.6% (77.5 to 85.6) | 1,345 ms | $1.96 |

1. **Tier 1 transfers, its calibration does not.** Accuracy is 75.0% and the ARC-fit temperature leaves it over-confident: mean confidence 86%
   against 75% accuracy, ECE 0.118. Refitting the temperature on the fit split (T = 9.62, was 7.50) brings test ECE to 0.032.
2. **The reasoner did not help on this task.** On the 123 test tickets under 0.9 confidence, tier 1 is right 60.2% of the time and the reasoner 51.2%,
   at 19 s each. The shipped 0.85 threshold sends 24% of tickets to it and lowers accuracy (73.1%, within noise of 75.0%) at 37 times the GPU time.
   The threshold refit on the fit split is 0.5: 3% escalated, the same accuracy as tier 1 alone. For this kind of text, run tier 1 only, or leave
   tier 2 for a task where it measurably helps (it did on ARC: 0.902 to 0.940).
3. **The hosted models are more accurate.** Haiku 5.5 scores 80.3% and Sonnet 5.5 81.6%, about 5 to 6 points above tier 1, with intervals that
   overlap but do not coincide. Tier 1 is about seven times faster than the fastest hosted call (131 ms against 927 ms, one request at a time,
   hosted time includes the network) and has no per-call fee. Haiku is already only $0.10 per 1,000 decisions, so on cost alone the local
   setup does not win at this volume; what it offers is latency, no data leaving the machine, and swappable models.
4. **Hosted models state better-calibrated confidence than tier 1 with the ARC temperature** (ECE 0.055 Haiku, 0.021 Sonnet, against 0.118), but a temperature refit
   on a few hundred labelled tickets closes the gap for tier 1 (0.032). The hosted confidence is what the model says, with no refit possible.
5. **The pipeline matches the simulation.** All 320 test tickets went through Redis, the deciders and the service at theta 0.5: 320 routed,
   0 dead-lettered, 0 missing, 75.0% accuracy, 3.1% escalated, the same as the offline numbers.

6. **Tier 1 deciding with Haiku as the escalation** ([`results/HYBRID.md`](results/HYBRID.md)). Escalating tickets under 0.85 confidence (24% of them)
   to Haiku lifts accuracy from 75.0% to 77.5% (73.1 to 82.2): decider latency stays 141 ms, overall average latency is 395 ms (p95 1.4 s) and the API
   cost is $0.024 per 1,000 decisions. On the 76 low-confidence tickets Haiku is right 65.8% of the time against tier 1's 55.3%. The gain is within noise
   at this sample size, and it is gradual: tier-1 confidence does not isolate the errors sharply enough, so reaching Haiku's own 80.3% needs
   escalating about 90% of tickets (threshold 0.95, the one the fit split picks), which is Haiku-only in practice. The hybrid is a latency and
   cost saving for a modest accuracy cost against Haiku alone, not an accuracy gain.

Caveats: class-balanced sampling, not live traffic; regulator complaints, not helpdesk tickets ([`DATASET.md`](DATASET.md)); one run per
configuration; the hosted prompt is one plain prompt that was not tuned (thinking off, low effort), and a tuned prompt, few-shot examples or higher
effort might score higher; tier 1 latency here is longer than the 45 ms ARC figure because complaints are far longer than ARC questions.

## Reproduce

Needs the service set up on a GPU ([`../service/README.md`](../service/README.md)) and Redis on port 6380.

```
pip install numpy pandas redis
python eval/prepare.py

python service/run.py &                    # port 5000; the commands below use PORT=5001 as in the runs reported
python eval/collect.py --split fit  --theta 0 --out results/raw/tier1_fit.jsonl
python eval/collect.py --split fit  --theta 2 --out results/raw/tier2_fit.jsonl --only-below 0.9 --tier1 results/raw/tier1_fit.jsonl
python eval/collect.py --split test --theta 0 --out results/raw/tier1_test.jsonl
python eval/collect.py --split test --theta 2 --out results/raw/tier2_test.jsonl --only-below 0.9 --tier1 results/raw/tier1_test.jsonl
python eval/run_pipeline.py --theta 0.85   # optional, about 15 minutes
python eval/report.py
```

Run `collect.py` one at a time: the service handles a single request at a time, so two collectors starve each other.

Why two passes: `theta 0` never escalates, so tier 1 sees every ticket quickly; `theta 2` always escalates, so the reasoner runs only on the
tickets that could ever be escalated (confidence below 0.9). Any threshold up to 0.9 is then simulated exactly from the two files.

## Hosted comparison

```
export ANTHROPIC_API_KEY=...
python eval/hosted.py --model claude-haiku-5-5
python eval/hosted.py --model claude-sonnet-5-5
python eval/compare.py
```

One plain request per test ticket (the complaint and the same eight team descriptions as lettered options, JSON answer with a stated
confidence, thinking off, low effort). `--max-calls` caps spending (default 400 calls); results are cached on disk so a re-run does not pay
twice. Prices in `hosted.py` are the list prices on the date noted there; update them if they have changed.
