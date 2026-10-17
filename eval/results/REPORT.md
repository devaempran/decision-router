# Applied evaluation: consumer complaint routing

320 fit tickets and 320 test tickets, 8 teams, the same number per team (see DATASET.md). Service temperature T = 7.50 (fit on ARC).

## Tier 1 (one forward pass)

| | accuracy | ECE | Brier | mean confidence |
|---|---|---|---|---|
| fit, ARC temperature | 78.1% | 0.098 | 0.361 | 87.5% |
| test, ARC temperature | 75.0% | 0.118 | 0.415 | 86.4% |
| fit, refit T = 9.62 | 78.1% | 0.041 | 0.349 | 77.4% |
| test, refit T = 9.62 | 75.0% | 0.032 | 0.398 | 76.3% |

Test accuracy 95% bootstrap interval: 70.3% to 79.7%. Tier 1 latency p50 131 ms, p95 194 ms.

Reliability (test, ARC temperature):

| confidence bin | n | mean confidence | accuracy |
|---|---|---|---|
| 0.3-0.4 | 2 | 38.5% | 50.0% |
| 0.4-0.5 | 8 | 44.7% | 50.0% |
| 0.5-0.6 | 12 | 53.8% | 33.3% |
| 0.6-0.7 | 10 | 64.5% | 40.0% |
| 0.7-0.8 | 26 | 75.1% | 61.5% |
| 0.8-0.9 | 65 | 86.6% | 69.2% |
| 0.9-1.0 | 197 | 93.1% | 84.3% |

## Escalation threshold

| theta | fit: escalated | fit: accuracy | test: escalated | test: accuracy | test: mean s/decision |
|---|---|---|---|---|---|
| 0.00 | 0.0% | 78.1% | 0.0% | 75.0% | 0.14 |
| 0.50 | 2.8% | 79.4% | 3.1% | 75.0% | 0.91 |
| 0.60 | 4.7% | 79.1% | 6.9% | 75.6% | 2.18 |
| 0.70 | 7.8% | 78.1% | 10.0% | 75.6% | 2.77 |
| 0.75 | 10.9% | 78.1% | 14.4% | 75.6% | 3.78 |
| 0.80 | 14.4% | 77.2% | 18.1% | 74.4% | 4.34 |
| 0.85 | 20.3% | 77.5% | 23.8% | 73.1% | 5.24 |
| 0.90 | 35.6% | 76.6% | 38.4% | 71.6% | 7.59 |

Chosen on the fit split: theta = 0.5 (the least escalation within half a point of the best fit accuracy).

## Test split, scored once

| threshold | escalated | accuracy (95% interval) | mean s/decision | GPU-seconds per 1,000 decisions |
|---|---|---|---|---|
| never (tier 1 only) | 0.0% | 75.0% (70.3% to 79.7%) | 0.14 | 141 |
| 0.85 (shipped default) | 23.8% | 73.1% (68.4% to 78.1%) | 5.24 | 5236 |
| 0.5 (chosen on the fit split) | 3.1% | 75.0% (70.3% to 79.7%) | 0.91 | 912 |

On the 123 test tickets with tier-1 confidence below 0.9: tier 1 scores 60.2%, the reasoner 51.2% (19.4 s each; 0.0% needed a forced answer).

## Through the full pipeline (Redis, deciders, per-team streams)

320 test tickets routed at theta 0.5: accuracy 75.0% (70.0% to 79.7%), 3.1% escalated.
