# Hybrid: local tier 1 decides, Claude Haiku 5.5 takes the uncertain tickets

320 test complaints, 8 teams. Haiku is the same plain prompted call as in COMPARISON.md. Every ticket goes to tier 1 first;
tickets below the threshold are then sent to Haiku too, so an escalated ticket's latency is tier 1 plus the Haiku call.

## Threshold sweep

| theta | fit: escalated | fit: accuracy | test: escalated | test: accuracy | test: overall avg latency | test: $ per 1,000 |
|---|---|---|---|---|---|---|
| 0.00 | 0.0% | 78.1% | 0.0% | 75.0% | 141 ms | $0.000 |
| 0.50 | 2.8% | 79.1% | 3.1% | 75.3% | 178 ms | $0.003 |
| 0.60 | 4.7% | 79.4% | 6.9% | 76.2% | 219 ms | $0.007 |
| 0.70 | 7.8% | 80.3% | 10.0% | 76.9% | 250 ms | $0.010 |
| 0.75 | 10.9% | 81.2% | 14.4% | 77.2% | 293 ms | $0.014 |
| 0.80 | 14.4% | 81.9% | 18.1% | 76.9% | 335 ms | $0.018 |
| 0.85 | 20.3% | 83.1% | 23.8% | 77.5% | 395 ms | $0.024 |
| 0.90 | 35.6% | 83.8% | 38.4% | 77.8% | 537 ms | $0.038 |
| 0.95 | 92.5% | 85.3% | 90.6% | 80.3% | 1048 ms | $0.090 |
| all to Haiku | 100.0% | 85.3% | 100.0% | 80.3% | 1137 ms | $0.099 |

Chosen on the fit split: theta = 0.95.

## Test split, scored once

| setup | escalated | accuracy (95% interval) | decider avg latency | overall avg latency | overall p95 | $ per 1,000 decisions |
|---|---|---|---|---|---|---|
| tier 1 only | 0.0% | 75.0% (70.3% to 79.7%) | 141 ms | 141 ms | 194 ms | $0.000 |
| hybrid at 0.85 | 23.8% | 77.5% (73.1% to 82.2%) | 141 ms | 395 ms | 1397 ms | $0.024 |
| hybrid at 0.9 | 38.4% | 77.8% (73.4% to 82.2%) | 141 ms | 537 ms | 1425 ms | $0.038 |
| hybrid at 0.95 | 90.6% | 80.3% (75.9% to 84.7%) | 141 ms | 1048 ms | 1510 ms | $0.090 |
| Haiku only | 100.0% | 80.3% (75.9% to 84.7%) | 141 ms | 1137 ms | 1512 ms | $0.099 |

On the 76 test tickets below 0.85 confidence: tier 1 scores 55.3%, Haiku 65.8%.

Decider latency is the tier-1 average over every ticket (the cost every decision pays). Overall latency adds the Haiku call for escalated tickets.
The Haiku-only row here still pays tier 1 first; Haiku called directly has a median of 927 ms (COMPARISON.md).
