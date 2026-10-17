# Local two-tier setup vs hosted models

320 test complaints, 8 teams. Hosted rows: one plain prompted call per ticket (hosted.py), list prices of 2026-10-06. Latency is wall-clock per decision on one RTX 5060 Ti (local) or over the network (hosted), one request at a time.

| setup | accuracy (95% interval) | latency p50 | latency p95 | calibration error of its own stated confidence | cost |
|---|---|---|---|---|---|
| local, tier 1 only | 75.0% (70.3% to 79.7%) | 131 ms | 194 ms | 0.118 | 141 GPU-s per 1,000 |
| local, two-tier at 0.85 | 73.1% (68.4% to 78.1%) | 149 ms | 31094 ms | n/a (the reasoner states no confidence) | 5236 GPU-s per 1,000 |
| local, two-tier at 0.5 | 75.0% (70.3% to 79.7%) | 132 ms | 203 ms | n/a (the reasoner states no confidence) | 912 GPU-s per 1,000 |
| claude-haiku-5-5 (prompted) | 80.3% (75.9% to 84.7%) | 927 ms | 1322 ms | 0.055 | $0.10 per 1,000 |
| claude-sonnet-5-5 (prompted) | 81.6% (77.5% to 85.6%) | 1345 ms | 1833 ms | 0.021 | $1.96 per 1,000 |
