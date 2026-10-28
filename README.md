# write-what-matters — recurrent-memory write routing

Project #90. Scoped CPU replication of the *claim* in arXiv
2610.34180, "Write What Matters: Disentangling Token Selection from
Slot Allocation in Recurrent Memory" (Chen Chen, Rui Cheng, submitted
25 Oct 2026), built clean-room from the live abstract only. No
author code was used (none found as of 2026-10-28); no LLM is used
anywhere; all tasks are synthetic.

## Claim (this replication's, not the paper's)

On synthetic long-context fact-recall tasks with a vector-sum slot
memory (128 slots, 4 slots per write), the better write policy
flips with fact density: always-write wins when facts are dense,
selective (gated) writing wins when facts are sparse — and the
sparse win is driven by content-based routing, not by writing less.

## Results (head-to-head, fresh seeds 1000-1009, 200 sequences/regime)

| Regime | always | selective | adaptive (desc.) | random (control) | oracle (diag.) |
|---|---|---|---|---|---|
| Dense (70% facts) | 0.918 | 0.790 | 0.782 | 0.608 | 1.000 |
| Sparse (8% facts) | 0.430 | 0.780 | 0.782 | 0.260 | 1.000 |

- G2 dense: always − selective = **+0.128, 95% CI [0.121, 0.135]**
- G2 sparse: selective − always = **+0.351, 95% CI [0.329, 0.372]**
- G3 sparse: selective − random = **+0.520, 95% CI [0.500, 0.540]**
- Paired bootstrap CIs, B = 10,000, seed 12345, over sequences.
  Protocol and bars were locked in PREREG.md before head-to-head.

## Reproduce

```bash
python3 -m pytest tests/ -q          # 6/6
python3 src/run_head_to_head.py      # seeds 1000-1009
```

Rerunning reproduces `results/head_to_head_rows.jsonl` byte-for-byte
(SHA-256 `894cdd73a661d0d6d92116dc8fcecdeaaf0f6f5f701b968157ebf71faa3c700b`).

## Honest limits (see PAPER.md for the full list)

Synthetic toy; the gate is a fixed noisy salience proxy, not a
learned gate; slot addressing is a deterministic hash, not learned
addressing; distractor coherence is a modelling choice that enables
the reversal; adaptive slots did NOT underperform here (divergence
from the paper, reported not hidden); the paper's 0.472/0.337 and
0.222/0.048 figures were NOT reproduced and are NOT claimed.

## Layout

- `src/memory.py` — slot memory, sequence generator, policies
- `src/pilot.py` — pilot sweep (seeds 0-9, calibration only)
- `src/run_head_to_head.py` — PREREG-locked head-to-head run
- `tests/` — determinism / invariant tests
- `results/` — pilot + head-to-head rows and summary
- `PREREG.md`, `KILLGATE.md` — frozen protocol and gate record
