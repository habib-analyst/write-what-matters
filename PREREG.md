# PREREG — write-what-matters (project #90)

Locked 2026-10-28, BEFORE the head-to-head run. Bars are frozen;
no bar-moving after seeing head-to-head results.

## Target claim (from the live arXiv 2610.34180 abstract, context only)

Density-dependent reversal in recurrent-memory write routing: with
4 slots per write, always-write beats learned selective writing on a
dense task, while selective writing beats always-write on a sparse
task. The paper's figures (0.472 vs 0.337 dense; 0.222 vs 0.048
sparse; 10 seeds) are CONTEXT ONLY and are not the bars below: this
is a clean-room re-implementation with its own scale, so bars are
directional with a minimum margin, not copies of the paper's numbers.

## Locked configuration (src/memory.py defaults)

- DIM 64, N_SLOTS 128, K_WRITE 4, TAU 0.30 (cosine threshold)
- Sequence length 200; dense fact_prob 0.70; sparse fact_prob 0.08
- 16 distractor prototypes (coherent repeated filler)
- Gate: fact scores N(0.68, 0.22), distractor scores N(0.32, 0.22),
  write iff score >= 0.5 (noisy salience proxy for a learned gate;
  NOT trained — TPR/FPR are reported, not tuned post-lock)
- Policies: always, selective, adaptive (descriptive only),
  random (matched-budget content-blind control), oracle (diagnostic)
- Pilot seeds 0-9 (8 sequences/seed) used ONLY for calibration.
  Pilot result at the locked config: dense gap +0.116
  CI [0.105, 0.127]; sparse gap +0.340 CI [0.304, 0.378].
  The locked config was chosen as the most balanced pilot regime
  (largest minimum headline gap across the sweep), not the largest
  single gap.

## Head-to-head protocol (fresh seeds, untouched by pilot/tests)

- Seeds 1000-1009, 20 sequences/seed, 200 sequences per regime.
- Primary unit: sequence, paired across policies (identical
  generated sequence, token vectors, and gate scores per policy).
- Statistics: paired bootstrap 95% CI, B = 10,000, seed 12345,
  percentile method over sequence indices.

## Frozen bars

- G2 dense: always − selective >= +0.05 AND 95% CI lower bound > 0.
- G2 sparse: selective − always >= +0.05 AND 95% CI lower bound > 0.
- G3 (mechanism): sparse selective − random >= +0.05 AND 95% CI
  lower bound > 0. Random writes the same count as selective per
  sequence but content-blind; if selective's sparse win were merely
  writing less, random would match it. (Dense selective − random
  reported descriptively; the sparse arm is the load-bearing one
  because that is where filtering is claimed to help.)
- Adaptive and oracle arms are descriptive only, not gated. In
  particular the paper's "adaptive underperforms fixed" is NOT a
  bar: the pilot already shows adaptive ≈ selective in this model,
  and that divergence will be reported honestly if it persists.
