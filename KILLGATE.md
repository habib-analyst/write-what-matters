# Kill-gate record — write-what-matters (project #90)

## Abstract read (fresh, 2026-10-28)

arXiv 2610.34180, "Write What Matters: Disentangling Token Selection
from Slot Allocation in Recurrent Memory", Chen Chen and Rui Cheng,
submitted 25 Oct 2026. Abstract confirms: Factorized Write Routing
(FWR); 4 slots per write; 10 seeds; two synthetic tasks querying
facts after long contexts; dense Feature Recall always-write 0.472
vs selective 0.337; sparse-fact Passkey selective 0.222 vs
always-write 0.048; adaptive rule underperforms fixed allocation
under the learned gate on both tasks; labelled oracle reaches 0.372
on Passkey. Watch-note caution carried forward: absolute accuracies
are low and the adaptive rule underperforms — pilot-calibrate
carefully, do not inflate.

## G0 — harness: PASS

Vector-sum slot memory (src/memory.py), seeded/deterministic,
dense vs sparse regimes, always/selective/adaptive/random/oracle
policies. 6/6 pytest green before head-to-head.

## G1 — author-code recheck: PASS (absence, dated 2026-10-28)

- arXiv abs page (fetched fresh 2026-10-28): no author code link;
  only generic "Code, Data and Media" toggles with no repo.
- Web searches 2026-10-28: exact title + "Factorized Write Routing";
  exact title + "code github"; arXiv ID + "code repository";
  github + "Factorized Write Routing" / title / authors — no author
  repository found (hits were unrelated papers and generic pages).
- Caveat (as with #89): this is a web-search absence 3 days after
  submission, not an exhaustive proof; re-check before push. No
  author code was used; the harness is clean-room from the abstract.

## G2 — density-dependent reversal: PASS (head-to-head, fresh seeds 1000-1009, 200 sequences/regime)

- Dense (fact_prob 0.70): always 0.918 vs selective 0.790;
  always − selective = **+0.128, 95% CI [0.121, 0.135]**
  (bar: >= +0.05, CI excluding 0).
- Sparse (fact_prob 0.08): selective 0.780 vs always 0.430;
  selective − always = **+0.351, 95% CI [0.329, 0.372]**
  (bar: >= +0.05, CI excluding 0).
- Pilot-to-head-to-head consistency: locked-config pilot gaps were
  +0.116 dense / +0.340 sparse — no drift, no bar-moving.

## G3 — routing mechanism (not a confound): PASS

- Sparse selective − random (matched write-count, content-blind) =
  **+0.520, 95% CI [0.500, 0.540]** (bar: >= +0.05, CI excluding 0).
  Writing less alone does not explain the sparse win; the gate must
  select facts. Dense selective − random = +0.182 [0.176, 0.188]
  (descriptive).
- Diagnostic oracle (write iff fact): 1.000 in both regimes;
  oracle − selective = +0.220 [0.205, 0.234] sparse — headroom in
  filtering, mirroring the paper's oracle diagnostic directionally.
- Divergence reported honestly: adaptive slot-count (k = 2/8 under
  the gate) did NOT underperform fixed k = 4 here (dense 0.782 vs
  0.790; sparse 0.782 vs 0.780), unlike the paper's learned adaptive
  rule. Adaptive was descriptive only, never a bar, and this model
  difference is stated in PAPER.md — it is not hidden.

## Determinism / tests

- 6/6 pytest green. Head-to-head rerun reproduces
  results/head_to_head_rows.jsonl byte-for-byte, SHA-256
  894cdd73a661d0d6d92116dc8fcecdeaaf0f6f5f701b968157ebf71faa3c700b.
- Compute: $0, CPU only, no downloads, no model runs.
