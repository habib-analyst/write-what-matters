"""Head-to-head run — PREREG-locked. Fresh seeds 1000-1009 only.

Writes results/head_to_head_rows.jsonl and results/head_to_head.json.
Re-running reproduces the rows file byte-for-byte (see README).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory  # noqa: E402

POLICIES = ["always", "selective", "adaptive", "random", "oracle"]
SEEDS = range(1000, 1010)
SEQS_PER_SEED = 20
REGIMES = {"dense": 0.70, "sparse": 0.08}


def paired_ci(a, b, n_boot=10000, seed=12345):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return {"mean": float(d.mean()),
            "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def main():
    assert memory.N_SLOTS == 128 and memory.TAU == 0.30 and memory.K_WRITE == 4, \
        "locked config drifted — do not run head-to-head off-PREREG"
    root = Path(__file__).resolve().parents[1]
    outdir = root / "results"
    outdir.mkdir(exist_ok=True)
    summary = {"config": {"n_slots": memory.N_SLOTS, "tau": memory.TAU,
                           "k_write": memory.K_WRITE, "length": 200,
                           "dense_p": 0.70, "sparse_p": 0.08,
                           "seeds": "1000-1009", "seqs_per_seed": SEQS_PER_SEED},
               "regimes": {}}
    with open(outdir / "head_to_head_rows.jsonl", "w") as f:
        for regime, p in REGIMES.items():
            rows = []
            for seed in SEEDS:
                for seq in range(SEQS_PER_SEED):
                    r = memory.run_sequence(seed, seq, 200, p, POLICIES)
                    rows.append(r)
                    f.write(json.dumps({"regime": regime, "seed": seed, "seq": seq, **r}) + "\n")
            reg = {"n_sequences": len(rows),
                   "means": {pol: float(np.mean([r[pol] for r in rows])) for pol in POLICIES},
                   "tpr": float(np.mean([r["_tpr"] for r in rows])),
                   "fpr": float(np.mean([r["_fpr"] for r in rows])),
                   "gaps": {}}
            if regime == "dense":
                reg["gaps"]["always_minus_selective"] = paired_ci(
                    [r["always"] for r in rows], [r["selective"] for r in rows])
            else:
                reg["gaps"]["selective_minus_always"] = paired_ci(
                    [r["selective"] for r in rows], [r["always"] for r in rows])
                reg["gaps"]["selective_minus_random"] = paired_ci(
                    [r["selective"] for r in rows], [r["random"] for r in rows])
                reg["gaps"]["oracle_minus_selective"] = paired_ci(
                    [r["oracle"] for r in rows], [r["selective"] for r in rows])
            reg["gaps"]["selective_minus_random_dense_desc"] = paired_ci(
                [r["selective"] for r in rows], [r["random"] for r in rows]) if regime == "dense" else None
            summary["regimes"][regime] = reg
            print(regime, json.dumps(reg, indent=2))
    (outdir / "head_to_head.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
