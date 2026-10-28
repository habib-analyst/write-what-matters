"""Pilot sweep: find a regime where the density reversal appears.

Pilot seeds 0-9, 8 sequences/seed. Prints per-regime mean accuracy
per policy and paired bootstrap CIs for the two headline gaps.
Writes results/pilot_rows.jsonl. TAU / N_SLOTS are swept by
monkeypatching module globals before running (they are read at
call time)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory  # noqa: E402

POLICIES = ["always", "selective", "adaptive", "random", "oracle"]


def paired_ci(a, b, n_boot=2000, seed=7):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def run_config(tau, n_slots, length=200, dense_p=0.70, sparse_p=0.08):
    memory.TAU = tau
    memory.N_SLOTS = n_slots
    out = {}
    for regime, p in (("dense", dense_p), ("sparse", sparse_p)):
        rows = []
        for seed in range(10):
            for seq in range(8):
                rows.append(memory.run_sequence(seed, seq, length, p, POLICIES))
        means = {pol: float(np.mean([r[pol] for r in rows])) for pol in POLICIES}
        if regime == "dense":
            gap, ci = paired_ci([r["always"] for r in rows], [r["selective"] for r in rows])
        else:
            gap, ci = paired_ci([r["selective"] for r in rows], [r["always"] for r in rows])
        out[regime] = {"means": means, "headline_gap": gap, "ci": ci,
                       "tpr": float(np.mean([r["_tpr"] for r in rows])),
                       "fpr": float(np.mean([r["_fpr"] for r in rows])),
                       "n_seq": len(rows)}
        out[regime]["rows"] = rows
    return out


def main():
    results = {}
    for tau, n_slots in [(0.35, 64), (0.30, 64), (0.45, 64), (0.35, 128), (0.25, 64)]:
        cfg = run_config(tau, n_slots)
        key = f"tau={tau}_slots={n_slots}"
        results[key] = {reg: {k: v for k, v in d.items() if k != "rows"}
                        for reg, d in cfg.items()}
        print(f"== {key} ==")
        for reg, d in cfg.items():
            print(f"  {reg}: " + " ".join(f"{p}={d['means'][p]:.3f}" for p in POLICIES)
                  + f" | headline_gap={d['headline_gap']:.3f} CI=[{d['ci'][0]:.3f},{d['ci'][1]:.3f}]"
                  + f" | TPR={d['tpr']:.3f} FPR={d['fpr']:.3f}")
        Path("results/pilot_rows.jsonl").parent.mkdir(exist_ok=True)
        with open("results/pilot_rows.jsonl", "a") as f:
            for reg, d in cfg.items():
                for r in d["rows"]:
                    f.write(json.dumps({"config": key, "regime": reg, **r}) + "\n")
    Path("results/pilot_summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
