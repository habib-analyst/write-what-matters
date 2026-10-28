import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import memory  # noqa: E402


def test_slots_distinct_in_range_and_deterministic():
    s1 = memory._slots_for("fact-1-0-3", 4)
    s2 = memory._slots_for("fact-1-0-3", 4)
    assert s1 == s2 and len(set(s1)) == 4
    assert all(0 <= s < memory.N_SLOTS for s in s1)


def test_prototype_coherence_same_slots():
    tokens, _ = memory.generate_sequence(0, 0, 200, 0.08)
    by_id = {}
    for t in tokens:
        if not t.is_fact:
            by_id.setdefault(t.token_id, set()).add(tuple(t.slots))
    assert by_id and all(len(v) == 1 for v in by_id.values())


def test_sequence_determinism():
    a = memory.run_sequence(3, 2, 200, 0.70, ["always", "selective"])
    b = memory.run_sequence(3, 2, 200, 0.70, ["always", "selective"])
    assert a == b


def test_recall_bounded_and_gate_rates_sane():
    out = memory.run_sequence(0, 0, 200, 0.70, ["always", "selective", "oracle"])
    for pol in ("always", "selective", "oracle"):
        assert 0.0 <= out[pol] <= 1.0
    assert 0.5 < out["_tpr"] < 1.0
    assert 0.0 < out["_fpr"] < 0.5


def test_invalid_policy_raises():
    tokens, fact_ids = memory.generate_sequence(0, 0, 50, 0.5)
    with pytest.raises(ValueError):
        memory.run_policy(tokens, fact_ids, "nope", np.random.default_rng(0))


def test_missed_fact_unrecallable():
    # Gate threshold above every score -> selective recalls nothing.
    old = memory.GATE_THRESHOLD
    memory.GATE_THRESHOLD = 10.0
    try:
        out = memory.run_sequence(0, 0, 200, 0.70, ["selective"])
        assert out["selective"] == 0.0
    finally:
        memory.GATE_THRESHOLD = old
