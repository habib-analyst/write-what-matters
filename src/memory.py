"""Clean-room vector-sum slot memory for the write-routing claim.

NOT the authors' code or architecture (none found; see KILLGATE.md).
A scoped re-implementation of the *claim* in arXiv 2610.34180:
recurrent memory with a fixed number of slots, each token written to
k = 4 slots, queries reading all slots. Slot contents are vector
sums; a fact is recalled iff the cosine between its vector and the
mean of its k slots clears a threshold tau.

Key modelling choices (fixed before head-to-head, see PREREG.md):
- Facts are unique unit vectors; each fact is written once.
- Distractors are drawn from a small prototype pool (16 prototypes):
  repeated filler, so the same distractor vector lands in the same
  slots repeatedly and interferes *coherently*, while distinct facts
  interfere incoherently. This is the standard synthetic-task
  construction for "distractors dominate" regimes and is what lets
  density change the sign of the policy gap (see pilot notes).
- Slot assignment is a deterministic hash of the token id
  (stand-in for learned slot addressing); identical tokens always
  hit identical slots.
- The selective gate is a noisy salience proxy for a learned gate:
  fact scores ~ N(0.68, 0.22), distractor scores ~ N(0.32, 0.22),
  write iff score >= 0.5. It is NOT trained; its TPR/FPR are
  consequences of the fixed score distributions, reported in results.
- Policies: always (write everything, k=4), selective (gated, k=4),
  adaptive (gated, k = 2 if score < 0.65 else 8; descriptive only),
  random (content-blind, matched write-count to selective per
  sequence; the G3 control), oracle (write iff fact, k=4; diagnostic
  upper bound, mirroring the paper's labelled-oracle diagnostic).
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

DIM = 64
N_SLOTS = 128  # PREREG-locked 2026-10-28 (see PREREG.md)
K_WRITE = 4
TAU = 0.30  # PREREG-locked 2026-10-28 (see PREREG.md)
N_DISTRACTOR_PROTOTYPES = 16
GATE_THRESHOLD = 0.5
FACT_SCORE_MEAN = 0.68
DISTRACTOR_SCORE_MEAN = 0.32
SCORE_SD = 0.22


def _stable_int(token_id: str) -> int:
    return int(hashlib.sha256(token_id.encode()).hexdigest()[:8], 16)


def _unit_vector(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=DIM)
    return v / np.linalg.norm(v)


def _slots_for(token_id: str, k: int) -> list[int]:
    """k distinct slots, deterministically hashed from the token id."""
    rng = np.random.default_rng(_stable_int(token_id) % (2**32))
    return sorted(rng.choice(N_SLOTS, size=k, replace=False).tolist())


class Token:
    __slots__ = ("token_id", "is_fact", "vector", "score", "slots")

    def __init__(self, token_id, is_fact, vector, score):
        self.token_id = token_id
        self.is_fact = is_fact
        self.vector = vector
        self.score = score
        self.slots = _slots_for(token_id, K_WRITE)


def generate_sequence(seed: int, seq_idx: int, length: int, fact_prob: float):
    """One synthetic long-context sequence. Returns (tokens, fact_ids)."""
    rng = np.random.default_rng((seed * 100003 + seq_idx * 7919) % (2**32))
    # Distractor prototypes are fixed per (seed, seq) draw: coherent pool.
    prototypes = [_unit_vector(rng) for _ in range(N_DISTRACTOR_PROTOTYPES)]
    proto_slots = {}
    tokens: list[Token] = []
    fact_ids: list[str] = []
    for pos in range(length):
        if rng.random() < fact_prob:
            tid = f"fact-{seed}-{seq_idx}-{pos}"
            vec = _unit_vector(rng)
            score = float(rng.normal(FACT_SCORE_MEAN, SCORE_SD))
            tokens.append(Token(tid, True, vec, score))
            fact_ids.append(tid)
        else:
            p = int(rng.integers(0, N_DISTRACTOR_PROTOTYPES))
            tid = f"dist-{seed}-{seq_idx}-p{p}"
            score = float(rng.normal(DISTRACTOR_SCORE_MEAN, SCORE_SD))
            tok = Token(tid, False, prototypes[p], score)
            # All occurrences of a prototype share slots (coherent write).
            if tid not in proto_slots:
                proto_slots[tid] = tok.slots
            tok.slots = proto_slots[tid]
            tokens.append(tok)
    return tokens, fact_ids


def run_policy(tokens, fact_ids, policy: str, rng: np.random.Generator) -> float:
    """Recall accuracy over the sequence's facts under one policy."""
    slots = np.zeros((N_SLOTS, DIM))
    written: dict[str, tuple[Token, list[int]]] = {}
    if policy == "random":
        # Matched-budget content-blind control: same write count as
        # selective would write on this sequence, chosen at random.
        n_sel = sum(1 for t in tokens if t.score >= GATE_THRESHOLD)
        chosen = set(rng.choice(len(tokens), size=n_sel, replace=False).tolist()) if n_sel else set()
    else:
        chosen = set()
    for i, tok in enumerate(tokens):
        if policy == "always":
            do_write, k = True, K_WRITE
        elif policy == "selective":
            do_write, k = tok.score >= GATE_THRESHOLD, K_WRITE
        elif policy == "adaptive":
            if tok.score < GATE_THRESHOLD:
                do_write, k = False, K_WRITE
            else:
                do_write, k = True, (2 if tok.score < 0.65 else 8)
        elif policy == "random":
            do_write, k = i in chosen, K_WRITE
        elif policy == "oracle":
            do_write, k = tok.is_fact, K_WRITE
        else:
            raise ValueError(policy)
        if not do_write:
            continue
        # Content-addressed slots: a deterministic draw per token, so
        # the read always uses exactly the slots the write touched.
        write_slots = tok.slots if k == K_WRITE else _slots_for_k(tok.token_id, k)
        for s in write_slots:
            slots[s] += tok.vector
        written[tok.token_id] = (tok, write_slots)
    if not fact_ids:
        return float("nan")
    hits = 0
    for fid in fact_ids:
        entry = written.get(fid)
        if entry is None:
            continue  # missed write: unrecallable by construction
        tok, write_slots = entry
        read = slots[write_slots].mean(axis=0)
        n = np.linalg.norm(read)
        if n == 0:
            continue
        cos = float(read @ tok.vector / n)  # tok.vector is unit norm
        if cos >= TAU:
            hits += 1
    return hits / len(fact_ids)


def _slots_for_k(token_id: str, k: int) -> list[int]:
    rng = np.random.default_rng((_stable_int(token_id) + 977) % (2**32))
    # Deterministic per token (content-addressed); read uses these slots.
    return sorted(rng.choice(N_SLOTS, size=k, replace=False).tolist())


def run_sequence(seed: int, seq_idx: int, length: int, fact_prob: float, policies):
    tokens, fact_ids = generate_sequence(seed, seq_idx, length, fact_prob)
    rng = np.random.default_rng((seed * 31 + seq_idx * 17 + 5) % (2**32))
    out = {}
    for pol in policies:
        out[pol] = run_policy(tokens, fact_ids, pol, rng)
    # Gate diagnostics (same for all policies on this sequence).
    facts = [t for t in tokens if t.is_fact]
    distractors = [t for t in tokens if not t.is_fact]
    out["_n_facts"] = len(facts)
    out["_tpr"] = float(np.mean([t.score >= GATE_THRESHOLD for t in facts])) if facts else float("nan")
    out["_fpr"] = float(np.mean([t.score >= GATE_THRESHOLD for t in distractors])) if distractors else float("nan")
    return out
