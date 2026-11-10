"""B0 — brute force over all field subsets (sufprov's method, design notes), exact reference for M <= ~22.

Computes every within-task pair's Hellinger bounds once (shared core function, sufprov normalisation), encodes each pair's
difference set as a bitmask, and evaluates every subset S by the separation rule of sufprov Prop. 2 / SI (S8):
U(S) = max hi over pairs whose difference set misses S, L(S) = max lo over the same pairs.
"""
from __future__ import annotations

import itertools

import numpy as np

from suffdisc.core import Relation, hellinger_rows, pair_bounds


def pair_table(rel: Relation, alpha: float = 0.05):
    """All within-task pairs: (masks uint64, lo, hi). Requires M <= 63."""
    assert rel.M <= 63
    eta = rel.radii(alpha)
    weights = (np.uint64(1) << np.arange(rel.M, dtype=np.uint64))
    masks, los, his = [], [], []
    for t in np.unique(rel.task):
        idx = np.flatnonzero(rel.task == t)
        if idx.size < 2:
            continue
        ii, jj = np.triu_indices(idx.size, k=1)
        a, b = idx[ii], idx[jj]
        h = hellinger_rows(rel.laws[a], rel.laws[b])
        lo, hi = pair_bounds(h, eta[a], eta[b])
        diff = rel.Z[a] != rel.Z[b]
        masks.append((diff.astype(np.uint64) * weights).sum(axis=1).astype(np.uint64))
        los.append(lo)
        his.append(hi)
    return np.concatenate(masks), np.concatenate(los), np.concatenate(his)


def all_subsets(rel: Relation, alpha: float = 0.05):
    """Return dict mask -> (L(S), U(S)) for every subset of the M fields."""
    masks, lo, hi = pair_table(rel, alpha)
    out = {}
    for s in range(1 << rel.M):
        sel = (masks & np.uint64(s)) == 0
        out[s] = (float(lo[sel].max(initial=0.0)), float(hi[sel].max(initial=0.0)))
    return out


def mask_to_fields(s: int, names: list[str]) -> list[str]:
    return [names[k] for k in range(len(names)) if (s >> k) & 1]


def minimal_sufficient(table: dict, M: int, eps: float, guard: float = 1e-10):
    """All inclusion-minimal sets with U(S) + guard <= eps (certified sufficient; exact laws: R(S) <= eps)."""
    suff = {s for s, (_, u) in table.items() if u + guard <= eps}
    return sorted(s for s in suff if not any((s & ~(1 << k)) in suff for k in range(M) if (s >> k) & 1))


def min_cost_sets(table: dict, costs, eps: float, guard: float = 1e-10):
    """All minimum-cost sufficient sets (ties kept)."""
    best, sets = None, []
    for s, (_, u) in table.items():
        if u + guard > eps:
            continue
        c = sum(costs[k] for k in range(len(costs)) if (s >> k) & 1)
        if best is None or c < best - 1e-12:
            best, sets = c, [s]
        elif abs(c - best) <= 1e-12:
            sets.append(s)
    return best, sorted(sets)
