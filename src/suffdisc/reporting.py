"""Boundary reporting (protocol §3; audit L3/D4-validator R2): "a set within 1e-10 of eps is reported as 'boundary' and listed".

Applied identically to the output of every method (B3, SUFFDISC-0) after it returns, so it never changes which set is returned
and costs the same for both. The candidate sets a method reports on are its returned set S and the single deletions S \\ {k};
for a no-solution result, the full field set. The returned S always satisfies U(S) + GUARD <= eps (guarded discovery), so the
sets that can be boundary are:
- S \\ {k} for every field k labelled 'minimality_unresolved' (a certified-needed field has L(S \\ {k}) > eps);
- the full field set when no certified-sufficient set exists.
A set T is boundary iff eps - GUARD < U(T) <= eps (and it is then listed with U(T)). U(T) is computed exactly with an early exit
at the first pair whose upper bound exceeds eps (then T is plainly not sufficient). Pairs are visited in a fixed random order of
groups / blocks (seed 0) so that typical non-sufficient sets exit after few blocks.
"""
from __future__ import annotations

import time

import numpy as np

from suffdisc.core import GUARD, Relation, pair_bounds
from suffdisc.engine.kernels import hellinger_block, sqrt_embed
from suffdisc.engine.suffdisc0 import _groups


def _upper_risk(rel: Relation, T: list[int], eps: float, eta: np.ndarray, R: np.ndarray, block: int = 1024) -> float:
    """U(T) if U(T) <= eps, else some value > eps (early exit)."""
    U = 0.0
    groups = _groups(rel.Z, rel.task, list(T))
    order = np.random.default_rng(0).permutation(len(groups))
    for gi in order:
        g = groups[gi]
        for a in range(0, g.size, block):
            ga = g[a:a + block]
            for b in range(a, g.size, block):
                gb = g[b:b + block]
                H = hellinger_block(R[ga], R[gb])
                _, hi = pair_bounds(H, eta[ga][:, None], eta[gb][None, :])
                if a == b:
                    hi = np.triu(hi, k=1)
                m = float(hi.max()) if hi.size else 0.0
                if m > eps:
                    return m
                U = max(U, m)
    return U


def boundary_report(rel: Relation, eps: float, result: dict, alpha: float = 0.05, eta: np.ndarray | None = None) -> dict:
    t0 = time.perf_counter()
    eta = rel.radii(alpha) if eta is None else eta
    R = sqrt_embed(rel.laws)
    listed = []
    if result.get("status") != "complete":
        return {"boundary_sets": [], "t_report": 0.0, "checked": 0}
    if result.get("certified_sufficient_exists"):
        S = list(result["S"])
        cands = [[f for f in S if f != k] for k, v in result.get("labels", {}).items() if v != "certified_needed"]
    else:
        cands = [list(range(rel.M))]
    for T in cands:
        U = _upper_risk(rel, T, eps, eta, R)
        if eps - GUARD < U <= eps:
            listed.append({"set": sorted(int(x) for x in T), "U": U})
    return {"boundary_sets": listed, "t_report": time.perf_counter() - t0, "checked": len(cands)}
