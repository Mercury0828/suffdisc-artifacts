"""C6 — independent exhaustive validator.

Deliberately simple and on a different code path from every discovery method: grouping by Python dict on raw codes, pure
pairwise float64 distances from its OWN Hellinger and radius implementations (it imports nothing from suffdisc.core), no
pruning, no GEMM, no pivots. It is the reference that K5 is judged against.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

_GUARD = 1e-10  # protocol §3 reporting guard; own constant (C6 imports nothing from the methods)


def _radius(n: float, alphabet: int, n_laws: int, alpha: float) -> float:
    # sufprov Eq. (12): eta = sqrt(1 - exp(-((B-1) ln 2 + ln(N_laws/alpha)) / n))
    c = (alphabet - 1) * math.log(2.0) + math.log(n_laws / alpha)
    return math.sqrt(1.0 - math.exp(-c / n))


def _groups(Z: np.ndarray, task: np.ndarray, S: list[int]) -> list[list[int]]:
    g: dict[tuple, list[int]] = defaultdict(list)
    for i in range(Z.shape[0]):
        g[(int(task[i]),) + tuple(int(Z[i, k]) for k in S)].append(i)
    return [v for v in g.values() if len(v) > 1]


def _pairwise(R: np.ndarray, idx: list[int]) -> np.ndarray:
    """Exact pairwise Hellinger matrix of the rows idx of sqrt-laws R (float64), by explicit differences."""
    X = R[idx]
    out = np.empty((len(idx), len(idx)), dtype=np.float64)
    for a in range(len(idx)):
        d = X - X[a]
        out[a] = np.sqrt((d * d).sum(axis=1)) / math.sqrt(2.0)
    return out


def _radius_v2(counts: np.ndarray, n_laws: int, alpha: float) -> float:
    """Radius rule v2, own implementation (scipy brentq; the method side uses suffdisc.radius_v2 with bisection): Agrawal
    Thm I.2 KL threshold, then min(global conversion, local conversion through per-cell caps)."""
    from scipy.optimize import brentq
    n = float(counts.sum())
    d = counts.size - 1
    L = math.log(n_laws / alpha)
    x = brentq(lambda v: v - d - d * math.log(v / d) - L, d * (1 + 1e-12), 100.0 * (L + d) + 100.0, xtol=1e-13, rtol=1e-15)
    t = x * (1 + 1e-12) / n
    eta = math.sqrt(1.0 - math.exp(-t / 2.0))
    if (counts > 0).all():
        R = 1.0
        for c in counts:
            a = c / n
            if a >= 1.0:
                continue
            f = lambda u: a * math.log(a / u) + (1 - a) * math.log((1 - a) / (1 - u)) - t
            u = min(1.0, brentq(f, a, 1.0 - 1e-15, xtol=1e-15, rtol=1e-15) * (1 + 1e-12))
            R = max(R, u / a)
        k = 4.0 if R <= 1.0 + 1e-12 else 2.0 * (R - 1.0 - math.log(R)) / (math.sqrt(R) - 1.0) ** 2
        eta = min(eta, math.sqrt(t / k))
    return eta


class C6:
    def __init__(self, Z, task, laws, shots=None, alpha: float = 0.05, counts=None, radius: str = "v1"):
        self.Z = np.asarray(Z)
        self.task = np.asarray(task)
        self.R = np.sqrt(np.asarray(laws, dtype=np.float64))
        n_laws, alphabet = self.R.shape
        if radius == "v2":
            self.eta = np.array([_radius_v2(np.asarray(c, dtype=np.float64), n_laws, alpha) for c in counts])
        elif shots is None:
            self.eta = np.zeros(n_laws)
        else:
            self.eta = np.array([_radius(float(s), alphabet, n_laws, alpha) for s in shots])

    def risk_bounds(self, S: list[int]):
        """(L(S), U(S), witness_lo, witness_hi): max over within-group pairs of the lower / upper pair bounds (SI (S8))."""
        L = U = 0.0
        wl = wu = None
        for idx in _groups(self.Z, self.task, list(S)):
            D = _pairwise(self.R, idx)
            e = self.eta[idx]
            lo = np.maximum(0.0, D - e[:, None] - e[None, :])
            hi = np.minimum(1.0, D + e[:, None] + e[None, :])
            np.fill_diagonal(lo, 0.0)
            np.fill_diagonal(hi, 0.0)
            a, b = np.unravel_index(int(np.argmax(lo)), lo.shape)
            if lo[a, b] > L:
                L, wl = float(lo[a, b]), (idx[a], idx[b])
            a, b = np.unravel_index(int(np.argmax(hi)), hi.shape)
            if hi[a, b] > U:
                U, wu = float(hi[a, b]), (idx[a], idx[b])
        return L, U, wl, wu

    def status(self, S, eps: float) -> str:
        """'sufficient' if U(S) + 1e-10 <= eps (protocol §3 reporting guard); 'boundary' if U(S) <= eps < U(S) + 1e-10;
        'insufficient' if L(S) > eps; else 'unresolved' (sufprov SI S1.5)."""
        L, U, _, _ = self.risk_bounds(S)
        if U + _GUARD <= eps:
            return "sufficient"
        if U <= eps and L <= eps:
            return "boundary"
        if L > eps:
            return "insufficient"
        return "unresolved"

    def check_output(self, S, eps: float) -> dict:
        """Full check of one returned set: sufficiency, and per field whether its removal is certified insufficient
        (certified needed) or not; plus whether every proper subset fails (by monotonicity, single deletions suffice)."""
        S = list(S)
        st = self.status(S, eps)
        labels = {}
        for k in S:
            rest = [f for f in S if f != k]
            L, U, _, _ = self.risk_bounds(rest)
            labels[k] = ("certified_needed" if L > eps else "removable" if U + _GUARD <= eps
                         else "boundary" if U <= eps else "unresolved")
        return {"status": st, "labels": labels,
                "certified_inclusion_minimal": st == "sufficient" and all(v == "certified_needed" for v in labels.values()),
                "no_removable_field": all(v != "removable" for v in labels.values())}
