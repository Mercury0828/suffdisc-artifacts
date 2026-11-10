"""Pair certification under the 'kl' rule of record (protocol §16, A-005) for use INSIDE discovery engines (D-013).

One object per relation; every method that uses it has identical certificate semantics.
  radius   : v2 (Agrawal threshold + local conversion), per record           -> triangle bounds (cheap, valid)
  refine   : only pairs the triangle leaves UNRESOLVED (lo <= eps < hi + GUARD):
             hi <- min(hi, KL-set Lagrangian-dual upper bound)   (suffdisc.pair_upper; any multiplier vector is valid, so a
                   fast multiplier search only loosens, never invalidates)
             lo <- max(lo, KL-set convex lower bound)            (suffdisc.kl_lower)
Fast multiplier search for the upper bound: a fixed 3^4 grid around Z0 (the median optimum on development pairs), then one
Nelder-Mead from the best grid point. The procedure depends only on the pair (no state carried between pairs), so every pair's
bound — and hence the possible-violation family that T2 refers to — is a fixed function of the data (development check, tools/dev/dual_fast_eval.py: same status as the 3-start reference on 300 mirror pairs,
≈ 3 ms/pair vs ≈ 38 ms). Results are cached per pair, so a pair met again in later iterations costs nothing.
Exact laws (shots None): radius 0, no refinement — identical to the exact rule.

Floating point: every decision is made on a fixed, order-independent predicate of the (canonical) pair.
  - the engine's GEMM value h = sqrt(max(0, 1 - <sqrt a, sqrt b>)) has |h^2 - H^2| <= E_B = 2 (B + 8) u (u = 2^-53; standard
    dot-product error bound on non-negative terms whose sum is <= 1, plus the sqrt and subtraction roundings); every bound is
    inflated by it: lo uses sqrt(max(0, h^2 - E_B)), hi uses sqrt(h^2 + E_B);
  - when an inflated bound lies within TAU = 1e-6 of a decision threshold (eps - GUARD for hi, eps for lo), h is replaced by the
    canonical value (stable difference formula on the pair in index order), so the decision does not depend on which GEMM block
    produced h; outside that band the GEMM and canonical decisions agree (|h_gemm - h_canon| <= 2 sqrt(E_B) << TAU);
  - KL refinements are computed on the pair in index order and cached per (i, j, eps).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

from suffdisc.core import GUARD, pair_bounds
from suffdisc.kl_lower import kl_lower
from suffdisc.pair_upper import _boxes, _neg_entropy, dual_value
from suffdisc.radius_v2 import kl_threshold, radius_v2

U53 = 2.0 ** -53
TAU = 1e-6

_OFFS = np.array(np.meshgrid(*[[-1.0, 0.0, 1.0]] * 4)).reshape(4, -1).T * 0.25


class PairDecider:
    """rule 'v1': sufprov Eq. (12) radii, triangle bounds; rule 'kl': v2 radii + KL refinement (protocol §16). Both with the
    floating-point inflation and canonical band recomputation above."""

    def __init__(self, laws: np.ndarray, shots: np.ndarray | None, alpha: float = 0.05, rule: str = "kl", eta_v1=None):
        self.A = np.ascontiguousarray(laws, dtype=np.float64)
        self.SA = np.sqrt(self.A)
        N, B = self.A.shape
        self.E = 2.0 * (B + 8) * U53
        self.margin = 6.0 * math.sqrt(self.E) + 1e-9  # certificates and prefilters: GEMM error (<= sqrt(E) per distance) + canonical shift
        self.rule = rule
        self.exact = shots is None
        self.cache: dict[tuple[int, int, float], tuple[float, float]] = {}
        self.n_refined = 0
        self.n_upper_nm = 0
        self.n_canonical = 0
        if self.exact:
            self.eta = np.zeros(N)
            return
        if rule == "v1":
            self.eta = np.asarray(eta_v1, dtype=np.float64)
            return
        shots = np.asarray(shots, dtype=np.int64)
        assert np.all(shots == shots[0]), "kl rule implemented for equal shot counts per record (protocol §16)"
        counts = np.rint(self.A * shots[:, None]).astype(np.int64)
        self.n = int(shots[0])
        self.eta = radius_v2(counts, N, alpha)
        self.t = kl_threshold(self.n, B, N, alpha)
        self.tvec = np.full(N, self.t)
        self.boxes: dict[int, tuple] = {}
        self.negent: dict[int, float] = {}
        self.zmed = np.array([-0.3, -0.3, 0.25, 0.25])  # Z0, fixed (pair-deterministic bounds)

    def _box(self, r):
        if r not in self.boxes:
            self.boxes[r] = _boxes(self.A[r], self.t)
            self.negent[r] = _neg_entropy(self.A[r])
        return self.boxes[r], self.negent[r]

    def _upper(self, i, j, eps):
        (XL, XU), ca = self._box(i)
        (YL, YU), cb = self._box(j)
        a, b, t = self.A[i], self.A[j], self.t

        def dv(z):
            return dual_value(np.asarray(z, dtype=np.float64), a, b, XL, XU, YL, YU, ca, cb, t, t)

        grid = self.zmed[None, :] + _OFFS
        vals = np.array([dv(z) for z in grid])
        k = int(vals.argmax())
        best, bz = float(vals[k]), grid[k]
        if math.sqrt(min(1.0, max(0.0, 1.0 - best))) + GUARD > eps:  # not yet certified pass: refine the multipliers
            self.n_upper_nm += 1
            o = minimize(lambda z: -dv(z), bz, method="Nelder-Mead",
                         options={"xatol": 1e-4, "fatol": 1e-7, "maxfev": 300})
            if -o.fun > best:
                best, bz = float(-o.fun), o.x
        return math.sqrt(min(1.0, max(0.0, 1.0 - best)))

    def _tri(self, h, ei, ej):
        lo = np.maximum(0.0, np.sqrt(np.maximum(0.0, h * h - self.E)) - ei - ej)
        hi = np.minimum(1.0, np.sqrt(h * h + self.E) + ei + ej)
        return lo, hi

    def canonical(self, i, j):
        d = self.SA[i] - self.SA[j]
        return math.sqrt(0.5 * float(np.dot(d, d)))

    def bounds(self, ia: np.ndarray, ib: np.ndarray, h: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
        """(lo, hi) for pairs (ia, ib) with engine value h, under the decider's rule at tolerance eps (order-independent)."""
        ia = np.asarray(ia, dtype=np.int64)
        ib = np.asarray(ib, dtype=np.int64)
        I, J = np.minimum(ia, ib), np.maximum(ia, ib)
        h = np.array(h, dtype=np.float64, copy=True)
        ei, ej = self.eta[I], self.eta[J]
        lo, hi = self._tri(h, ei, ej)
        band = np.flatnonzero((np.abs(hi - (eps - GUARD)) <= TAU) | (np.abs(lo - eps) <= TAU))
        if band.size:
            self.n_canonical += band.size
            h[band] = [self.canonical(int(I[x]), int(J[x])) for x in band]
            lo[band], hi[band] = self._tri(h[band], ei[band], ej[band])
        if self.exact or self.rule != "kl":
            return lo, hi
        unres = np.flatnonzero((hi + GUARD > eps) & (lo <= eps))
        for x in unres:
            i, j = int(I[x]), int(J[x])
            key = (i, j, eps)
            c = self.cache.get(key)
            if c is None:
                self.n_refined += 1
                u = self._upper(i, j, eps)
                lk = 0.0
                if u + GUARD > eps:
                    lk = float(kl_lower(self.A, self.tvec, np.array([i]), np.array([j]), eps)[0])
                c = (lk, u)
                self.cache[key] = c
            lo[x], hi[x] = max(lo[x], c[0]), min(hi[x], c[1])
        return lo, hi


def KLCertifier(laws, shots, alpha=0.05):  # backwards-compatible name
    return PairDecider(laws, shots, alpha, rule="kl")
