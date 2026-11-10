"""Radius rule v3 candidate (K6 ladder attempt 2b): Lagrangian-dual bound on rho = max_{p in C} H(p, a) over the Agrawal KL set
C = {p in simplex : KL(a||p) <= t}, with the per-cell caps L_k <= p_k <= U_k (binary kl(a_k||.) <= t; implied by the KL constraint
by data processing, so adding them changes nothing on the event but strengthens the dual). Candidate only, not the frozen rule.

H^2(p, a) = 1 - sum_k sqrt(a_k p_k). Weak duality: for every lam > 0 and real nu,
  max_{p in C} H^2 <= 1 + lam t + nu + sum_{a_k>0} a_k max_{r in [L_k/a_k, U_k/a_k]} (-sqrt(r) + lam ln r - nu r)
                     + sum_{a_k=0} max_{p in [0, U_k]} (-nu p)
(substituting p_k = a_k r_k; the cells with a_k = 0 do not enter H(p, a) or KL(a||p)). Each per-cell maximum is exact: with
s = sqrt(r), psi(s) = -s + 2 lam ln s - nu s^2 has its stationary points at the real roots of 2 nu s^2 + s - 2 lam = 0, and the
maximum over an interval is attained at an endpoint or a stationary point inside it. So every evaluated (lam, nu) gives a valid
upper bound; we minimise over (ln lam, nu) in [-18, 9] x [-50, 50] (no catastrophic cancellation) and keep the best evaluated value
(+1e-10). eta_v3 = min(eta_v2, sqrt(bound)).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

from suffdisc.radius_v2 import _bkl, kl_threshold, radius_v2


def _lower_cap(a: float, t: float) -> float:
    """Smallest l in (0, a] with binary kl(a || l) <= t (kl decreasing on (0, a]); bisection, returns a lower bound."""
    if a <= 0.0:
        return 0.0
    lo, hi = 0.0, a
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= 0.0 or _bkl(a, mid) > t:
            lo = mid
        else:
            hi = mid
    return lo


def _upper_cap_any(a: float, t: float) -> float:
    if a >= 1.0:
        return 1.0
    if a <= 0.0:
        return -math.expm1(-t)  # kl(0 || u) = -ln(1 - u) = t
    lo, hi = a, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid >= 1.0 or mid <= lo:  # binary64 resolution reached near 1: keep the conservative upper end hi
            break
        if _bkl(a, mid) <= t:
            lo = mid
        else:
            hi = mid
    return hi


def _psi_max(lam: float, nu: float, s_lo: float, s_hi: float) -> float:
    def psi(s):
        return -s + 2.0 * lam * math.log(s) - nu * s * s
    cands = [s_lo, s_hi]
    disc = 1.0 + 16.0 * nu * lam
    if disc >= 0.0:
        rd = math.sqrt(disc)
        cands.append(4.0 * lam / (1.0 + rd))  # = (-1 + rd) / (4 nu), rationalised (stable as nu -> 0; = 2 lam at nu = 0)
        if nu < 0.0:
            cands.append((-1.0 - rd) / (4.0 * nu))  # the second positive root exists only for nu < 0
    return max(psi(s) for s in cands if s_lo <= s <= s_hi and s > 0.0)


def dual_bound(a: np.ndarray, t: float) -> float:
    """Upper bound on max_{KL(a||p) <= t} H^2(p, a)."""
    pos = [(float(x), _lower_cap(float(x), t), _upper_cap_any(float(x), t)) for x in a if x > 0]
    zero_caps = [_upper_cap_any(0.0, t) for x in a if x <= 0]
    sl = [(x, math.sqrt(max(L / x, 1e-300)), math.sqrt(U / x)) for x, L, U in pos]

    def D(z):
        # clamp to a range where the sum has no catastrophic cancellation (rounding error << the 1e-10 margin)
        lam, nu = math.exp(min(max(z[0], -18.0), 9.0)), min(max(z[1], -50.0), 50.0)
        v = 1.0 + lam * t + nu
        for x, s_lo, s_hi in sl:
            v += x * _psi_max(lam, nu, s_lo, s_hi)
        for U in zero_caps:
            v += max(0.0, -nu * U)
        return v

    best = math.inf
    for z0 in ([math.log(0.5), 0.0], [math.log(5.0), -1.0], [math.log(0.05), 0.5]):
        r = minimize(D, np.array(z0), method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-13, "maxiter": 4000})
        best = min(best, D(r.x))
    return min(1.0, best + 1e-10)


def radius_v3(counts: np.ndarray, n_laws: int, alpha: float = 0.05) -> np.ndarray:
    counts = np.asarray(counts)
    eta2 = radius_v2(counts, n_laws, alpha)
    out = eta2.copy()
    cache: dict[int, float] = {}
    for r in range(counts.shape[0]):
        n = int(counts[r].sum())
        if n not in cache:
            cache[n] = kl_threshold(n, counts.shape[1], n_laws, alpha)
        b = dual_bound(counts[r] / n, cache[n])
        out[r] = min(eta2[r], math.sqrt(max(0.0, b)))
    return out
