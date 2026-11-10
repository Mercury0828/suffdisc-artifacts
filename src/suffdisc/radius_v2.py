"""Radius rule v2 (K6 repair ladder, attempt 1; round L1/repair-K6 R1). Candidate method improvement — NOT the frozen protocol
radius (protocol §3 = sufprov Eq. (12)); using it in a K6 test of record needs a pre-outcome amendment.

Per record r with counts C_r over B outcomes and n_r shots, empirical law a = C_r / n_r:
1. KL threshold (Agrawal, IEEE TIT 2020, arXiv:1904.02291, Thm I.2, verified from the PDF 2026-11-07): for eps > (k-1)/n,
   Pr[KL(a||p) >= eps] <= exp(-n eps) (e eps n / (k-1))^(k-1). With d = B - 1, per-record level alpha / N_rec, the threshold is
   t = x / n where x > d solves x - d - d ln(x/d) = ln(N_rec / alpha). Union over the N_rec records => simultaneous event.
2. Global conversion (as sufprov SI S1.5): H^2 = 1 - exp(-D_1/2 / 2) <= 1 - exp(-t/2).
3. Local conversion (derived in the round; proof checked by the orchestrator): if every a_i > 0, KL(a||p) = sum a_i (r_i - 1 -
   ln r_i) and H^2(a,p) = (1/2) sum a_i (sqrt(r_i) - 1)^2 with r_i = p_i / a_i; k(r) = 2 (r - 1 - ln r) / (sqrt(r) - 1)^2 is
   decreasing on r > 0 (k(1) = 4); on the event, p_i <= U_i where kl(a_i || U_i) = t (binary KL; data processing), so every
   r_i <= R = max_i U_i / a_i and KL >= k(R) H^2, i.e. H <= sqrt(t / k(R)). eta = min(global, local). No extra error budget.
"""
from __future__ import annotations

import math

import numpy as np


def kl_threshold(n: float, alphabet_size: int, n_laws: int, alpha: float) -> float:
    """t such that Pr[KL(a||p) >= t] <= alpha / n_laws by Agrawal Thm I.2 (bisection on x = n t, x > d)."""
    d = alphabet_size - 1
    L = math.log(n_laws / alpha)
    lo, hi = float(d), float(d) + 10.0 * (L + d) + 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid - d - d * math.log(mid / d) >= L:
            hi = mid
        else:
            lo = mid
    return hi / n  # hi satisfies the inequality, so the tail bound at hi / n is <= alpha / n_laws


def _bkl(a: float, u: float) -> float:
    out = a * math.log(a / u)
    if a < 1.0:
        if u >= 1.0:  # kl(a || 1) = +inf for a < 1 (guard against a rounded endpoint)
            return math.inf
        out += (1.0 - a) * math.log((1.0 - a) / (1.0 - u))
    return out


def _upper_cap(a: float, t: float) -> float:
    """Largest u in [a, 1) with binary kl(a || u) <= t (kl increasing in u >= a); bisection, returns an upper bound."""
    if a >= 1.0:
        return 1.0
    lo, hi = a, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _bkl(a, mid) <= t:
            lo = mid
        else:
            hi = mid
    return hi


def _k(R: float) -> float:
    if R <= 1.0 + 1e-12:
        return 4.0
    return 2.0 * (R - 1.0 - math.log(R)) / (math.sqrt(R) - 1.0) ** 2


def radius_v2(counts: np.ndarray, n_laws: int, alpha: float = 0.05) -> np.ndarray:
    """Per-record radius v2 from integer counts (N x B)."""
    counts = np.asarray(counts)
    N, B = counts.shape
    out = np.empty(N)
    cache: dict[int, float] = {}
    for r in range(N):
        n = int(counts[r].sum())
        if n not in cache:
            cache[n] = kl_threshold(n, B, n_laws, alpha)
        t = cache[n]
        eta = math.sqrt(-math.expm1(-t / 2.0))
        if (counts[r] > 0).all():
            a = counts[r] / n
            R = max(_upper_cap(float(ai), t) / float(ai) for ai in a)
            eta = min(eta, math.sqrt(t / _k(R)))
        out[r] = eta
    return out
