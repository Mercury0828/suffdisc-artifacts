"""Convex lower bound on the pairwise Hellinger distance over Agrawal KL confidence sets (K6 ladder attempt 2; PR-K6-attempt2).

Candidate method improvement — NOT the frozen protocol rule; using it in a test of record needs a pre-outcome amendment.

On the simultaneous event E = {KL(a_r || p_r) <= t_r for every record r} (Agrawal, IEEE TIT 2020, Thm I.2 with a union over the
N_rec records; the same event as radius v2, so no extra error budget), the true laws satisfy p_i in C_i, p_j in C_j with
C_r = {p in simplex : KL(a_r || p) <= t_r}. Hence H(p_i, p_j) >= m_ij := min_{p in C_i, q in C_j} H(p, q).

f(p, q) = H^2 = 1 - sum_k sqrt(p_k q_k) is jointly convex. For any interior (p*, q*) and every (p, q):
    f(p, q) >= f(p*, q*) + g_p . (p - p*) + g_q . (q - q*),   g_p = -sqrt(q*/p*)/2, g_q = -sqrt(p*/q*)/2,
so m_ij^2 >= f(p*, q*) - g_p.p* - g_q.q* + min_{p in C_i} g_p.p + min_{q in C_j} g_q.q.
Each linear minimum over a KL set is bounded below by its Lagrange dual (weak duality; every dual point is valid):
    min_{p in C} g.p >= nu + exp(sum_{a_k>0} a_k ln(g_k - nu) - t)   for nu < min_{a_k>0} g_k and nu <= min_{a_k=0} g_k
(derivation: L = g.p + lam (sum a ln(a/p) - t) + nu (1 - sum p); minimising over p > 0 and then over lam > 0 in closed form).
The anchor (p*, q*) is improved by Frank-Wolfe steps whose iterates stay feasible (x0 = (1 - tau) a + tau / B with
KL <= -ln(1 - tau) <= t / 2), so f at the iterate is an upper bound on m^2 and is used only for early stopping. The returned
value is the largest valid lower bound seen, minus 1e-12, square-rooted (0 if negative).
"""
from __future__ import annotations

import math

import numba as nb
import numpy as np


@nb.njit(cache=True)
def _lin_dual(g, a, t):
    """Lower bound on min_{p in simplex, KL(a||p) <= t} g.p and the primal point of the (approximately) optimal dual."""
    B = g.size
    mpos = np.inf
    mzero = np.inf
    for k in range(B):
        if a[k] > 0:
            if g[k] < mpos:
                mpos = g[k]
        else:
            if g[k] < mzero:
                mzero = g[k]
    rng_ = 0.0
    for k in range(B):
        d = abs(g[k] - mpos)
        if d > rng_:
            rng_ = d
    ub_strict = mpos  # nu must be < mpos
    ub = min(mpos, mzero)
    lo = ub - 2.0 * (rng_ + 1.0)

    def deriv(nu):
        s = 0.0
        w = 0.0
        for k in range(B):
            if a[k] > 0:
                s += a[k] * math.log(g[k] - nu)
                w += a[k] / (g[k] - nu)
        return 1.0 - math.exp(s - t) * w

    for _ in range(60):
        if deriv(lo) > 0:
            break
        lo -= 2.0 * (rng_ + 1.0)
    # bisection on the concave dual's derivative within [lo, ub)
    hi = ub
    if ub >= ub_strict:
        hi = ub_strict - 1e-15 * max(1.0, abs(ub_strict))
    if deriv(hi) >= 0:
        nu = hi
    else:
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if deriv(mid) > 0:
                lo = mid
            else:
                hi = mid
        nu = lo
    s = 0.0
    for k in range(B):
        if a[k] > 0:
            s += a[k] * math.log(g[k] - nu)
    lam = math.exp(s - t)
    val = nu + lam
    # primal recovery (feasible up to rounding; used only as a Frank-Wolfe direction)
    p = np.zeros(B)
    tot = 0.0
    for k in range(B):
        if a[k] > 0:
            p[k] = lam * a[k] / (g[k] - nu)
            tot += p[k]
    if tot < 1.0:
        kz = -1
        for k in range(B):
            if a[k] == 0 and (kz < 0 or g[k] < g[kz]):
                kz = k
        if kz >= 0 and g[kz] <= nu + 1e-12 * max(1.0, abs(nu)):
            p[kz] += 1.0 - tot
            tot = 1.0
    if tot > 0:
        for k in range(B):
            p[k] /= tot
    return val, p


@nb.njit(cache=True)
def _pair_lower(a1, t1, a2, t2, eps, iters):
    B = a1.size
    tau1 = min(0.5, -math.expm1(-t1 / 2.0))
    tau2 = min(0.5, -math.expm1(-t2 / 2.0))
    p = (1.0 - tau1) * a1 + tau1 / B
    q = (1.0 - tau2) * a2 + tau2 / B
    best = -1.0
    for it in range(iters):
        f = 1.0
        for k in range(B):
            f -= math.sqrt(p[k] * q[k])
        if eps >= 0 and f <= eps * eps:  # a feasible point within eps: the minimum cannot exceed eps (eps < 0: no early stop)
            break
        gp = np.empty(B)
        gq = np.empty(B)
        lin = 0.0
        for k in range(B):
            gp[k] = -0.5 * math.sqrt(q[k] / p[k])
            gq[k] = -0.5 * math.sqrt(p[k] / q[k])
            lin += gp[k] * p[k] + gq[k] * q[k]
        v1, s1 = _lin_dual(gp, a1, t1)
        v2, s2 = _lin_dual(gq, a2, t2)
        bnd = f - lin + v1 + v2
        if bnd > best:
            best = bnd
        if eps >= 0 and best > eps * eps + 1e-9:
            break
        # Frank-Wolfe step with a short ternary line search on the convex segment
        lo_, hi_ = 0.0, 1.0
        for _ in range(30):
            m1 = lo_ + (hi_ - lo_) / 3.0
            m2 = hi_ - (hi_ - lo_) / 3.0
            f1 = 1.0
            f2 = 1.0
            for k in range(B):
                f1 -= math.sqrt(((1 - m1) * p[k] + m1 * s1[k]) * ((1 - m1) * q[k] + m1 * s2[k]))
                f2 -= math.sqrt(((1 - m2) * p[k] + m2 * s1[k]) * ((1 - m2) * q[k] + m2 * s2[k]))
            if f1 <= f2:
                hi_ = m2
            else:
                lo_ = m1
        gam = min(0.5 * (lo_ + hi_), 0.999)  # stay interior
        for k in range(B):
            p[k] = (1 - gam) * p[k] + gam * s1[k]
            q[k] = (1 - gam) * q[k] + gam * s2[k]
    best -= 1e-12
    return math.sqrt(best) if best > 0 else 0.0


@nb.njit(cache=True)
def pair_lower_many(A, t, ii, jj, eps, iters):
    out = np.empty(ii.size)
    for x in range(ii.size):
        out[x] = _pair_lower(A[ii[x]], t[ii[x]], A[jj[x]], t[jj[x]], eps, iters)
    return out


def kl_lower(A: np.ndarray, t: np.ndarray, ii: np.ndarray, jj: np.ndarray, eps: float, iters: int = 40) -> np.ndarray:
    """Valid lower bounds on min H over C_i x C_j for the pairs (ii, jj); early-stops once > eps is certified or impossible."""
    return pair_lower_many(np.ascontiguousarray(A, dtype=np.float64), np.ascontiguousarray(t, dtype=np.float64),
                           np.ascontiguousarray(ii, dtype=np.int64), np.ascontiguousarray(jj, dtype=np.int64), float(eps), iters)
