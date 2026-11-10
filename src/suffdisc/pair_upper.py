"""Sharper pairwise upper bound over Agrawal KL confidence sets (K6 ladder attempt 3). Candidate only, not the frozen rule.

On the simultaneous event E (every true law in its set C_r = {p : KL(a_r || p) <= t_r}; the same event as radius v2, no extra
error budget), H(p_i, p_j) <= sqrt(1 - m) with m <= min_{p in C_i, q in C_j} BC(p, q), BC(p, q) = sum_k sqrt(p_k q_k).
The triangle bound H(a_i, a_j) + rho_i + rho_j assumes the two displacements are anti-aligned; in sqrt-space, displacements into
different cells are orthogonal, so the triangle is loose (D2b: feasible maxima ~0.12 where the triangle gives ~0.20).

Lower bound m by weak Lagrangian duality with the redundant per-cell boxes p_k in [L_k, U_k] (binary kl(a_k || .) <= t, implied by
the KL constraint by data processing). For lam, mu >= 0 and real nu, xi:
  min_{C_i x C_j} BC >= lam (sum a ln a - t) + mu (sum b ln b - t) - nu - xi + sum_k min_{box_k} phi_k,
  phi_k(x, y) = x y - 2 lam a_k ln x - 2 mu b_k ln y + nu x^2 + xi y^2,   x = sqrt(p_k), y = sqrt(q_k)
(the ln terms only for a_k > 0 / b_k > 0, where the box excludes 0). Each per-cell minimum over the rectangle is found exactly
among: the 4 corners; the stationary points on each edge (roots of 2 xi y^2 + x0 y - 2 mu b = 0, resp. the x-analogue); the
interior stationary points (z = x^2 solves (8 xi nu^2 - 2 nu) z^2 + (2 lam a - 2 mu b - 16 xi lam a nu) z + 8 xi lam^2 a^2 = 0,
y = (2 lam a - 2 nu z) / x). Rounding in a computed stationary point changes phi only to second order; a margin of 1e-9 per cell
is subtracted. Any evaluated multiplier vector gives a valid bound; we maximise numerically and keep the best evaluated value.
"""
from __future__ import annotations

import math

import numba as nb
import numpy as np
from scipy.optimize import minimize

from suffdisc.radius_v2 import kl_threshold
from suffdisc.radius_v3 import _lower_cap, _upper_cap_any

CELL_MARGIN = 1e-9


@nb.njit(cache=True)
def _phi(x, y, lam, mu, nu, xi, a, b):
    v = x * y + nu * x * x + xi * y * y
    if a > 0:
        v -= 2.0 * lam * a * math.log(x)
    if b > 0:
        v -= 2.0 * mu * b * math.log(y)
    return v


@nb.njit(cache=True)
def _edge_roots(c_quad, c_lin, c_const):
    """Positive roots of c_quad s^2 + c_lin s - c_const = 0 with c_lin >= 0, c_const >= 0 (stable forms)."""
    out = np.full(2, -1.0)
    D = c_lin * c_lin + 4.0 * c_quad * c_const
    if D < 0:
        return out
    rd = math.sqrt(D)
    den = c_lin + rd
    if den > 0:
        out[0] = 2.0 * c_const / den
    if c_quad < 0:
        out[1] = (-c_lin - rd) / (2.0 * c_quad)
    return out


@nb.njit(cache=True)
def _cell_min(lam, mu, nu, xi, a, b, xl, xu, yl, yu):
    best = np.inf
    xs = (xl, xu)
    ys = (yl, yu)
    for x in xs:
        for y in ys:
            if (a > 0 and x <= 0) or (b > 0 and y <= 0):
                continue
            v = _phi(x, y, lam, mu, nu, xi, a, b)
            if v < best:
                best = v
    # edges x = x0: d/dy = x0 - 2 mu b / y + 2 xi y = 0  <=>  2 xi y^2 + x0 y - 2 mu b = 0
    for x0 in xs:
        if a > 0 and x0 <= 0:
            continue
        r = _edge_roots(2.0 * xi, x0, 2.0 * mu * b)
        for y in r:
            if y > 0 and yl <= y <= yu:
                v = _phi(x0, y, lam, mu, nu, xi, a, b)
                if v < best:
                    best = v
    for y0 in ys:
        if b > 0 and y0 <= 0:
            continue
        r = _edge_roots(2.0 * nu, y0, 2.0 * lam * a)
        for x in r:
            if x > 0 and xl <= x <= xu:
                v = _phi(x, y0, lam, mu, nu, xi, a, b)
                if v < best:
                    best = v
    # interior stationary points
    c2 = 8.0 * xi * nu * nu - 2.0 * nu
    c1 = 2.0 * lam * a - 2.0 * mu * b - 16.0 * xi * lam * a * nu
    c0 = 8.0 * xi * lam * lam * a * a
    zs = np.full(2, -1.0)
    if abs(c2) < 1e-300:
        if abs(c1) > 1e-300:
            zs[0] = -c0 / c1
    else:
        D = c1 * c1 - 4.0 * c2 * c0
        if D >= 0:
            rd = math.sqrt(D)
            q = -0.5 * (c1 + math.copysign(rd, c1))
            if q != 0:
                zs[0] = q / c2
                zs[1] = c0 / q
            else:
                zs[0] = 0.0
    for z in zs:
        if z > 0:
            x = math.sqrt(z)
            y = (2.0 * lam * a - 2.0 * nu * z) / x
            if xl <= x <= xu and y > 0 and yl <= y <= yu:
                v = _phi(x, y, lam, mu, nu, xi, a, b)
                if v < best:
                    best = v
    return best


@nb.njit(cache=True)
def dual_value(z, A, Bv, XL, XU, YL, YU, ca, cb, t1, t2):
    lam = math.exp(min(max(z[0], -18.0), 9.0))
    mu = math.exp(min(max(z[1], -18.0), 9.0))
    nu = min(max(z[2], -50.0), 50.0)
    xi = min(max(z[3], -50.0), 50.0)
    v = lam * (ca - t1) + mu * (cb - t2) - nu - xi
    for k in range(A.size):
        v += _cell_min(lam, mu, nu, xi, A[k], Bv[k], XL[k], XU[k], YL[k], YU[k]) - CELL_MARGIN
    return v


def _boxes(a, t):
    L = np.array([_lower_cap(float(x), t) for x in a])
    U = np.array([_upper_cap_any(float(x), t) for x in a])
    return np.sqrt(L), np.sqrt(U)


def _neg_entropy(a):
    m = a > 0
    return float((a[m] * np.log(a[m])).sum())


def pair_upper(a, b, t1, t2, boxes=None, starts=None):
    """Valid upper bound on max_{C_i x C_j} H (on the event E). boxes: precomputed (XL, XU, YL, YU)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if boxes is None:
        XL, XU = _boxes(a, t1)
        YL, YU = _boxes(b, t2)
    else:
        XL, XU, YL, YU = boxes
    ca, cb = _neg_entropy(a), _neg_entropy(b)
    f = lambda z: -dual_value(np.asarray(z, dtype=np.float64), a, b, XL, XU, YL, YU, ca, cb, t1, t2)  # noqa: E731
    best = -np.inf
    for z0 in (starts or [[math.log(1.0), math.log(1.0), -0.5, -0.5], [math.log(10.0), math.log(10.0), -5.0, -5.0],
                          [math.log(0.1), math.log(0.1), 0.0, 0.0]]):
        r = minimize(f, np.array(z0, dtype=np.float64), method="Nelder-Mead",
                     options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 3000, "maxfev": 3000})
        best = max(best, -f(r.x))
    return math.sqrt(min(1.0, max(0.0, 1.0 - best)))


def pair_upper_many(Acounts_or_laws, n, n_laws, ii, jj, alpha=0.05, prior_hi=None, eps=None):
    """Upper bounds for pairs (ii, jj) of empirical laws A (N x B) at n shots. If prior_hi and eps are given, only pairs with
    prior_hi + 1e-10 > eps are refined (the others already pass); the result is min(prior_hi, refined)."""
    A = np.asarray(Acounts_or_laws, dtype=np.float64)
    t = kl_threshold(n, A.shape[1], n_laws, alpha)
    box = {}
    out = np.array(prior_hi, dtype=np.float64) if prior_hi is not None else np.ones(ii.size)
    for x in range(ii.size):
        if prior_hi is not None and eps is not None and prior_hi[x] + 1e-10 <= eps:
            continue
        i, j = int(ii[x]), int(jj[x])
        for r in (i, j):
            if r not in box:
                box[r] = _boxes(A[r], t)
        u = pair_upper(A[i], A[j], t, t, boxes=(box[i][0], box[i][1], box[j][0], box[j][1]))
        out[x] = min(out[x], u)
    return out
