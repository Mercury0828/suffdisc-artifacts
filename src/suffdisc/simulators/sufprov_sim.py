"""Re-implementation of sufprov's finite-class simulator from its Supplementary Information S2.1-S2.4 (arXiv:2610.09931v3).

Four-qubit tasks, exact density-matrix propagation, 16-outcome raw law with logical qubit 0 the most significant bit.
Context fields (SI Table S3): L layout, R routing, G gate noise, M readout, A angle bias, S sampler seed (law-invariant).
Only the angle-draw order is a parameter (`draw`; the "sequential" and array variants currently give identical angles). Other
conventions the quoted SI text does not fix are fixed in code (audit L3/D1-anchor R1): perfect |0000> initialisation,
R(theta) = exp(-i theta sigma / 2), layouts as logical-to-physical maps (all supplied permutations are self-inverse), SWAP =
CNOT(a,b) CNOT(b,a) CNOT(a,b) on the shortest line route with noise after each CNOT, disjoint gates serialised in increasing qubit
order, RNG reset per task with angles shared across contexts (QAOA draws gamma then beta), sampler seeds as law-invariant codes,
real diagonal clipped at 0 and renormalised. The QAOA family (seeds 42-55, depths 1-2) follows the project protocol, not the SI quote.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

LAYOUTS = [(0, 1, 2, 3), (0, 2, 1, 3), (3, 2, 1, 0)]          # logical -> physical
ROUTING = ["move_control", "move_target"]
GATE_NOISE = [0.002, 0.010, 0.030]
READOUT = [0.005, 0.020, 0.060]
ANGLE_BIAS = [0.0, 0.030, 0.080]
SEEDS = [7, 99]
EDGE_W = (0.75, 1.0, 1.25)                                     # weight of physical edge (j, j+1)
P1Q = 0.0005
NQ = 4

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)


def rot(axis: np.ndarray, theta: float) -> np.ndarray:
    return math.cos(theta / 2) * _I - 1j * math.sin(theta / 2) * axis


_CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)  # control = first qubit
_SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


class DM:
    """Density matrix on NQ physical qubits as a tensor with axes (row q0..q3, col q0..q3)."""

    def __init__(self):
        self.t = np.zeros((2,) * (2 * NQ), dtype=complex)
        self.t[(0,) * (2 * NQ)] = 1.0

    def apply(self, U: np.ndarray, qs: tuple[int, ...]) -> None:
        k = len(qs)
        Ut = U.reshape((2,) * (2 * k))
        # rows: contract U[a, b] with row axes qs
        t = np.tensordot(Ut, self.t, axes=(list(range(k, 2 * k)), list(qs)))
        t = np.moveaxis(t, list(range(k)), list(qs))
        cols = [NQ + q for q in qs]
        t = np.tensordot(t, Ut.conj(), axes=(cols, list(range(k, 2 * k))))
        t = np.moveaxis(t, list(range(2 * NQ - k, 2 * NQ)), cols)
        self.t = t

    def depolarize(self, qs: tuple[int, ...], p: float) -> None:
        """D_{T,p}(rho) = (1-p) rho + p I_T/2^|T| (x) Tr_T rho  (SI Eq. (S25))."""
        if p == 0.0:
            return
        k = len(qs)
        red = self.t
        # partial trace over qs: trace row axis q with col axis NQ+q
        axes_sorted = sorted(qs, reverse=True)
        for q in axes_sorted:
            red = np.trace(red, axis1=q, axis2=red.ndim // 2 + q)
        # red now has axes (rows of remaining, cols of remaining) in original relative order
        rem = [q for q in range(NQ) if q not in qs]
        eye = np.eye(2 ** k, dtype=complex).reshape((2,) * (2 * k)) / (2 ** k)
        full = np.tensordot(eye, red, axes=0)  # axes: (qs rows, qs cols, rem rows, rem cols)
        order_rows = list(qs) + rem
        src = list(range(k)) + list(range(2 * k, 2 * k + len(rem)))
        src_cols = list(range(k, 2 * k)) + list(range(2 * k + len(rem), 2 * k + 2 * len(rem)))
        perm = [0] * (2 * NQ)
        for pos, q in enumerate(order_rows):
            perm[q] = src[pos]
            perm[NQ + q] = src_cols[pos]
        full = np.transpose(full, perm)
        self.t = (1 - p) * self.t + p * full

    def probs(self) -> np.ndarray:
        m = self.t.reshape(2 ** NQ, 2 ** NQ)
        return np.real(np.diag(m)).copy()


class Machine:
    """Physical line 0-1-2-3 with noisy gates; tracks the logical -> physical layout."""

    def __init__(self, layout, G, A, routing):
        self.dm = DM()
        self.pos = list(layout)           # pos[logical] = physical
        self.G, self.A, self.routing = G, A, routing
        self.n_cnot = 0

    def g1(self, U, logical):
        q = self.pos[logical]
        self.dm.apply(U, (q,))
        self.dm.depolarize((q,), P1Q)

    def ry(self, theta, lq):
        self.g1(rot(_Y, theta + self.A), lq)

    def rx(self, theta, lq):
        self.g1(rot(_X, theta + self.A), lq)

    def rz(self, theta, lq):
        self.g1(rot(_Z, theta), lq)

    def h(self, lq):
        self.g1(_H, lq)

    def _phys_cnot(self, pc, pt):
        assert abs(pc - pt) == 1
        self.dm.apply(_CNOT, (pc, pt))
        self.dm.depolarize((min(pc, pt), max(pc, pt)), self.G * EDGE_W[min(pc, pt)])
        self.n_cnot += 1

    def _phys_swap(self, a, b):
        # SWAP = three CNOTs on the same edge; the two-qubit depolarizer commutes with any unitary on that edge
        self._phys_cnot(a, b)
        self._phys_cnot(b, a)
        self._phys_cnot(a, b)

    def cnot(self, lc, lt):
        pc, pt = self.pos[lc], self.pos[lt]
        if abs(pc - pt) == 1:
            self._phys_cnot(pc, pt)
            return
        mover_is_control = self.routing == "move_control"
        m, other = (pc, pt) if mover_is_control else (pt, pc)
        step = 1 if other > m else -1
        swaps = []
        cur = m
        while abs(cur - other) > 1:
            self._phys_swap(cur, cur + step)
            swaps.append((cur, cur + step))
            cur += step
        if mover_is_control:
            self._phys_cnot(cur, other)
        else:
            self._phys_cnot(other, cur)
        for a, b in reversed(swaps):
            self._phys_swap(a, b)

    def law(self, M) -> np.ndarray:
        p = self.dm.probs().reshape((2,) * NQ)   # axes = physical qubits 0..3, value bit per qubit
        flip = np.array([[1 - M, M], [M, 1 - M]])
        for q in range(NQ):
            p = np.moveaxis(np.tensordot(flip, p, axes=([1], [q])), 0, q)
        # relabel to logical order: outcome bits (logical 0 most significant)
        p = np.transpose(p, [self.pos[l] for l in range(NQ)])
        out = p.reshape(-1)
        out = np.clip(out, 0.0, None)
        return out / out.sum()


def task_angles(kind: str, depth: int, seed: int, draw: str = "sequential"):
    rng = np.random.default_rng(seed)
    if kind == "HEA":
        return rng.uniform(-math.pi, math.pi, size=(depth + 1, NQ))
    if draw == "sequential":
        return np.array([[rng.uniform(0.1, 1.0), rng.uniform(0.1, 1.0)] for _ in range(depth)])
    return rng.uniform(0.1, 1.0, size=(depth, 2))


def run_task(kind, depth, angles, layout, routing, G, M, A):
    m = Machine(layout, G, A, routing)
    if kind == "HEA":
        for layer in range(depth + 1):
            for q in range(NQ):
                m.ry(angles[layer, q], q)
            if layer < depth:
                for (c, t) in [(0, 1), (1, 2), (2, 3)]:
                    m.cnot(c, t)
    else:
        for q in range(NQ):
            m.h(q)
        for layer in range(depth):
            gamma, beta = angles[layer]
            for (c, t) in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                m.cnot(c, t)
                m.rz(-gamma, t)
                m.cnot(c, t)
            for q in range(NQ):
                m.rx(2 * beta, q)
    return m.law(M), m.n_cnot


def contexts():
    """All 324 complete contexts as integer codes (L, R, G, M, A, S)."""
    return list(itertools.product(range(3), range(2), range(3), range(3), range(3), range(2)))


def build_task_laws(kind, depth, seed=42, draw="sequential"):
    angles = task_angles(kind, depth, seed, draw)
    cache, Z, P = {}, [], []
    for (l, r, g, mm, a, s) in contexts():
        key = (l, r, g, mm, a)
        if key not in cache:
            cache[key] = run_task(kind, depth, angles, LAYOUTS[l], ROUTING[r], GATE_NOISE[g], READOUT[mm], ANGLE_BIAS[a])[0]
        Z.append((l, r, g, mm, a, s))
        P.append(cache[key])
    return np.array(Z, dtype=np.int32), np.array(P), angles
