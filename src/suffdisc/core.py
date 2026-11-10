"""Shared primitives used by EVERY method (equal-semantics rule, design notes★★+):
one Hellinger function with one normalisation, one confidence-radius rule, one three-valued rule, one relation format.

Normalisation (sufprov arXiv:2610.09931v3, Eq. (1); lit/GIVEN.md G1/G3): H(p, q) = ||sqrt(p) - sqrt(q)||_2 / sqrt(2), in [0, 1].
Radius (sufprov Eq. (12) / SI (S5)-(S7)): eta = sqrt(1 - exp(-c / n)), c = (B - 1) log 2 + log(N_rec / alpha), where B is the
full outcome-alphabet size, N_rec the number of laws in the simultaneous confidence family and alpha the failure probability.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np

SQRT2 = math.sqrt(2.0)
MISSING = -1  # code for a missing value; equal only to itself


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    """Hellinger distance in sufprov's normalisation, via differences of square roots (no 1 - sum sqrt(pq) cancellation)."""
    d = np.sqrt(np.asarray(p, dtype=np.float64)) - np.sqrt(np.asarray(q, dtype=np.float64))
    return float(np.sqrt(np.dot(d, d)) / SQRT2)


def hellinger_rows(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Row-wise Hellinger distances between matching rows of P and Q (float64, differences of square roots)."""
    D = np.sqrt(np.asarray(P, dtype=np.float64)) - np.sqrt(np.asarray(Q, dtype=np.float64))
    return np.sqrt(np.einsum("ij,ij->i", D, D)) / SQRT2


def confidence_radius(n_shots: np.ndarray | int, alphabet_size: int, n_laws: int, alpha: float) -> np.ndarray:
    """Simultaneous Hellinger confidence radius of sufprov Eq. (12) for each law (fixed shot counts, full alphabet)."""
    c = (alphabet_size - 1) * math.log(2.0) + math.log(n_laws / alpha)
    n = np.asarray(n_shots, dtype=np.float64)
    return np.sqrt(-np.expm1(-c / n))


def pair_bounds(h_hat: np.ndarray, eta_r: np.ndarray, eta_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lower / upper bounds of sufprov Eq. (11): max(0, H - eta_r - eta_s), min(1, H + eta_r + eta_s)."""
    lo = np.maximum(0.0, h_hat - eta_r - eta_s)
    hi = np.minimum(1.0, h_hat + eta_r + eta_s)
    return lo, hi


# Three-valued pair status for a tolerance eps
PASS, UNRESOLVED, VIOLATION = 0, 1, 2
# Reporting guard (protocol §3, sufprov SI S2.3): a pair passes only if hi + GUARD <= eps, so every reported 'sufficient' set has
# U(S) + GUARD <= eps; a pair with eps - GUARD < hi <= eps is a 'boundary' pair (counted, never passed). Audit L3/D4-validator R1.
GUARD = 1e-10


def pair_status(lo: np.ndarray, hi: np.ndarray, eps: float) -> np.ndarray:
    """PASS if hi + GUARD <= eps; certified VIOLATION if lo > eps; UNRESOLVED otherwise. Exact laws: lo == hi == H."""
    st = np.full(np.shape(lo), UNRESOLVED, dtype=np.int8)
    st[hi + GUARD <= eps] = PASS
    st[lo > eps] = VIOLATION
    return st


@dataclass
class Relation:
    """One record relation: N records, M candidate fields (integer codes after binning), a task key, outcome laws.

    laws: float64 N x B probability vectors (exact laws or empirical frequencies); shots: None for exact laws, else int N.
    """
    Z: np.ndarray                      # int32 N x M field codes (MISSING = -1)
    field_names: list[str]
    task: np.ndarray                   # int32 N
    laws: np.ndarray                   # float64 N x B
    shots: np.ndarray | None = None    # int64 N, or None for exact laws
    costs: np.ndarray | None = None    # float64 M (unit costs if None)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.Z = np.ascontiguousarray(self.Z, dtype=np.int32)
        self.task = np.ascontiguousarray(self.task, dtype=np.int32)
        self.laws = np.ascontiguousarray(self.laws, dtype=np.float64)
        if self.costs is None:
            self.costs = np.ones(self.Z.shape[1], dtype=np.float64)
        assert self.Z.shape[0] == self.laws.shape[0] == self.task.shape[0]
        assert self.Z.shape[1] == len(self.field_names)

    @property
    def N(self) -> int:
        return int(self.Z.shape[0])

    @property
    def M(self) -> int:
        return int(self.Z.shape[1])

    def radii(self, alpha: float) -> np.ndarray:
        """Per-record radius; zero for exact laws. The confidence family is all N laws of the relation."""
        if self.shots is None:
            return np.zeros(self.N)
        return confidence_radius(self.shots, self.laws.shape[1], self.N, alpha)

    def sha256(self) -> str:
        """Canonical relation hash (field names, codes, task keys, laws, shots, costs)."""
        h = hashlib.sha256()
        h.update("\x1f".join(self.field_names).encode())
        for arr in (self.Z, self.task, self.laws, self.costs):
            h.update(np.ascontiguousarray(arr).tobytes())
        if self.shots is not None:
            h.update(np.ascontiguousarray(self.shots, dtype=np.int64).tobytes())
        return h.hexdigest()
