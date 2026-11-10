"""Shared kernels for B3 and SUFFDISC-0 (same primitives, so a speed difference comes from the algorithm, not the code):
GEMM Hellinger blocks, bit-packed difference sets, a superset-free cover, and the hitting-set back end.

Masks: a field set is a row of W = ceil(M / 64) uint64 words.
"""
from __future__ import annotations

import math
import time

import numba as nb
import numpy as np

from suffdisc.core import pair_bounds


def n_words(M: int) -> int:
    return (M + 63) // 64


def sqrt_embed(laws: np.ndarray) -> np.ndarray:
    return np.sqrt(np.asarray(laws, dtype=np.float64))


def hellinger_block(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Hellinger distances between rows of Ra and Rb (sqrt-embedded laws) via one float64 GEMM: H = sqrt(max(0, 1 - <a,b>))."""
    G = Ra @ Rb.T
    np.subtract(1.0, G, out=G)
    np.maximum(G, 0.0, out=G)
    np.sqrt(G, out=G)
    return G


@nb.njit(cache=True)
def diff_masks(Z, ia, ib, W):
    """Difference-set masks of the record pairs (ia[k], ib[k]); Z is int32 N x M (missing = -1 equals only itself)."""
    P = ia.shape[0]
    M = Z.shape[1]
    out = np.zeros((P, W), dtype=np.uint64)
    for k in range(P):
        a = ia[k]
        b = ib[k]
        for f in range(M):
            if Z[a, f] != Z[b, f]:
                out[k, f >> 6] |= np.uint64(1) << np.uint64(f & 63)
    return out


@nb.njit(cache=True)
def _popcount(x):
    c = 0
    while x:
        x &= x - np.uint64(1)
        c += 1
    return c


@nb.njit(cache=True)
def popcounts(masks):
    out = np.zeros(masks.shape[0], dtype=np.int64)
    for i in range(masks.shape[0]):
        c = 0
        for w in range(masks.shape[1]):
            c += _popcount(masks[i, w])
        out[i] = c
    return out


@nb.njit(cache=True)
def _is_subset(a, b):
    for w in range(a.shape[0]):
        if a[w] & ~b[w]:
            return False
    return True


@nb.njit(cache=True)
def _merge_into_cover(cover, n_cover, cand):
    """cover: K x W buffer whose first n_cover rows are superset-free; cand: candidates sorted by popcount ascending and
    deduplicated. Inserts every candidate that has no subset in the cover and drops cover rows that are supersets of it.
    Returns the new count (rows compacted in place; caller guarantees capacity)."""
    alive = np.ones(cover.shape[0], dtype=np.bool_)
    n = n_cover
    for i in range(cand.shape[0]):
        c = cand[i]
        dominated = False
        for j in range(n):
            if alive[j] and _is_subset(cover[j], c):
                dominated = True
                break
        if dominated:
            continue
        for j in range(n):
            if alive[j] and _is_subset(c, cover[j]):
                alive[j] = False
        cover[n] = c
        alive[n] = True
        n += 1
    k = 0
    for j in range(n):
        if alive[j]:
            if k != j:
                cover[k] = cover[j]
            k += 1
    return k


class CoverDeadline(Exception):
    pass


class Cover:
    """Superset-free family of difference sets (bit-packed), growing buffer."""

    def __init__(self, W: int, cap: int = 1024):
        self.W = W
        self.buf = np.zeros((cap, W), dtype=np.uint64)
        self.n = 0
        self.inserted = 0

    @property
    def sets(self) -> np.ndarray:
        return self.buf[: self.n]

    def add(self, masks: np.ndarray, deadline: float | None = None, chunk: int = 4096) -> int:
        """Add candidate masks (deduplicated, popcount-sorted, merged in chunks so a deadline can interrupt);
        returns 1 if the cover changed, else 0. Raises CoverDeadline when time.perf_counter() passes the deadline."""
        if masks.shape[0] == 0:
            return 0
        u = np.unique(masks, axis=0)
        u = u[np.argsort(popcounts(u), kind="stable")]
        self.inserted += int(u.shape[0])
        changed = 0
        for s in range(0, u.shape[0], chunk):
            part = u[s:s + chunk]
            need = self.n + part.shape[0]
            if need > self.buf.shape[0]:
                nb_ = np.zeros((max(need, 2 * self.buf.shape[0]), self.W), dtype=np.uint64)
                nb_[: self.n] = self.buf[: self.n]
                self.buf = nb_
            n_before = self.n
            before_sum = int(self.buf[: self.n].sum(dtype=np.uint64)) if self.n else 0
            self.n = _merge_into_cover(self.buf, self.n, part)
            if self.n != n_before or (self.n and int(self.buf[: self.n].sum(dtype=np.uint64)) != before_sum):
                changed = 1
            if deadline is not None and time.perf_counter() > deadline:
                raise CoverDeadline
        return changed


def mask_from_fields(fields, W: int) -> np.ndarray:
    m = np.zeros(W, dtype=np.uint64)
    for f in fields:
        m[f >> 6] |= np.uint64(1) << np.uint64(f & 63)
    return m


def fields_from_mask(m: np.ndarray) -> list[int]:
    out = []
    for w in range(m.shape[0]):
        x = int(m[w])
        while x:
            b = x & -x
            out.append(w * 64 + b.bit_length() - 1)
            x ^= b
    return out


@nb.njit(cache=True)
def _greedy_hitting(sets, M, costs):
    """Greedy weighted hitting set: repeatedly add the field with the best (#unhit sets it hits) / cost; then prune redundant
    fields in reverse insertion order. Returns an inclusion-minimal hitting set of `sets` (as a field list)."""
    K = sets.shape[0]
    hit = np.zeros(K, dtype=np.bool_)
    chosen = []
    remaining = K
    while remaining > 0:
        best_f = -1
        best_score = -1.0
        for f in range(M):
            w = f >> 6
            bit = np.uint64(1) << np.uint64(f & 63)
            cnt = 0
            for k in range(K):
                if not hit[k] and (sets[k, w] & bit):
                    cnt += 1
            if cnt > 0:
                sc = cnt / costs[f]
                if sc > best_score:
                    best_score = sc
                    best_f = f
        if best_f < 0:
            return chosen, False  # some set is empty: no hitting set exists (irreducible)
        chosen.append(best_f)
        w = best_f >> 6
        bit = np.uint64(1) << np.uint64(best_f & 63)
        for k in range(K):
            if not hit[k] and (sets[k, w] & bit):
                hit[k] = True
                remaining -= 1
    # prune: drop a field if every set stays hit without it
    i = len(chosen) - 1
    while i >= 0:
        f = chosen[i]
        ok = True
        for k in range(K):
            covered = False
            for g in chosen:
                if g != f and (sets[k, g >> 6] & (np.uint64(1) << np.uint64(g & 63))):
                    covered = True
                    break
            if not covered:
                ok = False
                break
        if ok:
            chosen.pop(i)
        i -= 1
    return chosen, True


def minimal_hitting_set(sets: np.ndarray, M: int, costs=None) -> tuple[list[int], bool]:
    """Inclusion-minimal hitting set of the family (greedy + prune). ok=False iff some set is empty (irreducible pairs)."""
    if sets.shape[0] == 0:
        return [], True
    c = np.ones(M) if costs is None else np.asarray(costs, dtype=np.float64)
    chosen, ok = _greedy_hitting(np.ascontiguousarray(sets), M, c)
    return sorted(int(x) for x in chosen), bool(ok)


def witness_labels(S: list[int], cert_cover: np.ndarray, W: int) -> dict[int, bool]:
    """Field f in S is certified needed if some certified-violation difference set D has D ∩ S = {f}."""
    if not S:
        return {}
    sm = mask_from_fields(S, W)
    out = {f: False for f in S}
    if cert_cover.shape[0] == 0:
        return out
    inter = cert_cover & sm[None, :]
    cnt = popcounts(inter)
    for k in np.flatnonzero(cnt == 1):
        f = fields_from_mask(inter[k])[0]
        out[f] = True
    return out


@nb.njit(cache=True)
def _append_nondominated(cover, n_cover, cand):
    """cand sorted by popcount ascending and deduplicated; cover rows have popcount <= every candidate. Append each candidate
    that contains no cover row (no deletions are ever needed in this order). Returns the new count."""
    n = n_cover
    for i in range(cand.shape[0]):
        c = cand[i]
        dominated = False
        for j in range(n):
            if _is_subset(cover[j], c):
                dominated = True
                break
        if not dominated:
            cover[n] = c
            n += 1
    return n


class BucketCover:
    """Superset-free cover built in increasing popcount order  'cardinality-bucketed
    baseline'): candidates must arrive in non-decreasing popcount across calls; no deletions."""

    def __init__(self, W: int, cap: int = 1024):
        self.W = W
        self.buf = np.zeros((cap, W), dtype=np.uint64)
        self.n = 0
        self.last_pop = -1

    @property
    def sets(self) -> np.ndarray:
        return self.buf[: self.n]

    def add_sorted(self, masks: np.ndarray, deadline: float | None = None, chunk: int = 4096) -> None:
        if masks.shape[0] == 0:
            return
        u = np.unique(masks, axis=0)
        pc = popcounts(u)
        order = np.argsort(pc, kind="stable")
        u, pc = u[order], pc[order]
        assert pc[0] >= self.last_pop, "BucketCover requires non-decreasing popcounts"
        self.last_pop = int(pc[-1])
        for s in range(0, u.shape[0], chunk):
            part = u[s:s + chunk]
            need = self.n + part.shape[0]
            if need > self.buf.shape[0]:
                nb_ = np.zeros((max(need, 2 * self.buf.shape[0]), self.W), dtype=np.uint64)
                nb_[: self.n] = self.buf[: self.n]
                self.buf = nb_
            self.n = _append_nondominated(self.buf, self.n, part)
            if deadline is not None and time.perf_counter() > deadline:
                raise CoverDeadline


class RawFamily:
    """Witnessed constraints kept as a deduplicated family, without superset-free maintenance : 'It is unnecessary to maintain H as a superset-free cover')."""

    def __init__(self, W: int):
        self.W = W
        self._parts = []
        self._sets = np.zeros((0, W), dtype=np.uint64)
        self._seen = set()

    @property
    def sets(self) -> np.ndarray:
        if self._parts:
            self._sets = np.concatenate([self._sets] + self._parts)
            self._parts = []
        return self._sets

    @property
    def n(self) -> int:
        return int(self.sets.shape[0])

    def add(self, masks: np.ndarray, deadline: float | None = None) -> int:
        if masks.shape[0] == 0:
            return 0
        u = np.unique(masks, axis=0)
        new = [i for i in range(u.shape[0]) if u[i].tobytes() not in self._seen]
        for i in new:
            self._seen.add(u[i].tobytes())
        if new:
            self._parts.append(u[new])
        return int(len(new) > 0)


# ---------------------------------------------------------------------------------------------------------------------------
# v3 kernels (2026-11-07, after tuning batch tune1v2 showed (i) B3 bound by the linear-scan dominance check and (ii) SUFFDISC-0
# bound by re-running the greedy hitting set from scratch). Shared back end: both methods use _greedy_hitting_fast.
# ---------------------------------------------------------------------------------------------------------------------------

@nb.njit(cache=True)
def _greedy_hitting_fast(sets, M, costs):
    """Same output as _greedy_hitting (same greedy rule, same first-maximum tie-break, same reverse-order prune), but with
    per-field counters of unhit sets and per-set hit counters: O(sum of set sizes) instead of O(|S| * M * K)."""
    K, W = sets.shape
    cnt = np.zeros(M, dtype=np.int64)
    for k in range(K):
        for w in range(W):
            x = sets[k, w]
            if x:
                for b in range(64):
                    if (x >> np.uint64(b)) & np.uint64(1):
                        cnt[w * 64 + b] += 1
    hit = np.zeros(K, dtype=np.bool_)
    chosen = []
    remaining = K
    while remaining > 0:
        best_f = -1
        best_score = -1.0
        for f in range(M):
            if cnt[f] > 0:
                sc = cnt[f] / costs[f]
                if sc > best_score:
                    best_score = sc
                    best_f = f
        if best_f < 0:
            return chosen, False
        chosen.append(best_f)
        wf = best_f >> 6
        bit = np.uint64(1) << np.uint64(best_f & 63)
        for k in range(K):
            if not hit[k] and (sets[k, wf] & bit):
                hit[k] = True
                remaining -= 1
                for w in range(W):
                    x = sets[k, w]
                    if x:
                        for b in range(64):
                            if (x >> np.uint64(b)) & np.uint64(1):
                                cnt[w * 64 + b] -= 1
    # prune in reverse insertion order with per-set counters (equivalent to the quadratic check of _greedy_hitting)
    c = np.zeros(K, dtype=np.int64)
    for g in chosen:
        wg = g >> 6
        bg = np.uint64(1) << np.uint64(g & 63)
        for k in range(K):
            if sets[k, wg] & bg:
                c[k] += 1
    i = len(chosen) - 1
    while i >= 0:
        f = chosen[i]
        wf = f >> 6
        bit = np.uint64(1) << np.uint64(f & 63)
        ok = True
        for k in range(K):
            if (sets[k, wf] & bit) and c[k] == 1:
                ok = False
                break
        if ok:
            for k in range(K):
                if sets[k, wf] & bit:
                    c[k] -= 1
            chosen.pop(i)
        i -= 1
    return chosen, True


def minimal_hitting_set_fast(sets: np.ndarray, M: int, costs=None) -> tuple[list[int], bool]:
    """Drop-in replacement of minimal_hitting_set (identical output), linear in the family size per greedy step."""
    if sets.shape[0] == 0:
        return [], True
    c = np.ones(M) if costs is None else np.asarray(costs, dtype=np.float64)
    chosen, ok = _greedy_hitting_fast(np.ascontiguousarray(sets), M, c)
    return sorted(int(x) for x in chosen), bool(ok)


@nb.njit(cache=True)
def _prune_order(sets, order):
    """Inclusion-minimal hitting subset of `order` (fields in insertion order; every set must be hit by some field of `order`):
    drop fields in reverse insertion order while every set stays hit. Returns a keep flag per position of `order`."""
    K = sets.shape[0]
    n = order.shape[0]
    c = np.zeros(K, dtype=np.int64)
    for i in range(n):
        g = order[i]
        wg = g >> 6
        bg = np.uint64(1) << np.uint64(g & 63)
        for k in range(K):
            if sets[k, wg] & bg:
                c[k] += 1
    keep = np.ones(n, dtype=np.bool_)
    for i in range(n - 1, -1, -1):
        f = order[i]
        wf = f >> 6
        bit = np.uint64(1) << np.uint64(f & 63)
        ok = True
        for k in range(K):
            if (sets[k, wf] & bit) and c[k] == 1:
                ok = False
                break
        if ok:
            keep[i] = False
            for k in range(K):
                if sets[k, wf] & bit:
                    c[k] -= 1
    return keep


@nb.njit(cache=True)
def _unhit(sets, smask):
    out = np.zeros(sets.shape[0], dtype=np.bool_)
    for k in range(sets.shape[0]):
        h = False
        for w in range(sets.shape[1]):
            if sets[k, w] & smask[w]:
                h = True
                break
        out[k] = not h
    return out


def extend_and_prune(all_sets: np.ndarray, order: list[int], M: int, costs=None) -> tuple[list[int], bool]:
    """Incremental minimal hitting set: keep the current fields (insertion order), hit the sets they miss with the shared greedy
    rule, then prune against the whole family. Returns (fields in insertion order, ok); ok=False iff some set is empty."""
    W = all_sets.shape[1]
    sm = mask_from_fields(order, W)
    miss = all_sets[_unhit(np.ascontiguousarray(all_sets), sm)]
    if miss.shape[0]:
        c = np.ones(M) if costs is None else np.asarray(costs, dtype=np.float64)
        add, ok = _greedy_hitting_fast(np.ascontiguousarray(miss), M, c)
        if not ok:
            return [], False
        order = list(order) + [int(f) for f in add]
    if not order:
        return [], True
    keep = _prune_order(np.ascontiguousarray(all_sets), np.asarray(order, dtype=np.int64))
    return [f for f, k in zip(order, keep) if k], True


@nb.njit(cache=True)
def _indexed_append(buf, idx, l1, dens, n, cand, M, i0, gorder):
    """Append (in order) every candidate of cand[i0:] that contains no stored set, until the buffer is full.
    Stored set j is indexed by its agree fields: bit j of idx[f] is set iff f is NOT in set j. A stored set S_j is a subset of a
    candidate c iff j lies in the intersection of idx[f] over the agree fields f of c (fields not in c). gorder = field order
    (sparsest posting first) used for the intersection. Returns (n, next candidate index)."""
    cap = buf.shape[0]
    W = buf.shape[1]
    A = np.empty(M, dtype=np.int64)
    for i in range(i0, cand.shape[0]):
        if n >= cap:
            return n, i
        c = cand[i]
        na = 0
        for t in range(M):
            f = gorder[t]
            if not ((c[f >> 6] >> np.uint64(f & 63)) & np.uint64(1)):
                A[na] = f
                na += 1
        dominated = False
        if n > 0:
            if na == 0:
                dominated = True
            else:
                nw = (n + 63) >> 6
                nl1 = (nw + 63) >> 6
                for j in range(nl1):
                    y = l1[A[0], j]
                    for t in range(1, na):
                        if y == 0:
                            break
                        y &= l1[A[t], j]
                    while y:
                        b = 0
                        z = y
                        while (z & np.uint64(1)) == 0:
                            z >>= np.uint64(1)
                            b += 1
                        y &= y - np.uint64(1)
                        w = j * 64 + b
                        if w >= nw:
                            continue
                        x = idx[A[0], w]
                        for t in range(1, na):
                            if x == 0:
                                break
                            x &= idx[A[t], w]
                        if x:
                            dominated = True
                            break
                    if dominated:
                        break
        if dominated:
            continue
        for w in range(W):
            buf[n, w] = c[w]
        wn = n >> 6
        bn = np.uint64(1) << np.uint64(n & 63)
        for t in range(na):
            f = A[t]
            idx[f, wn] |= bn
            l1[f, wn >> 6] |= np.uint64(1) << np.uint64(wn & 63)
            dens[f] += 1
        n += 1
    return n, cand.shape[0]


class IndexedCover:
    """Superset-free cover built in non-decreasing popcount order (same semantics and same resulting family as BucketCover),
    with a two-level bitmap inverted index on agree fields for the subset query : a
    subset-query index). Memory: about M * cap / 8 bytes for the index."""

    def __init__(self, W: int, M: int, cap: int = 1 << 16):
        cap = max(64, 1 << (int(cap) - 1).bit_length())  # power of two >= 64 (index words and level-1 words stay aligned)
        self.W, self.M = W, M
        self.buf = np.zeros((cap, W), dtype=np.uint64)
        self.idx = np.zeros((M, cap // 64), dtype=np.uint64)
        self.l1 = np.zeros((M, (cap // 64 + 63) // 64), dtype=np.uint64)
        self.dens = np.zeros(M, dtype=np.int64)
        self.n = 0
        self.last_pop = -1

    @property
    def sets(self) -> np.ndarray:
        return self.buf[: self.n]

    def _grow(self):
        cap = self.buf.shape[0] * 2
        nb_ = np.zeros((cap, self.W), dtype=np.uint64)
        nb_[: self.n] = self.buf[: self.n]
        ni = np.zeros((self.M, cap // 64), dtype=np.uint64)
        ni[:, : self.idx.shape[1]] = self.idx
        nl = np.zeros((self.M, (cap // 64 + 63) // 64), dtype=np.uint64)
        nl[:, : self.l1.shape[1]] = self.l1
        self.buf, self.idx, self.l1 = nb_, ni, nl

    def add_sorted(self, masks: np.ndarray, deadline: float | None = None, chunk: int = 1 << 15) -> None:
        if masks.shape[0] == 0:
            return
        u = np.unique(masks, axis=0)
        pc = popcounts(u)
        order = np.argsort(pc, kind="stable")
        u, pc = u[order], pc[order]
        assert pc[0] >= self.last_pop, "IndexedCover requires non-decreasing popcounts"
        self.last_pop = int(pc[-1])
        for s in range(0, u.shape[0], chunk):
            part = np.ascontiguousarray(u[s:s + chunk])
            i = 0
            while i < part.shape[0]:
                gorder = np.argsort(self.dens, kind="stable").astype(np.int64)
                self.n, i = _indexed_append(self.buf, self.idx, self.l1, self.dens, self.n, part, self.M, i, gorder)
                if i < part.shape[0]:
                    self._grow()
            if deadline is not None and time.perf_counter() > deadline:
                raise CoverDeadline
