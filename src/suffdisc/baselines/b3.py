"""B3 — strong naive baseline, v2 (2026-11-07, cardinality-bucketed exact baseline): blockwise float64 GEMM all-pairs Hellinger within tasks -> bit-packed difference sets of possible / certified
violations -> superset-free covers D+ / D- built in increasing popcount order (no deletions; memory-bounded popcount ranges,
one streaming pass per range, distances recomputed because they are cheap) -> minimal hitting set (shared back end).
v1 (linear superset-free maintenance with deletions) is kept as b3_v1.py for the record.
v3 (2026-11-07, after tuning batch tune1v2: every far / mixed case timed out in the linear-scan dominance check): the cover is an
IndexedCover (two-level bitmap inverted index on agree fields; identical resulting family), the hitting set uses the shared fast
greedy (identical output), and the per-range memory budget is in bytes (1.5 GB of masks) instead of rows. v2 kept as b3_v2.py.

Knobs: block size; mask budget per range (tuned on tuning instances only). Time limit and checkpoints per protocol §11 / §14.
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np

from suffdisc.core import GUARD, Relation, pair_bounds
from suffdisc.engine.kernels import (IndexedCover, CoverDeadline, diff_masks, hellinger_block, minimal_hitting_set_fast, n_words,
                                     popcounts, sqrt_embed, witness_labels)


class TimeLimit(Exception):
    pass


def _blocks(rel: Relation, block: int):
    out = []
    for t in np.unique(rel.task):
        idx = np.flatnonzero(rel.task == t)
        starts = list(range(0, idx.size, block))
        for ai, a in enumerate(starts):
            for b in starts[ai:]:
                out.append((idx[a:a + block], idx[b:b + block], a == b))
    return out


def run_b3(rel: Relation, eps: float, alpha: float = 0.05, block: int = 1024, mask_bytes: float = 1.5e9,
           time_limit: float = 4 * 3600, ckpt_path: str | None = None, ckpt_every: float = 600.0, cover_cap: int = 1 << 16,
           _mutation_extra_pairs=None) -> dict:
    """cover_cap: initial IndexedCover capacity (tests force growth with a small value). _mutation_extra_pairs: D4 mutation M4
    only — (a, b) record pairs compared regardless of task (never used in a real run)."""
    t_start = time.perf_counter()
    W = n_words(rel.M)
    R = sqrt_embed(rel.laws)
    eta = rel.radii(alpha)
    Z = rel.Z
    mask_budget = max(1, int(mask_bytes // (8 * W)))
    st = {"phase": "hist", "range_i": 0, "block_i": 0, "hist_plus": np.zeros(rel.M + 1, np.int64),
          "hist_minus": np.zeros(rel.M + 1, np.int64), "plus": IndexedCover(W, rel.M, cover_cap), "minus": IndexedCover(W, rel.M, cover_cap), "coll_p": [],
          "coll_m": [], "n_dist": 0, "n_poss": 0, "n_cert": 0, "n_irreducible": 0, "n_empty_plus": 0, "t_dist": 0.0,
          "t_masks": 0.0, "t_cover": 0.0, "elapsed_prev": 0.0, "ranges": None, "passes": 0, "n_boundary": 0}
    if ckpt_path and os.path.exists(ckpt_path):
        st = pickle.load(open(ckpt_path, "rb"))
        st["resumed"] = st.get("resumed", 0) + 1
    deadline = t_start + time_limit - st["elapsed_prev"]
    last_ck = time.perf_counter()

    def elapsed():
        return st["elapsed_prev"] + time.perf_counter() - t_start

    def checkpoint(force=False):
        nonlocal last_ck
        if ckpt_path and (force or time.perf_counter() - last_ck >= ckpt_every):
            snap = dict(st)
            snap["elapsed_prev"] = elapsed()
            tmp = ckpt_path + ".tmp"
            pickle.dump(snap, open(tmp, "wb"))
            if os.path.exists(ckpt_path):
                os.replace(ckpt_path, ckpt_path + ".last-good")
            os.replace(tmp, ckpt_path)
            last_ck = time.perf_counter()

    def block_pairs(ia, ib, diag, count):
        t1 = time.perf_counter()
        H = hellinger_block(R[ia], R[ib])
        if count:
            st["n_dist"] += H.size if not diag else ia.size * (ia.size - 1) // 2
        lo, hi = pair_bounds(H, eta[ia][:, None], eta[ib][None, :])
        poss = hi + GUARD > eps  # protocol §3 reporting guard
        if diag:
            poss = np.triu(poss, k=1)
        pa, pb = np.nonzero(poss)
        cert = lo[pa, pb] > eps
        if count:
            st["n_boundary"] += int(((hi[pa, pb] <= eps) & ~cert).sum())
        t2 = time.perf_counter()
        st["t_dist"] += t2 - t1
        m = diff_masks(Z, ia[pa].astype(np.int64), ib[pb].astype(np.int64), W) if pa.size else np.zeros((0, W), np.uint64)
        st["t_masks"] += time.perf_counter() - t2
        return m, cert

    status = "complete"
    try:
        blocks = _blocks(rel, block)
        for a_, b_ in (_mutation_extra_pairs or []):
            blocks.append((np.array([a_]), np.array([b_]), False))
        if st["phase"] == "hist":
            for bi in range(st["block_i"], len(blocks)):
                ia, ib, diag = blocks[bi]
                m, cert = block_pairs(ia, ib, diag, True)
                if m.shape[0]:
                    pc = popcounts(m)
                    st["hist_plus"] += np.bincount(pc, minlength=rel.M + 1)
                    st["hist_minus"] += np.bincount(pc[cert], minlength=rel.M + 1)
                    st["n_poss"] += int(m.shape[0])
                    st["n_cert"] += int(cert.sum())
                    st["n_empty_plus"] += int((pc == 0).sum())
                    st["n_irreducible"] += int(((pc == 0) & cert).sum())
                st["block_i"] = bi + 1
                if elapsed() > time_limit:
                    raise TimeLimit
                checkpoint()
            ranges, lo_p, acc = [], 0, 0
            for p in range(rel.M + 1):
                acc += int(st["hist_plus"][p])
                if acc > mask_budget and p > lo_p:
                    ranges.append((lo_p, p))
                    lo_p, acc = p, int(st["hist_plus"][p])
            ranges.append((lo_p, rel.M + 1))
            st.update({"phase": "ranges", "ranges": ranges, "range_i": 0, "block_i": 0, "coll_p": [], "coll_m": []})
            checkpoint(force=True)
        while st["range_i"] < len(st["ranges"]):
            r_lo, r_hi = st["ranges"][st["range_i"]]
            if st["hist_plus"][r_lo:r_hi].sum() > 0:
                for bi in range(st["block_i"], len(blocks)):
                    ia, ib, diag = blocks[bi]
                    m, cert = block_pairs(ia, ib, diag, False)
                    if m.shape[0]:
                        pc = popcounts(m)
                        sel = (pc >= r_lo) & (pc < r_hi)
                        if sel.any():
                            st["coll_p"].append(np.unique(m[sel], axis=0))
                            if (sel & cert).any():
                                st["coll_m"].append(np.unique(m[sel & cert], axis=0))
                    st["block_i"] = bi + 1
                    if elapsed() > time_limit:
                        raise TimeLimit
                    checkpoint()
                t3 = time.perf_counter()
                if st["coll_p"]:
                    st["plus"].add_sorted(np.concatenate(st["coll_p"]), deadline)
                if st["coll_m"]:
                    st["minus"].add_sorted(np.concatenate(st["coll_m"]), deadline)
                st["t_cover"] += time.perf_counter() - t3
                st["passes"] += 1
            st.update({"range_i": st["range_i"] + 1, "block_i": 0, "coll_p": [], "coll_m": []})
            checkpoint(force=True)
    except (TimeLimit, CoverDeadline):
        status = "timeout"
    out = {"method": "B3", "version": "v3-indexed", "status": status, "eps": eps, "block": block, "mask_bytes": mask_bytes, "mask_budget_rows": mask_budget,
           "n_dist": st["n_dist"], "n_possible_viol_pairs": st["n_poss"], "n_certified_viol_pairs": st["n_cert"],
           "n_irreducible_pairs": st["n_irreducible"], "cover_plus": st["plus"].n, "cover_minus": st["minus"].n,
           "passes": st["passes"], "t_dist": st["t_dist"], "t_masks": st["t_masks"], "t_cover": st["t_cover"],
           "resumed": st.get("resumed", 0), "phase_at_end": st["phase"], "n_boundary_pair_evals": st.get("n_boundary", 0)}
    if status == "complete":
        t4 = time.perf_counter()
        if st["n_empty_plus"]:
            S, ok = [], False
        else:
            S, ok = minimal_hitting_set_fast(st["plus"].sets, rel.M, rel.costs)
        labels = witness_labels(S, st["minus"].sets, W) if ok else {}
        out.update({"t_hitting": time.perf_counter() - t4, "S": S, "certified_sufficient_exists": ok,
                    "labels": {int(k): ("certified_needed" if v else "minimality_unresolved") for k, v in labels.items()}})
    out["t_total"] = st["elapsed_prev"] + time.perf_counter() - t_start
    return out
