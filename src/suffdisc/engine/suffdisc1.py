"""SUFFDISC-1 = SUFFDISC-0 v3 + validator knobs (2026-11-08, after review): `pivot` (single-pivot group
certificate) and `net` (radius-aware h-net validator, L1 audited item A2: leader clustering at radius h = net_h * eps, cluster-pair
certificates by the triangle inequality, exact pairwise fallback on uncertified cluster pairs => the same possible-violation set as
plain pairwise validation). pivot=False, net=False is the partial-cover baseline with ordinary pairwise validation (HPIValid
principle, Birnick et al. PVLDB 13). Development code: knobs not yet tuned.

SUFFDISC-0 — first prototype (v2 2026-11-07: partial constraints kept as a deduplicated raw family; v1 kept as suffdisc0_v1.py; v3 2026-11-07 after tuning batch tune1v2 showed >= 97 % of the time in re-running the
greedy hitting set from scratch on M = 800: the candidate is now maintained incrementally — keep the current fields, hit the newly
found constraints with the shared greedy rule, prune against the whole family (extend_and_prune); first candidate by the shared
fast greedy; v2 kept as suffdisc0_v2.py): focused pair sampling -> candidate minimal hitting set of the partial cover
-> validation of the candidate on the partition pi_S with ball-pivot certification -> add every violation found -> repeat.
Same kernels and hitting-set back end as B3 (engine/kernels.py).

Correctness (guide T2(i), to be proved / audited in L2): a candidate S that validates (every group of pi_S has upper-bound
diameter <= eps) is certified sufficient; S is a minimal hitting set of the partial cover C+ whose members are difference
sets of true possible violations, so every proper subset of S misses a true possible violation => O-min.
Knobs (tuned on tuning instances only): sample rounds, window, sort-key width, block size for in-group GEMM.
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np

from suffdisc.core import GUARD, Relation, pair_bounds
from suffdisc.engine.kernels import (RawFamily, CoverDeadline, diff_masks, extend_and_prune, hellinger_block, mask_from_fields, minimal_hitting_set_fast, n_words,
                                     popcounts, sqrt_embed, witness_labels)


class TimeLimit(Exception):
    pass


class TestAbort(Exception):
    pass


def _groups(Z: np.ndarray, task: np.ndarray, S: list[int], rows: np.ndarray | None = None):
    """Groups (size >= 2) of records equal on task and on every field of S; returns list of index arrays."""
    idx = np.arange(Z.shape[0]) if rows is None else rows
    keys = [Z[idx, f] for f in reversed(S)] + [task[idx]]
    order = np.lexsort(keys) if keys else np.arange(idx.size)
    sk = np.stack([k[order] for k in keys], axis=1)
    change = np.any(sk[1:] != sk[:-1], axis=1)
    bounds = np.concatenate(([0], np.flatnonzero(change) + 1, [idx.size]))
    out = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e - s >= 2:
            out.append(idx[order[s:e]])
    return out


NET_MARGIN = 1e-7  # computed H = sqrt(1 - <a,b>) can break the triangle inequality by ~1e-8; certificates keep this margin


def run_suffdisc1(rel: Relation, eps: float, alpha: float = 0.05, rounds: int = 8, window: int = 4, key_width: int = 6,
                  block: int = 1024, time_limit: float = 4 * 3600, seed: int = 0, ckpt_path: str | None = None,
                  ckpt_every: float = 600.0, _test_abort_after_first_ckpt: bool = False, pivot: bool = True,
                  net: bool = False, net_h: float = 0.125, net_min: int = 256, net_kmax_frac: float = 0.25,
                  rule: str = "v1") -> dict:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    W = n_words(rel.M)
    R = sqrt_embed(rel.laws)
    # one decision function for every pair test (certify.PairDecider): rule 'v1' (sufprov radii) or 'kl' (protocol §16), both
    # with floating-point inflation and canonical, order-independent decisions near thresholds
    from suffdisc.certify import PairDecider, TAU
    cert = PairDecider(rel.laws, rel.shots, alpha, rule=rule, eta_v1=None if rel.shots is None else rel.radii(alpha))
    eta = cert.eta
    MARG = cert.margin
    PRE = TAU + MARG  # prefilters on the uninflated GEMM triangle bound: a superset of every PairDecider decision
    bounds = lambda ia, ib, h: cert.bounds(ia, ib, h, eps)  # noqa: E731
    Z, task = rel.Z, rel.task
    plus, minus = RawFamily(W), RawFamily(W)  # v2: witnessed constraints, no superset-free maintenance ())
    st = {"n_dist": 0, "t_sample": 0.0, "t_validate": 0.0, "t_hitting": 0.0, "iterations": 0, "n_pivot_certified_groups": 0,
          "n_full_groups": 0, "irreducible_seen": 0, "n_boundary": 0, "n_net_groups": 0, "n_net_abandoned": 0,
          "n_net_pairs_certified": 0}
    # checkpoint state (protocol 14): taken only at loop-iteration boundaries, so a resumed run repeats at most one iteration
    cs = {"phase": "sample", "hs_order": [], "elapsed_prev": 0.0, "resumed": 0}
    if ckpt_path and os.path.exists(ckpt_path):
        snap = pickle.load(open(ckpt_path, "rb"))
        plus, minus, st, cs = snap["plus"], snap["minus"], snap["st"], snap["cs"]
        cs["resumed"] += 1
    t0 -= cs["elapsed_prev"]  # elapsed time carries over a restart
    deadline = t0 + time_limit
    last_ck = time.perf_counter()

    def check_time():
        if time.perf_counter() - t0 > time_limit:
            raise TimeLimit

    def checkpoint():
        nonlocal last_ck
        if ckpt_path and time.perf_counter() - last_ck >= ckpt_every:
            cs["elapsed_prev"] = time.perf_counter() - t0
            tmp = ckpt_path + ".tmp"
            pickle.dump({"plus": plus, "minus": minus, "st": st, "cs": cs}, open(tmp, "wb"))
            if os.path.exists(ckpt_path):
                os.replace(ckpt_path, ckpt_path + ".last-good")
            os.replace(tmp, ckpt_path)
            last_ck = time.perf_counter()
            if os.environ.get("SUFFDISC_TEST_HARD_EXIT_AFTER_CKPT") == "1" and cs["resumed"] == 0:
                os._exit(9)  # resume test only: simulates a kill right after the first checkpoint
            if _test_abort_after_first_ckpt and cs["resumed"] == 0:
                raise TestAbort  # D4 part 1b only: in-process stand-in for a kill right after the first checkpoint

    def add_pairs(ia, ib, h):
        lo, hi = bounds(ia, ib, h)
        poss = hi + GUARD > eps  # protocol §3 reporting guard
        st["n_boundary"] += int(((hi <= eps) & poss & (lo <= eps)).sum())
        if not poss.any():
            return 0
        m = diff_masks(Z, ia[poss].astype(np.int64), ib[poss].astype(np.int64), W)
        certv = (lo[poss] > eps)
        st["irreducible_seen"] += int(((popcounts(m) == 0) & certv).sum())
        ch = plus.add(m, deadline)
        if certv.any():
            minus.add(m[certv], deadline)
        return int(poss.sum()) if ch or True else 0


    def _net_validate(g):
        """h-net validation of one group (exact: returns after adding every possible violation of the group, like pairwise)."""
        n = g.size
        h = net_h * eps
        cl = np.full(n, -1, dtype=np.int64)
        rad = np.zeros(n)
        centers = []
        rem = np.arange(n)
        kmax = max(1, int(net_kmax_frac * n))
        while rem.size:
            check_time()
            c = rem[0]
            dd = np.sqrt(np.maximum(0.0, 1.0 - R[g[rem]] @ R[g[c]]))
            st["n_dist"] += rem.size
            w = dd <= h
            w[0] = True
            cl[rem[w]] = len(centers)
            rad[rem[w]] = dd[w]
            centers.append(c)
            rem = rem[~w]
            if len(centers) > kmax:
                st["n_net_abandoned"] += 1
                return None  # no useful net: plain pairwise
        st["n_net_groups"] += 1
        k = len(centers)
        rk = np.zeros(k)
        ek = np.zeros(k)
        np.maximum.at(rk, cl, rad)
        np.maximum.at(ek, cl, eta[g])
        Ck = hellinger_block(R[g[centers]], R[g[centers]])
        st["n_dist"] += k * (k - 1) // 2
        U = Ck + rk[:, None] + rk[None, :] + ek[:, None] + ek[None, :]
        np.fill_diagonal(U, 2 * rk + 2 * ek)
        need = U + max(NET_MARGIN, MARG) > eps - GUARD
        members = [np.flatnonzero(cl == a) for a in range(k)]
        sizes = np.array([m.size for m in members])
        tot = 0
        for a in range(k):
            bs = np.flatnonzero(need[a, a:]) + a
            checked = int(sizes[a] * sizes[bs[bs > a]].sum() + (sizes[a] * (sizes[a] - 1) // 2 if need[a, a] else 0))
            st["n_net_pairs_certified"] += int(sizes[a] * sizes[a + 1:].sum() + sizes[a] * (sizes[a] - 1) // 2) - checked
            if bs.size == 0:
                continue
            ys = np.concatenate([members[b] for b in bs])
            for x0 in range(0, sizes[a], block):
                xa = members[a][x0:x0 + block]
                check_time()
                for s0 in range(0, ys.size, block):
                    yb = ys[s0:s0 + block]
                    keep = (cl[yb][None, :] != a) | (yb[None, :] > xa[:, None])
                    if not keep.any():
                        continue
                    H = hellinger_block(R[g[xa]], R[g[yb]])
                    st["n_dist"] += int(keep.sum())
                    lo, hi = pair_bounds(H, eta[g[xa]][:, None], eta[g[yb]][None, :])
                    pa, pb = np.nonzero(keep & (hi + GUARD + PRE > eps))
                    if pa.size:
                        tot += add_pairs(g[xa[pa]], g[yb[pb]], H[pa, pb])
        return tot

    status = "complete"
    S: list[int] = []
    ok = True
    try:
        # 1. focused sampling (HyFD-style): sort by random field subsets, compare near neighbours, within task
        ts = time.perf_counter()
        for _ in range(rounds if cs["phase"] == "sample" else 0):
            fields = rng.choice(rel.M, size=min(key_width, rel.M), replace=False)
            order = np.lexsort([Z[:, f] for f in fields] + [task])
            for w in range(1, window + 1):
                ia, ib = order[:-w], order[w:]
                same = task[ia] == task[ib]
                ia, ib = ia[same], ib[same]
                h = np.sqrt(np.maximum(0.0, 1.0 - np.einsum("ij,ij->i", R[ia], R[ib])))
                st["n_dist"] += ia.size
                add_pairs(ia, ib, h)
            check_time()
        if cs["phase"] == "sample":
            st["t_sample"] = time.perf_counter() - ts
            cs["phase"] = "loop"
        hs_order = cs["hs_order"]
        # 2. candidate -> validate -> refine
        while True:
            st["iterations"] += 1
            th = time.perf_counter()
            if (popcounts(plus.sets) == 0).any():
                ok = False
                S = []
                st["t_hitting"] += time.perf_counter() - th
                break
            if not hs_order and st["iterations"] == 1:
                S, ok = minimal_hitting_set_fast(plus.sets, rel.M, rel.costs)
                hs_order = list(S)
            else:
                hs_order, ok = extend_and_prune(plus.sets, hs_order, rel.M, rel.costs)
                S = sorted(hs_order)
            st["t_hitting"] += time.perf_counter() - th
            tv = time.perf_counter()
            found = 0
            for g in _groups(Z, task, S):
                check_time()
                if pivot:
                    c = g[0]
                    d = np.sqrt(np.maximum(0.0, 1.0 - R[g] @ R[c]))
                    st["n_dist"] += g.size - 1
                    if (d + eta[g]).max() + eta[c] + MARG <= (eps - GUARD) / 2:
                        st["n_pivot_certified_groups"] += 1
                        continue
                if net and g.size >= net_min:
                    r = _net_validate(g)
                    if r is not None:
                        found += r
                        continue
                st["n_full_groups"] += 1
                for a in range(0, g.size, block):
                    ga = g[a:a + block]
                    for b in range(a, g.size, block):
                        gb = g[b:b + block]
                        H = hellinger_block(R[ga], R[gb])
                        st["n_dist"] += H.size if a != b else ga.size * (ga.size - 1) // 2
                        lo, hi = pair_bounds(H, eta[ga][:, None], eta[gb][None, :])
                        poss = hi + GUARD + PRE > eps  # superset; add_pairs decides with the canonical predicate
                        if a == b:
                            poss = np.triu(poss, k=1)
                        pa, pb = np.nonzero(poss)
                        if pa.size:
                            found += add_pairs(ga[pa], gb[pb], H[pa, pb])
            st["t_validate"] += time.perf_counter() - tv
            if found == 0:
                break
            cs["hs_order"] = hs_order
            checkpoint()
        # 3. labels: witnesses from the certified cover; missing ones searched on pi_{S \ {f}} (cross pairs only)
        labels = witness_labels(S, minus.sets, W) if ok else {}
        tl = time.perf_counter()
        for f in [f for f, v in labels.items() if not v]:
            rest = [x for x in S if x != f]
            for g in _groups(Z, task, rest):
                check_time()
                vals = Z[g, f]
                if np.unique(vals).size < 2:
                    continue
                sm = mask_from_fields(S, W)
                hit = False
                for a in range(0, g.size, block):
                    ga = g[a:a + block]
                    for b in range(a, g.size, block):
                        gb = g[b:b + block]
                        check_time()
                        H = hellinger_block(R[ga], R[gb])
                        _, hi_t = pair_bounds(H, eta[ga][:, None], eta[gb][None, :])
                        cross = Z[ga, f][:, None] != Z[gb, f][None, :]
                        if a == b:
                            cross &= np.triu(np.ones_like(cross), k=1)
                        st["n_dist"] += int(cross.sum())
                        pa, pb = np.nonzero(cross & (hi_t + PRE > eps))  # lo <= hi_t: only these can be certified violations
                        if pa.size:
                            lo_r, _ = bounds(ga[pa], gb[pb], H[pa, pb])
                            keep = lo_r > eps
                            pa, pb = pa[keep], pb[keep]
                        if pa.size:
                            m = diff_masks(Z, ga[pa].astype(np.int64), gb[pb].astype(np.int64), W)
                            minus.add(m, deadline)
                            if (popcounts(m & sm[None, :]) == 1).any():
                                hit = True
                                break
                    if hit:
                        break
                if hit:
                    labels[f] = True
                    break
        st["t_labels"] = time.perf_counter() - tl
    except (TimeLimit, CoverDeadline):
        status = "timeout"
        labels = {}
    out = {"method": "SUFFDISC-1", "version": "v1-net", "validator": {"pivot": pivot, "net": net, "net_h": net_h,
           "net_min": net_min, "net_kmax_frac": net_kmax_frac}, "rule": rule,
           "kl_refined_pairs": cert.n_refined, "kl_upper_nm": cert.n_upper_nm, "n_canonical": cert.n_canonical, "status": status, "eps": eps, "knobs": {"rounds": rounds, "window": window,
           "key_width": key_width, "block": block, "seed": seed}, "cover_plus": plus.n, "cover_minus": minus.n, "resumed": cs["resumed"],
           **st}
    if status == "complete":
        out.update({"S": S, "certified_sufficient_exists": ok,
                    "labels": {int(k): ("certified_needed" if v else "minimality_unresolved") for k, v in labels.items()}})
    out["t_total"] = time.perf_counter() - t0
    return out
