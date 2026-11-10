"""K6 ladder attempt 2 on D2b (PR-K6-attempt2): KL-set convex lower bound (suffdisc.kl_lower) on top of radius v2.

Same record selection, meaningful fields and rng as d2b_mirror (counts ~ Multinomial(n, law), rng default_rng(80001 + n)).
Per eps: v2 pair statuses (triangle lo / hi with v2 radii), then the KL-set lower bound on every v2-unresolved pair; a pair becomes a
certified violation iff the bound > eps. Reports u under v2 and under v2 + KL lower bound, certified violations, the oracle split of
v2-unresolved pairs by true H (> eps or <= eps), a validity check (bound <= true H for every evaluated pair whose two true laws lie
in their KL sets), and the removal labels of the O-min set S that the recorded v2 mirror run returned (S is unchanged because the
upper bounds are unchanged). Usage:
python -m suffdisc.harness.k6_attempt2 <n_shots> <last3|all> <mirror_v2.json> <out.json>
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from suffdisc.core import pair_bounds, pair_status, PASS, UNRESOLVED, VIOLATION
from suffdisc.harness.k3_analysis import meaningful
from suffdisc.kl_lower import kl_lower
from suffdisc.radius_v2 import kl_threshold, radius_v2

EPS = [0.05, 0.10, 0.15, 0.20]


def hell(P, Q):
    return np.sqrt(np.maximum(0.0, 1.0 - np.sqrt(P * Q).sum(axis=-1)))


def kl_rows(A, P):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(A > 0, A * np.log(A / P), 0.0)
    return t.sum(axis=1)


def main(n, rows_sel, mirror_path, out_path):
    n = int(n)
    d = np.load("data/sim/dsa_pilot.npz")
    info = json.load(open("data/sim/dsa_pilot.json"))
    names = [str(x) for x in d["names"]]
    epochs = [r["epoch"] for r in info["design"]]
    if rows_sel == "last3":
        last3 = sorted(set(epochs))[-3:]
        rows = np.array([e in last3 for e in epochs])
    else:
        rows = np.ones(len(epochs), dtype=bool)
    keep = meaningful(d["Z"], info["field_report"], names)
    Z = d["Z"][rows][:, keep]
    kept = [names[i] for i in keep]
    k2 = [i for i in range(Z.shape[1]) if len(np.unique(Z[:, i])) > 1]
    Z = Z[:, k2]
    kept = [kept[i] for i in k2]
    laws = d["laws"][rows]
    laws = laws / laws.sum(axis=1, keepdims=True)
    task = d["task"][rows].astype(np.int32)
    rng = np.random.default_rng(80001 + n)
    counts = np.stack([rng.multinomial(n, p) for p in laws])
    N = len(laws)
    A = counts / n
    eta = radius_v2(counts, N)
    t = np.full(N, kl_threshold(n, 16, N, 0.05))
    inside = kl_rows(A, laws) <= t  # true law inside its KL set (the event E restricted to this draw)
    mirror = json.load(open(mirror_path))
    ii_l, jj_l = [], []
    for tk in np.unique(task):
        idx = np.flatnonzero(task == tk)
        a, b = np.triu_indices(idx.size, 1)
        ii_l.append(idx[a])
        jj_l.append(idx[b])
    ii, jj = np.concatenate(ii_l), np.concatenate(jj_l)
    hh = hell(A[ii], A[jj])
    htrue = hell(laws[ii], laws[jj])
    lo, hi = pair_bounds(hh, eta[ii], eta[jj])
    res = {"n": n, "rows": rows_sel, "N": N, "M": len(kept), "pairs": int(ii.size), "t": float(t[0]),
           "true_laws_inside_sets": int(inside.sum()), "by_eps": {}}
    print(f"n {n} rows {rows_sel} N {N} pairs {ii.size} t {t[0]:.5f} true laws inside KL sets {inside.sum()}/{N}")
    for e in EPS:
        st = pair_status(lo, hi, e)
        unr = np.flatnonzero(st == UNRESOLVED)
        t0 = time.perf_counter()
        lb = kl_lower(A, t, ii[unr], jj[unr], e)
        dt = time.perf_counter() - t0
        st2 = st.copy()
        st2[unr[lb > e]] = VIOLATION
        ev = inside[ii[unr]] & inside[jj[unr]]
        bad = int((lb[ev] > htrue[unr][ev] + 1e-9).sum())
        S = mirror["by_eps"][str(e)]["S"]
        labels = {}
        if S is not None and mirror["by_eps"][str(e)].get("status") == "complete" and S:
            Si = [kept.index(f) for f in S]
            for k in Si:
                rest = [f for f in Si if f != k]
                agree = (Z[ii][:, rest] == Z[jj][:, rest]).all(axis=1) if rest else np.ones(ii.size, bool)
                labels[kept[k]] = {"v2": bool((agree & (st == VIOLATION)).any()), "v2+kl": bool((agree & (st2 == VIOLATION)).any())}
        row = {"u_v2": float((st == UNRESOLVED).mean()), "u_v2kl": float((st2 == UNRESOLVED).mean()),
               "cert_v2": int((st == VIOLATION).sum()), "cert_v2kl": int((st2 == VIOLATION).sum()),
               "unresolved_v2": int(unr.size), "unresolved_true_gt_eps": int((htrue[unr] > e).sum()),
               "validity_pairs_checked": int(ev.sum()), "validity_violations": bad, "t_bound_s": dt,
               "S": S, "witnessed": labels}
        res["by_eps"][str(e)] = row
        print(f"eps {e}: u v2 {row['u_v2']:.3f} -> v2+kl {row['u_v2kl']:.3f}; cert {row['cert_v2']} -> {row['cert_v2kl']}; "
              f"v2-unresolved with true H > eps {row['unresolved_true_gt_eps']}/{unr.size}; validity bad {bad}/{ev.sum()}; "
              f"{dt:.1f} s; S {S} witnessed {labels}", flush=True)
    json.dump(res, open(out_path, "w"), indent=1, default=float)


if __name__ == "__main__":
    main(*sys.argv[1:5])
