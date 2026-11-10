"""K6 ladder attempt 3 on D2b (PR-K6-attempt3): pair bounds computed directly over the Agrawal KL confidence sets.

Rules compared on the same draws (counts ~ Multinomial(n, law), rng default_rng(80001 + n); records of the 3 most recent archive
epochs, N_rec = 96 = the real pilot's size; meaningful fields as d2b_mirror):
  v1  : sufprov Eq. (12) radius, triangle bounds (the frozen protocol rule);
  v2  : radius v2 (Agrawal threshold + local conversion), triangle bounds;
  kl  : lower bound = max(v2 triangle lo, KL-set convex lower bound, suffdisc.kl_lower);
        upper bound = min(v2 triangle hi, Lagrangian-dual bound over the KL sets, suffdisc.pair_upper).
All three are valid on their simultaneous events (v1: sufprov's; v2 / kl: the same Agrawal event). Per eps: u, certified
violations, irreducible pairs, the O-min set (greedy + prune hitting set of the possible-violation difference sets, i.e. pairs with
hi + GUARD > eps — exactly B3's family at this size), |S|, gamma, kappa, and removal witnesses (a certified violation whose difference
set meets S only in that field), plus an exhaustive search for fully witnessed non-vacuous sets of size <= 3 (K3(c) is
existential). Validity on D2b: every bound vs the true H for pairs whose two true laws lie in their KL sets.
Usage: python -m suffdisc.harness.k6_attempt3 <n_shots> <out.json>
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from suffdisc.core import GUARD, Relation, pair_bounds, pair_status, PASS, UNRESOLVED, VIOLATION
from suffdisc.engine.kernels import diff_masks, minimal_hitting_set_fast, n_words, witness_labels
from suffdisc.harness.k3_analysis import gamma_kappa, meaningful
from suffdisc.kl_lower import kl_lower
from suffdisc.pair_upper import pair_upper_many
from suffdisc.radius_v2 import kl_threshold, radius_v2

EPS = [0.05, 0.10, 0.15, 0.20]


def hell(P, Q):
    return np.sqrt(np.maximum(0.0, 1.0 - np.sqrt(P * Q).sum(axis=-1)))


def load_last3():
    d = np.load("data/sim/dsa_pilot.npz")
    info = json.load(open("data/sim/dsa_pilot.json"))
    names = [str(x) for x in d["names"]]
    epochs = [r["epoch"] for r in info["design"]]
    last3 = sorted(set(epochs))[-3:]
    rows = np.array([e in last3 for e in epochs])
    keep = meaningful(d["Z"], info["field_report"], names)
    Z = d["Z"][rows][:, keep]
    kept = [names[i] for i in keep]
    k2 = [i for i in range(Z.shape[1]) if len(np.unique(Z[:, i])) > 1]
    laws = d["laws"][rows]
    return Z[:, k2], [kept[i] for i in k2], laws / laws.sum(axis=1, keepdims=True), d["task"][rows].astype(np.int32)


def osets(Z, task, ii, jj, lo, hi, eps, names):
    W = n_words(Z.shape[1])
    poss = hi + GUARD > eps
    cert = lo > eps
    D = diff_masks(Z, ii[poss].astype(np.int64), jj[poss].astype(np.int64), W)
    irreducible = int((~D.any(axis=1)).sum())
    if D.shape[0] == 0:
        S, ok = [], True
    else:
        S, ok = minimal_hitting_set_fast(D, Z.shape[1])
    out = {"irreducible": irreducible, "exists": bool(ok)}
    if ok:
        Dc = diff_masks(Z, ii[cert].astype(np.int64), jj[cert].astype(np.int64), W)
        lab = witness_labels(S, Dc, W)
        g, k = gamma_kappa(Z, task, S)
        out.update({"S": [names[f] for f in S], "size": len(S), "gamma": g, "kappa": k,
                    "witnessed": {names[f]: bool(v) for f, v in lab.items()},
                    "all_witnessed": bool(S) and all(lab.values())})
    return out


def witnessed_search(Z, task, ii, jj, lo, hi, eps, names, kmax=3):
    """Exhaustive search over field sets of size 2..kmax: certified-sufficient (no possible-violation pair agrees on S),
    non-vacuous (gamma >= 2), every field with a removal witness (a certified violation agreeing on S minus that field).
    K3(c) asks whether such a set EXISTS; the greedy O-min above returns only one minimal set."""
    poss = hi + GUARD > eps
    cert = lo > eps
    Ag = (Z[ii] == Z[jj])
    Ap = Ag[poss].astype(np.int32)
    Ac = Ag[cert]
    M = Z.shape[1]
    found = []
    # size 2
    both = Ap.T @ Ap
    for f in range(M):
        for g in range(f + 1, M):
            if both[f, g] == 0:
                S = [f, g]
                if all((Ac[:, [x for x in S if x != k]].all(axis=1) & ~Ac[:, k]).any() for k in S):
                    gm, _ = gamma_kappa(Z, task, S)
                    if gm >= 2:
                        found.append(S)
    if kmax >= 3:
        for f in range(M):
            rows = Ap[:, f] == 1
            if not rows.any():
                continue
            sub = Ap[rows]
            cnt = sub.T @ sub
            for g in range(f + 1, M):
                if both[f, g] == 0:
                    continue  # {f,g} already sufficient => supersets not minimal
                for h in range(g + 1, M):
                    if cnt[g, h] == 0 and both[f, h] > 0 and both[g, h] > 0:
                        S = [f, g, h]
                        if all((Ac[:, [x for x in S if x != k]].all(axis=1) & ~Ac[:, k]).any() for k in S):
                            gm, _ = gamma_kappa(Z, task, S)
                            if gm >= 2:
                                found.append(S)
    best = sorted(found, key=lambda S: -gamma_kappa(Z, task, S)[0])[:5]
    return {"n_witnessed_sets": len(found), "examples": [[names[f] for f in S] + [round(gamma_kappa(Z, task, S)[0], 2)]
                                                        for S in best]}


def main(n, out_path):
    n = int(n)
    Z, names, laws, task = load_last3()
    N = len(laws)
    rng = np.random.default_rng(80001 + n)
    counts = np.stack([rng.multinomial(n, p) for p in laws])
    A = counts / n
    t = kl_threshold(n, 16, N, 0.05)
    with np.errstate(divide="ignore", invalid="ignore"):
        inside = np.where(A > 0, A * np.log(A / laws), 0.0).sum(axis=1) <= t
    ii_l, jj_l = [], []
    for tk in np.unique(task):
        idx = np.flatnonzero(task == tk)
        a, b = np.triu_indices(idx.size, 1)
        ii_l.append(idx[a])
        jj_l.append(idx[b])
    ii, jj = np.concatenate(ii_l), np.concatenate(jj_l)
    hh, htrue = hell(A[ii], A[jj]), hell(laws[ii], laws[jj])
    eta1 = Relation(Z=Z, field_names=names, task=task, laws=A, shots=np.full(N, n)).radii(0.05)
    eta2 = radius_v2(counts, N)
    lo1, hi1 = pair_bounds(hh, eta1[ii], eta1[jj])
    lo2, hi2 = pair_bounds(hh, eta2[ii], eta2[jj])
    t0 = time.perf_counter()
    hik = pair_upper_many(A, n, N, ii, jj, prior_hi=hi2, eps=min(EPS))
    t_up = time.perf_counter() - t0
    ev = inside[ii] & inside[jj]
    res = {"n": n, "N": N, "M": len(names), "pairs": int(ii.size), "t": t, "true_laws_inside": int(inside.sum()),
           "t_upper_s": t_up, "validity_upper_bad": int((hik[ev] + 1e-9 < htrue[ev]).sum()), "by_eps": {}}
    print(f"n {n}: pairs {ii.size}, true laws inside {inside.sum()}/{N}, dual upper bounds {t_up:.0f} s, "
          f"upper-bound validity violations {res['validity_upper_bad']}", flush=True)
    for e in EPS:
        lok = lo2.copy()
        st2 = pair_status(lo2, hi2, e)
        need = np.flatnonzero(st2 != VIOLATION)
        lok[need] = np.maximum(lo2[need], kl_lower(A, np.full(N, t), ii[need], jj[need], e))
        row = {}
        for rule, lo, hi in (("v1", lo1, hi1), ("v2", lo2, hi2), ("kl", lok, hik)):
            st = pair_status(lo, hi, e)
            r = {"u": float((st == UNRESOLVED).mean()), "certified": int((st == VIOLATION).sum()),
                 "pass": int((st == PASS).sum()), "validity_lower_bad": int((lo[ev] > htrue[ev] + 1e-9).sum())}
            r.update(osets(Z, task, ii, jj, lo, hi, e, names))
            r["search"] = witnessed_search(Z, task, ii, jj, lo, hi, e, names)
            row[rule] = r
            print(f"  eps {e} {rule:2s}: u {r['u']:.3f} cert {r['certified']:4d} pass {r['pass']:4d} | exists {r['exists']} "
                  f"S {r.get('S')} gamma {r.get('gamma', float('nan')):.2f} all_witnessed {r.get('all_witnessed')} "
                  f"lo-bad {r['validity_lower_bad']} | witnessed sets (|S|<=3, gamma>=2) {r['search']['n_witnessed_sets']} "
                  f"{r['search']['examples'][:2]}", flush=True)
        res["by_eps"][str(e)] = row
    json.dump(res, open(out_path, "w"), indent=1, default=float)


if __name__ == "__main__":
    main(*sys.argv[1:3])
