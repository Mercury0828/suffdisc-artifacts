"""D4 (K5): C6 vs B0 vs B3 / SUFFDISC-0 on the 1,000 small planted instances (protocol §9), plus mutation tests M1-M4.
PR-D4. Usage: python -m suffdisc.harness.d4_validator <out_jsonl> <n_procs>
r1 (2026-11-07, after audit L3/D4-validator R1; r0 kept as d4_validator_r0.py): reporting guard in B0 (U + 1e-10 <= eps);
M3 / M4 as pre-registered (one record's radius set to 0; one cross-task pair compared), the r0 global variants kept and labelled
M3g / M4g; mutation detection includes label disagreement; C6 vs B0 on ALL B0 minimal sets with their single deletions; C6 on the
full field set in no-solution cases.
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool

import numpy as np

from suffdisc.baselines.b0 import pair_table
from suffdisc.baselines.b3 import run_b3
from suffdisc.core import GUARD, Relation, pair_bounds
from suffdisc.engine.suffdisc0 import run_suffdisc0
from suffdisc.simulators.planted import generate
from suffdisc.validator.c6 import C6

EPS = [0.05, 0.10, 0.20]


def instance(seed: int) -> Relation:
    rng = np.random.default_rng(seed)
    fam = "far" if rng.random() < 0.5 else "near"
    M = int(rng.integers(8, 16))
    N = int(rng.integers(50, 301))
    rel, _ = generate(fam, M, N, seed)
    rel.meta = {"family": fam, "M": M, "N": N, "seed": seed, "shots": None}
    if seed % 2 == 1:
        n = 200 if seed % 4 == 1 else 1000
        r2 = np.random.default_rng(seed + 10**6)
        counts = np.stack([r2.multinomial(n, p / p.sum()) for p in rel.laws])
        rel = Relation(Z=rel.Z, field_names=rel.field_names, task=rel.task, laws=counts / n,
                       shots=np.full(rel.N, n), meta={**rel.meta, "shots": n})
    return rel


def b0_minimal(rel: Relation, eps: float):
    """Minimal certified-sufficient sets via the separation rule on all subsets (vectorised)."""
    masks, lo, hi = pair_table(rel)
    poss = np.unique(masks[hi + GUARD > eps])
    subsets = np.arange(1 << rel.M, dtype=np.uint64)
    if poss.size == 0:
        suff = np.ones(subsets.size, dtype=bool)
    else:
        suff = ~((subsets[:, None] & poss[None, :]) == 0).any(axis=1)
    suff_set = set(int(s) for s in np.flatnonzero(suff))
    mins = {s for s in suff_set if not any((s & ~(1 << k)) in suff_set for k in range(rel.M) if (s >> k) & 1)}
    return mins, bool(suff.any())


def pair_table_idx(rel: Relation):
    """Within-task pairs with their difference masks (M <= 62 so one int64 word) and bounds."""
    eta = rel.radii(0.05)
    R = np.sqrt(rel.laws)
    A, B = [], []
    for t in np.unique(rel.task):
        idx = np.flatnonzero(rel.task == t)
        i, j = np.triu_indices(idx.size, 1)
        A.append(idx[i])
        B.append(idx[j])
    ia = np.concatenate(A) if A else np.zeros(0, int)
    ib = np.concatenate(B) if B else np.zeros(0, int)
    h = np.sqrt(np.maximum(0.0, 1.0 - np.einsum("ij,ij->i", R[ia], R[ib])))
    lo, hi = pair_bounds(h, eta[ia], eta[ib])
    w = (1 << np.arange(rel.M, dtype=np.int64))
    masks = ((rel.Z[ia] != rel.Z[ib]) * w).sum(axis=1)
    return masks, lo, hi, ia, ib


def cross_task_candidate(rel: Relation, eps: float, S):
    eta = rel.radii(0.05)
    R = np.sqrt(rel.laws)
    sm = sum(1 << k for k in S)
    w = (1 << np.arange(rel.M, dtype=np.int64))
    for a in range(rel.N):
        b = np.arange(a + 1, rel.N)
        b = b[rel.task[b] != rel.task[a]]
        if b.size == 0:
            continue
        h = np.sqrt(np.maximum(0.0, 1.0 - R[b] @ R[a]))
        hi = np.minimum(1.0, h + eta[a] + eta[b])
        d = ((rel.Z[b] != rel.Z[a]) * w).sum(axis=1)
        ok = np.flatnonzero((hi + GUARD > eps) & (d != 0) & ((d & sm) == 0))
        if ok.size:
            return a, int(b[ok[0]])
    return None, None


def one(seed: int) -> dict:
    rel = instance(seed)
    c6 = C6(rel.Z, rel.task, rel.laws, rel.shots)
    out = {"seed": seed, **rel.meta, "cases": []}
    for eps in EPS:
        mins, exists = b0_minimal(rel, eps)
        case = {"eps": eps, "b0_exists": exists, "n_b0_minimal": len(mins), "methods": {}}
        for name, fn in [("B3", lambda: run_b3(rel, eps)), ("SUFFDISC-0", lambda: run_suffdisc0(rel, eps, seed=seed))]:
            r = fn()
            S = r["S"]
            m = {"S": S, "exists": r["certified_sufficient_exists"]}
            if r["certified_sufficient_exists"]:
                chk = c6.check_output(S, eps)
                m["in_b0_minimal"] = sum(1 << k for k in S) in mins
                m["c6_status"] = chk["status"]
                m["labels_agree"] = all((r["labels"][k] == "certified_needed") == (chk["labels"][k] == "certified_needed")
                                        for k in S)
                m["no_removable"] = chk["no_removable_field"]
                m["ok"] = bool(m["in_b0_minimal"] and chk["status"] == "sufficient" and m["labels_agree"] and m["no_removable"])
            else:
                m["ok"] = (not exists)
            case["methods"][name] = m
        # C6 vs B0 on the B0 minimal sets (and their single deletions): status must agree with the separation rule
        dis = 0
        for s in mins:
            S = [k for k in range(rel.M) if (s >> k) & 1]
            if c6.status(S, eps) != "sufficient":
                dis += 1
            for k in S:  # every single deletion of a B0-minimal set must not be C6-sufficient
                if c6.status([f for f in S if f != k], eps) == "sufficient":
                    dis += 1
        if not exists and c6.status(list(range(rel.M)), eps) == "sufficient":
            dis += 1  # no-solution case: C6 must not find the full field set sufficient
        case["c6_b0_disagreements"] = dis
        case["c6_b0_checks"] = len(mins) + sum(bin(s).count("1") for s in mins) + (0 if exists else 1)
        # mutations (on the first eps only, to bound cost)
        if eps == EPS[1] and case["methods"]["B3"].get("S"):
            S = case["methods"]["B3"]["S"]
            mut = {}
            mut["M1_drop_field"] = c6.status(S[:-1], eps) != "sufficient"
            Zm = rel.Z.copy()
            k = S[0]
            Zm[0, k] = Zm[0, k] + 1  # mis-binned value of one record in a returned field
            r_m = run_b3(Relation(Z=Zm, field_names=rel.field_names, task=rel.task, laws=rel.laws, shots=rel.shots), eps)
            def flagged(r):
                if not r["certified_sufficient_exists"]:
                    return exists
                chk = c6.check_output(r["S"], eps)
                lab = all((r["labels"][k] == "certified_needed") == (chk["labels"][k] == "certified_needed") for k in r["S"])
                return chk["status"] != "sufficient" or not chk["no_removable_field"] or not lab
            mut["M2_misbin"] = flagged(run_b3(Relation(Z=Zm, field_names=rel.field_names, task=rel.task, laws=rel.laws,
                                                       shots=rel.shots), eps))
            masks, lo, hi, ia, ib = pair_table_idx(rel)
            if rel.shots is not None:
                # M3 (pre-registered): the radius of ONE record set to 0 inside the method - the record with the most
                # UNRESOLVED within-task pairs at this eps (ties: smallest index)
                unres = (hi + GUARD > eps) & (lo <= eps)
                cnt = np.bincount(np.concatenate([ia[unres], ib[unres]]), minlength=rel.N)
                if cnt.max() > 0:
                    r0 = int(np.argmax(cnt))
                    sh = np.asarray(rel.shots, dtype=np.float64).copy()
                    sh[r0] = 1e18
                    mut["M3_radius_zero"] = flagged(run_b3(Relation(Z=rel.Z, field_names=rel.field_names, task=rel.task,
                                                                    laws=rel.laws, shots=sh), eps))
                else:
                    mut["M3_radius_zero"] = None  # no unresolved pair: not testable on this case
                mut["M3g_all_radii_zero"] = flagged(run_b3(Relation(Z=rel.Z, field_names=rel.field_names, task=rel.task,
                                                                    laws=rel.laws, shots=np.full(rel.N, 10**12)), eps))
            # M4 (pre-registered): ONE cross-task pair compared by the method - the first (a < b) cross-task pair whose upper
            # bound exceeds eps and whose nonempty difference set misses the returned set S (otherwise it cannot matter)
            ca, cb = cross_task_candidate(rel, eps, S)
            if ca is not None:
                mut["M4_cross_task"] = flagged(run_b3(rel, eps, _mutation_extra_pairs=[(ca, cb)]))
            else:
                mut["M4_cross_task"] = None
            mut["M4g_all_tasks_merged"] = flagged(run_b3(Relation(Z=rel.Z, field_names=rel.field_names,
                                                                  task=np.zeros(rel.N, dtype=np.int32), laws=rel.laws,
                                                                  shots=rel.shots), eps))
            case["mutations_caught"] = mut
        out["cases"].append(case)
    return out


def main(out_path: str, procs: int) -> None:
    seeds = list(range(61001, 62001))
    with Pool(procs) as pool, open(out_path, "w", encoding="utf-8") as fh:
        for res in pool.imap_unordered(one, seeds, chunksize=4):
            fh.write(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
            fh.flush()


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
