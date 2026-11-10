"""D2 (K3) and D5 (K6) analysis of record for the real pilot (protocol v1.2 §7, §12, §16). Written and tested on fake records
before any pilot counts file was opened.

Input: relation.npz from suffdisc.collectors.pilot_assemble (Z, task, names, counts, shots) and relation.json (field report).
Meaningful fields = the loader's report (§5 / K3(a)). For the full recorded counts and for subsamples n in {1,000, 2,000, 4,000,
8,192} drawn without replacement (multivariate hypergeometric, rng default_rng(70001 + n)), and for each eps in the grid, both rules
(kl = rule of record, v1 = frozen sufprov rule alongside): pair census (possible / certified / unresolved / irreducible / pass),
irreducible share of the violating pairs, O-min set (greedy + prune hitting set of the possible-violation family), gamma, kappa,
removal witnesses, and the exhaustive witnessed-set search (|S| in {2, 3}). K3 / K6 verdict fields are computed by
suffdisc.harness.d2_analysis.verdict exactly as design notes / protocol §12.
Usage: python -m suffdisc.harness.d2_analysis <relation.npz> <out.json>
"""
from __future__ import annotations

import json
import sys

import numpy as np

from suffdisc.core import Relation, pair_bounds, pair_status, PASS, UNRESOLVED, VIOLATION
from suffdisc.harness.k6_attempt3 import hell, osets, witnessed_search
from suffdisc.kl_lower import kl_lower
from suffdisc.pair_upper import pair_upper_many
from suffdisc.radius_v2 import kl_threshold, radius_v2

EPS = [0.05, 0.10, 0.15, 0.20]


def analyse_counts(Z, names, task, counts, ii, jj):
    N = counts.shape[0]
    n_r = counts.sum(axis=1)
    A = counts / n_r[:, None]
    hh = hell(A[ii], A[jj])
    eta1 = Relation(Z=Z, field_names=names, task=task, laws=A, shots=n_r).radii(0.05)
    eta2 = radius_v2(counts, N)
    lo1, hi1 = pair_bounds(hh, eta1[ii], eta1[jj])
    lo2, hi2 = pair_bounds(hh, eta2[ii], eta2[jj])
    assert np.all(n_r == n_r[0]), "kl rule implemented for equal shot counts per record"
    n = int(n_r[0])
    t = kl_threshold(n, counts.shape[1], N, 0.05)
    hik = pair_upper_many(A, n, N, ii, jj, prior_hi=hi2, eps=min(EPS))
    out = {"n": n, "eta_v1": float(eta1.max()), "eta_v2_median": float(np.median(eta2)), "by_eps": {}}
    for e in EPS:
        lok = lo2.copy()
        need = np.flatnonzero(pair_status(lo2, hi2, e) != VIOLATION)
        lok[need] = np.maximum(lo2[need], kl_lower(A, np.full(N, t), ii[need], jj[need], e))
        row = {}
        for rule, lo, hi in (("kl", lok, hik), ("v1", lo1, hi1)):
            st = pair_status(lo, hi, e)
            r = {"u": float((st == UNRESOLVED).mean()), "certified": int((st == VIOLATION).sum()),
                 "pass": int((st == PASS).sum()), "possible": int((st != PASS).sum()), "within_task_pairs": int(st.size)}
            r.update(osets(Z, task, ii, jj, lo, hi, e, names))
            viol = int((st == VIOLATION).sum())
            r["irreducible_certified"] = int(((st == VIOLATION) & (Z[ii] == Z[jj]).all(axis=1)).sum())
            r["irreducible_share_of_violating"] = (r["irreducible_certified"] / viol) if viol else 0.0
            r["search"] = witnessed_search(Z, task, ii, jj, lo, hi, e, names)
            row[rule] = r
        out["by_eps"][str(e)] = row
    return out


def verdict(full, grid, n_meaningful):
    """K3 on the full counts and K6 (protocol §12) on the grid, per rule."""
    res = {"K3a_fires": n_meaningful < 30}
    for rule in ("kl", "v1"):
        b_pass = []
        for e in EPS:
            r = full["by_eps"][str(e)][rule]
            nonvac = r.get("exists") and (r.get("gamma") or 0) >= 2 or r["search"]["n_witnessed_sets"] > 0
            if r["irreducible_share_of_violating"] <= 0.10 and nonvac:
                b_pass.append(e)
        c_pass = [e for e in b_pass if full["by_eps"][str(e)][rule]["search"]["n_witnessed_sets"] > 0
                  or full["by_eps"][str(e)][rule].get("all_witnessed")]
        k6a = all(grid["4000"]["by_eps"][str(e)][rule]["u"] > 0.5 for e in b_pass) if b_pass else True
        k6b = not any((g["by_eps"][str(e)][rule].get("exists") and (g["by_eps"][str(e)][rule].get("gamma") or 0) >= 2)
                      for k, g in grid.items() if k in ("1000", "2000", "4000") for e in EPS)
        res[rule] = {"K3b_pass_eps": b_pass, "K3b_fires": not b_pass, "K3c_pass_eps": c_pass, "K3c_fires": not c_pass,
                     "K6a_fires": k6a, "K6b_fires": k6b}
    return res


def main(npz, out_path):
    d = np.load(npz)
    meta = json.load(open(npz.replace(".npz", ".json"), encoding="utf-8"))
    names_all = [str(x) for x in d["names"]]
    keep = [i for i, nm in enumerate(names_all) if nm in set(meta["field_report"]["meaningful"])]
    Z = d["Z"][:, keep]
    names = [names_all[i] for i in keep]
    task = d["task"].astype(np.int32)
    counts = d["counts"].astype(np.int64)
    ii_l, jj_l = [], []
    for tk in np.unique(task):
        idx = np.flatnonzero(task == tk)
        a, b = np.triu_indices(idx.size, 1)
        ii_l.append(idx[a])
        jj_l.append(idx[b])
    ii, jj = np.concatenate(ii_l), np.concatenate(jj_l)
    res = {"relation": npz, "relation_fields_sha256": meta.get("relation_fields_sha256"), "N": int(Z.shape[0]),
           "fields_meaningful": len(keep), "field_report_counts": {k: len(v) for k, v in meta["field_report"].items()}}
    res["full"] = analyse_counts(Z, names, task, counts, ii, jj)
    print("full", json.dumps({e: {r: (v[r]["u"], v[r].get("S"), v[r].get("gamma"), v[r]["search"]["n_witnessed_sets"])
                                  for r in v} for e, v in res["full"]["by_eps"].items()}, default=float), flush=True)
    res["grid"] = {}
    for n in (1000, 2000, 4000, 8192):
        if n >= int(counts.sum(axis=1).min()):
            continue
        rng = np.random.default_rng(70001 + n)
        sub = np.stack([rng.multivariate_hypergeometric(c, n) for c in counts])
        res["grid"][str(n)] = analyse_counts(Z, names, task, sub, ii, jj)
        print(n, json.dumps({e: {r: (round(v[r]["u"], 3), v[r].get("gamma")) for r in v}
                             for e, v in res["grid"][str(n)]["by_eps"].items()}, default=float), flush=True)
    res["verdict"] = verdict(res["full"], res["grid"], len(keep)) if "4000" in res["grid"] else None
    print("verdict", json.dumps(res["verdict"]), flush=True)
    json.dump(res, open(out_path, "w", encoding="utf-8"), indent=1, default=float)


if __name__ == "__main__":
    main(*sys.argv[1:3])
