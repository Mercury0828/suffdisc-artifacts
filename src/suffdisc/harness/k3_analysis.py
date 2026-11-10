"""K3 / K6 analysis of a record relation (protocol §7, §12): meaningful fields, pair structure per eps, O-min by B3 (checked by
C6), non-vacuity gamma / kappa, removal witnesses; optional finite-shot resampling for K6.

Usage: python -m suffdisc.harness.k3_analysis <relation.npz> <out.json> [--shots-grid] [--subsample-from-counts]
The npz holds Z, task, laws (exact or empirical), names, and optionally counts / shots.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from suffdisc.baselines.b3 import run_b3
from suffdisc.core import Relation, pair_bounds, PASS, UNRESOLVED, VIOLATION, pair_status
from suffdisc.engine.kernels import diff_masks, n_words, popcounts
from suffdisc.validator.c6 import C6

EPS = [0.05, 0.10, 0.15, 0.20]


def meaningful(Z: np.ndarray, report: dict | None, names: list[str]) -> list[int]:
    if report is not None:
        keep = set(report["meaningful"])
        return [i for i, n in enumerate(names) if n in keep]
    return [i for i in range(Z.shape[1]) if len(np.unique(Z[:, i])) > 1]


def gamma_kappa(Z, task, S):
    keys = np.stack([task] + [Z[:, f] for f in S], axis=1)
    _, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    gamma = Z.shape[0] / cnt.size
    same = (cnt * (cnt - 1) // 2).sum()
    _, tc = np.unique(task, return_counts=True)
    within = (tc * (tc - 1) // 2).sum()
    return float(gamma), float(same / within) if within else 0.0


def pair_census(rel: Relation, eps_list, alpha=0.05):
    eta = rel.radii(alpha)
    W = n_words(rel.M)
    R = np.sqrt(rel.laws)
    tot = {e: {"possible": 0, "certified": 0, "unresolved": 0, "irreducible": 0, "within_task_pairs": 0} for e in eps_list}
    for t in np.unique(rel.task):
        idx = np.flatnonzero(rel.task == t)
        ii, jj = np.triu_indices(idx.size, 1)
        a, b = idx[ii], idx[jj]
        h = np.sqrt(np.maximum(0.0, 1.0 - np.einsum("ij,ij->i", R[a], R[b])))
        lo, hi = pair_bounds(h, eta[a], eta[b])
        empty = (rel.Z[a] == rel.Z[b]).all(axis=1)
        for e in eps_list:
            st = pair_status(lo, hi, e)
            d = tot[e]
            d["within_task_pairs"] += int(a.size)
            d["possible"] += int((st != PASS).sum())
            d["certified"] += int((st == VIOLATION).sum())
            d["unresolved"] += int((st == UNRESOLVED).sum())
            d["irreducible"] += int(((st == VIOLATION) & empty).sum())
    return tot


def analyse(rel: Relation, names_kept: list[str], alpha=0.05, eps_list=EPS, c6_check=True, c6_kwargs=None):
    out = {"N": rel.N, "M_meaningful": rel.M, "census": pair_census(rel, eps_list, alpha), "by_eps": {}}
    c6 = C6(rel.Z, rel.task, rel.laws, rel.shots, alpha, **(c6_kwargs or {})) if c6_check else None
    for e in eps_list:
        r = run_b3(rel, e, alpha, time_limit=3600)
        res = {"status": r["status"], "exists": r.get("certified_sufficient_exists"), "t": r["t_total"],
               "cover_plus": r["cover_plus"], "cover_minus": r["cover_minus"]}
        if r["status"] == "complete" and r["certified_sufficient_exists"]:
            S = r["S"]
            g, k = gamma_kappa(rel.Z, rel.task, S)
            res.update({"S": [names_kept[f] for f in S], "size": len(S), "gamma": g, "kappa": k,
                        "labels": {names_kept[f]: v for f, v in r["labels"].items()},
                        "all_witnessed": all(v == "certified_needed" for v in r["labels"].values())})
            if c6 is not None:
                chk = c6.check_output(S, e)
                res["c6"] = {"status": chk["status"], "certified_inclusion_minimal": chk["certified_inclusion_minimal"],
                             "labels_agree": all((r["labels"][f] == "certified_needed") == (chk["labels"][f] == "certified_needed")
                                                 for f in S)}
        out["by_eps"][str(e)] = res
    return out


def main(npz, out_path, *flags):
    d = np.load(npz, allow_pickle=False)
    meta = json.load(open(npz.replace(".npz", "") + ".npz.json")) if False else None
    names = [str(x) for x in d["names"]]
    report = None
    try:
        report = json.load(open(npz.replace(".npz", "") + ".json"))["field_report"]
    except Exception:
        pass
    keep = meaningful(d["Z"], report, names)
    Zk = d["Z"][:, keep]
    kept = [names[i] for i in keep]
    task = d["task"].astype(np.int32)
    result = {"relation": npz, "fields_total": len(names), "fields_meaningful": len(keep), "field_report_counts":
              {k: len(v) for k, v in (report or {}).items()}}
    base = Relation(Z=Zk, field_names=kept, task=task, laws=d["laws"],
                    shots=d["shots"] if "shots" in d.files else None)
    result["exact_or_observed"] = analyse(base, kept)
    if "--shots-grid" in flags:
        result["shots_grid"] = {}
        grid = (1000, 2000, 4000, 8192) if "--subsample-from-counts" in flags else (1000, 2000, 4000)  # 8,192: v1.1 informative
        for n in grid:
            if "--subsample-from-counts" in flags:
                rng = np.random.default_rng(70001 + n)
                counts = np.stack([rng.multivariate_hypergeometric(c.astype(np.int64), n) for c in d["counts"]])
            else:
                rng = np.random.default_rng(80001 + n)
                counts = np.stack([rng.multinomial(n, p / p.sum()) for p in d["laws"]])
            rel_n = Relation(Z=Zk, field_names=kept, task=task, laws=counts / n, shots=np.full(base.N, n))
            result["shots_grid"][str(n)] = analyse(rel_n, kept, c6_check=False)
    json.dump(result, open(out_path, "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in result.items() if k != "shots_grid"}, default=float)[:3000])


if __name__ == "__main__":
    main(*sys.argv[1:])
