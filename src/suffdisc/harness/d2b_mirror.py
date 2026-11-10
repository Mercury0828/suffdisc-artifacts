"""D2b pilot mirror and D5-on-D2b under radius rule v1 (frozen, sufprov Eq. 12) or v2 (K6 repair candidate, radius_v2.py).

Same selection as the recorded mirror runs (pre-registered mirror runs): records of the 3 most recent archive epochs (96) or all 1,440;
meaningful fields from the D2b field report, then re-applied on the selected records; counts ~ Multinomial(n, law) with
rng default_rng(80001 + n). Usage:
python -m suffdisc.harness.d2b_mirror <n_shots> <v1|v2> <last3|all> <out.json>
"""
from __future__ import annotations

import json
import sys

import numpy as np

from suffdisc.core import Relation
from suffdisc.harness.k3_analysis import analyse, meaningful
from suffdisc.radius_v2 import radius_v2


class RelationR(Relation):
    """Relation whose per-record radii are given (radius rule v2)."""

    def radii(self, alpha):  # noqa: D401
        return self.meta["eta"]


def main(n, rule, rows_sel, out_path):
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
    task = d["task"][rows].astype(np.int32)
    rng = np.random.default_rng(80001 + n)
    counts = np.stack([rng.multinomial(n, p / p.sum()) for p in laws])
    N = len(laws)
    if rule == "v1":
        rel = Relation(Z=Z, field_names=kept, task=task, laws=counts / n, shots=np.full(N, n))
        c6kw = {}
    else:
        eta = radius_v2(counts, N)
        rel = RelationR(Z=Z, field_names=kept, task=task, laws=counts / n, shots=np.full(N, n), meta={"eta": eta})
        c6kw = {"counts": counts, "radius": "v2"}
    eta = rel.radii(0.05)
    a = analyse(rel, kept, c6_kwargs=c6kw)
    res = {"n": n, "rule": rule, "rows": rows_sel, "N": N, "M": rel.M, "eta_min": float(eta.min()), "eta_max": float(eta.max()),
           "eta_median": float(np.median(eta)), "zero_count_records": int((counts == 0).any(axis=1).sum()), "by_eps": {}}
    print(f"n {n} rule {rule} rows {rows_sel} N {N} M {rel.M} eta min/med/max {eta.min():.4f}/{np.median(eta):.4f}/"
          f"{eta.max():.4f} records with a zero count {res['zero_count_records']}")
    for e in a["census"]:
        c = a["census"][e]
        b = a["by_eps"][str(e)]
        row = {"u": c["unresolved"] / c["within_task_pairs"], "certified": c["certified"], "possible": c["possible"],
               "irreducible": c["irreducible"], "S": b.get("S"), "size": b.get("size"), "gamma": b.get("gamma"),
               "kappa": b.get("kappa"), "labels": b.get("labels"), "c6": b.get("c6"), "status": b.get("status")}
        res["by_eps"][str(e)] = row
        print(f"eps {e} u {row['u']:.3f} cert {row['certified']} poss {row['possible']} | S {row['S']} |S| {row['size']} "
              f"gamma {row['gamma']} labels {row['labels']} c6 {row['c6']}")
    json.dump(res, open(out_path, "w"), indent=1, default=float)


if __name__ == "__main__":
    main(*sys.argv[1:5])
