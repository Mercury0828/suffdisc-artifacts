"""K4 statistic of record (protocol §10; §19 / A-008: --b3-rep0 = B3 case time from rep 0 only).
Usage: python -m suffdisc.harness.k4_stats <batch_dir> <out.json> [--b3-rep0]"""
import glob
import json
import math
import sys
from collections import defaultdict

import numpy as np

LIMIT = 4 * 3600.0


def rep_time(r):
    """(value, censored). complete -> (t_total, False); timeout -> (LIMIT, True); memcap -> (observed elapsed, True): a memory-cap
    kill proves only that the run lasted at least as long as observed, NOT 4 h. Censored values are
    lower bounds."""
    if r["status"] == "complete":
        return float(r["t_total"]), False
    if r["status"] == "memcap":
        return float(r.get("elapsed_lower_bound") or r.get("wall_this_launch") or 0.0), True
    return LIMIT, True


def case_time(reps):
    """Median of 3 (value, censored, status) triples. The median of the lower bounds is a lower bound of the true median; it is
    exact only if no censored value ranks at or below the median position, otherwise the case is censored (a lower bound)."""
    t = sorted(reps, key=lambda x: (x[0], x[1]))
    mid = len(t) // 2
    return t[mid][0], any(x[1] for x in t[:mid + 1]), t[mid][2]


def main(batch_dir, out_path, b3_rep0=False):
    rows = [json.load(open(f)) for f in glob.glob(f"{batch_dir}/results/*.json") if not f.endswith(".job.json")]
    times = defaultdict(list)
    for r in rows:
        j = r["job"]
        inst = j["instance"].split("/")[-1]
        if r["status"] == "not_run_median_already_censored":
            continue  # protocol amendment (D3): reps 0 and 1 censored => the median is censored whatever rep 2 does
        if b3_rep0 and r["method"] == "B3" and j.get("rep") != 0:
            continue  # A-008: only rep 0 of B3 is of record
        times[(inst, j["eps"], r["method"])].append(rep_time(r) + (r["status"],))
    cases = sorted({(i, e) for (i, e, _) in times})
    table, logr = [], []
    for inst, eps in cases:
        tb = times.get((inst, eps, "B3"), [])
        ts = times.get((inst, eps, "SUFFDISC-0"), [])
        cens2 = lambda reps: len(reps) == 2 and all(x[1] for x in reps)  # noqa: E731  (rep 2 skipped by the rule)
        b3_ok = len(tb) == 1 if b3_rep0 else (len(tb) >= 3 or cens2(tb))
        if not b3_ok or (len(ts) < 3 and not cens2(ts)):
            table.append({"instance": inst, "eps": eps, "incomplete_reps": True})
            continue
        (b, bc, _), (s, sc, _) = case_time(tb), case_time(ts)
        row = {"instance": inst, "eps": eps, "t_B3": b, "B3_censored": bc, "t_S0": s, "S0_censored": sc, "B3_reps": tb,
               "S0_reps": ts, "B3_memcap_reps": sum(x[2] == "memcap" for x in tb)}
        if b >= 600 or bc:
            if bc and sc:
                row["kind"] = "both_censored(K4c)"
            elif bc:
                row["r"] = b / s; row["kind"] = "B3_censored_lower_bound"  # b is B3's observed lower bound
            elif sc:
                row["r"] = b / LIMIT; row["kind"] = "S0_censored_upper_bound"
            else:
                row["r"] = b / s; row["kind"] = "both_finished"
            if "r" in row:
                logr.append((inst, math.log(row["r"])))
        else:
            row["kind"] = "B3_under_10min"
        table.append(row)
    complete = [r for r in table if not r.get("incomplete_reps")]
    k4a = bool(complete) and len(complete) == len(table) and all(r["t_B3"] <= 600 and not r["B3_censored"] for r in complete)
    k4c = any(r.get("kind") == "both_censored(K4c)" for r in complete)
    res = {"cases": table, "K4a_fires": k4a, "K4c_fires": k4c, "n_ratio_cases": len(logr),
           "n_cases_with_B3_memcap": sum(1 for r in complete if r.get("B3_memcap_reps"))}
    if logr:
        insts = sorted({i for i, _ in logr})
        by = {i: [v for j, v in logr if j == i] for i in insts}
        rng = np.random.default_rng(7)
        boots = []
        for _ in range(10000):
            pick = rng.choice(len(insts), size=len(insts), replace=True)
            vals = [v for k in pick for v in by[insts[k]]]
            boots.append(np.mean(vals))
        mean = float(np.mean([v for _, v in logr]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        res.update({"mean_log_r": mean, "geo_mean_r": math.exp(mean), "ci95_log_r": [float(lo), float(hi)],
                    "ci95_r": [math.exp(lo), math.exp(hi)], "K4b_fires": bool(lo < math.log(5))})
    json.dump(res, open(out_path, "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "cases"}, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--b3-rep0" in sys.argv)
