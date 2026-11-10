"""PR-D3b-r0 summary: per case times, distance evaluations, validation time, output equality. Usage: python -m suffdisc.harness.d3b_summary <batch_dir>"""
import glob
import json
import os
import sys
from collections import defaultdict

M = ("PC-pairwise", "SUFFDISC-1-pivot-only", "SUFFDISC-1")


def main(batch):
    R = defaultdict(dict)
    for f in glob.glob(os.path.join(batch, "results", "*.json")):
        if f.endswith(".job.json"):
            continue
        r = json.load(open(f, encoding="utf-8"))
        inst = os.path.basename(r["job"]["instance"])
        R[(inst, r["job"]["eps"])][r["job"]["method"]] = r
    rows = []
    for k in sorted(R):
        row = {"instance": k[0], "eps": k[1]}
        outs = set()
        for m in M:
            r = R[k].get(m)
            if r is None:
                continue
            row[m] = {"status": r["status"], "t": r.get("t_total") or r.get("elapsed_lower_bound") or r.get("wall_this_launch"),
                      "t_validate": r.get("t_validate"), "n_dist": r.get("n_dist"), "wall_over_cpu": None}
            if r.get("t_cpu"):
                row[m]["wall_over_cpu"] = round((r.get("t_total") or 0) / r["t_cpu"], 2)
            if r["status"] == "complete":
                outs.add(json.dumps([sorted(r["S"]), sorted(r["labels"].items())]))
        row["outputs_identical"] = len(outs) <= 1
        a, b = row.get("PC-pairwise"), row.get("SUFFDISC-1")
        if a and b and b["t"]:
            row["speedup"] = a["t"] / b["t"]
            row["speedup_is_lower_bound"] = a["status"] != "complete"
            if a["n_dist"] and b["n_dist"]:
                row["dist_ratio"] = a["n_dist"] / b["n_dist"]
        rows.append(row)
    json.dump(rows, open(os.path.join(batch, "summary.json"), "w"), indent=1)
    for r in rows:
        line = f"{r['instance']:28s} {r['eps']:.2f}"
        for m in M:
            if m in r:
                x = r[m]
                line += f" | {m[:12]:12s} {x['status'][:4]} {x['t']:9.1f}s nd={x['n_dist']}"
        line += f" | x{r.get('speedup', float('nan')):.1f} dist x{r.get('dist_ratio', float('nan')):.0f} same={r['outputs_identical']}"
        print(line)


if __name__ == "__main__":
    main(sys.argv[1])
