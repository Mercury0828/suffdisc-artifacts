"""Tabulate a tuning batch and apply the PR-TUNE-1 selection rule: per method, the setting with the smallest geometric-mean wall
time over the cases (instance, eps) that EVERY setting of that method completes; ties -> smaller peak RSS. Jobs with wall / CPU >
1.5 are flagged as contended. Usage: python -m suffdisc.harness.tune_summary <results_dir>"""
import collections
import glob
import json
import math
import os
import sys


def main(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        if f.endswith(".job.json"):
            continue
        r = json.load(open(f, encoding="utf-8"))
        job = json.load(open(f.replace(".json", ".job.json"), encoding="utf-8"))
        res = r.get("result", r)
        rows.append({"id": job["id"], "case": (os.path.basename(job["instance"]), job["eps"]), "method": job["method"],
                     "knobs": json.dumps(job["knobs"], sort_keys=True), "status": res.get("status"),
                     "wall": res.get("t_total"), "cpu": res.get("t_cpu"), "rss": res.get("peak_mem_bytes"),
                     "S": res.get("S"), "size": len(res["S"]) if res.get("S") is not None else None})
    by = collections.defaultdict(list)
    for x in rows:
        by[x["method"]].append(x)
    for m, xs in by.items():
        print(f"== {m}")
        settings = sorted({x["knobs"] for x in xs})
        cases = sorted({x["case"] for x in xs})
        done = {(x["knobs"], x["case"]): x for x in xs if x["status"] == "complete"}
        common = [c for c in cases if all((s, c) in done for s in settings)]
        print(f"cases {len(cases)}, completed by every setting {len(common)}: {common}")
        for c in cases:
            line = "  ".join(f"{done[(s, c)]['wall']:8.1f}" if (s, c) in done else f"{'cens':>8}" for s in settings)
            print(f"{c[0][:34]:34s} e{c[1]:<4} {line}")
        best = None
        for s in settings:
            if not common:
                continue
            gm = math.exp(sum(math.log(max(done[(s, c)]['wall'], 1e-6)) for c in common) / len(common))
            rss = max((done[(s, c)]['rss'] or 0) for c in common)
            ncens = sum(1 for c in cases if (s, c) not in done)
            print(f"  setting {s}: geo-mean wall {gm:.2f} s over {len(common)} common cases; max RSS {rss / 1e9:.2f} GB; censored {ncens}")
            if best is None or (gm, rss) < best[0]:
                best = ((gm, rss), s)
        print(f"  SELECTED {best[1] if best else None}")
        flag = [x["id"] for x in xs if x["cpu"] and x["wall"] and x["wall"] / x["cpu"] > 1.5]
        print(f"  contended (wall/CPU > 1.5): {len(flag)} {flag}")


if __name__ == "__main__":
    main(sys.argv[1])
