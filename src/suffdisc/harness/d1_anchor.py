"""D1a — reproduce sufprov §V / Table II with C1 (relation) + B0 (brute force) + C6 (independent validator). PR-D1a.

Usage: python d1_anchor.py <out_json> [draw_variant]
"""
import json
import sys
import time

import numpy as np

from suffdisc.baselines.b0 import all_subsets, mask_to_fields, minimal_sufficient, min_cost_sets
from suffdisc.core import Relation
from suffdisc.simulators.sufprov_sim import build_task_laws
from suffdisc.validator.c6 import C6

NAMES = ["L", "R", "G", "M", "A", "S"]
TASKS = [("HEA", 1), ("HEA", 3), ("QAOA", 1), ("QAOA", 2)]
EPS_SWEEP = [0.01, 0.05, 0.10, 0.15, 0.20]
GUARD = 1e-10


def main(out_path, draw="sequential"):
    t0 = time.time()
    report = {"draw_variant": draw, "tasks": {}}
    for kind, depth in TASKS:
        Z, P, angles = build_task_laws(kind, depth, seed=42, draw=draw)
        rel = Relation(Z=Z, field_names=NAMES, task=np.zeros(len(Z)), laws=P)
        table = all_subsets(rel)
        c6 = C6(Z, np.zeros(len(Z)), P)
        disagreements = 0
        for s, (lo, hi) in table.items():
            L, U, _, _ = c6.risk_bounds([k for k in range(6) if (s >> k) & 1])
            if abs(U - hi) > 1e-12 or abs(L - lo) > 1e-12:
                disagreements += 1
        name = f"{kind}-{depth}"
        full = (1 << 6) - 1
        rec = {"relation_sha256": rel.sha256(), "angles": angles.tolist(), "b0_c6_disagreements_64_subsets": disagreements,
               "singleton_s": {NAMES[k]: table[full & ~(1 << k)][1] for k in range(6)},
               "risk": {"".join(mask_to_fields(s, NAMES)) or "{}": table[s][1] for s in table}, "by_eps": {}}
        for eps in EPS_SWEEP:
            mins = minimal_sufficient(table, 6, eps, GUARD)
            best, opt = min_cost_sets(table, [1.0] * 6, eps, GUARD)
            opt_sorted = sorted(opt, key=lambda s: (table[s][1], s))
            rec["by_eps"][str(eps)] = {
                "min_cost": best,
                "min_cost_sets": [{"set": mask_to_fields(s, NAMES), "risk": table[s][1]} for s in opt_sorted],
                "n_inclusion_minimal": len(mins),
                "c6_check_selected": c6.check_output([k for k in range(6) if (opt_sorted[0] >> k) & 1], eps)["certified_inclusion_minimal"],
            }
        report["tasks"][name] = rec
        print(name, "disagreements", disagreements, "| eps=0.05:", rec["by_eps"]["0.05"]["min_cost_sets"], flush=True)
    report["seconds"] = round(time.time() - t0, 1)
    json.dump(report, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
