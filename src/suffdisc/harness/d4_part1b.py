"""D4 part 1b (PR-D4-1b; audit L3/D4-validator R1 item 4): the v3 code paths that part 1 (M <= 15, N <= 300) cannot reach —
multi-word masks (M 65-130), several distance blocks (block 256 with N 1,000-2,500), IndexedCover growth (initial capacity 64),
SUFFDISC-0 checkpoint resume — checked against C6 (B0 is infeasible at this M).

Usage: python -m suffdisc.harness.d4_part1b <out_jsonl> <n_procs>
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from multiprocessing import Pool

import numpy as np

from suffdisc.baselines.b3 import run_b3
from suffdisc.core import Relation
from suffdisc.engine.suffdisc0 import TestAbort, run_suffdisc0
from suffdisc.reporting import boundary_report
from suffdisc.simulators.planted import generate
from suffdisc.validator.c6 import C6

SEEDS = list(range(63001, 63025))
EPS = [0.10, 0.20]


def instance(seed: int) -> Relation:
    rng = np.random.default_rng(seed)
    fam = ["far", "near", "mixed"][seed % 3]
    M = int(rng.integers(65, 131))
    N = int(rng.integers(1000, 2501))
    rel, _ = generate(fam, M, N, seed)
    rel.meta = {"family": fam, "M": M, "N": N, "seed": seed, "shots": None}
    if seed % 2 == 1:
        r2 = np.random.default_rng(seed + 10**6)
        counts = np.stack([r2.multinomial(1000, p / p.sum()) for p in rel.laws])
        rel = Relation(Z=rel.Z, field_names=rel.field_names, task=rel.task, laws=counts / 1000,
                       shots=np.full(rel.N, 1000), meta={**rel.meta, "shots": 1000})
    return rel


def check(c6, r, eps, rel=None):
    listed = sorted(tuple(x["set"]) for x in boundary_report(rel, eps, r)["boundary_sets"]) if rel is not None else []
    M = c6.Z.shape[1]
    if not r["certified_sufficient_exists"]:
        st = c6.status(list(range(M)), eps)
        return {"exists": False, "c6_full_set_status": st,
                "boundary_agree": listed == ([tuple(range(M))] if st == "boundary" else [])}
    chk = c6.check_output(r["S"], eps)
    lab = all((r["labels"][k] == "certified_needed") == (chk["labels"][k] == "certified_needed") for k in r["S"])
    want = sorted(tuple(sorted(f for f in r["S"] if f != k)) for k, v in chk["labels"].items() if v == "boundary")
    return {"exists": True, "S": r["S"], "c6_status": chk["status"], "no_removable": chk["no_removable_field"],
            "labels_agree": lab, "boundary_agree": listed == want,
            "ok": bool(chk["status"] == "sufficient" and chk["no_removable_field"] and lab and listed == want)}


def one(seed: int) -> dict:
    rel = instance(seed)
    c6 = C6(rel.Z, rel.task, rel.laws, rel.shots)
    out = {"seed": seed, **rel.meta, "cases": []}
    for eps in EPS:
        case = {"eps": eps}
        b_def = run_b3(rel, eps)
        b_small = run_b3(rel, eps, block=256, cover_cap=64)
        s_def = run_suffdisc0(rel, eps, seed=seed, block=256)
        with tempfile.TemporaryDirectory(dir="data/runs") as d:
            ck = os.path.join(d, "s0.pkl")
            aborted = False
            try:
                run_suffdisc0(rel, eps, seed=seed, block=256, ckpt_path=ck, ckpt_every=0.0, _test_abort_after_first_ckpt=True)
            except TestAbort:
                aborted = True
            s_res = run_suffdisc0(rel, eps, seed=seed, block=256, ckpt_path=ck, ckpt_every=0.0)
        case["B3"] = check(c6, b_def, eps, rel)
        case["B3_block256_cap64"] = check(c6, b_small, eps, rel)
        case["B3_variants_same_S"] = b_def.get("S") == b_small.get("S") and b_def["cover_plus"] == b_small["cover_plus"]
        case["SUFFDISC-0"] = check(c6, s_def, eps, rel)
        case["S0_resume"] = {"aborted_after_ckpt": aborted, "resumed": s_res.get("resumed", 0),
                             "same_S_and_labels": s_res.get("S") == s_def.get("S") and s_res.get("labels") == s_def.get("labels")}
        case["exists_agree"] = b_def["certified_sufficient_exists"] == s_def["certified_sufficient_exists"]
        case["cover_plus_B3"] = b_def["cover_plus"]
        case["S0_iterations"] = s_def["iterations"]
        case["boundary_pairs"] = {"B3": b_def.get("n_boundary_pair_evals"), "S0": s_def.get("n_boundary")}
        out["cases"].append(case)
    return out


def main(out_path: str, procs: int) -> None:
    with Pool(procs) as pool, open(out_path, "w", encoding="utf-8") as fh:
        for res in pool.imap_unordered(one, SEEDS):
            fh.write(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
            fh.flush()


def summary(path: str) -> None:
    rows = [json.loads(x) for x in open(path, encoding="utf-8")]
    n = bad = vac = rex = 0
    notes = []
    for r in rows:
        for c in r["cases"]:
            for k in ("B3", "B3_block256_cap64", "SUFFDISC-0"):
                n += 1
                v = c[k]
                good = v["ok"] if v["exists"] else (v["c6_full_set_status"] != "sufficient" and v.get("boundary_agree", True))
                if not good:
                    bad += 1
                    notes.append((r["seed"], c["eps"], k, v))
            for flag in ("B3_variants_same_S", "exists_agree"):
                if not c[flag]:
                    bad += 1
                    notes.append((r["seed"], c["eps"], flag))
            res = c["S0_resume"]
            if not res["aborted_after_ckpt"]:
                vac += 1  # run ended in its first iteration: no checkpoint is ever taken, resume not exercised
            elif res["resumed"] >= 1 and res["same_S_and_labels"]:
                rex += 1
            else:
                bad += 1
                notes.append((r["seed"], c["eps"], "S0_resume", res))
    Ms = [r["M"] for r in rows]
    covers = [c["cover_plus_B3"] for r in rows for c in r["cases"]]
    print(f"D4 part 1b: instances {len(rows)} (M {min(Ms)}-{max(Ms)}), outputs checked {n}, failures {bad}; "
          f"B3 cover sizes {min(covers)}-{max(covers)} (initial capacity 64 => growth exercised when > 64); SUFFDISC-0 resume "
          f"exercised and identical {rex}, not exercisable (finished in its first iteration, no checkpoint) {vac}")
    for x in notes:
        print("  ", x)


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary(sys.argv[2])
    else:
        main(sys.argv[1], int(sys.argv[2]))
