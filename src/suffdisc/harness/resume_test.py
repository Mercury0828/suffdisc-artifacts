"""Kill -> restart -> resume test for B3 and SUFFDISC-0 checkpointing (design notes (6), protocol §14).

python -m suffdisc.harness.resume_test run <method> <ckpt> <out.json>   (worker: runs the method with 5 s checkpoints)
python -m suffdisc.harness.resume_test test <method>                      (driver: uninterrupted vs killed-and-resumed)
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

from suffdisc.core import Relation
from suffdisc.simulators.planted import load

INST, NREC, EPS = "data/sim/tuning-03-far-M800-N10000", 1600, 0.10
INST_S0, EPS_S0 = "data/sim/tuning-05-mixed-M800-N10000", 0.20


def small():
    rel, _ = load(INST)
    keep = np.arange(NREC)
    return Relation(Z=rel.Z[keep], field_names=rel.field_names, task=rel.task[keep], laws=rel.laws[keep])


def worker(method, ckpt, out):
    if method == "B3":
        from suffdisc.baselines.b3 import run_b3
        r = run_b3(small(), EPS, block=256, ckpt_path=ckpt, ckpt_every=5.0)
    else:
        from suffdisc.engine.suffdisc0 import run_suffdisc0
        rel, _ = load(INST_S0)  # SUFFDISC-0 v3 finishes in seconds: checkpoint at every iteration boundary, and the killed run
        r = run_suffdisc0(rel, EPS_S0, ckpt_path=ckpt, ckpt_every=0.0)  # exits hard right after its first checkpoint (env hook)
    json.dump(r, open(out, "w"), default=str)


def test(method):
    d = "data/runs/resume_test_" + method
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        os.remove(os.path.join(d, f))
    env = dict(os.environ, PYTHONPATH="src", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    t0 = time.time()
    subprocess.run([sys.executable, "-m", "suffdisc.harness.resume_test", "run", method, f"{d}/a.pkl", f"{d}/a.json"], env=env, check=True)
    t_full = time.time() - t0
    if method == "B3":
        p = subprocess.Popen([sys.executable, "-m", "suffdisc.harness.resume_test", "run", method, f"{d}/b.pkl", f"{d}/b.json"], env=env)
        time.sleep(max(12.0, 0.5 * t_full))
        p.kill()
        p.wait()
    else:
        p = subprocess.run([sys.executable, "-m", "suffdisc.harness.resume_test", "run", method, f"{d}/b.pkl", f"{d}/b.json"],
                           env=dict(env, SUFFDISC_TEST_HARD_EXIT_AFTER_CKPT="1"))
        print("killed run exit code", p.returncode, "result written:", os.path.exists(f"{d}/b.json"))
    killed_ckpt_exists = os.path.exists(f"{d}/b.pkl")
    subprocess.run([sys.executable, "-m", "suffdisc.harness.resume_test", "run", method, f"{d}/b.pkl", f"{d}/b.json"], env=env, check=True)
    a, b = json.load(open(f"{d}/a.json")), json.load(open(f"{d}/b.json"))
    if method != "B3":
        a["n_possible_viol_pairs"] = a["n_certified_viol_pairs"] = b["n_possible_viol_pairs"] = b["n_certified_viol_pairs"] = None
    rep = {"method": method, "version": b.get("version"), "status": (a["status"], b["status"]),"t_uninterrupted": t_full, "checkpoint_existed_after_kill": killed_ckpt_exists, "resumed_count": b["resumed"],
           "same_S": a["S"] == b["S"], "same_cover_sizes": (a["cover_plus"], a["cover_minus"]) == (b["cover_plus"], b["cover_minus"]),
           "same_pair_counts": (a["n_possible_viol_pairs"], a["n_certified_viol_pairs"]) == (b["n_possible_viol_pairs"], b["n_certified_viol_pairs"]),
           "a": {k: a.get(k) for k in ("S", "cover_plus", "n_possible_viol_pairs", "t_total", "iterations", "labels")},
           "b": {k: b.get(k) for k in ("S", "cover_plus", "n_possible_viol_pairs", "t_total", "resumed", "iterations", "labels")},
           "same_labels": a.get("labels") == b.get("labels")}
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(f"runs/resume_test_{method}.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        worker(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        test(sys.argv[2])
