"""One job: load instance, pin to a core, run the method single-threaded, write the result row atomically."""
from __future__ import annotations

import json
import os
import sys
import time

import psutil


def main(job_path: str, core: str, out_path: str) -> None:
    job = json.load(open(job_path, encoding="utf-8"))
    proc = psutil.Process()
    try:
        proc.cpu_affinity([int(core)])
    except Exception:
        pass
    from suffdisc.simulators.planted import load
    t0 = time.perf_counter()
    rel, info = load(job["instance"])
    t_load = time.perf_counter() - t0
    knobs = dict(job.get("knobs", {}))
    c0 = time.process_time()
    tl = float(job.get("time_limit", 4 * 3600))
    if job["method"] == "B3":
        from suffdisc.baselines.b3 import run_b3
        ck = os.path.join("data/runs", job.get("batch", "batch"), "ckpt", job["id"] + ".pkl")
        os.makedirs(os.path.dirname(ck), exist_ok=True)
        res = run_b3(rel, job["eps"], time_limit=tl, ckpt_path=ck, **knobs)
    elif job["method"] == "SUFFDISC-0":
        from suffdisc.engine.suffdisc0 import run_suffdisc0
        ck = os.path.join("data/runs", job.get("batch", "batch"), "ckpt", job["id"] + ".pkl")
        os.makedirs(os.path.dirname(ck), exist_ok=True)
        res = run_suffdisc0(rel, job["eps"], time_limit=tl, ckpt_path=ck, **knobs)
    elif job["method"] in ("SUFFDISC-1", "PC-pairwise", "SUFFDISC-1-pivot-only"):
        # one implementation, validator knobs set per method in the spec (equal semantics and code for the comparison, D-013)
        from suffdisc.engine.suffdisc1 import run_suffdisc1
        ck = os.path.join("data/runs", job.get("batch", "batch"), "ckpt", job["id"] + ".pkl")
        os.makedirs(os.path.dirname(ck), exist_ok=True)
        res = run_suffdisc1(rel, job["eps"], time_limit=tl, ckpt_path=ck, **knobs)
        res["method"] = job["method"]
    else:
        raise ValueError(job["method"])
    t_cpu = time.process_time() - c0  # CPU seconds of the run (diagnoses contention from other machine load; K4 uses wall-clock)
    mi = proc.memory_info()
    res.update({"job": job, "relation_sha256": info["relation_sha256"], "t_load": t_load, "t_cpu": t_cpu,
                "peak_mem_bytes": int(getattr(mi, "peak_wset", getattr(mi, "rss", 0))), "core": int(core)})
    tmp = out_path + ".tmp"
    json.dump(res, open(tmp, "w", encoding="utf-8"), indent=1, default=str)
    from suffdisc.harness.driver import replace_retry
    replace_retry(tmp, out_path)


if __name__ == "__main__":
    main(*sys.argv[1:4])
