"""Manifest-based batch driver (protocol §11, §14).

batch spec (JSON): {"name", "jobs": [{"id", "instance", "method", "eps", "rep", "knobs": {...}, "time_limit"}], "concurrency"}
- manifest runs/<name>/manifest.json: status per job (pending / done / failed), rewritten atomically after every job;
- each job runs in its own single-threaded worker process pinned to one P-core; result -> runs/<name>/results/<id>.json (atomic);
- restart skips `done` jobs unless --fresh.
Usage: python -m suffdisc.harness.driver <spec.json> [--fresh]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import psutil

P_CORES = [0, 2, 4, 6, 8, 10, 12, 14]  # logical CPUs of the 8 P-cores (one hyper-thread each) on the i9-13900K


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=str)
    replace_retry(tmp, path)


def replace_retry(tmp, path, tries=60):
    """os.replace with retries: on Windows a reader holding the target open (indexer, editor, reviewer) makes it fail with
    PermissionError for a moment (D3 launch 1 died this way, 2026-11-07)."""
    for k in range(tries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(0.5)


def main(spec_path: str, fresh: bool = False) -> None:
    spec = json.load(open(spec_path, encoding="utf-8"))
    root = os.path.join("runs", spec["name"])
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    man_path = os.path.join(root, "manifest.json")
    man = {} if fresh or not os.path.exists(man_path) else json.load(open(man_path, encoding="utf-8"))
    for j in spec["jobs"]:
        man.setdefault(j["id"], {"status": "pending"})
    atomic_json(man_path, man)
    pending = [j for j in spec["jobs"] if man[j["id"]]["status"] != "done"]
    conc = int(spec.get("concurrency", 4))
    running: dict[int, tuple] = {}
    free_cores = list(spec.get("cores") or P_CORES[:conc])  # "cores": explicit CPUs (e.g. an E-core when another project times on CPU 0)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
               PYTHONPATH="src")
    mem_cap = float(spec.get("mem_cap_bytes", 0)) or None
    retries: dict[str, int] = {}

    def result_status(jid):
        f = os.path.join(root, "results", jid + ".json")
        if not os.path.exists(f):
            return None
        try:
            return json.load(open(f, encoding="utf-8")).get("status")
        except Exception:
            return None

    def sibling_ids(j):
        return [j["id"].rsplit("__r", 1)[0] + f"__r{r}" for r in range(3) if r != j["rep"]]

    while pending or running:
        while pending and free_cores:
            j = pending.pop(0)
            if spec.get("skip_rep_if_median_censored") and j.get("rep") == 2:
                sib = sibling_ids(j)
                st = [result_status(x) for x in sib]
                live = {q["id"] for q in pending} | {r[1]["id"] for r in running.values()}

                def waits(q):
                    return q.get("rep") == 2 and any(x in live for x in sibling_ids(q))
                if waits(j):
                    pending.append(j)  # decide once both earlier repetitions have finished; other work goes first
                    if any(not waits(q) for q in pending):
                        continue
                    break
                if all(x in ("timeout", "memcap") for x in st):
                    row = {"status": "not_run_median_already_censored", "method": j["method"], "eps": j["eps"], "job": j,
                           "t_total": None, "note": f"reps {sib} both censored ({st}); the median of 3 is censored whatever rep 2 does"}
                    atomic_json(os.path.join(root, "results", j["id"] + ".json"), row)
                    man[j["id"]] = {"status": "done", "skipped": True}
                    atomic_json(man_path, man)
                    print(j["id"], man[j["id"]], flush=True)
                    continue
            need = float(spec.get("min_free_bytes_to_launch", 0))
            if need and psutil.virtual_memory().available < need:
                pending.insert(0, j)  # machine memory is held by other processes: wait instead of crashing at start-up
                break
            core = free_cores.pop(0)
            jp = os.path.join(root, "results", j["id"] + ".job.json")
            atomic_json(jp, j)
            p = subprocess.Popen([sys.executable, "-X", "utf8", "-m", "suffdisc.harness.worker", jp, str(core),
                                  os.path.join(root, "results", j["id"] + ".json")], env=env)
            running[p.pid] = (p, j, core, time.time())
            man[j["id"]] = {"status": "running", "core": core, "started": time.time()}
            atomic_json(man_path, man)
        time.sleep(2)
        for pid in list(running):
            p, j, core, t0 = running[pid]
            hard = j.get("time_limit", 4 * 3600) + 1800
            if p.poll() is None and time.time() - t0 > hard:
                p.kill()
            if p.poll() is None and mem_cap:
                try:
                    pr = psutil.Process(pid)
                    rss = sum(x.memory_info().rss for x in [pr] + pr.children(recursive=True))
                except Exception:
                    rss = 0
                if rss > mem_cap:
                    for x in [pr] + pr.children(recursive=True):
                        try:
                            x.kill()
                        except Exception:
                            pass
                    p.wait()
                    row = {"status": "memcap", "method": j["method"], "eps": j["eps"], "job": j, "t_total": None,
                           "rss_at_kill": rss, "wall_this_launch": time.time() - t0, "mem_cap_bytes": mem_cap,
                           "note": "killed by the driver: process-tree RSS above the pre-registered per-job memory cap"}
                    atomic_json(os.path.join(root, "results", j["id"] + ".json"), row)
            if p.poll() is not None:
                res = os.path.join(root, "results", j["id"] + ".json")
                ok = (p.returncode == 0 and os.path.exists(res)) or result_status(j["id"]) == "memcap"
                man[j["id"]] = {"status": "done" if ok else "failed", "returncode": p.returncode, "wall": time.time() - t0}
                atomic_json(man_path, man)
                if not ok and retries.get(j["id"], 0) < 5:  # crash (e.g. machine-wide memory starvation): resume from checkpoint
                    retries[j["id"]] = retries.get(j["id"], 0) + 1
                    pending.append(j)
                if ok:  # the result row is saved atomically; a finished job's checkpoint is never needed again (disk budget)
                    ck = os.path.join("data/runs", j.get("batch", spec["name"]), "ckpt", j["id"] + ".pkl")
                    if os.path.exists(ck):
                        os.remove(ck)
                free_cores.append(core)
                del running[pid]
                print(j["id"], man[j["id"]], flush=True)


if __name__ == "__main__":
    main(sys.argv[1], "--fresh" in sys.argv)
