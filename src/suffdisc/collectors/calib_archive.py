"""Free calibration-archive collector for ibm_cleveland (no QPU job, no QPU seconds).

For each requested UTC timestamp, calls backend.properties(datetime=ts) (returns the snapshot closest to, but older than, ts)
and stores the full snapshot as JSON under <out_dir>/props_<last_update_date>.json. A manifest (<out_dir>/manifest.json) is
rewritten atomically after every snapshot, so an interrupted run resumes without refetching.

Usage: python -I calib_archive.py <out_dir> <start YYYY-MM-DD> <end YYYY-MM-DD> <hours between requests>
"""
import datetime as dt
import json
import os
import sys
import time

from qiskit_ibm_runtime import QiskitRuntimeService


def atomic_write(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, default=str)
    os.replace(tmp, path)


def main(out_dir: str, start: str, end: str, step_h: float) -> None:
    os.makedirs(out_dir, exist_ok=True)
    man_path = os.path.join(out_dir, "manifest.json")
    manifest = json.load(open(man_path, encoding="utf-8")) if os.path.exists(man_path) else {"requests": {}}
    service = QiskitRuntimeService()
    backend = service.backend("ibm_cleveland")
    t = dt.datetime.fromisoformat(start).replace(tzinfo=dt.timezone.utc)
    t_end = dt.datetime.fromisoformat(end).replace(tzinfo=dt.timezone.utc)
    while t <= t_end:
        key = t.isoformat()
        if manifest["requests"].get(key, {}).get("status") == "done":
            t += dt.timedelta(hours=step_h)
            continue
        t0 = time.time()
        try:
            props = backend.properties(datetime=t)
            snap = props.to_dict()
            lud = str(props.last_update_date)
            fname = "props_" + lud.replace(":", "").replace(" ", "T").replace("+", "p").replace("-", "") + ".json"
            fpath = os.path.join(out_dir, fname)
            if not os.path.exists(fpath):
                atomic_write(fpath, snap)
            manifest["requests"][key] = {"status": "done", "last_update_date": lud, "file": fname,
                                         "seconds": round(time.time() - t0, 1)}
        except Exception as exc:  # recorded, retried on the next run
            manifest["requests"][key] = {"status": "error", "error": repr(exc)}
        atomic_write(man_path, manifest)
        print(key, manifest["requests"][key], flush=True)
        t += dt.timedelta(hours=step_h)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]))
