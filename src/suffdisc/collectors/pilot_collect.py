"""D2 hardware collector for the frozen pilot (protocol §6). SUBMITS QPU JOBS — run only after the authors's explicit approval.

Resumable: data/records\\pilot\\manifest.json holds one entry per planned (epoch_label, dd) job with its job id and status;
an entry with a job id is never re-submitted (design notes (5)).
  python -I pilot_collect.py submit <pilot_dir> <epoch_label> --approved-by-authors
  python -I pilot_collect.py fetch  <pilot_dir>
"""
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # own source dir (calib_epoch.py)
from calib_epoch import snapshot_at, value_epoch, values  # noqa: E402

import qiskit_ibm_runtime  # noqa: E402
from qiskit import qpy
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

SHOTS = 32768  # protocol v1.1 (A-004 Option B, authors decision D-011, 2026-11-07)


def atomic(path, obj):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w", encoding="utf-8"), indent=1, default=str)
    os.replace(tmp, path)


def submit(pilot_dir, epoch_label):
    man_path = os.path.join(pilot_dir, "manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else {"jobs": {}, "usage_log": []}
    service = QiskitRuntimeService()
    backend = service.backend("ibm_cleveland")
    with open(os.path.join(pilot_dir, "pilot_isa.qpy"), "rb") as fh:
        isas = qpy.load(fh)
    man["usage_log"].append({"when": dt.datetime.now(dt.timezone.utc).isoformat(), "event": f"before {epoch_label}",
                             "usage": service.usage()})
    atomic(man_path, man)
    for dd in (False, True):
        key = f"{epoch_label}|dd={dd}"
        if man["jobs"].get(key, {}).get("job_id"):
            print("skip (already submitted)", key)
            continue
        sampler = SamplerV2(mode=backend)
        sampler.options.default_shots = SHOTS
        sampler.options.dynamical_decoupling.enable = dd
        if dd:
            sampler.options.dynamical_decoupling.sequence_type = "XY4"
        sampler.options.twirling.enable_gates = False
        sampler.options.twirling.enable_measure = False
        job = sampler.run([(c,) for c in isas], shots=SHOTS)
        man["jobs"][key] = {"job_id": job.job_id(), "submitted": dt.datetime.now(dt.timezone.utc).isoformat(),
                            "dd": dd, "epoch_label": epoch_label, "status": "submitted",
                            "pubs": [c.name for c in isas], "shots": SHOTS}
        atomic(man_path, man)
        print("submitted", key, job.job_id())


def fetch(pilot_dir):
    man_path = os.path.join(pilot_dir, "manifest.json")
    man = json.load(open(man_path))
    service = QiskitRuntimeService()
    backend = service.backend("ibm_cleveland")
    for key, ent in man["jobs"].items():
        if ent.get("status") == "done":
            continue
        job = service.job(ent["job_id"])
        st = str(job.status())
        if st not in ("DONE", "JobStatus.DONE"):
            ent["status"] = st
            continue
        res = job.result()
        spans = res.metadata.get("execution", {}).get("execution_spans")
        start = spans.start if spans is not None else None
        if start is not None and start.tzinfo is None:  # spans are naive UTC; properties() would read naive as local time
            start = start.replace(tzinfo=dt.timezone.utc)
        snap = snapshot_at(backend, start).to_dict() if start is not None else None
        if snap is not None:  # protocol v1.3 (A-006): calib.epoch = publication time of the values current at the span start
            snap["value_epoch"], _ = value_epoch(backend, start)
        counts = [pub.data.meas.get_counts() if hasattr(pub.data, "meas") else pub.data.c.get_counts() for pub in res]
        # metadata + snapshot (field values) and outcomes go to SEPARATE files: the loader encodes fields from job_*.json and
        # hashes the relation before any counts_*.json is opened (protocol §7)
        out = {"job_id": ent["job_id"], "metrics": job.metrics(), "usage_estimation": getattr(job, "usage_estimation", None),
               "exec_span_start": start.isoformat() if start is not None else None,
               "exec_span_stop": (spans.stop if spans.stop.tzinfo else spans.stop.replace(tzinfo=dt.timezone.utc)).isoformat()
               if spans is not None else None,
               "creation_time": str(job.creation_date), "session_id": getattr(job, "session_id", None),
               "backend": job.backend().name, "sdk_version": qiskit_ibm_runtime.__version__,
               "runtime_version": res.metadata.get("version"), "dd": ent["dd"], "epoch_label": ent["epoch_label"],
               "shots": ent.get("shots", SHOTS), "snapshot": snap, "pubs": ent["pubs"]}
        atomic(os.path.join(pilot_dir, f"counts_{ent['job_id']}.json"), {"job_id": ent["job_id"], "pubs": ent["pubs"],
                                                                         "counts": counts})
        atomic(os.path.join(pilot_dir, f"job_{ent['job_id']}.json"), out)
        ent["status"] = "done"
        atomic(man_path, man)
    man["usage_log"].append({"when": dt.datetime.now(dt.timezone.utc).isoformat(), "event": "fetch", "usage": service.usage()})
    epoch_check(pilot_dir, man)
    atomic(man_path, man)


def epoch_check(pilot_dir, man):
    """Protocol v1.4 (A-007): both jobs of an epoch must run on one set of calibration values (identical qubit / gate values in
    the span-start snapshots). Metadata only (job_*.json); never opens counts_*.json."""
    labels = sorted({e["epoch_label"] for e in man["jobs"].values()})
    man.setdefault("epoch_check", {})
    for lab in labels:
        ents = [e for e in man["jobs"].values() if e["epoch_label"] == lab]
        if len(ents) != 2 or any(e.get("status") != "done" for e in ents):
            continue
        snaps = [json.load(open(os.path.join(pilot_dir, f"job_{e['job_id']}.json"), encoding="utf-8"))["snapshot"] for e in ents]
        same = values(snaps[0]) == values(snaps[1])
        man["epoch_check"][lab] = {"same_calibration_values": same,
                                   "value_epochs": [sn.get("value_epoch") for sn in snaps]}
        print("epoch", lab, "same calibration values:", same, flush=True)


def resnap(pilot_dir):
    """Re-take the properties snapshot (and value_epoch) of every fetched job from its recorded span start, as aware UTC. Fixes
    jobs fetched before 2026-11-08 11:00 UTC, whose naive span start was read as local time (UTC-6). Opens job_*.json only, never
    counts_*.json; the old snapshot is kept under 'snapshot_superseded'."""
    backend = QiskitRuntimeService().backend("ibm_cleveland")
    man = json.load(open(os.path.join(pilot_dir, "manifest.json")))
    for ent in man["jobs"].values():
        if ent.get("status") != "done":
            continue
        path = os.path.join(pilot_dir, f"job_{ent['job_id']}.json")
        j = json.load(open(path, encoding="utf-8"))
        start = dt.datetime.fromisoformat(j["exec_span_start"])
        if start.tzinfo is None:
            start = start.replace(tzinfo=dt.timezone.utc)
        snap = snapshot_at(backend, start).to_dict()
        snap["value_epoch"], _ = value_epoch(backend, start)
        if "snapshot_superseded" not in j:
            j["snapshot_superseded"] = {"last_update_date": j["snapshot"]["last_update_date"],
                                        "value_epoch": j["snapshot"].get("value_epoch"),
                                        "reason": "naive span start read as local time (UTC-6) by properties(datetime=)"}
        j["snapshot"], j["exec_span_start"] = snap, start.isoformat()
        if j.get("exec_span_stop") and "+" not in j["exec_span_stop"][10:]:
            j["exec_span_stop"] = dt.datetime.fromisoformat(j["exec_span_stop"]).replace(tzinfo=dt.timezone.utc).isoformat()
        atomic(path, j)
        print(ent["job_id"], "snapshot", snap["last_update_date"], "value_epoch", snap["value_epoch"])
    epoch_check(pilot_dir, man)
    atomic(os.path.join(pilot_dir, "manifest.json"), man)


if __name__ == "__main__":
    if sys.argv[1] == "submit":
        assert "--approved-by-authors" in sys.argv, "QPU jobs need the authors's explicit approval (H0 (2))"
        submit(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "resnap":
        resnap(sys.argv[2])
    else:
        fetch(sys.argv[2])
