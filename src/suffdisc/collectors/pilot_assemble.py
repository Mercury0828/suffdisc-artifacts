"""Assemble the D2 pilot relation (protocol §5, §7) in two separated steps.

  python -m suffdisc.collectors.pilot_assemble fields <pilot_dir> <out_dir>
      field values only: job_*.json (metadata + properties snapshot) + pilot_isa.json -> relation_fields.npz (Z, task, names),
      relation_fields.json (field report, per-record design / job id / PUB index), and relation_fields.sha256. Never opens
      counts_*.json.
  python -m suffdisc.collectors.pilot_assemble laws <pilot_dir> <out_dir>
      refuses unless relation_fields.sha256 exists and matches; then reads counts_*.json -> relation.npz (Z, task, laws, counts,
      shots, names) in the same record order, for suffdisc.harness.k3_analysis.
Law vectors are indexed by int(bitstring, 2) over the full 16-outcome alphabet (protocol §6).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

import numpy as np

from suffdisc.loader.pilot_loader import encode_records

FIELDS, BINS = "configs/fields_v1.yaml", "configs/fields_v1_bins.json"


def _jobs(pilot_dir):
    man = json.load(open(os.path.join(pilot_dir, "manifest.json"), encoding="utf-8"))
    excluded = set(man.get("excluded", {}))  # protocol v1.4 (A-007): epochs excluded before any counts file was opened
    keys = sorted(k for k, e in man["jobs"].items() if e.get("status") == "done" and e["epoch_label"] not in excluded)
    return [(k, man["jobs"][k]["job_id"]) for k in keys]


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def fields(pilot_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    meta = {m["name"]: m for m in json.load(open(os.path.join(pilot_dir, "pilot_isa.json"), encoding="utf-8"))["circuits"]}
    recs, tasks, index = [], [], []
    for key, jid in _jobs(pilot_dir):
        j = json.load(open(os.path.join(pilot_dir, f"job_{jid}.json"), encoding="utf-8"))
        assert "counts" not in j, "job file must not hold outcomes"
        for pub_i, name in enumerate(j["pubs"]):
            m = meta[name]
            recs.append({"design": {"task": m["task"], "segment": m["segment"], "orientation": m["orientation"], "dd": j["dd"],
                                    "shots": j["shots"], "opt_level": 1, "seed_transpiler": 11},
                         "circuit": {"depth": m["depth"], "ops": m["ops"]}, "snapshot": j["snapshot"],
                         "job": {"id": jid, "session_id": j.get("session_id"), "creation_time": j.get("creation_time"),
                                 "exec_span_start": j["exec_span_start"], "exec_span_stop": j.get("exec_span_stop"),
                                 "backend": j.get("backend"), "sdk_version": j.get("sdk_version"),
                                 "runtime_version": j.get("runtime_version")}})
            tasks.append(0 if m["task"] == "ghz4" else 1)
            index.append({"manifest_key": key, "job_id": jid, "pub": pub_i, "name": name, "dd": j["dd"],
                          "epoch_label": j["epoch_label"], "calib_epoch": j["snapshot"].get("value_epoch") or j["snapshot"]["last_update_date"]})
    Z, names, report = encode_records(recs, FIELDS, BINS)
    npz = os.path.join(out_dir, "relation_fields.npz")
    np.savez_compressed(npz, Z=Z, task=np.array(tasks), names=np.array(names))
    json.dump({"field_report": report, "records": index, "fields_yaml_sha256": _sha(FIELDS), "bins_sha256": _sha(BINS)},
              open(os.path.join(out_dir, "relation_fields.json"), "w", encoding="utf-8"), indent=1)
    h = _sha(npz)
    open(os.path.join(out_dir, "relation_fields.sha256"), "w").write(h + "\n")
    print({"N": Z.shape[0], "M": Z.shape[1], **{k: len(v) for k, v in report.items()}, "sha256": h})


def laws(pilot_dir, out_dir):
    npz = os.path.join(out_dir, "relation_fields.npz")
    want = open(os.path.join(out_dir, "relation_fields.sha256")).read().strip()
    assert _sha(npz) == want, "relation_fields.npz changed after hashing"
    F = np.load(npz)
    idx = json.load(open(os.path.join(out_dir, "relation_fields.json"), encoding="utf-8"))["records"]
    # protocol v1.4 (A-007): refuse if any frozen record belongs to an epoch excluded or failing the same-calibration check
    # (e.g. fields frozen before the exclusion) -- rebuild `fields` instead; checked before any counts file is opened
    man = json.load(open(os.path.join(pilot_dir, "manifest.json"), encoding="utf-8"))
    excluded, chk = set(man.get("excluded", {})), man.get("epoch_check", {})
    labels = {ent["epoch_label"] for ent in idx}
    assert not labels & excluded, f"relation_fields holds excluded epochs {sorted(labels & excluded)}: re-run `fields`"
    assert all(chk.get(lab, {}).get("same_calibration_values") for lab in labels), "epoch without a passing same-calibration check"
    cache, C = {}, np.zeros((len(idx), 16), dtype=np.int64)
    for r, ent in enumerate(idx):
        if ent["job_id"] not in cache:
            cache[ent["job_id"]] = json.load(open(os.path.join(pilot_dir, f"counts_{ent['job_id']}.json"), encoding="utf-8"))
        cj = cache[ent["job_id"]]
        assert cj["pubs"][ent["pub"]] == ent["name"]
        for bits, c in cj["counts"][ent["pub"]].items():
            C[r, int(bits, 2)] += int(c)
    shots = C.sum(axis=1)
    np.savez_compressed(os.path.join(out_dir, "relation.npz"), Z=F["Z"], task=F["task"], names=F["names"],
                        laws=C / shots[:, None], counts=C, shots=shots)
    fj = json.load(open(os.path.join(out_dir, "relation_fields.json"), encoding="utf-8"))
    json.dump({**fj, "relation_fields_sha256": want}, open(os.path.join(out_dir, "relation.json"), "w", encoding="utf-8"), indent=1)
    print({"N": int(C.shape[0]), "shots_min": int(shots.min()), "shots_max": int(shots.max())})


if __name__ == "__main__":
    {"fields": fields, "laws": laws}[sys.argv[1]](sys.argv[2], sys.argv[3])
