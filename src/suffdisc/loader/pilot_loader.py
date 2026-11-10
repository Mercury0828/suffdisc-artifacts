"""C1 record loader for ibm_cleveland pilot / DS-A records (protocol v1 §5). Field VALUES only — never reads outcomes.

A raw record is a dict:
  design   : {"task", "segment", "orientation", "dd", "shots", "opt_level", "seed_transpiler"}
  circuit  : {"depth", "ops": {gate_name: count}}
  snapshot : BackendProperties.to_dict() current at the execution-span start
  job      : {"id", "session_id", "creation_time", "exec_span_start", "exec_span_stop", "backend", "sdk_version",
              "runtime_version"}  (ISO-8601 strings for times)
encode_records(list_of_raw, fields_yaml, bins_json) -> (Z int32 N x M, field_names, report) with ⊥ = -1.
"""
from __future__ import annotations

import datetime as dt
import json
import math

import numpy as np
import yaml

MISSING = -1
UNITS = {"T1": "us", "T2": "us", "readout_length": "ns", "gate_length": "ns"}


def _qprop(snap, q, name):
    for p in snap["qubits"][q]:
        if p["name"] == name:
            want = UNITS.get(name, "")
            if (p.get("unit") or "") != want:
                return None
            v = p.get("value")
            return float(v) if v is not None and math.isfinite(float(v)) else None
    return None


def _gprop(snap, gate, qubits, name):
    key = tuple(sorted(qubits)) if gate == "cz" else tuple(qubits)
    for g in snap["gates"]:
        if g["gate"] == gate and (tuple(sorted(g["qubits"])) if gate == "cz" else tuple(g["qubits"])) == key:
            for p in g["parameters"]:
                if p["name"] == name:
                    want = UNITS.get(name, "")
                    if (p.get("unit") or "") != want:
                        return None
                    v = p.get("value")
                    return float(v) if v is not None and math.isfinite(float(v)) else None
    return None


def _parse(ts: str) -> dt.datetime:
    d = dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    return d if d.tzinfo is not None else d.replace(tzinfo=dt.timezone.utc)  # Runtime execution spans are naive UTC


def raw_values(rec: dict, cfg: dict) -> dict:
    """Unbinned value of every field for one record (None = missing)."""
    d, c, s, j = rec["design"], rec["circuit"], rec["snapshot"], rec["job"]
    seg = cfg["layouts"][d["segment"]]
    layout = list(seg) if d["orientation"] == "forward" else list(seg)[::-1]
    v = {"exec.segment": d["segment"], "exec.orientation": d["orientation"], "exec.dd": str(d["dd"]),
         "exec.shots": int(d["shots"]), "exec.opt_level": int(d["opt_level"]), "exec.seed_transpiler": int(d["seed_transpiler"]),
         "circ.depth": int(c["depth"])}
    for gname in ["cz", "sx", "x", "rz"]:
        v[f"circ.n_{gname}"] = int(c["ops"].get(gname, 0))
    ep = s.get("value_epoch") or s["last_update_date"]  # protocol v1.3 (A-006): value-changing update, not a republication
    v["calib.epoch"] = str(ep)
    try:
        hours = (_parse(j["exec_span_start"]) - _parse(ep)).total_seconds() / 3600.0
        v["calib.age_bin"] = 0 if hours < 4 else 1 if hours < 8 else 2 if hours < 16 else 3
    except Exception:
        v["calib.age_bin"] = None
    for i in range(4):
        q = layout[i]
        for p in ["T1", "T2", "readout_error", "prob_meas0_prep1", "prob_meas1_prep0", "readout_length"]:
            v[f"pos.q{i}.{p}"] = _qprop(s, q, p)
        for g in ["sx", "x"]:
            v[f"pos.q{i}.{g}.gate_error"] = _gprop(s, g, (q,), "gate_error")
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        for p in ["gate_error", "gate_length"]:
            v[f"pos.e{a}{b}.cz.{p}"] = _gprop(s, "cz", (layout[a], layout[b]), p)
    allq = sorted({q for lay in cfg["layouts"].values() for q in lay})
    for q in allq:
        for p in ["T1", "T2", "readout_error"]:
            v[f"abs.q{q}.{p}"] = _qprop(s, q, p)
        v[f"abs.q{q}.sx.gate_error"] = _gprop(s, "sx", (q,), "gate_error")
    allc = sorted({tuple(sorted((lay[k], lay[k + 1]))) for lay in cfg["layouts"].values() for k in range(3)})
    for a, b in allc:
        v[f"abs.cz{a}_{b}.gate_error"] = _gprop(s, "cz", (a, b), "gate_error")
    for k_out, k_in in [("job.id", "id"), ("job.session_id", "session_id"), ("job.creation_time", "creation_time"),
                        ("job.exec_span_start", "exec_span_start"), ("job.exec_span_stop", "exec_span_stop"),
                        ("job.backend", "backend"), ("job.sdk_version", "sdk_version"), ("job.runtime_version", "runtime_version")]:
        v[k_out] = j.get(k_in)
    return v


def encode_records(records: list[dict], fields_yaml: str, bins_json: str):
    cfg = yaml.safe_load(open(fields_yaml, encoding="utf-8"))
    bins = json.load(open(bins_json, encoding="utf-8"))
    fields = cfg["fields"]
    names = [f["name"] for f in fields]
    vals = [raw_values(r, cfg) for r in records]
    N, M = len(records), len(fields)
    Z = np.full((N, M), MISSING, dtype=np.int32)
    report = {"declared_identifiers": [], "data_identifiers": [], "constant": [], "mostly_missing": [], "meaningful": []}
    for m, f in enumerate(fields):
        col = [vals[r].get(f["name"]) for r in range(N)]
        if f["kind"] == "continuous":
            edges = np.asarray(bins[f["name"]]["edges"])
            for r, x in enumerate(col):
                if x is not None:
                    Z[r, m] = int(np.searchsorted(edges, x, side="right"))
        else:
            codes = {}
            for r, x in enumerate(col):
                if x is not None:
                    Z[r, m] = codes.setdefault(str(x), len(codes))
        present = Z[:, m][Z[:, m] != MISSING]
        n_distinct = len(np.unique(present))
        if f.get("identifier"):
            report["declared_identifiers"].append(f["name"])
        elif n_distinct > 0.9 * N:
            report["data_identifiers"].append(f["name"])
        elif (N - present.size) >= 0.5 * N:
            report["mostly_missing"].append(f["name"])
        elif len(np.unique(Z[:, m])) <= 1:
            report["constant"].append(f["name"])
        else:
            report["meaningful"].append(f["name"])
    return Z, names, report
