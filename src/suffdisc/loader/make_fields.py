"""Generate configs/fields_v1.yaml (field universe) and configs/fields_v1_bins.json (quantile bin edges) for ibm_cleveland
pilot records (protocol §5). Uses only field VALUES from the free calibration archive — never outcomes.

Usage: python -I make_fields.py <archive_dir> <out_yaml> <out_bins_json>
"""
import glob
import hashlib
import json
import sys

import numpy as np
import yaml

LAYOUTS = {"L0": [90, 91, 98, 111], "L1": [17, 27, 26, 25], "L2": [142, 143, 144, 145], "L3": [43, 56, 63, 62]}
QPROPS = ["T1", "T2", "readout_error", "prob_meas0_prep1", "prob_meas1_prep0", "readout_length"]
GATEPROPS_1Q = ["sx", "x"]
ABS_QPROPS = ["T1", "T2", "readout_error"]


def load_archive(archive_dir):
    snaps = []
    for f in sorted(glob.glob(archive_dir + "/props_*.json")):
        d = json.load(open(f, encoding="utf-8"))
        q = {i: {p["name"]: p["value"] for p in plist} for i, plist in enumerate(d["qubits"])}
        g = {}
        for gate in d["gates"]:
            key = (gate["gate"], tuple(sorted(gate["qubits"])))
            g[key] = {p["name"]: p["value"] for p in gate["parameters"]}
        snaps.append({"lud": d["last_update_date"], "q": q, "g": g})
    return snaps


def quartile_edges(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    return [float(x) for x in np.quantile(v, [0.25, 0.5, 0.75])], int(v.size)


def main(archive_dir, out_yaml, out_bins):
    snaps = load_archive(archive_dir)
    fields, bins = [], {}

    def add(name, family, granularity, kind, path, identifier=False, values=None, note=None):
        f = {"name": name, "family": family, "granularity": granularity, "kind": kind, "path": path,
             "equality": "binned" if kind == "continuous" else "exact", "missing": "bottom_equals_only_bottom",
             "cost": 1.0, "identifier": identifier}
        if note:
            f["note"] = note
        if kind == "continuous":
            edges, n = quartile_edges(values)
            bins[name] = {"edges": edges, "n_archive_values": n, "rule": "4 quantile bins, left-closed"}
        fields.append(f)

    for name, kind, path in [("exec.segment", "categorical", "design.segment"), ("exec.orientation", "categorical", "design.orientation"),
                             ("exec.dd", "categorical", "design.dd"), ("exec.shots", "integer", "design.shots"),
                             ("exec.opt_level", "integer", "design.opt_level"),
                             ("exec.seed_transpiler", "integer", "design.seed_transpiler")]:
        add(name, "execution", "record", kind, path)
    for name in ["depth", "n_cz", "n_sx", "n_x", "n_rz"]:
        add("circ." + name, "transpiled_circuit", "record", "integer", "isa_circuit." + name)
    add("calib.epoch", "calibration_time", "record", "categorical", "snapshot.value_epoch (fallback snapshot.last_update_date; protocol v1.3 §17)",
        note="coarse time: the calibration epoch, not an exact timestamp")
    add("calib.age_bin", "calibration_time", "record", "categorical", "hours(exec_start - snapshot.last_update_date) in [0,4),[4,8),[8,16),>=16")
    # positional per logical qubit
    for i in range(4):
        phys = sorted({LAYOUTS[k][i] for k in LAYOUTS} | {LAYOUTS[k][3 - i] for k in LAYOUTS})  # both orientations
        for p in QPROPS:
            vals = [s["q"][q].get(p) for s in snaps for q in phys]
            add(f"pos.q{i}.{p}", "calibration_positional", f"logical qubit {i} (physical = layout[{i}])", "continuous",
                f"snapshot.qubits[layout[{i}]].{p}", values=vals)
        for gname in GATEPROPS_1Q:
            vals = [s["g"].get((gname, (q,)), {}).get("gate_error") for s in snaps for q in phys]
            add(f"pos.q{i}.{gname}.gate_error", "calibration_positional", f"logical qubit {i}", "continuous",
                f"snapshot.gates[{gname}, layout[{i}]].gate_error", values=vals)
    for e in [(0, 1), (1, 2), (2, 3)]:
        couplers = sorted({tuple(sorted((v[e[0]], v[e[1]]))) for k in LAYOUTS for v in (LAYOUTS[k], LAYOUTS[k][::-1])})
        for p in ["gate_error", "gate_length"]:
            vals = [s["g"].get(("cz", c), {}).get(p) for s in snaps for c in couplers]
            add(f"pos.e{e[0]}{e[1]}.cz.{p}", "calibration_positional", f"logical edge {e}", "continuous",
                f"snapshot.gates[cz, (layout[{e[0]}], layout[{e[1]}])].{p}", values=vals)
    # absolute per physical qubit / coupler of the four layouts
    allq = sorted({q for v in LAYOUTS.values() for q in v})
    for q in allq:
        for p in ABS_QPROPS:
            add(f"abs.q{q}.{p}", "calibration_absolute", f"physical qubit {q}", "continuous", f"snapshot.qubits[{q}].{p}",
                values=[s["q"][q].get(p) for s in snaps])
        add(f"abs.q{q}.sx.gate_error", "calibration_absolute", f"physical qubit {q}", "continuous",
            f"snapshot.gates[sx,{q}].gate_error", values=[s["g"].get(("sx", (q,)), {}).get("gate_error") for s in snaps])
    allc = sorted({tuple(sorted((v[j], v[j + 1]))) for v in LAYOUTS.values() for j in range(3)})
    for c in allc:
        add(f"abs.cz{c[0]}_{c[1]}.gate_error", "calibration_absolute", f"coupler {c}", "continuous",
            f"snapshot.gates[cz,{c}].gate_error", values=[s["g"].get(("cz", c), {}).get("gate_error") for s in snaps])
    for name, path in [("job.id", "job.job_id"), ("job.session_id", "job.session_id"), ("job.creation_time", "job.creation_date"),
                       ("job.exec_span_start", "job.result.metadata.execution.execution_spans.start"),
                       ("job.exec_span_stop", "job.result.metadata.execution.execution_spans.stop")]:
        add(name, "job_metadata", "record", "categorical", path, identifier=True)
    for name, path in [("job.backend", "job.backend"), ("job.sdk_version", "qiskit.__version__"),
                       ("job.runtime_version", "qiskit_ibm_runtime.__version__")]:
        add(name, "job_metadata", "record", "categorical", path)

    bins_bytes = (json.dumps(bins, indent=1, sort_keys=True) + chr(10)).encode("utf-8")
    open(out_bins, "wb").write(bins_bytes)  # LF bytes; the recorded hash is of these exact bytes
    doc = {"version": "fields_v1", "frozen": "2026-11-07", "backend": "ibm_cleveland", "layouts": LAYOUTS,
           "archive": {"dir": archive_dir, "epochs": len(snaps), "first": snaps[0]["lud"], "last": snaps[-1]["lud"]},
           "bins_file": out_bins.replace("\\", "/").split("/")[-1],
           "bins_sha256": hashlib.sha256(bins_bytes).hexdigest(),
           "units": {"T1": "us", "T2": "us", "readout_length": "ns", "gate_length": "ns", "errors_and_probabilities": ""},
           "identifier_rule": "declared identifier OR > 0.9 N distinct non-missing values on the analysed relation (values only; A-002)",
           "binning": "4 bins; edges = numpy default quantiles 0.25/0.5/0.75 of archive values; bin = searchsorted(edges, v, side='right')",
           "orientation": "forward: logical i -> segment[i]; reversed: logical i -> segment[3-i]",
           "meaningful_rule": "not constant on the pilot, not an identifier, missing in < 50 % of records",
           "n_fields": len(fields), "fields": fields}
    yaml.safe_dump(doc, open(out_yaml, "w", encoding="utf-8"), sort_keys=False, width=140)
    print(len(fields), "fields;", len(bins), "binned; bins sha256", doc["bins_sha256"])


if __name__ == "__main__":
    main(*sys.argv[1:4])
