"""D2b: calibration-archive simulation of the frozen pilot design (protocol §8). Exact laws, no QPU.

Usage: python dsa_pilot.py <archive_dir> <pilot_dir> <out_npz>
Writes the field-encoded relation (Z, field names, task, laws) and per-record design metadata.
"""
import glob
import json
import sys

import numpy as np
from qiskit import qpy
from qiskit_ibm_runtime.models import BackendProperties
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from suffdisc.loader.pilot_loader import encode_records


def strip_measure(isa, meas_phys):
    qc = isa.copy_empty_like()
    for ins in isa.data:
        if ins.operation.name not in ("measure", "barrier"):
            qc.append(ins)
    qc.save_probabilities(meas_phys, label="p")  # ordering: meas_phys[0] is the least significant bit
    return qc


def readout_apply(p_phys, snap_q, meas_phys):
    """Apply per-qubit assignment matrices; returns law over 4 classical bits, clbit 0 most significant."""
    p = p_phys.reshape((2,) * 4)  # axes: meas_phys[3], meas_phys[2], meas_phys[1], meas_phys[0] (little-endian index)
    p = np.transpose(p, (3, 2, 1, 0))  # axes now clbit0..clbit3
    for c, q in enumerate(meas_phys):
        e10 = snap_q[q].get("prob_meas1_prep0", 0.0)
        e01 = snap_q[q].get("prob_meas0_prep1", 0.0)
        A = np.array([[1 - e10, e01], [e10, 1 - e01]])  # A[measured, prepared]
        p = np.moveaxis(np.tensordot(A, p, axes=([1], [c])), 0, c)
    out = np.clip(p.reshape(-1), 0, None)
    return out / out.sum()


PILOT_QUBITS = {90, 91, 98, 111, 17, 27, 26, 25, 142, 143, 144, 145, 43, 56, 63, 62}


def epoch_laws(args):
    f, pilot_dir = args
    meta = json.load(open(f"{pilot_dir}/pilot_isa.json"))["circuits"]
    with open(f"{pilot_dir}/pilot_isa.qpy", "rb") as fh:
        isas = qpy.load(fh)
    sim = AerSimulator(method="density_matrix")
    snap = json.load(open(f))
    snap_aer = json.loads(json.dumps(snap))
    for q, pl in enumerate(snap_aer["qubits"]):  # Aer requires a 'frequency' key; with temperature=0 it has no effect
        if not any(p["name"] == "frequency" for p in pl):
            pl.append({"date": pl[0]["date"], "name": "frequency", "unit": "GHz", "value": 5.0})
        # Aer also requires T1/T2 for every qubit. Missing values occur only on qubits outside the pilot layouts (qubit 99 in
        # 7 snapshots, checked 2026-11-07); those qubits are never touched by the circuits (Aer truncates them), so a
        # placeholder there cannot change any law. A missing value on a pilot qubit raises instead.
        for k in ("T1", "T2"):
            if not any(p["name"] == k for p in pl):
                if q in PILOT_QUBITS:
                    raise ValueError(f"missing {k} on pilot qubit {q}")
                pl.append({"date": pl[0]["date"], "name": k, "unit": "us", "value": 1.0e6})
    nm = NoiseModel.from_backend_properties(BackendProperties.from_dict(snap_aer), readout_error=False)
    snap_q = {i: {p["name"]: p["value"] for p in pl} for i, pl in enumerate(snap["qubits"])}
    laws = []
    for isa, m in zip(isas, meta):
        qc = strip_measure(isa, m["measured_physical_by_clbit"])
        res = sim.run(qc, noise_model=nm, shots=1).result()
        laws.append(readout_apply(np.asarray(res.data(0)["p"]), snap_q, m["measured_physical_by_clbit"]))
    return f, laws


def main(archive_dir, pilot_dir, out_npz, procs="6"):
    from multiprocessing import Pool
    meta = json.load(open(f"{pilot_dir}/pilot_isa.json"))["circuits"]
    files = sorted(glob.glob(f"{archive_dir}/props_*.json"))
    with Pool(int(procs)) as pool:
        results = dict(pool.map(epoch_laws, [(f, pilot_dir) for f in files]))
    recs, laws, tasks = [], [], []
    for f in files:
        snap = json.load(open(f))
        for m, law in zip(meta, results[f]):
            for dd in (False, True):
                start = snap["last_update_date"]
                recs.append({"design": {"task": m["task"], "segment": m["segment"], "orientation": m["orientation"], "dd": dd,
                                        "shots": 8192, "opt_level": 1, "seed_transpiler": 11},
                             "circuit": {"depth": m["depth"], "ops": m["ops"]}, "snapshot": snap,
                             "job": {"id": None, "session_id": None, "creation_time": None, "exec_span_start": start,
                                     "exec_span_stop": None, "backend": "ibm_cleveland", "sdk_version": None,
                                     "runtime_version": None}})
                laws.append(law)
                tasks.append(0 if m["task"] == "ghz4" else 1)
    Z, names, report = encode_records(recs, "configs/fields_v1.yaml", "configs/fields_v1_bins.json")
    np.savez_compressed(out_npz, Z=Z, task=np.array(tasks), laws=np.array(laws), names=np.array(names))
    design = [{**r["design"], "epoch": r["snapshot"]["last_update_date"]} for r in recs]
    json.dump({"field_report": report, "design": design}, open(out_npz + ".json", "w"), indent=0)
    print({k: len(v) for k, v in report.items()})


if __name__ == "__main__":
    main(*sys.argv[1:5])
