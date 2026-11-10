"""Build and transpile the pilot circuits (protocol §6) — no job, no QPU seconds.

Outputs <out_dir>/pilot_isa.qpy (16 ISA circuits) and <out_dir>/pilot_isa.json (per circuit: task, segment, orientation, layout,
depth, ops, measured physical qubits in classical-bit order, SHA-256 of the QPY bytes, target snapshot time).
"""
import datetime as dt
import hashlib
import io
import json
import sys

from qiskit import QuantumCircuit, qpy
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService

SEGMENTS = {"L0": [90, 91, 98, 111], "L1": [17, 27, 26, 25], "L2": [142, 143, 144, 145], "L3": [43, 56, 63, 62]}
GAMMA, BETA = 0.7, 0.4


def ghz4() -> QuantumCircuit:
    qc = QuantumCircuit(4, 4, name="ghz4")
    qc.h(0)
    for i in range(3):
        qc.cx(i, i + 1)
    qc.measure(range(4), range(4))
    return qc


def qaoa1_path() -> QuantumCircuit:
    qc = QuantumCircuit(4, 4, name="qaoa1_path")
    qc.h(range(4))
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        qc.rzz(2 * GAMMA, a, b)
    qc.rx(2 * BETA, range(4))
    qc.measure(range(4), range(4))
    return qc


def main(out_dir: str) -> None:
    backend = QiskitRuntimeService().backend("ibm_cleveland")
    circuits, meta = [], []
    for task, build in [("ghz4", ghz4), ("qaoa1_path", qaoa1_path)]:
        for seg, qs in SEGMENTS.items():
            for ori in ["forward", "reversed"]:
                layout = qs if ori == "forward" else qs[::-1]
                pm = generate_preset_pass_manager(optimization_level=1, backend=backend, initial_layout=layout,
                                                  seed_transpiler=11)
                isa = pm.run(build())
                isa.name = f"{task}|{seg}|{ori}"
                meas = {}
                for ins in isa.data:
                    if ins.operation.name == "measure":
                        meas[isa.find_bit(ins.clbits[0]).index] = isa.find_bit(ins.qubits[0]).index
                buf = io.BytesIO()
                qpy.dump(isa, buf)
                circuits.append(isa)
                meta.append({"name": isa.name, "task": task, "segment": seg, "orientation": ori, "layout": layout,
                             "depth": isa.depth(), "ops": {k: int(v) for k, v in isa.count_ops().items()},
                             "measured_physical_by_clbit": [meas[c] for c in range(4)],
                             "qpy_sha256": hashlib.sha256(buf.getvalue()).hexdigest()})
    with open(f"{out_dir}/pilot_isa.qpy", "wb") as fh:
        qpy.dump(circuits, fh)
    json.dump({"built_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "circuits": meta},
              open(f"{out_dir}/pilot_isa.json", "w"), indent=1)
    for m in meta:
        print(m["name"], m["depth"], m["ops"], m["measured_physical_by_clbit"])


if __name__ == "__main__":
    main(sys.argv[1])
