"""Save ibm_cleveland static configuration (coupling map, dt, instruction durations) — no job, no QPU seconds."""
import json, sys
from qiskit_ibm_runtime import QiskitRuntimeService
b = QiskitRuntimeService().backend("ibm_cleveland")
cfg = b.configuration()
out = {"name": b.name, "num_qubits": b.num_qubits, "dt": getattr(cfg, "dt", None),
       "coupling_map": [list(e) for e in b.coupling_map.get_edges()],
       "basis_gates": list(getattr(cfg, "basis_gates", [])),
       "max_shots": getattr(cfg, "max_shots", None), "rep_delay_range": getattr(cfg, "rep_delay_range", None),
       "default_rep_delay": getattr(cfg, "default_rep_delay", None), "processor_type": getattr(cfg, "processor_type", None),
       "supported_features": getattr(cfg, "supported_features", None)}
json.dump(out, open(sys.argv[1], "w"), default=str, indent=1)
print({k: (v if k != "coupling_map" else len(v)) for k, v in out.items()})
