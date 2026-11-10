"""Value-changing calibration updates on ibm_cleveland (protocol v1.3, amendment A-006).

Observed 2026-11-07/08: the backend republishes its properties every 10-60 min, most republications change no value (only
`last_update_date` moves), and the value-changing updates are rolling partial batches (readout / gate errors, thresholds, T1/T2)
every ~1.5-3 h. A "calibration update" in protocol §5 / §6 is therefore a published snapshot whose VALUES differ from the
previous snapshot's; a timestamp-only republication is not one.

values(props)        -> {entry: value} over every qubit and gate parameter (per-entry dates ignored)
value_epoch(b, t)    -> last_update_date (str) of the earliest snapshot, walking back from t in STEP_MIN steps, whose values equal
                        those of the snapshot at t (= when the current values were published, to within STEP_MIN), and the
                        number of hours from it to t
"""
from __future__ import annotations

import datetime as dt

STEP_MIN = 10


def values(props_dict: dict) -> dict:
    out = {}
    for i, ql in enumerate(props_dict["qubits"]):
        for e in ql:
            out[f"q{i}.{e['name']}"] = e["value"]
    for g in props_dict["gates"]:
        for e in g["parameters"]:
            out[f"{g['gate']}{tuple(g['qubits'])}.{e['name']}"] = e["value"]
    return out


def _parse(ts) -> dt.datetime:
    """Naive datetimes are UTC (Runtime execution spans are naive UTC). qiskit-ibm-runtime's properties(datetime=) reads a naive
    datetime as LOCAL time, so every call below passes an aware UTC datetime."""
    d = ts if isinstance(ts, dt.datetime) else dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    return d.replace(tzinfo=dt.timezone.utc) if d.tzinfo is None else d.astimezone(dt.timezone.utc)


def snapshot_at(backend, t):
    """Properties snapshot as of t (aware UTC; see _parse)."""
    return backend.properties(datetime=_parse(t))


def value_epoch(backend, t: dt.datetime, max_hours: float = 24.0):
    t = _parse(t)
    p0 = backend.properties(datetime=t)
    v0 = values(p0.to_dict())
    epoch = p0.last_update_date
    s = t
    while (t - s).total_seconds() < max_hours * 3600:
        s = s - dt.timedelta(minutes=STEP_MIN)
        p = backend.properties(datetime=s)
        if p is None or values(p.to_dict()) != v0:
            break
        epoch = p.last_update_date
    epoch = _parse(epoch)
    return str(epoch), (t - epoch).total_seconds() / 3600.0
