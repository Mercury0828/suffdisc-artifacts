"""Fresh-simulation confirmation of the kl rule (PR-K6-fresh, 2026-11-08; review paper-orientation R1 item 6). Identical to
k6_attempt3 (same rules, bounds, O-min and witnessed search, rng default_rng(80001 + n)) except the records: the 3 EARLIEST daily
archive epochs instead of the 3 latest (the development mirror), N_rec = 96. No knob or rule is changed after seeing these results.
Usage: python -m suffdisc.harness.k6_fresh <n_shots> <out.json>"""
import json
import sys

import numpy as np

from suffdisc.harness import k6_attempt3 as k
from suffdisc.harness.k3_analysis import meaningful


def load_first3():
    d = np.load("data/sim/dsa_pilot.npz")
    info = json.load(open("data/sim/dsa_pilot.json"))
    names = [str(x) for x in d["names"]]
    epochs = [r["epoch"] for r in info["design"]]
    first3 = sorted(set(epochs))[:3]
    rows = np.array([e in first3 for e in epochs])
    keep = meaningful(d["Z"], info["field_report"], names)
    Z = d["Z"][rows][:, keep]
    kept = [names[i] for i in keep]
    k2 = [i for i in range(Z.shape[1]) if len(np.unique(Z[:, i])) > 1]
    laws = d["laws"][rows]
    return Z[:, k2], [kept[i] for i in k2], laws / laws.sum(axis=1, keepdims=True), d["task"][rows].astype(np.int32)


if __name__ == "__main__":
    k.load_last3 = load_first3
    k.main(*sys.argv[1:3])
