"""PR-D3b (2026-11-08): instances + batch spec for the timed comparison SUFFDISC-1 vs the partial-cover baseline (HPIValid principle,
pairwise validation) and the pivot-only ablation, on near-eps and mixed planted instances at N in {1e5, 3e5, 1e6}.
Seeds 71101 + index (development range; disjoint from probe 41xxx, tuning 51xxx, D4 61xxx, scale probe 71001-71003, reported
>= 90001). Usage: python -m suffdisc.harness.make_d3b <sim_dir> <spec_out.json>"""
import json
import os
import sys

from suffdisc.simulators.planted import generate, save

INST = [(fam, 200, N) for fam in ("near", "mixed") for N in (100_000, 300_000, 1_000_000)]
METHODS = {"SUFFDISC-1": {"pivot": True, "net": True, "net_h": 0.125, "net_min": 256},
           "PC-pairwise": {"pivot": False, "net": False},
           "SUFFDISC-1-pivot-only": {"pivot": True, "net": False}}
BASE = {"rounds": 2, "window": 2}  # SUFFDISC-0 knobs selected on tuning instances (RESULT TUNE-1v3); validator knobs = analysis defaults


def main(sim_dir, spec_out):
    jobs = []
    for i, (fam, M, N) in enumerate(INST):
        iid = f"d3b-{i:02d}-{fam}-M{M}-N{N}"
        path = os.path.join(sim_dir, iid)
        if not os.path.exists(path + ".npz"):
            rel, info = generate(fam, M, N, 71101 + i)
            save(rel, info, path)
            print(iid, info["relation_sha256"], flush=True)
        for eps in (0.1, 0.2):
            for m, kn in METHODS.items():
                for rep in range(3):
                    jobs.append({"instance": path.replace("\\", "/"), "eps": eps, "rep": rep, "time_limit": 14400, "batch": "d3b",
                                 "id": f"{iid}__e{eps}__{m}__r{rep}", "method": m, "knobs": {**BASE, **kn}})
    jobs.sort(key=lambda j: (j["rep"], j["instance"], j["eps"], j["method"]))  # every case gets a first repetition early
    spec = {"name": "d3b", "concurrency": 1, "jobs": jobs, "mem_cap_bytes": 12e9, "skip_rep_if_median_censored": True,
            "min_free_bytes_to_launch": 14e9}
    json.dump(spec, open(spec_out, "w"), indent=1)
    print(len(jobs), "jobs")


if __name__ == "__main__":
    main(*sys.argv[1:3])
