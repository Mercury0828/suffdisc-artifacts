"""DS-P planted-instance generator (protocol v1 §9). Deterministic given (family, M, N, seed); frozen in D0.

Record structure (all draws from numpy default_rng(seed), in the order written below):
  1. Latents: 8 categorical latents per record, cardinalities LAT_CARD, drawn uniformly, shape N x 8 (one rng.integers call
     per latent, latent order 0..7).
  2. Task key: task(r) = r mod 4.
  3. Field roles for the M fields: 4 planted fields = latents 0..3 (exact copies); of the remaining M-4 fields, the first
     round(0.2 (M-4)) are decoys, the next round(0.75 (M-4)) derived, the rest (about 5 %) noise.
     (Generator v1.1, 2026-11-07: derived 0.4 -> 0.75 and noise 0.4 -> 0.05 before any method ran, so that per-execution
     i.i.d. fields are rare, as in provenance records; v1.0 instances were deleted unused.)
       decoy  d: copy of planted field (d mod 4); then 10 % of records (rng.random < 0.1) get a fresh uniform value in that
                 field's range.
       derived: k = rng.integers(2, 9) output values; uses one latent from {4..7} (if rng.random < 0.5) or two (rng.choice
                 without replacement); value = lookup[latent codes] with lookup = rng.integers(0, k, size=prod(cards)).
       noise  : 8 values with probabilities proportional to (v+1)^-1.5 (v = 0..7), i.i.d. per record.
     Field columns are then permuted by perm = rng.permutation(M); the manifest records the planted columns.
  4. Laws (exact, per task t and planted cell c = (l0, l1, l2, l3)):
       far  : Dirichlet(1) over |Omega| = 256, one draw per (t, c) in lexicographic (t, c) order.
       near : sufprov QAOA-1 simulator (simulators/sufprov_sim.py), task seed 42 + t, routing "move_control",
              layout LAYOUTS[l1], G = NEAR_G[l2], M = NEAR_M[l3], A = NEAR_A[l0]; |Omega| = 16 zero-padded to 256 only in
              the mixed family.
       mixed: tasks 0, 1 use the far map, tasks 2, 3 the near map (padded to 256); cross-task pairs are never compared, so
              padding changes no distance.
  The planted set {latents 0..3} is sufficient by construction (law is a function of task and planted cell); it is not
  claimed to be the unique minimal sufficient set.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

from suffdisc.core import Relation
from suffdisc.simulators import sufprov_sim as sim

LAT_CARD = [2, 3, 4, 4, 6, 8, 8, 16]
NEAR_G = [0.002, 0.008, 0.016, 0.030]
NEAR_M = [0.005, 0.020, 0.040, 0.060]
NEAR_A = [0.0, 0.080]
N_TASKS = 4
NOISE_P = np.array([(v + 1) ** -1.5 for v in range(8)])
NOISE_P = NOISE_P / NOISE_P.sum()


def _near_laws():
    out = {}
    for t in range(N_TASKS):
        ang = sim.task_angles("QAOA", 1, 42 + t)
        for c in itertools.product(*[range(k) for k in LAT_CARD[:4]]):
            law, _ = sim.run_task("QAOA", 1, ang, sim.LAYOUTS[c[1]], "move_control", NEAR_G[c[2]], NEAR_M[c[3]], NEAR_A[c[0]])
            out[(t, c)] = law
    return out


_NEAR_CACHE = None


def generate(family: str, M: int, N: int, seed: int) -> tuple[Relation, dict]:
    global _NEAR_CACHE
    assert family in ("far", "near", "mixed") and M >= 8
    rng = np.random.default_rng(seed)
    lat = np.stack([rng.integers(0, k, size=N) for k in LAT_CARD], axis=1).astype(np.int32)
    task = (np.arange(N) % N_TASKS).astype(np.int32)
    n_rest = M - 4
    n_dec = int(round(0.2 * n_rest))
    n_der = int(round(0.75 * n_rest))
    cols, roles = [lat[:, j].copy() for j in range(4)], ["planted"] * 4
    for d in range(n_dec):
        src = d % 4
        v = lat[:, src].copy()
        flip = rng.random(N) < 0.1
        v[flip] = rng.integers(0, LAT_CARD[src], size=int(flip.sum()))
        cols.append(v)
        roles.append(f"decoy_of_{src}")
    for _ in range(n_der):
        k = int(rng.integers(2, 9))
        if rng.random() < 0.5:
            use = [int(rng.integers(4, 8))]
        else:
            use = sorted(int(x) for x in rng.choice(np.arange(4, 8), size=2, replace=False))
        cards = [LAT_CARD[u] for u in use]
        lookup = rng.integers(0, k, size=int(np.prod(cards)))
        idx = np.zeros(N, dtype=np.int64)
        for u, c in zip(use, cards):
            idx = idx * c + lat[:, u]
        cols.append(lookup[idx].astype(np.int32))
        roles.append("derived_" + "_".join(map(str, use)))
    for _ in range(M - len(cols)):
        cols.append(rng.choice(8, size=N, p=NOISE_P).astype(np.int32))
        roles.append("noise")
    perm = rng.permutation(M)
    Z = np.stack([cols[i] for i in perm], axis=1).astype(np.int32)
    roles = [roles[i] for i in perm]
    planted_cols = [int(np.flatnonzero(perm == j)[0]) for j in range(4)]

    cells = list(itertools.product(*[range(k) for k in LAT_CARD[:4]]))
    far = {}
    if family in ("far", "mixed"):
        for t in range(N_TASKS):
            for c in cells:
                far[(t, c)] = rng.dirichlet(np.ones(256))
    if family in ("near", "mixed") and _NEAR_CACHE is None:
        _NEAR_CACHE = _near_laws()
    B = 16 if family == "near" else 256
    laws = np.zeros((N, B))
    keys = [tuple(int(x) for x in row) for row in lat[:, :4]]
    for r in range(N):
        t = int(task[r])
        use_far = family == "far" or (family == "mixed" and t < 2)
        law = far[(t, keys[r])] if use_far else _NEAR_CACHE[(t, keys[r])]
        laws[r, : law.size] = law
    names = [f"f{j:04d}" for j in range(M)]
    rel = Relation(Z=Z, field_names=names, task=task, laws=laws)
    info = {"generator": "planted v1.1", "family": family, "M": M, "N": N, "seed": seed, "planted_columns": planted_cols,
            "roles_count": {r: roles.count(r) for r in sorted(set(roles))}, "alphabet": B,
            "relation_sha256": rel.sha256()}
    return rel, info


def save(rel: Relation, info: dict, path: str) -> None:
    np.savez_compressed(path, Z=rel.Z.astype(np.int8), task=rel.task.astype(np.int8), laws=rel.laws)
    json.dump(info, open(path + ".json", "w"), indent=1)


def load(path: str) -> tuple[Relation, dict]:
    d = np.load(path if path.endswith(".npz") else path + ".npz")
    info = json.load(open(path.replace(".npz", "") + ".json"))
    rel = Relation(Z=d["Z"].astype(np.int32), field_names=[f"f{j:04d}" for j in range(d["Z"].shape[1])],
                   task=d["task"].astype(np.int32), laws=d["laws"])
    assert rel.sha256() == info["relation_sha256"], "relation hash mismatch"
    return rel, info


PROBE = [(fam, M, N) for fam in ("far", "near", "mixed") for M in (200, 800) for N in (10_000, 100_000)]
TUNING = [(fam, 200, 10_000) for fam in ("far", "near", "mixed")] + [(fam, 800, 10_000) for fam in ("far", "near", "mixed")]


def make_manifest(out_dir: str, which: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    spec = PROBE if which == "probe" else TUNING
    base = 41001 if which == "probe" else 51001
    man_path = os.path.join(out_dir, f"{which}_manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else {}
    for i, (fam, M, N) in enumerate(spec):
        iid = f"{which}-{i:02d}-{fam}-M{M}-N{N}"
        if iid in man:
            continue
        rel, info = generate(fam, M, N, base + i)
        save(rel, info, os.path.join(out_dir, iid))
        man[iid] = info
        tmp = man_path + ".tmp"
        json.dump(man, open(tmp, "w"), indent=1)
        os.replace(tmp, man_path)
        print(iid, info["relation_sha256"], flush=True)


if __name__ == "__main__":
    make_manifest(sys.argv[1], sys.argv[2])
