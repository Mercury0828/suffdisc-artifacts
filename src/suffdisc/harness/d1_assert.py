"""Automated published-target assertions for D1a (review L3/D1-anchor R1, non-blocking item (a)).

Checks runs/d1/d1a_sequential.json against the sufprov Table II digits quoted in lit/GIVEN.md G1, with the PR-D1a tolerances
(|delta| < 5e-7 on 7-digit values, < 5e-5 on 4-digit values) AND the stricter half-unit-of-last-digit rounding test.
Usage: python -m suffdisc.harness.d1_assert runs/d1/d1a_sequential.json
"""
import json
import sys

# (task, eps, expected set of cost-minimal sets (as strings), {set: (published value, digits)})
TARGETS = [
    ("HEA-1", "0.05", {"LRGMA"}, {"LRGMA": (0.0, 7)}),
    ("HEA-3", "0.05", {"LRGMA"}, {"LRGMA": (0.0, 7)}),
    ("QAOA-1", "0.05", {"LGMA", "LRGM"}, {"LGMA": (0.0416650, 7), "LRGM": (0.0496812, 7)}),
    ("QAOA-2", "0.05", {"LGMA"}, {"LGMA": (0.0250, 4)}),
]
EXTRA = [("QAOA-1", "LGM", 0.0514011, 7)]  # R({L,G,M}) on QAOA-1


def tol(digits):
    return 5e-7 if digits >= 7 else 5e-5


def main(path):
    d = json.load(open(path))
    fails = []
    for task, eps, sets, vals in TARGETS:
        got = d["tasks"][task]["by_eps"][eps]["min_cost_sets"]
        got_sets = {"".join(x["set"]) for x in got}
        if got_sets != sets:
            fails.append(f"{task} eps {eps}: sets {sorted(got_sets)} != {sorted(sets)}")
        risk = {"".join(x["set"]): x["risk"] for x in got}
        for s, (pub, dig) in vals.items():
            v = risk.get(s)
            if v is None or abs(v - pub) >= tol(dig) or abs(v - pub) > 0.5 * 10 ** -dig:
                fails.append(f"{task} {s}: {v} vs published {pub} ({dig} digits)")
    for task, s, pub, dig in EXTRA:
        v = d["tasks"][task]["risk"][s]
        if abs(v - pub) >= tol(dig) or abs(v - pub) > 0.5 * 10 ** -dig:
            fails.append(f"{task} R({s}) = {v} vs published {pub}")
    print("D1a published-target assertions:", "PASS" if not fails else "FAIL", *fails, sep="\n  ")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
