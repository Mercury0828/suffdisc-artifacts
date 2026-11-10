"""Summary of a D4 part-1 results file (PR-D4). Usage: python -m suffdisc.harness.d4_summary <results.jsonl> <label> [<compare.jsonl>]
With a second file, also counts cases whose returned set differs between the two files (per method; informative only, O-min answers
are not unique)."""
import json
import sys
from collections import Counter


def load(p):
    return {r["seed"]: r for r in map(json.loads, open(p, encoding="utf-8"))}


def main(path, label, compare=None):
    rows = load(path)
    fam = Counter(r["family"] for r in rows.values())
    fs = sum(1 for r in rows.values() if r["shots"])
    outputs = failures = disagree = none = 0
    mut = Counter()
    mut_n = Counter()
    for r in rows.values():
        for c in r["cases"]:
            disagree += c["c6_b0_disagreements"]
            if not c["b0_exists"]:
                none += 1
            for m, v in c["methods"].items():
                outputs += 1
                failures += 0 if v["ok"] else 1
            for k, v in c.get("mutations_caught", {}).items():
                if v is not None:
                    mut_n[k] += 1
                    mut[k] += bool(v)
    print(f"{label}; instances {len(rows)} finite-shot {fs} {dict(fam)}")
    print(f"outputs {outputs} failures {failures} | C6 vs B0 disagreements {disagree} | cases without certified-sufficient set {none}")
    print("mutations caught/tested", {k: (mut[k], mut_n[k]) for k in sorted(mut_n)})
    prereg = ["M1_drop_field", "M2_misbin", "M3_radius_zero", "M4_cross_task"]
    print("pre-registered rule 'each of M1-M4 caught on >= 1 instance':",
          "PASS" if all(mut[k] >= 1 for k in prereg) else "FAIL", "| zero failures and zero disagreements:",
          "PASS" if failures == 0 and disagree == 0 else "FAIL")
    if any("c6_b0_checks" in c for r in rows.values() for c in r["cases"]):
        print("C6 vs B0 checks performed (minimal sets + single deletions + no-solution full sets):",
              sum(c.get("c6_b0_checks", 0) for r in rows.values() for c in r["cases"]))
    if compare:
        other = load(compare)
        diff = Counter()
        for s, r in rows.items():
            for c, d in zip(r["cases"], other[s]["cases"]):
                for m in c["methods"]:
                    if c["methods"][m]["S"] != d["methods"][m]["S"]:
                        diff[m] += 1
        print("cases whose returned set differs from", compare, dict(diff))


if __name__ == "__main__":
    main(*sys.argv[1:4])
