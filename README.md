# SUFFDISC — artefacts

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24120475.svg)](https://doi.org/10.5281/zenodo.24120475)

Code, data and recorded results for the paper

> J. Shen and H. Zhong, *SUFFDISC: Certified Discovery of Metric Functional Dependencies from Sampled Distributions*, submitted to the IEEE International Conference on Data Engineering (ICDE).

SUFFDISC discovers inclusion-minimal field sets whose groups of execution records have outcome distributions within a Hellinger
threshold, with a finite-shot guarantee for the true distributions.

## Layout

| path | content |
|---|---|
| `src/suffdisc/engine/suffdisc1.py` | SUFFDISC (`pivot`/`net` switches give PC-pairwise, SUFFDISC-pivot and SUFFDISC) |
| `src/suffdisc/engine/kernels.py` | Hellinger kernel, bitset difference sets, hitting-set back end |
| `src/suffdisc/baselines/b3.py`, `b0.py` | direct route B3, brute-force B0 |
| `src/suffdisc/certify.py`, `radius_v2.py`, `radius_v3.py`, `kl_lower.py`, `pair_upper.py` | pair bounds (radius, KL-set lower/upper bounds, floating-point inflation) |
| `src/suffdisc/validator/c6.py` | independent checker C6 |
| `src/suffdisc/simulators/planted.py` | planted instances (DS-P), deterministic from seeds |
| `src/suffdisc/simulators/dsa_pilot.py` | calibration-archive simulation (DS-A) |
| `src/suffdisc/harness/` | batch driver, worker, analyses and summaries |
| `data/real_records/` | the 96 real execution records (DS-R): fields, task, outcome counts |
| `data/calibration_archive_sim/` | DS-A relation with exact laws |
| `results/` | the recorded runs behind every number in the paper |

## Paper ↔ artefact

| paper | artefact |
|---|---|
| Table VI, Fig. 2 (B3 vs SUFFDISC-pivot) | `results/d3/results/*.json` |
| Table VII, Fig. 3 (validators) | `results/d3b_r0/results/*.json`, `results/d3b_r0/summary.json` |
| Correctness (Section V-D) | `results/d4/`, `results/dev_net_equiv_20261108.jsonl`, `results/d1/` |
| Tables VIII, X, Fig. 4 (real records) | `results/d2_pilot_analysis.json`, `data/real_records/` |
| Table IX (DS-A sets) | `results/d2b/d2b_k3_d5.json` |
| Table XI (simulated shots) | `results/d2b/k6fresh_*` |

## Reproduce

Python 3.12 with the packages in `requirements.txt`. From the repository root:

```bash
pip install -r requirements.txt
export PYTHONPATH=src            # Windows PowerShell: $env:PYTHONPATH="src"
# summary of the validator comparison from the shipped results
python -m suffdisc.harness.d3b_summary results/d3b_r0
# regenerate the planted probe instances (seeds 41001-41012) and the scale instances
python -m suffdisc.simulators.planted instances probe
python -m suffdisc.harness.make_d3b instances runs/d3b_spec.json
# run one batch (one single-threaded process per run; set "cores" in the spec)
python -m suffdisc.harness.driver runs/d3b_spec.json
# real-record analysis (certification at full shots and subsampled shots)
python -m suffdisc.harness.d2_analysis data/real_records/relation.npz results/d2_pilot_analysis_rerun.json
```

Timings in the paper are CPU seconds on one core of an Intel i9-13900K with single-threaded BLAS. Distance-evaluation counts do
not depend on the machine.

## License

MIT, see `LICENSE`.
