# Reproducing "Emergent cohesion via self-caging in maximally entangled rod packings"

One-click pipeline for the figures of arXiv:2606.03952 (paper + SI PDFs in
the repo root). Full plan and verification log:
`plan_for_reproducing_sweep_results.md`.

## Quick start

```bash
make -C repro build        # headless binary
make -C repro manifest     # 739-run manifest + scene JSONs (Table S1 protocol)
make -C repro pilot        # gates + Fig S4/End Matter/robustness groups (~2 h)
make -C repro sweep        # full Fig 4C,D sweep: 720 runs (overnight-weekend)
make -C repro figures      # figures/*.pdf from whatever runs are finished
```

`repro/run_sweep.py` is resume-safe: re-running skips finished runs
(config-hash match), so interrupted sweeps just continue.

## Layout

| Piece | Role |
|---|---|
| `assets/packings_metadata.csv` | one row per committed packing: Z, R/l, ē(0), ḡ_t, ḡ_r, A*-finite (committed, regenerable) |
| `repro/build_packing_metadata.py` | static packing metrics + ē(0) via the production binary |
| `repro/fill_free_volume.py` | ḡ columns via [rod-free-volume](https://github.com/yeonsu-jung/rod-free-volume) (SI Table S2 defaults) |
| `repro/gen_manifest.py` | manifest + per-run scenes from the Table S1 template |
| `repro/run_sweep.py` | resume-safe process-pool runner (`runs/<id>/`, gitignored) |
| `repro/analysis/` | one module per figure → `figures/*.pdf` |

## Run groups

- **c1** (720): every packing × μ∈{0,0.1,0.2,0.4} × 3 kick seeds → Fig 4C,D + S3
- **c2** (9): N=200, α∈{100,200,1000}, μ∈{0.2,0.3,0.4}, stride-1 → Fig S4
- **c3** (3): reference packing, per-contact geomspace sampling → End Matter (1/t)
- **c4** (7): solver robustness — iterations, dt, cfm, scale invariance (s=2)

Protocol constants (Table S1 / SI S3.5): dt=1e-3, PGS 200 iters, β=0,
cfm=0.05, ω=1, restitution 1, kick σ_v=0.1 / σ_ω=0.2 with axial spin
removed, t_f = 100·t_u = 320 time units (t_u = 0.32l/v₀ = 3.2).

## External inputs

- Packings (`assets/initial-configs/**/x_relaxed.txt`) are frozen outputs of
  [entanglement-optimization](https://github.com/yeonsu-jung/entanglement-optimization).
- ḡ_t/ḡ_r come from one offline pass of
  [rod-free-volume](https://github.com/yeonsu-jung/rod-free-volume)
  (`make -C repro metadata RFV=/path/to/rod_free_volume`); results are
  committed so the default pipeline has no external dependency.

## Data policy

Raw `runs/` output stays local (gitignored; archive as a tarball or GitHub
Release asset). Processed parquet/CSV summaries and `figures/*.pdf` are
committed.
