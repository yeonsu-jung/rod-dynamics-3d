# Result reproduction plan

Goal: **one-click reproduction** of the figures in *"Emergent cohesion via self-caging in
maximally entangled rod packings"* (arXiv:2606.03952, `2606.03952v1.pdf` + `SI.pdf` in repo
root). Target UX: `make pilot` (≈30 min sanity subset) and `make figures` (full pipeline:
sweep → analysis → `figures/*.pdf`), resumable and provenance-stamped.

---

## 0. Figure inventory — what generates what

| Figure | Content | Data source | Status in this repo |
|---|---|---|---|
| Fig 1B–D | renders of maximally entangled / relaxed packings | entanglement-optimization repo (packing gen) + viewer stills | packings in `assets/initial-configs`; renders manual/optional |
| Fig 2 | R/l vs N/(Zα), contact-point radius of gyration + Eqs. 3–4 theory | **static** contact analysis of relaxed packings (contact ⇔ d_ij < 1.01·d, Z≈4) | needs small analysis script (pybind module can do detection) |
| Fig 3 | schematic | — | n/a |
| Fig 4A,B | ḡ_t, ḡ_r vs N/(Zα) | `rod-free-volume` binary (external repo, SI Table S2) | run once → commit per-packing metadata CSV |
| **Fig 4C,D** | retention ē(t_f)/ē(0) vs ḡ_t, ḡ_r, colored by μ | **the dynamics sweep** (this repo) | main compute; all pieces exist |
| Fig S1 | two-rod ACN vs d, θ, a | analytic formulas S23–S25 | tiny standalone python script |
| Fig S2 | ē vs α for many N | entanglement-optimization repo | out of scope (document pointer) |
| Fig S3 | retention histograms, diverging vs finite A*, by μ | same sweep as Fig 4C,D + A* from metadata CSV | free once sweep exists |
| Fig S4 | collision count vs t/t_u (μ∈{0.2,0.3,0.4}, α∈{100,200,1000}) | NSC per-step collision counts | needs a proper `collisions` column (see A2) |
| End Matter | v_rel(t) ~ (g₀/μ)(1/t); per-collision decay (1−μΛ)^k | per-contact vn_pre/vn_post (`--network` export, `--debug-normal-velocity-csv`) | export implemented; analysis script needed |
| Videos 2–4 | dynamics renders | GL build | optional, manual |

## 1. Protocol — pinned from the paper (single source of truth)

**`assets/scenes/default_entangled_nsc.json` is exactly Table S1** — treat its `physics` +
`bodies` blocks as the canonical protocol and generate every sweep scene from it:

- l = 1, d = 1/α (from init-CSV header), ρ = 1000, m = ρπr²l
- dt = 0.001, NSC PGS: `velocity_iters: 200, beta: 0.0, cfm: 0.05, omega: 1.0`
- **restitution = 1.0** via `bodies[0]` (⚠ code default for CSV-loaded rods is 0.15 —
  `defaultRestitution`, `src/app/main.cpp:2562` — every generated scene must set it)
- Kick (SI S3.5): `randomInit: {mode: "gaussian", vSigma: 0.1, wSigma: 0.2,
  projectParallelSpin: true}` (v₀=0.1, ω₀=2v₀/l; axial spin removed) — already supported
- μ ∈ {0.0, 0.1, 0.2, 0.4} for Fig 4/S3; add 0.3 for Fig S4
- No gravity, no damping, free boundary (periodic disabled)
- Timescale: t_u = d*/v₀ = 0.32/0.1 = **3.2**; t_f = 100·t_u = **320** → **320,000 steps**
- ē(t): sum of |pairwise linking| (entanglement-cpp, already wired into all run loops);
  the retention ratio ē(t_f)/ē(0) cancels the n_p normalization
- Static contact criterion (Fig 2): d_ij < 1.01·d

Measured cost (this machine, 1 thread, Table S1 settings, N=200 α=50): **2.2 ms/step**
→ ≈ **12 min per full run**; N=500 estimated ~30–40 min (re-measure in Phase B).
The app auto-serializes at small N, so parallelism = many independent single-thread
processes (ideal for a process-pool runner).

## 2. Phase A — close repo gaps before sweeping

- **A1 Restitution verification.** ✅ DONE (2026-07-03). Crossed-rod point-contact test:
  ε=1, μ=0, cfm=0 gives an *exact* elastic velocity exchange (KE ratio 1.00000). Findings:
  - With cfm>0 each collision loses velocity cfm·λ, and λ scales with contact mass →
    the **effective restitution depends on rod diameter**: for paper-thin rods (d=0.02,
    α=50) the Table S1 cfm=0.05 loss is ~1.6%/collision (e_eff≈0.984); for d=0.1 rods it
    reaches ~33%. This is inherent to the paper's own protocol (same solver + cfm), so we
    reproduce with cfm=0.05 as published — but C4 adds cfm∈{0, 0.01, 0.05} sensitivity.
  - Low restitution (e≤0.15) + cfm=0.05 behaves fully plastic (cfm eats the small bounce).
  - Fixed a real footgun: scene `bodies[]` capsules ignored `radius` (only `diameter`
    counted, silently defaulting to d=0.1). Now radius-without-diameter is honored.
  - Exactly parallel rods form a line contact that a single contact point cannot
    represent (rods pivot around the tracked point) — degenerate, measure-zero in real
    packings, inherent to point-contact solvers incl. the paper's. Avoid parallel-rod
    micro-tests.
  - Regressions added to `tests/smoke_test.py` (exact exchange @cfm=0, near-elastic
    @Table S1 cfm, capsule-radius honoring).
- **A2 Collision counter.** Fig S4 needs "collisions per step", not "active contacts".
  Define collision = manifold with v_n_pre < −tol this step; emit a `collisions` CSV column
  next to `contacts`. (Cheap: vn_pre already computed per manifold.)
- **A3 ē(t) cadence.** Pick `entanglementEvery` so overhead <5% of step time (straight rods
  = 1 segment each; measure at N=500). Log to its own CSV column/file with sim-time stamp.
- **A4 Packing metadata table.** `assets/packings_metadata.csv`: one row per packing —
  id, path, N, α, gen-seed, Z (1.01d criterion), ē(0), R/l, ḡ_t, ḡ_r, A*finite? .
  Z/ē(0)/R come from our own tools; ḡ/A* from one offline `rod-free-volume` pass
  (external repo, results committed so the pipeline stays self-contained).
- **A5 Scene generator + manifest.** `repro/gen_manifest.py` → `repro/manifest.csv`
  (run_id, packing_id, μ, kick seed, steps, output dir) and one generated scene JSON per
  run under `repro/scenes/` (never hand-edited; derived from the Table S1 template).
- **A6 Runner.** `repro/run_sweep.py`: process pool of single-thread headless runs;
  **resume-safe** (skip run dirs whose config hash + completion marker match); per-run dir
  stores scene copy, git SHA, seed, wall time, stdout log.

## 3. Phase B — verification gates (pilot before burning CPU)

1. **Elastic gate:** A1 test passes; energy drift over 10k contact-free steps ≈ 0.
2. **Kick gate:** sampled σ_v, σ_ω match 0.1/0.2; ⟨ω·axis⟩ ≈ 0; bitwise determinism for a
   fixed seed (already regression-tested — rerun here).
3. **t_u gate:** one run (N=200, α=100, μ=0.2): ACN of initially crossing pairs halves at
   t ≈ 3.2 (SI S3.6). Calibrates the t/t_u axis for Fig S4.
4. **S4-shape gate:** 9-run mini-sweep (μ∈{0.2,0.3,0.4} × α∈{100,200,1000}, N=200) shows
   the qualitative Fig S4 signature: fast collision drop, late-time rise for μ≥0.3 at high α.
5. **End Matter gate:** pilot v_rel(t) shows a 1/t regime with prefactor scaling like g₀/μ
   across μ ∈ {0.1,0.2,0.4}.

Any gate failure = stop and fix physics/protocol before Phase C.

## 4. Phase C — the sweeps

| Sweep | Grid | Runs | Est. cost |
|---|---|---|---|
| C1 main (Fig 4C,D, S3) | all packings in assets (3 gen-seeds × ~10 α × N∈{200,500}) × μ∈{0,0.1,0.2,0.4} × **3 kick seeds** | ~720 | ~150 h core-time → a weekend on 16 cores |
| C2 Fig S4 | μ=0.3 additions for 3 α at N=200 | +9 | ~2 h |
| C3 End Matter | 3–5 reference runs with per-contact export (`--network`, stride-sampled to bound IO) | +5 | ~1.5 h |
| C4 robustness (our own checks) | scale-invariance s∈{1,2} (`--init-scale`, kBT/slop scaled); hyperparams (iters 100/200/400, dt ×½/×2, cfm, slop) on one reference case; contact-model comparison NSC vs Hertz–Mindlin vs harmonic vs wall-clock (`step_ms`) | +~25 | ~6 h |

Kick-seed replication: **3 seeds per packing** (decided) → C1 ≈ 720 runs,
~150 h core-time ≈ a weekend on 16 cores, or split over nights.

Raw outputs per run: `profile.csv` (KE, contacts, collisions, step_ms), entanglement
series, final state; network export only for C3. Raw dirs gitignored; processed
parquet + figures committed.

## 5. Phase D — analysis & figures

`repro/analysis/`: one module per figure (`fig2.py`, `fig4ab.py`, `fig4cd.py`, `figS1.py`,
`figS3.py`, `figS4.py`, `end_matter.py`) + shared style; each reads manifest + run dirs (or
metadata CSV), writes `data/processed/*.parquet` and `figures/*.pdf`. Overlay theory
curves where the paper has them (Eqs. 3–4 on Fig 2; 1/t on End Matter). Axes/binning
copied from the paper (log-width ḡ bins, marker ∝ √N_bin, min–max bands per α).

## 6. Phase E — one-click wiring

- `Makefile` (or a thin `repro/Makefile`): `pilot` → `sweep` → `figures` → `all`;
  `figures` runs from committed processed data so reviewers don't need the sweep.
- `repro/README.md`: exact commands, expected runtimes, external-repo pointers.
- CI: keep the existing smoke test; optionally add one 2k-step mini-run + analysis
  import check (minutes, stays in free tier). Full sweep is never CI.

## 7. External-repo boundaries

- **entanglement-optimization** (packing generation; Fig 1B, S2): committed
  `x_relaxed.txt` files are frozen inputs; document regeneration but don't depend on it.
- **rod-free-volume** (ḡ_t, ḡ_r, A*; Fig 4A,B, S3 split): run once offline, commit
  results into `assets/packings_metadata.csv`; optionally submodule it later.

## 8. Decisions (resolved 2026-07-03)

1. **Coverage:** Fig 4A,B and Fig 2 run over **all committed N and α**. The End Matter
   figure needs only a few simple reference runs (C3) — no extra packing generation.
2. **Kick seeds:** **3 per packing** → C1 ≈ 720 runs.
3. **Raw-data home:** raw `runs/` stays gitignored locally; git LFS is used only for the
   small *processed* artifacts (parquet, final figures, packing metadata), not raw sweep
   output. Rationale: GitHub LFS free tier is 1 GB storage / 1 GB-month bandwidth and
   every clone re-downloads LFS objects — tens of GB of raw CSV would burn quota
   immediately. Full raw data → tarball on external drive and/or a GitHub Release asset
   (2 GB/file, free, not fetched on clone).
4. **Videos 2–4:** yes — reproduce via GL capture in Phase E (add a frame-dump/record
   path to the GL build if missing, then ffmpeg to mp4).

## Suggested execution order

A1→A2 (physics correctness) → A4–A6 (infrastructure) → B gates → C1 kicked off overnight
→ D analysis for Fig S4/End Matter first (cheapest validation against the paper) → full
Fig 4/S3 → Fig 2/4A,B (static, independent — can run any time after A4) → E wiring.
