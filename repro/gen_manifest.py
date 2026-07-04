#!/usr/bin/env python3
"""Generate the sweep manifest + per-run scene JSONs for paper reproduction.

Groups
  c1   main sweep (Fig 4C,D + S3): every committed packing x mu x 3 kick seeds
  c2   Fig S4 subset: N=200, alpha in {100,200,1000}, mu in {0.2,0.3,0.4}
  c3   End Matter: reference packing, mu in {0.1,0.2,0.4}, per-contact sampling
  c4   robustness: solver hyperparameters + scale invariance on the reference

Protocol (Table S1 / SI S3.5): dt=1e-3, PGS 200 iters, beta=0, cfm=0.05,
omega=1, restitution=1, kick vSigma=0.1 wSigma=0.2 (axial spin removed),
t_f = 100 t_u = 320 time units -> 320k steps.

Outputs: repro/manifest.csv, repro/scenes/<run_id>.json (both regenerable,
not committed). Run with repro/run_sweep.py.
"""
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
META = REPO / "assets" / "packings_metadata.csv"
SCENES = REPO / "repro" / "scenes"
MANIFEST = REPO / "repro" / "manifest.csv"

STEPS = 320_000          # t_f = 100 t_u = 320 time units at dt = 1e-3
MUS_MAIN = [0.0, 0.1, 0.2, 0.4]
KICK_SEEDS = [1, 2, 3]
# Self-caged reference (finite A*, alpha > alpha_c ~ 3N/Z): the End Matter
# 1/t law and the hyperparameter checks need sustained contacts, which a
# fragile (diverging-A*) packing sheds within a few t_u. Matches the
# paper's cohesion demo (Video 2: N=200, alpha=300).
REFERENCE = "6,7,8/2025-02-16_17_EntangledRelaxedPacking-N0200-AR0300-Scale1"


def short_id(p):
    grp = p["seed_group"].split(",")[0]
    date = p["packing_id"].split("/")[1].split("_EntangledRelaxed")[0]
    date = date.replace("2025-", "").replace("-", "").replace("_", ".")
    return f"g{grp}-{date}-N{int(p['N']):04d}A{int(p['alpha_nominal']):04d}"


def scene_json(p, mu, seed, extra_physics=None, init_scale=None):
    d = float(p["d"])
    if init_scale:
        d *= init_scale
    phys = {
        "dt": 0.001, "gravity": [0.0, 0.0, 0.0],
        "lin_damp": 0.0, "ang_damp": 0.0,
        "nsc": {"enabled": True, "mu": mu, "velocity_iters": 200,
                "beta": 0.0, "cfm": 0.05, "omega": 1.0,
                "use_spatial_hash": True},
    }
    kick = {"enabled": True, "mode": "gaussian", "vSigma": 0.1,
            "wSigma": 0.2, "seed": seed, "projectParallelSpin": True}
    if init_scale:  # dynamical similarity: v ~ s, omega ~ v/l ~ const
        kick["vSigma"] = 0.1 * init_scale
        kick["wSigma"] = 0.2
    if extra_physics:
        for k, v in extra_physics.items():
            if k in ("dt",):
                phys[k] = v
            else:
                phys["nsc"][k] = v
    return {
        "scene": {
            "initCsvPath": p["path"],
            "randomInit": kick,
            "bodies": [{"length": 1.0, "diameter": d, "density": 1000.0,
                        "restitution": 1.0, "friction": mu,
                        "friction_s": mu, "friction_d": mu}],
        },
        "physics": phys,
        "render": {"vsync": False},
    }


def main():
    packings = list(csv.DictReader(META.open()))
    by_id = {p["packing_id"]: p for p in packings}
    ref = by_id[REFERENCE]

    SCENES.mkdir(parents=True, exist_ok=True)
    rows = []

    def emit(group, p, mu, seed, steps=STEPS, csv_stride=10, ent_period=1000,
             extra_args="", tag="", extra_physics=None, init_scale=None):
        rid = f"{group}-{short_id(p)}-mu{mu:g}-k{seed}" + (f"-{tag}" if tag
                                                           else "")
        sp = SCENES / f"{rid}.json"
        sp.write_text(json.dumps(
            scene_json(p, mu, seed, extra_physics, init_scale), indent=1))
        cli = ""
        if init_scale:
            cli = f"--init-scale {init_scale}"
        if extra_args:
            cli = (cli + " " + extra_args).strip()
        rows.append({
            "run_id": rid, "group": group, "packing_id": p["packing_id"],
            "N": p["N"], "alpha": p["alpha_nominal"], "mu": mu,
            "kick_seed": seed, "steps": steps, "csv_stride": csv_stride,
            "ent_period": ent_period,
            "scene": str(sp.relative_to(REPO)), "extra_args": cli,
        })

    # ── c1: main sweep ──
    for p in packings:
        for mu in MUS_MAIN:
            for seed in KICK_SEEDS:
                emit("c1", p, mu, seed)

    # ── c2: Fig S4 (fine time resolution) ──
    for p in packings:
        if p["seed_group"] != "6,7,8" or int(p["N"]) != 200:
            continue
        if int(p["alpha_nominal"]) not in (100, 200, 1000):
            continue
        for mu in (0.2, 0.3, 0.4):
            emit("c2", p, mu, 1, csv_stride=1, ent_period=100)

    # ── c3: End Matter (per-contact velocity sampling, geomspace) ──
    em_args = ("--early-pair-diagnostics --early-pair-start 1 "
               "--early-pair-end {end} --early-pair-schedule geomspace "
               "--early-pair-geom-count 400 "
               "--early-pair-contact-csv {run_dir}/contacts_sampled.csv")
    # alpha=300: cohesive only at high mu; alpha=1000: cohesive for all mu,
    # so the 1/t tail and its 1/mu prefactor can be compared across mu.
    ref1000 = by_id["6,7,8/2025-11-05_21_EntangledRelaxedPacking-"
                    "N0200-AR1000-Scale1"]
    for p3 in (ref, ref1000):
        for mu in (0.1, 0.2, 0.4):
            emit("c3", p3, mu, 1, csv_stride=1, ent_period=100,
                 extra_args=em_args.format(end=STEPS, run_dir="{run_dir}"))

    # ── c4: robustness on the reference packing (mu=0.2, seed 1) ──
    for tag, xp in [("it100", {"velocity_iters": 100}),
                    ("it400", {"velocity_iters": 400}),
                    ("cfm0", {"cfm": 0.0}),
                    ("cfm001", {"cfm": 0.01})]:
        emit("c4", ref, 0.2, 1, tag=tag, extra_physics=xp)
    emit("c4", ref, 0.2, 1, tag="dt0500", steps=640_000, csv_stride=20,
         ent_period=2000, extra_physics={"dt": 0.0005})
    emit("c4", ref, 0.2, 1, tag="dt2000", steps=160_000, csv_stride=5,
         ent_period=500, extra_physics={"dt": 0.002})
    # scale invariance: s=2 (velocities x2, slop x2, same step count)
    emit("c4", ref, 0.2, 1, tag="scale2", init_scale=2,
         extra_physics={"slop": 2e-4})

    with MANIFEST.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_by_group = {}
    for r in rows:
        n_by_group[r["group"]] = n_by_group.get(r["group"], 0) + 1
    print(f"wrote {MANIFEST}: {len(rows)} runs {n_by_group}")
    print(f"scenes in {SCENES}")


if __name__ == "__main__":
    main()
