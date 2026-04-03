#!/usr/bin/env python3
"""sweep_reptation.py — Parameter sweep for frictional reptation.

Launches headless simulations over combinations of friction and confinement,
with optional thermal initialization, and collects per-run summary CSVs into a
single combined file. For reptation studies that care about signed axial
displacement, the script can also emit per-run per-rod CSVs for later analysis.

Usage:
    python scripts/sweep_reptation.py [--exe PATH] [--out-dir DIR] [--dry-run]
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import os
import subprocess
import sys
import json

import numpy as np


def format_tag_value(value):
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def resolve_nonthermal_init(args):
    if args.fixed_reptation:
        return {
            "init_mode": "fixed-axial-transverse",
            "fixed_vn": args.fixed_vn,
            "fixed_vt": args.fixed_vt,
            "fixed_va": args.fixed_va,
            "fixed_vx": 0.0,
            "fixed_vy": 0.0,
            "fixed_vz": 0.0,
            "fixed_wx": 0.0,
            "fixed_wy": args.fixed_w,
            "fixed_wz": 0.0,
            "descriptor": (
                f"initfix_vn{format_tag_value(args.fixed_vn)}_"
                f"vt{format_tag_value(args.fixed_vt)}_"
                f"va{format_tag_value(args.fixed_va)}_"
                f"w{format_tag_value(args.fixed_w)}"
            ),
            "summary": {
                "init_family": "fixed-reptation",
                "init_mode": "fixed-axial-transverse",
                "init_vn": args.fixed_vn,
                "init_vt": args.fixed_vt,
                "init_va": args.fixed_va,
                "init_vx": 0.0,
                "init_vy": 0.0,
                "init_vz": 0.0,
                "init_wx": 0.0,
                "init_wy": args.fixed_w,
                "init_wz": 0.0,
            },
        }

    if args.init_mode == "gaussian-axial-transverse":
        return {
            "init_mode": "gaussian-axial-transverse",
            "fixed_vn": 0.0,
            "fixed_vt": 0.0,
            "fixed_va": 0.0,
            "fixed_vx": 0.0,
            "fixed_vy": 0.0,
            "fixed_vz": 0.0,
            "fixed_wx": 0.0,
            "fixed_wy": 0.0,
            "fixed_wz": 0.0,
            "descriptor": (
                f"initgatr_svn{format_tag_value(args.sigma_vn)}_"
                f"svt{format_tag_value(args.sigma_vt)}_"
                f"sva{format_tag_value(args.sigma_va)}_"
                f"sw{format_tag_value(args.sigma_w_reptation)}"
            ),
            "summary": {
                "init_family": "gaussian-reptation",
                "init_mode": "gaussian-axial-transverse",
                "init_vn": "",
                "init_vt": "",
                "init_va": "",
                "init_vx": "",
                "init_vy": "",
                "init_vz": "",
                "init_wx": "",
                "init_wy": "",
                "init_wz": "",
            },
        }

    if args.init_mode == "random":
        return {
            "init_mode": "random",
            "fixed_vn": args.fixed_vn,
            "fixed_vt": args.fixed_vt,
            "fixed_va": args.fixed_va,
            "fixed_vx": args.fixed_vx,
            "fixed_vy": args.fixed_vy,
            "fixed_vz": args.fixed_vz,
            "fixed_wx": args.fixed_wx,
            "fixed_wy": args.fixed_wy,
            "fixed_wz": args.fixed_wz,
            "descriptor": f"initrand_sv{format_tag_value(args.sigma_v)}_sw{format_tag_value(args.sigma_w)}",
            "summary": {
                "init_family": "random",
                "init_mode": "random",
                "init_vn": "",
                "init_vt": "",
                "init_va": "",
                "init_vx": "",
                "init_vy": "",
                "init_vz": "",
                "init_wx": "",
                "init_wy": "",
                "init_wz": "",
            },
        }

    if args.init_mode == "fixed-axial-transverse":
        return {
            "init_mode": "fixed-axial-transverse",
            "fixed_vn": args.fixed_vn,
            "fixed_vt": args.fixed_vt,
            "fixed_va": args.fixed_va,
            "fixed_vx": 0.0,
            "fixed_vy": 0.0,
            "fixed_vz": 0.0,
            "fixed_wx": args.fixed_wx,
            "fixed_wy": args.fixed_wy,
            "fixed_wz": args.fixed_wz,
            "descriptor": (
                f"initfat_vn{format_tag_value(args.fixed_vn)}_"
                f"vt{format_tag_value(args.fixed_vt)}_"
                f"va{format_tag_value(args.fixed_va)}_"
                f"wx{format_tag_value(args.fixed_wx)}_"
                f"wy{format_tag_value(args.fixed_wy)}_"
                f"wz{format_tag_value(args.fixed_wz)}"
            ),
            "summary": {
                "init_family": "fixed",
                "init_mode": "fixed-axial-transverse",
                "init_vn": args.fixed_vn,
                "init_vt": args.fixed_vt,
                "init_va": args.fixed_va,
                "init_vx": "",
                "init_vy": "",
                "init_vz": "",
                "init_wx": args.fixed_wx,
                "init_wy": args.fixed_wy,
                "init_wz": args.fixed_wz,
            },
        }

    return {
        "init_mode": "fixed-cartesian",
        "fixed_vn": 0.0,
        "fixed_vt": 0.0,
        "fixed_va": 0.0,
        "fixed_vx": args.fixed_vx,
        "fixed_vy": args.fixed_vy,
        "fixed_vz": args.fixed_vz,
        "fixed_wx": args.fixed_wx,
        "fixed_wy": args.fixed_wy,
        "fixed_wz": args.fixed_wz,
        "descriptor": (
            f"initfc_vx{format_tag_value(args.fixed_vx)}_"
            f"vy{format_tag_value(args.fixed_vy)}_"
            f"vz{format_tag_value(args.fixed_vz)}_"
            f"wx{format_tag_value(args.fixed_wx)}_"
            f"wy{format_tag_value(args.fixed_wy)}_"
            f"wz{format_tag_value(args.fixed_wz)}"
        ),
        "summary": {
            "init_family": "fixed",
            "init_mode": "fixed-cartesian",
            "init_vn": "",
            "init_vt": "",
            "init_va": "",
            "init_vx": args.fixed_vx,
            "init_vy": args.fixed_vy,
            "init_vz": args.fixed_vz,
            "init_wx": args.fixed_wx,
            "init_wy": args.fixed_wy,
            "init_wz": args.fixed_wz,
        },
    }


def add_velocity_initialization(cmd, init_mode, rng, sigma_v, sigma_w,
                                sigma_vn, sigma_vt, sigma_va, sigma_w_reptation,
                                fixed_vx, fixed_vy, fixed_vz,
                                fixed_vn, fixed_vt, fixed_va,
                                fixed_wx, fixed_wy, fixed_wz):
    if init_mode == "random":
        v0 = rng.normal(0, sigma_v)
        w0 = rng.normal(0, sigma_w, size=2)
        cmd.extend([
            "--set-velocity", "0", "0", f"{v0:.6f}", "0",
            "--set-ang-velocity", "0", f"{w0[0]:.6f}", "0", f"{w0[1]:.6f}",
        ])
        return {
            "label": f"random axial/tumble v0={v0:.4f}",
            "vx": 0.0,
            "vy": float(v0),
            "vz": 0.0,
            "wx": float(w0[0]),
            "wy": 0.0,
            "wz": float(w0[1]),
        }

    if init_mode == "gaussian-axial-transverse":
        vx = rng.normal(0.0, sigma_vn)
        vy = rng.normal(0.0, sigma_va)
        vz = rng.normal(0.0, sigma_vt)
        wy = rng.normal(0.0, sigma_w_reptation)
        cmd.extend([
            "--set-velocity", "0", f"{vx:.6f}", f"{vy:.6f}", f"{vz:.6f}",
            "--set-ang-velocity", "0", "0.000000", f"{wy:.6f}", "0.000000",
        ])
        return {
            "label": (
                f"gaussian axial/transverse sigma_vn={sigma_vn:.4f} sigma_vt={sigma_vt:.4f} "
                f"sigma_va={sigma_va:.4f} sigma_w={sigma_w_reptation:.4f}"
            ),
            "vx": float(vx),
            "vy": float(vy),
            "vz": float(vz),
            "wx": 0.0,
            "wy": float(wy),
            "wz": 0.0,
        }

    if init_mode == "fixed-cartesian":
        cmd.extend([
            "--set-velocity", "0", f"{fixed_vx:.6f}", f"{fixed_vy:.6f}", f"{fixed_vz:.6f}",
            "--set-ang-velocity", "0", f"{fixed_wx:.6f}", f"{fixed_wy:.6f}", f"{fixed_wz:.6f}",
        ])
        return {
            "label": (
                f"fixed cartesian v=({fixed_vx:.4f},{fixed_vy:.4f},{fixed_vz:.4f}) "
                f"w=({fixed_wx:.4f},{fixed_wy:.4f},{fixed_wz:.4f})"
            ),
            "vx": float(fixed_vx),
            "vy": float(fixed_vy),
            "vz": float(fixed_vz),
            "wx": float(fixed_wx),
            "wy": float(fixed_wy),
            "wz": float(fixed_wz),
        }

    if init_mode == "fixed-axial-transverse":
        vx = fixed_vn
        vy = fixed_va
        vz = fixed_vt
        cmd.extend([
            "--set-velocity", "0", f"{vx:.6f}", f"{vy:.6f}", f"{vz:.6f}",
            "--set-ang-velocity", "0", f"{fixed_wx:.6f}", f"{fixed_wy:.6f}", f"{fixed_wz:.6f}",
        ])
        return {
            "label": (
                f"fixed axial/transverse vn={fixed_vn:.4f} vt={fixed_vt:.4f} va={fixed_va:.4f} "
                f"w=({fixed_wx:.4f},{fixed_wy:.4f},{fixed_wz:.4f})"
            ),
            "vx": float(vx),
            "vy": float(vy),
            "vz": float(vz),
            "wx": float(fixed_wx),
            "wy": float(fixed_wy),
            "wz": float(fixed_wz),
        }

    raise ValueError(f"Unsupported init mode: {init_mode}")


def run_one_local_job(job):
    result = subprocess.run(job["cmd"], capture_output=True, text=True)
    return job, result


def run_local_jobs(jobs, max_workers):
    if not jobs:
        return

    print(f"Running {len(jobs)} local simulations with jobs={max_workers}")
    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_one_local_job, job) for job in jobs]
        for future in as_completed(futures):
            job, result = future.result()
            if result.returncode == 0:
                print(f"  completed [{job['index']}/{job['total']}] mu={job['mu']} gap={job['gap']} trial={job['trial']}")
                continue

            failures.append((job, result))
            print(
                f"  WARNING: failed [{job['index']}/{job['total']}] "
                f"mu={job['mu']} gap={job['gap']} trial={job['trial']} "
                f"exit={result.returncode}",
                file=sys.stderr,
            )
            if result.stderr:
                print(f"  stderr: {result.stderr[:400]}", file=sys.stderr)

    if failures:
        raise SystemExit(f"{len(failures)} local simulation(s) failed")


def make_scene(base: dict, R: float, friction: float,
               use_cuda: bool,
               use_nsc: bool = False, sigma_v: float = None,
               sigma_w: float = None,
               thermal: bool = False,
               rod_length: float | None = None,
               rod_diameter: float | None = None) -> dict:
    d = json.loads(json.dumps(base))

    # Apply tube radius
    d.setdefault("scene", {}).setdefault("cylinder", {})["radius"] = R

    bodies = d.setdefault("scene", {}).setdefault("bodies", [])
    if bodies:
        if rod_length is not None:
            bodies[0]["length"] = rod_length
        if rod_diameter is not None:
            bodies[0]["diameter"] = rod_diameter

    # Thermal mode: compute translational and rotational thermal energies
    if thermal and sigma_v is not None:
        import math
        rod_length = 1.0
        rod_diameter = 0.01  # from reptation.json base
        rod_density = 1000.0  # from reptation.json base
        if "scene" in d and "bodies" in d["scene"] and d["scene"]["bodies"]:
            b0 = d["scene"]["bodies"][0]
            rod_length = b0.get("length", rod_length)
            rod_diameter = b0.get("diameter", rod_diameter)
            rod_density = b0.get("density", rod_density)
        
        rod_radius = rod_diameter / 2.0
        rod_mass = rod_density * math.pi * rod_radius**2 * rod_length
        rod_I_perp = (rod_mass / 12.0) * (rod_length**2 + 3.0 * rod_radius**2)
        kBT_trans = rod_mass * sigma_v**2
        kBT_rot = rod_I_perp * sigma_w**2 if sigma_w is not None else kBT_trans
        
        d["scene"]["randomInit"] = {
            "enabled": True,
            "mode": "thermal",
            "kBT": kBT_trans,
            "kBTTrans": kBT_trans,
            "kBTRot": kBT_rot,
            "seed": 42,
            "projectParallelSpin": True,
        }
    else:
        # Disable randomInit if not using thermal (rely on CLI --set-velocity)
        if "randomInit" in d.get("scene", {}):
            d["scene"]["randomInit"]["enabled"] = False 

    # Apply friction
    if use_nsc:
        nsc = d.setdefault("physics", {}).setdefault("nsc", {})
        nsc["enabled"] = True
        nsc["mu"] = friction
    else:
        sc = d.setdefault("physics", {}).setdefault("soft_contact", {})
        sc["enabled"] = True
        sc["mu"] = friction
        sc["mu_static"] = friction
        
    if use_cuda:
        d.setdefault("physics", {}).setdefault("soft_contact", {})["use_cuda"] = True
        
    return d


def main():
    parser = argparse.ArgumentParser(description="Reptation parameter sweep")
    parser.add_argument("--exe", default="./build/rigidbody_viewer_3d",
                        help="Path to headless binary")
    parser.add_argument("--scene", default="assets/scenes/reptation.json",
                        help="Base scene JSON")
    parser.add_argument("--out-dir", default="results/reptation",
                        help="Directory for output CSVs")
    parser.add_argument("--steps", type=int, default=100_000,
                        help="Max simulation steps")
    parser.add_argument("--dt", type=float, default=0.001,
                        help="Simulation timestep override passed to the binary")
    parser.add_argument("--stop-ke", type=float, default=1e-8,
                        help="KE threshold for early stop")
    parser.add_argument("--stop-ke-avg-window", type=int, default=5,
                        help="Rolling average window for stop-KE")
    parser.add_argument("--no-stop-ke", action="store_true",
                        help="Do not pass KE early-stop flags to the binary")
    parser.add_argument("--stop-slide-vel-threshold", type=float, default=None,
                        help="Optional axial sliding-speed threshold for early stop")
    parser.add_argument("--stop-slide-vel-min-steps", type=int, default=0,
                        help="Minimum steps before axial sliding-speed stop check")
    parser.add_argument("--mus", type=float, nargs="+",
                        # default=[0.01, 0.05, 0.1, 0.3, 0.5, 1.0],
                        default=[1.0],
                        help="Friction coefficients to sweep")
    parser.add_argument("--radii", type=float, nargs="+",
                        default=None,
                        help="Cylinder radii to sweep")
    parser.add_argument("--gaps", type=float, nargs="+",
                        default=None,
                        help="Gap values to sweep, where gap = R_cyl - rod_radius")
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of random trials per (mu, R)")
    parser.add_argument("--rod-length", type=float, default=1.0,
                        help="Rod length for the sweep scene")
    parser.add_argument("--aspect-ratio", type=float, default=None,
                        help="Rod aspect ratio L/d. If provided, overrides scene diameter")
    parser.add_argument("--rod-diameter", type=float, default=None,
                        help="Rod diameter. Used if --aspect-ratio is not set")
    parser.add_argument("--sigma-v", type=float, default=1.0,
                        help="Std-dev of initial axial velocity (MB) (Used for kBT in thermal init)")
    parser.add_argument("--sigma-w", type=float, default=0.5,
                        help="Std-dev of initial angular velocity components (Used to set rotational kBT in thermal mode)")
    parser.add_argument("--thermal", action="store_true",
                        help="Use native C++ thermal (Maxwell-Boltzmann) initialization")
    parser.add_argument("--fixed-reptation", action="store_true",
                        help="Convenience preset for reptation studies: uses fixed axial/transverse init with scalar axial spin --fixed-w")
    parser.add_argument(
        "--init-mode",
        choices=["random", "fixed-axial-transverse", "fixed-cartesian", "gaussian-axial-transverse"],
        default="random",
        help=(
            "Initialization pathway for non-thermal runs: random preserves the existing axial/tumbling kick; "
            "fixed-axial-transverse maps (vn, va, vt) to (x, y, z); fixed-cartesian uses explicit (vx, vy, vz); "
            "gaussian-axial-transverse samples reptation coordinates component-wise."
        ),
    )
    parser.add_argument("--gap-radius-basis", choices=["radius", "diameter"], default="radius",
                        help="Interpret --gaps as R = rod_radius + gap or R = rod_diameter + gap")
    parser.add_argument("--fixed-vn", type=float, default=0.0,
                        help="Fixed transverse-normal translational velocity for fixed-axial-transverse mode (mapped to x)")
    parser.add_argument("--fixed-vt", type=float, default=0.0,
                        help="Fixed transverse-tangential translational velocity for fixed-axial-transverse mode (mapped to z)")
    parser.add_argument("--fixed-va", type=float, default=0.0,
                        help="Fixed axial translational velocity for fixed-axial-transverse mode (mapped to y)")
    parser.add_argument("--fixed-w", type=float, default=0.0,
                        help="Convenience scalar spin about the rod axis for --fixed-reptation (mapped to wy)")
    parser.add_argument("--fixed-vx", type=float, default=0.0,
                        help="Fixed x translational velocity for fixed-cartesian mode")
    parser.add_argument("--fixed-vy", type=float, default=0.0,
                        help="Fixed y translational velocity for fixed-cartesian mode")
    parser.add_argument("--fixed-vz", type=float, default=0.0,
                        help="Fixed z translational velocity for fixed-cartesian mode")
    parser.add_argument("--fixed-wx", type=float, default=0.0,
                        help="Fixed x angular velocity for fixed non-thermal modes")
    parser.add_argument("--fixed-wy", type=float, default=0.0,
                        help="Fixed y angular velocity for fixed non-thermal modes")
    parser.add_argument("--fixed-wz", type=float, default=0.0,
                        help="Fixed z angular velocity for fixed non-thermal modes")
    parser.add_argument("--sigma-vn", type=float, default=0.0,
                        help="Gaussian std-dev for reptation vn in gaussian-axial-transverse mode")
    parser.add_argument("--sigma-vt", type=float, default=0.0,
                        help="Gaussian std-dev for reptation vt in gaussian-axial-transverse mode")
    parser.add_argument("--sigma-va", type=float, default=0.0,
                        help="Gaussian std-dev for reptation va in gaussian-axial-transverse mode")
    parser.add_argument("--sigma-w-reptation", type=float, default=0.0,
                        help="Gaussian std-dev for reptation scalar w in gaussian-axial-transverse mode")
    parser.add_argument("--use-cuda", action="store_true",
                        help="Set use_cuda in soft contact scene settings")
                        
    # NSC arguments for alignment
    parser.add_argument("--nsc", action="store_true",
                        help="Use NSC (impulse-based) contact solver instead of soft contact.")
    parser.add_argument("--nsc-iters", type=int, default=200)
    parser.add_argument("--nsc-beta", type=float, default=0.0)
    parser.add_argument("--nsc-cfm", type=float, default=0.05)
    parser.add_argument("--nsc-omega", type=float, default=1.0)
    parser.add_argument("--nsc-pos-iters", type=int, default=5)
    parser.add_argument("--nsc-pos-psor", type=int, default=50)

    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--sbatch", action="store_true",
                        help="Submit as a Slurm Job Array instead of sequential local execution")
    parser.add_argument("--cpus", type=int, default=1,
                        help="Number of CPUs per Slurm task")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Number of local runs to execute concurrently")
    parser.add_argument("--combined-csv", default=None,
                        help="Path for combined summary CSV (default: <out-dir>/combined.csv)")
    parser.add_argument("--perrod", action="store_true",
                        help="Emit per-run per-rod CSVs for later signed displacement analysis")
    parser.add_argument("--perrod-stride", type=int, default=100,
                        help="Stride for per-run per-rod CSVs when --perrod is enabled")
    parser.add_argument("--perrod-max", type=int, default=None,
                        help="Maximum number of per-rod frames to write when --perrod is enabled")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    combined_path = args.combined_csv or os.path.join(args.out_dir, "combined.csv")

    with open(args.scene, "r") as f:
        base_scene = json.load(f)

    base_bodies = base_scene.get("scene", {}).get("bodies", [])
    if not base_bodies:
        raise SystemExit("Base scene must contain at least one body for reptation sweep")

    rod_length = args.rod_length
    if args.aspect_ratio is not None:
        if args.aspect_ratio <= 0:
            raise SystemExit("--aspect-ratio must be positive")
        rod_diameter = rod_length / args.aspect_ratio
    elif args.rod_diameter is not None:
        rod_diameter = args.rod_diameter
    else:
        rod_diameter = base_bodies[0].get("diameter", 0.01)

    rod_radius = rod_diameter / 2.0
    if args.gaps is not None:
        gap_offset = rod_radius if args.gap_radius_basis == "radius" else rod_diameter
        gap_radius_pairs = [(gap, gap + gap_offset) for gap in args.gaps]
    elif args.radii is not None:
        gap_radius_pairs = [(None, radius) for radius in args.radii]
    else:
        gap_radius_pairs = [(None, radius) for radius in [0.2, 0.3, 0.5]]

    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    if args.thermal and args.fixed_reptation:
        raise SystemExit("--fixed-reptation cannot be combined with --thermal")
    if args.thermal and args.init_mode != "random":
        raise SystemExit("--thermal cannot be combined with a non-random --init-mode")

    init_cfg = None if args.thermal else resolve_nonthermal_init(args)

    total = len(args.mus) * len(gap_radius_pairs) * args.trials
    run_idx = 0
    array_commands = []
    local_jobs = []

    for mu, (gap_input, R) in itertools.product(args.mus, gap_radius_pairs):
        for trial in range(args.trials):
            run_idx += 1
            rng = np.random.default_rng(seed=trial)
            if gap_input is None:
                gap_label = R - rod_radius
            else:
                gap_label = gap_input
            
            tag = f"AR{args.aspect_ratio if args.aspect_ratio is not None else rod_length / rod_diameter:g}_gap{gap_label}_mu{mu}_{'initthermal' if args.thermal else init_cfg['descriptor']}_t{trial}"
            summary_path = os.path.join(args.out_dir, f"rept_{tag}.csv")
            scene_path = os.path.join(args.out_dir, f"scene_{tag}.json")
            perrod_path = os.path.join(args.out_dir, f"perrod_{tag}.csv") if args.perrod else None
            
            scene_data = make_scene(base_scene, R=R, friction=mu, use_cuda=args.use_cuda,
                                    use_nsc=args.nsc, sigma_v=args.sigma_v,
                                    sigma_w=args.sigma_w,
                                    thermal=args.thermal, rod_length=rod_length,
                                    rod_diameter=rod_diameter)
            
            # Explicit seed override via JSON
            if args.thermal:
                scene_data["scene"]["randomInit"]["seed"] = trial

            with open(scene_path, "w") as f:
                json.dump(scene_data, f, indent=2)

            cmd = [
                args.exe,
                "--headless", "--steps", str(args.steps),
                "--scene", scene_path,
                "--reptation-summary", summary_path,
                "--quiet",
            ]

            if args.dt is not None:
                cmd.extend(["--dt", str(args.dt)])

            if not args.no_stop_ke:
                cmd.extend([
                    "--stop-ke-threshold", str(args.stop_ke),
                    "--stop-ke-avg-window", str(args.stop_ke_avg_window),
                ])

            if args.stop_slide_vel_threshold is not None:
                cmd.extend([
                    "--stop-slide-vel-threshold", str(args.stop_slide_vel_threshold),
                    "--stop-slide-vel-min-steps", str(args.stop_slide_vel_min_steps),
                ])

            if perrod_path is not None:
                cmd.extend([
                    "--perrod", perrod_path,
                    "--perrod-stride", str(args.perrod_stride),
                ])
                if args.perrod_max is not None:
                    cmd.extend(["--perrod-max", str(args.perrod_max)])

            if args.nsc:
                cmd.extend([
                    "--nsc",
                    "--nsc-iters", str(args.nsc_iters),
                    "--nsc-beta", str(args.nsc_beta),
                    "--nsc-pos-iters", str(args.nsc_pos_iters),
                    "--nsc-pos-psor", str(args.nsc_pos_psor)
                ])
                if args.nsc_cfm != 0.0:
                    cmd.extend(["--nsc-cfm", str(args.nsc_cfm)])
                if args.nsc_omega != 1.0:
                    cmd.extend(["--nsc-omega", str(args.nsc_omega)])
            
            if not args.thermal:
                init_info = add_velocity_initialization(
                    cmd,
                    init_mode=init_cfg["init_mode"],
                    rng=rng,
                    sigma_v=args.sigma_v,
                    sigma_w=args.sigma_w,
                    sigma_vn=args.sigma_vn,
                    sigma_vt=args.sigma_vt,
                    sigma_va=args.sigma_va,
                    sigma_w_reptation=args.sigma_w_reptation,
                    fixed_vx=init_cfg["fixed_vx"],
                    fixed_vy=init_cfg["fixed_vy"],
                    fixed_vz=init_cfg["fixed_vz"],
                    fixed_vn=init_cfg["fixed_vn"],
                    fixed_vt=init_cfg["fixed_vt"],
                    fixed_va=init_cfg["fixed_va"],
                    fixed_wx=init_cfg["fixed_wx"],
                    fixed_wy=init_cfg["fixed_wy"],
                    fixed_wz=init_cfg["fixed_wz"],
                )
                print(
                    f"[{run_idx}/{total}] mu={mu} gap={gap_label} R={R} trial={trial} {init_info['label']}"
                )
            else:
                print(f"[{run_idx}/{total}] mu={mu} gap={gap_label} R={R} trial={trial} (thermal kBT)")

            if args.dry_run:
                print("  " + " ".join(cmd))
                continue

            if args.sbatch:
                import shlex
                # Ensure the path contains absolute structure or resolve via cd
                array_commands.append(" ".join(shlex.quote(c) for c in cmd))
            else:
                local_jobs.append({
                    "index": run_idx,
                    "total": total,
                    "mu": mu,
                    "gap": gap_label,
                    "trial": trial,
                    "cmd": cmd,
                })

    if args.sbatch and array_commands:
        cmds_file = os.path.join(args.out_dir, "array_commands.txt")
        with open(cmds_file, "w") as f:
            f.write("\n".join(array_commands) + "\n")
            
        master_sb = f"""#!/bin/bash
#SBATCH -n 1
#SBATCH -c {args.cpus}
#SBATCH -t 0-12:00:00
#SBATCH -p seas_compute
#SBATCH --mem=8G
#SBATCH --array=1-{len(array_commands)}
#SBATCH -o {os.path.abspath(args.out_dir)}/output_%A_%a.out
#SBATCH -e {os.path.abspath(args.out_dir)}/errors_%A_%a.err
#SBATCH --job-name=sweep_rept_array

set -euo pipefail

cd {os.path.abspath('.')}

CMD=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {os.path.abspath(cmds_file)})
echo "Running Array task $SLURM_ARRAY_TASK_ID"
echo "Command: $CMD"

eval "$CMD"
echo "Job complete."
"""
        master_sbatch_path = os.path.join(args.out_dir, "Master_Sbatch.sh")
        with open(master_sbatch_path, "w") as f:
            f.write(master_sb)
            
        print(f"Submitting SLURM Array Job with {len(array_commands)} tasks...")
        subprocess.run(["sbatch", "Master_Sbatch.sh"], cwd=args.out_dir)

    elif not args.dry_run and not args.sbatch:
        run_local_jobs(local_jobs, max_workers=args.jobs)

        # Combine all individual summary CSVs into one (Local execution)
        combine_summaries(
            args.out_dir,
            combined_path,
            thermal=args.thermal,
            init_cfg=init_cfg,
            sigma_v=args.sigma_v,
            sigma_w=args.sigma_w,
            rod_radius=rod_radius,
            rod_length=rod_length,
            rod_diameter=rod_diameter,
            gap_radius_basis=args.gap_radius_basis,
            stop_slide_vel_threshold=args.stop_slide_vel_threshold,
            stop_slide_vel_min_steps=args.stop_slide_vel_min_steps,
        )
        print(f"\nDone. Combined summary: {combined_path}")


def combine_summaries(out_dir, combined_path, thermal=False,
                      init_cfg=None, sigma_v=None, sigma_w=None,
                      rod_radius=None,
                      rod_length=None, rod_diameter=None,
                      gap_radius_basis="radius",
                      stop_slide_vel_threshold=None,
                      stop_slide_vel_min_steps=None):
    """Concatenate all rept_*.csv files into a single CSV."""
    import glob
    files = sorted(glob.glob(os.path.join(out_dir, "rept_*.csv")))
    if not files:
        print("No summary files found to combine.")
        return

    header = None
    rows = []
    with open(combined_path, "w") as out:
        for f in files:
            with open(f) as inp:
                lines = [line.strip() for line in inp.readlines() if line.strip()]
                if not lines:
                    continue
                local_header = lines[0]
                if header is None:
                    header = local_header
                for line in lines[1:]:
                    rows.append(line)

        if header is None:
            print("No summary rows found to combine.")
            return

        extra_cols = ["gap", "init_family", "init_mode", "sigma_v_input", "sigma_w_input",
                      "init_vn", "init_vt", "init_va", "init_vx", "init_vy", "init_vz",
                      "init_wx", "init_wy", "init_wz"]
        if rod_radius is not None:
            extra_cols.append("rod_radius")
        if rod_length is not None:
            extra_cols.append("rod_length_input")
        if rod_diameter is not None:
            extra_cols.append("rod_diameter_input")
        if stop_slide_vel_threshold is not None:
            extra_cols.append("stop_slide_vel_threshold")
            extra_cols.append("stop_slide_vel_min_steps")

        out.write(header + "," + ",".join(extra_cols) + "\n")
        header_cols = header.split(",")
        idx_R = header_cols.index("R_cyl") if "R_cyl" in header_cols else None
        for row in rows:
            values = row.split(",")
            extras = []
            if idx_R is not None and rod_radius is not None:
                if gap_radius_basis == "diameter" and rod_diameter is not None:
                    gap = float(values[idx_R]) - rod_diameter
                else:
                    gap = float(values[idx_R]) - rod_radius
                extras.append(str(gap))
            else:
                extras.append("")
            if thermal:
                extras.extend([
                    "thermal",
                    "thermal",
                    str(sigma_v),
                    str(sigma_w),
                    "", "", "", "", "", "", "", "", "",
                ])
            else:
                summary = init_cfg["summary"] if init_cfg is not None else {}
                extras.extend([
                    str(summary.get("init_family", "")),
                    str(summary.get("init_mode", "")),
                    str(sigma_v),
                    str(sigma_w),
                    str(summary.get("init_vn", "")),
                    str(summary.get("init_vt", "")),
                    str(summary.get("init_va", "")),
                    str(summary.get("init_vx", "")),
                    str(summary.get("init_vy", "")),
                    str(summary.get("init_vz", "")),
                    str(summary.get("init_wx", "")),
                    str(summary.get("init_wy", "")),
                    str(summary.get("init_wz", "")),
                ])
            if rod_radius is not None:
                extras.append(str(rod_radius))
            if rod_length is not None:
                extras.append(str(rod_length))
            if rod_diameter is not None:
                extras.append(str(rod_diameter))
            if stop_slide_vel_threshold is not None:
                extras.append(str(stop_slide_vel_threshold))
                extras.append(str(stop_slide_vel_min_steps))
            out.write(row + "," + ",".join(extras) + "\n")


if __name__ == "__main__":
    main()
