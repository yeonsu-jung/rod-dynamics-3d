#!/usr/bin/env python3

import argparse
import random
import shlex
import subprocess
from pathlib import Path

import numpy as np

from get_random_packings import RelaxedPacking, choose_random_relaxed_packing


REPO_ROOT = Path("/Users/yeonsu/GitHub/rod-dynamics-3d")
BUILD_DIR = REPO_ROOT / "build"
THIS_DIR = Path("/Users/yeonsu/GitHub/rod-dynamics-3d/parametric_study/free_rod_sweep" )
BINARY_PATH = BUILD_DIR / "rigidbody_viewer_3d"
SCENE_PATH = REPO_ROOT / "parametric_study" / "free_rod_sweep" / "default_scene.json"
# INIT_CSV_PATH = REPO_ROOT / "initial-configs" / "relaxation_3rd_multithreading" / "N1000" / "355,359,829" / "x_relaxed_AR200.txt"


TEST_ROD_PATH = THIS_DIR / "test_rod_endpoints.csv"
DEFAULT_LOG_PATH = THIS_DIR / "sweep_free_rod.log"

DEFAULT_STEPS = 20000
DEFAULT_TEST_ROD_STRIDE = 10000
DEFAULT_NSC_MU = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the free-rod simulation.")
    parser.add_argument("--mu", type=float, default=DEFAULT_NSC_MU, help=f"NSC friction coefficient passed as --nsc-mu (default: {DEFAULT_NSC_MU})")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help=f"Headless step count (default: {DEFAULT_STEPS})")
    parser.add_argument(
        "--test-rod-stride",
        type=int,
        default=DEFAULT_TEST_ROD_STRIDE,
        help=f"Test-rod endpoint CSV sampling stride (default: {DEFAULT_TEST_ROD_STRIDE})",
    )
    parser.add_argument(
        "--free-rod-id",
        type=int,
        default=None,
        help="Rod index to leave unfixed for --fix-every-except (default: random valid rod from the chosen packing)",
    )
    parser.set_defaults(headless=True)
    parser.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        help="Run without graphics (default).",
    )
    parser.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="Run with graphics instead of headless mode.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help=f"Write binary stdout/stderr to this file (default: {DEFAULT_LOG_PATH})",
    )
    return parser.parse_args()



def build_command(args: argparse.Namespace, init_csv_path: Path, free_rod_id: int, mu: float) -> list[str]:
    cmd = [
        str(BINARY_PATH),
        "--scene",
        str(SCENE_PATH),
        "--init-csv",
        str(init_csv_path),
        "--test-rod-endpoints",
        str(TEST_ROD_PATH),
        "--test-rod-id",
        str(free_rod_id),
        "--test-rod-endpoints-stride",
        str(args.test_rod_stride),
        "--fix-every-except",
        str(free_rod_id),
        "--nsc-mu",
        str(mu),
    ]
    if args.headless:
        cmd.extend(["--steps", str(args.steps), "--headless"])
    return cmd


def main():
    # choose a set of packing randomly
    # choose two test rods - the ones with MaxFSA, MaxFTA
    # sweep with friction = geomspace(0.01,1,10)
    # multithreading with subprocess

    args = parse_args()
    
    packing = choose_random_relaxed_packing()
    init_csv_path = packing.path
    
    from look_up_extreme_rods import look_up_extreme_rod
    rod_index_MaxFSA, MaxFSA_value = look_up_extreme_rod(packing.n_rods,packing.aspect_ratio,packing.seed_triplet,"MaxFSA")
    rod_index_MaxFTA, MaxFTA_value = look_up_extreme_rod(packing.n_rods,packing.aspect_ratio,packing.seed_triplet,"MaxFTA")
    
    # print(rod_index_MaxFSA)
    # print(rod_index_MaxFTA)    
    # print(init_csv_path)
    
    mu_list = np.geomspace(0.01,1,10)
    
    # start of one cycle for MaxFSA
    
    for mu in mu_list:
    
        free_rod_id = rod_index_MaxFSA
        cmd = build_command(args, init_csv_path, free_rod_id, mu)
        # print(cmd)
        # print(" ".join(shlex.quote(part) for part in cmd))
        # print this cmd to somewhere
        
        log_path = args.log.resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Binary log: {log_path}")
        with log_path.open("w", encoding="utf-8") as log_file:
            subprocess.run(
                cmd,
                cwd=BUILD_DIR,
                check=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        
        # get endpoints trajectory
        ep_traj = np.loadtxt(TEST_ROD_PATH, skiprows=1, delimiter=",", ndmin=2)[:, 3:]    
        centroids = (ep_traj[:,:3] + ep_traj[:,3:])/2

        dumb_path_length = np.linalg.norm(centroids[-1] - centroids[0]) # todo: get proper sliding length
        print(dumb_path_length)
        # this is one cycle

if __name__ == "__main__":
    raise SystemExit(main())
