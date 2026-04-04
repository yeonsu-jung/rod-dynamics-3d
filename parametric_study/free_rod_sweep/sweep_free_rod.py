#!/usr/bin/env python3

import argparse
import shlex
import subprocess
from pathlib import Path

import numpy as np


REPO_ROOT = Path("/Users/yeonsu/GitHub/rod-dynamics-3d")
BUILD_DIR = REPO_ROOT / "build"
THIS_DIR = Path("/Users/yeonsu/GitHub/rod-dynamics-3d/parametric_study/free_rod_sweep" )
BINARY_PATH = BUILD_DIR / "rigidbody_viewer_3d"
SCENE_PATH = REPO_ROOT / "parametric_study" / "free_rod_sweep" / "default_scene.json"
INIT_CSV_PATH = REPO_ROOT / "initial-configs" / "relaxation_3rd_multithreading" / "N1000" / "355,359,829" / "x_relaxed_AR200.txt"
TEST_ROD_PATH = THIS_DIR / "test_rod_endpoints.csv"
DEFAULT_LOG_PATH = THIS_DIR / "sweep_free_rod.log"

DEFAULT_STEPS = 20000
DEFAULT_TEST_ROD_STRIDE = 10000
DEFAULT_FREE_ROD_ID = 543
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
        default=DEFAULT_FREE_ROD_ID,
        help=f"Rod index to leave unfixed for --fix-every-except (default: {DEFAULT_FREE_ROD_ID})",
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


def build_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        str(BINARY_PATH),
        "--scene",
        str(SCENE_PATH),
        "--init-csv",
        str(INIT_CSV_PATH),
        "--test-rod-endpoints",
        str(TEST_ROD_PATH),
        "--test-rod-id",
        str(args.free_rod_id),
        "--test-rod-endpoints-stride",
        str(args.test_rod_stride),
        "--fix-every-except",
        str(args.free_rod_id),
        "--nsc-mu",
        str(args.mu),
    ]
    if args.headless:
        cmd.extend(["--headless", "--steps", str(args.steps)])
    return cmd


def main() -> int:
    args = parse_args()

    for required_path in (BUILD_DIR, BINARY_PATH, SCENE_PATH, INIT_CSV_PATH):
        if not required_path.exists():
            raise FileNotFoundError(f"Required path does not exist: {required_path}")

    cmd = build_command(args)
    print(" ".join(shlex.quote(part) for part in cmd))
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
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
