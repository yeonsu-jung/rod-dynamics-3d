import json
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXECUTABLE = os.path.join(ROOT, "build-debug", "rigidbody_viewer_3d")
SCENE = os.path.join(ROOT, "assets", "scenes", "spinning_rod_soft.json")
PROFILE = os.path.join(ROOT, "spinning_rod_profile.csv")


def run_sim(steps: int = 4000, dt: float = 5e-4) -> None:
    if not os.path.exists(EXECUTABLE):
        raise RuntimeError(f"Executable not found: {EXECUTABLE}. Build the debug target first.")
    if not os.path.exists(SCENE):
        raise RuntimeError(f"Scene file not found: {SCENE}")

    cmd = [
        EXECUTABLE,
        "--headless",
        "--scene", SCENE,
        "--soft-contact",
        "--steps", str(steps),
        "--dt", str(dt),
        "--csv", PROFILE,
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def load_ke(path: str):
    import csv
    frames = []
    ke_total = []
    ke_after_int = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Header from App::logCsvFrame uses: frame,rods,...,KE,KE_after_integrate,...
            frames.append(int(row["frame"]))
            ke_total.append(float(row["KE"]))
            ke_after_int.append(float(row["KE_after_integrate"]))
    return np.array(frames), np.array(ke_total), np.array(ke_after_int)


def main():
    if not os.path.exists(PROFILE):
        run_sim()

    frames, ke_total, ke_after_int = load_ke(PROFILE)

    # Use dt from the run (0.0005 by default). If you change dt in the scene,
    # keep this in sync or infer it from the CSV if needed.
    t = frames * float(5e-4)

    plt.figure(figsize=(6, 4))
    plt.plot(t, ke_total, label="KE_total")
    plt.plot(t, ke_after_int, label="KE_afterIntegrate", alpha=0.7)
    plt.xlabel("time (s)")
    plt.ylabel("Kinetic energy (J)")
    plt.title("Spinning rod soft-contact KE vs time")
    plt.legend()
    plt.tight_layout()
    out_png = os.path.join(ROOT, "spinning_rod_ke.png")
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot to {out_png}")


if __name__ == "__main__":
    main()
