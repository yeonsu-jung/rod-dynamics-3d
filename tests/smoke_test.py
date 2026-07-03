#!/usr/bin/env python3
"""Smoke test for the rod_dynamics_py module.

Run via ctest (which sets PYTHONPATH to the build directory), or manually:

    PYTHONPATH=build python3 tests/smoke_test.py

Checks that the module loads, a scene steps with finite state, and guards
the contact/config regressions fixed in 2026-07:
  - Hertz-Mindlin contact is repulsive (force sign inversion)
  - quaternion array vs object JSON forms agree (component permutation)
  - sphere-capsule contacts across a periodic boundary are detected and
    repulsive (missing shift_b / PBC, inverted normal)
"""

import json
import math
import sys
import tempfile
from pathlib import Path

import rod_dynamics_py as rd

REPO_ROOT = Path(__file__).resolve().parent.parent

failures = []


def check(name, condition, detail=""):
    status = "ok" if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(name)


def finite3(v):
    return all(math.isfinite(x) for x in v)


def make_sim(tmpdir, name, cfg):
    path = Path(tmpdir) / f"{name}.json"
    path.write_text(json.dumps(cfg))
    return rd.Simulator(str(path))


def separation(sim, i=0, j=1):
    rods = sim.rods()
    a, b = rods[i]["position"], rods[j]["position"]
    return math.dist(a, b)


def test_basic_step(tmpdir):
    """Load a real scene from assets, step it, and require finite state."""
    scene = REPO_ROOT / "assets" / "scenes" / "confined_n2_box0.50.json"
    sim = rd.Simulator(str(scene))
    d0 = sim.diagnostics()
    check("scene loads with bodies", d0["rod_count"] > 0,
          f"rod_count={d0['rod_count']}")

    sim.step(100)
    d = sim.diagnostics()
    check("frame index advances", d["frame_index"] == 100,
          f"frame_index={d['frame_index']}")
    check("KE finite after 100 steps", math.isfinite(d["last_ke"]),
          f"KE={d['last_ke']}")
    for r in sim.rods():
        if not (finite3(r["position"]) and finite3(r["linear_velocity"])):
            check("rod state finite", False, str(r["position"]))
            return
    check("rod state finite", True)


def test_hertz_mindlin_repulsion(tmpdir):
    """Two overlapping spheres must push apart with bounded KE."""
    cfg = {
        "scene": {"bodies": [
            {"pos": [-0.09, 0, 0], "shape": "sphere", "radius": 0.1,
             "density": 2500, "restitution": 0.5, "friction": 0.3},
            {"pos": [0.09, 0, 0], "shape": "sphere", "radius": 0.1,
             "density": 2500, "restitution": 0.5, "friction": 0.3},
        ]},
        "physics": {"dt": 1e-5, "gravity": [0, 0, 0],
                    "hertz_mindlin": {"enabled": True,
                                      "youngs_modulus": 1e7,
                                      "poisson_ratio": 0.25,
                                      "restitution_coeff": 0.5,
                                      "friction_coeff": 0.3,
                                      "enable_tangential": True,
                                      "enable_rolling": False}},
    }
    sim = make_sim(tmpdir, "hm_overlap", cfg)
    d0 = separation(sim)
    sim.step(2000)
    d1 = separation(sim)
    ke = sim.diagnostics()["last_ke"]
    vxa = sim.rods()[0]["linear_velocity"][0]
    vxb = sim.rods()[1]["linear_velocity"][0]
    check("HM overlapping spheres separate", d1 > d0,
          f"{d0:.4f} -> {d1:.4f}")
    check("HM velocities point apart", vxa < 0 < vxb,
          f"vxa={vxa:.4f}, vxb={vxb:.4f}")
    check("HM KE bounded", math.isfinite(ke) and ke < 100.0, f"KE={ke:.4f}")


def test_quaternion_forms(tmpdir):
    """Array [w,x,y,z] and object {w,x,y,z} forms must give the same rotation."""
    c = math.sqrt(0.5)  # 90 degrees about x
    body = {"pos": [0, 0, 0], "shape": "capsule", "radius": 0.05,
            "length": 1.0, "density": 1000}
    cfg = {"scene": {"bodies": [body]},
           "physics": {"dt": 1e-3, "gravity": [0, 0, 0]}}

    body["rot_quat"] = [c, c, 0, 0]
    q_array = make_sim(tmpdir, "q_array", cfg).rods()[0]
    body["rot_quat"] = {"w": c, "x": c, "y": 0, "z": 0}
    q_object = make_sim(tmpdir, "q_object", cfg).rods()[0]

    same = all(abs(a - b) < 1e-6 for a, b in
               zip(q_array["orientation_wxyz"], q_object["orientation_wxyz"]))
    check("quat array/object forms agree", same,
          f"{q_array['orientation_wxyz']} vs {q_object['orientation_wxyz']}")
    # 90 deg about x maps the local Y rod axis onto Z
    ea = q_array["endpoint_a"]
    check("quat rotation applied correctly",
          abs(ea[2] + 0.5) < 1e-4 and abs(ea[1]) < 1e-4,
          f"endpoint_a={ea}")


def test_sphere_capsule_pbc(tmpdir):
    """Sphere-capsule pair overlapping only through a periodic boundary must
    be detected and pushed apart (sphere -x, capsule +x)."""
    def cfg(pbc):
        return {
            "scene": {
                "bodies": [
                    {"pos": [0.45, 0, 0], "shape": "sphere", "radius": 0.1,
                     "density": 1000},
                    {"pos": [-0.47, 0, 0], "shape": "capsule", "radius": 0.05,
                     "length": 0.4, "density": 1000},
                ],
                "periodic": {"enabled": pbc,
                             "min": [-0.5, -0.5, -0.5],
                             "max": [0.5, 0.5, 0.5]},
            },
            "physics": {"dt": 1e-4, "gravity": [0, 0, 0],
                        "soft_contact": {"enabled": True}},
        }

    sim_off = make_sim(tmpdir, "sc_nopbc", cfg(False))
    sim_off.step(50)
    check("no contact without PBC",
          sim_off.diagnostics()["last_hit_count"] == 0)

    sim_on = make_sim(tmpdir, "sc_pbc", cfg(True))
    sim_on.step(50)
    d = sim_on.diagnostics()
    vx_s = sim_on.rods()[0]["linear_velocity"][0]
    vx_c = sim_on.rods()[1]["linear_velocity"][0]
    check("contact detected through PBC", d["last_hit_count"] >= 1,
          f"hits={d['last_hit_count']}")
    check("PBC pair pushed apart", vx_s < 0 < vx_c,
          f"vx_sphere={vx_s:.4f}, vx_capsule={vx_c:.4f}")


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_basic_step(tmpdir)
        test_hertz_mindlin_repulsion(tmpdir)
        test_quaternion_forms(tmpdir)
        test_sphere_capsule_pbc(tmpdir)

    if failures:
        print(f"\n{len(failures)} check(s) failed: {failures}")
        return 1
    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
