import math
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Disk:
    """2D disk with position, velocity, mass, and radius."""
    x: np.ndarray  # position (2,)
    v: np.ndarray  # velocity (2,)
    m: float
    r: float


@dataclass
class Contact:
    """Contact data for two disks: indices, normal, and penetration."""
    i: int
    j: int
    n: np.ndarray  # normal from i -> j (2,)
    penetration: float


def detect_pair_contact(i: int, j: int, disks: List[Disk]) -> Contact | None:
    """Detect normal contact between two disks i, j. Returns Contact or None."""
    d1, d2 = disks[i], disks[j]
    dx = d2.x - d1.x
    dist = float(np.linalg.norm(dx))
    if dist == 0.0:
        n = np.array([1.0, 0.0])
    else:
        n = dx / dist

    h = d1.r + d2.r
    penetration = max(0.0, h - dist)
    if penetration <= 0.0:
        return None
    return Contact(i=i, j=j, n=n, penetration=penetration)


def solve_normal_impulse_min_ke(d1: Disk, d2: Disk, n: np.ndarray, restitution: float = 0.0) -> float:
    """
    Solve for a single normal impulse j_n that minimizes post-step KE,
    subject to j_n >= 0 (no pulling).

    This is a tiny QP:
       minimize 1/2 * (v^+ - v^-)' M (v^+ - v^-)
       subject to j_n >= 0,
    with v^+ = v^- + M^{-1} * J^T * j_n

    For two disks in 2D with one normal constraint, we can do it analytically.
    """
    m1, m2 = d1.m, d2.m

    # Relative normal velocity before impulse
    v_rel = d2.v - d1.v
    v_n = float(np.dot(v_rel, n))

    # Effective mass along the normal direction:
    # M_eff^-1 = n^T (M^-1_1 + M^-1_2) n = 1/m1 + 1/m2  (scalar in this simple case)
    inv_meff = 1.0 / m1 + 1.0 / m2

    # Target post-impact normal velocity v_n+ = -e v_n
    # Solve: v_n+ = v_n + j_n * inv_meff  =>  j_n* = (v_n+ - v_n) / inv_meff
    v_n_target = -restitution * v_n
    j_star = (v_n_target - v_n) / inv_meff

    # Constraint: j_n >= 0 (no tensile force)
    j_n = max(0.0, j_star)

    return j_n


def apply_normal_impulse(d1: Disk, d2: Disk, j_n: float, n: np.ndarray):
    """
    Apply a normal impulse j_n * n to the pair (equal and opposite).
    """
    if j_n == 0.0:
        return

    # Impulse on disk 2 along +n, on disk 1 along -n
    impulse = j_n * n  # on disk 2
    d2.v += impulse / d2.m
    d1.v -= impulse / d1.m


def apply_friction_impulse(d1: Disk, d2: Disk, contact: Contact, mu: float):
    """Apply tangential friction impulse at a contact, capped by mu * j_n.

    This assumes normal impulse has already been applied and updates only the
    tangential relative velocity using a simple Coulomb model.
    """
    if mu <= 0.0:
        return

    n = contact.n
    # Relative velocity after normal impulse
    v_rel = d2.v - d1.v
    v_n = float(np.dot(v_rel, n))
    v_t_vec = v_rel - v_n * n
    v_t = float(np.linalg.norm(v_t_vec))
    if v_t == 0.0:
        return

    # Effective mass in tangential direction is same scalar as normal here
    m1, m2 = d1.m, d2.m
    inv_meff_t = 1.0 / m1 + 1.0 / m2

    # We need the normal impulse magnitude that was applied. For this simple
    # example we approximate it from the change in v_n over this step:
    #   j_n ~= (v_n_before - v_n_after) / inv_meff
    # but since we don't store v_n_before here, we'll instead compute a
    # friction impulse that tries to zero tangential velocity and then clamp.

    # Unconstrained friction impulse to kill tangential velocity:
    j_t_star = -v_t / inv_meff_t
    t_dir = v_t_vec / v_t

    # Approximate max friction from a guessed normal impulse scale: this is
    # purely for illustration, not exact MuJoCo.
    # In a more precise solver you would track j_n per contact.
    j_n_scale = abs(j_t_star)  # local scale proxy
    j_t_max = mu * j_n_scale

    j_t = j_t_star
    if abs(j_t) > j_t_max:
        j_t = math.copysign(j_t_max, j_t_star)

    impulse_t = j_t * t_dir
    d2.v += impulse_t / d2.m
    d1.v -= impulse_t / d1.m


def step_disks_qp(disks: List[Disk], dt: float, g: np.ndarray,
                  mu: float = 0.0, restitution: float = 0.0,
                  gs_iterations: int = 1) -> float:
    """One simulation step for a list of disks using a simple contact QP.

    - Applies external acceleration g.
    - Detects all pairwise contacts.
    - Runs a Gauss–Seidel loop over contacts:
        * normal projection (with restitution)
        * optional friction projection
    - Integrates positions.
    Returns total KE after the step.
    """
    # 1) External acceleration
    for d in disks:
        d.v += dt * g

    # 2) Detect contacts based on current positions
    contacts: List[Contact] = []
    n_disks = len(disks)
    for i in range(n_disks):
        for j in range(i + 1, n_disks):
            c = detect_pair_contact(i, j, disks)
            if c is not None:
                contacts.append(c)

    # 3) Gauss–Seidel over contacts for impulses
    for _ in range(gs_iterations):
        for c in contacts:
            d1, d2 = disks[c.i], disks[c.j]
            # normal projection
            j_n = solve_normal_impulse_min_ke(d1, d2, c.n, restitution=restitution)
            apply_normal_impulse(d1, d2, j_n, c.n)
            # friction projection (very simplified)
            if mu > 0.0:
                apply_friction_impulse(d1, d2, c, mu)

    # 4) Integrate positions
    for d in disks:
        d.x += dt * d.v

    # 5) KE diagnostic
    ke = 0.0
    for d in disks:
        ke += 0.5 * d.m * float(np.dot(d.v, d.v))
    return ke


def run_demo():
    # Three disks in 2D for a simple multi-contact + friction example
    disks = [
        Disk(x=np.array([-0.5, 0.0]), v=np.array([2.0, 0.5]), m=1.0, r=0.1),
        Disk(x=np.array([+0.5, 0.0]), v=np.array([-1.5, 0.0]), m=1.0, r=0.1),
        Disk(x=np.array([0.0, 0.4]), v=np.array([0.0, -1.0]), m=1.0, r=0.1),
    ]

    dt = 1e-3
    steps = 3000
    g = np.array([0.0, 0.0])  # no gravity for this demo
    mu = 0.
    restitution = 1.0

    t_hist = []
    ke_hist = []
    x_hist = [[] for _ in disks]
    v_hist = [[] for _ in disks]

    for k in range(steps):
        t = k * dt
        ke = step_disks_qp(disks, dt, g, mu=mu, restitution=restitution,
                           gs_iterations=4)

        t_hist.append(t)
        ke_hist.append(ke)
        for i, d in enumerate(disks):
            x_hist[i].append(float(d.x[0]))
            v_hist[i].append(float(d.v[0]))

    t_hist = np.array(t_hist)

    # Plot positions and velocities for disk 0 (as a representative)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))

    for i in range(len(disks)):
        axs[0].plot(t_hist, x_hist[i], label=f"x{i}")
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("position x")
    axs[0].legend()
    axs[0].set_title("Positions (x)")

    for i in range(len(disks)):
        axs[1].plot(t_hist, v_hist[i], label=f"v{i}")
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("velocity vx")
    axs[1].legend()
    axs[1].set_title("Velocities (vx)")

    axs[2].plot(t_hist, ke_hist, label="KE")
    axs[2].set_xlabel("time")
    axs[2].set_ylabel("kinetic energy")
    axs[2].set_title("Total KE")
    axs[2].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_demo()