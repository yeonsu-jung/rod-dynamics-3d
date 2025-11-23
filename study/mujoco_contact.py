import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Tuple as Tup

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


@dataclass
class Disk:
    """2D disk with position, velocity, mass, and radius."""
    x: np.ndarray  # position (2,)
    v: np.ndarray  # velocity (2,)
    m: float
    r: float


@dataclass
class Box:
    """Periodic simulation box in 2D.

    Lx, Ly are box lengths along x and y. Coordinates are wrapped into
    [0, L) in each dimension.
    """
    Lx: float
    Ly: float


@dataclass
class Contact:
    """Contact data for two disks: indices, normal, and penetration."""
    i: int
    j: int
    n: np.ndarray  # normal from i -> j (2,)
    penetration: float


def minimum_image(dx: np.ndarray, box: Box) -> np.ndarray:
    """Apply minimum-image convention for displacement dx under periodic BCs."""
    L = np.array([box.Lx, box.Ly])
    # Shift dx into [-L/2, L/2) in each dimension
    return dx - L * np.round(dx / L)


def wrap_position(x: np.ndarray, box: Box) -> np.ndarray:
    """Wrap a 2D position into [0, L) x [0, L) for the periodic box."""
    x_wrapped = np.empty_like(x)
    x_wrapped[0] = x[0] % box.Lx
    x_wrapped[1] = x[1] % box.Ly
    return x_wrapped


def detect_pair_contact(i: int, j: int, disks: List[Disk], box: Box | None = None) -> Contact | None:
    """Detect normal contact between two disks i, j. Returns Contact or None.

    If box is provided, uses minimum-image separation under periodic BCs.
    """
    d1, d2 = disks[i], disks[j]
    dx = d2.x - d1.x
    if box is not None:
        dx = minimum_image(dx, box)
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


def build_spatial_hash(disks: List[Disk], box: Box, cell_size: float) -> Dict[Tup[int, int], List[int]]:
    """Build a simple spatial hash (uniform grid) for disks in a periodic box.

    Returns a dict mapping integer cell coordinates to a list of disk indices.
    """
    grid: Dict[Tup[int, int], List[int]] = {}
    nx = int(math.floor(box.Lx / cell_size))
    ny = int(math.floor(box.Ly / cell_size))
    nx = max(nx, 1)
    ny = max(ny, 1)

    for idx, d in enumerate(disks):
        # cell indices with periodic wrap
        cx = int(math.floor(d.x[0] / cell_size)) % nx
        cy = int(math.floor(d.x[1] / cell_size)) % ny
        key = (cx, cy)
        grid.setdefault(key, []).append(idx)

    return grid


def gather_nearby_pairs(grid: Dict[Tup[int, int], List[int]], box: Box,
                        cell_size: float) -> List[Tup[int, int]]:
    """Collect potentially colliding pairs from a spatial hash.

    Checks each cell and its neighbors (3x3 stencil) under periodic wrap.
    """
    pairs: List[Tup[int, int]] = []
    if not grid:
        return pairs

    # Deduce grid resolution from keys
    xs = [k[0] for k in grid.keys()]
    ys = [k[1] for k in grid.keys()]
    nx = max(xs) + 1
    ny = max(ys) + 1

    seen = set()
    neighbors = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),  (0, 0),  (1, 0),
        (-1, 1),  (0, 1),  (1, 1),
    ]

    for (cx, cy), indices in grid.items():
        for dx, dy in neighbors:
            nx_cell = (cx + dx) % nx
            ny_cell = (cy + dy) % ny
            nbr = grid.get((nx_cell, ny_cell))
            if not nbr:
                continue
            for i in indices:
                for j in nbr:
                    if i >= j:
                        continue
                    key = (i, j)
                    if key in seen:
                        continue
                    seen.add(key)
                    pairs.append(key)

    return pairs


def step_disks_qp(disks: List[Disk], dt: float, g: np.ndarray,
                  mu: float = 0.0, restitution: float = 0.0,
                  gs_iterations: int = 1,
                  box: Box | None = None,
                  broadphase_cell_size: float | None = None) -> float:
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

    # 2) Build contact list using either naive O(N^2) or spatial hash
    contacts: List[Contact] = []
    n_disks = len(disks)

    if broadphase_cell_size is not None and box is not None:
        # spatial hash broadphase
        grid = build_spatial_hash(disks, box, broadphase_cell_size)
        pairs = gather_nearby_pairs(grid, box, broadphase_cell_size)
        for i, j in pairs:
            c = detect_pair_contact(i, j, disks, box=box)
            if c is not None:
                contacts.append(c)
    else:
        # fallback: all pairs
        for i in range(n_disks):
            for j in range(i + 1, n_disks):
                c = detect_pair_contact(i, j, disks, box=box)
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

    # 4) Integrate positions and wrap into box if given
    for d in disks:
        d.x += dt * d.v
        if box is not None:
            d.x = wrap_position(d.x, box)

    # 5) KE diagnostic
    ke = 0.0
    for d in disks:
        ke += 0.5 * d.m * float(np.dot(d.v, d.v))
    return ke


def run_demo():
    # Periodic box settings
    box = Box(Lx=4.0, Ly=4.0)

    # Many-disk initialization in a periodic box
    rng = np.random.default_rng(0)
    n_disks = 600
    radius = 0.05
    mass = 1.0

    disks: List[Disk] = []
    for _ in range(n_disks):
        x = np.array([
            rng.uniform(0.0, box.Lx),
            rng.uniform(0.0, box.Ly),
        ])
        # zero-mean random velocities
        v = rng.normal(0.0, 1.0, size=2)
        disks.append(Disk(x=x, v=v, m=mass, r=radius))

    dt = 5e-3
    steps = 2000
    g = np.array([0.0, 0.0])  # no gravity for this demo
    mu = 0.0
    restitution = 1.0

    # Broadphase cell size around 2 radii (a bit larger than contact distance)
    cell_size = 2.5 * radius

    # Pre-run simulation to store history for plotting and animation
    t_hist = []
    ke_hist = []

    # For visualization, track only a subset of disks to keep memory small
    n_track = min(10, n_disks)
    track_indices = list(range(n_track))
    x_hist = [[] for _ in track_indices]
    y_hist = [[] for _ in track_indices]
    v_hist = [[] for _ in track_indices]

    for k in range(steps):
        t = k * dt
        ke = step_disks_qp(
            disks,
            dt,
            g,
            mu=mu,
            restitution=restitution,
            gs_iterations=4,
            box=box,
            broadphase_cell_size=cell_size,
        )

        t_hist.append(t)
        ke_hist.append(ke)
        for local_i, idx in enumerate(track_indices):
            d = disks[idx]
            x_hist[local_i].append(float(d.x[0]))
            y_hist[local_i].append(float(d.x[1]))
            v_hist[local_i].append(float(d.v[0]))

    t_hist = np.array(t_hist)

    # Static diagnostic plots (positions x, velocities vx, KE) for tracked disks
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))

    for i in range(len(track_indices)):
        axs[0].plot(t_hist, x_hist[i], label=f"x{track_indices[i]}")
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("position x")
    axs[0].legend()
    axs[0].set_title("Positions (x)")

    for i in range(len(track_indices)):
        axs[1].plot(t_hist, v_hist[i], label=f"v{track_indices[i]}")
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

    # 2D animation of disks as moving circles
    fig_anim, ax_anim = plt.subplots(figsize=(4, 4))
    ax_anim.set_aspect("equal", adjustable="box")

    # Determine bounds from trajectories with some padding
    all_x = np.concatenate([np.array(xs) for xs in x_hist])
    all_y = np.concatenate([np.array(ys) for ys in y_hist])
    pad = 0.2
    xmin, xmax = 0.0, box.Lx
    ymin, ymax = 0.0, box.Ly
    ax_anim.set_xlim(xmin, xmax)
    ax_anim.set_ylim(ymin, ymax)
    ax_anim.set_title("Disk motion")

    # Create circle patches for each tracked disk
    circles = []
    for i, idx in enumerate(track_indices):
        d = disks[idx]
        circ = plt.Circle((x_hist[i][0], y_hist[i][0]), d.r, fill=False)
        ax_anim.add_patch(circ)
        circles.append(circ)

    time_text = ax_anim.text(0.02, 0.95, "", transform=ax_anim.transAxes)

    def init():
        for i, idx in enumerate(track_indices):
            circles[i].center = (x_hist[i][0], y_hist[i][0])
        time_text.set_text("")
        return circles + [time_text]

    def update(frame):
        for i, idx in enumerate(track_indices):
            circles[i].center = (x_hist[i][frame], y_hist[i][frame])
        time_text.set_text(f"t = {t_hist[frame]:.3f}")
        return circles + [time_text]

    anim = animation.FuncAnimation(
        fig_anim,
        update,
        init_func=init,
        frames=steps,
        interval=1000 * dt,
        blit=True,
    )

    plt.show()


if __name__ == "__main__":
    run_demo()