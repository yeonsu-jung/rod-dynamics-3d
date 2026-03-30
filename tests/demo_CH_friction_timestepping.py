# =============================================================================
# PROJECT CHRONO - http://projectchrono.org
#
# Time-stepping dynamics with friction contact (kinetic) mini-demo in pure Python.
#
# This script extends the friction contact demo to simulate system dynamics over time:
# - Random initial velocities on 2D disks
# - Euler implicit semi-explicit timestepper (velocity solve + position update)
# - Dynamic friction with Coulomb model (kinetic only)
# - Optional gravity
# - Energy tracking and trajectory visualization
#
# Chrono mapping:
# - ChTimestepperEulerImplicitProjected pattern (velocity problem → position stabilization)
# - NSC contact system with friction cone projection
# =============================================================================

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class Body2D:
    pos: np.ndarray      # shape (2,)
    vel: np.ndarray      # shape (2,)
    radius: float
    mass: float


@dataclass
class Contact:
    a: int
    b: int
    normal: np.ndarray
    phi: float
    mu_k: float


def detect_overlaps(bodies: List[Body2D], mu_k: float = 0.3, overlap_tol: float = 1e-12) -> List[Contact]:
    """Naive O(n^2) collision detection."""
    contacts: List[Contact] = []

    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            pa = bodies[i].pos
            pb = bodies[j].pos
            d = pb - pa
            dist = float(np.linalg.norm(d))
            rsum = bodies[i].radius + bodies[j].radius
            phi = dist - rsum

            if phi < -overlap_tol:
                if dist > 1e-12:
                    normal = d / dist
                else:
                    normal = np.array([1.0, 0.0], dtype=float)

                contacts.append(Contact(a=i, b=j, normal=normal, phi=phi, mu_k=mu_k))

    return contacts


def build_jacobian_normal_row(contact: Contact, nbodies: int) -> np.ndarray:
    """Jacobian row for normal constraint."""
    row = np.zeros(2 * nbodies, dtype=float)
    i0 = 2 * contact.a
    i1 = 2 * contact.b
    row[i0 : i0 + 2] = -contact.normal
    row[i1 : i1 + 2] = +contact.normal
    return row


def build_jacobian_tangent_row(contact: Contact, nbodies: int) -> np.ndarray:
    """Jacobian row for tangential (friction) constraint."""
    tangent = np.array([-contact.normal[1], contact.normal[0]], dtype=float)
    row = np.zeros(2 * nbodies, dtype=float)
    i0 = 2 * contact.a
    i1 = 2 * contact.b
    row[i0 : i0 + 2] = -tangent
    row[i1 : i1 + 2] = +tangent
    return row


def build_inv_mass_vector(bodies: List[Body2D]) -> np.ndarray:
    invm = np.zeros(2 * len(bodies), dtype=float)
    for i, body in enumerate(bodies):
        val = 0.0 if body.mass <= 0.0 else 1.0 / body.mass
        invm[2 * i] = val
        invm[2 * i + 1] = val
    return invm


def flatten_velocities(bodies: List[Body2D]) -> np.ndarray:
    v = np.zeros(2 * len(bodies), dtype=float)
    for i, body in enumerate(bodies):
        v[2 * i : 2 * i + 2] = body.vel
    return v


def scatter_velocities(bodies: List[Body2D], v: np.ndarray) -> None:
    for i, body in enumerate(bodies):
        body.vel = v[2 * i : 2 * i + 2].copy()


def solve_contacts_psor_friction(
    bodies: List[Body2D],
    contacts: List[Contact],
    dt: float,
    beta: float = 0.2,
    omega: float = 1.0,
    cfm: float = 0.0,
    max_iters: int = 40,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Projected SOR solver for contact with dynamic friction.

    CHRONO MAPPING (velocity-level):
    Coupled solve of normal and friction impulses with projections.
    """
    nb = len(bodies)
    nc = len(contacts)

    if nc == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float), flatten_velocities(bodies)

    invm = build_inv_mass_vector(bodies)
    v = flatten_velocities(bodies)
    lambdas_n = np.zeros(nc, dtype=float)
    lambdas_t = np.zeros(nc, dtype=float)

    rows_n = [build_jacobian_normal_row(c, nb) for c in contacts]
    rows_t = [build_jacobian_tangent_row(c, nb) for c in contacts]

    b_n = np.array([beta * c.phi / dt for c in contacts], dtype=float)
    b_t = np.zeros(nc, dtype=float)

    g_n = np.zeros(nc, dtype=float)
    g_t = np.zeros(nc, dtype=float)
    for i in range(nc):
        g_n_i = float(np.dot(rows_n[i] * invm, rows_n[i])) + cfm
        g_n[i] = max(g_n_i, 1e-12)
        g_t_i = float(np.dot(rows_t[i] * invm, rows_t[i])) + cfm
        g_t[i] = max(g_t_i, 1e-12)

    for _ in range(max_iters):
        for i in range(nc):
            # Normal impulse (unilateral)
            w_n = float(np.dot(rows_n[i], v) + b_n[i] + cfm * lambdas_n[i])
            delta_n = -(omega / g_n[i]) * w_n
            old_lambda_n = lambdas_n[i]
            new_lambda_n = max(0.0, old_lambda_n + delta_n)
            true_delta_n = new_lambda_n - old_lambda_n
            lambdas_n[i] = new_lambda_n

            if true_delta_n != 0.0:
                v += (invm * rows_n[i]) * true_delta_n

            # Friction impulse (friction cone)
            w_t = float(np.dot(rows_t[i], v) + b_t[i] + cfm * lambdas_t[i])
            delta_t = -(omega / g_t[i]) * w_t
            old_lambda_t = lambdas_t[i]
            unclamped_lambda_t = old_lambda_t + delta_t
            max_friction = contacts[i].mu_k * max(lambdas_n[i], 0.0)
            new_lambda_t = np.clip(unclamped_lambda_t, -max_friction, max_friction)
            true_delta_t = new_lambda_t - old_lambda_t
            lambdas_t[i] = new_lambda_t

            if true_delta_t != 0.0:
                v += (invm * rows_t[i]) * true_delta_t

    scatter_velocities(bodies, v)
    return lambdas_n, lambdas_t, v


def project_positions(
    bodies: List[Body2D],
    max_iters: int = 10,
    slop: float = 1e-4,
    omega: float = 1.0,
    cfm: float = 0.0,
    psor_iters: int = 100,
) -> int:
    """Position stabilization (normal constraints only)."""
    for it in range(max_iters):
        all_contacts = detect_overlaps(bodies)
        contacts = [
            Contact(c.a, c.b, c.normal, c.phi + slop, c.mu_k)
            for c in all_contacts
            if c.phi < -slop
        ]

        if not contacts:
            return it

        nb = len(bodies)
        nc = len(contacts)
        invm = build_inv_mass_vector(bodies)
        rows = [build_jacobian_normal_row(c, nb) for c in contacts]
        rhs = np.array([c.phi for c in contacts], dtype=float)
        lambdas = np.zeros(nc, dtype=float)
        dpos = np.zeros(2 * nb, dtype=float)

        g = np.zeros(nc, dtype=float)
        for i, row in enumerate(rows):
            g_i = float(np.dot(row * invm, row)) + cfm
            g[i] = max(g_i, 1e-12)

        for _ in range(psor_iters):
            for i in range(nc):
                row = rows[i]
                residual_i = float(np.dot(row, dpos) + rhs[i] + cfm * lambdas[i])

                delta = -(omega / g[i]) * residual_i
                old_lambda = lambdas[i]
                new_lambda = max(0.0, old_lambda + delta)
                true_delta = new_lambda - old_lambda
                lambdas[i] = new_lambda

                if true_delta != 0.0:
                    dpos += (invm * row) * true_delta

        for ibody, body in enumerate(bodies):
            body.pos += dpos[2 * ibody : 2 * ibody + 2]

    return max_iters


def compute_kinetic_energy(bodies: List[Body2D]) -> float:
    """Total kinetic energy."""
    ke = 0.0
    for body in bodies:
        v_sq = float(np.dot(body.vel, body.vel))
        ke += 0.5 * body.mass * v_sq
    return ke


def compute_potential_energy(bodies: List[Body2D], gravity: np.ndarray, ref_height: float) -> float:
    """Total potential energy (relative to fixed reference height)."""
    pe = 0.0
    g_mag = -np.dot(gravity, np.array([0.0, 0.0]))  # magnitude of gravity
    for body in bodies:
        if body.mass > 0.0:
            dy = body.pos[1] - ref_height
            pe += body.mass * g_mag * dy
    return pe


def render_frame(
    bodies: List[Body2D],
    time_val: float,
    ke: float,
    pe: float,
    contacts: int,
    output_file: str,
    domain_bounds: Tuple[float, float, float, float],
    figsize: Tuple[int, int] = (12, 12),
) -> None:
    """Render a frame showing all bodies and save as PNG.
    
    Args:
        domain_bounds: (xmin, xmax, ymin, ymax) fixed domain for rendering
    """
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw bodies
    for body in bodies:
        circle = Circle(body.pos, body.radius, fill=False, edgecolor='blue', linewidth=1)
        ax.add_patch(circle)
    
    # Use fixed domain bounds
    xmin, xmax, ymin, ymax = domain_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Title with simulation info
    title = f"t={time_val:.3f}s | KE={ke:.4f} | PE={pe:.4f} | Contacts={contacts}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=60, bbox_inches='tight')
    plt.close(fig)


def create_video_from_frames(
    frame_dir: str,
    output_video: str,
    fps: int = 30,
) -> bool:
    """Create MP4 video from PNG frames using ffmpeg."""
    try:
        frame_pattern = os.path.join(frame_dir, "frame_%06d.png")
        cmd = [
            "ffmpeg",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-y",  # overwrite output
            output_video,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            return True
        else:
            print(f"FFmpeg error: {result.stderr}")
            return False
    except Exception as e:
        print(f"Video creation failed: {e}")
        return False


def make_random_velocity_bodies(
    n_bodies: int,
    radius: float = 0.35,
    mass: float = 1.0,
    seed: int = 42,
) -> List[Body2D]:
    """Create bodies in a grid with random initial velocities."""
    rng = np.random.default_rng(seed)

    side = int(np.ceil(np.sqrt(n_bodies)))
    nominal = 2.0 * radius
    spacing = nominal * 0.98
    jitter = nominal * 0.02

    bodies: List[Body2D] = []
    for idx in range(n_bodies):
        ix = idx % side
        iy = idx // side
        pos = np.array(
            [
                ix * spacing + rng.uniform(-jitter, jitter),
                iy * spacing + rng.uniform(-jitter, jitter),
            ],
            dtype=float,
        )
        # Random velocity
        vel = rng.uniform(-1.0, 1.0, 2).astype(float)
        bodies.append(Body2D(pos=pos, vel=vel, radius=radius, mass=mass))

    pts = np.array([b.pos for b in bodies])
    centroid = np.mean(pts, axis=0)
    for b in bodies:
        b.pos = b.pos - centroid

    return bodies


def simulate(
    bodies: List[Body2D],
    name: str,
    dt: float = 1e-2,
    num_steps: int = 100,
    gravity: np.ndarray = np.array([0.0, 0.0], dtype=float),
    mu_k: float = 0.3,
    damping: float = 0.0,
    render_video: bool = False,
    video_dir: str = "/tmp/chrono_video",
    video_fps: int = 30,
) -> None:
    """
    Time-stepping simulation with friction contact.

    CHRONO MAPPING (semi-explicit Euler):
    For each step:
      1. v_new = v_old + dt * (f_ext / M)              [external forces]
      2. Solve contact problem (velocity level)        [contact forces]
      3. dpos = v_new * dt                             [position update]
      4. Position projection (normal constraints only) [constraint stabilization]
    """
    print(f"\n=== {name} ===")
    print(f"Bodies: {len(bodies)}")
    print(f"Time steps: {num_steps}, dt = {dt} s")
    print(f"Friction coefficient: {mu_k}")
    print(f"Gravity: {gravity}")
    print(f"Damping: {damping}")
    if render_video:
        print(f"Video rendering: enabled ({video_fps} fps)")

    # Create video directory if rendering
    if render_video and HAS_MATPLOTLIB:
        os.makedirs(video_dir, exist_ok=True)
        for f in os.listdir(video_dir):
            os.remove(os.path.join(video_dir, f))
    elif render_video and not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping video rendering")
        render_video = False

    # Calculate domain bounds from initial configuration (fixed for all frames)
    domain_bounds = None
    if render_video and HAS_MATPLOTLIB:
        positions = np.array([b.pos for b in bodies])
        radii = np.array([b.radius for b in bodies])
        margin = 2.0  # extra margin around initial configuration
        xmin = positions[:, 0].min() - radii.max() - margin
        xmax = positions[:, 0].max() + radii.max() + margin
        ymin = positions[:, 1].min() - radii.max() - margin
        ymax = positions[:, 1].max() + radii.max() + margin
        domain_bounds = (xmin, xmax, ymin, ymax)

    # Initial energy (fixed reference height for consistency)
    ref_height = bodies[0].pos[1] if len(bodies) > 0 else 0.0
    ke0 = compute_kinetic_energy(bodies)
    pe0 = compute_potential_energy(bodies, gravity, ref_height)
    e0 = ke0 + pe0

    print(f"Initial energy: KE={ke0:.6f}, PE={pe0:.6f}, Total={e0:.6f}")

    energy_history = [e0]
    contact_history = []
    time_history = [0.0]

    t_start = time.perf_counter()
    
    # Determine frame skip to match video fps
    frame_skip = max(1, int(num_steps / (num_steps * dt * video_fps)))
    frame_count = 0

    for step in range(num_steps):
        current_time = step * dt

        # Detect contacts
        contacts = detect_overlaps(bodies, mu_k=mu_k, overlap_tol=1e-12)
        contact_history.append(len(contacts))

        # Semi-explicit Euler: apply external forces
        g_accel = gravity
        for body in bodies:
            if body.mass > 0.0:
                body.vel += g_accel * dt

        # Apply damping
        if damping > 0.0:
            for body in bodies:
                body.vel *= (1.0 - damping * dt)

        # CHRONO MAPPING (velocity level):
        # Solve contact impulses with friction
        lambdas_n, lambdas_t, _ = solve_contacts_psor_friction(
            bodies,
            contacts,
            dt=dt,
            beta=0.2,
            omega=1.0,
            cfm=0.0,
            max_iters=20,
        )

        # Update positions
        for body in bodies:
            body.pos += body.vel * dt

        # CHRONO MAPPING (position level):
        # Position projection for constraint stabilization
        proj_iters = project_positions(
            bodies,
            max_iters=5,
            slop=1e-4,
            omega=1.0,
            cfm=0.0,
            psor_iters=50,
        )

        # Energy tracking
        ke = compute_kinetic_energy(bodies)
        pe = compute_potential_energy(bodies, gravity, ref_height)
        e = ke + pe
        energy_history.append(e)
        time_history.append(current_time + dt)

        # Render frame if requested
        if render_video and step % frame_skip == 0:
            frame_path = os.path.join(video_dir, f"frame_{frame_count:06d}.png")
            render_frame(bodies, current_time + dt, ke, pe, len(contacts), frame_path, domain_bounds)
            frame_count += 1

        if (step + 1) % max(1, num_steps // 10) == 0 or step == 0:
            e_factor = (e - e0) / max(abs(e0), 1e-10)
            print(
                f"  Step {step+1:4d}: t={current_time+dt:.4f}s, "
                f"contacts={len(contacts):3d}, "
                f"KE={ke:.6f}, PE={pe:.6f}, E={e:.6f}, "
                f"ΔE/E₀={e_factor:+.3%}"
            )

    t_end = time.perf_counter()

    print(f"\nSimulation completed in {(t_end - t_start) * 1000.0:.2f} ms")
    print(f"Final energy: KE={ke:.6f}, PE={pe:.6f}, Total={e:.6f}")
    print(f"Energy change: {energy_history[-1] - energy_history[0]:+.6f} ({(energy_history[-1] - energy_history[0]) / max(abs(energy_history[0]), 1e-10):+.3%})")
    print(f"Max contacts in single step: {max(contact_history)}")
    print(f"Avg contacts per step: {np.mean(contact_history):.1f}")

    # Create video if rendering
    if render_video and HAS_MATPLOTLIB:
        print("\nCreating video from frames...")
        video_output = os.path.join(os.path.dirname(video_dir), f"{name.lower().replace(' ', '_')}_dynamics.mp4")
        if create_video_from_frames(video_dir, video_output, fps=video_fps):
            print(f"✓ Video saved to: {video_output}")
        else:
            print("✗ Video creation failed. Make sure ffmpeg is installed.")

    # Plot energy evolution
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
        fig.suptitle(f"{name} - Energy Evolution")

        # Energy vs time
        ax = axes[0]
        ax.plot(time_history, energy_history, "b-", linewidth=2, label="Total Energy")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Energy (J)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title("Total Energy Over Time")

        # Contact count vs time
        ax = axes[1]
        ax.plot(time_history[:-1], contact_history, "r-", linewidth=1, label="Number of Contacts")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Contact Count")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title("Active Contacts Over Time")

        plt.show()
    except Exception as e:
        print(f"\nMatplotlib visualization unavailable: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time-stepping dynamics with friction contact."
    )
    parser.add_argument(
        "--n-bodies",
        type=int,
        default=9,
        help="Number of bodies (will be arranged in a grid).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=200,
        help="Number of time steps.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=5e-3,
        help="Time step size (seconds).",
    )
    parser.add_argument(
        "--mu-k",
        type=float,
        default=0.3,
        help="Kinetic friction coefficient.",
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=0.0,
        help="Magnitude of gravity (m/s²).",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.0,
        help="Velocity damping coefficient (0-1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initial conditions.",
    )
    parser.add_argument(
        "--render-video",
        action="store_true",
        help="Render video of simulation dynamics.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=30,
        help="Frame rate for video output.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("CHRONO-LIKE TIME-STEPPING WITH FRICTION CONTACT")
    print("=" * 70)

    bodies = make_random_velocity_bodies(
        n_bodies=args.n_bodies,
        radius=0.3,
        mass=1.0,
        seed=args.seed,
    )

    gravity_vec = np.array([0.0, -args.gravity], dtype=float)

    simulate(
        bodies,
        name=f"{args.n_bodies}-Body Dynamics",
        dt=args.dt,
        num_steps=args.n_steps,
        gravity=gravity_vec,
        mu_k=args.mu_k,
        damping=args.damping,
        render_video=args.render_video,
        video_fps=args.video_fps,
    )


if __name__ == "__main__":
    main()
