"""Utility script to load per-rod position CSV, build an (F, N, 3) array,
compute periodic center of mass (COM), and visualize trajectories.

Features:
    - Efficient reshape without manual loops (optional pivot path).
    - Periodic center-of-mass computation for each frame in a cubic box.
    - Static 3D scatter of all positions or final frame, with COM path overlay.
    - Optional lightweight animation of motion (Matplotlib FuncAnimation) including moving COM marker + trail.
    - Export COM time series to CSV/NPZ if requested.
    - CLI arguments for path, box size, stride, frame limit and animation export.

Usage examples:
  python read_positions.py --csv parametric_study/runs/.../confined_n20_hard_mu0_10_noise_f1e-05_t1e-05_seed1.csv --box-size 0.9 --show
  python read_positions.py --csv run.csv --animate --save-animation path.mp4 --stride 5 --frame-limit 1000

Notes:
  - The CSV is expected to contain columns: frame, rod, px, py, pz.
  - Periodic domain assumed centered at origin with side length = box_size.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from matplotlib import pyplot as plt
from matplotlib import animation
try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:  # Should not occur if plotly installed, but keep safe
    _PLOTLY_AVAILABLE = False

def resolve_csv_path(csv_path: Path) -> Path:
    """Attempt to resolve a CSV path by searching common locations.

    Priority:
      1. Given path as-is.
      2. Same directory as this script.
      3. 'runs' subdirectory next to this script.
      4. parametric_study/runs under current working directory.
    Raises FileNotFoundError with attempted paths listed if not found.
    """
    attempted = []
    if csv_path.exists():
        return csv_path
    script_dir = Path(__file__).parent
    candidates = [
        script_dir / csv_path.name,
        script_dir / 'runs' / csv_path.name,
        Path.cwd() / 'parametric_study' / 'runs' / csv_path.name,
    ]
    for c in candidates:
        attempted.append(str(c))
        if c.exists():
            print(f"[resolve_csv_path] Resolved '{csv_path}' -> '{c}'")
            return c
    raise FileNotFoundError(f"Could not locate CSV '{csv_path}'. Tried: " + ", ".join(attempted))

def load_positions_csv(csv_path: Path, stride: int = 1, frame_limit: int | None = None) -> pd.DataFrame:
    """Load position CSV with optional frame subsampling.

    Parameters
    ----------
    csv_path : Path
        Path to per-rod CSV.
    stride : int
        Keep every k-th frame (after sorting unique frames).
    frame_limit : int | None
        Limit to first frame_limit frames after stride filtering.
    """
    real_path = resolve_csv_path(csv_path)
    df = pd.read_csv(real_path, usecols=["frame", "rod", "px", "py", "pz"])
    # Apply stride by filtering frames list then selecting subset of rows.
    if stride > 1 or frame_limit is not None:
        frames_sorted = np.sort(df["frame"].unique())
        frames_sorted = frames_sorted[::stride]
        if frame_limit is not None:
            frames_sorted = frames_sorted[:frame_limit]
        df = df[df["frame"].isin(frames_sorted)].copy()
    return df

def build_position_array(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert long-form dataframe to (F, N, 3) array.

    Returns
    -------
    positions : ndarray[F, N, 3]
    frames    : ndarray[F]
    rods      : ndarray[N]
    """
    frames = np.sort(df["frame"].unique())
    rods = np.sort(df["rod"].unique())
    F, N = len(frames), len(rods)
    frame_index = {f: i for i, f in enumerate(frames)}
    rod_index = {r: i for i, r in enumerate(rods)}
    arr = np.zeros((F, N, 3), dtype=np.float64)
    for row in df.itertuples(index=False):
        fi = frame_index[row.frame]
        ri = rod_index[row.rod]
        arr[fi, ri, 0] = row.px
        arr[fi, ri, 1] = row.py
        arr[fi, ri, 2] = row.pz
    return arr, frames, rods

def periodic_center_of_mass(positions: np.ndarray, box_size: float) -> np.ndarray:
    """Compute periodic COM for each frame under cubic box [-L/2, L/2]^3.

    Uses complex exponentials to avoid discontinuities near boundaries.
    positions: (F, N, 3)
    Returns: (F, 3) COM positions wrapped back into [-L/2, L/2].
    """
    L = box_size
    half = L / 2.0
    F = positions.shape[0]
    com = np.zeros((F, 3), dtype=np.float64)
    for i in range(F):
        for dim in range(3):
            # Shift to [0, L) then map to unit circle.
            shifted = (positions[i, :, dim] + half) % L
            angles = 2.0 * np.pi * shifted / L
            # Mean on unit circle.
            mean_vec = np.exp(1j * angles).mean()
            # Angle of mean vector back to [0, 2pi)
            mean_angle = np.angle(mean_vec) % (2.0 * np.pi)
            # Convert back to length coordinate in [0, L) then shift to [-L/2, L/2]
            coord = (mean_angle / (2.0 * np.pi)) * L - half
            com[i, dim] = coord
    return com

def plot_static(positions: np.ndarray, com: np.ndarray | None, frames: np.ndarray, show_all: bool = True) -> None:
    """Static 3D scatter: either all points or last frame; overlay COM path."""
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    if show_all:
        ax.scatter(positions[:, :, 0].ravel(), positions[:, :, 1].ravel(), positions[:, :, 2].ravel(), s=2, alpha=0.35)
    else:
        ax.scatter(positions[-1, :, 0], positions[-1, :, 1], positions[-1, :, 2], s=12, color='tab:blue')
    if com is not None:
        ax.plot(com[:, 0], com[:, 1], com[:, 2], color='black', linewidth=1.2, label='Periodic COM')
        ax.scatter(com[0, 0], com[0, 1], com[0, 2], color='red', s=25, label='COM start')
    ax.set_title(f"Trajectory Points (frames={len(frames)})")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if com is not None:
        ax.legend(loc='upper left')
    plt.tight_layout()

def animate_positions(positions: np.ndarray, com: np.ndarray | None, interval: int = 40, tail: int = 0) -> animation.FuncAnimation:
    """Create a simple animation of rod positions and COM.

    tail: number of previous frames to retain as faded trail (0 for none).
    """
    F, N, _ = positions.shape
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter([], [], [], s=12, color='tab:blue')
    com_line = None
    com_marker = None
    if com is not None:
        com_line, = ax.plot([], [], [], color='black', linewidth=1.2, label='COM trail')
        com_marker = ax.scatter([], [], [], color='red', s=30, label='COM current')
    ax.set_xlim(np.min(positions[:, :, 0]), np.max(positions[:, :, 0]))
    ax.set_ylim(np.min(positions[:, :, 1]), np.max(positions[:, :, 1]))
    ax.set_zlim(np.min(positions[:, :, 2]), np.max(positions[:, :, 2]))
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title('Rod Positions Animation')

    def update(frame: int):
        pts = positions[frame]
        scat._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        artists = [scat]
        if com is not None and com_line is not None and com_marker is not None:
            span_start = max(0, frame - tail)
            com_line.set_data(com[span_start:frame+1, 0], com[span_start:frame+1, 1])
            com_line.set_3d_properties(com[span_start:frame+1, 2])
            # update current marker
            com_marker._offsets3d = (np.array([com[frame, 0]]), np.array([com[frame, 1]]), np.array([com[frame, 2]]))
            artists.extend([com_line, com_marker])
        return artists

    anim = animation.FuncAnimation(fig, update, frames=F, interval=interval, blit=False)
    return anim

def plot_plotly_static(positions: np.ndarray, com: np.ndarray | None, frames: np.ndarray, show_all: bool, html_out: Path | None):
    """Create an interactive Plotly 3D scatter and optionally save to HTML.

    If show_all True, plots all points with low opacity; else only final frame.
    COM path (if provided) drawn as a line; start and end marked.
    """
    if not _PLOTLY_AVAILABLE:
        raise RuntimeError("Plotly not available; install plotly first.")
    if show_all:
        xs = positions[:, :, 0].ravel()
        ys = positions[:, :, 1].ravel()
        zs = positions[:, :, 2].ravel()
        scatter = go.Scatter3d(x=xs, y=ys, z=zs,
                               mode='markers',
                               marker=dict(size=2, opacity=0.35, color='royalblue'),
                               name='All points')
        data = [scatter]
    else:
        last = positions[-1]
        scatter = go.Scatter3d(x=last[:, 0], y=last[:, 1], z=last[:, 2],
                               mode='markers', marker=dict(size=5, color='royalblue'),
                               name=f'Frame {frames[-1]}')
        data = [scatter]
    if com is not None:
        com_line = go.Scatter3d(x=com[:, 0], y=com[:, 1], z=com[:, 2],
                                mode='lines', line=dict(width=4, color='black'),
                                name='COM path')
        com_start = go.Scatter3d(x=[com[0,0]], y=[com[0,1]], z=[com[0,2]],
                                 mode='markers', marker=dict(size=5, color='red'), name='COM start')
        com_end = go.Scatter3d(x=[com[-1,0]], y=[com[-1,1]], z=[com[-1,2]],
                               mode='markers', marker=dict(size=5, color='green'), name='COM end')
        data.extend([com_line, com_start, com_end])
    fig = go.Figure(data=data)
    fig.update_layout(title=f"Rod Trajectories (frames={len(frames)})", scene=dict(aspectmode='data'))
    if html_out:
        fig.write_html(str(html_out))
        print(f"Wrote Plotly HTML: {html_out}")
    else:
        fig.show()

def plot_plotly_animated(positions: np.ndarray, com: np.ndarray | None, frames: np.ndarray, html_out: Path | None, sample: int):
    """Animate positions over frames using Plotly frames API.

    sample: keep every k-th frame for animation (to control size)."""
    if not _PLOTLY_AVAILABLE:
        raise RuntimeError("Plotly not available; install plotly first.")
    F = positions.shape[0]
    frame_indices = list(range(0, F, sample))
    # Base (first frame)
    base_pts = positions[frame_indices[0]]
    scat = go.Scatter3d(x=base_pts[:,0], y=base_pts[:,1], z=base_pts[:,2],
                        mode='markers', marker=dict(size=4, color='royalblue'), name='Rods')
    data = [scat]
    if com is not None:
        com_marker = go.Scatter3d(x=[com[frame_indices[0],0]], y=[com[frame_indices[0],1]], z=[com[frame_indices[0],2]],
                                  mode='markers', marker=dict(size=6, color='red'), name='COM')
        data.append(com_marker)
    frames_plotly = []
    for fi in frame_indices:
        pts = positions[fi]
        frame_data = [go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2],
                                   mode='markers', marker=dict(size=4, color='royalblue'))]
        if com is not None:
            frame_data.append(go.Scatter3d(x=[com[fi,0]], y=[com[fi,1]], z=[com[fi,2]],
                                           mode='markers', marker=dict(size=6, color='red')))
        frames_plotly.append(go.Frame(data=frame_data, name=str(fi)))
    fig = go.Figure(data=data, frames=frames_plotly)
    fig.update_layout(
        title='Rod Positions Animation (Plotly)',
        scene=dict(aspectmode='data'),
        updatemenus=[{
            'type': 'buttons',
            'buttons': [
                {'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}]},
                {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0}}]}
            ]
        }]
    )
    if html_out:
        fig.write_html(str(html_out))
        print(f"Wrote Plotly animated HTML: {html_out}")
    else:
        fig.show()

def main():
    parser = argparse.ArgumentParser(description="Load per-rod positions and visualize.")
    parser.add_argument("--csv", type=Path, required=False,
                        default=Path("parametric_study/runs/_both_models_demo_patch_test/confined_n20_hard_mu0_10_noise_f1e-05_t1e-05_seed1.csv"),
                        help="Path to per-rod positions CSV.")
    parser.add_argument("--box-size", type=float, default=0.9, help="Periodic box side length.")
    parser.add_argument("--stride", type=int, default=1, help="Keep every k-th frame.")
    parser.add_argument("--frame-limit", type=int, default=None, help="Limit number of frames after stride.")
    parser.add_argument("--animate", action="store_true", help="Animate positions over time.")
    parser.add_argument("--tail", type=int, default=0, help="Trail length (frames) for COM path during animation.")
    parser.add_argument("--save-animation", type=Path, default=None, help="Optional path to save animation (mp4/gif).")
    parser.add_argument("--show-all", action="store_true", help="Show all trajectory points instead of final frame in static plot.")
    parser.add_argument("--no-com", action="store_true", help="Disable periodic COM calculation.")
    parser.add_argument("--show", action="store_true", help="Display plots interactively.")
    parser.add_argument("--com-out", type=Path, default=None, help="Write periodic COM time series to CSV path (frame,com_x,com_y,com_z).")
    parser.add_argument("--com-npz", type=Path, default=None, help="Optional NPZ output storing com array and frames.")
    parser.add_argument("--plotly", action="store_true", help="Use Plotly for interactive 3D visualization (static).")
    parser.add_argument("--plotly-html", type=Path, default=None, help="Output HTML path for Plotly static plot.")
    parser.add_argument("--plotly-animate", action="store_true", help="Use Plotly frames for animation.")
    parser.add_argument("--plotly-sample", type=int, default=5, help="Sample every k-th frame for Plotly animation.")
    args = parser.parse_args()

    df = load_positions_csv(args.csv, stride=args.stride, frame_limit=args.frame_limit)
    positions, frames, rods = build_position_array(df)
    print(f"Loaded positions: frames={len(frames)}, rods={len(rods)}, array shape={positions.shape}")

    com = None if args.no_com else periodic_center_of_mass(positions, args.box_size)
    if com is not None:
        print("First 3 COM positions:")
        print(com[:3])
        if args.com_out is not None:
            out_df = pd.DataFrame({"frame": frames, "com_x": com[:,0], "com_y": com[:,1], "com_z": com[:,2]})
            out_df.to_csv(args.com_out, index=False)
            print(f"Wrote COM CSV: {args.com_out}")
        if args.com_npz is not None:
            np.savez(args.com_npz, com=com, frames=frames)
            print(f"Wrote COM NPZ: {args.com_npz}")

    # Static plot always created (can be suppressed by skipping --show)
    if not args.plotly and not args.plotly_animate:
        # Matplotlib static path
        plot_static(positions, com, frames, show_all=args.show_all)
    else:
        # Plotly path
        if args.plotly:
            plot_plotly_static(positions, com, frames, args.show_all, args.plotly_html)
        if args.plotly_animate:
            plot_plotly_animated(positions, com, frames, args.plotly_html, args.plotly_sample)

    if args.animate and not (args.plotly or args.plotly_animate):
        anim = animate_positions(positions, com, tail=args.tail)
        if args.save_animation is not None:
            out = args.save_animation
            fmt = out.suffix.lower().lstrip('.')
            print(f"Saving animation to {out} (format={fmt}) ...")
            if fmt == 'gif':
                anim.save(out, writer='pillow', fps=25)
            else:
                anim.save(out, writer='ffmpeg', fps=25)
        # Show legend if COM present
        if com is not None:
            plt.legend(loc='upper left')
        if args.show:
            plt.show()
    else:
        if args.show and not (args.plotly or args.plotly_animate):
            plt.show()

if __name__ == "__main__":
    main()