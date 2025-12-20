"""Analyze rod contact network dumps.

Reads the CSV written by `rigidbody_viewer_3d --network <path>`:

  frame,rod_i,rod_j,contact_x,contact_y,contact_z,normal_x,normal_y,normal_z,
  distance,force_a_x,...,friction_b_z

It can:
- extract one frame's contacts
- build a NetworkX graph (nodes=rods, edges=contacts; multiplicity preserved)
- compute per-rod contact counts / unique-neighbor degrees
- optionally convert contact points into rod-local axial coordinate and
  azimuthal angle around the rod axis (requires poses)

Examples:
  python study/network/contact_analysis.py --network study/network/test.csv --frame last

  # Use per-rod pose log to compute axial/azimuth coords
  python study/network/contact_analysis.py --network study/network/test.csv --frame 100 \
	--perrod build/perrod.csv --rod-length 1.0

  # Static rods from init-csv endpoints (assumes pose doesn't change)
  python study/network/contact_analysis.py --network study/network/test.csv --frame first \
	--init-csv initial-configs/.../attempts.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


try:
	import networkx as nx
except Exception as e:  # pragma: no cover
	raise SystemExit(
		"This script requires networkx. Install with: pip install networkx\n"
		f"Import error: {e}"
	)


def _try_import_matplotlib():
	try:
		import matplotlib.pyplot as plt  # type: ignore
		return plt
	except Exception as e:  # pragma: no cover
		raise SystemExit(
			"Plotting requires matplotlib. Install with: pip install matplotlib\n"
			f"Import error: {e}"
		)


Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class ContactRow:
	frame: int
	i: int
	j: int
	p: Vec3
	n: Vec3
	distance: float


@dataclass(frozen=True)
class RodPose:
	p: Vec3
	q: Tuple[float, float, float, float]  # (w,x,y,z)


def _dot(a: Vec3, b: Vec3) -> float:
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a: Vec3, b: Vec3) -> Vec3:
	return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vec3, b: Vec3) -> Vec3:
	return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(a: Vec3, s: float) -> Vec3:
	return (a[0] * s, a[1] * s, a[2] * s)


def _norm(a: Vec3) -> float:
	return math.sqrt(_dot(a, a))


def _normalize(a: Vec3) -> Vec3:
	n = _norm(a)
	if n <= 0:
		return (0.0, 0.0, 0.0)
	return (a[0] / n, a[1] / n, a[2] / n)


def _cross(a: Vec3, b: Vec3) -> Vec3:
	return (
		a[1] * b[2] - a[2] * b[1],
		a[2] * b[0] - a[0] * b[2],
		a[0] * b[1] - a[1] * b[0],
	)


def _quat_rotate(q: Tuple[float, float, float, float], v: Vec3) -> Vec3:
	"""Rotate v by quaternion q=(w,x,y,z)."""
	w, x, y, z = q
	# Using the optimized form: v' = v + 2*cross(q_xyz, cross(q_xyz, v) + w*v)
	qv = (x, y, z)
	t = _mul(_cross(qv, v), 2.0)
	return _add(v, _add(_mul(t, w), _mul(_cross(qv, t), 1.0)))


def _clamp(x: float, a: float, b: float) -> float:
	return a if x < a else b if x > b else x


def segseg_distance(a0: Vec3, a1: Vec3, b0: Vec3, b1: Vec3) -> float:
	"""Minimum distance between two 3D line segments."""
	u = _sub(a1, a0)
	v = _sub(b1, b0)
	w0 = _sub(a0, b0)
	a = _dot(u, u)
	b = _dot(u, v)
	c = _dot(v, v)
	d = _dot(u, w0)
	e = _dot(v, w0)

	eps = 1e-12
	denom = a * c - b * b

	if a <= eps and c <= eps:
		return _norm(_sub(a0, b0))
	if a <= eps:
		t = _clamp(e / c if c > eps else 0.0, 0.0, 1.0)
		pb = _add(b0, _mul(v, t))
		return _norm(_sub(a0, pb))
	if c <= eps:
		s = _clamp(-d / a, 0.0, 1.0)
		pa = _add(a0, _mul(u, s))
		return _norm(_sub(pa, b0))

	if denom > eps:
		s = _clamp((b * e - c * d) / denom, 0.0, 1.0)
	else:
		s = 0.0

	t = (b * s + e) / c
	if t < 0.0:
		t = 0.0
		s = _clamp(-d / a, 0.0, 1.0)
	elif t > 1.0:
		t = 1.0
		s = _clamp((b - d) / a, 0.0, 1.0)

	pa = _add(a0, _mul(u, s))
	pb = _add(b0, _mul(v, t))
	return _norm(_sub(pa, pb))


def rod_endpoints_from_pose(pose: RodPose, rod_length: float) -> Tuple[Vec3, Vec3]:
	axis = _normalize(_quat_rotate(pose.q, (0.0, 1.0, 0.0)))
	half = 0.5 * rod_length
	return _sub(pose.p, _mul(axis, half)), _add(pose.p, _mul(axis, half))


def _grid_key(p: Vec3, cell: float) -> Tuple[int, int, int]:
	return (
		int(math.floor(p[0] / cell)),
		int(math.floor(p[1] / cell)),
		int(math.floor(p[2] / cell)),
	)


def iter_perrod_frames(
	perrod_csv: Path,
	start_frame: int,
	max_frames: int,
) -> Iterable[Tuple[int, Dict[int, RodPose]]]:
	"""Yield (frame, poses) for frames in [start_frame, start_frame+max_frames).

	Reads `perrod.csv` once (much faster than seeking per frame).
	"""
	end_frame = start_frame + max_frames
	with perrod_csv.open("r", newline="") as f:
		r = csv.DictReader(f)
		required = {"frame", "rod", "px", "py", "pz", "qw", "qx", "qy", "qz"}
		missing = [k for k in required if k not in (r.fieldnames or [])]
		if missing:
			raise ValueError(f"Missing columns in {perrod_csv}: {missing}")

		cur_frame: Optional[int] = None
		poses: Dict[int, RodPose] = {}
		for row in r:
			fr = _safe_int(row["frame"], default=-1)
			if fr < start_frame:
				continue
			if fr >= end_frame:
				break

			if cur_frame is None:
				cur_frame = fr
			elif fr != cur_frame:
				yield cur_frame, poses
				poses = {}
				cur_frame = fr

			rid = _safe_int(row["rod"], default=-1)
			if rid < 0:
				continue
			poses[rid] = RodPose(
				p=(
					_safe_float(row["px"]),
					_safe_float(row["py"]),
					_safe_float(row["pz"]),
				),
				q=(
					_safe_float(row["qw"]),
					_safe_float(row["qx"]),
					_safe_float(row["qy"]),
					_safe_float(row["qz"]),
				),
			)

		if cur_frame is not None:
			yield cur_frame, poses


def find_nearby_pairs_for_frame(
	poses: Dict[int, RodPose],
	rod_length: float,
	rod_diameter: float,
	thresh_mult: float,
) -> List[Tuple[int, int, float]]:
	"""Find rod pairs whose axis-segment distance is < thresh_mult * rod_diameter."""
	if rod_length <= 0 or rod_diameter <= 0:
		raise ValueError("rod_length and rod_diameter must be > 0")
	if thresh_mult <= 0:
		raise ValueError("thresh_mult must be > 0")

	thresh = thresh_mult * rod_diameter
	# Conservative bounding sphere radius for the capsule
	R = math.sqrt((0.5 * rod_length) ** 2 + (0.5 * rod_diameter) ** 2)
	cell = max(1e-9, (2.0 * R + thresh))

	rods = sorted(poses.keys())
	centers = {rid: poses[rid].p for rid in rods}
	grid: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
	for rid in rods:
		grid[_grid_key(centers[rid], cell)].append(rid)

	out: List[Tuple[int, int, float]] = []
	seen: set[Tuple[int, int]] = set()

	for rid in rods:
		cx, cy, cz = _grid_key(centers[rid], cell)
		for dx in (-1, 0, 1):
			for dy in (-1, 0, 1):
				for dz in (-1, 0, 1):
					for other in grid.get((cx + dx, cy + dy, cz + dz), []):
						if other <= rid:
							continue
						key = (rid, other)
						if key in seen:
							continue
						seen.add(key)

						dcen = _norm(_sub(centers[rid], centers[other]))
						if dcen > (2.0 * R + thresh):
							continue

						a0, a1 = rod_endpoints_from_pose(poses[rid], rod_length)
						b0, b1 = rod_endpoints_from_pose(poses[other], rod_length)
						dmin = segseg_distance(a0, a1, b0, b1)
						if dmin < thresh:
							out.append((rid, other, dmin))

	return out


def _safe_int(s: str, default: int = 0) -> int:
	try:
		return int(float(s))
	except Exception:
		return default


def _safe_float(s: str, default: float = 0.0) -> float:
	try:
		return float(s)
	except Exception:
		return default


def scan_first_last_frame(network_csv: Path) -> Tuple[int, int]:
	first: Optional[int] = None
	last: Optional[int] = None
	with network_csv.open("r", newline="") as f:
		r = csv.DictReader(f)
		for row in r:
			if "frame" not in row:
				continue
			try:
				fr = int(float(row["frame"]))
			except Exception:
				continue
			if first is None:
				first = fr
			last = fr
	if first is None or last is None:
		raise ValueError(f"No data rows in {network_csv}")
	return first, last


def scan_first_last_contact_frame(network_csv: Path) -> Tuple[int, int]:
	"""Return first/last frame that contains at least one rod-rod contact row."""
	first: Optional[int] = None
	last: Optional[int] = None
	with network_csv.open("r", newline="") as f:
		r = csv.DictReader(f)
		for row in r:
			try:
				fr = int(float(row.get("frame", "")))
				i = int(float(row.get("rod_i", "-1")))
				j = int(float(row.get("rod_j", "-1")))
			except Exception:
				continue
			if i < 0 or j < 0:
				continue
			if first is None:
				first = fr
			last = fr
	if first is None or last is None:
		raise ValueError(f"No contact rows found in {network_csv}")
	return first, last


def read_contacts_for_frame(network_csv: Path, frame: int) -> List[ContactRow]:
	out: List[ContactRow] = []
	with network_csv.open("r", newline="") as f:
		r = csv.DictReader(f)
		required = {
			"frame",
			"rod_i",
			"rod_j",
			"contact_x",
			"contact_y",
			"contact_z",
			"normal_x",
			"normal_y",
			"normal_z",
			"distance",
		}
		missing = [k for k in required if k not in (r.fieldnames or [])]
		if missing:
			raise ValueError(
				f"Missing columns in {network_csv}: {missing}. "
				"(Did you generate this with --network ?)"
			)

		for row in r:
			fr = _safe_int(row["frame"], default=-1)
			if fr < frame:
				continue
			if fr > frame:
				break
			i = _safe_int(row["rod_i"], default=-1)
			j = _safe_int(row["rod_j"], default=-1)
			# Sentinel row for empty-contact frames
			if i < 0 or j < 0:
				continue

			p = (
				_safe_float(row["contact_x"]),
				_safe_float(row["contact_y"]),
				_safe_float(row["contact_z"]),
			)
			n = (
				_safe_float(row["normal_x"]),
				_safe_float(row["normal_y"]),
				_safe_float(row["normal_z"]),
			)
			out.append(
				ContactRow(
					frame=fr,
					i=i,
					j=j,
					p=p,
					n=n,
					distance=_safe_float(row["distance"]),
				)
			)
	return out


def build_graph(contacts: Iterable[ContactRow]) -> nx.Graph:
	"""Undirected graph; edge attribute 'count' increments per contact row."""
	g = nx.Graph()
	for c in contacts:
		a, b = (c.i, c.j) if c.i <= c.j else (c.j, c.i)
		if g.has_edge(a, b):
			g[a][b]["count"] += 1
		else:
			g.add_edge(a, b, count=1)
	return g


def per_rod_stats(contacts: Iterable[ContactRow], g: nx.Graph) -> Dict[int, Dict[str, int]]:
	"""Returns per-rod: total_contact_rows, unique_neighbors."""
	row_counts: Counter[int] = Counter()
	for c in contacts:
		row_counts[c.i] += 1
		row_counts[c.j] += 1

	out: Dict[int, Dict[str, int]] = {}
	rods = set(row_counts.keys()) | set(g.nodes)
	for rid in sorted(rods):
		out[rid] = {
			"total_contact_rows": int(row_counts.get(rid, 0)),
			"unique_neighbors": int(g.degree(rid)) if rid in g else 0,
		}
	return out


def load_perrod_poses(perrod_csv: Path, frame: int) -> Dict[int, RodPose]:
	"""Load poses (p,q) for all rods at the given frame from perrod.csv."""
	poses: Dict[int, RodPose] = {}
	with perrod_csv.open("r", newline="") as f:
		r = csv.DictReader(f)
		required = {"frame", "rod", "px", "py", "pz", "qw", "qx", "qy", "qz"}
		missing = [k for k in required if k not in (r.fieldnames or [])]
		if missing:
			raise ValueError(f"Missing columns in {perrod_csv}: {missing}")

		for row in r:
			fr = _safe_int(row["frame"], default=-1)
			if fr < frame:
				continue
			if fr > frame:
				break
			rid = _safe_int(row["rod"], default=-1)
			if rid < 0:
				continue
			poses[rid] = RodPose(
				p=(
					_safe_float(row["px"]),
					_safe_float(row["py"]),
					_safe_float(row["pz"]),
				),
				q=(
					_safe_float(row["qw"]),
					_safe_float(row["qx"]),
					_safe_float(row["qy"]),
					_safe_float(row["qz"]),
				),
			)
	return poses


def load_init_endpoints(init_csv: Path) -> Dict[int, Tuple[Vec3, Vec3]]:
	"""Load endpoints from an init-csv style file.

	Accepts:
	- CSV with header x0,y0,z0,x1,y1,z1 (optional extra cols)
	- whitespace-separated 6 floats per row (headerless)
	- comment lines starting with '#'
	"""
	endpoints: Dict[int, Tuple[Vec3, Vec3]] = {}
	idx = 0
	with init_csv.open("r") as f:
		for raw in f:
			line = raw.strip()
			if not line or line.startswith("#"):
				continue
			low = line.lower().replace("\t", " ")
			if "x0" in low and "y0" in low and "z0" in low and "x1" in low:
				continue
			# parse either CSV or whitespace
			if "," in line:
				parts = [p.strip() for p in line.split(",") if p.strip()]
			else:
				parts = line.split()
			if len(parts) < 6:
				continue
			vals = [float(parts[k]) for k in range(6)]
			p0 = (vals[0], vals[1], vals[2])
			p1 = (vals[3], vals[4], vals[5])
			endpoints[idx] = (p0, p1)
			idx += 1
	return endpoints


def pose_from_endpoints(p0: Vec3, p1: Vec3) -> RodPose:
	"""Construct an approximate pose where local +Y aligns with (p1-p0).

	We only need the axis direction for axial/azimuth coordinates, so the
	quaternion doesn't need to match the simulator exactly.
	"""
	c = _mul(_add(p0, p1), 0.5)
	axis = _sub(p1, p0)
	d = _normalize(axis)
	# Create a quaternion that rotates +Y=(0,1,0) to d.
	up = (0.0, 1.0, 0.0)
	dot_ud = max(-1.0, min(1.0, _dot(up, d)))
	if dot_ud > 0.999999:
		q = (1.0, 0.0, 0.0, 0.0)
	elif dot_ud < -0.999999:
		# 180 degrees around Z
		q = (0.0, 0.0, 0.0, 1.0)
	else:
		cax = _cross(up, d)
		s = math.sqrt((1.0 + dot_ud) * 2.0)
		invs = 1.0 / s
		q = (s * 0.5, cax[0] * invs, cax[1] * invs, cax[2] * invs)
	return RodPose(p=c, q=q)


def contact_coords_rod_local(
	pose: RodPose,
	rod_length: float,
	contact_point: Vec3,
) -> Tuple[float, float, float]:
	"""Return (axial_s, radial_r, azimuth_theta) for one rod.

	- axial_s: projection along rod axis (0 at center), in world units
	- radial_r: distance from axis
	- azimuth_theta: angle around axis in [-pi,pi]
	"""
	# local +Y is rod axis in sim
	axis = _normalize(_quat_rotate(pose.q, (0.0, 1.0, 0.0)))
	r = _sub(contact_point, pose.p)
	s = _dot(r, axis)
	perp = _sub(r, _mul(axis, s))
	rr = _norm(perp)

	# stable basis around axis
	ref = (1.0, 0.0, 0.0)
	if abs(_dot(axis, ref)) > 0.9:
		ref = (0.0, 0.0, 1.0)
	e1 = _normalize(_cross(axis, ref))
	e2 = _cross(axis, e1)

	if rr < 1e-12:
		theta = 0.0
	else:
		theta = math.atan2(_dot(perp, e2), _dot(perp, e1))

	# Clamp s slightly for reporting convenience
	halfL = 0.5 * rod_length
	if halfL > 0:
		s = max(-halfL, min(halfL, s))
	return s, rr, theta


def main(argv: Optional[List[str]] = None) -> int:
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--network", type=Path, required=False, help="Network CSV path (required unless --nearby-pairs)")

	# Nearby-pairs scan (distance-based; uses perrod poses)
	ap.add_argument(
		"--nearby-pairs",
		action="store_true",
		help="Scan frames and report pairs with segment distance < k*rod_diameter",
	)
	ap.add_argument(
		"--start-frame",
		type=int,
		default=0,
		help="Start frame for --nearby-pairs scan (default: 0)",
	)
	ap.add_argument(
		"--frames",
		type=int,
		default=100,
		help="Number of frames to scan for --nearby-pairs (default: 100)",
	)
	ap.add_argument(
		"--k",
		type=float,
		default=3.0,
		help="Threshold multiplier: distance < k*rod_diameter (default: 3)",
	)
	ap.add_argument(
		"--rod-diameter",
		type=float,
		default=None,
		help="Rod diameter (required for --nearby-pairs)",
	)
	ap.add_argument(
		"--out-pairs",
		type=Path,
		default=None,
		help="Optional CSV output for nearby pairs: frame,rod_i,rod_j,dmin",
	)
	ap.add_argument(
		"--out-nearby-stats",
		type=Path,
		default=None,
		help="Optional CSV output for nearby-pairs time series: frame,count,mean_dmin,std_dmin,mean_gap",
	)
	ap.add_argument(
		"--plot-nearby",
		action="store_true",
		help="Plot mean_dmin vs frame for nearby pairs (requires --nearby-pairs)",
	)
	ap.add_argument(
		"--plot-out",
		type=Path,
		default=None,
		help="Output plot path (e.g. nearby_mean.png). If omitted, shows an interactive window.",
	)
	ap.add_argument(
		"--frame",
		default="last",
		help=(
			"Frame to analyze: integer, 'first', 'last', 'first-contact', or "
			"'last-contact' (default: last)"
		),
	)
	ap.add_argument(
		"--perrod",
		type=Path,
		default=None,
		help="Optional per-rod CSV (for rod poses): frame,rod,px,py,pz,...,qw,qx,qy,qz",
	)
	ap.add_argument(
		"--init-csv",
		type=Path,
		default=None,
		help="Optional init-csv endpoints file (static pose assumption)",
	)
	ap.add_argument(
		"--rod-length",
		type=float,
		default=None,
		help="Rod length (needed for axial clamping; optional but recommended)",
	)
	ap.add_argument(
		"--top",
		type=int,
		default=10,
		help="Show top-N rods by contacts (default: 10)",
	)
	ap.add_argument(
		"--out-contacts",
		type=Path,
		default=None,
		help="Optional output CSV for per-contact per-rod local coords",
	)
	args = ap.parse_args(argv)

	if args.network is None:
		if not args.nearby_pairs:
			raise SystemExit("--network is required unless --nearby-pairs is set")
	else:
		if not args.network.exists():
			raise SystemExit(f"Network CSV not found: {args.network}")

	if args.nearby_pairs:
		if args.perrod is None:
			raise SystemExit("--nearby-pairs requires --perrod")
		if args.rod_length is None:
			raise SystemExit("--nearby-pairs requires --rod-length")
		if args.rod_diameter is None:
			raise SystemExit("--nearby-pairs requires --rod-diameter")
		if args.frames <= 0:
			raise SystemExit("--frames must be > 0")

		out_f = None
		writer = None
		if args.out_pairs is not None:
			args.out_pairs.parent.mkdir(parents=True, exist_ok=True)
			out_f = args.out_pairs.open("w", newline="")
			writer = csv.writer(out_f)
			writer.writerow(["frame", "rod_i", "rod_j", "dmin"])

		stats_f = None
		stats_w = None
		if args.out_nearby_stats is not None:
			args.out_nearby_stats.parent.mkdir(parents=True, exist_ok=True)
			stats_f = args.out_nearby_stats.open("w", newline="")
			stats_w = csv.writer(stats_f)
			stats_w.writerow(["frame", "count", "mean_dmin", "std_dmin", "mean_gap"])

		total_pairs = 0
		frames_list: List[int] = []
		mean_list: List[float] = []
		std_list: List[float] = []
		count_list: List[int] = []
		for fr, poses in iter_perrod_frames(args.perrod, args.start_frame, args.frames):
			pairs = find_nearby_pairs_for_frame(
				poses,
				rod_length=float(args.rod_length),
				rod_diameter=float(args.rod_diameter),
				thresh_mult=float(args.k),
			)
			total_pairs += len(pairs)
			count = len(pairs)
			if count:
				ds = [d for _, _, d in pairs]
				mean = sum(ds) / count
				var = sum((d - mean) ** 2 for d in ds) / count
				std = math.sqrt(var)
				mean_gap = mean - float(args.rod_diameter)
			else:
				mean = 0.0
				std = 0.0
				mean_gap = 0.0
			frames_list.append(fr)
			mean_list.append(mean)
			std_list.append(std)
			count_list.append(count)
			print(f"frame={fr}: nearby_pairs={count} mean_dmin={mean:.6g} std_dmin={std:.6g} (k={args.k})")
			if stats_w is not None:
				stats_w.writerow([fr, count, mean, std, mean_gap])
			if writer is not None:
				for i, j, dmin in pairs:
					writer.writerow([fr, i, j, dmin])

		if out_f is not None:
			out_f.close()
		if stats_f is not None:
			stats_f.close()

		if args.plot_nearby:
			plt = _try_import_matplotlib()
			if not frames_list:
				raise SystemExit("No frames scanned; cannot plot")
			x = frames_list
			y = mean_list
			plt.figure(figsize=(8, 4.5))
			plt.plot(x, y, lw=1.5, label="mean dmin")
			# Optional: show +/- std as a light band
			if any(s > 0 for s in std_list):
				lo = [m - s for m, s in zip(mean_list, std_list)]
				hi = [m + s for m, s in zip(mean_list, std_list)]
				plt.fill_between(x, lo, hi, alpha=0.2, label="±1 std")
			plt.xlabel("frame")
			plt.ylabel("mean nearest distance dmin")
			plt.title(f"Nearby pairs: dmin < {args.k} * diameter")
			plt.grid(True, alpha=0.3)
			plt.legend(loc="best")
			plt.tight_layout()
			if args.plot_out is not None:
				args.plot_out.parent.mkdir(parents=True, exist_ok=True)
				plt.savefig(args.plot_out)
				print(f"Wrote plot {args.plot_out}")
			else:
				plt.show()
		if args.out_pairs is not None:
			print(f"Wrote {args.out_pairs} (total_pairs={total_pairs})")
		if args.out_nearby_stats is not None:
			print(f"Wrote {args.out_nearby_stats}")
		if args.out_pairs is None and args.out_nearby_stats is None:
			print(f"Done (total_pairs={total_pairs})")
		return 0

	first, last = scan_first_last_frame(args.network)
	if isinstance(args.frame, str):
		fstr = args.frame.strip().lower()
		if fstr == "first":
			frame = first
		elif fstr == "last":
			frame = last
		elif fstr == "first-contact":
			frame = scan_first_last_contact_frame(args.network)[0]
		elif fstr == "last-contact":
			frame = scan_first_last_contact_frame(args.network)[1]
		else:
			frame = int(fstr)
	else:
		frame = int(args.frame)

	contacts = read_contacts_for_frame(args.network, frame)
	g = build_graph(contacts)
	stats = per_rod_stats(contacts, g)

	print(f"frame={frame} contacts_rows={len(contacts)} nodes={g.number_of_nodes()} edges={g.number_of_edges()}")

	# Top rods by contact rows
	top = sorted(stats.items(), key=lambda kv: kv[1]["total_contact_rows"], reverse=True)
	print(f"Top {min(args.top, len(top))} rods by total_contact_rows:")
	for rid, s in top[: args.top]:
		print(f"  rod {rid:4d}: rows={s['total_contact_rows']:4d}  unique_neighbors={s['unique_neighbors']:3d}")

	# Degree distribution
	degs = [d for _, d in g.degree()]
	deg_hist = Counter(degs)
	if deg_hist:
		print("Degree histogram (unique_neighbors -> count):")
		for k in sorted(deg_hist):
			print(f"  {k:3d} -> {deg_hist[k]}")
	else:
		print("Degree histogram: (empty graph)")

	# Optional: per-contact per-rod local coords
	if args.out_contacts is not None:
		rod_length = args.rod_length
		if rod_length is None:
			raise SystemExit("--out-contacts requires --rod-length (for axial clamping)")

		poses: Dict[int, RodPose] = {}
		if args.perrod is not None:
			poses = load_perrod_poses(args.perrod, frame)
		elif args.init_csv is not None:
			eps = load_init_endpoints(args.init_csv)
			poses = {rid: pose_from_endpoints(p0, p1) for rid, (p0, p1) in eps.items()}
		else:
			raise SystemExit("--out-contacts requires --perrod or --init-csv to provide rod poses")

		args.out_contacts.parent.mkdir(parents=True, exist_ok=True)
		with args.out_contacts.open("w", newline="") as f:
			w = csv.writer(f)
			w.writerow([
				"frame",
				"contact_idx",
				"rod",
				"other",
				"contact_x",
				"contact_y",
				"contact_z",
				"axial_s",
				"radial_r",
				"azimuth_theta",
			])

			missing_pose = 0
			for k, c in enumerate(contacts):
				for rod, other in ((c.i, c.j), (c.j, c.i)):
					pose = poses.get(rod)
					if pose is None:
						missing_pose += 1
						continue
					s, rr, th = contact_coords_rod_local(pose, rod_length, c.p)
					w.writerow([
						frame,
						k,
						rod,
						other,
						c.p[0],
						c.p[1],
						c.p[2],
						s,
						rr,
						th,
					])

		if missing_pose:
			print(f"Wrote {args.out_contacts} (note: missing poses for {missing_pose} rod-entries)")
		else:
			print(f"Wrote {args.out_contacts}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())