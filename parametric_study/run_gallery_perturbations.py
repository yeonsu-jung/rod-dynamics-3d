#!/usr/bin/env python3

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GALLERY = ROOT / "packing_gallery" / "gallery.json"
DEFAULT_SCENE = ROOT / "assets" / "scenes" / "default_entangled.json"
DEFAULT_OUTPUT_ROOT = ROOT / "packing_gallery" / "runs"
DEFAULT_BINARY_CANDIDATES = [
    ROOT / "build_wsl" / "rigidbody_viewer_3d",
    ROOT / "build_wsl_gl" / "rigidbody_viewer_3d",
    ROOT / "build_wsl_cuda" / "rigidbody_viewer_3d",
    ROOT / "build_wsl_gl_cuda" / "rigidbody_viewer_3d",
    ROOT / "build_wsl_gl_dbg" / "rigidbody_viewer_3d",
    ROOT / "build" / "rigidbody_viewer_3d",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run gallery packings with three perturbation cases: tight rod, "
            "loose rod, and all rods."
        )
    )
    parser.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--dt", type=float, default=0.0005)
    parser.add_argument("--lin-vel", type=float, default=0.1)
    parser.add_argument("--ang-vel", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--friction", type=float, default=None)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--output-frames", type=int, default=300)
    parser.add_argument("--trajectory-frames", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--packing-index", type=int, nargs="*", default=None)
    parser.add_argument("--n", type=int, nargs="*", default=None)
    parser.add_argument("--ar", type=float, nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["tight", "loose", "all"],
        default=["tight", "loose", "all"],
    )
    return parser.parse_args()


def resolve_binary(explicit_binary: Optional[Path]) -> Path:
    if explicit_binary is not None:
        binary = explicit_binary.resolve()
        if not binary.exists():
            raise FileNotFoundError(f"Binary not found: {binary}")
        return binary

    for candidate in DEFAULT_BINARY_CANDIDATES:
        if candidate.exists():
            return candidate

    tried = "\n".join(str(path) for path in DEFAULT_BINARY_CANDIDATES)
    raise FileNotFoundError(f"No simulator binary found. Tried:\n{tried}")


def sanitize_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def load_gallery_entries(gallery_path: Path) -> List[Dict]:
    with gallery_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Gallery must contain a list of packings: {gallery_path}")
    return data


def select_entries(entries: Sequence[Dict], args: argparse.Namespace) -> List[Dict]:
    selected: List[Dict] = []
    allowed_indices = set(args.packing_index) if args.packing_index is not None else None
    allowed_n = set(args.n) if args.n else None
    allowed_ar = {float(value) for value in args.ar} if args.ar else None

    for index, entry in enumerate(entries):
        if allowed_indices is not None and index not in allowed_indices:
            continue
        if allowed_n is not None and int(entry["N"]) not in allowed_n:
            continue
        if allowed_ar is not None and float(entry["AR"]) not in allowed_ar:
            continue
        selected.append(entry)

    if args.limit > 0:
        selected = selected[: args.limit]
    return selected


def resolve_packing_path(gallery_path: Path, entry: Dict) -> Path:
    source = Path(entry["source_file"])
    if source.is_absolute():
        resolved = source
    else:
        relative_source = str(source)
        if relative_source.startswith("./"):
            relative_source = relative_source[2:]
        resolved = gallery_path.parent / relative_source
    if not resolved.exists():
        raise FileNotFoundError(f"Packing file not found: {resolved}")
    return resolved.resolve()


def make_scene(base_scene: Dict, entry: Dict, args: argparse.Namespace) -> Dict:
    scene = json.loads(json.dumps(base_scene))
    scene.setdefault("scene", {})
    scene.setdefault("physics", {})
    scene["scene"].setdefault("populate", {})
    scene["scene"].setdefault("randomInit", {})
    scene["physics"].setdefault("soft_contact", {})

    scene["scene"]["populate"]["count"] = int(entry["n_rods"])
    scene["scene"]["populate"]["radius"] = float(entry["rod_radius"])
    scene["scene"]["populate"]["length"] = 1.0

    scene["scene"]["randomInit"] = {
        "enabled": True,
        "vSigma": float(args.lin_vel),
        "wSpeed": float(args.ang_vel),
        "seed": int(args.seed),
    }

    if scene["scene"].get("randomForce"):
        scene["scene"]["randomForce"]["enabled"] = False

    if "bodies" in scene["scene"] and scene["scene"]["bodies"]:
        body0 = scene["scene"]["bodies"][0]
        body0["length"] = 1.0
        body0["diameter"] = 2.0 * float(entry["rod_radius"])

    scene["physics"]["dt"] = float(args.dt)
    if args.friction is not None:
        scene["physics"]["soft_contact"]["mu"] = float(args.friction)
        scene["physics"]["soft_contact"]["mu_static"] = float(args.friction)
    return scene


def build_packing_label(entry: Dict, packing_path: Path) -> str:
    copied = entry.get("copied_to")
    if copied:
        label = Path(copied).stem
    else:
        label = f"packing_N{entry['N']}_AR{entry['AR']}"
    seed_token = sanitize_token(packing_path.parent.name)
    if seed_token and seed_token not in label:
        label = f"{label}_{seed_token}"
    return sanitize_token(label)


def case_specs(entry: Dict, enabled_cases: Iterable[str]) -> List[Dict[str, Optional[int]]]:
    specs = {
        "tight": {
            "name": "tight",
            "rod_index": int(entry["tight_rod"]["index"]),
            "gap": float(entry["tight_rod"]["min_gap"]),
        },
        "loose": {
            "name": "loose",
            "rod_index": int(entry["loose_rod"]["index"]),
            "gap": float(entry["loose_rod"]["min_gap"]),
        },
        "all": {
            "name": "all",
            "rod_index": None,
            "gap": None,
        },
    }
    return [specs[name] for name in enabled_cases]


def compute_stride(steps: int, frames: int) -> int:
    return max(1, steps // max(1, frames))


def write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")


def run_case(
    entry: Dict,
    packing_path: Path,
    label: str,
    case: Dict[str, Optional[int]],
    base_scene: Dict,
    binary_path: Path,
    args: argparse.Namespace,
) -> Dict[str, object]:
    output_stride = compute_stride(args.steps, args.output_frames)
    trajectory_stride = compute_stride(args.steps, args.trajectory_frames)

    run_dir = args.output_root / label / case["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    scene_payload = make_scene(base_scene, entry, args)
    scene_path = run_dir / "scene.json"
    write_json(scene_path, scene_payload)

    metadata = {
        "packing_label": label,
        "packing_source": str(packing_path),
        "N": int(entry["N"]),
        "AR": float(entry["AR"]),
        "rod_radius": float(entry["rod_radius"]),
        "case": case["name"],
        "perturb_rod": case["rod_index"],
        "min_gap": case["gap"],
        "steps": int(args.steps),
        "dt": float(args.dt),
        "lin_vel": float(args.lin_vel),
        "ang_vel": float(args.ang_vel),
        "seed": int(args.seed),
        "output_stride": output_stride,
        "trajectory_stride": trajectory_stride,
        "output_frames": int(args.output_frames),
        "trajectory_frames": int(args.trajectory_frames),
        "friction": scene_payload["physics"]["soft_contact"].get("mu"),
    }
    write_json(run_dir / "run_metadata.json", metadata)

    command = [
        str(binary_path),
        "--headless",
        "--scene",
        str(scene_path),
        "--init-csv",
        str(packing_path),
        "--output",
        str(run_dir / "output.csv"),
        "--output-stride",
        str(output_stride),
        "--output-max",
        str(args.output_frames),
        "--perrod",
        str(run_dir / "perrod.csv"),
        "--perrod-stride",
        str(trajectory_stride),
        "--perrod-max",
        str(args.trajectory_frames),
        "--steps",
        str(args.steps),
        "--dt",
        str(args.dt),
        "--seed",
        str(args.seed),
        "--no-network",
        "--no-csv",
        "--entanglement",
        "--entanglement-period",
        str(args.steps),
    ]

    if args.threads > 0:
        command.extend(["--threads", str(args.threads)])
    if case["rod_index"] is not None:
        command.extend(["--perturb-rod", str(case["rod_index"])])

    write_text(run_dir / "command.sh", " ".join(shlex.quote(part) for part in command) + "\n")

    if args.dry_run:
        return {
            "run_dir": str(run_dir),
            "case": case["name"],
            "returncode": 0,
            "dry_run": True,
        }

    completed = subprocess.run(
        command,
        cwd=run_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    write_text(run_dir / "stdout.log", completed.stdout)
    write_text(run_dir / "stderr.log", completed.stderr)

    if completed.returncode != 0:
        raise RuntimeError(
            f"Simulation failed for {label}/{case['name']} with exit code "
            f"{completed.returncode}. See {run_dir / 'stderr.log'}"
        )

    return {
        "run_dir": str(run_dir),
        "case": case["name"],
        "returncode": completed.returncode,
        "dry_run": False,
    }


def main() -> int:
    args = parse_args()
    gallery_path = args.gallery.resolve()
    scene_path = args.scene.resolve()
    output_root = args.output_root.resolve()
    binary_path = resolve_binary(args.binary)
    args.gallery = gallery_path
    args.scene = scene_path
    args.output_root = output_root
    if args.binary is not None:
        args.binary = binary_path

    entries = load_gallery_entries(gallery_path)
    selected_entries = select_entries(entries, args)
    if not selected_entries:
        raise SystemExit("No packings matched the requested filters.")

    with scene_path.open("r", encoding="utf-8") as handle:
        base_scene = json.load(handle)

    output_root.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, object]] = []
    for entry in selected_entries:
        packing_path = resolve_packing_path(gallery_path, entry)
        label = build_packing_label(entry, packing_path)
        for case in case_specs(entry, args.cases):
            result = run_case(
                entry=entry,
                packing_path=packing_path,
                label=label,
                case=case,
                base_scene=base_scene,
                binary_path=binary_path,
                args=args,
            )
            manifest.append(result)
            print(f"[{case['name']}] {label} -> {result['run_dir']}")

    write_json(output_root / "manifest.json", manifest)
    print(f"Completed {len(manifest)} simulations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())