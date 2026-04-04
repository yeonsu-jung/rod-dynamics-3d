from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random


REPO_ROOT = Path(__file__).resolve().parents[2]
INIT_FILE_ROOT_PATH = REPO_ROOT / "initial-configs" / "relaxation_3rd_multithreading"


@dataclass(frozen=True)
class RelaxedPacking:
    path: Path
    n_rods: int
    aspect_ratio: int
    seed_triplet: str


def _parse_relaxed_path(path: Path) -> RelaxedPacking:
    return RelaxedPacking(
        path=path,
        n_rods=int(path.parent.parent.name.removeprefix("N")),
        aspect_ratio=int(path.stem.removeprefix("x_relaxed_AR")),
        seed_triplet=path.parent.name,
    )


def choose_random_relaxed_packing(
    root: Path = INIT_FILE_ROOT_PATH,
    *,
    n_rods: int | None = None,
    aspect_ratio: int | None = None,
    rng: random.Random | None = None,
) -> RelaxedPacking:
    rng = rng or random.Random()

    candidates = []
    for path in root.rglob("x_relaxed_AR*.txt"):
        packing = _parse_relaxed_path(path)
        if n_rods is not None and packing.n_rods != n_rods:
            continue
        if aspect_ratio is not None and packing.aspect_ratio != aspect_ratio:
            continue
        candidates.append(packing)

    if not candidates:
        raise FileNotFoundError(
            f"No relaxed packing found under {root} for "
            f"n_rods={n_rods!r}, aspect_ratio={aspect_ratio!r}"
        )

    return rng.choice(candidates)


if __name__ == "__main__":
    packing = choose_random_relaxed_packing()
    print(f"Picked: {packing.path}")
    print(f"N={packing.n_rods}, AR={packing.aspect_ratio}, seed={packing.seed_triplet}")
