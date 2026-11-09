# rod-dynamics-3d

A C++17 rigid-body engine for dense packings of slender rods (capsules) in frictional contact. It simulates hundreds to thousands of rods with hard (nonsmooth) or penalty contact, can track the pairwise entanglement of the packing as it evolves, and runs either headless for parameter sweeps or with an OpenGL viewer. Python bindings expose a step-by-step interface.

The engine is used for the simulations in Y. Jung and L. Mahadevan, "Emergent cohesion via self-caging in maximally entangled rod packings", [arXiv:2606.03952](https://arxiv.org/abs/2606.03952). [`repro/`](repro/README.md) regenerates that paper's figures.

## Features

**Contact models.** Select one with `--contact-model <name>` or `physics.contact_model` in the scene JSON:

| Name | Model |
|---|---|
| `nsc` | Nonsmooth contact: velocity-level projected SOR (Gauss–Seidel at ω = 1) with Coulomb friction-cone projection and restitution, followed by normal-only position stabilization; warm-started. Follows Chrono's implicit-Euler projected time stepper. |
| `harmonic` | Smooth penalty contact from a contact potential (after DisMech), with static and kinetic friction, integrated by velocity Verlet. |
| `hertz-mindlin` | Hertzian normal force and Mindlin tangential friction for spheres (granular DEM). |
| `mujoco` | Experimental MuJoCo-style penalty model, for A/B tests against `harmonic` (no periodic boundaries). |

**Geometry and boundaries.** Capsule–capsule closest points (Lumelsky's segment–segment algorithm), spheres and mixed sphere–capsule scenes, free or periodic boundaries with minimum-image contacts, and an optional floor.

**Performance.** Spatial-hash and AABB broad phase; OpenMP-parallel contact detection, which switches to one thread for small systems so many independent runs can share a machine; an optional CUDA broad phase that checks all pairs on the GPU (`-DENABLE_CUDA=ON`); optional Tracy profiler zones (`-DTRACY_ENABLE=ON`).

**Observables.** Pairwise linking numbers of the rods, computed every N frames ([entanglement-cpp](#build)). Per-step profile CSV (kinetic energy, contact and collision counts, time spent in each solver phase), per-rod trajectories, a contact-network export, energy-balance and normal-velocity diagnostics, and PNG frame dumps for movies.

**Initial conditions.** Rod endpoints from a packing file (`--init-csv`), or random non-overlapping placement. Gaussian velocity kicks with the axial spin removed, and optional per-rod overrides or fixed rods.

## Build

Requirements:
- a C++17 compiler and CMake ≥ 3.18;
- Python 3 with development headers (for the bindings).

CMake fetches pybind11, nlohmann/json, GLM, zlib, and (for the viewer) GLFW and GLAD if they are not installed. OpenMP is used when found. On macOS this requires `libomp` from Homebrew.

```bash
git clone --recursive https://github.com/yeonsu-jung/rod-dynamics-3d.git
cd rod-dynamics-3d

# Headless binary and Python module (no OpenGL needed)
cmake -B build-headless -DBUILD_HEADLESS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-headless -j

# Interactive viewer (needs OpenGL)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The linking-number code is a submodule, [`external/entanglement-cpp`](https://github.com/yeonsu-jung/entanglement-cpp). If you cloned without `--recursive`, CMake runs `git submodule update --init` for it at configure time.

| CMake option | Default | Effect |
|---|---|---|
| `BUILD_HEADLESS` | `OFF` | Build without OpenGL (servers, clusters, CI) |
| `ENABLE_CUDA` | `OFF` | Compile the CUDA all-pairs broad phase |
| `TRACY_ENABLE` | `OFF` | Tracy profiler zones |
| `BUILD_SANITIZERS` | `ON` | AddressSanitizer and UBSan in Debug builds |

## Quick start

Run 200 rods from a relaxed packing for 1,000 steps with nonsmooth contact, computing entanglement every 100 steps:

```bash
./build-headless/rigidbody_viewer_3d \
  --scene assets/scenes/default_entangled_nsc.json \
  --init-csv "assets/initial-configs/37,178,56/2025-02-16_17_EntangledRelaxedPacking-N0200-AR0050-Scale1/x_relaxed.txt" \
  --headless --steps 1000 --entanglement --ent-period 100 --csv profile.csv
```

The same command without `--headless`, run with the binary in `build/`, opens the viewer. `--help` lists every option, including solver overrides (`--nsc-iters`, `--nsc-mu`, `--nsc-cfm`, `--dt`), output (`--perrod`, `--network`, `--export`) and early-stopping criteria.

A scene file has three blocks:
- `scene`: boundaries, rod properties in `bodies`, random placement in `populate`, and initial kicks in `randomInit`;
- `physics`: time step, damping, and the contact-model parameters;
- `render`.

[`assets/scenes/default_entangled_nsc.json`](assets/scenes/default_entangled_nsc.json) holds the protocol of arXiv:2606.03952 (Table S1).

### Python

```python
import rod_dynamics_py as rd   # PYTHONPATH=build-headless

sim = rd.Simulator("assets/scenes/default_entangled_nsc.json",
                   "assets/initial-configs/.../x_relaxed.txt")
sim.step(100)
print(sim.diagnostics())   # frame_index, dt, last_ke, last_hit_count, rod_count, ...
rod = sim.rods()[0]        # position, orientation_wxyz, linear/angular velocity,
                           # force, torque, mass, endpoints, ...
```

## Tests and CI

```bash
ctest --test-dir build-headless --output-on-failure
```

The smoke test (`tests/smoke_test.py`) loads the Python module and steps a scene with finite state. It also guards against fixed regressions:
- the sign of the Hertz–Mindlin force;
- quaternion parsing from JSON arrays and objects;
- sphere–capsule contacts across a periodic boundary.

GitHub Actions builds the headless target from a fresh clone on Ubuntu and runs the tests on every push and pull request (`.github/workflows/ci.yml`). `scripts/compare_contact_models.py` runs one scene under several contact models and compares cost per step, contact counts, and final kinetic energy.

## Reproducing arXiv:2606.03952

```bash
make -C repro build manifest   # headless binary; 739-run manifest from the Table S1 protocol
make -C repro pilot            # verification groups and the static figures (~2 h)
make -C repro sweep            # main sweep for Fig. 4C,D and S3 (720 runs)
make -C repro figures          # figures/*.pdf from the finished runs
```

The runner is resume-safe. See [`repro/README.md`](repro/README.md) for the run groups and the external inputs. The packings in `assets/initial-configs/` come from [entanglement-optimization](https://github.com/yeonsu-jung/entanglement-optimization).

## Repository layout

| Path | Contents |
|---|---|
| `src/physics`, `include/physics` | Rigid bodies, contact detection, contact solvers, integrators, CUDA broad phase |
| `src/sim`, `include/sim` | Headless environment and entanglement observer |
| `src/app` | Command-line application and viewer |
| `src/python` | pybind11 module `rod_dynamics_py` |
| `src/tools` | Converters for trajectory and contact-network output; broad-phase benchmark |
| `assets/scenes` | Scene JSON files |
| `assets/initial-configs` | Relaxed rod packings used as initial conditions |
| `repro/` | Paper reproduction pipeline and figure scripts |
| `docs/` | Design notes on the contact models, friction, periodic boundaries, and CUDA |
| `tests/` | Smoke test and examples |

## Citation

```bibtex
@article{jung2026selfcaging,
  title   = {Emergent cohesion via self-caging in maximally entangled rod packings},
  author  = {Jung, Yeonsu and Mahadevan, L.},
  journal = {arXiv preprint arXiv:2606.03952},
  year    = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
