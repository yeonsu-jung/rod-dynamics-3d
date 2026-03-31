# Thermal RandomInit — Cluster Deployment Instructions

## What was done (local, Mar 31 2026)

Two commits pushed to `main` on GitHub:

1. **C++ thermal randomInit** (`e6d4eaa`): Three-mode `randomInit` in the engine:
   - `"thermal"`: kBT-based equipartition. σ_v = √(kBT/m), σ_ω = √(kBT/I⊥). One parameter.
   - `"gaussian"`: independent σ_v / σ_ω Gaussians.
   - `"uniform"`: legacy (vSigma uniform range + fixed wSpeed on S²).
   - New fields in `RandomInitCfg`: `mode`, `wSigma`, `kBT`, `projectParallelSpin`.
   - Files: `include/config/config.hpp`, `src/config/config.cpp`, `src/app/main.cpp`, `assets/scenes/reptation.json`.

2. **Python `--sigma-v` flag** (`891ef68`): `submit_entangled.py` now accepts `--sigma-v <float>`:
   - Computes `kBT = ρ π r² L × sigma_v²` per-AR (so all rods get the same velocity scale regardless of diameter).
   - Writes `"mode": "thermal"` + kBT into the scene JSON.
   - Mutually exclusive with legacy `--init-velocity-sigma` / `--w-speed`.
   - New script: `parametric_study/iter_submit_n_nsc_thermal.sh` (sigma_v=0.1, mu=1.0, N200).

## What needs to happen on the cluster

```bash
cd /n/home01/yjung/Github/rod-dynamics-3d
git pull origin main
cd build_head && make -j8 rigidbody_viewer_3d && cd ..
```

Then run:
```bash
bash parametric_study/iter_submit_n_nsc_thermal.sh
```

Or dry-run first:
```bash
DRY_RUN=true bash parametric_study/iter_submit_n_nsc_thermal.sh
```

## Key physics notes

- **Diameter does NOT affect kinematics** in the slender limit (d ≪ L). Mass m cancels in σ_ω/σ_v = √(m/I⊥) = √(12)/L. The d-dependent correction is O(d²/L²) ~ 10⁻⁵ for AR≥100.
- For σ_v = 0.1 L (L=1): σ_ω ≈ 0.35 rad/time. kBT depends on AR through mass.
- `projectParallelSpin: true` by default — ω∥ (rod-axis spin) is zeroed in thermal mode, giving clean 5-DOF dynamics on R³×S².

## Customization

Edit `iter_submit_n_nsc_thermal.sh` to change:
- `SIGMA_V=0.1` — translational velocity scale
- `FRICTIONS="1.0"` — friction coefficients (comma-separated for sweep)
- `LIMIT=5` — number of seed folders
- `STEPS=200000` — simulation steps
- Output goes to: `/n/holylabs/.../runs/test_nsc_thermal/`

## Code changes summary

### RandomInitCfg (include/config/config.hpp)
```cpp
struct RandomInitCfg {
    bool enabled = false;
    std::string mode{"thermal"};  // "thermal", "gaussian", or "uniform"
    float vSigma = 0.3f;         // (uniform/gaussian) linear velocity sigma
    float wSpeed = 1.5f;         // (uniform) fixed angular speed
    float wSigma = 0.5f;         // (gaussian) angular velocity sigma
    float kBT = 1.0f;            // (thermal) effective temperature
    unsigned int seed = 0;
    bool projectParallelSpin = true;  // zero out rod-axis ω component
};
```

### Thermal mode logic (src/app/main.cpp)
- σ_v = √(kBT / mass)
- σ_ω = √(kBT / I_perp) where I_perp = I_body[0][0]
- 3 Gaussian components for v, 3 Gaussian components for ω
- If projectParallelSpin: subtract ω∥ = (ω·û)û where û = q × [0,1,0]

### submit_entangled.py --sigma-v
- Computes per-AR kBT: `kBT = density * pi * (diameter/2)^2 * rod_length * sigma_v^2`
- This ensures all AR values get the same σ_v regardless of rod diameter
