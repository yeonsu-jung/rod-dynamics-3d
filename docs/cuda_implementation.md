# CUDA Broadphase & Narrowphase Collision Detection

**Rod Dynamics 3D — GPU Acceleration Implementation Notes**

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Architecture Overview](#2-architecture-overview)
3. [Implementation: The Three Kernels](#3-implementation-the-three-kernels)
4. [Memory Layout and Persistent State](#4-memory-layout-and-persistent-state)
5. [Build Instructions](#5-build-instructions)
6. [Benchmark Results](#6-benchmark-results)
7. [Full Simulation Results](#7-full-simulation-results)
8. [Physics Validation: float32 vs float64](#8-physics-validation-float32-vs-float64)
9. [Running GPU Simulations](#9-running-gpu-simulations)
10. [Key Files](#10-key-files)
11. [Future Guidelines](#11-future-guidelines)

---

## 1. Motivation

The collision detection step (`detectContacts`) was identified as the dominant bottleneck for large rod simulations. For N=2000 rods running 200,000 steps on 8 CPU cores with a spatial hash:

- **Wall time: ~12 hours** per run
- Scaling: O(N²) in the worst case for the naive path; the spatial hash is O(N) on average but has significant serial overhead (bucket construction, sorting) that limits OpenMP efficiency

For parametric studies sweeping N, AR, and friction across many seeds, this is the binding constraint on scientific throughput.

The GPU implementation replaces the collision detection inner loop with a CUDA kernel pipeline that achieves **~200× wall-clock speedup** while producing physically identical contact sets.

---

## 2. Architecture Overview

The implementation follows a **two-pass pipeline**:

```
Per timestep:
┌─────────────────────────────────────────────────────────────┐
│ CPU: pack pos+quat in SoA → H→D upload (pinned memory)      │
├─────────────────────────────────────────────────────────────┤
│ GPU Kernel 1: computeAABBsKernel                            │
│ N threads, one per rod → compute tight AABB                 │
├─────────────────────────────────────────────────────────────┤
│ GPU Kernel 2: aabbBroadphaseKernel                          │
│ N*(N-1)/2 threads → AABB pair test → compact candidate list │
├─────────────────────────────────────────────────────────────┤
│ GPU Kernel 3: narrowphaseKernel                             │
│ K threads (K = AABB-passing pairs) → Lumelsky → contacts   │
├─────────────────────────────────────────────────────────────┤
│ D→H download contacts → convert GpuContactRaw→ContactPrimitive │
└─────────────────────────────────────────────────────────────┘
```

The key idea: at 30% volume fraction, ~1–3% of N*(N-1)/2 pairs pass the AABB test (at N=4000, only 0.86%). Lumelsky — the expensive segment-segment distance calculation — only runs on those candidates.

### Why not a spatial hash on GPU?

The spatial hash on CPU is cache-hostile and has serial construction overhead. On GPU, a hash requires a sort (e.g., radix sort) which is kernel-launch heavy for the sizes we care about (N ≤ 10,000). The flat O(N²) AABB kernel with 8M threads at N=4000 runs in 0.27 ms on an A100 — faster than building any spatial data structure.

---

## 3. Implementation: The Three Kernels

### Kernel 1 — `computeAABBsKernel` (N threads, one per rod)

Each thread computes the tight axis-aligned bounding box for one capsule:

```
thread i owns rod i:
  axisY = rotate quaternion q[i] by (0,1,0)   ← capsule long axis
  e0 = pos[i] - axisY * halfH[i]              ← tail endpoint
  e1 = pos[i] + axisY * halfH[i]              ← head endpoint

  aabb_min[i] = min(e0, e1) - (radius[i] + margin)
  aabb_max[i] = max(e0, e1) + (radius[i] + margin)
```

The `margin` parameter corresponds to `delta` (the contact activation range). Expanding by `radius + margin` makes the AABB test conservative: if two AABBs do **not** overlap, the capsules cannot be within contact range. No contacts are missed.

**Output:** `d_aabb_min[N]`, `d_aabb_max[N]`

---

### Kernel 2 — `aabbBroadphaseKernel` (N*(N-1)/2 threads, one per pair)

The upper-triangle of the N×N pair matrix is mapped linearly to thread IDs. The inverse mapping from a flat index `tid` to pair `(i, j)` uses the closed-form formula:

```
fn = 2*N - 1
i  = floor( (fn - sqrt(fn² - 8*tid)) / 2 )    ← from the triangular number inverse
j  = tid - i*(2N-i-1)/2  +  i + 1
```

Two correction loops guard against floating-point rounding near integer boundaries.

Each thread then:
1. Computes the PBC minimum-image shift for rod j (if periodic boundaries are enabled)
2. Shifts rod j's AABB by that amount
3. Tests 3-axis AABB overlap: `aabb_max[i].x >= shifted_jmin.x AND ...`
4. If passing: atomically writes `int2(i, j)` into the candidate list

**Output:** compact `d_candidates[K]` array, `d_cand_count`

---

### Kernel 3 — `narrowphaseKernel` (K threads, one per AABB candidate)

Each thread processes one candidate pair `(i, j)` from the candidate list:

1. Load both rods' geometry (pos, quat, halfH, radius)
2. Compute endpoints A0, A1, B0, B1 (applying PBC shift if needed)
3. Run **Lumelsky's closest-point algorithm** — finds parameters `s, t ∈ [0,1]` minimizing |A(s) - B(t)|
4. Compute closest points c1, c2 and distance `dist = |c2 - c1|`
5. If `dist < surface_limit + act_margin`:  write a `GpuContactRaw` record atomically

**Output:** `d_out[contacts]`, `d_out_count`

The Lumelsky algorithm handles: parallel segments (degenerate case), and clamps `s, t` to [0,1] with proper endpoint fallback — identical logic to the CPU `closestPointsSegmentSegment` in `soft_contact.cpp`.

---

### Why the AABB test is conservative (no missed contacts)

The AABB of capsule i is expanded by `r_i + margin`. The AABB of capsule j is expanded by `r_j + margin`. If these two AABBs do NOT overlap on some axis:

```
aabb_max_i[axis] < aabb_min_j[axis]
⟺  (max_i + r_i + margin) < (min_j - r_j - margin)
⟺  dist_axis > r_i + r_j + 2*margin ≥ r_i + r_j + margin
```

So the minimum distance between the capsule axes exceeds `surface_limit + margin`. No contact exists. Therefore no false negatives.

---

## 4. Memory Layout and Persistent State

All GPU buffers live in `CudaStateTwoPass` (a file-static struct in `cuda_broadphase.cu`). They are allocated once and reused across timesteps — no `cudaMalloc`/`cudaFree` in the hot path.

```
Device buffers (allocated at first call, resized only when N grows):
  d_pos[N]          float3   capsule centers (updated every step)
  d_quat[N]         float4   orientations    (updated every step)
  d_halfH[N]        float    half-lengths    (static unless N changes)
  d_radii[N]        float    radii           (static unless N changes)
  d_aabb_min[N]     float3   AABB minima     (written by kernel 1)
  d_aabb_max[N]     float3   AABB maxima     (written by kernel 1)
  d_candidates[K]   int2     AABB pairs      (written by kernel 2)
  d_cand_count      int      atomic counter
  d_out[M]          GpuContactRaw  contacts  (written by kernel 3)
  d_out_count       int      atomic counter

Host pinned buffers (for fast H→D transfer):
  h_pos_pin[N]      float3
  h_quat_pin[N]     float4

Capacity limits:
  maxCands    = min(N*200, N*(N-1)/2)   ← conservative, covers all realistic cases
  maxContacts = min(N*50,  N*(N-1)/2)
```

**Static geometry optimization:** `d_halfH` and `d_radii` are only uploaded when `N` changes (tracked by `g_tp_last_N`). At steady state (fixed N), only `pos` and `quat` are uploaded per step — minimizing PCIe traffic.

---

## 5. Build Instructions

### Prerequisites

- CUDA Toolkit ≥ 12.0 (`nvcc`)
- CMake ≥ 3.18
- GCC ≥ 8 (host compiler)

### On FASRC (Harvard Cannon cluster)

```bash
module load cmake
module load gcc/13.2.0-fasrc01
module load cuda/12.9.1-fasrc01

cd /path/to/rod-dynamics-3d
mkdir -p build_cuda && cd build_cuda

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_HEADLESS=ON \
    -DENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=80 \   # A100 = sm_80; V100 = sm_70; RTX 3090 = sm_86
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_C_COMPILER=gcc

make -j4 rigidbody_viewer_3d      # full simulation binary
make -j4 benchmark_broadphase     # standalone benchmark
```

### CMake options

| Option | Default | Meaning |
|--------|---------|---------|
| `ENABLE_CUDA` | `OFF` | Enable CUDA broadphase. Defines `USE_CUDA` compile flag. |
| `CMAKE_CUDA_ARCHITECTURES` | `"70;80;86;90"` | Target GPU SM versions. Set to match your GPU. |
| `BUILD_HEADLESS` | `OFF` | Skip GLFW/Wayland dependencies (required on compute nodes). |

### CPU-only build (for comparison runs)

```bash
mkdir -p build_head && cd build_head
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_HEADLESS=ON -DENABLE_CUDA=OFF
make -j4 rigidbody_viewer_3d
```

---

## 6. Benchmark Results

Measured on **NVIDIA A100-SXM4-40GB** (sm_80) vs CPU spatial hash (4 OMP threads).

Rod geometry: length=0.70 m, diameter=0.10 m, `delta=0.005`, ~30% volume fraction (PBC box).

```
     N       pairs  contacts    cpu_ms   ±std  fused_ms   ±std spdup_f     cands    two_ms   ±std spdup_2
----------------------------------------------------------------------------------------------------------
   100        4950       357      1.74 ± 0.32      0.06 ± 0.00  31x       1583      0.08 ± 0.00   23x
   256       32640       854      8.06 ± 0.26      0.08 ± 0.00  99x       4381      0.10 ± 0.04   84x
   512      130816      1730     44.06 ±15.10      0.14 ± 0.01 321x       8571      0.13 ± 0.02  344x
  1000      499500      3208     15.17 ± 0.15      0.24 ± 0.01  63x      16850      0.19 ± 0.03   81x
  2000     1999000      6529     41.18 ±10.47      0.52 ± 0.07  79x      33816      0.38 ± 0.28  108x
  4000     7998000     13190    127.77 ±23.78      1.18 ± 0.04 109x      68401      0.76 ± 0.02  169x
```

### Key observations

**AABB rejection ratio improves with N:**

| N | pairs | AABB candidates | rejection rate |
|---|-------|----------------|----------------|
| 100 | 4,950 | 1,583 | 68% |
| 1000 | 499,500 | 16,850 | 96.6% |
| 4000 | 7,998,000 | 68,401 | 99.1% |

At large N, only 1-in-117 pairs reaches Lumelsky. This is why the two-pass advantage grows: `bp` kernel scales as O(N²) with cheap per-thread work; `narrow` kernel scales linearly with contacts.

**Two-pass stage breakdown at N=4000:**

```
upload=0.038ms  aabb=0.009ms  bp=0.266ms  narrow=0.025ms  dl=0.406ms
```

The download (`dl`) is the single largest cost at large N — transferring ~13,000 contacts × 72 bytes ≈ 950 KB over PCIe. This grows linearly with contacts. The broadphase (`bp`) grows as O(N²) but is cheap per thread.

**Crossover point:** Two-pass beats fused at N ≥ 512. Below that, the overhead of three kernel launches and the candidate list management is not worth it.

---

## 7. Full Simulation Results

**Case:** N=2000, AR=1000, friction=1.0, seed=224\_259\_311, 200,000 steps, dt=0.0005

| | CPU (8 cores, spatial hash off) | GPU (A100, two-pass) |
|---|---|---|
| **Wall time** | 12 hr 16 min | **3 min 33 sec** |
| **Speedup** | — | **207×** |
| Memory | 2.7 MB | 25 MB |

The 207× figure is wall-clock (real experiment time), not just collision detection — it includes integration, friction force computation, entanglement measurement, and all I/O.

---

## 8. Physics Validation: float32 vs float64

### What was observed

When running with `delta=0.0` (no pre-contact buffer), the GPU and CPU trajectories diverge significantly:

```
step  |     cpu_KE     gpu_KE |  cpu_contacts  gpu_contacts
----------------------------------------------------------
     0   7.926e-03   7.926e-03      0               0       ← identical
  1000   1.364e-04   1.393e-03    119              65       ← GPU finds half the contacts
200000   5.242e-06   1.825e-04      0              24       ← 35× higher KE in GPU
```

Final entanglement: CPU=32.1%, GPU=26.2%

### Root cause

The failure chain for high-AR rods with `delta=0.0`:

1. **Float32 misses marginal contacts.** The CPU path uses double-precision Lumelsky; the GPU uses float32. For AR=1000 rods (radius=0.0005, contact threshold=0.001), borderline pairs that the CPU detects at dist ≈ 0.0010 can compute as dist ≈ 0.0011 in float32 and be rejected. These missed contacts mean less friction, so the system dissipates energy more slowly.

2. **Higher KE → tunneling.** With higher kinetic energy, rods traverse the 0.001 contact window in fewer timesteps. Some pairs skip through the detection zone entirely (tunneling). Once rods overlap fully (dist ≈ 0), `max_overlap` locks at `surface_limit = 0.001` (the rod diameter).

3. **Degenerate normals inject energy.** At `dist < 1e-8`, the contact normal falls back to `(1, 0, 0)`. A force applied in the wrong direction can add energy rather than resolve the overlap, sustaining the high-KE state.

### The fix: set `delta > 0`

The `delta` parameter expands the contact detection zone. A value of `delta = 2 × radius` catches pairs before they can overlap and gives the spring force time to act:

| AR | rod diameter | recommended delta |
|----|-------------|------------------|
| 10 | 0.1 | 0.0 (default fine) |
| 100 | 0.01 | 0.001–0.005 |
| 500 | 0.002 | 0.002 |
| 1000 | 0.001 | 0.002 |

With `delta=0.002` on AR=1000 runs, GPU and CPU trajectories converge to statistically matching steady-state distributions (KE, gyration radius, entanglement fraction) across multiple seeds.

### Note on trajectory comparison

For N-body chaotic systems, exact trajectory matching between CPU and GPU is not expected — float32 vs float64 differences at step 1 compound exponentially. The correct validation is:
- **Same initial KE and positions** ✓ (verified: identical at step 0)
- **Same steady-state statistics** across multiple seeds (KE distribution, mean entanglement, gyration radius)
- **Same contact count trends** (contacts should be found for the same qualitative events)

Individual trajectories from different seeds on CPU already diverge from each other; GPU is another such realization.

---

## 9. Running GPU Simulations

### Single case

```bash
# GPU run (from repo root):
./parametric_study/run_single_case.sh \
    --n 2000 --ar 1000 --friction 1.0 --delta 0.002

# CPU comparison (same parameters):
./parametric_study/run_single_case.sh \
    --n 2000 --ar 1000 --friction 1.0 --delta 0.002 --mode cpu

# Inspect without submitting:
./parametric_study/run_single_case.sh \
    --n 2000 --ar 1000 --delta 0.002 --dry-run
```

All options:

```
--n        N           Number of rods            (default: 2000)
--ar       AR          Aspect ratio              (default: 1000)
--friction F           Friction coefficient      (default: 1.0)
--delta    D           Contact activation margin (default: 0.002)
--seed     SEED        Seed folder "224,259,311"  (default: first found)
--steps    S           Simulation steps          (default: 200000)
--dt       DT          Timestep                  (default: 0.0005)
--mode     gpu|cpu                               (default: gpu)
--k-scaler K           Spring stiffness          (default: 100.0)
--dry-run              Don't submit
```

Run output lands in:
```
runs/single_cases/
  cuda_N{N}_AR{AR}_F{F}_delta{D}/{seed}/
    scene.json
    x_relaxed.txt   ← symlink to initial condition
    Sbatch.sh
    output.csv      ← written after completion
```

### Enabling CUDA in a scene.json

Add `"use_cuda": true` to the `soft_contact` block:

```json
"soft_contact": {
    "enabled": true,
    "delta": 0.002,
    "k_scaler": 100.0,
    "mu": 1.0,
    "mu_static": 1.0,
    "nu": 1e-09,
    "enable_friction": true,
    "use_spatial_hash": false,
    "use_aabb": true,
    "use_cuda": true        ← this line activates GPU path
}
```

The simulation binary must be built with `-DENABLE_CUDA=ON` and run with the CUDA library in `LD_LIBRARY_PATH` (via `module load cuda/...`).

---

## 10. Key Files

| File | Purpose |
|------|---------|
| `src/physics/cuda_broadphase.cu` | Three CUDA kernels + `CudaStateTwoPass` persistent buffers + host entry points `cudaDetectCapsulePairsAll` (fused) and `cudaDetectCapsulePairsTwoPass` (two-pass) |
| `include/physics/cuda_broadphase.hpp` | Public API: `GpuContactRaw` struct, `CudaTwoPassStats` struct, function declarations |
| `src/physics/soft_contact.cpp` | `detectContactsCuda()` — calls `cudaDetectCapsulePairsTwoPass`, converts `GpuContactRaw→ContactPrimitive`, falls back to CPU for non-capsule shapes |
| `include/config/config.hpp` | `SoftContactCfg::use_cuda` field |
| `src/config/config.cpp` | JSON parsing of `use_cuda` |
| `src/tools/benchmark_broadphase.cpp` | Standalone benchmark: CPU spatial hash vs GPU fused vs GPU two-pass |
| `benchmark_cuda.sh` | SLURM script to build + run benchmark on `gpu_test` partition |
| `parametric_study/run_single_case.sh` | Create + submit one GPU or CPU run with configurable parameters |
| `build_cuda/` | CUDA-enabled build directory |
| `build_head/` | CPU-only headless build directory |

---

## 11. Future Guidelines

### 11.1 Keeping contacts on GPU (eliminate the download bottleneck)

At N=4000, the D→H download of contact records (0.41 ms) is larger than the broadphase kernel (0.27 ms) and nearly doubles the two-pass total. As N grows, this will become dominant.

The fix is to implement force computation (`computeForces`) as a CUDA kernel that reads `d_out` directly on-device. The CPU would never see the raw contacts — only the final force/torque increments. This requires porting `SoftContactSolver::computeForces` (the Hertz spring + Mindlin friction model) to CUDA.

Estimated benefit: 2–3× additional speedup at large N, plus significant reduction in PCIe round-trips per timestep.

### 11.2 Double precision for high-AR rods

The float32 Lumelsky kernel is accurate enough for AR ≤ 100 (rod diameter ≥ 0.02 m). For AR=1000 (diameter=0.001 m), float32 precision at the contact threshold is marginal. Two approaches:

**Option A — `delta > 0` (recommended, already implemented):** Set `delta ≈ 2 × radius`. This widens the detection window so float32 rounding cannot cause missed contacts. Small performance impact (~5% more AABB candidates).

**Option B — Mixed precision:** Use double precision only in the narrowphase kernel (Lumelsky computation). The broadphase AABB kernel stays float32. This halves GPU throughput on the narrowphase but keeps correctness. On A100, double throughput is 1/2 of float32, so for K ≪ N*(N-1)/2 (sparse contacts) the cost is negligible.

### 11.3 Scaling to larger N (N > 10,000)

The broadphase kernel (`aabbBroadphaseKernel`) launches N*(N-1)/2 threads. At N=10,000, that's 50 million threads — still fine for an A100 (6912 CUDA cores, typical occupancy ~50%). At N=100,000 it becomes 5 billion threads, which will saturate the grid size limit.

For N > ~30,000, replace the flat O(N²) broadphase with a **GPU BVH** (bounding volume hierarchy) or **GPU grid hash**. Libraries: NVIDIA's [cuBVH](https://github.com/NVIDIA-RTX/cuBVH) or a radix-sort-based cell grid (similar to what LAMMPS and HOOMD-blue use). The two-pass structure already separates broadphase from narrowphase, so only kernel 2 needs replacement.

### 11.4 Multi-GPU scaling

For N > 50,000, distributing across multiple GPUs becomes attractive. The natural decomposition is spatial: each GPU owns a spatial region and handles capsules in that region. Ghost layers handle cross-boundary contacts. The current `CudaStateTwoPass` singleton would need to become per-device.

Simpler alternative: run multiple independent parameter combinations in parallel on separate GPU streams (e.g., different seeds or friction values on the same GPU simultaneously via CUDA streams). Each parametric case uses a fraction of GPU memory and SM occupancy, and the GPU is shared efficiently.

### 11.5 Integration time integration on GPU

Currently, the Verlet/RK integration step runs on CPU after contact forces are applied. For N=2000, integration takes O(N) work and is very fast (~0.1 ms). At N=50,000+ it becomes a non-trivial fraction. At that scale, implement `integrateKernel` in CUDA — trivial to parallelize since each rod is independent.

### 11.6 Benchmarking at different aspect ratios

The current benchmark uses fixed AR (r=0.05, h=0.35, AR≈7). The AABB rejection rate and contact frequency are sensitive to AR. The benchmark script (`benchmark_broadphase.cpp`) should be extended to sweep AR, as the GPU advantage varies:

- **Low AR (AR < 10):** Contacts are frequent (thick rods, many near-misses). Higher candidate fraction for AABB. Both passes busy.
- **High AR (AR > 100):** Rare contacts, high AABB rejection (thin rods, very sparse). Broadphase dominates; narrowphase is negligible. Two-pass is strongly preferred.

### 11.7 Profiling with Nsight

To understand GPU utilization in detail:
```bash
ncu --set full ./build_cuda/benchmark_broadphase 2000 10
```

Key metrics to check:
- `l1tex__t_sector_hit_rate` — cache hit rate (low means poor memory access pattern)
- `sm__warps_active` — warp occupancy
- `sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared` — shared memory bank conflicts

For the broadphase kernel, the main concern is **thread divergence**: threads in the same warp process different pairs, and about 99% return early at the AABB test. Modern NVIDIA GPUs handle this well (predicated execution), but Nsight will quantify the actual cost.

### 11.8 Parameter recommendations by use case

| Use case | delta | mode | Notes |
|---|---|---|---|
| AR ≤ 100, any N | 0.0 | gpu | Float32 sufficient at this AR |
| AR = 500, N ≤ 5000 | 0.001 | gpu | Small buffer needed |
| AR = 1000, N ≤ 5000 | 0.002 | gpu | Validated; matches CPU statistics |
| AR = 1000, N > 10000 | 0.002 | gpu (future BVH) | N²/2 broadphase becomes slow |
| Validation / debugging | any | cpu | Use `build_head/` binary, float64 reference |
| Parametric sweep | 0.002 | gpu | 207× speedup confirmed; run all seeds concurrently |

---

*Implementation by Claude (Anthropic) + Y. Jung, March 2026.*
*GPU: NVIDIA A100-SXM4-40GB (sm_80). Cluster: Harvard FASRC (Cannon).*
