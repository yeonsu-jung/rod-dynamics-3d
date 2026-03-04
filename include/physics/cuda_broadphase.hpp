/**
 * @file cuda_broadphase.hpp
 * @brief CUDA-accelerated naive O(N^2) capsule-capsule broadphase
 *
 * Each pair (i,j) is checked in parallel on GPU using the same Lumelsky
 * closest-point algorithm as the CPU narrowphase. For N rods this launches
 * N*(N-1)/2 threads simultaneously – dramatically faster than the serial
 * sort bottleneck and even the OpenMP path above ~512 rods.
 *
 * Build with:  cmake -DENABLE_CUDA=ON ...
 * Enable at runtime: set use_cuda=true in SoftContactCfg (or JSON config).
 */

#pragma once

#include <vector>

struct RigidBody;
struct ContactPrimitive;

/**
 * @brief Raw capsule-capsule contact returned from GPU.
 *
 * Plain-old-data so it can live in a flat device/host buffer.
 * Indices a/b are remapped to the original bodies[] indices by the host
 * wrapper before handing back to the solver.
 */
struct GpuContactRaw {
    int   a, b;             ///< original body indices
    float px_a, py_a, pz_a; ///< closest point on body a (world space)
    float px_b, py_b, pz_b; ///< closest point on body b (world space)
    float nx,  ny,  nz;     ///< contact normal (from a → b)
    float dist;             ///< centre-to-centre distance
    float surface_limit;    ///< r_a + r_b
    float s, t;             ///< Lumelsky params on [0,1] for type classification
    float shift_bx, shift_by, shift_bz; ///< PBC minimum-image shift applied to b
};

#ifdef USE_CUDA

/**
 * @brief Detect all capsule-capsule contacts using a naive GPU broadphase.
 *
 * Extracts capsule-only bodies, uploads minimal SoA data to GPU, launches
 * N*(N-1)/2 threads (one per pair), and downloads the detected contacts.
 *
 * Non-capsule bodies (Box floors, Spheres) are silently skipped; call the
 * existing CPU paths for those.
 *
 * @param bodies           Full bodies vector (may contain non-capsules)
 * @param activation_margin  Extra range: contact detected if dist < (r_a+r_b)+margin
 * @param pbc_enabled      Periodic boundary conditions flag
 * @param pbc_sx/sy/sz     PBC box dimensions (ignored if !pbc_enabled)
 * @param out_raw          Output: detected contacts appended here (not cleared)
 */
void cudaDetectCapsulePairsAll(
    const std::vector<RigidBody>& bodies,
    float activation_margin,
    bool  pbc_enabled,
    float pbc_sx, float pbc_sy, float pbc_sz,
    std::vector<GpuContactRaw>& out_raw);

#endif // USE_CUDA
