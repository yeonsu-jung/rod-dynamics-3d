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

/**
 * @brief Per-stage timing from the two-pass pipeline.
 */
struct CudaTwoPassStats {
    double upload_ms      = 0.0; ///< H→D data transfer
    double aabb_ms        = 0.0; ///< per-body AABB kernel
    double broadphase_ms  = 0.0; ///< AABB pair-test kernel
    double narrowphase_ms = 0.0; ///< Lumelsky narrowphase kernel
    double download_ms    = 0.0; ///< D→H result transfer
    int    candidates     = 0;   ///< AABB-passing pairs
    int    contacts       = 0;   ///< actual contacts found
};

/**
 * @brief Two-pass GPU collision: AABB broadphase then exact Lumelsky narrowphase.
 *
 * Pass 1 – AABB kernel (N threads): compute a tight per-body axis-aligned
 * bounding box; expand by radius + activation_margin.
 *
 * Pass 2 – pair-test kernel (N*(N-1)/2 threads): test every AABB pair (with
 * PBC minimum-image shift); write passing pairs into a compact int2 list.
 *
 * Pass 3 – narrowphase kernel (K threads, one per AABB candidate): run the
 * full Lumelsky segment-segment algorithm and write GpuContactRaw records.
 *
 * @param bodies             Full bodies vector
 * @param activation_margin  Extra detection range beyond r_a+r_b
 * @param pbc_enabled        PBC flag
 * @param pbc_sx/sy/sz       PBC box extents
 * @param out_raw            Output contacts (appended, not cleared)
 * @param stats              Optional timing/count output (pass nullptr to skip)
 */
void cudaDetectCapsulePairsTwoPass(
    const std::vector<RigidBody>& bodies,
    float activation_margin,
    bool  pbc_enabled,
    float pbc_sx, float pbc_sy, float pbc_sz,
    std::vector<GpuContactRaw>& out_raw,
    CudaTwoPassStats* stats = nullptr);

#endif // USE_CUDA
