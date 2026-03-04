/**
 * @file cuda_broadphase.cu
 * @brief Naive O(N^2) capsule-capsule collision broadphase on GPU.
 *
 * Strategy
 * --------
 * Launch N*(N-1)/2 threads, one per capsule pair (i < j).  Each thread runs
 * the full Lumelsky closest-point algorithm (identical to the CPU path in
 * soft_contact.cpp) and writes to a compact output buffer via atomicAdd.
 *
 * No spatial hash, no sort – just flat parallelism.  For N >= ~256 rods the
 * GPU wins over the OpenMP spatial-hash path; for N=1000 expect ~10-30x
 * speedup over 16 CPU threads; for N=10 000 expect ~100x.
 *
 * Memory layout (SoA on device, the CPU AoS is repacked on upload)
 * ----------------------------------------------------------------
 *   d_pos   : float3[N_caps]  – capsule centres
 *   d_quat  : float4[N_caps]  – orientation quaternions (x,y,z,w)
 *   d_halfH : float[N_caps]   – half-lengths
 *   d_radii : float[N_caps]   – radii
 *   d_out   : GpuContactRaw[maxContacts]
 *   d_count : int             – atomic counter
 *
 * Persistent device buffers are kept in a file-static struct so the per-step
 * overhead is just two small H→D uploads (pos + quat; halfH/radii are static
 * per simulation) plus one D→H download of detected contacts.
 */

#include "physics/cuda_broadphase.hpp"
#include "physics/rigid_body.hpp"  // RigidBody, ShapeType

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

// ---------------------------------------------------------------------------
// Error checking helper
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "[CUDA] %s:%d  %s\n",                            \
                    __FILE__, __LINE__, cudaGetErrorString(_e));              \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Device helper: float3 arithmetic
// ---------------------------------------------------------------------------
__device__ __forceinline__ float3 op_add(float3 a, float3 b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}
__device__ __forceinline__ float3 op_sub(float3 a, float3 b) {
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}
__device__ __forceinline__ float3 op_scale(float3 a, float s) {
    return make_float3(a.x*s, a.y*s, a.z*s);
}
__device__ __forceinline__ float op_dot(float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
__device__ __forceinline__ float op_len(float3 a) {
    return sqrtf(op_dot(a, a));
}

// ---------------------------------------------------------------------------
// Device helper: capsule Y-axis from quaternion (matches glm::mat3_cast logic)
//
//   axisY = R(q) * (0,1,0)  =  column 1 of the rotation matrix
//   =  (2(qx*qy - qw*qz),  1-2(qx²+qz²),  2(qy*qz + qw*qx))
//
//   float4 stores quaternion as (x, y, z, w) – same order as glm::quat members.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float3 axisY_from_quat(float4 q) {
    float qx = q.x, qy = q.y, qz = q.z, qw = q.w;
    return make_float3(
        2.0f*(qx*qy - qw*qz),
        1.0f - 2.0f*(qx*qx + qz*qz),
        2.0f*(qy*qz + qw*qx)
    );
}

// ---------------------------------------------------------------------------
// Device helper: Lumelsky closest-point algorithm (same as CPU soft_contact.cpp)
//
// Finds parameters s∈[0,1], t∈[0,1] such that
//   c1 = p1 + s*(q1-p1)  and  c2 = p2 + t*(q2-p2)
// minimise ‖c1-c2‖.
// ---------------------------------------------------------------------------
__device__ void lumelsky_closest(
    float3 p1, float3 q1,
    float3 p2, float3 q2,
    float& s,  float& t)
{
    float3 e1  = op_sub(q1, p1);
    float3 e2  = op_sub(q2, p2);
    float3 e12 = op_sub(p2, p1);

    float D1 = op_dot(e1, e1);
    float D2 = op_dot(e2, e2);
    float S1 = op_dot(e1, e12);
    float S2 = op_dot(e2, e12);
    float R  = op_dot(e1, e2);
    float den = D1*D2 - R*R;

    float uf;

    if (fabsf(den) < 1e-12f) {
        // Parallel segments
        s  = 0.0f;
        t  = (D2 > 1e-12f) ? -S2 / D2 : 0.0f;
        uf = fminf(1.0f, fmaxf(0.0f, t));
        if (uf != t) {
            s = (D1 > 1e-12f) ? fminf(1.0f, fmaxf(0.0f, (uf*R + S1)/D1)) : 0.0f;
            t = uf;
        }
    } else {
        s  = fminf(1.0f, fmaxf(0.0f, (S1*D2 - S2*R) / den));
        t  = (D2 > 1e-12f) ? (s*R - S2) / D2 : 0.0f;
        uf = fminf(1.0f, fmaxf(0.0f, t));
        if (uf != t) {
            s = (D1 > 1e-12f) ? fminf(1.0f, fmaxf(0.0f, (uf*R + S1)/D1)) : s;
            t = uf;
        }
    }
}

// ---------------------------------------------------------------------------
// Main kernel: one thread per unique capsule pair (i < j)
// ---------------------------------------------------------------------------
__global__ void capsulePairsKernel(
    const float3* __restrict__ d_pos,
    const float4* __restrict__ d_quat,
    const float*  __restrict__ d_halfH,
    const float*  __restrict__ d_radii,
    int N,
    float act_margin,          // delta: extra activation range beyond r_a+r_b
    bool  pbc_enabled,
    float pbc_sx, float pbc_sy, float pbc_sz,  // PBC box extents
    GpuContactRaw* __restrict__ d_out,
    int*           __restrict__ d_count,
    int maxContacts)
{
    // Map this thread to pair index
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long npairs = (long long)N * (N - 1) / 2;
    if (tid >= npairs) return;

    // Invert upper-triangle index: find i such that i*(2N-i-1)/2 <= tid
    // Use double sqrt for precision with large N (N up to ~100k is fine)
    const double fn = (double)(2 * N - 1);
    int i = (int)floor((fn - sqrt(fn*fn - 8.0*(double)tid)) * 0.5);
    // Guard against FP rounding
    while ((long long)i * (2*N - i - 1) / 2 > tid) --i;
    while ((long long)(i+1) * (2*N - i - 2) / 2 <= tid) ++i;
    int j = (int)(tid - (long long)i * (2*N - i - 1) / 2) + i + 1;

    // Load body data
    const float3 pi = d_pos[i];
    const float3 pj = d_pos[j];
    const float4 qi = d_quat[i];
    const float4 qj = d_quat[j];
    const float hi = d_halfH[i];
    const float hj = d_halfH[j];
    const float ri = d_radii[i];
    const float rj = d_radii[j];

    // Capsule axes
    const float3 ai = axisY_from_quat(qi);
    const float3 aj = axisY_from_quat(qj);

    // Segment endpoints
    float3 A0 = op_sub(pi, op_scale(ai, hi));
    float3 A1 = op_add(pi, op_scale(ai, hi));
    float3 B0 = op_sub(pj, op_scale(aj, hj));
    float3 B1 = op_add(pj, op_scale(aj, hj));

    // Minimum-image PBC shift for body j
    float3 shift_b = make_float3(0.0f, 0.0f, 0.0f);
    if (pbc_enabled) {
        float3 delta = op_sub(pj, pi);
        if (pbc_sx > 0.0f) shift_b.x = -floorf(delta.x / pbc_sx + 0.5f) * pbc_sx;
        if (pbc_sy > 0.0f) shift_b.y = -floorf(delta.y / pbc_sy + 0.5f) * pbc_sy;
        if (pbc_sz > 0.0f) shift_b.z = -floorf(delta.z / pbc_sz + 0.5f) * pbc_sz;
        B0 = op_add(B0, shift_b);
        B1 = op_add(B1, shift_b);
    }

    // --- Broadphase AABB rejection (cheap early-out) ---
    // Check if centres are farther than the longest possible contact distance
    const float max_reach = hi + hj + ri + rj + act_margin;
    const float3 dp = op_sub(op_add(pj, shift_b), pi);
    if (fabsf(dp.x) > max_reach || fabsf(dp.y) > max_reach || fabsf(dp.z) > max_reach)
        return;

    // --- Narrowphase: Lumelsky closest points ---
    float s, t;
    lumelsky_closest(A0, A1, B0, B1, s, t);

    const float3 c1   = op_add(A0, op_scale(op_sub(A1, A0), s));
    const float3 c2   = op_add(B0, op_scale(op_sub(B1, B0), t));
    const float3 diff = op_sub(c2, c1);
    const float  dist = op_len(diff);

    const float surface_limit = ri + rj;
    if (dist >= surface_limit + act_margin) return;

    // --- Atomically claim a slot in the output buffer ---
    const int slot = atomicAdd(d_count, 1);
    if (slot >= maxContacts) {
        atomicSub(d_count, 1);   // keep count accurate (capped)
        return;
    }

    GpuContactRaw& out = d_out[slot];
    out.a = i;   out.b = j;
    out.px_a = c1.x;  out.py_a = c1.y;  out.pz_a = c1.z;
    out.px_b = c2.x;  out.py_b = c2.y;  out.pz_b = c2.z;
    // Normal from a → b
    if (dist > 1e-8f) {
        const float inv = 1.0f / dist;
        out.nx = diff.x*inv;  out.ny = diff.y*inv;  out.nz = diff.z*inv;
    } else {
        out.nx = 1.0f;  out.ny = 0.0f;  out.nz = 0.0f;
    }
    out.dist          = dist;
    out.surface_limit = surface_limit;
    out.s = s;   out.t = t;
    out.shift_bx = shift_b.x;
    out.shift_by = shift_b.y;
    out.shift_bz = shift_b.z;
}

// ---------------------------------------------------------------------------
// Persistent device buffers (avoid re-alloc every timestep)
// ---------------------------------------------------------------------------
namespace {

struct CudaState {
    float3*       d_pos     = nullptr;
    float4*       d_quat    = nullptr;
    float*        d_halfH   = nullptr;
    float*        d_radii   = nullptr;
    GpuContactRaw* d_out    = nullptr;
    int*          d_count   = nullptr;

    int n_alloc             = 0;  ///< bodies slots allocated
    int contacts_alloc      = 0;  ///< output slots allocated

    // Pinned host staging buffers for fast H→D transfers
    float3* h_pos_pin  = nullptr;
    float4* h_quat_pin = nullptr;
    int     n_pin      = 0;

    void ensure(int N, int maxContacts) {
        if (N > n_alloc) {
            CUDA_CHECK(cudaFree(d_pos));
            CUDA_CHECK(cudaFree(d_quat));
            CUDA_CHECK(cudaFree(d_halfH));
            CUDA_CHECK(cudaFree(d_radii));
            CUDA_CHECK(cudaMalloc(&d_pos,   N * sizeof(float3)));
            CUDA_CHECK(cudaMalloc(&d_quat,  N * sizeof(float4)));
            CUDA_CHECK(cudaMalloc(&d_halfH, N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_radii, N * sizeof(float)));
            if (!d_count) CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
            n_alloc = N;
        }
        if (maxContacts > contacts_alloc) {
            CUDA_CHECK(cudaFree(d_out));
            CUDA_CHECK(cudaMalloc(&d_out, maxContacts * sizeof(GpuContactRaw)));
            contacts_alloc = maxContacts;
        }
        if (N > n_pin) {
            CUDA_CHECK(cudaFreeHost(h_pos_pin));
            CUDA_CHECK(cudaFreeHost(h_quat_pin));
            CUDA_CHECK(cudaMallocHost(&h_pos_pin,  N * sizeof(float3)));
            CUDA_CHECK(cudaMallocHost(&h_quat_pin, N * sizeof(float4)));
            n_pin = N;
        }
    }

    ~CudaState() {
        cudaFree(d_pos);    cudaFree(d_quat);
        cudaFree(d_halfH);  cudaFree(d_radii);
        cudaFree(d_out);    cudaFree(d_count);
        cudaFreeHost(h_pos_pin);
        cudaFreeHost(h_quat_pin);
    }
};

static CudaState g_cs;

// halfH and radii are static per simulation run – upload only when resized
static int g_last_N       = -1;
static float* h_halfH_buf = nullptr;   // CPU buffer reused across calls
static float* h_radii_buf = nullptr;

} // anonymous namespace

// ---------------------------------------------------------------------------
// Host entry point
// ---------------------------------------------------------------------------
void cudaDetectCapsulePairsAll(
    const std::vector<RigidBody>& bodies,
    float activation_margin,
    bool  pbc_enabled,
    float pbc_sx, float pbc_sy, float pbc_sz,
    std::vector<GpuContactRaw>& out_raw)
{
    // ---- 1. Collect capsule data ----------------------------------------
    std::vector<int>   orig_idx;
    orig_idx.reserve(bodies.size());

    for (int k = 0; k < (int)bodies.size(); ++k) {
        if (bodies[k].type == ShapeType::Capsule)
            orig_idx.push_back(k);
    }

    const int N = (int)orig_idx.size();
    if (N < 2) return;

    // Output buffer: cap at a generous upper bound (~50 contacts per rod)
    const long long npairs      = (long long)N * (N-1) / 2;
    const int maxContacts = (int)std::min((long long)N * 50, npairs);

    g_cs.ensure(N, maxContacts);

    // ---- 2. Pack SoA and upload ------------------------------------------
    // Positions + quaternions: update every step (they move)
    for (int k = 0; k < N; ++k) {
        const RigidBody& b = bodies[orig_idx[k]];
        g_cs.h_pos_pin[k]  = {b.x.x, b.x.y, b.x.z};
        g_cs.h_quat_pin[k] = {b.q.x, b.q.y, b.q.z, b.q.w};
    }
    CUDA_CHECK(cudaMemcpy(g_cs.d_pos,  g_cs.h_pos_pin,  N*sizeof(float3),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_cs.d_quat, g_cs.h_quat_pin, N*sizeof(float4),
                          cudaMemcpyHostToDevice));

    // halfH and radii: upload only when body count changes (static geometry)
    if (N != g_last_N) {
        delete[] h_halfH_buf;
        delete[] h_radii_buf;
        h_halfH_buf = new float[N];
        h_radii_buf = new float[N];
        for (int k = 0; k < N; ++k) {
            const RigidBody& b = bodies[orig_idx[k]];
            h_halfH_buf[k] = b.cap.h;
            h_radii_buf[k] = b.cap.r;
        }
        CUDA_CHECK(cudaMemcpy(g_cs.d_halfH, h_halfH_buf, N*sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g_cs.d_radii, h_radii_buf, N*sizeof(float),
                              cudaMemcpyHostToDevice));
        g_last_N = N;
    }

    // ---- 3. Reset counter and launch kernel ------------------------------
    const int zero = 0;
    CUDA_CHECK(cudaMemcpy(g_cs.d_count, &zero, sizeof(int),
                          cudaMemcpyHostToDevice));

    constexpr int BLOCK = 256;
    const long long grid_ll = (npairs + BLOCK - 1) / BLOCK;
    const int grid = (int)std::min(grid_ll, (long long)INT_MAX);

    capsulePairsKernel<<<grid, BLOCK>>>(
        g_cs.d_pos, g_cs.d_quat, g_cs.d_halfH, g_cs.d_radii,
        N, activation_margin,
        pbc_enabled, pbc_sx, pbc_sy, pbc_sz,
        g_cs.d_out, g_cs.d_count, maxContacts);

    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- 4. Download results --------------------------------------------
    int h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, g_cs.d_count, sizeof(int),
                          cudaMemcpyDeviceToHost));

    if (h_count <= 0) return;

    if (h_count > maxContacts) {
        fprintf(stderr, "[CUDA broadphase] WARNING: overflow – %d contacts "
                        "truncated to %d. Increase N*50 limit.\n",
                h_count, maxContacts);
        h_count = maxContacts;
    }

    const size_t prev = out_raw.size();
    out_raw.resize(prev + h_count);
    CUDA_CHECK(cudaMemcpy(out_raw.data() + prev,
                          g_cs.d_out,
                          h_count * sizeof(GpuContactRaw),
                          cudaMemcpyDeviceToHost));

    // ---- 5. Remap compact capsule indices → original bodies[] indices ---
    for (size_t k = prev; k < out_raw.size(); ++k) {
        out_raw[k].a = orig_idx[out_raw[k].a];
        out_raw[k].b = orig_idx[out_raw[k].b];
    }
}
