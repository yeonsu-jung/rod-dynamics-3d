/**
 * @file benchmark_broadphase.cpp
 * @brief Benchmark: CPU spatial-hash vs GPU fused O(N²) vs GPU two-pass AABB+Lumelsky
 *
 * Usage:
 *   ./benchmark_broadphase [N] [iters]
 *   N     – number of rods  (default: swept: 100 256 512 1000 2000 4000)
 *   iters – timed iterations (default: 50)
 *
 * How it generates bodies
 * -----------------------
 * N rods are placed randomly inside a cubic PBC box whose side length is
 * chosen to give ~30 % volume fraction – matching a typical dense simulation.
 *   L = cbrt(N * V_rod / 0.30)    V_rod ≈ π r² (2h) + 4/3 π r³
 *
 * Rod geometry (default pbc_1000_rods scene):
 *   length 0.70 m, diameter 0.10 m  (radius = 0.05, half-height = 0.35)
 *
 * Output columns
 * --------------
 * CPU path:
 *   N  pairs  contacts  cpu_ms ±std
 *
 * GPU fused (USE_CUDA only):
 *   fused_ms ±std  speedup_f
 *
 * GPU two-pass (USE_CUDA only):
 *   cands  two_ms ±std  speedup_2
 *   Per-stage breakdown:  upload | aabb | bp | narrow | dl  (mean over iters)
 */

#include "physics/rigid_body.hpp"
#include "physics/soft_contact.hpp"
#include "config/config.hpp"
#ifdef USE_CUDA
#include "physics/cuda_broadphase.hpp"
#endif

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// physics library references this global (defined in main.cpp in the app).
// 0 = use all available threads.
int g_thread_limit = 0;

// ---------------------------------------------------------------------------
// Random rod generator
// ---------------------------------------------------------------------------
static std::vector<RigidBody> makeRods(int N, float radius, float halfH,
                                       float boxL, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> pos(-boxL * 0.5f, boxL * 0.5f);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    std::vector<RigidBody> bodies;
    bodies.reserve(N);

    for (int k = 0; k < N; ++k) {
        glm::vec3 p(pos(rng), pos(rng), pos(rng));

        // Uniform quaternion (Shoemake 1992)
        float u1 = uni(rng), u2 = uni(rng), u3 = uni(rng);
        float s1  = std::sqrt(1.0f - u1), s2 = std::sqrt(u1);
        float t1  = 2.0f * static_cast<float>(M_PI) * u2;
        float t2  = 2.0f * static_cast<float>(M_PI) * u3;
        glm::quat q(s1 * std::sin(t1), s1 * std::cos(t1),
                    s2 * std::sin(t2), s2 * std::cos(t2));
        q = glm::normalize(q);

        bodies.push_back(RigidBody::makeCapsule(p, q, 1000.0f, radius, halfH));
    }
    return bodies;
}

// ---------------------------------------------------------------------------
// Timer helper
// ---------------------------------------------------------------------------
struct Stats {
    double mean    = 0.0;
    double std_dev = 0.0;
};

static Stats summarise(const std::vector<double>& samples) {
    if (samples.empty()) return {};
    double sum  = std::accumulate(samples.begin(), samples.end(), 0.0);
    double mean = sum / samples.size();
    double sq   = 0.0;
    for (double v : samples) sq += (v - mean) * (v - mean);
    return {mean, std::sqrt(sq / samples.size())};
}

// ---------------------------------------------------------------------------
// Benchmark a single N
// ---------------------------------------------------------------------------
static void benchmarkN(int N, int iters) {
    constexpr float radius = 0.05f;
    constexpr float halfH  = 0.35f;

    const float vrod = static_cast<float>(M_PI) * radius * radius * 2.0f * halfH
                     + (4.0f / 3.0f) * static_cast<float>(M_PI) * radius * radius * radius;
    const float boxL = std::cbrt(static_cast<float>(N) * vrod / 0.30f);

    auto bodies = makeRods(N, radius, halfH, boxL);

    SoftContactCfg cfg;
    cfg.enabled       = true;
    cfg.delta         = 0.005;
    cfg.k_scaler      = 1000.0f;
    cfg.use_aabb      = true;

    SoftContactSolver solver(cfg);
    solver.setPBC(true,
                  glm::vec3(-boxL * 0.5f),
                  glm::vec3( boxL * 0.5f));

    using Clock = std::chrono::high_resolution_clock;

    // ---- CPU: spatial hash ------------------------------------
    cfg.use_spatial_hash = true;
    cfg.use_cuda         = false;
    solver.setConfig(cfg);

    for (int i = 0; i < 3; ++i) solver.detectContacts(bodies);

    std::vector<double> cpu_times;
    cpu_times.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = Clock::now();
        solver.detectContacts(bodies);
        auto t1 = Clock::now();
        cpu_times.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    int    n_contacts_cpu = static_cast<int>(solver.getNumContacts());
    Stats  cpu_stat       = summarise(cpu_times);

#ifdef USE_CUDA
    // ---- GPU: fused O(N²) -------------------------------------
    cfg.use_cuda = true;
    solver.setConfig(cfg);

    for (int i = 0; i < 3; ++i) solver.detectContacts(bodies);

    std::vector<double> fused_times;
    fused_times.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = Clock::now();
        solver.detectContacts(bodies);
        auto t1 = Clock::now();
        fused_times.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    int   n_contacts_fused = solver.getStats().cuda_contacts;
    Stats fused_stat       = summarise(fused_times);
    double speedup_fused   = cpu_stat.mean / fused_stat.mean;

    // ---- GPU: two-pass AABB + narrowphase ---------------------
    // Call the two-pass function directly (bypass solver path)
    auto runTwoPass = [&](CudaTwoPassStats* tp_stats) {
        std::vector<GpuContactRaw> raw;
        cudaDetectCapsulePairsTwoPass(
            bodies, static_cast<float>(cfg.delta),
            true,
            boxL, boxL, boxL,
            raw, tp_stats);
        return (int)raw.size();
    };

    // Warmup
    for (int i = 0; i < 3; ++i) runTwoPass(nullptr);

    std::vector<double> tp_times;
    tp_times.reserve(iters);
    CudaTwoPassStats tp_acc{};
    int n_contacts_tp = 0;
    int n_cands_tp    = 0;
    for (int i = 0; i < iters; ++i) {
        CudaTwoPassStats tps{};
        auto t0 = Clock::now();
        n_contacts_tp = runTwoPass(&tps);
        auto t1 = Clock::now();
        tp_times.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
        tp_acc.upload_ms      += tps.upload_ms;
        tp_acc.aabb_ms        += tps.aabb_ms;
        tp_acc.broadphase_ms  += tps.broadphase_ms;
        tp_acc.narrowphase_ms += tps.narrowphase_ms;
        tp_acc.download_ms    += tps.download_ms;
        n_cands_tp = tps.candidates;  // stable across iters
    }
    Stats  tp_stat      = summarise(tp_times);
    double speedup_tp   = cpu_stat.mean / tp_stat.mean;
    // Mean per-stage (ms)
    double s_upload = tp_acc.upload_ms      / iters;
    double s_aabb   = tp_acc.aabb_ms        / iters;
    double s_bp     = tp_acc.broadphase_ms  / iters;
    double s_np     = tp_acc.narrowphase_ms / iters;
    double s_dl     = tp_acc.download_ms    / iters;
#endif

    // ---- Print ------------------------------------------------
    const long long npairs = (long long)N * (N - 1) / 2;

    std::cout << std::setw(6)  << N
              << std::setw(12) << npairs
              << std::setw(10) << n_contacts_cpu
              << std::fixed << std::setprecision(2)
              << std::setw(10) << cpu_stat.mean
              << " ±" << std::setw(5) << cpu_stat.std_dev
#ifdef USE_CUDA
              << std::setw(10) << fused_stat.mean
              << " ±" << std::setw(5) << fused_stat.std_dev
              << std::setw(7)  << speedup_fused << "x"
              << std::setw(10) << n_cands_tp
              << std::setw(10) << tp_stat.mean
              << " ±" << std::setw(5) << tp_stat.std_dev
              << std::setw(7)  << speedup_tp << "x"
#endif
              << "\n"
#ifdef USE_CUDA
              // Per-stage breakdown for two-pass
              << "       two-pass stages (mean ms):"
              << "  upload=" << std::setprecision(3) << s_upload
              << "  aabb="   << s_aabb
              << "  bp="     << s_bp
              << "  narrow=" << s_np
              << "  dl="     << s_dl
              << "  [fused_contacts=" << n_contacts_fused
              << "  tp_contacts=" << n_contacts_tp << "]\n"
#endif
              ;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::vector<int> ns;
    int iters = 50;

    if (argc >= 2) {
        ns.push_back(std::atoi(argv[1]));
        if (argc >= 3) iters = std::atoi(argv[2]);
    } else {
        ns = {100, 256, 512, 1000, 2000, 4000};
    }

    std::cout << "Broadphase benchmark  (iters=" << iters << ")\n";
    std::cout << "Rod geometry: length=0.70 m, diameter=0.10 m, volume_fraction≈30%\n";
    std::cout << "PBC enabled\n\n";

    // Header
    std::cout << std::setw(6)  << "N"
              << std::setw(12) << "pairs"
              << std::setw(10) << "contacts"
              << std::setw(10) << "cpu_ms"
              << std::setw(8)  << "±std"
#ifdef USE_CUDA
              << std::setw(10) << "fused_ms"
              << std::setw(8)  << "±std"
              << std::setw(8)  << "spdup_f"
              << std::setw(10) << "cands"
              << std::setw(10) << "two_ms"
              << std::setw(8)  << "±std"
              << std::setw(8)  << "spdup_2"
#endif
              << "\n";
    std::cout << std::string(
#ifdef USE_CUDA
        110
#else
        46
#endif
        , '-') << "\n";

    for (int N : ns) {
        benchmarkN(N, iters);
        std::cout << "\n";
    }

    return 0;
}
