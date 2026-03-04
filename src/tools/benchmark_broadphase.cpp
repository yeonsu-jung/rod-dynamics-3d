/**
 * @file benchmark_broadphase.cpp
 * @brief Standalone benchmark: CPU spatial-hash vs CUDA naive O(N²) broadphase
 *
 * Usage:
 *   ./benchmark_broadphase [N] [iters]
 *   N     – number of rods  (default: swept: 100 128 256 512 1000 2000 4000)
 *   iters – timed iterations (default: 50)
 *
 * How it generates bodies
 * -----------------------
 * N rods are placed randomly inside a cubic PBC box whose side length is
 * chosen to give ~30 % volume fraction – matching a typical dense simulation.
 *   L = cbrt(N * V_rod / 0.30)    V_rod ≈ π r² (2h) + 4/3 π r³
 *
 * The rods use the same geometry as the default pbc_1000_rods scene:
 *   length 0.70 m, diameter 0.10 m  (radius = 0.05, half-height = 0.35)
 *
 * Output
 * ------
 * For each N it prints a table row:
 *   N  pairs  contacts  cpu_hash_ms  [cuda_ms  speedup]
 */

#include "physics/rigid_body.hpp"
#include "physics/soft_contact.hpp"
#include "config/config.hpp"

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
        // Random position inside box
        glm::vec3 p(pos(rng), pos(rng), pos(rng));

        // Random orientation via uniform quaternion (Shoemake 1992)
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
    double mean = 0.0;
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
    // Rod geometry matching pbc_1000_rods.json
    constexpr float radius = 0.05f;
    constexpr float halfH  = 0.35f;

    // Box side for ~30 % volume fraction
    const float vrod = static_cast<float>(M_PI) * radius * radius * 2.0f * halfH
                     + (4.0f / 3.0f) * static_cast<float>(M_PI) * radius * radius * radius;
    const float boxL = std::cbrt(static_cast<float>(N) * vrod / 0.30f);

    auto bodies = makeRods(N, radius, halfH, boxL);

    // Shared config: PBC enabled, delta=0.005 (matches default scene)
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

    // Warmup
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
    int n_contacts_cpu = static_cast<int>(solver.getNumContacts());
    Stats cpu_stat = summarise(cpu_times);

#ifdef USE_CUDA
    // ---- GPU: CUDA naive O(N²) --------------------------------
    cfg.use_cuda = true;
    solver.setConfig(cfg);

    // Warmup (CUDA context init on first call)
    for (int i = 0; i < 3; ++i) solver.detectContacts(bodies);

    std::vector<double> gpu_times;
    gpu_times.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = Clock::now();
        solver.detectContacts(bodies);
        auto t1 = Clock::now();
        gpu_times.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    int n_contacts_gpu  = solver.getStats().cuda_contacts;
    Stats gpu_stat      = summarise(gpu_times);
    double speedup      = cpu_stat.mean / gpu_stat.mean;
#endif

    // ---- Print ------------------------------------------------
    long long npairs = (long long)N * (N - 1) / 2;
    std::cout << std::setw(6)  << N
              << std::setw(12) << npairs
              << std::setw(10) << n_contacts_cpu
              << std::fixed << std::setprecision(2)
              << std::setw(12) << cpu_stat.mean
              << " ±" << std::setw(6)  << cpu_stat.std_dev
#ifdef USE_CUDA
              << std::setw(12) << gpu_stat.mean
              << " ±" << std::setw(6)  << gpu_stat.std_dev
              << std::setw(9)  << speedup << "x"
              << "  [contacts_gpu=" << n_contacts_gpu << "]"
#endif
              << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    // Parse optional arguments
    std::vector<int> ns;
    int iters = 50;

    if (argc >= 2) {
        ns.push_back(std::atoi(argv[1]));
        if (argc >= 3) iters = std::atoi(argv[2]);
    } else {
        // Default sweep
        ns = {100, 256, 512, 1000, 2000, 4000};
    }

    std::cout << "Broadphase benchmark  (iters=" << iters << ")\n";
    std::cout << "Rod geometry: length=0.70 m, diameter=0.10 m, volume_fraction≈30%\n";
    std::cout << "PBC enabled\n\n";

    std::cout << std::setw(6)  << "N"
              << std::setw(12) << "pairs"
              << std::setw(10) << "contacts"
              << std::setw(12) << "cpu_ms"
              << std::setw(9)  << "±std"
#ifdef USE_CUDA
              << std::setw(12) << "cuda_ms"
              << std::setw(9)  << "±std"
              << std::setw(9)  << "speedup"
#endif
              << "\n";
    std::cout << std::string(70, '-') << "\n";

    for (int N : ns) {
        benchmarkN(N, iters);
    }

    return 0;
}
