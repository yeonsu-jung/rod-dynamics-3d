/**
 * @file main.cpp
 * @brief 3D Rod Dynamics Simulation - Main Application
 */

#ifndef HEADLESS_BUILD
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#endif

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <cstdint>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <array>
#ifdef _OPENMP
#include <omp.h>
#endif

// Global thread limit (0 = use hardware_concurrency)
static int g_thread_limit = 0;

// Unified CLI print control: when true, use a single formatted status line
static const bool CLI_UNIFIED_PRINT = true;

#ifdef TRACY_ENABLE
#  if __has_include(<tracy/Tracy.hpp>)
#    include <tracy/Tracy.hpp>
#  elif __has_include(<Tracy.hpp>)
#    include <Tracy.hpp>
#  else
// Provide no-op stubs if Tracy headers are not available
#    ifndef ZoneScopedN
#      define ZoneScopedN(x)
#    endif
#    ifndef FrameMark
#      define FrameMark
#    endif
namespace tracy { inline void SetThreadName(const char*) {} }
#  endif
#endif

#include "physics/rigid_body.hpp"
#include "physics/collision.hpp"
#include "physics/solver.hpp"
#include "physics/integrator.hpp"
#include "physics/soft_contact.hpp"
#include "physics/mujoco_contact.hpp"
#include "physics/hertz_mindlin.hpp"

#ifndef HEADLESS_BUILD
#include "gfx/renderer.hpp"
#include "gfx/mesh.hpp"
#include "gfx/camera.hpp"
#endif

#include "config/config.hpp"

#ifndef ASSETS_DIR
#define ASSETS_DIR "."
#endif

// entanglement-cpp (external)
#include "linking_number.h"
// Forward declaration for pairwise API defined in external/entanglement-cpp/linking_number.cpp
// double pairwise_abs_linking_sum_with_cutoff(const std::vector<std::array<double,6>>&, double, long long*, int);
extern double pairwise_abs_linking_sum_with_cutoff(
    const std::vector<std::array<double,6>>& rods_array,
    double cutoff,
    long long* out_pairs,
    int num_threads
);

#ifdef GLAD_GL_KHR_debug
static void GLAPIENTRY glDebugCallback(GLenum, GLenum, GLuint, GLenum sev, GLsizei, const GLchar* msg, const void*)
{
    if (sev == GL_DEBUG_SEVERITY_NOTIFICATION) return;
    std::cerr << "[GL] " << msg << "\n";
}
#endif

class App {
public:
    App() = default;
    ~App() = default;

    int run();
    void setConfig(const AppCfg& config);
    void setProfiling(bool enabled) { profilingEnabled = enabled; }
    void enableCsv(const std::string& path) {
        csvPath = path.empty() ? std::string("profile.csv") : path;
        csvStream.open(csvPath, std::ios::out | std::ios::trunc);
        if (!csvStream) {
            std::cerr << "Failed to open CSV file: " << csvPath << "\n";
            csvEnabled = false; return;
        }
        csvEnabled = true; csvHeaderWritten = false;
    }

    // Enable contact dump diagnostics from CLI
    void configureContactDump(const std::string& path, double thresh, int trigger) {
        contactDumpEnabled = true;
        contactDumpPath = path;
        if (thresh >= 0.0) contactDumpThresh = thresh;
        contactDumpTrigger = trigger; // 0:any, +1:up, -1:down
        // open lazily in dumpContactsCSV
    }

    // Headless control
    void setHeadless(bool h) { headless = h; }
    void setHeadlessSteps(int s) { headlessSteps = s; if (perRodEnabled && perRodMaxFrames > 0) perRodSkip = std::max(1, headlessSteps / perRodMaxFrames); }
    // Enable per-rod CSV output (path, maximum sampled frames)
    void enablePerRod(const std::string& path, int maxFrames);

    void enableAdaptiveSubsteps(bool on) { adaptiveSubsteps = on; }
    void setAdaptiveParams(int minS, int maxS, int hitThresh, double dKEUp, double dKEDown) {
        asMin = std::max(1, minS);
        asMax = std::max(asMin, maxS);
        asHitThresh = hitThresh;
        asKEUpThresh = dKEUp;
        asKEDownThresh = dKEDown;
    }
    void setStabilization(float beta_min, int highContactThresh, float betaScale) {
        betaMin = std::max(0.0f, beta_min);
        betaHighContactThresh = highContactThresh;
        betaHighContactScale = std::max(0.0f, betaScale);
    }

    // Entanglement controls
    void setEntanglement(bool enable, double cutoff, int period, int threads) {
        entanglementEnabled = enable;
        entanglementCutoff = cutoff;
        entanglementEvery = std::max(1, period);
        entanglementThreads = threads;
    }

private:
    // ---- Window and OpenGL ----
#ifndef HEADLESS_BUILD
    GLFWwindow* window = nullptr;
    bool vsync = true;
#endif
    bool headless = false;
    int headlessSteps = 1000;

    // ---- Renderer and meshes ----
#ifndef HEADLESS_BUILD
    Renderer rnd;
    Mesh cube, cyl, sphere;

    // ---- Camera ----
    OrbitCamera cam;
    bool dragging = false;
    double lastX = 0.0, lastY = 0.0;
#endif

    // ---- Simulation ----
    bool paused = false;
    glm::vec3 gravity{0.0f, -10.0f, 0.0f};
    float dt = 1.0f / 600.0f;
    AppCfg settings{};
    SolverConfig solver{};
    SoftContactSolver softContactSolver{};
    MujocoContactSolver mjContactSolver{};
    HertzMindlinSolver hertzMindlinSolver{};

    // Periodic box
    bool usePBC = false;
    glm::vec3 pbcMin{-3,-1,-3}, pbcMax{3,3,3};
    float cellSize = 0.6f; // broadphase grid cell size

    // ---- Physics objects ----
    std::vector<RigidBody> rods;
    RigidBody floorRB;

    // ---- Initialization ----
    bool initWindow(int width = 1200, int height = 800, 
                   const char* title = "Rigid Bodies - Rods (Capsules)");
    bool initGraphics();

    // ---- Scene management ----
    static RigidBody createRod(const BodyCfg& config);
    void resetScene();

    // ---- Event callbacks ----
#ifndef HEADLESS_BUILD
    static void keyCB(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void cursorCB(GLFWwindow* window, double x, double y);
    static void mouseCB(GLFWwindow* window, int button, int action, int mods);
    static void scrollCB(GLFWwindow* window, double xoffset, double yoffset);
#endif

    // ---- Profiling (built-in lightweight) ----
    struct Times {
        double integrate = 0, sleepUpdate = 0, broadphase = 0, warmstart = 0,
               buildIslands = 0, solve = 0, floorSolve = 0, posCorrect = 0, pbcWrap = 0, render = 0;
        // New fine-grained broadphase
        double bpCount = 0, bpPrefix = 0, bpFill = 0, bpPairs = 0, bpLongLong = 0;
        void reset(){ integrate = sleepUpdate = broadphase = warmstart = buildIslands = solve = floorSolve = posCorrect = pbcWrap = render = 0; bpCount = bpPrefix = bpFill = bpPairs = bpLongLong = 0; }
        Times& operator+=(const Times& o){ integrate+=o.integrate; sleepUpdate+=o.sleepUpdate; broadphase+=o.broadphase; warmstart+=o.warmstart; buildIslands+=o.buildIslands; solve+=o.solve; floorSolve+=o.floorSolve; posCorrect+=o.posCorrect; pbcWrap+=o.pbcWrap; render+=o.render; bpCount+=o.bpCount; bpPrefix+=o.bpPrefix; bpFill+=o.bpFill; bpPairs+=o.bpPairs; bpLongLong+=o.bpLongLong; return *this; }
    };
    struct ScopedAccum {
        using clock = std::chrono::high_resolution_clock;
        double* acc = nullptr; clock::time_point t0;
        explicit ScopedAccum(double* dst): acc(dst), t0(clock::now()){}
        ~ScopedAccum(){ if(acc){ auto t1=clock::now(); *acc += std::chrono::duration<double,std::milli>(t1 - t0).count(); } }
    };
    bool profilingEnabled = false;
    Times curTimes{}, sumTimes{};
    int sumFrames = 0;
    std::chrono::high_resolution_clock::time_point lastTitleUpdate{};
#ifndef HEADLESS_BUILD
    void maybeUpdateWindowTitle();
#endif

    // CSV logging
    bool csvEnabled = false;
    std::ofstream csvStream;
    std::string csvPath;
    bool csvHeaderWritten = false;
    uint64_t frameIndex = 0;
    size_t lastHitCount = 0;
    size_t lastIslandCount = 0;
    // New: energy tracking
    double lastKE = 0.0;
    // KE checkpoints per frame (diagnostics)
    double keAfterIntegrate = 0.0;
    double keAfterWarmstart = 0.0;
    double keAfterSolve = 0.0;
    double keAfterPosCorrect = 0.0;
    double keAfterPBCWrap = 0.0;
    void logCsvFrame();
    // Soft-contact potential energy (total over contacts) captured after latest force computation
    double lastSoftPotentialEnergy = 0.0;
    // Optional separate CSV logging for soft contact potential energy
    bool softPEEnabled = false;
    std::ofstream softPEStream;
    bool softPEHeaderWritten = false;
    std::string softPEPath;
public: // expose soft PE enabling API
    void enableSoftPE(const std::string& path) {
        softPEPath = path.empty() ? std::string("soft_pe.csv") : path;
        softPEStream.open(softPEPath, std::ios::out | std::ios::trunc);
        if (!softPEStream) {
            std::cerr << "Failed to open soft PE CSV file: " << softPEPath << "\n";
            softPEEnabled = false; return;
        }
        softPEEnabled = true; softPEHeaderWritten = false;
    }
    
    // Enable center-of-mass tracking
    void enableCOM(const std::string& path) {
        comPath = path.empty() ? std::string("com.csv") : path;
        comStream.open(comPath, std::ios::out | std::ios::trunc);
        if (!comStream) {
            std::cerr << "Failed to open COM CSV file: " << comPath << "\n";
            comEnabled = false; return;
        }
        comEnabled = true; comHeaderWritten = false;
    }
    
    // Enable contact network tracking
    void enableNetwork(const std::string& path) {
        networkPath = path.empty() ? std::string("network.csv") : path;
        networkStream.open(networkPath, std::ios::out | std::ios::trunc);
        if (!networkStream) {
            std::cerr << "Failed to open network CSV file: " << networkPath << "\n";
            networkEnabled = false; return;
        }
        networkEnabled = true; networkHeaderWritten = false;
    }
    
private:
    void logSoftPEFrame() {
        if (!softPEEnabled || !softPEStream) return;
        if (!softPEHeaderWritten) { softPEStream << "frame,soft_PE\n"; softPEHeaderWritten = true; }
        softPEStream << frameIndex << ',' << lastSoftPotentialEnergy << '\n';
        if ((frameIndex & 0x3F) == 0) softPEStream.flush();
    }
    
    // Center-of-mass CSV logging
    bool comEnabled = false;
    std::ofstream comStream;
    std::string comPath;
    bool comHeaderWritten = false;
    void logCOMFrame();
    glm::vec3 computeCOM() const;
    
    // Contact network CSV logging
    bool networkEnabled = false;
    std::ofstream networkStream;
    std::string networkPath;
    bool networkHeaderWritten = false;
    void logNetworkFrame();  // Will detect contact mode automatically
    
    // Per-rod CSV logging
    bool perRodEnabled = false;
    int perRodMaxFrames = 1000;
    std::string perRodPath;
    std::ofstream perRodStream;
    bool perRodHeaderWritten = false;
    int perRodSkip = 1; // sample every N frames
    int perRodWrittenFrames = 0;
    void logPerRodFrame();
    // Compute total kinetic energy for current rods
    double totalKE() const;

    // Adaptive substeps (runtime-tunable)
    bool adaptiveSubsteps = false;
    int  asMin = 1;
    int  asMax = 1;
    int  asHitThresh = INT32_MAX;
    double asKEUpThresh = 1e300;   // huge => disabled by default
    double asKEDownThresh = -1e300; // very negative => disabled
    double lastFrameKEDelta = 0.0;  // KE_n - KE_{n-1}
    double prevFrameKE = 0.0;       // KE of previous frame

    // Positional stabilization tuning (Baumgarte scaling)
    float betaMin = 0.0f;           // clamp lower bound on beta during dyn scaling
    int   betaHighContactThresh = INT32_MAX;
    float betaHighContactScale = 1.0f; // multiply solver.baumgarte by this when many contacts

    // Declare Hit before using in dumpContactsCSV
    struct Hit; // forward declaration

    // Contact dump diagnostics
    bool contactDumpEnabled = false;
    std::string contactDumpPath;
    std::ofstream contactDumpStream;
    bool contactDumpHeaderWritten = false;
    double contactDumpThresh = 0.0; // absolute KE increase/decrease threshold to trigger (J)
    int contactDumpTrigger = 0; // 0:any, +1:up, -1:down
    bool contactDumpTriggeredThisFrame = false;
    void dumpContactsCSV(const std::vector<Hit>& hits, const char* stageLabel);
    
    // ---- Simulation ----
    struct Hit { 
        int a = -1, b = -1; 
        Contact c{}; 
    };
    
    void physicsStep();
    void renderFrame();
    void stepWithSubsteps();

    // ---- Helpers ----
    static inline glm::ivec3 gridDims(const glm::vec3& bmin, const glm::vec3& bmax, float cs) {
        glm::vec3 size = bmax - bmin;
        glm::ivec3 n = glm::max(glm::ivec3(1), glm::ivec3(glm::floor(size / cs)));
        return n;
    }
    static inline glm::ivec3 cellIndex(const glm::vec3& p, const glm::vec3& bmin, const glm::vec3& bmax, const glm::ivec3& n) {
        glm::vec3 size = bmax - bmin;
        glm::vec3 rel = (p - bmin) / size; // in [0,1)
        glm::ivec3 idx = glm::clamp(glm::ivec3(rel * glm::vec3(n)), glm::ivec3(0), n - 1);
        return idx;
    }
    static inline int64_t packKey(const glm::ivec3& i, const glm::ivec3& n) {
        // pack 3 indices into 64-bit (assumes n components < 2^21)
        return (int64_t(i.x) << 42) ^ (int64_t(i.y) << 21) ^ int64_t(i.z);
    }
    static inline size_t linearIndex(const glm::ivec3& i, const glm::ivec3& n) {
        return size_t(i.x) + size_t(n.x) * (size_t(i.y) + size_t(n.y) * size_t(i.z));
    }
    static inline uint64_t pairKey(int a, int b) {
        if (b < a) std::swap(a,b);
        return (uint64_t(uint32_t(a)) << 32) | uint64_t(uint32_t(b));
    }
    static inline void wrapPos(glm::vec3& p, const glm::vec3& bmin, const glm::vec3& bmax) {
        const glm::vec3 size = bmax - bmin;
        for (int k = 0; k < 3; ++k) {
            if (size[k] <= 0.0f) continue;
            while (p[k] < bmin[k]) p[k] += size[k];
            while (p[k] >= bmax[k]) p[k] -= size[k];
        }
    }

    // ---- Sleeping (simple) ----
    float sleepLinThresh = 0.02f;   // m/s
    float sleepAngThresh = 0.05f;   // rad/s
    float sleepTimeThresh = 0.6f;   // s
    std::vector<float> sleepTimer;  // per-body accumulated below-threshold time
    std::vector<uint8_t> sleeping;  // 0/1 flags
    inline void wake(int i) {
        if (i < 0 || i >= (int)rods.size()) return;
        sleeping[i] = 0; sleepTimer[i] = 0.f;
    }

    template <class F>
    static void parallel_for(size_t begin, size_t end, F fn) {
        // Simple static partitioning
        const size_t N = end - begin;
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        const unsigned T = (g_thread_limit > 0) ? std::min<unsigned>(g_thread_limit, (unsigned)N) : std::min<unsigned>(hw, (unsigned)N);
        if (T <= 1 || N < 1024) { // small tasks run single-threaded
            for (size_t i = begin; i < end; ++i) fn(i);
            return;
        }
        std::vector<std::thread> threads; threads.reserve(T);
        size_t chunk = (N + T - 1) / T;
        for (unsigned t = 0; t < T; ++t) {
            size_t s = begin + t * chunk;
            size_t e = std::min(begin + (t + 1) * chunk, end);
            if (s >= e) break;
            threads.emplace_back([=]() {
                for (size_t i = s; i < e; ++i) fn(i);
            });
        }
        for (auto& th : threads) th.join();
    }

    // ---- Broadphase scratch (reused across frames) ----
    glm::ivec3 gridN{0};
    // Flattened grid: counts -> prefix-sum offsets -> items
    std::vector<uint32_t> gridCounts;   // size = cellCount
    std::vector<uint32_t> gridOffsets;  // size = cellCount+1
    std::vector<uint32_t> gridWrite;    // temp write cursors, size = cellCount
    std::vector<int>      gridItems;    // flattened body indices
    std::vector<std::vector<Hit>> thHitsScratch;         // per-thread hit buffers
    std::vector<std::vector<int>> thSeenAt;              // per-thread seen stamps sized [numRods]
    std::vector<std::vector<int>> thCellSeenAt;         // per-thread cell visited stamps sized [cellCount]
    std::vector<Hit> hitsScratch;                        // merged hits

    // Warm-start cache: previous-frame impulses per pair (a<b)
    std::unordered_map<uint64_t, AppliedImpulse> warmCache;
    std::vector<uint64_t> hitKeysScratch; // keys for current hits

    // Random force application
    bool useRandomForce = false;
    std::mt19937 genRandomForce{std::random_device{}()};
    std::normal_distribution<float> normal_f{0.0f, 1.0f};
    std::uniform_real_distribution<float> uni_u{-1.0f, 1.0f};
    std::uniform_real_distribution<float> uni_phi{0.0f, 2.0f * float(M_PI)};
    float tauMag = 0.1f;
    float fSigma = 0.0f;

    glm::vec3 uniform_dir_s2(std::mt19937& gen) {
        float u = uni_u(gen);
        float phi = uni_phi(gen);
        float s = std::sqrt(std::max(0.0f, 1.0f - u*u));
        return glm::vec3(s * std::cos(phi), u, s * std::sin(phi));
    }

    // Entanglement metrics (sum of |linking number| over all rod pairs)
    bool entanglementEnabled = false;
    int  entanglementEvery = 60;     // compute every N frames
    double entanglementCutoff = -1;  // <=0 disables pruning in library
    int  entanglementThreads = 0;    // 0 => auto
    double lastEntanglementSum = 0.0;
    long long lastEntanglementPairs = 0;
    void computeEntanglement();
    // Print a unified CLI status line (frame, rods, KE, entanglement)
    void printCliStatus(const std::string& prefix = "") const;
};

// ---- Implementation ----

#ifndef HEADLESS_BUILD
bool App::initWindow(int width, int height, const char* title) {
    if (!glfwInit()) { 
        std::cerr << "GLFW init failed\n"; 
        return false; 
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, std::max(1, settings.render.msaa_samples));
    
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window) { 
        std::cerr << "GLFW window creation failed\n"; 
        glfwTerminate(); 
        return false; 
    }
    
    glfwMakeContextCurrent(window);

    vsync = settings.render.vsync;
    glfwSwapInterval(vsync ? 1 : 0);

    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) { 
        std::cerr << "GLAD load failed\n"; 
        return false; 
    }

    // OpenGL state setup
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    
    if (settings.render.cull) { 
        glEnable(GL_CULL_FACE); 
        glCullFace(GL_BACK); 
    } else { 
        glDisable(GL_CULL_FACE); 
    }
    
    glDisable(GL_BLEND);

#ifdef GLAD_GL_KHR_debug
    if (GLAD_GL_KHR_debug) { 
        glEnable(GL_DEBUG_OUTPUT); 
        glDebugMessageCallback(glDebugCallback, nullptr); 
    }
#endif

    // Set up event callbacks
    glfwSetWindowUserPointer(window, this);
    glfwSetKeyCallback(window, &App::keyCB);
    glfwSetCursorPosCallback(window, &App::cursorCB);
    glfwSetMouseButtonCallback(window, &App::mouseCB);
    glfwSetScrollCallback(window, &App::scrollCB);
    
    return true;
}

bool App::initGraphics() {
    if (!rnd.init(ASSETS_DIR)) {
        std::cerr << "Renderer init failed (check " << ASSETS_DIR << "/shaders).\n";
        return false;
    }
    cube = makeCubeMesh();
    cyl = makeCappedCylinderMesh(16);  // Reduced from 40 to 16 for better performance with many rods
    sphere = makeSphereMesh(16, 12);   // 16 slices, 12 stacks for smooth spheres
    return true;
}
#endif

RigidBody App::createRod(const BodyCfg& config) {
    // rot_quat is glm::vec4 {w,x,y,z} (resolved by config)
    glm::quat q(config.rot_quat.x, config.rot_quat.y, config.rot_quat.z, config.rot_quat.w);
    
    RigidBody rb;
    
    // Create body based on shape type
    if (config.shape == "sphere") {
        rb = RigidBody::makeSphere(config.pos, config.density, config.radius,
                                    config.restitution, config.friction);
    } else {
        // Default to capsule/rod
        rb = RigidBody::makeRodLD(config.pos, q, config.density, config.length, 
                                   config.diameter, config.restitution, config.friction);
    }
    
    // Advanced friction (optional): default to legacy if not provided
    if (config.friction_s > 0.0f) rb.frictionS = config.friction_s; else rb.frictionS = -1.0f;
    if (config.friction_d > 0.0f) rb.frictionD = config.friction_d; else rb.frictionD = -1.0f;
    rb.rollingFriction = config.rolling_friction;
    rb.v = config.v_lin; 
    rb.w = config.v_ang;
    return rb;
}

void App::resetScene() {
    dt = settings.physics.dt;
    gravity = settings.physics.gravity;
    solver = settings.physics.solver;
    
    // Configure soft contact solvers
    softContactSolver.setConfig(settings.physics.soft_contact);
    MujocoContactCfg mjCfg;
    mjCfg.enabled = settings.physics.soft_contact.enabled && settings.physics.use_mujoco_contact;
    // Map a few basic parameters; we start conservatively and can tune later.
    mjCfg.normal_k = settings.physics.soft_contact.k_scaler; // reuse scaler as stiffness
    mjCfg.normal_damping = 1.0; // default damping factor; could be exposed later
    mjCfg.friction_mu = settings.physics.soft_contact.mu;
    mjCfg.vel_eps = settings.physics.soft_contact.nu;
    mjContactSolver.setConfig(mjCfg);
    
    // Configure Hertz-Mindlin solver (convert config struct)
    HertzMindlinCfg hmCfg;
    hmCfg.youngs_modulus = settings.physics.hertz_mindlin.youngs_modulus;
    hmCfg.poisson_ratio = settings.physics.hertz_mindlin.poisson_ratio;
    hmCfg.restitution_coeff = settings.physics.hertz_mindlin.restitution_coeff;
    hmCfg.friction_coeff = settings.physics.hertz_mindlin.friction_coeff;
    hmCfg.rolling_friction_coeff = settings.physics.hertz_mindlin.rolling_friction_coeff;
    hmCfg.enable_tangential = settings.physics.hertz_mindlin.enable_tangential;
    hmCfg.enable_rolling = settings.physics.hertz_mindlin.enable_rolling;
    hmCfg.verbose = settings.physics.hertz_mindlin.verbose;
    hmCfg.computeDamping();  // Recompute damping from restitution
    hertzMindlinSolver.setConfig(hmCfg);

    g_lin_damp = settings.physics.lin_damp;
    g_ang_damp = settings.physics.ang_damp;
    g_w_max = settings.physics.w_max;

    // Periodic config
    usePBC = settings.scene.periodic.enabled;
    pbcMin = settings.scene.periodic.min;
    pbcMax = settings.scene.periodic.max;
    cellSize = settings.scene.periodic.cellSize;
    g_pbc_enabled = usePBC;
    g_pbc_min = pbcMin; 
    g_pbc_max = pbcMax;

    // Random initialization for PBC study
    const bool useRandomInit = usePBC && settings.scene.randomInit.enabled;
    if (useRandomInit) {
        gravity = glm::vec3(0.0f);
    }

    // Random force settings
    useRandomForce = settings.scene.randomForce.enabled;
    if (useRandomForce) {
        fSigma = settings.scene.randomForce.fSigma;
        tauMag = settings.scene.randomForce.tauMag;
        genRandomForce.seed(settings.scene.randomForce.seed ? settings.scene.randomForce.seed : std::random_device{}());
    }

    // Floor (only if not using PBC)
    const auto& floorConfig = settings.scene.floor;
    glm::quat qF(floorConfig.rot_quat.x, floorConfig.rot_quat.y, 
                 floorConfig.rot_quat.z, floorConfig.rot_quat.w);
    floorRB = RigidBody::makeStaticFloor(
        floorConfig.pos, qF, 
        floorConfig.half_extents.x, floorConfig.half_extents.y, floorConfig.half_extents.z,
            floorConfig.restitution, floorConfig.friction
    );

    rods.clear();
    sleeping.clear();
    sleepTimer.clear();

    // Procedural population overrides explicit bodies if requested
    if (settings.scene.populate.count > 0) {
        const int N = settings.scene.populate.count;
        std::random_device rd;
        std::mt19937 gen(settings.scene.populate.seed ? settings.scene.populate.seed : rd());
        std::uniform_real_distribution<float> urand(0.0f, 1.0f);

        // Use first body's dimensions if provided, else defaults
        BodyCfg base = settings.scene.bodies.empty() ? BodyCfg{} : BodyCfg(settings.scene.bodies.front());
        // const float L = base.length; // unused
        const float D = base.diameter;
        const float spacing = settings.scene.populate.spacingMul * D;

        const std::string mode = settings.scene.populate.mode;
        glm::vec3 bmin = usePBC ? pbcMin : glm::vec3(-10.0f);
        glm::vec3 bmax = usePBC ? pbcMax : glm::vec3(+10.0f);
        glm::vec3 boxSize = bmax - bmin;
        if (mode == "grid") {
            glm::vec3 extent = pbcMax - pbcMin;
            // Compute grid dims to fit N
            int nx = std::max(1, int(extent.x / spacing));
            int ny = std::max(1, int(extent.y / spacing));
            int nz = std::max(1, int(extent.z / spacing));
            long long capacity = 1LL * nx * ny * nz;
            if (capacity < N) {
                // Increase spacing slightly to accommodate N
                float scale = std::cbrt(float(N) / std::max(1.0f, float(capacity)));
                int nx2 = std::max(1, int(nx / scale));
                int ny2 = std::max(1, int(ny / scale));
                int nz2 = std::max(1, int(nz / scale));
                nx = nx2; ny = ny2; nz = nz2;
            }

            rods.reserve(N);
            int placed = 0;
            for (int ix = 0; ix < nx && placed < N; ++ix) {
                for (int iy = 0; iy < ny && placed < N; ++iy) {
                    for (int iz = 0; iz < nz && placed < N; ++iz) {
                        BodyCfg cfg = base;
                        // Jittered grid center
                        glm::vec3 cellMin = pbcMin + glm::vec3(ix * spacing, iy * spacing, iz * spacing);
                        glm::vec3 p = cellMin + glm::vec3(0.5f * spacing);
                        // Small jitter within cell
                        glm::vec3 jitter{ (urand(gen)-0.5f)*0.3f*spacing,
                                          (urand(gen)-0.5f)*0.3f*spacing,
                                          (urand(gen)-0.5f)*0.3f*spacing };
                        cfg.pos = p + jitter;
                        // Random orientation around a random axis
                        glm::vec3 axis = glm::normalize(glm::vec3(urand(gen)-0.5f, urand(gen)-0.5f, urand(gen)-0.5f));
                        float angle = (urand(gen) - 0.5f) * 3.14159f; // [-pi/2, pi/2]
                        glm::quat q = glm::angleAxis(angle, axis);
                        cfg.rot_quat = glm::vec4(q.w, q.x, q.y, q.z);
                        rods.push_back(createRod(cfg));
                        ++placed;
                    }
                }
            }
        } else if (mode == "nonoverlap") {
            // Non-overlapping initializer (works with PBC or NPBC)
            rods.clear();
            rods.reserve(N);
            const glm::vec3 bmin = pbcMin;
            const glm::vec3 bmax = pbcMax;
            const glm::vec3 boxSize = bmax - bmin;
            const float halfL = 0.5f * base.length;
            const float R = 0.5f * base.diameter;
            const float diam2 = (2.0f * R) * (2.0f * R);
            // Choose placement grid cell size ~ rod length
            const float cs = (cellSize > 0.0f ? cellSize : std::max(0.25f * base.length, 2.5f * R));
            glm::ivec3 n = gridDims(bmin, bmax, cs);
            const int numCells = std::max(1, n.x * n.y * n.z);
            std::vector<std::vector<int>> cells(numCells);
            auto linIdx = [&](int ix,int iy,int iz){ return ix + n.x * (iy + n.y * iz); };
            auto wrapI = [&](int a,int dim){ if (a<0) return a+dim; if (a>=dim) return a-dim; return a; };
            auto minImage = [&](glm::vec3 d){
                if (!usePBC) return d;
                glm::vec3 r = d;
                for (int k=0;k<3;++k){ float L=boxSize[k]; if (L>0.0f) r[k] -= L*std::floor(r[k]/L + 0.5f); }
                return r;
            };
            auto segAABB = [&](const glm::vec3& c, const glm::vec3& u, glm::vec3& aabbMin, glm::vec3& aabbMax){
                glm::vec3 ext = glm::abs(u) * halfL + glm::vec3(R);
                aabbMin = c - ext; aabbMax = c + ext;
            };
            auto rangesForAABB = [&](const glm::vec3& mn, const glm::vec3& mx, glm::ivec3& i0, glm::ivec3& i1){
                i0 = glm::floor((mn - bmin) / cs);
                i1 = glm::floor((mx - bmin) / cs);
            };
            auto uniform_dir_s2 = [&](std::mt19937& g){
                float u = 2.0f * urand(g) - 1.0f;
                float phi = 2.0f * float(M_PI) * urand(g);
                float s = std::sqrt(std::max(0.0f, 1.0f - u*u));
                return glm::vec3(s * std::cos(phi), u, s * std::sin(phi));
            };
            auto quat_from_axisY = [&](const glm::vec3& dir){
                const glm::vec3 y(0,1,0);
                float d = glm::clamp(glm::dot(y, dir), -1.0f, 1.0f);
                float ang = std::acos(d);
                if (ang < 1e-6f) return glm::quat(1,0,0,0);
                glm::vec3 axis = glm::normalize(glm::cross(y, dir));
                return glm::angleAxis(ang, axis);
            };
            auto segseg_dist2 = [&](const glm::vec3& p0,const glm::vec3& p1,const glm::vec3& q0,const glm::vec3& q1){
                // robust segment-segment distance (no PBC inside; caller applies min-image via shifting q by centroid-based shift)
                glm::vec3 u = p1 - p0; glm::vec3 v = q1 - q0; glm::vec3 w0 = p0 - q0;
                float uu = glm::dot(u,u), vv = glm::dot(v,v), uv = glm::dot(u,v);
                float wu = glm::dot(w0,u), wv = glm::dot(w0,v);
                float D = uu*vv - uv*uv; float s, t;
                const float eps = 1e-12f;
                if (std::abs(D) < eps) { s = 0.0f; t = (vv>=eps)? (-wv/vv):0.0f; }
                else { s = (uv*wv - vv*wu)/D; t = (uu*wv - uv*wu)/D; }
                s = glm::clamp(s, 0.0f, 1.0f);
                t = (s*uv + wv) / (vv >= eps ? vv : 1.0f);
                t = glm::clamp(t, 0.0f, 1.0f);
                float su = (-wu + t*uv) / (uu >= eps ? uu : 1.0f);
                if (!(t > 1e-6f && t < 1.0f-1e-6f)) {
                    if (su < 0.0f) s = 0.0f; else if (su > 1.0f) s = su;
                }
                glm::vec3 d = (w0 + s*u) - t*v;
                return glm::dot(d,d);
            };
            const int maxAttempts = std::max(1000, settings.scene.populate.maxAttempts);

            // Store accepted centroids and directions for candidate checks
            std::vector<glm::vec3> C; C.reserve(N);
            std::vector<glm::vec3> U; U.reserve(N);

            for (int i = 0; i < N; ++i) {
                bool placed = false;
                for (int att = 0; att < maxAttempts && !placed; ++att) {
                    // Sample uniform centroid and direction
                    glm::vec3 r{urand(gen), urand(gen), urand(gen)};
                    glm::vec3 c = bmin + r * boxSize;
                    glm::vec3 udir = glm::normalize(uniform_dir_s2(gen));

                    // Build endpoints
                    glm::vec3 p0 = c - udir * halfL;
                    glm::vec3 p1 = c + udir * halfL;

                    // Gather candidates via AABB cells
                    glm::vec3 mn, mx; segAABB(c, udir, mn, mx);
                    glm::ivec3 a0, a1; rangesForAABB(mn, mx, a0, a1);
                    bool collide = false;
                    // Iterate overlapped cells (handle PBC by wrapping indices)
                    for (int iz = a0.z; iz <= a1.z && !collide; ++iz)
                    for (int iy = a0.y; iy <= a1.y && !collide; ++iy)
                    for (int ix = a0.x; ix <= a1.x && !collide; ++ix) {
                        int cx = usePBC ? wrapI(ix,n.x) : glm::clamp(ix,0,n.x-1);
                        int cy = usePBC ? wrapI(iy,n.y) : glm::clamp(iy,0,n.y-1);
                        int cz = usePBC ? wrapI(iz,n.z) : glm::clamp(iz,0,n.z-1);
                        const auto& bucket = cells[linIdx(cx,cy,cz)];
                        for (int j : bucket) {
                            // Shift previous rod to minimum image wrt this centroid (approx)
                            glm::vec3 cj = C[j];
                            glm::vec3 uj = U[j];
                            glm::vec3 shift(0);
                            if (usePBC) shift = minImage(cj - c);
                            glm::vec3 q0 = (cj + shift) - uj * halfL;
                            glm::vec3 q1 = (cj + shift) + uj * halfL;
                            float d2 = segseg_dist2(p0, p1, q0, q1);
                            if (d2 < diam2) { collide = true; break; }
                        }
                    }

                    if (!collide) {
                        // Accept: record and insert into cells
                        C.push_back(c); U.push_back(udir);
                        BodyCfg cfg = base;
                        cfg.pos = c;
                        glm::quat q = quat_from_axisY(udir);
                        cfg.rot_quat = glm::vec4(q.w, q.x, q.y, q.z);
                        rods.push_back(createRod(cfg));
                        // Insert to cells
                        for (int iz = a0.z; iz <= a1.z; ++iz)
                        for (int iy = a0.y; iy <= a1.y; ++iy)
                        for (int ix = a0.x; ix <= a1.x; ++ix) {
                            int cx = usePBC ? wrapI(ix,n.x) : glm::clamp(ix,0,n.x-1);
                            int cy = usePBC ? wrapI(iy,n.y) : glm::clamp(iy,0,n.y-1);
                            int cz = usePBC ? wrapI(iz,n.z) : glm::clamp(iz,0,n.z-1);
                            cells[linIdx(cx,cy,cz)].push_back(i);
                        }
                        placed = true;
                    }
                }
                if (!placed) {
                    std::cerr << "[populate] nonoverlap: failed to place rod " << i << "/" << N << " after attempts= " << maxAttempts << "\n";
                    break;
                }
            }
        } else if (mode == "random") {
            // Random placement without overlap check
            auto uniform_dir_s2_local = [&](std::mt19937& gen) -> glm::vec3 {
                float u1 = urand(gen), u2 = urand(gen);
                float theta = 2.0f * 3.14159f * u1;
                float phi = acos(2.0f * u2 - 1.0f);
                return glm::vec3(sin(phi)*cos(theta), sin(phi)*sin(theta), cos(phi));
            };
            auto quat_from_axisY_local = [](const glm::vec3& axis) -> glm::quat {
                glm::vec3 up(0,1,0);
                glm::vec3 axis_norm = glm::normalize(axis);
                float dot = glm::dot(up, axis_norm);
                if (abs(dot - 1.0f) < 1e-6f) return glm::quat(1,0,0,0);
                if (abs(dot + 1.0f) < 1e-6f) return glm::quat(0,1,0,0);
                glm::vec3 cross = glm::cross(up, axis_norm);
                float s = sqrt(2.0f * (1.0f + dot));
                return glm::quat(s * 0.5f, cross.x / s, cross.y / s, cross.z / s);
            };
            rods.reserve(N);
            for (int i = 0; i < N; ++i) {
                glm::vec3 r{urand(gen), urand(gen), urand(gen)};
                glm::vec3 c = bmin + r * boxSize;
                glm::vec3 udir = glm::normalize(uniform_dir_s2_local(gen));
                BodyCfg cfg = base;
                cfg.pos = c;
                glm::quat q = quat_from_axisY_local(udir);
                cfg.rot_quat = glm::vec4(q.w, q.x, q.y, q.z);
                rods.push_back(createRod(cfg));
            }
        } else {
            // Fallback: two default rods if scene is empty
            BodyCfg rodA{}, rodB{};
            
            rodA.pos = {-1.6f, 0.6f, 0.0f}; 
            rodA.rot_quat = {1, 0, 0, 0};
            rodA.density = 1000.0f; rodA.length = 0.5f; rodA.diameter = 0.10f; 
            rodA.restitution = 0.15f; rodA.friction = 0.6f; rodA.v_lin = {+2.2f, 0, 0};
            
            rodB.pos = {+1.2f, 1.0f, 0.2f}; 
            rodB.rot_quat = {1, 0, 0, 0};
            rodB.density = 1000.0f; rodB.length = 0.5f; rodB.diameter = 0.10f; 
            rodB.restitution = 0.15f; rodB.friction = 0.6f; rodB.v_lin = {-1.0f, 0, 0};
            
            rods.push_back(createRod(rodA));
            rods.push_back(createRod(rodB));
        }
    }

    // If rods still empty, populate from explicit bodies or fallback
    if (rods.empty()) {
        if (!settings.scene.bodies.empty()) {
            rods.reserve(settings.scene.bodies.size());
            for (const auto& bodyConfig : settings.scene.bodies) {
                rods.push_back(createRod(bodyConfig));
            }
        } else {
            // Fallback: two default rods if scene is empty
            BodyCfg rodA{}, rodB{};
            
            rodA.pos = {-1.6f, 0.6f, 0.0f}; 
            rodA.rot_quat = {1, 0, 0, 0};
            rodA.density = 1000.0f; rodA.length = 0.5f; rodA.diameter = 0.10f; 
            rodA.restitution = 0.15f; rodA.friction = 0.6f; rodA.v_lin = {+2.2f, 0, 0};
            
            rodB.pos = {+1.2f, 1.0f, 0.2f}; 
            rodB.rot_quat = {1, 0, 0, 0};
            rodB.density = 1000.0f; rodB.length = 0.5f; rodB.diameter = 0.10f; 
            rodB.restitution = 0.15f; rodB.friction = 0.6f; rodB.v_lin = {-1.0f, 0, 0};
            
            rods.push_back(createRod(rodA));
            rods.push_back(createRod(rodB));
        }
    }

    if (useRandomInit) {
        // Gaussian translational velocities, Uniform S2 direction with fixed magnitude for angular
        std::random_device rd;
        std::mt19937 gen(settings.scene.randomInit.seed ? settings.scene.randomInit.seed : rd());
        std::uniform_real_distribution<float> uniform(-settings.scene.randomInit.vSigma, settings.scene.randomInit.vSigma);
        std::uniform_real_distribution<float> uni(0.0f, 1.0f);
        const float wSpeed = settings.scene.randomInit.wSpeed;

        auto uniform_dir_s2 = [&](std::mt19937& g) {
            float u = 2.0f * uni(g) - 1.0f; // cos(theta) in [-1,1]
            float phi = 2.0f * float(M_PI) * uni(g);
            float s = std::sqrt(std::max(0.0f, 1.0f - u*u));
            return glm::vec3(s * std::cos(phi), u, s * std::sin(phi));
        };

        for (auto& rb : rods) {
            rb.v = { uniform(gen), uniform(gen), uniform(gen) };
            rb.w = wSpeed * uniform_dir_s2(gen);
        }
    }

    // Adaptive broadphase cell size if requested (<= 0 => auto)
    if (usePBC && cellSize <= 0.0f && !rods.empty()) {
        double sumD = 0.0;
        for (const auto& rb : rods) sumD += double(rb.cap.r) * 2.0; // diameter
        double avgD = sumD / double(rods.size());
        // Slightly larger than diameter to keep occupancy per cell modest
        cellSize = float(std::max(0.05, 1.25 * avgD));
        // Reset grid buffers to force reallocation on next step
        gridN = glm::ivec3(0);
        gridCounts.clear(); gridOffsets.clear(); gridWrite.clear(); gridItems.clear();
    }

    // init sleeping arrays
    sleeping.assign(rods.size(), 0);
    sleepTimer.assign(rods.size(), 0.f);

    // Reset KE history for adaptive decisions
    lastKE = totalKE();
    prevFrameKE = lastKE;
    lastFrameKEDelta = 0.0;
}

#ifndef HEADLESS_BUILD
void App::keyCB(GLFWwindow* window, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    
    auto* self = static_cast<App*>(glfwGetWindowUserPointer(window));
    switch (key) {
        case GLFW_KEY_ESCAPE: 
            glfwSetWindowShouldClose(window, 1); 
            break;
        case GLFW_KEY_SPACE:  
            self->paused = !self->paused;   
            break;
        case GLFW_KEY_R:      
            self->resetScene();             
            break;
        case GLFW_KEY_V:
            self->vsync = !self->vsync;
            glfwSwapInterval(self->vsync ? 1 : 0);
            break;
        default: 
            break;
    }
}

void App::cursorCB(GLFWwindow* window, double x, double y) {
    auto* self = static_cast<App*>(glfwGetWindowUserPointer(window));
    if (!self->dragging) { 
        self->lastX = x; 
        self->lastY = y; 
        return; 
    }
    
    float dx = float(x - self->lastX);
    float dy = float(y - self->lastY);
    self->lastX = x; 
    self->lastY = y;
    
    self->cam.yaw -= dx * 0.005f;
    self->cam.pitch -= dy * 0.005f;
    
    // Clamp pitch to prevent over-rotation
    if (self->cam.pitch < -1.2f) self->cam.pitch = -1.2f;
    if (self->cam.pitch > +1.2f) self->cam.pitch = +1.2f;
}

void App::mouseCB(GLFWwindow* window, int button, int action, int) {
    auto* self = static_cast<App*>(glfwGetWindowUserPointer(window));
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        self->dragging = (action == GLFW_PRESS);
    }
}

void App::scrollCB(GLFWwindow* window, double, double dy) {
    auto* self = static_cast<App*>(glfwGetWindowUserPointer(window));
    self->cam.dist *= std::exp(-0.1f * float(dy));
    
    // Clamp camera distance
    if (self->cam.dist < 2.0f)  self->cam.dist = 2.0f;
    if (self->cam.dist > 30.0f) self->cam.dist = 30.0f;
}
#endif

#ifndef HEADLESS_BUILD
void App::maybeUpdateWindowTitle(){
    if (!profilingEnabled || !window) return;
    using clock = std::chrono::high_resolution_clock;
    auto now = clock::now();
    if (lastTitleUpdate.time_since_epoch().count() == 0) lastTitleUpdate = now;
    sumTimes += curTimes; sumFrames++;
    curTimes.reset();
    double sec = std::chrono::duration<double>(now - lastTitleUpdate).count();
    if (sec < 0.5) return;
    double invF = sumFrames > 0 ? 1.0 / double(sumFrames) : 0.0;
    double fps = sec > 0 ? double(sumFrames) / sec : 0.0;
    double bp = sumTimes.broadphase * invF;
    double sv = (sumTimes.solve + sumTimes.floorSolve) * invF;
    double rd = sumTimes.render * invF;
    double bpPairs = sumTimes.bpPairs * invF;
    std::ostringstream ss; ss.setf(std::ios::fixed); ss.precision(1);
    ss << "Frame " << frameIndex << " | Rods: " << rods.size() 
       << " | KE: " << std::setprecision(6) << lastKE << " J"
       << " | FPS " << std::setprecision(0) << fps 
       << " | BP " << std::setprecision(1) << bp << " ms (pairs " << bpPairs << ")"
       << " | Solve " << sv << " ms | Render " << rd << " ms";
    glfwSetWindowTitle(window, ss.str().c_str());
    sumTimes = Times{}; sumFrames = 0; lastTitleUpdate = now;
}
#endif

void App::logCsvFrame(){
    if (!csvEnabled || !csvStream) return;
    if (!csvHeaderWritten) {
        csvStream << "frame,rods,integrate_ms,sleep_ms,broadphase_ms,bpCount_ms,bpPrefix_ms,bpFill_ms,bpPairs_ms,bpLongLong_ms,warmstart_ms,buildIslands_ms,solve_ms,floorSolve_ms,posCorrect_ms,pbcWrap_ms,render_ms,contacts,islands,KE,KE_after_integrate,KE_after_warmstart,KE_after_solve,KE_after_posCorrect,KE_after_pbcWrap,soft_PE,jn_sum,jt_sum,impulse_count,ent_pairs,ent_sum\n";
        csvHeaderWritten = true;
    }
    csvStream
        << frameIndex << ',' << rods.size() << ','
        << curTimes.integrate << ','
        << curTimes.sleepUpdate << ','
        << curTimes.broadphase << ','
        << curTimes.bpCount << ','
        << curTimes.bpPrefix << ','
        << curTimes.bpFill << ','
        << curTimes.bpPairs << ','
        << curTimes.bpLongLong << ','
        << curTimes.warmstart << ','
        << curTimes.buildIslands << ','
        << curTimes.solve << ','
        << curTimes.floorSolve << ','
        << curTimes.posCorrect << ','
        << curTimes.pbcWrap << ','
        << curTimes.render << ','
        << lastHitCount << ','
        << lastIslandCount << ','
        << lastKE << ','
        << keAfterIntegrate << ','
        << keAfterWarmstart << ','
        << keAfterSolve << ','
        << keAfterPosCorrect << ','
        << keAfterPBCWrap << ','
    << lastSoftPotentialEnergy << ','
        << g_diag_jn_sum << ','
        << g_diag_jt_sum << ','
        << g_diag_impulse_count << ','
        << lastEntanglementPairs << ','
        << lastEntanglementSum
        << '\n';
    if ((frameIndex & 0x3F) == 0) csvStream.flush();
}

double App::totalKE() const {
    double KE = 0.0;
    for (const auto& rb : rods) {
        double v2 = glm::dot(rb.v, rb.v);
        KE += 0.5 * double(rb.mass) * v2;
        glm::mat3 Iw = rb.R() * rb.I_body * glm::transpose(rb.R());
        glm::vec3 Iw_w = Iw * rb.w;
        KE += 0.5 * double(glm::dot(rb.w, Iw_w));
    }
    return KE;
}

// Per-rod logging implementation
void App::enablePerRod(const std::string& path, int maxFrames) {
    perRodPath = path.empty() ? std::string("perrod.csv") : path;
    perRodStream.open(perRodPath, std::ios::out | std::ios::trunc);
    if (!perRodStream) {
        std::cerr << "Failed to open per-rod CSV file: " << perRodPath << "\n";
        perRodEnabled = false; return;
    }
    perRodEnabled = true; perRodHeaderWritten = false;
    perRodMaxFrames = std::max(1, maxFrames);
    perRodWrittenFrames = 0;
    // Compute sampling skip when running headless (approximate total frames known)
    perRodSkip = 1;
    if (headless && headlessSteps > 0) perRodSkip = std::max(1, headlessSteps / perRodMaxFrames);
}

void App::logPerRodFrame() {
    if (!perRodEnabled || !perRodStream) return;
    if (!perRodHeaderWritten) {
        perRodStream << "frame,rod,px,py,pz,vx,vy,vz,wx,wy,wz,qw,qx,qy,qz,KE_lin,KE_rot,KE_total\n";
        perRodHeaderWritten = true;
    }
    if (perRodWrittenFrames >= perRodMaxFrames) return;
    if ((frameIndex % perRodSkip) != 0) return;
    for (size_t i = 0; i < rods.size(); ++i) {
        const auto& rb = rods[i];
        double ke_lin = 0.5 * double(rb.mass) * double(glm::dot(rb.v, rb.v));
        glm::mat3 Iw = rb.R() * rb.I_body * glm::transpose(rb.R());
        glm::vec3 Iw_w = Iw * rb.w;
        double ke_rot = 0.5 * double(glm::dot(Iw_w, rb.w));
        double ke_total = ke_lin + ke_rot;
        perRodStream
            << frameIndex << ',' << i << ','
            << rb.x.x << ',' << rb.x.y << ',' << rb.x.z << ','
            << rb.v.x << ',' << rb.v.y << ',' << rb.v.z << ','
            << rb.w.x << ',' << rb.w.y << ',' << rb.w.z << ','
            << rb.q.w << ',' << rb.q.x << ',' << rb.q.y << ',' << rb.q.z << ','
            << ke_lin << ',' << ke_rot << ',' << ke_total << '\n';
    }
    ++perRodWrittenFrames;
    if ((frameIndex & 0x3F) == 0) perRodStream.flush();
}

// Center-of-mass computation with PBC handling
glm::vec3 App::computeCOM() const {
    if (rods.empty()) return glm::vec3(0);
    
    double totalMass = 0.0;
    glm::dvec3 com(0.0);
    
    if (usePBC) {
        // Use circular/ring method for COM in periodic box
        // Map coordinates to angles on a circle, average using complex numbers
        // This correctly handles particles spread throughout the periodic domain
        
        const glm::vec3 boxSize = pbcMax - pbcMin;
        const double pi = 3.14159265358979323846;
        
        // For each dimension, compute angle-averaged position
        for (int k = 0; k < 3; ++k) {
            if (boxSize[k] <= 0.0f) {
                // Non-periodic dimension
                double sum = 0.0;
                totalMass = 0.0;
                for (const auto& rb : rods) {
                    double m = double(rb.mass);
                    totalMass += m;
                    sum += m * double(rb.x[k]);
                }
                com[k] = sum / totalMass;
            } else {
                // Periodic dimension: use circular mapping
                // Map position to angle: theta = 2*pi * (x - xmin) / L
                double cos_sum = 0.0;
                double sin_sum = 0.0;
                totalMass = 0.0;
                
                for (const auto& rb : rods) {
                    double m = double(rb.mass);
                    totalMass += m;
                    
                    // Normalize position to [0, 1] within box
                    double normalized = (double(rb.x[k]) - double(pbcMin[k])) / double(boxSize[k]);
                    double theta = 2.0 * pi * normalized;
                    
                    cos_sum += m * std::cos(theta);
                    sin_sum += m * std::sin(theta);
                }
                
                // Average angle
                double avg_theta = std::atan2(sin_sum / totalMass, cos_sum / totalMass);
                
                // Ensure avg_theta is in [0, 2*pi)
                if (avg_theta < 0.0) avg_theta += 2.0 * pi;
                
                // Convert back to position
                com[k] = double(pbcMin[k]) + (avg_theta / (2.0 * pi)) * double(boxSize[k]);
            }
        }
        
        return glm::vec3(com);
        
    } else {
        // Simple COM for non-periodic case
        for (const auto& rb : rods) {
            double m = double(rb.mass);
            totalMass += m;
            com += m * glm::dvec3(rb.x);
        }
        return glm::vec3(com / totalMass);
    }
}

void App::logCOMFrame() {
    if (!comEnabled || !comStream) return;
    if (!comHeaderWritten) {
        comStream << "frame,com_x,com_y,com_z,total_mass,num_rods\n";
        comHeaderWritten = true;
        std::cerr << "[COM] Header written\n";
    }
    
    glm::vec3 com = computeCOM();
    double totalMass = 0.0;
    for (const auto& rb : rods) totalMass += double(rb.mass);
    
    comStream << frameIndex << ',' 
              << com.x << ',' << com.y << ',' << com.z << ','
              << totalMass << ',' << rods.size() << '\n';
    
    if ((frameIndex & 0x3F) == 0) comStream.flush();
}

void App::logNetworkFrame() {
    if (!networkEnabled || !networkStream) return;
    if (!networkHeaderWritten) {
        networkStream << "frame,rod_i,rod_j,contact_x,contact_y,contact_z,normal_x,normal_y,normal_z,distance,"
                      << "force_a_x,force_a_y,force_a_z,force_b_x,force_b_y,force_b_z,"
                      << "friction_a_x,friction_a_y,friction_a_z,friction_b_x,friction_b_y,friction_b_z\n";
        networkHeaderWritten = true;
    }
    
    // Handle both soft and hard contact modes
    if (settings.physics.soft_contact.enabled) {
        if (settings.physics.use_mujoco_contact) {
            // MuJoCo soft contacts (no force data available yet)
            const auto& contacts = mjContactSolver.getContacts();
            for (const auto& c : contacts) {
                glm::vec3 midpoint = 0.5f * (c.pA + c.pB);
                networkStream << frameIndex << ',' 
                             << c.a << ',' << c.b << ','
                             << midpoint.x << ',' << midpoint.y << ',' << midpoint.z << ','
                             << c.n.x << ',' << c.n.y << ',' << c.n.z << ','
                             << c.dist << ','
                             << "0,0,0,0,0,0,0,0,0,0,0,0\n";  // Placeholder zeros for forces
            }
        } else {
            // Standard soft contacts - with force data
            const auto& contacts = softContactSolver.getContacts();
            for (const auto& c : contacts) {
                glm::vec3 midpoint = 0.5f * (c.point_a + c.point_b);
                networkStream << frameIndex << ',' 
                             << c.body_a << ',' << c.body_b << ','
                             << midpoint.x << ',' << midpoint.y << ',' << midpoint.z << ','
                             << c.normal.x << ',' << c.normal.y << ',' << c.normal.z << ','
                             << c.distance << ','
                             << c.force_a.x << ',' << c.force_a.y << ',' << c.force_a.z << ','
                             << c.force_b.x << ',' << c.force_b.y << ',' << c.force_b.z << ','
                             << c.friction_a.x << ',' << c.friction_a.y << ',' << c.friction_a.z << ','
                             << c.friction_b.x << ',' << c.friction_b.y << ',' << c.friction_b.z << '\n';
            }
        }
    } else {
        // Hard contacts - use hitsScratch from broadphase (no force data available)
        for (const auto& hit : hitsScratch) {
            if (hit.b < 0) continue; // Skip floor contacts
            
            networkStream << frameIndex << ',' 
                         << hit.a << ',' << hit.b << ','
                         << hit.c.point.x << ',' << hit.c.point.y << ',' << hit.c.point.z << ','
                         << hit.c.normal.x << ',' << hit.c.normal.y << ',' << hit.c.normal.z << ','
                         << -hit.c.penetration << ','
                         << "0,0,0,0,0,0,0,0,0,0,0,0\n";  // Placeholder zeros for forces
        }
    }
    
    if ((frameIndex & 0x3F) == 0) networkStream.flush();
}

void App::dumpContactsCSV(const std::vector<Hit>& hits, const char* stageLabel) {
    if (!contactDumpEnabled) return;
    if (!contactDumpStream.is_open()) {
        contactDumpStream.open(contactDumpPath, std::ios::out | std::ios::app);
        if (!contactDumpStream) { std::cerr << "Failed to open contact dump file: " << contactDumpPath << "\n"; contactDumpEnabled = false; return; }
    }
    if (!contactDumpHeaderWritten) {
        contactDumpStream << "frame,stage,idx,a,b,px,py,pz,nx,ny,nz,pen,shiftBx,shiftBy,shiftBz,vn,vt\n";
        contactDumpHeaderWritten = true;
    }
    size_t idx = 0;
    for (const auto& h : hits) {
        const RigidBody& A = rods[h.a];
        const RigidBody& B = (h.b >= 0) ? rods[h.b] : floorRB;
        glm::vec3 rA = h.c.point - A.x;
        glm::vec3 rB = h.c.point - (B.x + h.c.shiftB);
        glm::vec3 vA = A.v + glm::cross(A.w, rA);
        glm::vec3 vB = B.v + glm::cross(B.w, rB);
        glm::vec3 rel = vB - vA;
        float vn = glm::dot(rel, h.c.normal);
        glm::vec3 t = rel - h.c.normal * vn;
        float vt = glm::length(t);
        contactDumpStream
            << frameIndex << ',' << stageLabel << ',' << idx++ << ',' << h.a << ',' << h.b << ','
            << h.c.point.x << ',' << h.c.point.y << ',' << h.c.point.z << ','
            << h.c.normal.x << ',' << h.c.normal.y << ',' << h.c.normal.z << ','
            << h.c.penetration << ','
            << h.c.shiftB.x << ',' << h.c.shiftB.y << ',' << h.c.shiftB.z << ','
            << vn << ',' << vt << '\n';
    }
    if ((frameIndex & 0x3F) == 0) contactDumpStream.flush();
}

// ---- Simulation ----

void App::computeEntanglement() {
    std::vector<std::array<double,6>> segs;
    segs.reserve(rods.size());
    // Convert each rod to a segment [a,b] from capsule axis endpoints
    for (const auto& rb : rods) {
        if (rb.type != ShapeType::Capsule) continue;
        glm::vec3 a, b; rb.capsuleEndpoints(a, b);
        // If PBC enabled, map endpoints into primary box to keep segments short
        if (usePBC) {
            auto wrap = [&](glm::vec3& p){ wrapPos(p, pbcMin, pbcMax); };
            wrap(a); wrap(b);
        }
        segs.push_back({double(a.x), double(a.y), double(a.z), double(b.x), double(b.y), double(b.z)});
    }
#ifdef _OPENMP
    int threads = (entanglementThreads > 0 ? entanglementThreads : omp_get_max_threads());
#else
    int threads = (entanglementThreads > 0 ? entanglementThreads : 1);
#endif
    long long pairs = 0;
    double sum_abs = pairwise_abs_linking_sum_with_cutoff(segs, entanglementCutoff, &pairs, threads);
    lastEntanglementSum = sum_abs;
    lastEntanglementPairs = pairs;

    // Print to CLI for quick feedback
    if (entanglementEnabled) {
        if (CLI_UNIFIED_PRINT) {
            printCliStatus("[Entanglement]");
        } else {
            std::cout << "[Entanglement] frame=" << frameIndex
                      << " pairs=" << lastEntanglementPairs
                      << " sum=" << std::fixed << std::setprecision(6) << lastEntanglementSum
                      << std::defaultfloat << "\n";
        }
        // flush occasionally
        std::cout.flush();
    }
}

void App::physicsStep() {
#ifdef TRACY_ENABLE
    ZoneScopedN("PhysicsStep");
#endif
    // Reset diagnostic accumulators before this step
    resetFrameImpulseAccumulators();

    // Apply random forces if enabled
    if (useRandomForce) {
        for (auto& rb : rods) {
            rb.f   += float(sqrt(dt) * fSigma) * glm::vec3(normal_f(genRandomForce), normal_f(genRandomForce), normal_f(genRandomForce));
            rb.tau += float(sqrt(dt) * tauMag * normal_f(genRandomForce)) * uniform_dir_s2(genRandomForce);
            // rb.f +=   sqrt(dt) * fSigma * glm::vec3(normal_f(genRandomForce), normal_f(genRandomForce), normal_f(genRandomForce));
            // rb.tau += sqrt(dt) * tauMag * uniform_dir_s2(genRandomForce);
        }
    }

    if (settings.physics.hertz_mindlin.enabled) {
        // ===== Hertz-Mindlin contact model for spheres =====
        // Uses Velocity Verlet with contact forces
        // 1) contacts & forces at time t
        hertzMindlinSolver.detectContacts(rods);
        hertzMindlinSolver.computeForces(rods, dt);
        lastSoftPotentialEnergy = hertzMindlinSolver.getLastPotentialEnergy();
        lastHitCount = hertzMindlinSolver.getNumContacts();
        
        // 2) half-step velocities + position/orientation advance
        {
#ifdef TRACY_ENABLE
        ZoneScopedN("IntegrateHalfPos");
#endif
        ScopedAccum tIntegrateHP(profilingEnabled ? &curTimes.integrate : nullptr);
        parallel_for(0, rods.size(), [&](size_t i){
            if (!sleeping[i]) integrateHalfPos(rods[i], gravity, dt);
        });
        }
        // Clear forces before recompute at t+dt
        for (auto& rb : rods) { rb.f = glm::vec3(0); rb.tau = glm::vec3(0); }
        
        // 3) contacts & forces at time t+dt (updated positions)
        hertzMindlinSolver.detectContacts(rods);
        hertzMindlinSolver.computeForces(rods, dt);
        lastSoftPotentialEnergy = hertzMindlinSolver.getLastPotentialEnergy();
        lastHitCount = hertzMindlinSolver.getNumContacts();
        
        // 4) second half velocity update
        {
#ifdef TRACY_ENABLE
        ZoneScopedN("IntegrateHalfVel");
#endif
        ScopedAccum tIntegrateHV(profilingEnabled ? &curTimes.integrate : nullptr);
        parallel_for(0, rods.size(), [&](size_t i){
            if (!sleeping[i]) integrateHalfVel(rods[i], dt);
        });
        }
        keAfterSolve = totalKE();
    } else if (settings.physics.soft_contact.enabled) {
        // ===== Full Velocity Verlet sequence for soft contacts =====
        // 1) contacts & forces at time t
        if (settings.physics.use_mujoco_contact) {
            mjContactSolver.detectContacts(rods);
            mjContactSolver.computeForces(rods, dt);
            lastSoftPotentialEnergy = mjContactSolver.getLastPotentialEnergy(); // PE at configuration t
        } else {
            softContactSolver.detectContacts(rods);
            softContactSolver.computeForces(rods, dt);
            lastSoftPotentialEnergy = softContactSolver.getLastPotentialEnergy(); // PE at configuration t
        }
        // if (settings.physics.soft_contact.verbose && frameIndex % 200 == 0) {
        //     std::cout << "[Verlet] frame=" << frameIndex << " contacts(t)=" << softContactSolver.getNumContacts() << '\n';
        // }
        // 2) half-step velocities + position/orientation advance
        {
#ifdef TRACY_ENABLE
        ZoneScopedN("IntegrateHalfPos");
#endif
        ScopedAccum tIntegrateHP(profilingEnabled ? &curTimes.integrate : nullptr);
        parallel_for(0, rods.size(), [&](size_t i){
            if (!sleeping[i]) integrateHalfPos(rods[i], gravity, dt);
        });
        }
        // Clear forces before recompute at t+dt
        for (auto& rb : rods) { rb.f = glm::vec3(0); rb.tau = glm::vec3(0); }
        // 3) contacts & forces at time t+dt (updated positions)
        if (settings.physics.use_mujoco_contact) {
            mjContactSolver.detectContacts(rods);
            mjContactSolver.computeForces(rods, dt);
            lastSoftPotentialEnergy = mjContactSolver.getLastPotentialEnergy(); // overwrite with PE at configuration t+dt
        } else {
            softContactSolver.detectContacts(rods);
            softContactSolver.computeForces(rods, dt);
            lastSoftPotentialEnergy = softContactSolver.getLastPotentialEnergy(); // overwrite with PE at configuration t+dt
            lastHitCount = softContactSolver.getNumContacts(); // Update contact count for CSV logging
        }
        // if (settings.physics.soft_contact.verbose && frameIndex % 200 == 0) {
        //     std::cout << "[Verlet] frame=" << frameIndex << " contacts(t+dt)=" << softContactSolver.getNumContacts() << '\n';
        // }
        // 4) second half velocity update
        {
#ifdef TRACY_ENABLE
        ZoneScopedN("IntegrateSecondHalf");
#endif
        ScopedAccum tIntegrateSH(profilingEnabled ? &curTimes.integrate : nullptr);
        parallel_for(0, rods.size(), [&](size_t i){
            if (!sleeping[i]) integrateSecondHalf(rods[i], gravity, dt);
        });
        }
        // KE after full Verlet integrate
        keAfterIntegrate = totalKE();
    } else {
        // Original path (hard contacts or no soft penalty)
        {
#ifdef TRACY_ENABLE
        ZoneScopedN("IntegrateEuler");
#endif
        ScopedAccum tIntegrate(profilingEnabled ? &curTimes.integrate : nullptr);
        parallel_for(0, rods.size(), [&](size_t i){ if (!sleeping[i]) integrate(rods[i], gravity, dt); });
        }
        keAfterIntegrate = totalKE();
    }

    // Update sleeping state (after integration, before collision)
    {
#ifdef TRACY_ENABLE
    ZoneScopedN("SleepUpdate");
#endif
    ScopedAccum tSleep(profilingEnabled ? &curTimes.sleepUpdate : nullptr);
    for (size_t i = 0; i < rods.size(); ++i) {
        if (sleeping[i]) continue;
        float vs = glm::length(rods[i].v);
        float ws = glm::length(rods[i].w);
        if (vs < sleepLinThresh && ws < sleepAngThresh) {
            sleepTimer[i] += dt;
            if (sleepTimer[i] > sleepTimeThresh) {
                sleeping[i] = 1; sleepTimer[i] = 0.f;
                rods[i].v = glm::vec3(0);
                rods[i].w = glm::vec3(0);
            }
        } else {
            sleepTimer[i] = 0.f;
        }
    }
    }
    // end sleep update

    // Hard collision resolution (skipped if using soft contacts)
    if (!settings.physics.soft_contact.enabled) {
    
    // Broadphase: uniform grid within periodic box or AABB when not periodic
    auto& hits = hitsScratch; // reuse buffer across whole step
    hits.clear();
    hits.reserve(std::max<size_t>(rods.size(), 16) * 2);
    {
#ifdef TRACY_ENABLE
    ZoneScopedN("Broadphase");
#endif
    ScopedAccum tBroad(profilingEnabled ? &curTimes.broadphase : nullptr);

    const int numRods = static_cast<int>(rods.size());

    if (usePBC) {
        const glm::ivec3 N = gridDims(pbcMin, pbcMax, cellSize);
        // (Re)allocate flattened buffers if grid dims changed
        const size_t cellCount = size_t(N.x) * size_t(N.y) * size_t(N.z);
        if (N != gridN) {
            gridN = N;
            gridCounts.assign(cellCount, 0u);
            gridOffsets.assign(cellCount + 1, 0u);
            gridWrite.assign(cellCount, 0u);
            gridItems.clear(); gridItems.shrink_to_fit();
        } else {
            if (gridCounts.size() != cellCount) gridCounts.assign(cellCount, 0u);
            else std::fill(gridCounts.begin(), gridCounts.end(), 0u);
            if (gridOffsets.size() != cellCount + 1) gridOffsets.assign(cellCount + 1, 0u);
            else std::fill(gridOffsets.begin(), gridOffsets.end(), 0u);
            if (gridWrite.size() != cellCount) gridWrite.assign(cellCount, 0u);
            else std::fill(gridWrite.begin(), gridWrite.end(), 0u);
        }

        // Helpers and precompute per-rod cell spans and bounds
        auto wrapIndex = [&](int a, int dim) {
            if (a < 0) return a + dim; if (a >= dim) return a - dim; return a;
        };
        auto axisAABB = [&](const RigidBody& rb, glm::vec3& bmin, glm::vec3& bmax) {
            const glm::vec3 a = rb.axisY();
            const float h = rb.cap.h;
            const float r = rb.cap.r;
            const glm::vec3 ext = glm::vec3(r) + glm::abs(a) * h;
            bmin = rb.x - ext;
            bmax = rb.x + ext;
        };
        std::vector<glm::ivec3> i0s(numRods), i1s(numRods);
        std::vector<float> rBound(numRods);
        for (int i = 0; i < numRods; ++i) {
            glm::vec3 bmin, bmax; axisAABB(rods[i], bmin, bmax);
            i0s[i] = glm::floor((bmin - pbcMin) / cellSize);
            i1s[i] = glm::floor((bmax - pbcMin) / cellSize);
            rBound[i] = rods[i].cap.h + rods[i].cap.r;
        }
        const glm::vec3 boxSize = pbcMax - pbcMin;
        auto minImage = [&](glm::vec3 d) {
            for (int k = 0; k < 3; ++k) {
                const float L = boxSize[k];
                if (L > 0.0f) d[k] -= L * std::floor(d[k] / L + 0.5f);
            }
            return d;
        };

        auto tBPCountStart = std::chrono::high_resolution_clock::now();
        // Hybrid grid: classify long vs grid-inserted rods by span threshold
        const int LONG_SPAN = std::max(1, settings.scene.periodic.longSpan);
        std::vector<int> gridIdx; gridIdx.reserve(numRods);
        std::vector<int> longIdx; longIdx.reserve(numRods/8 + 4);
        for (int i = 0; i < numRods; ++i) {
            glm::ivec3 span = (i1s[i] - i0s[i]) + glm::ivec3(1);
            if (span.x > LONG_SPAN || span.y > LONG_SPAN || span.z > LONG_SPAN) longIdx.push_back(i);
            else gridIdx.push_back(i);
        }

        // Pass 1: count per-cell occupancy (only grid-inserted rods)
        for (int idx = 0; idx < (int)gridIdx.size(); ++idx) {
            int i = gridIdx[idx];
            const glm::ivec3& i0 = i0s[i];
            const glm::ivec3& i1 = i1s[i];
            for (int iz = i0.z; iz <= i1.z; ++iz)
            for (int iy = i0.y; iy <= i1.y; ++iy)
            for (int ix = i0.x; ix <= i1.x; ++ix) {
                const int cx = wrapIndex(ix, N.x);
                const int cy = wrapIndex(iy, N.y);
                const int cz = wrapIndex(iz, N.z);
                const size_t gi = linearIndex({cx,cy,cz}, N);
                ++gridCounts[gi];
            }
        }
        auto tBPCountEnd = std::chrono::high_resolution_clock::now();
        curTimes.bpCount += std::chrono::duration<double,std::milli>(tBPCountEnd - tBPCountStart).count();

        auto tBPPrefixStart = std::chrono::high_resolution_clock::now();
        // Prefix sum to offsets
        uint32_t totalItems = 0;
        for (size_t c = 0; c < cellCount; ++c) {
            gridOffsets[c] = totalItems;
            totalItems += gridCounts[c];
        }
        gridOffsets[cellCount] = totalItems;
        auto tBPPrefixEnd = std::chrono::high_resolution_clock::now();
        curTimes.bpPrefix += std::chrono::duration<double,std::milli>(tBPPrefixEnd - tBPPrefixStart).count();

        // Prepare items storage and write cursors
        gridItems.resize(totalItems);
        std::copy(gridOffsets.begin(), gridOffsets.begin() + cellCount, gridWrite.begin());

        auto tBPFillStart = std::chrono::high_resolution_clock::now();
        // Pass 2: fill items using write cursors (only grid-inserted rods)
        for (int idx = 0; idx < (int)gridIdx.size(); ++idx) {
            int i = gridIdx[idx];
            const glm::ivec3& i0 = i0s[i];
            const glm::ivec3& i1 = i1s[i];
            for (int iz = i0.z; iz <= i1.z; ++iz)
            for (int iy = i0.y; iy <= i1.y; ++iy)
            for (int ix = i0.x; ix <= i1.x; ++ix) {
                const int cx = wrapIndex(ix, N.x);
                const int cy = wrapIndex(iy, N.y);
                const int cz = wrapIndex(iz, N.z);
                const size_t gi = linearIndex({cx,cy,cz}, N);
                uint32_t w = gridWrite[gi]++;
                gridItems[w] = i;
            }
        }
        auto tBPFillEnd = std::chrono::high_resolution_clock::now();
        curTimes.bpFill += std::chrono::duration<double,std::milli>(tBPFillEnd - tBPFillStart).count();

        auto tBPPairsStart = std::chrono::high_resolution_clock::now();
        // Parallel neighbor checks with per-thread buffers and stamp-based de-dup per i
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        constexpr int MT_THRESHOLD = 200; // heuristic to avoid MT overhead on small N
        const int gridCount = (int)gridIdx.size();
        const unsigned T = (gridCount >= MT_THRESHOLD) ? std::min<unsigned>(g_thread_limit > 0 ? g_thread_limit : hw, std::max(1, gridCount)) : 1u;

        if (thHitsScratch.size() < T) thHitsScratch.resize(T);
        if (thSeenAt.size() < T) thSeenAt.resize(T);
        if (thCellSeenAt.size() < T) thCellSeenAt.resize(T);

        std::vector<std::thread> threads; threads.reserve(T);
        if (gridCount > 0) {
            const size_t chunk = (size_t(gridCount) + T - 1) / T;
            for (unsigned t = 0; t < T; ++t) {
                size_t s = size_t(t) * chunk;
                size_t e = std::min(s + chunk, size_t(gridCount));
                if (s >= e) break;
                // prep buffers
                auto& localHits = thHitsScratch[t];
                localHits.clear();
                localHits.reserve((e - s) * 8);
                auto& seenAt = thSeenAt[t];
                if (seenAt.size() != size_t(numRods)) seenAt.assign(size_t(numRods), -1);
                else std::fill(seenAt.begin(), seenAt.end(), -1);
                auto& cellSeenAt = thCellSeenAt[t];
                if (cellSeenAt.size() != cellCount) cellSeenAt.assign(cellCount, -1);
                else std::fill(cellSeenAt.begin(), cellSeenAt.end(), -1);

                threads.emplace_back([&, s, e, t]() {
                    auto& local = thHitsScratch[t];
                    auto& seen = thSeenAt[t];
                    auto& seenCell = thCellSeenAt[t];
                    for (size_t u = s; u < e; ++u) {
                        int i = gridIdx[u];
                        const glm::ivec3& i0 = i0s[i];
                        const glm::ivec3& i1 = i1s[i];
                        // Grid neighbors (27 cells)
                        for (int iz = i0.z; iz <= i1.z; ++iz)
                        for (int iy = i0.y; iy <= i1.y; ++iy)
                        for (int ix = i0.x; ix <= i1.x; ++ix) {
                            const int cx = wrapIndex(ix, N.x);
                            const int cy = wrapIndex(iy, N.y);
                            const int cz = wrapIndex(iz, N.z);
                            for (int dz = -1; dz <= 1; ++dz)
                            for (int dy = -1; dy <= 1; ++dy)
                            for (int dx = -1; dx <= 1; ++dx) {
                                const int nx = wrapIndex(cx + dx, N.x);
                                const int ny = wrapIndex(cy + dy, N.y);
                                const int nz = wrapIndex(cz + dz, N.z);
                                const size_t ni = linearIndex({nx,ny,nz}, N);
                                if (seenCell[ni] == i) continue; // already visited this neighbor cell for i
                                seenCell[ni] = i;
                                const uint32_t start = gridOffsets[ni];
                                const uint32_t end   = gridOffsets[ni+1];
                                for (uint32_t k = start; k < end; ++k) {
                                    int j = gridItems[k];
                                    if (j <= i) continue;
                                    if (seen[j] == i) continue; // already considered for this i
                                    // bounding-sphere precheck in PBC
                                    glm::vec3 d = rods[j].x - rods[i].x;
                                    d = minImage(d);
                                    const float R = rBound[i] + rBound[j];
                                    if (glm::dot(d,d) > R*R) { continue; }
                                    seen[j] = i;
                                    if (Contact c = collideCapsuleCapsule(rods[i], rods[j]); c.hit) {
                                        local.push_back({i, j, c});
                                    }
                                }
                            }
                        }
                        // Also test against all long rods not in the grid
                        for (int j : longIdx) {
                            if (j <= i) continue;
                            if (seen[j] == i) continue;
                            glm::vec3 d = rods[j].x - rods[i].x;
                            d = minImage(d);
                            const float R = rBound[i] + rBound[j];
                            if (glm::dot(d,d) > R*R) continue;
                            if (Contact c = collideCapsuleCapsule(rods[i], rods[j]); c.hit) {
                                local.push_back({i, j, c});
                            }
                        }
                    }
                });
            }
            for (auto& th : threads) th.join();
        } else {
            if (thHitsScratch.size() > 0) thHitsScratch[0].clear();
        }
        auto tBPPairsEnd = std::chrono::high_resolution_clock::now();
        curTimes.bpPairs += std::chrono::duration<double,std::milli>(tBPPairsEnd - tBPPairsStart).count();
        // Merge thread buffers
        size_t totalHits = 0; for (auto& v : thHitsScratch) totalHits += v.size();
        hits.reserve(hits.size() + totalHits);
        for (auto& v : thHitsScratch) { hits.insert(hits.end(), v.begin(), v.end()); }

        auto tBPLongStart = std::chrono::high_resolution_clock::now();
        // Long-long pairs: small naive pass (expected few)
        if (!longIdx.empty()) {
            for (size_t a = 0; a < longIdx.size(); ++a) {
                int i = longIdx[a];
                for (size_t b = a + 1; b < longIdx.size(); ++b) {
                    int j = longIdx[b];
                    glm::vec3 d = rods[j].x - rods[i].x;
                    d = minImage(d);
                    const float R = rBound[i] + rBound[j];
                    if (glm::dot(d,d) > R*R) continue;
                    if (Contact c = collideCapsuleCapsule(rods[i], rods[j]); c.hit) {
                        hits.push_back({i, j, c});
                    }
                }
            }
        }
        auto tBPLongEnd = std::chrono::high_resolution_clock::now();
        curTimes.bpLongLong += std::chrono::duration<double,std::milli>(tBPLongEnd - tBPLongStart).count();
    } else {
        // Non-PBC: naive all-pairs
        for (int i = 0; i < numRods; i++) {
            for (int j = i + 1; j < numRods; j++) {
                if (Contact contact = collideCapsuleCapsule(rods[i], rods[j]); contact.hit) {
                    hits.push_back({i, j, contact});
                }
            }
        }
        // Rod-floor collisions
        for (int i = 0; i < numRods; i++) {
            if (Contact contact = collideCapsuleFloor(rods[i], floorRB); contact.hit) {
                hits.push_back({i, -1, contact}); // -1 indicates floor collision
            }
        }
    }
    lastHitCount = hits.size();
    }
    // end broadphase

    // Velocity solving iterations (sequential for stability)
    // Warm start: apply cached impulses to seed solver
    {
#ifdef TRACY_ENABLE
    ZoneScopedN("WarmStart");
#endif
    ScopedAccum tWarm(profilingEnabled ? &curTimes.warmstart : nullptr);
    hitKeysScratch.resize(hits.size());
    for (size_t h = 0; h < hits.size(); ++h) {
        auto& hit = hits[h];
        if (hit.b >= 0) {
            // Wake bodies involved in contacts
            if (sleeping[hit.a]) wake(hit.a);
            if (sleeping[hit.b]) wake(hit.b);
            uint64_t key = pairKey(hit.a, hit.b);
            hitKeysScratch[h] = key;
            auto it = warmCache.find(key);
            if (it != warmCache.end()) {
                const auto& wi = it->second;
                applyWarmStart(rods[hit.a], rods[hit.b], hit.c, wi.jn, wi.jt, wi.tangent);
            }
        } else {
            hitKeysScratch[h] = 0ull;
        }
    }
    }
    // end warmstart

    // KE after warm-start
    keAfterWarmstart = totalKE();
    if (!contactDumpTriggeredThisFrame && contactDumpEnabled) {
        double dKE = keAfterWarmstart - keAfterIntegrate;
        bool up = dKE > contactDumpThresh, down = -dKE > contactDumpThresh;
        if ((contactDumpTrigger == 0 && (up || down)) || (contactDumpTrigger > 0 && up) || (contactDumpTrigger < 0 && down)) {
            dumpContactsCSV(hits, "after_warmstart");
            contactDumpTriggeredThisFrame = true;
        }
    }

    // Targeted high-speed restitution sweeps (normal-only) before main island solve
    if (solver.ngsNormalSweeps > 0) {
        std::vector<PairContact> pc; pc.reserve(hits.size());
        for (const auto& h : hits) pc.push_back({h.a, h.b, h.c});
        ngsRestitutionSweeps(pc, rods, solver);
    }

    // Solve and capture first-iteration impulses to update cache
    std::vector<AppliedImpulse> firstImp; firstImp.resize(hits.size());

    // Build islands from rod-rod contacts (b>=0). Floor hits handled separately.
    const int R = int(rods.size());
    std::vector<int> comp(R, -1);
    std::vector<std::vector<int>> adj(R);
    std::vector<size_t> floorHitIdx;
    floorHitIdx.reserve(hits.size());
    {
        #ifdef TRACY_ENABLE
        ZoneScopedN("BuildIslands");
        #endif
        ScopedAccum tBuild(profilingEnabled ? &curTimes.buildIslands : nullptr);
        for (size_t h = 0; h < hits.size(); ++h) {
            const auto& hit = hits[h];
            if (hit.b >= 0) {
                adj[hit.a].push_back(hit.b);
                adj[hit.b].push_back(hit.a);
            } else {
                floorHitIdx.push_back(h);
            }
        }
        std::vector<std::vector<int>> islands; islands.reserve(R);
        std::vector<int> stack; stack.reserve(R);
        for (int i = 0; i < R; ++i) {
            if (comp[i] != -1) continue;
            if (adj[i].empty()) { comp[i] = -2; continue; } // not in any rod-rod contact
            // BFS/DFS
            comp[i] = int(islands.size());
            islands.push_back({});
            auto& nodes = islands.back(); nodes.push_back(i);
            stack.clear(); stack.push_back(i);
            while (!stack.empty()) {
                int u = stack.back(); stack.pop_back();
                for (int v : adj[u]) {
                    if (comp[v] == -1) {
                        comp[v] = comp[i];
                        nodes.push_back(v);
                        stack.push_back(v);
                    }
                }
            }
        }
        // Collect hit indices per island
        std::vector<std::vector<size_t>> islandHits(islands.size());
        for (size_t h = 0; h < hits.size(); ++h) {
            const auto& hit = hits[h];
            if (hit.b < 0) continue;
            int ic = (hit.a < R) ? comp[hit.a] : -1;
            if (ic >= 0) islandHits[size_t(ic)].push_back(h);
        }

        // Solve islands in parallel when beneficial
        lastIslandCount = islandHits.size();
        auto solveIsland = [&](size_t idx){
            const auto& hlist = islandHits[idx];
            if (hlist.empty()) return;
            for (int iter = 0; iter < solver.velIters; ++iter) {
                for (size_t k = 0; k < hlist.size(); ++k) {
                    size_t h = hlist[k];
                    auto& hit = hits[h];
                    if (iter == 0) {
                        AppliedImpulse out{};
                        applyImpulse(rods[hit.a], rods[hit.b], hit.c, &out);
                        firstImp[h] = out;
                    } else {
                        applyImpulse(rods[hit.a], rods[hit.b], hit.c);
                    }
                }
            }
        };

        const size_t islandCount = islandHits.size();
        {
            #ifdef TRACY_ENABLE
            ZoneScopedN("Solve");
            #endif
            ScopedAccum tSolve(profilingEnabled ? &curTimes.solve : nullptr);
            if (islandCount > 1) {
                parallel_for(0, islandCount, solveIsland);
            } else if (islandCount == 1) {
                solveIsland(0);
            }
        }
    }

    // Solve floor hits (no warm cache capture)
    if (!usePBC && !floorHitIdx.empty()) {
        #ifdef TRACY_ENABLE
        ZoneScopedN("FloorSolve");
        #endif
        ScopedAccum tFloor(profilingEnabled ? &curTimes.floorSolve : nullptr);
        for (int iter = 0; iter < solver.velIters; ++iter) {
            for (size_t h : floorHitIdx) {
                applyImpulse(rods[hits[h].a], floorRB, hits[h].c);
            }
        }
    }

    // KE after solve (after islands and floor)
    keAfterSolve = totalKE();
    if (!contactDumpTriggeredThisFrame && contactDumpEnabled) {
        double dKE = keAfterSolve - keAfterWarmstart;
        bool up = dKE > contactDumpThresh, down = -dKE > contactDumpThresh;
        if ((contactDumpTrigger == 0 && (up || down)) || (contactDumpTrigger > 0 && up) || (contactDumpTrigger < 0 && down)) {
            dumpContactsCSV(hits, "after_solve");
            contactDumpTriggeredThisFrame = true;
        }
    }

    // Positional correction
    {
    #ifdef TRACY_ENABLE
    ZoneScopedN("PositionalCorrection");
    #endif
    ScopedAccum tPos(profilingEnabled ? &curTimes.posCorrect : nullptr);
    // Stabilization tuning: scale beta if many contacts; clamp to betaMin
    SolverConfig pcCfg = solver;
    if (lastHitCount >= (size_t)betaHighContactThresh) {
        pcCfg.baumgarte = std::max(betaMin, solver.baumgarte * betaHighContactScale);
    } else {
        pcCfg.baumgarte = std::max(betaMin, solver.baumgarte);
    }
    for (auto& hit : hits) {
        if (hit.b >= 0) {
            positionalCorrection(rods[hit.a], rods[hit.b], hit.c, pcCfg);
        } else if (!usePBC) {
            positionalCorrection(rods[hit.a], floorRB, hit.c, pcCfg);
        }
    }
    }

    // KE after positional correction
    keAfterPosCorrect = totalKE();
    if (!contactDumpTriggeredThisFrame && contactDumpEnabled) {
        double dKE = keAfterPosCorrect - keAfterSolve;
        bool up = dKE > contactDumpThresh, down = -dKE > contactDumpThresh;
        if ((contactDumpTrigger == 0 && (up || down)) || (contactDumpTrigger > 0 && up) || (contactDumpTrigger < 0 && down)) {
            dumpContactsCSV(hits, "after_posCorrect");
            contactDumpTriggeredThisFrame = true;
        }
    }

    // PBC wrap after corrections to avoid drift across boundaries
    if (usePBC) {
        #ifdef TRACY_ENABLE
        ZoneScopedN("PBCWrap");
        #endif
        ScopedAccum tWrap(profilingEnabled ? &curTimes.pbcWrap : nullptr);
        for (auto& rb : rods) {
            wrapPos(rb.x, pbcMin, pbcMax);
        }
    }

    // Track kinetic energy after PBC wrap (final state for the frame)
    keAfterPBCWrap = totalKE();
    lastKE = keAfterPBCWrap;

    } // end hard collision resolution (!soft_contact.enabled)

    // Update adaptive metrics
    lastFrameKEDelta = keAfterPBCWrap - prevFrameKE;
    prevFrameKE = keAfterPBCWrap;

    // After physics step bookkeeping
    lastKE = totalKE();
    // Optionally compute entanglement every N frames
    if (entanglementEnabled && (frameIndex % entanglementEvery == 0)) {
        computeEntanglement();
    }
    logCsvFrame();
    logSoftPEFrame();
    logCOMFrame();
    logNetworkFrame();
}

#ifndef HEADLESS_BUILD
void App::renderFrame() {
    #ifdef TRACY_ENABLE
    ZoneScopedN("RenderFrame");
    #endif
    int width, height; 
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);
    glClearColor(settings.render.bg.r, settings.render.bg.g, settings.render.bg.b, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    ScopedAccum tRender(profilingEnabled ? &curTimes.render : nullptr);
    float aspect = (height > 0) ? float(width) / float(height) : 1.0f;
    glm::mat4 projection = glm::perspective(glm::radians(50.0f), aspect, 0.05f, 100.0f);
    glm::mat4 view = cam.view();

    RenderUniforms uniforms;
    uniforms.P = projection; 
    uniforms.V = view;
    uniforms.eye = cam.eye();
    uniforms.lightDir = glm::normalize(settings.render.lightDir);
    uniforms.useGrid = settings.render.grid.enabled;
    uniforms.gridScale = settings.render.grid.scale;
    uniforms.gridC1 = settings.render.grid.c1;
    uniforms.gridC2 = settings.render.grid.c2;

    // Color palette for different rods
    static const glm::vec3 rodColors[] = {
        {0.30f, 0.70f, 1.00f}, // Blue
        {1.00f, 0.55f, 0.25f}, // Orange
        {0.60f, 0.90f, 0.40f}, // Green
        {0.90f, 0.40f, 0.80f}, // Purple
        {0.95f, 0.85f, 0.30f}, // Yellow
    };
    constexpr int numColors = sizeof(rodColors) / sizeof(rodColors[0]);

    // Draw rods: use instancing beyond a small threshold to reduce draw calls
    const size_t N = rods.size();
    const size_t INST_THRESHOLD = 64;
    if (N > INST_THRESHOLD) {
        // Separate bodies by shape type for instanced rendering
        std::vector<glm::mat4> capsuleModels, sphereModels;
        std::vector<glm::vec3> capsuleColors, sphereColors;
        
        for (size_t i = 0; i < N; ++i) {
            glm::mat4 model = rods[i].modelMatrix();
            glm::vec3 color = rodColors[i % numColors];
            
            if (rods[i].type == ShapeType::Sphere) {
                sphereModels.push_back(model);
                sphereColors.push_back(color);
            } else {
                capsuleModels.push_back(model);
                capsuleColors.push_back(color);
            }
        }
        
        // Draw capsules
        if (!capsuleModels.empty()) {
            RenderUniforms common = uniforms; common.useGrid = false;
            rnd.drawInstances(cyl, capsuleModels.data(), capsuleColors.data(), capsuleModels.size(), common);
        }
        
        // Draw spheres
        if (!sphereModels.empty()) {
            RenderUniforms common = uniforms; common.useGrid = false;
            rnd.drawInstances(sphere, sphereModels.data(), sphereColors.data(), sphereModels.size(), common);
        }
    } else {
        for (size_t i = 0; i < N; ++i) {
            uniforms.M = rods[i].modelMatrix();
            uniforms.color = rodColors[i % numColors];
            uniforms.useGrid = false;
            
            // Select mesh based on shape type
            const Mesh& mesh = (rods[i].type == ShapeType::Sphere) ? sphere : cyl;
            rnd.draw(mesh, uniforms);
        }
    }

    if (!usePBC) {
        // Draw floor
        uniforms.M = floorRB.modelMatrix(); 
        uniforms.useGrid = true; 
        uniforms.color = {1.0f, 1.0f, 1.0f};
        rnd.draw(cube, uniforms);
    }
}
#endif

void App::stepWithSubsteps() {
    // Determine substeps (adaptive if enabled)
    int baseSub = std::max(1, settings.physics.substeps);
    int substeps = baseSub;
    if (adaptiveSubsteps) {
        bool heavyContacts = (lastHitCount >= (size_t)asHitThresh);
        bool keUp = (lastFrameKEDelta > asKEUpThresh);
        bool keDown = (lastFrameKEDelta < asKEDownThresh);
        if (heavyContacts || keUp || keDown) substeps = asMax; else substeps = std::max(asMin, baseSub);
    }
    float frameDt = dt;
    float subDt = frameDt / float(substeps);
    float saveDt = dt;
    for (int s = 0; s < substeps; ++s) {
        dt = subDt;
        physicsStep();
    }
    dt = saveDt;
}

int App::run() {
#ifdef HEADLESS_BUILD
    // Force headless mode when built without graphics
    headless = true;
    resetScene();
    std::cout << "Running headless for " << headlessSteps << " steps...\n";
    for (int step = 0; step < headlessSteps; ++step) {
        if (!paused) stepWithSubsteps();
        if (entanglementEnabled && (frameIndex % entanglementEvery == 0)) {
            computeEntanglement();
        }
        if (perRodEnabled) logPerRodFrame();
        // CSV logging if enabled
        logCsvFrame();
        ++frameIndex;
        if ((step & 0x3FF) == 0) {
            printCliStatus("[Headless] ");
        }
    }
    if (csvEnabled) csvStream.flush();
    if (perRodEnabled) perRodStream.flush();
    std::cout << "Headless run complete. Frames=" << frameIndex << "\n";
    return 0;
#else
    if (headless) {
        // Headless: don't initialize window/graphics, run tight physics loop
        resetScene();
        std::cout << "Running headless for " << headlessSteps << " steps...\n";
        for (int step = 0; step < headlessSteps; ++step) {
            if (!paused) stepWithSubsteps();
            if (entanglementEnabled && (frameIndex % entanglementEvery == 0)) {
                computeEntanglement();
            }
            if (perRodEnabled) logPerRodFrame();
            // CSV logging if enabled
            logCsvFrame();
            ++frameIndex;
            if ((step & 0x3FF) == 0) {
                printCliStatus("[Headless] ");
            }
        }
        // ensure CSV flushed and closed
        if (csvEnabled) csvStream.flush();
        if (perRodEnabled) perRodStream.flush();
        if (softPEEnabled) softPEStream.flush();
        if (comEnabled) comStream.flush();
        if (networkEnabled) networkStream.flush();
        std::cout << "Headless run complete. Frames=" << frameIndex << "\n";
        return 0;
    }

    if (!initWindow()) return -1;
    if (!initGraphics()) return -1;
    resetScene();
    
    lastTitleUpdate = std::chrono::high_resolution_clock::now();

    auto lastTime = std::chrono::high_resolution_clock::now();
    double accumulator = 0.0;
    #ifdef TRACY_ENABLE
    tracy::SetThreadName("Main");
    #endif
    
    while (!glfwWindowShouldClose(window)) {
        auto currentTime = std::chrono::high_resolution_clock::now();
        double deltaTime = std::chrono::duration<double>(currentTime - lastTime).count();
        lastTime = currentTime;

        // Limit frame time to prevent spiral of death
        accumulator = std::min(accumulator + deltaTime, 1.0 / 15.0);
        
        while (accumulator >= dt) {
            if (!paused) {
                stepWithSubsteps();
            }
            accumulator -= dt;
        }

        renderFrame();
        if (perRodEnabled) logPerRodFrame();
        // CSV logging uses current per-frame times before they are reset by maybeUpdateWindowTitle
        logCsvFrame();
        maybeUpdateWindowTitle();
        glfwSwapBuffers(window);
        glfwPollEvents();
        #ifdef TRACY_ENABLE
        FrameMark;
        #endif
        ++frameIndex;
    }
    if (csvEnabled) csvStream.flush();
    if (perRodEnabled) perRodStream.flush();
    if (softPEEnabled) softPEStream.flush();
    if (comEnabled) comStream.flush();
    if (networkEnabled) networkStream.flush();
    glfwTerminate();
    return 0;
#endif
}

void App::setConfig(const AppCfg& config) {
    settings = config;
}

void App::printCliStatus(const std::string& prefix) const {
    if (!CLI_UNIFIED_PRINT) return;
    // frame, rods, KE, ent_pairs, ent_sum
    std::cout << prefix
              << "frame=" << frameIndex
              << " rods=" << rods.size()
              << " KE=" << std::fixed << std::setprecision(6) << lastKE
              << " ent_pairs=" << lastEntanglementPairs
              << " ent_sum=" << std::fixed << std::setprecision(6) << lastEntanglementSum
              << std::defaultfloat << "\n";
}

// ---- Main Function ----

int main(int argc, char** argv) {
    std::string scenePath = std::string(ASSETS_DIR) + "/scenes/default.json";
    bool enableProfile = false;
    std::string csvPath;
    bool headlessFlag = false;
    int headlessSteps = 1000;
    std::string perRodPath;
    int perRodMaxFrames = 1000;
    bool disableWarmStart = false;
    bool enableEnergySafeguard = false;
    // CLI overrides
    int cliSubsteps = -1;
    int cliVelIters = -1;
    int cliSplitImpulse = -1; // -1=unset, 0=false, 1=true
    int cliSplitOrient  = -1; // -1=unset, 0=false, 1=true
    int cliSeed = 0; // 0 means no override
    int cliNgsSweeps = -1;
    float cliNgsVth = -1.0f;
    int cliThreads = -1;
    float cliDt = -1.0f;
    std::string cliContactDumpPath;
    double cliContactDumpThresh = -1.0;
    std::string cliContactDumpTrig;
    std::string cliSoftPEPath; // optional soft potential energy output file
    std::string cliCOMPath;    // center-of-mass tracking
    std::string cliNetworkPath; // contact network tracking
    // Adaptive substeps CLI
    int cliAdaptive = -1; // -1 unset, 0 off, 1 on
    int cliAsMin = -1, cliAsMax = -1, cliAsHit = -1;
    double cliAsKEUp = std::numeric_limits<double>::quiet_NaN();
    double cliAsKEDown = std::numeric_limits<double>::quiet_NaN();
    // Stabilization CLI
    float cliBetaMin = std::numeric_limits<float>::quiet_NaN();
    int   cliBetaHit = -1;
    float cliBetaScale = std::numeric_limits<float>::quiet_NaN();
    // Entanglement CLI
    bool cliEntanglement = false;
    double cliEntanglementCutoff = -1.0;
    int cliEntanglementPeriod = 60;
    int cliEntanglementThreads = 0;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            std::cout << "Rod Dynamics 3D - Rigid Body Simulation\n\n";
            std::cout << "Usage: " << argv[0] << " [options]\n\n";
            std::cout << "Scene Configuration:\n";
            std::cout << "  --scene <path>              Load scene from JSON file (default: assets/scenes/default.json)\n\n";
            std::cout << "Execution Modes:\n";
            std::cout << "  --headless                  Run without graphics\n";
            std::cout << "  --steps <N>                 Number of steps for headless mode (default: 1000)\n\n";
            std::cout << "Output & Logging:\n";
            std::cout << "  --csv [path]                Enable CSV profile output (default: profile.csv)\n";
            std::cout << "  --perrod [path]             Enable per-rod trajectory CSV (default: perrod.csv)\n";
            std::cout << "  --perrod-max <N>            Max frames to log in per-rod CSV (default: 1000)\n";
            std::cout << "  --soft-pe <path>            Log soft contact potential energy\n";
            std::cout << "  --com <path>                Track center-of-mass (default: com.csv)\n";
            std::cout << "  --network <path>            Track contact network (default: network.csv)\n";
            std::cout << "  --contact-dump <path>       Log contact details to CSV\n";
            std::cout << "  --contact-dump-thresh <T>   KE threshold for contact dump\n";
            std::cout << "  --contact-dump-trigger <M>  Trigger mode: any|up|down\n\n";
            std::cout << "Solver Configuration:\n";
            std::cout << "  --dt <float>                Timestep size\n";
            std::cout << "  --substeps <N>              Substeps per frame\n";
            std::cout << "  --velIters <N>              Velocity solver iterations\n";
            std::cout << "  --ngs-sweeps <N>            NGS sweeps for constraint solver\n";
            std::cout << "  --ngs-vth <float>           NGS velocity threshold\n";
            std::cout << "  --split-impulse             Enable split impulse\n";
            std::cout << "  --no-split-impulse          Disable split impulse\n";
            std::cout << "  --split-orient              Enable split orientation correction\n";
            std::cout << "  --no-split-orient           Disable split orientation correction\n";
            std::cout << "  --no-warmstart              Disable warm starting\n";
            std::cout << "  --energy-safeguard          Enable energy safeguard\n\n";
            std::cout << "Adaptive Substeps:\n";
            std::cout << "  --adaptive-substeps         Enable adaptive substeps\n";
            std::cout << "  --no-adaptive-substeps      Disable adaptive substeps\n";
            std::cout << "  --as-min <N>                Minimum substeps\n";
            std::cout << "  --as-max <N>                Maximum substeps\n";
            std::cout << "  --as-hit <N>                Hit count threshold\n";
            std::cout << "  --as-ke-up <float>          KE increase threshold\n";
            std::cout << "  --as-ke-down <float>        KE decrease threshold\n\n";
            std::cout << "Stabilization:\n";
            std::cout << "  --beta-min <float>          Minimum beta value\n";
            std::cout << "  --beta-hit <N>              Hit count for beta adjustment\n";
            std::cout << "  --beta-scale <float>        Beta scaling factor\n\n";
            std::cout << "Entanglement:\n";
            std::cout << "  --entanglement              Enable entanglement computation\n";
            std::cout << "  --ent-cutoff <float>        Distance cutoff for linking (default: 5.0)\n";
            std::cout << "  --ent-period <N>            Compute every N frames (default: 60)\n";
            std::cout << "  --ent-threads <N>           Thread count (0=auto)\n\n";
            std::cout << "Other:\n";
            std::cout << "  --seed <N>                  Random seed\n";
            std::cout << "  --threads <N>               Thread limit (0=auto)\n";
            std::cout << "  --profile                   Enable profiling\n";
            std::cout << "  --help, -h                  Show this help message\n\n";
            std::cout << "Examples:\n";
            std::cout << "  " << argv[0] << " --scene my_scene.json\n";
            std::cout << "  " << argv[0] << " --headless --steps 10000 --csv --perrod\n";
            std::cout << "  " << argv[0] << " --scene confined.json --dt 0.001 --velIters 20\n";
            return 0;
        } else if (std::string(argv[i]) == "--scene" && i + 1 < argc) {
            scenePath = argv[++i];
        } else if (std::string(argv[i]) == "--profile") {
            enableProfile = true;
        } else if (std::string(argv[i]) == "--csv") {
            if (i + 1 < argc && argv[i+1][0] != '-') {
                csvPath = argv[++i];
            } else {
                csvPath = "profile.csv";
            }
        } else if (std::string(argv[i]) == "--headless") {
            headlessFlag = true;
        } else if (std::string(argv[i]) == "--steps" && i + 1 < argc) {
            headlessSteps = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "--perrod") {
            if (i + 1 < argc && argv[i+1][0] != '-') perRodPath = argv[++i]; else perRodPath = "perrod.csv";
        } else if (std::string(argv[i]) == "--perrod-max" && i + 1 < argc) {
            perRodMaxFrames = std::max(1, std::stoi(argv[++i]));
        } else if (std::string(argv[i]) == "--no-warmstart") {
            disableWarmStart = true;
        } else if (std::string(argv[i]) == "--energy-safeguard") {
            enableEnergySafeguard = true;
        } else if (std::string(argv[i]) == "--substeps" && i + 1 < argc) {
            cliSubsteps = std::max(1, std::stoi(argv[++i]));
        } else if (std::string(argv[i]) == "--velIters" && i + 1 < argc) {
            cliVelIters = std::max(1, std::stoi(argv[++i]));
        } else if (std::string(argv[i]) == "--split-impulse") {
            cliSplitImpulse = 1;
        } else if (std::string(argv[i]) == "--no-split-impulse") {
            cliSplitImpulse = 0;
        } else if (std::string(argv[i]) == "--split-orient") {
            cliSplitOrient = 1;
        } else if (std::string(argv[i]) == "--no-split-orient") {
            cliSplitOrient = 0;
        } else if (std::string(argv[i]) == "--seed" && i + 1 < argc) {
            cliSeed = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "--ngs-sweeps" && i + 1 < argc) {
            cliNgsSweeps = std::max(0, std::stoi(argv[++i]));
        } else if (std::string(argv[i]) == "--ngs-vth" && i + 1 < argc) {
            cliNgsVth = std::max(0.0f, std::stof(argv[++i]));
        } else if (std::string(argv[i]) == "--threads" && i + 1 < argc) {
            cliThreads = std::max(0, std::stoi(argv[++i]));
        } else if (std::string(argv[i]) == "--dt" && i + 1 < argc) {
            cliDt = std::max(0.0f, std::stof(argv[++i]));
        } else if (std::string(argv[i]) == "--contact-dump" && i + 1 < argc) {
            cliContactDumpPath = argv[++i];
        } else if (std::string(argv[i]) == "--contact-dump-thresh" && i + 1 < argc) {
            cliContactDumpThresh = std::stod(argv[++i]);
        } else if (std::string(argv[i]) == "--contact-dump-trigger" && i + 1 < argc) {
            cliContactDumpTrig = argv[++i]; // any|up|down
        } else if (std::string(argv[i]) == "--soft-pe" && i + 1 < argc) {
            cliSoftPEPath = argv[++i];
        } else if (std::string(argv[i]) == "--com" && i + 1 < argc) {
            cliCOMPath = argv[++i];
        } else if (std::string(argv[i]) == "--network" && i + 1 < argc) {
            cliNetworkPath = argv[++i];
        } else if (std::string(argv[i]) == "--adaptive-substeps") {
            cliAdaptive = 1;
        } else if (std::string(argv[i]) == "--no-adaptive-substeps") {
            cliAdaptive = 0;
        } else if (std::string(argv[i]) == "--as-min" && i + 1 < argc) {
            cliAsMin = std::max(1, std::stoi(argv[++i]));
        } else if (std::string(argv[i]) == "--as-max" && i + 1 < argc) {
            cliAsMax = std::max(1, std::stoi(argv[++i]));
        } else if (std::string(argv[i]) == "--as-hit" && i + 1 < argc) {
            cliAsHit = std::max(0, std::stoi(argv[++i]));
        } else if (std::string(argv[i]) == "--as-keup" && i + 1 < argc) {
            cliAsKEUp = std::stod(argv[++i]);
        } else if (std::string(argv[i]) == "--as-kedown" && i + 1 < argc) {
            cliAsKEDown = std::stod(argv[++i]);
        } else if (std::string(argv[i]) == "--beta-min" && i + 1 < argc) {
            cliBetaMin = std::stof(argv[++i]);
        } else if (std::string(argv[i]) == "--beta-hit" && i + 1 < argc) {
            cliBetaHit = std::max(0, std::stoi(argv[++i]));
        } else if (std::string(argv[i]) == "--beta-scale" && i + 1 < argc) {
            cliBetaScale = std::stof(argv[++i]);
        } else if (std::string(argv[i]) == "--entanglement") {
            cliEntanglement = true;
        } else if (std::string(argv[i]) == "--entanglement-cutoff" && i + 1 < argc) {
            cliEntanglementCutoff = std::stod(argv[++i]);
        } else if (std::string(argv[i]) == "--entanglement-period" && i + 1 < argc) {
            cliEntanglementPeriod = std::max(1, std::stoi(argv[++i]));
        } else if (std::string(argv[i]) == "--entanglement-threads" && i + 1 < argc) {
            cliEntanglementThreads = std::max(0, std::stoi(argv[++i]));
        }
    }

    AppCfg settings = defaultAppCfg();
    
    // Load scene configuration (keep defaults if load fails)
    if (!loadConfigFromFile(scenePath, settings)) {
        std::cerr << "Warning: Could not load scene file '" << scenePath 
                  << "', using defaults.\n";
    }

    // Apply CLI overrides to settings
    if (cliSubsteps > 0) settings.physics.substeps = cliSubsteps;
    if (cliVelIters > 0) settings.physics.solver.velIters = cliVelIters;
    if (cliSplitImpulse != -1) settings.physics.solver.splitImpulse = (cliSplitImpulse != 0);
    if (cliSplitOrient  != -1) settings.physics.solver.splitOrient  = (cliSplitOrient  != 0);
    if (cliSeed != 0) {
        settings.scene.populate.seed = cliSeed;
        settings.scene.randomInit.seed = cliSeed;
    }
    if (cliNgsSweeps >= 0) settings.physics.solver.ngsNormalSweeps = cliNgsSweeps;
    if (cliNgsVth >= 0.0f) settings.physics.solver.ngsHighVThresh = cliNgsVth;
    if (cliThreads >= 0) g_thread_limit = cliThreads;
    if (cliDt > 0.0f) settings.physics.dt = cliDt;

    App app;
    app.setConfig(settings);
    app.setProfiling(enableProfile);
    if (!csvPath.empty()) app.enableCsv(csvPath);
    if (headlessFlag) {
        // Provide default csv path if none given
        if (csvPath.empty()) app.enableCsv("profile_headless.csv");
        app.setHeadless(true);
        app.setHeadlessSteps(headlessSteps);
    }
    // Enable per-rod logging if requested (do after headless/steps set so sampling skip can be computed)
    if (!perRodPath.empty()) {
        app.enablePerRod(perRodPath, perRodMaxFrames);
    }
    if (!cliSoftPEPath.empty()) {
        app.enableSoftPE(cliSoftPEPath);
        std::cerr << "[app] Soft contact potential energy logging enabled: " << cliSoftPEPath << "\n";
    }
    if (!cliCOMPath.empty()) {
        app.enableCOM(cliCOMPath);
        std::cerr << "[app] Center-of-mass tracking enabled: " << cliCOMPath << "\n";
    }
    if (!cliNetworkPath.empty()) {
        app.enableNetwork(cliNetworkPath);
        std::cerr << "[app] Contact network tracking enabled: " << cliNetworkPath << "\n";
    }
    // Global toggles for solver diagnostics/testing
    if (disableWarmStart) {
        std::cerr << "[app] Warm-start disabled via --no-warmstart\n";
        setWarmstartEnabled(false);
    }
    if (enableEnergySafeguard) {
        std::cerr << "[app] Energy safeguard enabled via --energy-safeguard\n";
        setEnergySafeguard(true);
    }
    if (cliSubsteps > 1) {
        std::cerr << "[app] Substeps set to " << cliSubsteps << " via --substeps\n";
        settings.physics.substeps = cliSubsteps;
    }
    if (cliSplitImpulse == 1) {
        std::cerr << "[app] Split impulse enabled via --split-impulse\n";
        settings.physics.solver.splitImpulse = true;
    }
    if (cliSplitOrient == 1) {
        std::cerr << "[app] Split orientation correction enabled via --split-orient\n";
        settings.physics.solver.splitOrient = true;
    }
    if (cliThreads >= 0) {
        std::cerr << "[app] Thread limit set to " << cliThreads << " via --threads\n";
    }
    if (cliDt > 0.0f) {
        std::cerr << "[app] Timestep set to " << cliDt << " via --dt\n";
    }
    // Apply adaptive substeps and stabilization config to app
    if (cliAdaptive != -1) {
        app.enableAdaptiveSubsteps(cliAdaptive != 0);
        std::cerr << "[app] Adaptive substeps " << ((cliAdaptive != 0)?"on":"off") << "\n";
    }
    if (cliAsMin > 0 || cliAsMax > 0 || cliAsHit >= 0 || !std::isnan(cliAsKEUp) || !std::isnan(cliAsKEDown)) {
        app.setAdaptiveParams(cliAsMin>0?cliAsMin:1, cliAsMax>0?cliAsMax:settings.physics.substeps>0?settings.physics.substeps:1, cliAsHit>=0?cliAsHit:INT32_MAX, !std::isnan(cliAsKEUp)?cliAsKEUp:1e300, !std::isnan(cliAsKEDown)?cliAsKEDown:-1e300);
    }
    if (!std::isnan(cliBetaMin) || cliBetaHit >= 0 || !std::isnan(cliBetaScale)) {
        app.setStabilization(!std::isnan(cliBetaMin)?cliBetaMin:0.0f, cliBetaHit>=0?cliBetaHit:INT32_MAX, !std::isnan(cliBetaScale)?cliBetaScale:1.0f);
        std::cerr << "[app] Stabilization configured\n";
    }
    // Apply entanglement config
    if (cliEntanglement) {
        app.setEntanglement(true, cliEntanglementCutoff, cliEntanglementPeriod, cliEntanglementThreads);
        std::cerr << "[app] Entanglement enabled with cutoff=" << cliEntanglementCutoff << ", period=" << cliEntanglementPeriod << ", threads=" << cliEntanglementThreads << "\n";
    }
    
    App a;
    a.setHeadless(headlessFlag);
    a.setHeadlessSteps(headlessSteps);
    if (!csvPath.empty()) a.enableCsv(csvPath);
    if (!perRodPath.empty()) a.enablePerRod(perRodPath, perRodMaxFrames);
    if (!cliSoftPEPath.empty()) a.enableSoftPE(cliSoftPEPath);
    if (!cliCOMPath.empty()) a.enableCOM(cliCOMPath);
    if (!cliNetworkPath.empty()) a.enableNetwork(cliNetworkPath);
    a.setProfiling(enableProfile);
    if (cliAdaptive != -1) a.enableAdaptiveSubsteps(cliAdaptive == 1);
    if (cliAsMin > 0 || cliAsMax > 0 || cliAsHit >= 0 || !std::isnan(cliAsKEUp) || !std::isnan(cliAsKEDown)) {
        // Provide defaults consistent with App's internal defaults
        int defMin = 1, defMax = 1, defHit = INT32_MAX;
        double defUp = 1e300, defDown = -1e300;
        a.setAdaptiveParams(cliAsMin > 0 ? cliAsMin : defMin,
                            cliAsMax > 0 ? cliAsMax : defMax,
                            cliAsHit >= 0 ? cliAsHit : defHit,
                            std::isnan(cliAsKEUp) ? defUp : cliAsKEUp,
                            std::isnan(cliAsKEDown) ? defDown : cliAsKEDown);
    }
    if (!std::isnan(cliBetaMin) || cliBetaHit >= 0 || !std::isnan(cliBetaScale)) {
        float defBetaMin = 0.0f; int defBetaHit = INT32_MAX; float defBetaScale = 1.0f;
        a.setStabilization(std::isnan(cliBetaMin) ? defBetaMin : cliBetaMin,
                           cliBetaHit >= 0 ? cliBetaHit : defBetaHit,
                           std::isnan(cliBetaScale) ? defBetaScale : cliBetaScale);
    }
    // Entanglement options
    if (cliEntanglement) {
        a.setEntanglement(true, cliEntanglementCutoff, cliEntanglementPeriod, cliEntanglementThreads);
    }
    a.setConfig(settings);
    return a.run();
}
