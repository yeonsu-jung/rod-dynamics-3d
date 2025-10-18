/**
 * @file main.cpp
 * @brief 3D Rod Dynamics Simulation - Main Application
 */

#include <glad/glad.h>
#include <GLFW/glfw3.h>

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

#include "gfx/renderer.hpp"
#include "gfx/mesh.hpp"
#include "gfx/camera.hpp"

#include "config/config.hpp"

#ifndef ASSETS_DIR
#define ASSETS_DIR "."
#endif

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

private:
    // ---- Window and OpenGL ----
    GLFWwindow* window = nullptr;
    bool vsync = true;

    // ---- Renderer and meshes ----
    Renderer rnd;
    Mesh cube, cyl;

    // ---- Camera ----
    OrbitCamera cam;
    bool dragging = false;
    double lastX = 0.0, lastY = 0.0;

    // ---- Simulation ----
    bool paused = false;
    glm::vec3 gravity{0.0f, -10.0f, 0.0f};
    float dt = 1.0f / 600.0f;
    AppCfg settings{};
    SolverConfig solver{};

    // Periodic box
    bool usePBC = false;
    glm::vec3 pbcMin{-3,-1,-3}, pbcMax{3,3,3};
    float cellSize = 0.6f; // broadphase grid cell size

    // ---- Physics objects ----
    std::vector<RigidBody> rods;
    RigidBody floorRB;

    // ---- Initialization ----
    bool initWindow(int width = 1200, int height = 800, 
                   const char* title = "Rigid Bodies – Rods (Capsules)");
    bool initGraphics();

    // ---- Scene management ----
    static RigidBody createRod(const BodyCfg& config);
    void resetScene();

    // ---- Event callbacks ----
    static void keyCB(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void cursorCB(GLFWwindow* window, double x, double y);
    static void mouseCB(GLFWwindow* window, int button, int action, int mods);
    static void scrollCB(GLFWwindow* window, double xoffset, double yoffset);

    // ---- Profiling (built-in lightweight) ----
    struct Times {
        double integrate = 0, sleepUpdate = 0, broadphase = 0, warmstart = 0,
               buildIslands = 0, solve = 0, floorSolve = 0, posCorrect = 0, pbcWrap = 0, render = 0;
        void reset(){ integrate = sleepUpdate = broadphase = warmstart = buildIslands = solve = floorSolve = posCorrect = pbcWrap = render = 0; }
        Times& operator+=(const Times& o){ integrate+=o.integrate; sleepUpdate+=o.sleepUpdate; broadphase+=o.broadphase; warmstart+=o.warmstart; buildIslands+=o.buildIslands; solve+=o.solve; floorSolve+=o.floorSolve; posCorrect+=o.posCorrect; pbcWrap+=o.pbcWrap; render+=o.render; return *this; }
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
    void maybeUpdateWindowTitle();

    // CSV logging
    bool csvEnabled = false;
    std::ofstream csvStream;
    std::string csvPath;
    bool csvHeaderWritten = false;
    uint64_t frameIndex = 0;
    size_t lastHitCount = 0;
    size_t lastIslandCount = 0;
    void logCsvFrame();
    
    // ---- Simulation ----
    struct Hit { 
        int a = -1, b = -1; 
        Contact c{}; 
    };
    
    void physicsStep();
    void renderFrame();

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
        const unsigned T = std::min<unsigned>(hw, (unsigned)N);
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
};

// ---- Implementation ----

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
    cyl = makeCappedCylinderMesh(40);
    return true;
}

void App::setConfig(const AppCfg& config) {
    settings = config;
    cam.yaw = settings.render.yaw;
    cam.pitch = settings.render.pitch;
    cam.dist = settings.render.dist;
}

RigidBody App::createRod(const BodyCfg& config) {
    // rot_quat is glm::vec4 {w,x,y,z} (resolved by config)
    glm::quat q(config.rot_quat.x, config.rot_quat.y, config.rot_quat.z, config.rot_quat.w);
    RigidBody rb = RigidBody::makeRodLD(config.pos, q, config.density, config.length, 
                                       config.diameter, config.restitution, config.friction);
    rb.v = config.v_lin; 
    rb.w = config.v_ang;
    return rb;
}

void App::resetScene() {
    dt = settings.physics.dt;
    gravity = settings.physics.gravity;
    solver = settings.physics.solver;

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
        BodyCfg base = settings.scene.bodies.empty() ? BodyCfg{} : settings.scene.bodies.front();
        // const float L = base.length; // unused
        const float D = base.diameter;
        const float spacing = settings.scene.populate.spacingMul * D;

        if (settings.scene.populate.grid) {
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
        } else {
            // Uniform random centroids over the domain [pbcMin, pbcMax]
            rods.reserve(N);
            const glm::vec3 extent = pbcMax - pbcMin;
            for (int i = 0; i < N; ++i) {
                BodyCfg cfg = base;
                glm::vec3 r{urand(gen), urand(gen), urand(gen)};
                cfg.pos = pbcMin + r * extent;
                // Random orientation similar to grid path
                glm::vec3 axis = glm::normalize(glm::vec3(urand(gen)-0.5f, urand(gen)-0.5f, urand(gen)-0.5f));
                float angle = (urand(gen) - 0.5f) * 3.14159f; // [-pi/2, pi/2]
                glm::quat q = glm::angleAxis(angle, axis);
                cfg.rot_quat = glm::vec4(q.w, q.x, q.y, q.z);
                rods.push_back(createRod(cfg));
            }
        }
    } else if (!settings.scene.bodies.empty()) {
        rods.reserve(settings.scene.bodies.size());
        for (const auto& bodyConfig : settings.scene.bodies) {
            rods.push_back(createRod(bodyConfig));
        }
    } else {
        // Fallback: two default rods if scene is empty
        BodyCfg rodA{}, rodB{};
        
        rodA.pos = {-1.6f, 0.6f, 0.0f}; 
        rodA.rot_quat = {1, 0, 0, 0};
        rodA.density = 1000.0f; 
        rodA.length = 0.5f; 
        rodA.diameter = 0.10f; 
        rodA.restitution = 0.15f; 
        rodA.friction = 0.6f; 
        rodA.v_lin = {+2.2f, 0, 0};
        
        rodB.pos = {+1.2f, 1.0f, 0.2f}; 
        rodB.rot_quat = {1, 0, 0, 0};
        rodB.density = 1000.0f; 
        rodB.length = 0.5f; 
        rodB.diameter = 0.10f; 
        rodB.restitution = 0.15f; 
        rodB.friction = 0.6f; 
        rodB.v_lin = {-1.0f, 0, 0};
        
        rods.push_back(createRod(rodA));
        rods.push_back(createRod(rodB));
    }

    if (useRandomInit) {
        // Gaussian translational velocities, Uniform S2 direction with fixed magnitude for angular
        std::random_device rd;
        std::mt19937 gen(settings.scene.randomInit.seed ? settings.scene.randomInit.seed : rd());
        std::normal_distribution<float> normal(0.0f, settings.scene.randomInit.vSigma);
        std::uniform_real_distribution<float> uni(0.0f, 1.0f);
        const float wSpeed = settings.scene.randomInit.wSpeed;

        auto uniform_dir_s2 = [&](std::mt19937& g) {
            float u = 2.0f * uni(g) - 1.0f; // cos(theta) in [-1,1]
            float phi = 2.0f * float(M_PI) * uni(g);
            float s = std::sqrt(std::max(0.0f, 1.0f - u*u));
            return glm::vec3(s * std::cos(phi), u, s * std::sin(phi));
        };

        for (auto& rb : rods) {
            rb.v = { normal(gen), normal(gen), normal(gen) };
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
}

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
    std::ostringstream ss; ss.setf(std::ios::fixed); ss.precision(1);
    ss << "Rods: " << rods.size() << " | FPS " << std::setprecision(0) << fps << std::setprecision(1)
       << " | BP " << bp << " ms | Solve " << sv << " ms | Render " << rd << " ms";
    glfwSetWindowTitle(window, ss.str().c_str());
    // reset accumulators
    sumTimes = Times{}; sumFrames = 0; lastTitleUpdate = now;
}

void App::logCsvFrame(){
    if (!csvEnabled || !csvStream) return;
    if (!csvHeaderWritten) {
        csvStream << "frame,rods,integrate_ms,sleep_ms,broadphase_ms,warmstart_ms,buildIslands_ms,solve_ms,floorSolve_ms,posCorrect_ms,pbcWrap_ms,render_ms,contacts,islands\n";
        csvHeaderWritten = true;
    }
    csvStream
        << frameIndex << ',' << rods.size() << ','
        << curTimes.integrate << ','
        << curTimes.sleepUpdate << ','
        << curTimes.broadphase << ','
        << curTimes.warmstart << ','
        << curTimes.buildIslands << ','
        << curTimes.solve << ','
        << curTimes.floorSolve << ','
        << curTimes.posCorrect << ','
        << curTimes.pbcWrap << ','
        << curTimes.render << ','
        << lastHitCount << ','
        << lastIslandCount
        << '\n';
    // Light flush to keep file current without huge overhead
    if ((frameIndex & 0x3F) == 0) csvStream.flush();
}

// ---- Simulation ----

void App::physicsStep() {
#ifdef TRACY_ENABLE
    ZoneScopedN("PhysicsStep");
#endif
    // Integrate all rods (parallelized)
    {
#ifdef TRACY_ENABLE
    ZoneScopedN("Integrate");
#endif
    ScopedAccum tIntegrate(profilingEnabled ? &curTimes.integrate : nullptr);
    parallel_for(0, rods.size(), [&](size_t i){
        if (!sleeping[i]) {
            integrate(rods[i], gravity, dt);
        }
    });
    }
    // end integrate scope

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

        auto wrapIndex = [&](int a, int dim) {
            if (a < 0) return a + dim; if (a >= dim) return a - dim; return a;
        };

        auto axisAABB = [&](const RigidBody& rb, glm::vec3& bmin, glm::vec3& bmax) {
            const glm::vec3 a = rb.axisY(); // already unit from rotation matrix
            const float h = rb.cap.h;
            const float r = rb.cap.r;
            const glm::vec3 ext = glm::vec3(r) + glm::abs(a) * h; // tight AABB extents
            bmin = rb.x - ext;
            bmax = rb.x + ext;
        };

        // Precompute per-rod overlapped cell ranges and bounding-sphere radii
        std::vector<glm::ivec3> i0s(numRods), i1s(numRods);
        std::vector<float> rBound(numRods);
        for (int i = 0; i < numRods; ++i) {
            glm::vec3 bmin, bmax; axisAABB(rods[i], bmin, bmax);
            i0s[i] = glm::floor((bmin - pbcMin) / cellSize);
            i1s[i] = glm::floor((bmax - pbcMin) / cellSize);
            rBound[i] = rods[i].cap.h + rods[i].cap.r; // bounding sphere radius
        }

        const glm::vec3 boxSize = pbcMax - pbcMin;
        auto minImage = [&](glm::vec3 d) {
            for (int k = 0; k < 3; ++k) {
                const float L = boxSize[k];
                if (L > 0.0f) d[k] -= L * std::floor(d[k] / L + 0.5f);
            }
            return d;
        };

        // Hybrid grid: classify long vs grid-inserted rods by span threshold
        const int LONG_SPAN = 4; // if a rod spans more than this many cells in any axis, treat as long
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

        // Prefix sum to offsets
        uint32_t totalItems = 0;
        for (size_t c = 0; c < cellCount; ++c) {
            gridOffsets[c] = totalItems;
            totalItems += gridCounts[c];
        }
        gridOffsets[cellCount] = totalItems;
        // Prepare items storage and write cursors
        gridItems.resize(totalItems);
        std::copy(gridOffsets.begin(), gridOffsets.begin() + cellCount, gridWrite.begin());

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

        // Parallel neighbor checks with per-thread buffers and stamp-based de-dup per i
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        constexpr int MT_THRESHOLD = 200; // heuristic to avoid MT overhead on small N
        const int gridCount = (int)gridIdx.size();
        const unsigned T = (gridCount >= MT_THRESHOLD) ? std::min<unsigned>(hw, std::max(1, gridCount)) : 1u;

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
                            seen[j] = i;
                            if (Contact c = collideCapsuleCapsule(rods[i], rods[j]); c.hit) {
                                local.push_back({i, j, c});
                            }
                        }
                    }
                });
            }
            for (auto& th : threads) th.join();
        } else {
            // Ensure thread-local containers are cleared if not used
            if (thHitsScratch.size() > 0) thHitsScratch[0].clear();
        }
        // Merge thread buffers
        size_t totalHits = 0; for (auto& v : thHitsScratch) totalHits += v.size();
        hits.reserve(hits.size() + totalHits);
        for (auto& v : thHitsScratch) { hits.insert(hits.end(), v.begin(), v.end()); }

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

    // Positional correction
    {
    #ifdef TRACY_ENABLE
    ZoneScopedN("PositionalCorrection");
    #endif
    ScopedAccum tPos(profilingEnabled ? &curTimes.posCorrect : nullptr);
    for (auto& hit : hits) {
        if (hit.b >= 0) {
            positionalCorrection(rods[hit.a], rods[hit.b], hit.c, solver);
        } else if (!usePBC) {
            positionalCorrection(rods[hit.a], floorRB, hit.c, solver);
        }
    }
    }

    // Rebuild warm cache with current-frame impulses (from first iteration)
    if (!hits.empty()) {
        std::unordered_map<uint64_t, AppliedImpulse> nextCache;
        nextCache.reserve(hits.size()*2);
        for (size_t h = 0; h < hits.size(); ++h) {
            if (hits[h].b < 0) continue;
            uint64_t key = hitKeysScratch[h];
            nextCache[key] = firstImp[h];
        }
        warmCache.swap(nextCache);
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
}

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
        std::vector<glm::mat4> models; models.resize(N);
        std::vector<glm::vec3> colors; colors.resize(N);
        for (size_t i = 0; i < N; ++i) {
            models[i] = rods[i].modelMatrix();
            colors[i] = rodColors[i % numColors];
        }
        // Disable grid for instances
        RenderUniforms common = uniforms; common.useGrid = false;
        rnd.drawInstances(cyl, models.data(), colors.data(), N, common);
    } else {
        for (size_t i = 0; i < N; ++i) {
            uniforms.M = rods[i].modelMatrix();
            uniforms.color = rodColors[i % numColors];
            uniforms.useGrid = false;
            rnd.draw(cyl, uniforms);
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

int App::run() {
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
                physicsStep();
            }
            accumulator -= dt;
        }

        renderFrame();
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
    
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

// ---- Main Function ----

int main(int argc, char** argv) {
    std::string scenePath = std::string(ASSETS_DIR) + "/scenes/default.json";
    bool enableProfile = false;
    std::string csvPath;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--scene" && i + 1 < argc) {
            scenePath = argv[++i];
        } else if (std::string(argv[i]) == "--profile") {
            enableProfile = true;
        } else if (std::string(argv[i]) == "--csv") {
            if (i + 1 < argc && argv[i+1][0] != '-') {
                csvPath = argv[++i];
            } else {
                csvPath = "profile.csv";
            }
        }
    }

    AppCfg settings = defaultAppCfg();
    
    // Load scene configuration (keep defaults if load fails)
    if (!loadConfigFromFile(scenePath, settings)) {
        std::cerr << "Warning: Could not load scene file '" << scenePath 
                  << "', using defaults.\n";
    }

    App app;
    app.setConfig(settings);
    app.setProfiling(enableProfile);
    if (!csvPath.empty()) app.enableCsv(csvPath);
    
    return app.run();
}
