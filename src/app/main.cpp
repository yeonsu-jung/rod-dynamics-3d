/**
 * @file main.cpp
 * @brief 3D Rod Dynamics Simulation - Main Application
 */

#ifndef HEADLESS_BUILD
#include <GLFW/glfw3.h>
#include <glad/glad.h>
#endif

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#ifndef HEADLESS_BUILD
#include "lodepng.h"
#endif
#include <omp.h>

// Global thread limit (0 = use hardware_concurrency)
int g_thread_limit = 0;

// True when the user explicitly passed --threads N (N>0); suppresses auto-serial
bool g_user_threads_set = false;

// Global quiet flag: when true, suppress non-essential verbose output
extern bool gQuiet;

// Suppress periodic headless runtime progress while keeping startup logs.
bool gHeadlessProgressEnabled = true;

#ifdef _OPENMP
static void logOpenMpStartupConfig() {
  const int runtimeDefaultMax = omp_get_max_threads();
  if (g_thread_limit > 0) {
    omp_set_num_threads(g_thread_limit);
  }
  const int effectiveRuntimeMax = omp_get_max_threads();

  std::cout << "[Info] OpenMP enabled. Runtime max: " << runtimeDefaultMax;
  if (g_thread_limit > 0) {
    std::cout << " | requested thread limit: " << g_thread_limit
              << " | effective runtime max: " << effectiveRuntimeMax;
  } else {
    std::cout << " | requested thread limit: auto"
              << " | effective runtime max: " << effectiveRuntimeMax;
  }
  std::cout << "\n";
}
#endif

// Unified CLI print control: when true, use a single formatted status line
static const bool CLI_UNIFIED_PRINT = true;

#ifdef TRACY_ENABLE
#if __has_include(<tracy/Tracy.hpp>)
#include <tracy/Tracy.hpp>
#elif __has_include(<Tracy.hpp>)
#include <Tracy.hpp>
#else
// Provide no-op stubs if Tracy headers are not available
#ifndef ZoneScopedN
#define ZoneScopedN(x)
#endif
#ifndef FrameMark
#define FrameMark
#endif
namespace tracy {
inline void SetThreadName(const char *) {}
} // namespace tracy
#endif
#endif
// Physics globals
float g_minMoment = 0.0f;

#include "physics/collision.hpp"
#include "physics/contact_geometry.hpp"
#include "physics/contact_geometry_adapters.hpp"
#include "physics/hertz_mindlin.hpp"
#include "physics/integrator.hpp"
#include "physics/mujoco_contact.hpp"
#include "physics/nsc_contact.hpp"
#include "physics/rigid_body.hpp"
#include "physics/soft_contact.hpp"

#ifndef HEADLESS_BUILD
#include "gfx/camera.hpp"
#include "gfx/mesh.hpp"
#include "gfx/renderer.hpp"
#endif

#include "config/config.hpp"
#include "app/python_api.hpp"

#ifndef ASSETS_DIR
#define ASSETS_DIR "."
#endif

// entanglement-cpp (external)
#include "linking_number.h"
// Forward declaration for pairwise API defined in
// external/entanglement-cpp/linking_number.cpp double
// pairwise_abs_linking_sum_with_cutoff(const
// std::vector<std::array<double,6>>&, double, long long*, int);
extern double pairwise_abs_linking_sum_with_cutoff(
    const std::vector<std::array<double, 6>> &rods_array, double cutoff,
    long long *out_pairs, int num_threads);

#ifdef GLAD_GL_KHR_debug
static void GLAPIENTRY glDebugCallback(GLenum, GLenum, GLuint, GLenum sev,
                                       GLsizei, const GLchar *msg,
                                       const void *) {
  if (sev == GL_DEBUG_SEVERITY_NOTIFICATION)
    return;
  std::cerr << "[GL] " << msg << "\n";
}
#endif

class App {
public:
  bool showLabel = true;
  float rodDiameterOverride = -1.0f;
  bool autoExitAfterPlayback = false;
  App() = default;
  ~App() = default;

  int run();
  // Playback a snapshots NDJSON file (no physics) with optional frame dumping
  int runPlayback(const std::string &ndjsonPath, const std::string &dumpDir,
                  int playbackFps, bool orbit, float orbitSpeed, bool camPosSet,
                  const glm::vec3 &camPos, bool camTargetSet,
                  const glm::vec3 &camTarget, bool autoFrame, float scale,
                  float camScale, bool skipDupes, bool hideWindow, bool noFloor,
                  int exportStride, const std::string &moviePath,
                  bool autoExit);
  void setConfig(const AppCfg &config);
  void setProfiling(bool enabled) {
    profilingEnabled = enabled;
    if (!gQuiet) std::cerr << "[Debug] setProfiling: " << enabled << "\n";
  }
  void setInitCsvPath(const std::string &path) { initCsvPath = path; }
  void setSaveInitPath(const std::string &path) { saveInitPath = path; }
  void setInitStateCsvPath(const std::string &path) { initStateCsvPath = path; }
  void setLogOnSnapshotOnly(bool on) { logOnSnapshotOnly = on; }
  void enableCsv(const std::string &path) {
    csvPath = path.empty() ? std::string("profile.csv") : path;
    csvStream.open(csvPath, std::ios::out | std::ios::trunc);
    if (!csvStream) {
      std::cerr << "Failed to open CSV file: " << csvPath << "\n";
      csvEnabled = false;
      return;
    }
    csvEnabled = true;
    csvHeaderWritten = false;
  }

  void setNetworkStride(int stride) { networkStride = std::max(1, stride); }
  void setNetworkMax(int maxFrames) { networkMaxFrames = maxFrames; }
  void setOutputStride(int stride) { outputStride = std::max(1, stride); }
  void setOutputMax(int maxFrames) { outputMaxFrames = maxFrames; }
  void setPerRodStride(int stride) {
    perRodSkip = std::max(1, stride);
    explicitPerRodStride = true;
  }

  // Enable contact dump diagnostics from CLI
  void configureContactDump(const std::string &path, double thresh,
                            int trigger) {
    contactDumpEnabled = true;
    contactDumpPath = path;
    if (thresh >= 0.0)
      contactDumpThresh = thresh;
    contactDumpTrigger = trigger; // 0:any, +1:up, -1:down
                                  // open lazily in dumpContactsCSV
  }

  // Headless control
  void setHeadless(bool h) { headless = h; }
  void setHeadlessSteps(int s) {
    headlessSteps = s;
    if (perRodEnabled && perRodMaxFrames > 0 && !explicitPerRodStride)
      perRodSkip = std::max(1, headlessSteps / perRodMaxFrames);
    if (testRodEndpointsEnabled && testRodEndpointsMaxFrames > 0 &&
        !explicitTestRodEndpointsStride)
      testRodEndpointsStride =
          std::max(1, headlessSteps / testRodEndpointsMaxFrames);
  }
  void setStopKEThreshold(double v) { stopKEThreshold = v; }
  void setStopKEMinSteps(int n) { stopKEMinSteps = std::max(0, n); }
  void setStopKEAvgWindow(int n) { stopKEAvgWindow = std::max(1, n); }
  void setStopSlideVelThreshold(double v) { stopSlideVelThreshold = v; }
  void setStopSlideVelMinSteps(int n) { stopSlideVelMinSteps = std::max(0, n); }

  // Render stride control
  void setRenderStride(int s) { renderStride = std::max(1, s); }
  // CSV stride control
  void setCsvStride(int s) { csvStride = std::max(1, s); }
  void setCliStatusStride(int s) { cliStatusStride = std::max(1, s); }
  // Enable per-rod CSV output (path, maximum sampled frames)
  void enablePerRod(const std::string &path, int maxFrames);
  // Enable lightweight endpoint trajectory CSV for a single test rod.
  void enableTestRodEndpoints(const std::string &path);
  void setTestRodIndex(int idx) { testRodIndex = idx; }
  void setTestRodEndpointsStride(int stride) {
    testRodEndpointsStride = std::max(1, stride);
    explicitTestRodEndpointsStride = true;
  }
  void setTestRodEndpointsMaxFrames(int maxFrames) {
    testRodEndpointsMaxFrames = std::max(1, maxFrames);
    if (testRodEndpointsEnabled && headless && headlessSteps > 0 &&
        !explicitTestRodEndpointsStride)
      testRodEndpointsStride =
          std::max(1, headlessSteps / testRodEndpointsMaxFrames);
  }

  void enableAdaptiveSubsteps(bool on) { adaptiveSubsteps = on; }

  void enableReptSummary(const std::string& path, int rodIdx = 0) {
    reptSummaryEnabled = true;
    reptSummaryPath = path;
    reptRodIdx = rodIdx;
  }
  void setAdaptiveParams(int minS, int maxS, int hitThresh, double dKEUp,
                         double dKEDown) {
    asMin = std::max(1, minS);
    asMax = std::max(asMin, maxS);
    asHitThresh = hitThresh;
    asKEUpThresh = dKEUp;
    asKEDownThresh = dKEDown;
  }
  void setStabilization(float beta_min, int highContactThresh,
                        float betaScale) {
    betaMin = std::max(0.0f, beta_min);
    betaHighContactThresh = highContactThresh;
    betaHighContactScale = std::max(0.0f, betaScale);
  }
  void setDebugNormalVelocity(bool enabled) { debugNormalVelocity = enabled; }
  void setDebugNormalVelocityCsv(const std::string &path) {
    debugNormalVelocityCsvPath = path;
  }
  void setEnergyBalanceCsv(const std::string &path) {
    energyBalanceCsvPath = path;
  }
  void configureEarlyPairDiagnostics(const EarlyPairDiagnosticsCfg &cfg) {
    earlyPairDiagnostics = cfg;
    earlyPairDiagnostics.stride = std::max(1, earlyPairDiagnostics.stride);
    earlyPairDiagnostics.geomspace_samples =
        std::max(1, earlyPairDiagnostics.geomspace_samples);
    std::string &scheduleMode = earlyPairDiagnostics.schedule_mode;
    if (scheduleMode != "linear" && scheduleMode != "geomspace") {
      if (!gQuiet) {
        std::cerr << "[early-pair] Unknown schedule_mode='" << scheduleMode
                  << "'. Falling back to linear.\n";
      }
      scheduleMode = "linear";
    }

    earlyPairSampleFrames.clear();
    if (scheduleMode != "geomspace") {
      return;
    }

    const int startStep = earlyPairDiagnostics.start_step;
    const int endStep = earlyPairDiagnostics.end_step;
    if (endStep >= 0 && endStep < startStep) {
      earlyPairSampleFrames.insert(startStep);
      return;
    }

    const int positiveStart = std::max(1, startStep);
    const int positiveEnd = std::max(positiveStart, endStep);
    earlyPairSampleFrames.insert(startStep);
    earlyPairSampleFrames.insert(endStep);
    if (startStep <= 0 && endStep >= 1) {
      earlyPairSampleFrames.insert(1);
    }

    if (positiveEnd == positiveStart) {
      earlyPairSampleFrames.insert(positiveStart);
      return;
    }

    const int sampleCount = earlyPairDiagnostics.geomspace_samples;
    const double logStart = std::log(static_cast<double>(positiveStart));
    const double logEnd = std::log(static_cast<double>(positiveEnd));
    const double denom = std::max(1, sampleCount - 1);
    for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
      const double alpha = static_cast<double>(sampleIndex) / denom;
      const double value = std::exp(logStart + alpha * (logEnd - logStart));
      const int frame = std::clamp(
          static_cast<int>(std::llround(value)), positiveStart, positiveEnd);
      earlyPairSampleFrames.insert(frame);
    }
  }
  bool shouldSampleEarlyPairDiagnostics(int frame) const {
    if (!earlyPairDiagnostics.enabled)
      return false;
    if (frame < earlyPairDiagnostics.start_step)
      return false;
    if (earlyPairDiagnostics.end_step >= 0 &&
        frame > earlyPairDiagnostics.end_step)
      return false;
    if (earlyPairDiagnostics.schedule_mode == "geomspace") {
      return earlyPairSampleFrames.find(frame) != earlyPairSampleFrames.end();
    }
    int offset = frame - earlyPairDiagnostics.start_step;
    return offset % std::max(1, earlyPairDiagnostics.stride) == 0;
  }

  // Entanglement controls
  void setEntanglement(bool enable, double cutoff, int period, int threads) {
    entanglementEnabled = enable;
    entanglementCutoff = cutoff;
    entanglementEvery = std::max(1, period);
    entanglementThreads = threads;
  }
  void setPaused(bool p) { paused = p; }
  void setBackgroundColor(const glm::vec3 &color) {
    settings.render.bg = color;
  }
  void setPerturbationRod(int idx) { perturbationRodIndex = idx; }

  // Configure velocity override to be applied on scene reset
  void setOverrideVelocity(int idx, const glm::vec3 &vel) {
    overrideVelEnabled = true;
    overrideVelId = idx;
    overrideVel = vel;
  }

  // Configure angular velocity override to be applied on scene reset
  void setOverrideAngVelocity(int idx, const glm::vec3 &w) {
    overrideAngVelEnabled = true;
    overrideAngVelId = idx;
    overrideAngVel = w;
  }

  void initializePythonSession() {
    headless = true;
    resetScene();
  }

  void stepPythonSession(int steps = 1) {
    if (steps < 0) {
      throw std::invalid_argument("steps must be non-negative");
    }
    for (int i = 0; i < steps; ++i) {
      stepWithSubsteps();
      ++frameIndex;
    }
  }

  const std::vector<RigidBody> &pythonRods() const { return rods; }
  uint64_t pythonFrameIndex() const { return frameIndex; }
  double pythonLastKE() const { return lastKE; }
  size_t pythonLastHitCount() const { return lastHitCount; }
  size_t pythonLastIslandCount() const { return lastIslandCount; }
  float pythonDt() const { return dt; }

private:
  // ---- Window and OpenGL ----
#ifndef HEADLESS_BUILD
  GLFWwindow *window = nullptr;
  bool vsync = true;
#endif
  bool headless = false;
  int headlessSteps = 1000;
  double stopKEThreshold = -1.0;
  int stopKEMinSteps = 0;
  int stopKEAvgWindow = 1;  // 1 = instantaneous (legacy), N = rolling avg
  std::vector<double> stopKEBuffer;
  int stopKEBufIdx = 0;
  double stopSlideVelThreshold = -1.0;
  int stopSlideVelMinSteps = 0;

  int renderStride = 1;
  int csvStride = 1;
  int cliStatusStride = 1024;

  // New strides and limits
  int outputStride = 1;
  int outputMaxFrames = -1;
  int outputWrittenFrames = 0;

  int networkMaxFrames = -1;
  int networkWrittenFrames = 0;

  // ---- Renderer and meshes ----
#ifndef HEADLESS_BUILD
  Renderer rnd;
  Mesh cube, cyl, sphere;

  // ---- Camera ----
  OrbitCamera cam;
  glm::vec3 camTarget{0.0f}; // target/orbit center (playback configurable)
  bool dragging = false;
  double lastX = 0.0, lastY = 0.0;
#endif

  // ---- Simulation ----
  bool paused = false;
  bool stepSingle = false;
  glm::vec3 gravity{0.0f, -10.0f, 0.0f};
  float dt = 1.0f / 600.0f;
  AppCfg settings{};

  // Visualization
  bool showContactForces = true;   // Default ON?
  bool forceFadingEnabled = false; // Toggle for temporal fading
  float contactForceScale = 0.5f;
  int viewRodIndex = -1;         // -1 = view all, >= 0 view specific rod
  int perturbationRodIndex = -1; // -1 = all, >=0 = specific rod

  struct VisualContact {
    glm::vec3 p0, p1;
    float timeLeft; // Seconds
    int idxA = -1;
    int idxB = -1;
  };
  std::vector<VisualContact> fadingContacts;
  float forceFadeDuration = 1.0f;

  SoftContactSolver softContactSolver{};
  MujocoContactSolver mjContactSolver{};
  HertzMindlinSolver hertzMindlinSolver{};
  NscContactSolver nscSolver{};

  // Periodic box
  bool usePBC = false;
  glm::vec3 pbcMin{-3, -1, -3}, pbcMax{3, 3, 3};
  float cellSize = 0.6f; // broadphase grid cell size
  // Rendering overrides (playback)
  bool disableFloorRender = false; // when true, suppress floor even if !usePBC

  // ---- Physics objects ----
  std::vector<RigidBody> rods;
  RigidBody floorRB;

  // State for relative displacement calculation
  std::vector<glm::vec3> lastRodPos;
  glm::vec3 lastCOM{0.0f};
  bool hasLastPos = false;
  double lastRelDispSq = 0.0;
  uint64_t lastRelDispFrame = (uint64_t)-1;

  // Initial CSV path (optional)
  std::string initCsvPath;
  std::string saveInitPath;
  std::string initStateCsvPath;

  // Velocity override config
  bool overrideVelEnabled = false;
  int overrideVelId = -1;
  glm::vec3 overrideVel{0.0f};

  // Angular velocity override config
  bool overrideAngVelEnabled = false;
  int overrideAngVelId = -1;
  glm::vec3 overrideAngVel{0.0f};

  // ---- Initialization ----
  bool initWindow(int width = 1200, int height = 800,
                  const char *title = "Rigid Bodies - Rods (Capsules)");
  bool initGraphics();

  // ---- Scene management ----
  static RigidBody createRod(const BodyCfg &config);
  void resetScene();

  // ---- Event callbacks ----
#ifndef HEADLESS_BUILD
  static void keyCB(GLFWwindow *window, int key, int scancode, int action,
                    int mods);
  static void cursorCB(GLFWwindow *window, double x, double y);
  static void mouseCB(GLFWwindow *window, int button, int action, int mods);
  static void scrollCB(GLFWwindow *window, double xoffset, double yoffset);
#endif

  // ---- Profiling (built-in lightweight) ----
  struct Times {
    double integrate = 0, sleepUpdate = 0, broadphase = 0, warmstart = 0,
           buildIslands = 0, solve = 0, floorSolve = 0, posCorrect = 0,
           pbcWrap = 0, render = 0;
    // New fine-grained broadphase
    double bpCount = 0, bpPrefix = 0, bpFill = 0, bpPairs = 0, bpLongLong = 0;
    void reset() {
      integrate = sleepUpdate = broadphase = warmstart = buildIslands = solve =
          floorSolve = posCorrect = pbcWrap = render = 0;
      bpCount = bpPrefix = bpFill = bpPairs = bpLongLong = 0;
    }
    Times &operator+=(const Times &o) {
      integrate += o.integrate;
      sleepUpdate += o.sleepUpdate;
      broadphase += o.broadphase;
      warmstart += o.warmstart;
      buildIslands += o.buildIslands;
      solve += o.solve;
      floorSolve += o.floorSolve;
      posCorrect += o.posCorrect;
      pbcWrap += o.pbcWrap;
      render += o.render;
      bpCount += o.bpCount;
      bpPrefix += o.bpPrefix;
      bpFill += o.bpFill;
      bpPairs += o.bpPairs;
      bpLongLong += o.bpLongLong;
      return *this;
    }
  };
  struct ScopedAccum {
    using clock = std::chrono::high_resolution_clock;
    double *acc = nullptr;
    clock::time_point t0;
    explicit ScopedAccum(double *dst) : acc(dst), t0(clock::now()) {}
    ~ScopedAccum() {
      if (acc) {
        auto t1 = clock::now();
        *acc += std::chrono::duration<double, std::milli>(t1 - t0).count();
      }
    }
  };
  bool profilingEnabled = false;
  // Wall-clock time of the last stepWithSubsteps() call (all substeps),
  // in milliseconds. Always measured; the common axis for comparing
  // contact models by compute cost.
  double lastStepWallMs = 0.0;
  Times curTimes{}, sumTimes{};
  int sumFrames = 0;
  std::chrono::high_resolution_clock::time_point lastTitleUpdate{};
#ifndef HEADLESS_BUILD
  void maybeUpdateWindowTitle();
#endif

  // ---- Snapshot capture ----
  bool snapshotEnabled = false;
  int snapStride = 0;       // capture every snapStride frames (0 => disabled)
  int snapFrames = 0;       // total snapshots to capture
  int snapStartFrame = 0;   // frame to start capturing
  int snapshotCount = 0;    // how many captured so far
  std::string snapshotPath; // NDJSON output path
  std::ofstream snapshotStream; // output stream

  // Stride for network logging
  int networkStride = 1;

  // When true, selected logs (CSV, reldisp) are emitted only on snapshot frames
  bool logOnSnapshotOnly = false;

  // Square wave logging: output only when (frame % period) < width
  bool logWaveEnabled = false;
  uint64_t logWavePeriod = 1000;
  uint64_t logWaveWidth = 100;

  inline bool shouldLogThisFrame() const {
    // 1. Snapshot-only mode takes precedence if enabled
    if (logOnSnapshotOnly) {
      if (!snapshotEnabled || snapStride <= 0)
        return true; // fallback
      if (frameIndex < (uint64_t)snapStartFrame)
        return false;
      return ((frameIndex - snapStartFrame) % snapStride) == 0 &&
             snapshotCount < snapFrames;
    }

    // 2. Square wave logging
    if (logWaveEnabled) {
      uint64_t phase = frameIndex % logWavePeriod;
      if (phase >= logWaveWidth)
        return false;
    }

    return true;
  }

public:
  void enableSnapshots(int stride, int frames, const std::string &path,
                       int startFrame = 0) {
    if (stride <= 0 || frames <= 0) {
      std::cerr << "[snap] Invalid stride or frames; disabling snapshots\n";
      return;
    }
    snapStride = stride;
    snapFrames = frames;
    snapStartFrame = startFrame;
    snapshotPath = path.empty() ? std::string("snapshots.ndjson") : path;
    snapshotStream.open(snapshotPath, std::ios::out | std::ios::trunc);
    if (!snapshotStream) {
      std::cerr << "[snap] Failed to open " << snapshotPath << "\n";
      snapStride = snapFrames = 0;
      return;
    }
    snapshotEnabled = true;
    snapshotCount = 0;
    std::cerr << "[snap] Enabled. stride=" << snapStride
              << " frames=" << snapFrames << " start=" << snapStartFrame
              << " path=" << snapshotPath << "\n";
  }

  void setLogWave(uint64_t period, uint64_t width) {
    if (period == 0)
      return;
    logWaveEnabled = true;
    logWavePeriod = period;
    logWaveWidth = width;
    if (!gQuiet) std::cerr << "[app] Square wave logging enabled. Period=" << period
              << " Width=" << width << "\n";
  }

  void setConstantRandomAccel(bool enable, float sigma) {
    useConstantRandomAccel = enable;
    constAccelSigma = sigma;
  }

private:
  // Constant Random Acceleration
  bool useConstantRandomAccel = false;
  float constAccelSigma = 0.0f;
  std::vector<glm::vec3> constantForces;

  void writeSnapshotLine() {
    if (!snapshotEnabled || !snapshotStream)
      return;
    // Frame/time header
    double simTime = double(frameIndex) * double(settings.physics.dt);
    snapshotStream << '{' << "\"frame\":" << frameIndex
                   << ",\"time\":" << simTime << ",\"bodies\":[";
    for (size_t i = 0; i < rods.size(); ++i) {
      const auto &rb = rods[i];
      if (i)
        snapshotStream << ',';
      snapshotStream << '{' << "\"id\":" << i << ",\"shape\":\"";
      if (rb.type == ShapeType::Sphere) {
        snapshotStream << "sphere\",";
        snapshotStream << "\"pos\":[" << rb.x.x << ',' << rb.x.y << ','
                       << rb.x.z << "],";
        snapshotStream << "\"vel\":[" << rb.v.x << ',' << rb.v.y << ','
                       << rb.v.z << "],";
        snapshotStream << "\"omega\":[" << rb.w.x << ',' << rb.w.y << ','
                       << rb.w.z << "],";
        snapshotStream << "\"radius\":" << rb.sphere.r;
      } else if (rb.type == ShapeType::Capsule) {
        snapshotStream << "capsule\",";
        snapshotStream << "\"pos\":[" << rb.x.x << ',' << rb.x.y << ','
                       << rb.x.z << "],";
        snapshotStream << "\"quat\":[" << rb.q.w << ',' << rb.q.x << ','
                       << rb.q.y << ',' << rb.q.z << "],";
        snapshotStream << "\"vel\":[" << rb.v.x << ',' << rb.v.y << ','
                       << rb.v.z << "],";
        snapshotStream << "\"omega\":[" << rb.w.x << ',' << rb.w.y << ','
                       << rb.w.z << "],";
        snapshotStream << "\"radius\":" << rb.cap.r
                       << ",\"halfHeight\":" << rb.cap.h;
      } else {
        snapshotStream << "box\",";
        snapshotStream << "\"pos\":[" << rb.x.x << ',' << rb.x.y << ','
                       << rb.x.z << "],";
        snapshotStream << "\"quat\":[" << rb.q.w << ',' << rb.q.x << ','
                       << rb.q.y << ',' << rb.q.z << "],";
        snapshotStream << "\"vel\":[" << rb.v.x << ',' << rb.v.y << ','
                       << rb.v.z << "],";
        snapshotStream << "\"omega\":[" << rb.w.x << ',' << rb.w.y << ','
                       << rb.w.z << "],";
        snapshotStream << "\"hx\":" << rb.box.hx << ",\"hy\":" << rb.box.hy
                       << ",\"hz\":" << rb.box.hz;
      }
      snapshotStream << '}';
    }
    snapshotStream << "]}" << '\n';
    if ((frameIndex & 0x1F) == 0)
      snapshotStream.flush();
    ++snapshotCount;
    if (snapshotCount >= snapFrames) {
      snapshotEnabled = false;
      snapshotStream.flush();
      snapshotStream.close();
      std::cerr << "[snap] Reached target snapshots (" << snapFrames
                << "). Stopping capture.\n";
    }
  }

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
  // Soft-contact potential energy (total over contacts) captured after latest
  // force computation
  double lastSoftPotentialEnergy = 0.0;
  // Optional separate CSV logging for soft contact potential energy
  bool softPEEnabled = false;
  std::ofstream softPEStream;
  bool softPEHeaderWritten = false;
  std::string softPEPath;

public: // expose soft PE enabling API
  void enableSoftPE(const std::string &path) {
    softPEPath = path.empty() ? std::string("soft_pe.csv") : path;
    softPEStream.open(softPEPath, std::ios::out | std::ios::trunc);
    if (!softPEStream) {
      std::cerr << "Failed to open soft PE CSV file: " << softPEPath << "\n";
      softPEEnabled = false;
      return;
    }
    softPEEnabled = true;
    softPEHeaderWritten = false;
  }

  // Enable center-of-mass tracking
  void enableCOM(const std::string &path) {
    comPath = path.empty() ? std::string("com.csv") : path;
    comStream.open(comPath, std::ios::out | std::ios::trunc);
    if (!comStream) {
      std::cerr << "Failed to open COM CSV file: " << comPath << "\n";
      comEnabled = false;
      return;
    }
    comEnabled = true;
    comHeaderWritten = false;
  }

  // Enable contact network tracking
  void enableNetwork(const std::string &path) {
    networkPath = path.empty() ? std::string("network.csv") : path;
    networkStream.open(networkPath, std::ios::out | std::ios::trunc);
    if (!networkStream) {
      std::cerr << "Failed to open network CSV file: " << networkPath << "\n";
      networkEnabled = false;
      return;
    }
    networkEnabled = true;
    networkHeaderWritten = false;
  }

  // If enabled, write a sentinel row for frames with zero contacts.
  // Sentinel schema: rod_i=-1, rod_j=-1 and all other numeric fields 0.
  void setNetworkEmitEmptyFrames(bool v) { networkEmitEmptyFrames = v; }

private:
  void logSoftPEFrame() {
    if (!softPEEnabled || !softPEStream)
      return;
    if (!softPEHeaderWritten) {
      softPEStream << "frame,soft_PE\n";
      softPEHeaderWritten = true;
    }
    softPEStream << frameIndex << ',' << lastSoftPotentialEnergy << '\n';
    if ((frameIndex & 0x3F) == 0)
      softPEStream.flush();
  }

  // Center-of-mass CSV logging
  bool comEnabled = false;
  std::ofstream comStream;
  std::string comPath;
  bool comHeaderWritten = false;
  void logCOMFrame();
  glm::vec3 computeCOM() const;

  // Relative displacement (ri - rc) tracking
public:
  void enableRelDisp(const std::string &path) {
    relDispPath = path.empty() ? std::string("reldisp.csv") : path;
    relDispStream.open(relDispPath, std::ios::out | std::ios::trunc);
    if (!relDispStream) {
      std::cerr << "Failed to open relative displacement CSV file: "
                << relDispPath << "\n";
      relDispEnabled = false;
      return;
    }
    relDispEnabled = true;
    relDispHeaderWritten = false;
  }

private:
  bool relDispEnabled = false;
  std::ofstream relDispStream;
  std::string relDispPath;
  bool relDispHeaderWritten = false;
  void logRelDispFrame();

  // Contact network CSV logging
  bool networkEnabled = false;
  std::ofstream networkStream;
  std::string networkPath;
  bool networkHeaderWritten = false;
  bool networkEmitEmptyFrames = false;
  void logNetworkFrame(); // Will detect contact mode automatically

  // Playback state for interactive navigation
  bool inPlaybackMode = false;
  int currentPlaybackFrame = 0;
  int totalPlaybackFrames = 0;
  std::vector<std::string> playbackFrameData;
  std::unordered_map<int, std::vector<VisualContact>> playbackContacts;
  float playbackSpeedMultiplier = 1.0f; // Speed control (0.5x, 1x, 2x, etc.)
  void loadPlaybackFrame(int frameIndex);

  // Per-rod CSV logging
  bool perRodEnabled = false;
  int perRodMaxFrames = 1000;
  std::string perRodPath;
  std::ofstream perRodStream;
  bool perRodHeaderWritten = false;
  int perRodSkip = 1; // sample every N frames
  int perRodWrittenFrames = 0;
  bool explicitPerRodStride = false;
  void logPerRodFrame();

  // Lightweight trajectory for a single rod: only axis endpoints over time.
  bool testRodEndpointsEnabled = false;
  std::string testRodEndpointsPath;
  std::ofstream testRodEndpointsStream;
  bool testRodEndpointsHeaderWritten = false;
  int testRodIndex = -1; // -1 => derive from settings.scene.fixEveryExcept
  int testRodEndpointsStride = 1;
  int testRodEndpointsMaxFrames = INT32_MAX;
  int testRodEndpointsWrittenFrames = 0;
  bool explicitTestRodEndpointsStride = false;
  void logTestRodEndpointsFrame();
  // Compute total kinetic energy for current rods
  double totalKE() const;

  // --- Reptation summary tracking ---
  bool reptSummaryEnabled = false;
  std::string reptSummaryPath;
  int reptRodIdx = 0;       // tracked rod (default: rod 0)
  double reptTotalPath = 0.0; // sum |dx·a| per frame
  int reptWallHits = 0;     // cumulative wall contact events
  bool reptInitialized = false;
  glm::vec3 reptPrevPos{0.0f};
  glm::vec3 reptStartPos{0.0f};

  void reptAccumulate() {
    if (!reptSummaryEnabled || reptRodIdx < 0 ||
        reptRodIdx >= (int)rods.size()) return;
    const auto& rb = rods[reptRodIdx];
    const glm::vec3 cylAxis = glm::normalize(settings.scene.cylinder.axis);
    if (!reptInitialized) {
      reptStartPos = rb.x;
      reptPrevPos = rb.x;
      reptInitialized = true;
      return;
    }
    glm::vec3 dx = rb.x - reptPrevPos;
    reptTotalPath += std::abs((double)glm::dot(dx, cylAxis));
    reptPrevPos = rb.x;
  }

  void writeReptSummary() {
    if (!reptSummaryEnabled) return;
    const glm::vec3 cylAxis = glm::normalize(settings.scene.cylinder.axis);
    // Net displacement along axis
    double netDisp = 0.0;
    if (reptRodIdx >= 0 && reptRodIdx < (int)rods.size()) {
      glm::vec3 dx = rods[reptRodIdx].x - reptStartPos;
      netDisp = std::abs((double)glm::dot(dx, cylAxis));
    }
    double finalKE = totalKE();
    double simTime = frameIndex * settings.physics.dt;

    // Write header if file doesn't exist or is empty
    bool writeHeader = false;
    {
      std::ifstream chk(reptSummaryPath);
      writeHeader = !chk.good() || chk.peek() == std::ifstream::traits_type::eof();
    }
    std::ofstream ofs(reptSummaryPath, std::ios::app);
    if (writeHeader)
      ofs << "mu,R_cyl,L_rod,d_rod,v0_lin,v0_ang,net_displacement,total_path_length,wall_hits,sim_time,final_KE\n";

    const auto& rb = (reptRodIdx >= 0 && reptRodIdx < (int)rods.size())
                       ? rods[reptRodIdx] : rods[0];
    float rodLen = (rb.cap.h + rb.cap.r) * 2.0f;
    float rodDiam = rb.cap.r * 2.0f;
    float v0_lin = overrideVelEnabled ? glm::length(overrideVel) : 0.0f;
    float v0_ang = overrideAngVelEnabled ? glm::length(overrideAngVel) : 0.0f;

    float activeMu = settings.physics.soft_contact.enabled
                       ? (float)settings.physics.soft_contact.mu
                       : settings.physics.nsc.mu;
    ofs << std::setprecision(8)
        << activeMu << ","
        << settings.scene.cylinder.radius << ","
        << rodLen << "," << rodDiam << ","
        << v0_lin << "," << v0_ang << ","
        << netDisp << "," << reptTotalPath << ","
        << reptWallHits << ","
        << simTime << "," << finalKE << "\n";
    ofs.close();
    std::cout << "[Reptation] Summary → " << reptSummaryPath
              << "  net=" << netDisp << " path=" << reptTotalPath << "\n";
  }

  // Invoke optional logs after a frame is fully updated
  void logOptionalFrames() {
    logCsvFrame();
    logSoftPEFrame();
    logCOMFrame();
    logNetworkFrame();
    logPerRodFrame();
    logTestRodEndpointsFrame();
    logRelDispFrame();
    logOutputFrame();
  }
  // Aggregate squared radius of gyration (average L2 distance from COM)
  double computeGyrationSq() const {
    if (rods.empty())
      return 0.0;
    glm::vec3 rc =
        const_cast<App *>(this)
            ->computeCOM(); // computeCOM not const; cast is safe for read
    glm::vec3 boxSize = pbcMax - pbcMin;
    double sum = 0.0;
    for (const auto &rb : rods) {
      glm::vec3 d = rb.x - rc;
      if (usePBC) {
        for (int k = 0; k < 3; ++k) {
          if (boxSize[k] > 0.0f) {
            d[k] -= boxSize[k] * std::floor(d[k] / boxSize[k] + 0.5f);
          }
        }
      }
      sum += double(glm::dot(d, d));
    }
    return sum / double(rods.size());
  }

  // Compute Mean Squared Relative Displacement (internal deformation)
  // u_i = (p_i(t) - p_i(t-1)) - (C(t) - C(t-1))
  // Metric = (1/N) * sum( ||u_i||^2 )
  double computeRelativeMotionSq() {
    // Return cached value if already computed for this frame
    if (lastRelDispFrame == frameIndex) {
      return lastRelDispSq;
    }

    if (rods.empty())
      return 0.0;

    glm::vec3 currentCOM = computeCOM();

    if (!hasLastPos || lastRodPos.size() != rods.size()) {
      // First frame or reset: initialize history and return 0
      lastRodPos.resize(rods.size());
      for (size_t i = 0; i < rods.size(); ++i) {
        lastRodPos[i] = rods[i].x;
      }
      lastCOM = currentCOM;
      hasLastPos = true;
      lastRelDispSq = 0.0;
      lastRelDispFrame = frameIndex;
      return 0.0;
    }

    glm::vec3 boxSize = pbcMax - pbcMin;
    double sumSq = 0.0;

    // Calculate COM displacement with PBC
    glm::vec3 dCOM = currentCOM - lastCOM;
    if (usePBC) {
      for (int k = 0; k < 3; ++k) {
        if (boxSize[k] > 0.0f) {
          dCOM[k] -= boxSize[k] * std::floor(dCOM[k] / boxSize[k] + 0.5f);
        }
      }
    }

    for (size_t i = 0; i < rods.size(); ++i) {
      // Calculate rod displacement with PBC
      glm::vec3 dPos = rods[i].x - lastRodPos[i];
      if (usePBC) {
        for (int k = 0; k < 3; ++k) {
          if (boxSize[k] > 0.0f) {
            dPos[k] -= boxSize[k] * std::floor(dPos[k] / boxSize[k] + 0.5f);
          }
        }
      }

      // Relative displacement: rod motion minus global COM motion
      glm::vec3 u = dPos - dCOM;
      sumSq += double(glm::dot(u, u));

      // Update history
      lastRodPos[i] = rods[i].x;
    }

    lastCOM = currentCOM;
    lastRelDispSq = sumSq / double(rods.size());
    lastRelDispFrame = frameIndex;
    return lastRelDispSq;
  }

  // --- Compact output CSV (subset of metrics) ---
public:
  void enableOutput(const std::string &path) {
    outputPath = path.empty() ? std::string("output.csv") : path;
    outputStream.open(outputPath, std::ios::out | std::ios::trunc);
    if (!outputStream) {
      std::cerr << "Failed to open output CSV file: " << outputPath << "\n";
      outputEnabled = false;
      return;
    }
    outputEnabled = true;
    outputHeaderWritten = false;
  }

private:
  bool outputEnabled = false;
  std::ofstream outputStream;
  std::string outputPath;
  bool outputHeaderWritten = false;
  bool earlyPairContactHeaderWritten = false;
  bool earlyPairDistanceHeaderWritten = false;
  bool earlyPairVelocitySummaryHeaderWritten = false;
  std::ofstream earlyPairContactStream;
  std::ofstream earlyPairDistanceStream;
  std::ofstream earlyPairVelocitySummaryStream;
  std::unordered_set<int> earlyPairSampleFrames;
  // Debug flag: when true, minPairGap will print info about the worst (most
  // overlapping) pair
  bool debugMinGap = false;
  // When true, enforce a nonpenetration check right after initialization
  // (before first step)
  bool checkInitNonpenetration = false;

public:
  void setDebugMinGap(bool on) { debugMinGap = on; }
  void setCheckInitNonpenetration(bool on) { checkInitNonpenetration = on; }

private:
  void logOutputFrame();
  struct PairGapInfo {
    double signedGap = 0.0;
    double distanceMetric = 0.0;
    double surfaceLimit = 0.0;
    const char *pairType = "other";
  };
  struct PairKinematicsInfo {
    PairGapInfo gap{};
    glm::vec3 pointA{0.0f};
    glm::vec3 pointB{0.0f};
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
    glm::vec3 vRel{0.0f};
    double vNormal = 0.0;
    double vTangentialMagnitude = 0.0;
  };
  bool ensureEarlyPairContactStream();
  bool ensureEarlyPairDistanceStream();
  bool ensureEarlyPairVelocitySummaryStream();
  void logDetectedContactsFrame(const std::vector<CommonContactGeometry> &contacts,
                                const char *solver, int sampleFrame);
  void logAllPairDistancesFrame(int sampleFrame);
  PairGapInfo computePairGapInfo(const RigidBody &A, const RigidBody &B) const;
  PairKinematicsInfo computePairKinematicsInfo(const RigidBody &A,
                                               const RigidBody &B) const;
  // Compute minimum signed surface-to-surface gap between all pairs:
  // positive = separation, negative = overlap. Definitions:
  //   - capsule–capsule: axis distance minus (rA + rB)
  //   - sphere–sphere: center distance minus (r1 + r2)
  //   - sphere–capsule: axis distance minus (r_sphere + r_capsule)
  double minPairGap() const {
    const size_t N = rods.size();
    if (N < 2)
      return 0.0;
    auto segseg_dist = [](const glm::vec3 &p0, const glm::vec3 &p1,
                          const glm::vec3 &q0, const glm::vec3 &q1) {
      // robust segment-segment distance (non-PBC; caller applies any PBC
      // wrapping)
      glm::vec3 u = p1 - p0;
      glm::vec3 v = q1 - q0;
      glm::vec3 w0 = p0 - q0;
      float uu = glm::dot(u, u), vv = glm::dot(v, v), uv = glm::dot(u, v);
      float wu = glm::dot(w0, u), wv = glm::dot(w0, v);
      float D = uu * vv - uv * uv;
      float s, t;
      const float eps = 1e-12f;

      auto fixBound = [](float &x) {
        if (x < 0.0f)
          x = 0.0f;
        else if (x > 1.0f)
          x = 1.0f;
      };

      if (std::abs(D) < eps) {
        // Parallel
        s = 0.0f;
        t = (vv > eps) ? (-wv / vv) : 0.0f;
        fixBound(t);
      } else {
        s = (uv * wv - vv * wu) / D;
        fixBound(s);
        t = (s * uv + wv) / (vv > eps ? vv : 1.0f);
        float t_unclamped = t;
        fixBound(t);

        if (std::abs(t - t_unclamped) > 1e-6f) {
          s = (t * uv - wu) / (uu > eps ? uu : 1.0f);
          fixBound(s);
        }
      }
      glm::vec3 d = (w0 + s * u) - t * v;
      return std::sqrt(std::max(0.0f, glm::dot(d, d)));
    };
    auto sphere_sphere_gap = [](const RigidBody &A, const RigidBody &B) {
      float center = glm::length(B.x - A.x);
      float sumR = A.sphere.r + B.sphere.r;
      return double(center) - double(sumR);
    };
    auto capsule_capsule_gap = [&](const RigidBody &A, const RigidBody &B) {
      glm::vec3 a0, a1, b0, b1;
      A.capsuleEndpoints(a0, a1);
      B.capsuleEndpoints(b0, b1);
      // Apply PBC by shifting B's segment into minimum image w.r.t A's center
      glm::vec3 centerDelta = B.x - A.x;
      if (usePBC) {
        glm::vec3 boxSize = pbcMax - pbcMin;
        for (int k = 0; k < 3; ++k) {
          float L = boxSize[k];
          if (L > 0.0f)
            centerDelta[k] -= L * std::floor(centerDelta[k] / L + 0.5f);
        }
      }
      glm::vec3 shift = centerDelta - (B.x - A.x);
      glm::vec3 b0Wrapped = b0 + shift;
      glm::vec3 b1Wrapped = b1 + shift;
      float axisDist = segseg_dist(a0, a1, b0Wrapped, b1Wrapped);
      float sumR = A.cap.r + B.cap.r;
      return double(axisDist) - double(sumR);
    };
    auto sphere_capsule_gap = [&](const RigidBody &S, const RigidBody &C) {
      glm::vec3 a0, a1;
      C.capsuleEndpoints(a0, a1);
      // project sphere center onto segment
      glm::vec3 u = a1 - a0;
      float L2 = glm::dot(u, u);
      float t = L2 > 0 ? glm::dot(S.x - a0, u) / L2 : 0.0f;
      t = glm::clamp(t, 0.0f, 1.0f);
      glm::vec3 closest = a0 + t * u;
      float axisDist = glm::length(S.x - closest);
      float sumR = S.sphere.r + C.cap.r;
      return double(axisDist) - double(sumR);
    };
    double minGap = 1e300;
    int minI = -1, minJ = -1;
    float dbgCenter = 0.0f, dbgAxis = 0.0f, dbgRA = 0.0f, dbgRB = 0.0f;
    for (size_t i = 0; i < N; i++) {
      for (size_t j = i + 1; j < N; j++) {
        const RigidBody &A = rods[i];
        const RigidBody &B = rods[j];
        double g = 0.0;
        if (A.type == ShapeType::Sphere && B.type == ShapeType::Sphere) {
          g = sphere_sphere_gap(A, B);
        } else if (A.type == ShapeType::Capsule &&
                   B.type == ShapeType::Capsule) {
          g = capsule_capsule_gap(A, B);
        } else if (A.type == ShapeType::Sphere &&
                   B.type == ShapeType::Capsule) {
          g = sphere_capsule_gap(A, B);
        } else if (A.type == ShapeType::Capsule &&
                   B.type == ShapeType::Sphere) {
          g = sphere_capsule_gap(B, A);
        } else {
          // Fallback for unsupported shapes (e.g., boxes): use center distance
          float center = glm::length(B.x - A.x);
          g = double(center);
        }
        if (g < minGap) {
          minGap = g;
          minI = int(i);
          minJ = int(j);
          if (A.type == ShapeType::Sphere && B.type == ShapeType::Sphere) {
            dbgCenter = glm::length(B.x - A.x);
            dbgAxis = dbgCenter;
            dbgRA = A.sphere.r;
            dbgRB = B.sphere.r;
          } else if (A.type == ShapeType::Capsule &&
                     B.type == ShapeType::Capsule) {
            glm::vec3 a0, a1, b0, b1;
            A.capsuleEndpoints(a0, a1);
            B.capsuleEndpoints(b0, b1);
            dbgAxis = segseg_dist(a0, a1, b0, b1);
            dbgCenter = glm::length(B.x - A.x);
            dbgRA = A.cap.r;
            dbgRB = B.cap.r;
          } else if (A.type == ShapeType::Sphere &&
                     B.type == ShapeType::Capsule) {
            dbgCenter = glm::length(B.x - A.x);
            dbgRA = A.sphere.r;
            dbgRB = B.cap.r;
            glm::vec3 a0, a1;
            B.capsuleEndpoints(a0, a1);
            glm::vec3 u = a1 - a0;
            float L2 = glm::dot(u, u);
            float t = L2 > 0 ? glm::dot(A.x - a0, u) / L2 : 0.0f;
            t = glm::clamp(t, 0.0f, 1.0f);
            glm::vec3 closest = a0 + t * u;
            dbgAxis = glm::length(A.x - closest);
          } else if (A.type == ShapeType::Capsule &&
                     B.type == ShapeType::Sphere) {
            dbgCenter = glm::length(B.x - A.x);
            dbgRA = A.cap.r;
            dbgRB = B.sphere.r;
            glm::vec3 a0, a1;
            A.capsuleEndpoints(a0, a1);
            glm::vec3 u = a1 - a0;
            float L2 = glm::dot(u, u);
            float t = L2 > 0 ? glm::dot(B.x - a0, u) / L2 : 0.0f;
            t = glm::clamp(t, 0.0f, 1.0f);
            glm::vec3 closest = a0 + t * u;
            dbgAxis = glm::length(B.x - closest);
          } else {
            dbgCenter = glm::length(B.x - A.x);
            dbgAxis = dbgCenter;
            dbgRA = dbgRB = 0.0f;
          }
        }
      }
    }
    if (debugMinGap && minI >= 0 && minJ >= 0) {
      std::cerr << "[minPairGap-debug] frame=" << frameIndex << " i=" << minI
                << " j=" << minJ << " type_i=" << int(rods[minI].type)
                << " type_j=" << int(rods[minJ].type)
                << " center_dist=" << dbgCenter << " axis_dist=" << dbgAxis
                << " rA=" << dbgRA << " rB=" << dbgRB << " gap=" << minGap
                << " max_overlap=" << (minGap < 0.0 ? -minGap : 0.0) << "\n";
    }
    return minGap;
  }

  // Adaptive substeps (runtime-tunable)
  bool adaptiveSubsteps = false;
  int asMin = 1;
  int asMax = 1;
  int asHitThresh = INT32_MAX;
  double asKEUpThresh = 1e300;    // huge => disabled by default
  double asKEDownThresh = -1e300; // very negative => disabled
  double lastFrameKEDelta = 0.0;  // KE_n - KE_{n-1}
  double prevFrameKE = 0.0;       // KE of previous frame

  // Positional stabilization tuning (Baumgarte scaling)
  float betaMin = 0.0f; // clamp lower bound on beta during dyn scaling
  int betaHighContactThresh = INT32_MAX;
  float betaHighContactScale =
      1.0f; // multiply solver.baumgarte by this when many contacts
  bool debugNormalVelocity = false;
  std::string debugNormalVelocityCsvPath;
  std::string energyBalanceCsvPath;

  // Declare Hit before using in dumpContactsCSV
  struct Hit; // forward declaration

  // Contact dump diagnostics
  bool contactDumpEnabled = false;
  std::string contactDumpPath;
  std::ofstream contactDumpStream;
  bool contactDumpHeaderWritten = false;
  double contactDumpThresh =
      0.0; // absolute KE increase/decrease threshold to trigger (J)
  int contactDumpTrigger = 0; // 0:any, +1:up, -1:down
  EarlyPairDiagnosticsCfg earlyPairDiagnostics{};

  void dumpContactsCSV(const std::vector<Hit> &hits, const char *stageLabel);

  // ---- Simulation ----
  struct Hit {
    int a = -1, b = -1;
    Contact c{};
  };

  void physicsStep();
  void renderFrame();
  void stepWithSubsteps();

  // ---- Initial configuration loader (CSV with segment endpoints) ----
  // CSV schema: optional header lines starting with '#' providing metadata,
  // then a header line: x0,y0,z0,x1,y1,z1 followed by rows of endpoints.
  // Populates 'rods' with capsules aligned to each segment; diameter taken from
  // settings.scene.bodies[0].diameter when available, else uses the
  // distance-derived length with a default diameter.
  bool loadInitialConfigCSV(const std::string &path);
  bool saveInitialConfigCSV(const std::string &path);
  bool loadInitialStateCSV(const std::string &path);

  // ---- Helpers ----
  static inline glm::ivec3 gridDims(const glm::vec3 &bmin,
                                    const glm::vec3 &bmax, float cs) {
    glm::vec3 size = bmax - bmin;
    glm::ivec3 n = glm::max(glm::ivec3(1), glm::ivec3(glm::floor(size / cs)));
    return n;
  }
  static inline glm::ivec3 cellIndex(const glm::vec3 &p, const glm::vec3 &bmin,
                                     const glm::vec3 &bmax,
                                     const glm::ivec3 &n) {
    glm::vec3 size = bmax - bmin;
    glm::vec3 rel = (p - bmin) / size; // in [0,1)
    glm::ivec3 idx =
        glm::clamp(glm::ivec3(rel * glm::vec3(n)), glm::ivec3(0), n - 1);
    return idx;
  }
  static inline int64_t packKey(const glm::ivec3 &i, const glm::ivec3 &n) {
    // pack 3 indices into 64-bit (assumes n components < 2^21)
    return (int64_t(i.x) << 42) ^ (int64_t(i.y) << 21) ^ int64_t(i.z);
  }
  static inline size_t linearIndex(const glm::ivec3 &i, const glm::ivec3 &n) {
    return size_t(i.x) +
           size_t(n.x) * (size_t(i.y) + size_t(n.y) * size_t(i.z));
  }
  static inline uint64_t pairKey(int a, int b) {
    if (b < a)
      std::swap(a, b);
    return (uint64_t(uint32_t(a)) << 32) | uint64_t(uint32_t(b));
  }
  static inline void wrapPos(glm::vec3 &p, const glm::vec3 &bmin,
                             const glm::vec3 &bmax) {
    const glm::vec3 size = bmax - bmin;
    for (int k = 0; k < 3; ++k) {
      if (size[k] <= 0.0f)
        continue;
      while (p[k] < bmin[k])
        p[k] += size[k];
      while (p[k] >= bmax[k])
        p[k] -= size[k];
    }
  }

  // ---- Sleeping (simple) ----
  float sleepLinThresh = -1.0f;  // m/s (Disabled)
  float sleepAngThresh = -1.0f;  // rad/s (Disabled)
  float sleepTimeThresh = 0.6f;  // s
  std::vector<float> sleepTimer; // per-body accumulated below-threshold time
  std::vector<uint8_t> sleeping; // 0/1 flags
  inline void wake(int i) {
    if (i < 0 || i >= (int)rods.size())
      return;
    sleeping[i] = 0;
    sleepTimer[i] = 0.f;
  }

  // ---- Broadphase scratch (reused across frames) ----
  glm::ivec3 gridN{0};
  // Flattened grid: counts -> prefix-sum offsets -> items
  std::vector<uint32_t> gridCounts;  // size = cellCount
  std::vector<uint32_t> gridOffsets; // size = cellCount+1
  std::vector<uint32_t> gridWrite;   // temp write cursors, size = cellCount
  std::vector<int> gridItems;        // flattened body indices
  std::vector<std::vector<Hit>> thHitsScratch; // per-thread hit buffers
  std::vector<std::vector<int>>
      thSeenAt; // per-thread seen stamps sized [numRods]
  std::vector<std::vector<int>>
      thCellSeenAt; // per-thread cell visited stamps sized [cellCount]
  std::vector<Hit> hitsScratch; // merged hits

  // Warm-start cache: previous-frame impulses per pair (a<b)

  // Random force application
  bool useRandomForce = false;
  std::mt19937 genRandomForce{std::random_device{}()};
  std::normal_distribution<float> normal_f{0.0f, 1.0f};
  std::uniform_real_distribution<float> uni_u{-1.0f, 1.0f};
  std::uniform_real_distribution<float> uni_phi{0.0f, 2.0f * float(M_PI)};
  float tauMag = 0.1f;
  float fSigma = 0.0f;

  glm::vec3 uniform_dir_s2(std::mt19937 &gen) {
    float u = uni_u(gen);
    float phi = uni_phi(gen);
    float s = std::sqrt(std::max(0.0f, 1.0f - u * u));
    return glm::vec3(s * std::cos(phi), u, s * std::sin(phi));
  }

  // Entanglement metrics (sum of |linking number| over all rod pairs)
  bool entanglementEnabled = false;
  int entanglementEvery = 60;     // compute every N frames
  double entanglementCutoff = -1; // <=0 disables pruning in library
  int entanglementThreads = 0;    // 0 => auto
  double lastEntanglementSum = 0.0;
  long long lastEntanglementPairs = 0;
  void computeEntanglement();
  // Print a unified CLI status line (frame, rods, KE, entanglement)
  void printCliStatus(const std::string &prefix = "") const;
};

// ---- Implementation ----

#ifndef HEADLESS_BUILD
bool App::initWindow(int width, int height, const char *title) {
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
  cyl = makeCappedCylinderMesh(
      16); // Reduced from 40 to 16 for better performance with many rods
  sphere = makeSphereMesh(16, 12); // 16 slices, 12 stacks for smooth spheres
  return true;
}
#endif

RigidBody App::createRod(const BodyCfg &config) {
  // rot_quat is glm::vec4 {w,x,y,z} (resolved by config)
  glm::quat q(config.rot_quat.x, config.rot_quat.y, config.rot_quat.z,
              config.rot_quat.w);

  RigidBody rb;

  // Create body based on shape type
  if (config.shape == "sphere") {
    rb = RigidBody::makeSphere(config.pos, config.density, config.radius,
                               config.restitution, config.friction);
  } else {
    // Default to capsule/rod
    rb = RigidBody::makeRodLD(config.pos, q, config.density, config.length,
                              config.diameter, config.restitution,
                              config.friction);
  }

  // Advanced friction (optional): default to legacy if not provided
  if (config.friction_s > 0.0f)
    rb.frictionS = config.friction_s;
  else
    rb.frictionS = -1.0f;
  if (config.friction_d > 0.0f)
    rb.frictionD = config.friction_d;
  else
    rb.frictionD = -1.0f;
  rb.rollingFriction = config.rolling_friction;

  if (config.is_static) {
    rb.mass = 0.0f;
    rb.invMass = 0.0f;
    rb.I_body = glm::mat3(0.0f);
    rb.I_body_inv = glm::mat3(0.0f);
    rb.v = glm::vec3(0.0f);
    rb.w = glm::vec3(0.0f);
  } else {
    rb.v = config.v_lin;
    rb.w = config.v_ang;
  }
  return rb;
}

void App::resetScene() {
  dt = settings.physics.dt;
  gravity = settings.physics.gravity;

  // Configure soft contact solvers
  softContactSolver.setConfig(settings.physics.soft_contact);
  MujocoContactCfg mjCfg;
  mjCfg.enabled = settings.physics.soft_contact.enabled &&
                  settings.physics.use_mujoco_contact;
  // Map a few basic parameters; we start conservatively and can tune later.
  mjCfg.normal_k =
      settings.physics.soft_contact.k_scaler; // reuse scaler as stiffness
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
  hmCfg.rolling_friction_coeff =
      settings.physics.hertz_mindlin.rolling_friction_coeff;
  hmCfg.enable_tangential = settings.physics.hertz_mindlin.enable_tangential;
  hmCfg.enable_rolling = settings.physics.hertz_mindlin.enable_rolling;
  hmCfg.verbose = settings.physics.hertz_mindlin.verbose;
  hmCfg.computeDamping(); // Recompute damping from restitution
  hertzMindlinSolver.setConfig(hmCfg);

  // Configure NSC solver
  nscSolver.setConfig(settings.physics.nsc);
  nscSolver.setDebugNormalVelocity(debugNormalVelocity);
  nscSolver.setDebugNormalVelocityCsvPath(debugNormalVelocityCsvPath);
  nscSolver.setEnergyBalanceCsvPath(energyBalanceCsvPath);

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

  if (!gQuiet) std::cout << "[Debug] Calling softContactSolver.setPBC..." << std::endl;
  // Pass PBC settings to soft contact solver
  softContactSolver.setPBC(usePBC, pbcMin, pbcMax);
  nscSolver.setPBC(usePBC, pbcMin, pbcMax);
  if (!gQuiet) std::cout << "[Debug] setPBC done." << std::endl;

  // Random initialization for PBC study
  const bool useRandomInit = settings.scene.randomInit.enabled;
  if (useRandomInit) {
    if (!gQuiet) std::cout << "[Debug] Using random init..." << std::endl;
    gravity = glm::vec3(0.0f);
  } else {
    if (!gQuiet) std::cout << "[Debug] Populating scene..." << std::endl;
  }

  // Random force settings
  useRandomForce = settings.scene.randomForce.enabled;
  if (useRandomForce) {
    fSigma = settings.scene.randomForce.fSigma;
    tauMag = settings.scene.randomForce.tauMag;
    genRandomForce.seed(settings.scene.randomForce.seed
                            ? settings.scene.randomForce.seed
                            : std::random_device{}());
  }

  // Floor (only if not using PBC)
  const auto &floorConfig = settings.scene.floor;
  glm::vec3 fPos = floorConfig.pos;
  if (!floorConfig.enabled) {
    disableFloorRender = true;
    fPos.y = -1e6f; // Move effectively infinite distance away
  }
  glm::quat qF(floorConfig.rot_quat.x, floorConfig.rot_quat.y,
               floorConfig.rot_quat.z, floorConfig.rot_quat.w);
  floorRB = RigidBody::makeStaticFloor(
      fPos, qF, floorConfig.half_extents.x, floorConfig.half_extents.y,
      floorConfig.half_extents.z, floorConfig.restitution,
      floorConfig.friction);

  rods.clear();
  sleeping.clear();
  sleepTimer.clear();

  // If an initial state CSV is specified, load it (highest priority)
  if (!initStateCsvPath.empty()) {
    if (loadInitialStateCSV(initStateCsvPath)) {
      std::cerr << "[init-state] Loaded state from " << initStateCsvPath
                << " (rods=" << rods.size() << ")\n";
    } else {
      std::cerr << "[init-state] Failed to load state from " << initStateCsvPath
                << "\n";
    }
  }
  // If an initial CSV configuration is specified in the scene config, load it
  // first (Populate/randomInit logic below will be skipped if rods are
  // populated here.)
  else if (!initCsvPath.empty()) {
    if (!loadInitialConfigCSV(initCsvPath)) {
      std::cerr << "[init-csv] Failed to load initial CSV: " << initCsvPath
                << "\n";
    } else {
      if (!gQuiet) std::cerr << "[init-csv] Loaded initial configuration from "
                << initCsvPath << " (rods=" << rods.size() << ")\n";
      if (!gQuiet && !rods.empty()) {
        const auto &r = rods[0];
        if (r.type == ShapeType::Capsule) {
          std::cout << "[init-csv] Rod 0: radius=" << r.cap.r
                    << " diameter=" << (2.0f * r.cap.r)
                    << " halfHeight=" << r.cap.h
                    << " length=" << (2.0f * r.cap.h) << "\n";
        }
      }
    }
  }

  // Optional: run a one-shot nonpenetration check immediately after any CSV
  // load
  if (checkInitNonpenetration && !rods.empty()) {
    double g0 = minPairGap();
    std::cerr << "[init-check] minPairGap (pre-step) = " << g0
              << " (max_overlap=" << (g0 < 0.0 ? -g0 : 0.0) << ")\n";
  }

  // Procedural population overrides explicit bodies if requested
  if (rods.empty() && settings.scene.populate.count > 0) {
    const int N = settings.scene.populate.count;
    std::random_device rd;
    std::mt19937 gen(settings.scene.populate.seed ? settings.scene.populate.seed
                                                  : rd());
    std::uniform_real_distribution<float> urand(0.0f, 1.0f);

    // Check if populating spheres or rods
    const bool populatingSpheres = (settings.scene.populate.shape == "sphere");

    // Use first body's dimensions if provided, else defaults/legacy populate
    // settings
    BodyCfg base{};
    if (!settings.scene.bodies.empty()) {
      base = settings.scene.bodies[0];
    } else {
      // Legacy behavior: use populate.radius for diameter
      base.diameter = 2.0f * settings.scene.populate.radius;
      base.radius = settings.scene.populate.radius;
    }

    const float D = populatingSpheres ? (2.0f * base.radius) : base.diameter;
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
        nx = nx2;
        ny = ny2;
        nz = nz2;
      }

      rods.reserve(N);
      int placed = 0;
      for (int ix = 0; ix < nx && placed < N; ++ix) {
        for (int iy = 0; iy < ny && placed < N; ++iy) {
          for (int iz = 0; iz < nz && placed < N; ++iz) {
            BodyCfg cfg = base;
            // Jittered grid center
            glm::vec3 cellMin =
                pbcMin + glm::vec3(ix * spacing, iy * spacing, iz * spacing);
            glm::vec3 p = cellMin + glm::vec3(0.5f * spacing);
            // Small jitter within cell
            glm::vec3 jitter{(urand(gen) - 0.5f) * 0.3f * spacing,
                             (urand(gen) - 0.5f) * 0.3f * spacing,
                             (urand(gen) - 0.5f) * 0.3f * spacing};
            cfg.pos = p + jitter;
            // Random orientation around a random axis
            glm::vec3 axis = glm::normalize(glm::vec3(
                urand(gen) - 0.5f, urand(gen) - 0.5f, urand(gen) - 0.5f));
            float angle = (urand(gen) - 0.5f) * 3.14159f; // [-pi/2, pi/2]
            glm::quat q = glm::angleAxis(angle, axis);
            cfg.rot_quat = glm::vec4(q.w, q.x, q.y, q.z);
            rods.push_back(createRod(cfg));
            ++placed;
          }
        }
      }
    } else if (mode == "nonoverlap" && populatingSpheres) {
      // ===== SPHERE NON-OVERLAPPING PLACEMENT =====
      rods.clear();
      rods.reserve(N);
      const glm::vec3 bmin = pbcMin;
      const glm::vec3 bmax = pbcMax;
      const glm::vec3 boxSize = bmax - bmin;
      const float R = settings.scene.populate.radius;
      const float diameter = 2.0f * R;
      const float minCenterDist = spacing > 0.0f ? spacing : diameter;
      const float minDist2 =
          minCenterDist *
          minCenterDist; // Sphere-sphere minimum center distance squared

      // Spatial grid for fast neighbor lookup
      const float cs =
          (cellSize > 0.0f ? cellSize
                           : std::max(minCenterDist, 1.25f * diameter));
      glm::ivec3 n = gridDims(bmin, bmax, cs);
      const int numCells = std::max(1, n.x * n.y * n.z);
      std::vector<std::vector<int>> cells(numCells);

      auto linIdx = [&](int ix, int iy, int iz) {
        int idx = ix + n.x * (iy + n.y * iz);
        return std::max(0, std::min(idx, numCells - 1));
      };
      auto wrapI = [&](int a, int dim) {
        while (a < 0)
          a += dim;
        while (a >= dim)
          a -= dim;
        return a;
      };
      auto minImage = [&](glm::vec3 d) {
        if (!usePBC)
          return d;
        glm::vec3 r = d;
        for (int k = 0; k < 3; ++k) {
          float L = boxSize[k];
          if (L > 0.0f)
            r[k] -= L * std::floor(r[k] / L + 0.5f);
        }
        return r;
      };

      const int maxAttempts =
          std::max(1000, settings.scene.populate.maxAttempts);
      std::vector<glm::vec3> centers;
      centers.reserve(N);

      for (int i = 0; i < N; ++i) {
        bool placed = false;
        for (int att = 0; att < maxAttempts && !placed; ++att) {
          // Random position in box
          glm::vec3 r{urand(gen), urand(gen), urand(gen)};
          glm::vec3 pos = bmin + r * boxSize;

          // Check cells around this position
          glm::ivec3 cell = glm::floor((pos - bmin) / cs);
          cell = glm::clamp(cell, glm::ivec3(0), n - glm::ivec3(1));
          bool collide = false;

          for (int dz = -1; dz <= 1 && !collide; ++dz)
            for (int dy = -1; dy <= 1 && !collide; ++dy)
              for (int dx = -1; dx <= 1 && !collide; ++dx) {
                int cx = usePBC ? wrapI(cell.x + dx, n.x)
                                : glm::clamp(cell.x + dx, 0, n.x - 1);
                int cy = usePBC ? wrapI(cell.y + dy, n.y)
                                : glm::clamp(cell.y + dy, 0, n.y - 1);
                int cz = usePBC ? wrapI(cell.z + dz, n.z)
                                : glm::clamp(cell.z + dz, 0, n.z - 1);
                const auto &bucket = cells[linIdx(cx, cy, cz)];

                for (int j : bucket) {
                  glm::vec3 other = centers[j];
                  glm::vec3 delta = minImage(pos - other);
                  float dist2 = glm::dot(delta, delta);
                  if (dist2 < minDist2) {
                    collide = true;
                    break;
                  }
                }
              }

          if (!collide) {
            // Accept sphere
            centers.push_back(pos);
            base.pos = pos;
            base.shape = "sphere";
            base.radius = R;
            base.density = settings.scene.populate.density;
            rods.push_back(
                RigidBody::makeSphere(pos, settings.scene.populate.density, R,
                                      base.restitution, base.friction));

            // Add to spatial grid
            glm::ivec3 cellIdx = glm::floor((pos - bmin) / cs);
            int cx = usePBC ? wrapI(cellIdx.x, n.x)
                            : glm::clamp(cellIdx.x, 0, n.x - 1);
            int cy = usePBC ? wrapI(cellIdx.y, n.y)
                            : glm::clamp(cellIdx.y, 0, n.y - 1);
            int cz = usePBC ? wrapI(cellIdx.z, n.z)
                            : glm::clamp(cellIdx.z, 0, n.z - 1);
            cells[linIdx(cx, cy, cz)].push_back(i);
            placed = true;
          }
        }
        if (!placed) {
          std::cerr << "[populate] nonoverlap spheres: failed to place sphere "
                    << i << "/" << N << " after attempts= " << maxAttempts
                    << "\n";
          break;
        }
      }
    } else if (mode == "nonoverlap") {
      // Non-overlapping initializer for RODS (works with PBC or NPBC)
      rods.clear();
      rods.reserve(N);
      const glm::vec3 bmin = pbcMin;
      const glm::vec3 bmax = pbcMax;
      const glm::vec3 boxSize = bmax - bmin;
      const float halfL = 0.5f * base.length;
      const float R = 0.5f * base.diameter;
      const float diam2 = (2.0f * R) * (2.0f * R);
      // Choose placement grid cell size ~ rod length
      const float cs =
          (cellSize > 0.0f ? cellSize
                           : std::max(0.25f * base.length, 2.5f * R));
      glm::ivec3 n = gridDims(bmin, bmax, cs);
      const int numCells = std::max(1, n.x * n.y * n.z);
      std::vector<std::vector<int>> cells(numCells);

      // Initialize constant random forces (acceleration)
      if (useConstantRandomAccel) {
        std::cout
            << "[resetScene] Initializing constant random acceleration (sigma="
            << constAccelSigma << ")\n";
        constantForces.resize(rods.size());
        std::random_device rd;
        std::mt19937 gen(
            settings.scene.populate.seed ? settings.scene.populate.seed : rd());
        std::normal_distribution<float> norm(0.0f, constAccelSigma);

        for (size_t i = 0; i < rods.size(); ++i) {
          float ax = norm(gen);
          float ay = norm(gen);
          float az = norm(gen);
          // F = m * a
          constantForces[i] = rods[i].mass * glm::vec3(ax, ay, az);
        }
      } else {
        constantForces.clear();
      }

      auto linIdx = [&](int ix, int iy, int iz) {
        return ix + n.x * (iy + n.y * iz);
      };
      auto wrapI = [&](int a, int dim) {
        int res = a % dim;
        if (res < 0)
          res += dim;
        return res;
      };

      auto segAABB = [&](const glm::vec3 &c, const glm::vec3 &u,
                         glm::vec3 &aabbMin, glm::vec3 &aabbMax) {
        glm::vec3 ext = glm::abs(u) * halfL + glm::vec3(R);
        aabbMin = c - ext;
        aabbMax = c + ext;
      };
      auto rangesForAABB = [&](const glm::vec3 &mn, const glm::vec3 &mx,
                               glm::ivec3 &i0, glm::ivec3 &i1) {
        i0 = glm::floor((mn - bmin) / cs);
        i1 = glm::floor((mx - bmin) / cs);
      };
      auto uniform_dir_s2 = [&](std::mt19937 &g) {
        float u = 2.0f * urand(g) - 1.0f;
        float phi = 2.0f * float(M_PI) * urand(g);
        float s = std::sqrt(std::max(0.0f, 1.0f - u * u));
        return glm::vec3(s * std::cos(phi), u, s * std::sin(phi));
      };
      auto quat_from_axisY = [&](const glm::vec3 &dir) {
        const glm::vec3 y(0, 1, 0);
        float d = glm::clamp(glm::dot(y, dir), -1.0f, 1.0f);
        float ang = std::acos(d);
        if (ang < 1e-6f)
          return glm::quat(1, 0, 0, 0);
        glm::vec3 axis = glm::normalize(glm::cross(y, dir));
        return glm::angleAxis(ang, axis);
      };
      auto dist2_seg_pbc = [&](const glm::vec3 &p0, const glm::vec3 &p1,
                               const glm::vec3 &q0, const glm::vec3 &q1,
                               const glm::vec3 &L) {
        float minD2 = std::numeric_limits<float>::max();

        // Check all 27 images of q
        for (int k = -1; k <= 1; ++k) {
          for (int j = -1; j <= 1; ++j) {
            for (int i = -1; i <= 1; ++i) {
              glm::vec3 shift{i * L.x, j * L.y, k * L.z};
              glm::vec3 q0w = q0 + shift;
              glm::vec3 q1w = q1 + shift;

              // Standard segment-segment distance
              glm::vec3 u = p1 - p0;
              glm::vec3 v = q1w - q0w;
              glm::vec3 w0 = p0 - q0w;

              const float eps = 1e-12f;
              float uu = glm::dot(u, u), vv = glm::dot(v, v),
                    uv = glm::dot(u, v);
              float wu = glm::dot(w0, u), wv = glm::dot(w0, v);
              float D = uu * vv - uv * uv;

              float s, t;
              if (std::abs(D) < eps) {
                s = 0.0f;
                t = (vv >= eps) ? (-wv / vv) : 0.0f;
              } else {
                s = (uv * wv - vv * wu) / D;
                t = (uu * wv - uv * wu) / D;
              }

              if (s < 0.0f)
                s = 0.0f;
              else if (s > 1.0f)
                s = 1.0f;
              t = (s * uv + wv) / (vv >= eps ? vv : 1.0f);
              if (t < 0.0f)
                t = 0.0f;
              else if (t > 1.0f)
                t = 1.0f;

              float su = (-wu + t * uv) / (uu >= eps ? uu : 1.0f);
              if (!(t > 1e-6f && t < 1.0f - 1e-6f)) {
                if (su < 0.0f)
                  s = 0.0f;
                else if (su > 1.0f)
                  s = su;
                else
                  s = su;
              }

              glm::vec3 d = (w0 + s * u) - t * v;
              float d2 = glm::dot(d, d);

              if (d2 < minD2)
                minD2 = d2;
            }
          }
        }
        return minD2;
      };
      const int maxAttempts =
          std::max(1000, settings.scene.populate.maxAttempts);

      // Store accepted centroids and directions for candidate checks
      std::vector<glm::vec3> C;
      C.reserve(N);
      std::vector<glm::vec3> U;
      U.reserve(N);

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
          glm::vec3 mn, mx;
          segAABB(c, udir, mn, mx);
          glm::ivec3 a0, a1;
          rangesForAABB(mn, mx, a0, a1);
          bool collide = false;
          // Iterate overlapped cells (handle PBC by wrapping indices)
          for (int iz = a0.z; iz <= a1.z && !collide; ++iz)
            for (int iy = a0.y; iy <= a1.y && !collide; ++iy)
              for (int ix = a0.x; ix <= a1.x && !collide; ++ix) {
                int cx = usePBC ? wrapI(ix, n.x) : glm::clamp(ix, 0, n.x - 1);
                int cy = usePBC ? wrapI(iy, n.y) : glm::clamp(iy, 0, n.y - 1);
                int cz = usePBC ? wrapI(iz, n.z) : glm::clamp(iz, 0, n.z - 1);
                const auto &bucket = cells[linIdx(cx, cy, cz)];
                for (int j : bucket) {
                  glm::vec3 cj = C[j];
                  glm::vec3 uj = U[j];
                  glm::vec3 q0 = cj - uj * halfL;
                  glm::vec3 q1 = cj + uj * halfL;

                  float d2;
                  if (usePBC) {
                    d2 = dist2_seg_pbc(p0, p1, q0, q1, boxSize);
                  } else {
                    // For non-PBC, simple single check
                    // We can reuse dist2_seg_pbc logic but passing infinite L
                    // or just a simplified call. For safety, let's just use
                    // the core logic or call the helper with a flag.
                    // Or, since dist2_seg_pbc is robust, we can just use it
                    // if we modify it to skip images if not usePBC.
                    // Instead, let's just pass a huge number for L or wrap it?
                    // Proper way:
                    d2 =
                        dist2_seg_pbc(p0, p1, q0, q1,
                                      glm::vec3(0.0f)); // 0 means do logic? No.
                    // dist2_seg_pbc uses L for shifting. If L=0, shift=0.
                    // If we pass 0, it checks 27 times the same thing.
                    // Let's optimize:
                    // But wait, the lambda I wrote iterates 27 times.
                    // That is slow for non-PBC.
                    // I should check usePBC inside the lambda or before
                    // calling.
                  }

                  // Actually, let's fix the lambda to only loop if checking PBC
                  // is needed or requested. BUT, the goal is to fix PBC. The
                  // code I pasted above does 27 checks. I will overwrite it
                  // effectively.

                  // Let's refine the replacement content to handle usePBC flag
                  // inside or out. I will assume for now we just use the robust
                  // check always for this 'nonoverlap' mode or pass L=0 and fix
                  // the loop range? Easier: just replace with the 27-loop and
                  // rely on boxSize. If usePBC is false, boxSize is technically
                  // relevant for boundaries but not for shifting images
                  // usually? Actually if usePBC is false, we shouldn't shift.
                  // I'll make the lambda smart: check 'usePBC' captured from
                  // context.

                  if (usePBC) {
                    d2 = dist2_seg_pbc(p0, p1, q0, q1, boxSize);
                  } else {
                    // Single check
                    glm::vec3 u = p1 - p0;
                    glm::vec3 v = q1 - q0;
                    glm::vec3 w0 = p0 - q0;
                    float uu = glm::dot(u, u), vv = glm::dot(v, v),
                          uv = glm::dot(u, v);
                    float wu = glm::dot(w0, u), wv = glm::dot(w0, v);
                    float D = uu * vv - uv * uv;
                    float s, t;
                    if (std::abs(D) < 1e-12f) {
                      s = 0.0f;
                      t = (vv >= 1e-12f) ? (-wv / vv) : 0.0f;
                    } else {
                      s = (uv * wv - vv * wu) / D;
                      t = (uu * wv - uv * wu) / D;
                    }
                    s = glm::clamp(s, 0.0f, 1.0f);
                    t = glm::clamp((s * uv + wv) / (vv >= 1e-12f ? vv : 1.0f),
                                   0.0f, 1.0f);
                    float su = (-wu + t * uv) / (uu >= 1e-12f ? uu : 1.0f);
                    if (t > 1e-6f && t < 1.0f - 1e-6f) {
                    } else {
                      s = glm::clamp(su, 0.0f, 1.0f);
                    }
                    glm::vec3 d = (w0 + s * u) - t * v;
                    d2 = glm::dot(d, d);
                  }

                  if (d2 < diam2) {
                    collide = true;
                    break;
                  }
                }
              }

          if (!collide) {
            // Accept: record and insert into cells
            C.push_back(c);
            U.push_back(udir);
            BodyCfg cfg = base;
            cfg.pos = c;
            glm::quat q = quat_from_axisY(udir);
            cfg.rot_quat = glm::vec4(q.w, q.x, q.y, q.z);
            rods.push_back(createRod(cfg));
            // Insert to cells
            for (int iz = a0.z; iz <= a1.z; ++iz)
              for (int iy = a0.y; iy <= a1.y; ++iy)
                for (int ix = a0.x; ix <= a1.x; ++ix) {
                  int cx = usePBC ? wrapI(ix, n.x) : glm::clamp(ix, 0, n.x - 1);
                  int cy = usePBC ? wrapI(iy, n.y) : glm::clamp(iy, 0, n.y - 1);
                  int cz = usePBC ? wrapI(iz, n.z) : glm::clamp(iz, 0, n.z - 1);
                  cells[linIdx(cx, cy, cz)].push_back(i);
                }
            placed = true;
          }
        }
        if (!placed) {
          std::cerr << "[populate] nonoverlap: failed to place rod " << i << "/"
                    << N << " after attempts= " << maxAttempts << "\n";
          break;
        }
      }
    } else if (mode == "random") {
      // Random placement without overlap check
      auto uniform_dir_s2_local = [&](std::mt19937 &gen) -> glm::vec3 {
        float u1 = urand(gen), u2 = urand(gen);
        float theta = 2.0f * 3.14159f * u1;
        float phi = acos(2.0f * u2 - 1.0f);
        return glm::vec3(sin(phi) * cos(theta), sin(phi) * sin(theta),
                         cos(phi));
      };
      auto quat_from_axisY_local = [](const glm::vec3 &axis) -> glm::quat {
        glm::vec3 up(0, 1, 0);
        glm::vec3 axis_norm = glm::normalize(axis);
        float dot = glm::dot(up, axis_norm);
        if (abs(dot - 1.0f) < 1e-6f)
          return glm::quat(1, 0, 0, 0);
        if (abs(dot + 1.0f) < 1e-6f)
          return glm::quat(0, 1, 0, 0);
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
      rodA.density = 1000.0f;
      rodA.length = 0.5f;
      rodA.diameter = 0.10f;
      rodA.restitution = 0.15f;
      rodA.friction = 0.6f;
      rodA.v_lin = {+2.2f, 0, 0};
    }

    // Adaptive broadphase cell size if requested (<= 0 => auto)
    if (usePBC && cellSize <= 0.0f && !rods.empty()) {
      double sumD = 0.0;
      for (const auto &rb : rods)
        sumD += double(rb.cap.r) * 2.0; // diameter
      double avgD = sumD / double(rods.size());
      // Slightly larger to keep occupancy per cell modest
      cellSize = float(std::max(0.05, 1.25 * avgD));
      // Reset grid buffers to force reallocation on next step
      gridN = glm::ivec3(0);
      gridCounts.clear();
      gridOffsets.clear();
      gridWrite.clear();
      gridItems.clear();
    }
  }

  // Explicit bodies from config (if not populated/loaded)
  if (rods.empty() && !settings.scene.bodies.empty()) {
    for (const auto &b : settings.scene.bodies) {
      rods.push_back(createRod(b));
    }
    if (!gQuiet) std::cout << "[App] Loaded " << rods.size()
              << " explicit bodies from scene config.\n";
  }

  if (useRandomInit) {
    const auto &ri = settings.scene.randomInit;
    std::random_device rd;
    std::mt19937 gen(ri.seed ? ri.seed : rd());
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    // Helper: uniform direction on S2
    auto uniform_dir_s2 = [&](std::mt19937 &g) {
      float u = 2.0f * uni(g) - 1.0f; // cos(theta) in [-1,1]
      float phi = 2.0f * float(M_PI) * uni(g);
      float s = std::sqrt(std::max(0.0f, 1.0f - u * u));
      return glm::vec3(s * std::cos(phi), u, s * std::sin(phi));
    };

    // Helper: build an arbitrary unit vector perpendicular to u
    auto arbitrary_perp = [](const glm::vec3 &u) -> glm::vec3 {
      glm::vec3 ref = (std::abs(u.x) < 0.9f) ? glm::vec3(1, 0, 0)
                                                : glm::vec3(0, 1, 0);
      return glm::normalize(glm::cross(u, ref));
    };

    const std::string &mode = ri.mode;

    int idx = 0;
    for (auto &rb : rods) {
      const bool willBeFixed = (settings.scene.fixEveryExcept >= 0 &&
                                idx != settings.scene.fixEveryExcept);
      if (!willBeFixed &&
          (perturbationRodIndex == -1 || idx == perturbationRodIndex)) {

        if (mode == "thermal") {
          // Equipartition: sigma_v = sqrt(kBT_trans/m), sigma_w = sqrt(kBT_rot/I_perp)
          const float kBT_trans = (ri.kBTTrans >= 0.0f) ? ri.kBTTrans : ri.kBT;
          const float kBT_rot = (ri.kBTRot >= 0.0f) ? ri.kBTRot : ri.kBT;
          const float sigma_v =
            (rb.mass > 0.0f) ? std::sqrt(kBT_trans / rb.mass) : 0.0f;
          // I_perp = I_body[0][0] (transverse MOI for rod axis along Y)
          const float I_perp = rb.I_body[0][0];
          const float sigma_w =
            (I_perp > 0.0f) ? std::sqrt(kBT_rot / I_perp) : 0.0f;

          std::normal_distribution<float> dv(0.0f, sigma_v);
          std::normal_distribution<float> dw(0.0f, sigma_w);

          rb.v = {dv(gen), dv(gen), dv(gen)};

          // Rod axis in world frame (body-frame Y rotated by quaternion)
          glm::vec3 u_ax = rb.q * glm::vec3(0, 1, 0);
          glm::vec3 e1 = arbitrary_perp(u_ax);
          glm::vec3 e2 = glm::cross(u_ax, e1);
          rb.w = dw(gen) * e1 + dw(gen) * e2; // omega_parallel = 0 by construction

        } else if (mode == "gaussian") {
          // Independent Gaussian for both v and omega
          std::normal_distribution<float> dv(0.0f, ri.vSigma);
          std::normal_distribution<float> dw(0.0f, ri.wSigma);

          rb.v = {dv(gen), dv(gen), dv(gen)};
          rb.w = {dw(gen), dw(gen), dw(gen)};

          // Optionally project out parallel spin
          if (ri.projectParallelSpin) {
            glm::vec3 u_ax = rb.q * glm::vec3(0, 1, 0);
            rb.w -= glm::dot(rb.w, u_ax) * u_ax;
          }

        } else {
          // "uniform" (legacy): uniform translational, fixed-magnitude angular on S2
          std::uniform_real_distribution<float> uniform(
              -ri.vSigma, ri.vSigma);
          rb.v = {uniform(gen), uniform(gen), uniform(gen)};
          rb.w = ri.wSpeed * uniform_dir_s2(gen);

          // Optionally project out parallel spin
          if (ri.projectParallelSpin) {
            glm::vec3 u_ax = rb.q * glm::vec3(0, 1, 0);
            rb.w -= glm::dot(rb.w, u_ax) * u_ax;
          }
        }
      }
      idx++;
    }

    if (!gQuiet) {
      std::cout << "[RandomInit] mode=\"" << mode << "\"";
      if (mode == "thermal") {
        std::cout << " kBT=" << ri.kBT;
        if (ri.kBTTrans >= 0.0f)
          std::cout << " kBTTrans=" << ri.kBTTrans;
        if (ri.kBTRot >= 0.0f)
          std::cout << " kBTRot=" << ri.kBTRot;
      }
      else if (mode == "gaussian")
        std::cout << " vSigma=" << ri.vSigma << " wSigma=" << ri.wSigma;
      else
        std::cout << " vSigma=" << ri.vSigma << " wSpeed=" << ri.wSpeed;
      std::cout << " seed=" << ri.seed
                << " projectParallelSpin=" << ri.projectParallelSpin << "\n";
    }
  }

  // Fix rod(s) if requested
  if (settings.scene.fixCentroidRod && !rods.empty()) {
    int numToFix = std::min(settings.scene.numFixedRods, (int)rods.size());
    std::vector<int> fixedIndices;

    std::cout << "[Scene] Fixing " << numToFix
              << " rod(s) (requested: " << settings.scene.numFixedRods
              << ", available: " << rods.size() << ")\n";

    if (numToFix > 0) {
      // First rod: use selection method
      int first = -1;
      std::string method = settings.scene.fixedRodSelectionMethod;

      if (method == "centroid") {
        // Find rod closest to centroid
        glm::vec3 centroid(0.0f);
        for (const auto &r : rods)
          centroid += r.x;
        centroid /= float(rods.size());

        float minDist2 = std::numeric_limits<float>::max();
        for (int i = 0; i < (int)rods.size(); ++i) {
          glm::vec3 diff = rods[i].x - centroid;
          float d2 = glm::dot(diff, diff);
          if (d2 < minDist2) {
            minDist2 = d2;
            first = i;
          }
        }
        if (first >= 0) {
          std::cout << "[Scene] Fixed centroid rod " << first << " at "
                    << rods[first].x.x << "," << rods[first].x.y << ","
                    << rods[first].x.z
                    << " (dist to centroid: " << std::sqrt(minDist2) << ")\n";
        }
      } else if (method == "horizontal") {
        // Find most horizontal rod
        float maxHorizontality = -1.0f;
        for (int i = 0; i < (int)rods.size(); ++i) {
          if (rods[i].type == ShapeType::Capsule) {
            glm::vec3 yAxis = rods[i].q * glm::vec3(0, 1, 0);
            float horizontality =
                std::sqrt(yAxis.x * yAxis.x + yAxis.z * yAxis.z);
            if (horizontality > maxHorizontality) {
              maxHorizontality = horizontality;
              first = i;
            }
          }
        }
        if (first >= 0) {
          glm::vec3 yAxis = rods[first].q * glm::vec3(0, 1, 0);
          float angle_from_vertical =
              std::acos(std::abs(yAxis.y)) * 180.0f / M_PI;
          std::cout << "[Scene] Fixed horizontal rod " << first << " at "
                    << rods[first].x.x << "," << rods[first].x.y << ","
                    << rods[first].x.z
                    << " (angle from vertical: " << angle_from_vertical
                    << " deg)\n";
        }
      } else {
        std::cerr << "[Scene] Unknown fixedRodSelectionMethod: " << method
                  << ". Valid options: 'centroid', 'horizontal'\n";
      }

      if (first >= 0) {
        fixedIndices.push_back(first);
      }

      // Additional rods: random selection
      if (numToFix > 1 && first >= 0) {
        std::random_device rd;
        std::mt19937 rng(
            settings.scene.populate.seed ? settings.scene.populate.seed : rd());
        std::vector<int> available;
        for (int i = 0; i < (int)rods.size(); ++i) {
          if (i != first)
            available.push_back(i);
        }
        std::shuffle(available.begin(), available.end(), rng);

        int remaining = numToFix - 1;
        for (int i = 0; i < remaining && i < (int)available.size(); ++i) {
          fixedIndices.push_back(available[i]);
        }
        std::cout << "[Scene] Fixed " << (numToFix - 1)
                  << " additional random rod(s): ";
        for (size_t i = 1; i < fixedIndices.size(); ++i) {
          std::cout << fixedIndices[i];
          if (i < fixedIndices.size() - 1)
            std::cout << ", ";
        }
        std::cout << "\n";
      }

      // Apply fixed state to all selected rods
      for (int idx : fixedIndices) {
        rods[idx].invMass = 0.0f;
        rods[idx].I_body_inv = glm::mat3(0.0f);
        rods[idx].v = glm::vec3(0.0f);
        rods[idx].w = glm::vec3(0.0f);
      }

      std::cout << "[Scene] Total fixed rods: " << fixedIndices.size() << "\n";
    }
  }

  // Fix all rods except one (--fix-every-except)
  if (settings.scene.fixEveryExcept >= 0 && !rods.empty()) {
    int freeIdx = settings.scene.fixEveryExcept;
    if (freeIdx >= (int)rods.size()) {
      std::cerr << "[Scene] --fix-every-except index " << freeIdx
                << " out of range (only " << rods.size()
                << " rods). Clamping to 0.\n";
      freeIdx = 0;
    }
    int fixedCount = 0;
    for (int i = 0; i < (int)rods.size(); ++i) {
      if (i == freeIdx)
        continue;
      rods[i].invMass = 0.0f;
      rods[i].I_body_inv = glm::mat3(0.0f);
      rods[i].v = glm::vec3(0.0f);
      rods[i].w = glm::vec3(0.0f);
      ++fixedCount;
    }
    std::cout << "[Scene] Fixed " << fixedCount << " rod(s); rod " << freeIdx
              << " is free.\n";
  }

  // Apply manual velocity override if configured
  if (overrideVelEnabled && overrideVelId >= 0 &&
      overrideVelId < (int)rods.size()) {
    rods[overrideVelId].v = overrideVel;
    std::cerr << "[resetScene] Overrode velocity for rod " << overrideVelId
              << " to " << overrideVel.x << "," << overrideVel.y << ","
              << overrideVel.z << "\n";
  }

  // Apply manual angular velocity override if configured
  if (overrideAngVelEnabled && overrideAngVelId >= 0 &&
      overrideAngVelId < (int)rods.size()) {
    rods[overrideAngVelId].w = overrideAngVel;
    std::cerr << "[resetScene] Overrode angular velocity for rod " << overrideAngVelId
              << " to " << overrideAngVel.x << "," << overrideAngVel.y << ","
              << overrideAngVel.z << "\n";
  }

  // Initialize constant random forces (acceleration)
  if (useConstantRandomAccel) {
    std::cout
        << "[resetScene] Initializing constant random acceleration (sigma="
        << constAccelSigma << ")\n";
    constantForces.resize(rods.size());
    std::random_device rd;
    std::mt19937 gen(settings.scene.populate.seed ? settings.scene.populate.seed
                                                  : rd());
    std::normal_distribution<float> norm(0.0f, constAccelSigma);

    for (size_t i = 0; i < rods.size(); ++i) {
      if (rods[i].invMass <= 0.0f) {
        constantForces[i] = glm::vec3(0.0f);
        continue;
      }
      float ax = norm(gen);
      float ay = norm(gen);
      float az = norm(gen);
      // F = m * a
      constantForces[i] = rods[i].mass * glm::vec3(ax, ay, az);
    }
  } else {
    constantForces.clear();
  }

  // init sleeping arrays
  sleeping.assign(rods.size(), 0);
  sleepTimer.assign(rods.size(), 0.f);

  // Auto-tuning: for small N, switch to serial execution if user didn't force
  // parallel
  static bool autoSerialMode = false;
  if (!g_user_threads_set && (g_thread_limit == 0 || autoSerialMode)) {
    if (rods.size() > 0 && rods.size() < 256) {
      if (g_thread_limit != 1) {
        std::cout
            << (gQuiet ? "" : "[App] Auto-switching to serial mode (threads=1) for small N=")
            << rods.size() << "\n";
        g_thread_limit = 1;
        autoSerialMode = true;
      }
    } else {
      if (autoSerialMode) {
        std::cout << "[App] Restoring parallel mode (default threads) for N="
                  << rods.size() << "\n";
        g_thread_limit = 0;
        autoSerialMode = false;
      }
    }
  }

  // Save populated configuration if requested
  if (!saveInitPath.empty() && !rods.empty()) {
    saveInitialConfigCSV(saveInitPath);
  }

  // Reset KE history for adaptive decisions
  lastKE = totalKE();
  prevFrameKE = lastKE;
  lastFrameKEDelta = 0.0;
}

// Load initial configuration from CSV with endpoints per rod:
// x0,y0,z0,x1,y1,z1
// OR center/angles if header detected:
// x y z phi theta length
bool App::loadInitialConfigCSV(const std::string &path) {
  std::filesystem::path p(path);
  std::ifstream in(p);
  if (!in) {
    // Attempt fallback search upward for relative paths (common when
    // running from build/)
    std::cerr << "[init-csv] Primary open failed: " << path
              << ". Trying fallbacks...\n";
    std::filesystem::path cur = std::filesystem::current_path();
    bool found = false;
    for (int up = 0; up < 5 && !found; ++up) {
      std::filesystem::path candidate = cur / p;
      if (std::filesystem::exists(candidate)) {
        in.open(candidate);
        if (in) {
          found = true;
          std::cerr << "[init-csv] Opened via candidate: " << candidate.string()
                    << "\n";
          break;
        }
      }
      cur = cur.parent_path();
    }
    if (!found) {
      std::cerr << "[init-csv] Cannot locate file after fallback attempts: "
                << path << "\n";
      return false;
    }
  }
  // Defaults from settings (if available)
  float defaultLength = 0.5f;
  float defaultDiameter = 0.05f;
  float defaultDensity = 1000.0f;
  float defaultRestitution = 0.15f;
  float defaultFriction = 0.6f;
  if (settings.scene.populate.count > 0 ||
      settings.scene.populate.density > 0) {
    const auto &p = settings.scene.populate;
    // Assuming baseRad and baseDensity are members of App or accessible
    // here and that p.radius and p.density are valid for default values.
    // This part of the snippet seems to be a mix-up with populate logic.
    // Reinterpreting to set defaults based on populate settings if
    // available.
    if (p.radius > 0.0f)
      defaultDiameter = p.radius * 2.0f; // Assuming radius is half diameter
    if (p.density > 0.0f)
      defaultDensity = p.density;
  }

  // If explicit bodies are provided in config (e.g. from JSON), use the first
  // one as template for physical properties of rods loaded from CSV.
  if (!settings.scene.bodies.empty()) {
    const auto &b = settings.scene.bodies[0];
    defaultLength = b.length;
    defaultDiameter = b.diameter;
    defaultDensity = b.density;
    defaultRestitution = b.restitution;
    defaultFriction = b.friction;
    // Note: friction_s / friction_d not yet used in createRod logic below
    // unless we update it
  }

  // Format detection
  enum InputFormat { ENDPOINTS_CSV, CENTER_ANGLES_TXT };
  InputFormat fmt = ENDPOINTS_CSV;
  char delimiter = ',';
  bool ignoreDiameterCol =
      false; // Flag to prevent treating attempt count as diameter
  bool hadSchemaHeader = false;

  auto isEndpointsHeader = [](std::string s) {
    // Accept both comma-separated and whitespace-separated headers.
    // Examples:
    //   x0,y0,z0,x1,y1,z1
    //   x0 y0 z0 x1 y1 z1
    //   x0,y0,z0,x1,y1,z1,attempts
    for (char &c : s) {
      if (c == ',' || c == '\t')
        c = ' ';
      else
        c = char(std::tolower((unsigned char)c));
    }
    // collapse whitespace
    std::string t;
    t.reserve(s.size());
    bool prevSpace = true;
    for (char c : s) {
      bool sp = (c == ' ' || c == '\r' || c == '\n');
      if (sp) {
        if (!prevSpace)
          t.push_back(' ');
        prevSpace = true;
      } else {
        t.push_back(c);
        prevSpace = false;
      }
    }
    // trim
    while (!t.empty() && t.front() == ' ')
      t.erase(t.begin());
    while (!t.empty() && t.back() == ' ')
      t.pop_back();

    return t.find("x0 y0 z0 x1 y1 z1") != std::string::npos;
  };

  // Parse optional metadata headers starting with '#'
  std::string line;
  bool sawHeader = false;
  size_t lineCount = 0;
  size_t dataRows = 0;
  size_t skippedMalformed = 0;
  while (std::getline(in, line)) {
    // Stop if we have loaded the requested number of rods
    if (settings.scene.populate.count > 0 &&
        dataRows >= static_cast<size_t>(settings.scene.populate.count)) {
      break;
    }
    ++lineCount;
    if (line.empty())
      continue;
    // Check for specific format header (checking before comment check in case
    // it's not commented)
    if (isEndpointsHeader(line)) {
      fmt = ENDPOINTS_CSV;
      delimiter = (line.find(',') != std::string::npos) ? ',' : ' ';
      sawHeader = true;
      hadSchemaHeader = true;
      if (line.find("attempts") != std::string::npos) {
        ignoreDiameterCol = true;
      }
      continue;
    }

    if (line[0] == '#') {
      // Check for Text format header if commented
      if (line.find("Rod configuration: x y z phi theta length") !=
          std::string::npos) {
        fmt = CENTER_ANGLES_TXT;
        delimiter = ' ';
        sawHeader = true;
        hadSchemaHeader = true;
        continue;
      }

      // Metadata overrides
      // scene) e.g., "# rod_length=1" "# rod_diameter=0.01" "# pbc=true"
      // "# box_size=1.1"
      auto eq = line.find('=');
      if (eq != std::string::npos) {
        std::string key = line.substr(1, eq - 1);
        std::string val = line.substr(eq + 1);
        // trim spaces
        auto trim = [](std::string s) {
          size_t a = s.find_first_not_of(" \t\r\n");
          size_t b = s.find_last_not_of(" \t\r\n");
          if (a == std::string::npos)
            return std::string();
          return s.substr(a, b - a + 1);
        };
        key = trim(key);
        val = trim(val);
        try {
          if (key == "rod_length")
            defaultLength = std::stof(val);
          else if (key == "rod_diameter") {
            defaultDiameter = std::stof(val);
            if (!gQuiet) std::cout << "[init-csv] Parsed rod_diameter=" << defaultDiameter
                      << "\n";
          } else if (key == "rod_radius") {
            defaultDiameter = std::stof(val) * 2.0f;
            if (!gQuiet) std::cout << "[init-csv] Parsed rod_radius="
                      << (defaultDiameter * 0.5f) << "\n";
          } else if (key == "pbc") {
            bool v = (val == "1" || val == "true" || val == "True");
            usePBC = v;
            g_pbc_enabled = v;
          } else if (key == "box_size") {
            float L = std::stof(val);
            // Set a symmetric box centered around origin: [-L/2, +L/2] in
            // each axis
            pbcMin = glm::vec3(-0.5f * L);
            pbcMax = glm::vec3(+0.5f * L);
            g_pbc_min = pbcMin;
            g_pbc_max = pbcMax;
          }
        } catch (...) {
          // ignore malformed header values
        }
      }
      continue;
    }

    // Heuristics if no header seen yet:
    // If line contains no commas and 6 numbers separated by space ->
    // CENTER_ANGLES
    if (!sawHeader && fmt == ENDPOINTS_CSV &&
        line.find(',') == std::string::npos) {
      // Check tokens
      std::stringstream ss(line);
      int validNums = 0;
      double tmp;
      while (ss >> tmp)
        validNums++;
      if (validNums == 6) {
        fmt = ENDPOINTS_CSV;
        delimiter = ' ';
      }
      sawHeader = true; // Assume data starts now
    }

    if (!sawHeader) {
      sawHeader = true;
      // No header: default to endpoint schema; choose delimiter from first data
      // line.
      delimiter = (line.find(',') != std::string::npos) ? ',' : ' ';
    }

    // Data row parsing
    std::vector<double> vals;
    if (delimiter == ',' && line.find(',') != std::string::npos) {
      std::stringstream ss(line);
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) {
          try {
            vals.push_back(std::stod(tok));
          } catch (...) {
          }
        }
      }
    } else {
      // whitespace separated (space/tab)
      std::stringstream ss(line);
      double v;
      while (ss >> v)
        vals.push_back(v);
    }

    if (vals.size() < 6) {
      ++skippedMalformed;
      continue;
    }

    glm::vec3 p0, p1;
    if (fmt == CENTER_ANGLES_TXT) {
      // x y z phi theta length
      float cx = float(vals[0]);
      float cy = float(vals[1]);
      float cz = float(vals[2]);
      float phi = float(vals[3]);
      float theta = float(vals[4]);
      float L = float(vals[5]);

      // theta is usually azimuthal (0..2pi), phi is polar (0..pi) in typical
      // physics checking rsa_pbc.cpp: u = {sin(phi)*cos(theta),
      // sin(phi)*sin(theta), cos(phi)} x,y,z corresponds to 0,1,2
      float s = std::sin(phi);
      glm::vec3 u(s * std::cos(theta), s * std::sin(theta), std::cos(phi));
      p0 = glm::vec3(cx, cy, cz) - u * (0.5f * L);
      p1 = glm::vec3(cx, cy, cz) + u * (0.5f * L);

      // Override default length for this rod if provided
      // But createRod uses BodyCfg which has length.
      // We will store endpoints. When createRod is called with just endpoints?
      // We actually need to reconstruction pos/rot.
      // Actually below we construct BodyCfg.

      // Let's ensure defaultLength is updated if not variable?
      // Or better, just use p0,p1 to define the rod.
    } else {
      p0 = glm::vec3(vals[0], vals[1], vals[2]);
      p1 = glm::vec3(vals[3], vals[4], vals[5]);
    }

    // Create rod from endpoints p0, p1
    glm::vec3 center = 0.5f * (p0 + p1);
    glm::vec3 axis = p1 - p0;
    float len = glm::length(axis);
    if (len < 1e-6f) {
      ++skippedMalformed;
      continue;
    }
    glm::vec3 dir = axis / len;

    // Orientation: align local Y (0,1,0) to dir
    // Quat rotation from (0,1,0) to dir
    glm::vec3 up(0, 1, 0);
    glm::quat q;
    float d = glm::dot(up, dir);
    if (d > 0.999999f) {
      q = glm::quat(1, 0, 0, 0);
    } else if (d < -0.999999f) {
      q = glm::quat(0, 0, 0, 1); // 180 deg around Z
    } else {
      glm::vec3 c = glm::cross(up, dir);
      float s = std::sqrt((1.0f + d) * 2.0f);
      float invs = 1.0f / s;
      q = glm::quat(s * 0.5f, c.x * invs, c.y * invs, c.z * invs);
    }

    // Normalize
    q = glm::normalize(q);

    float diameter = defaultDiameter;
    if (vals.size() >= 7 && !ignoreDiameterCol) {
      // If we don't have a header, a 7th column is often an attempt count
      // (integer-ish and typically much larger than a diameter). Avoid
      // accidentally interpreting attempts as diameter.
      if (!hadSchemaHeader) {
        double v = vals[6];
        double nearestInt = std::round(v);
        bool looksInteger = std::abs(v - nearestInt) < 1e-9;
        if (!(looksInteger && v >= 1.0) && v > 0.0 && v < 0.5) {
          diameter = float(v);
        }
      } else {
        diameter = float(vals[6]);
      }
    }

    BodyCfg cfg;
    cfg.shape = "capsule"; // Ensure it's a rod
    cfg.pos = center;
    cfg.rot_quat = glm::vec4(q.w, q.x, q.y, q.z);
    cfg.length = len; // Use actual length from file
    cfg.diameter = diameter;
    cfg.density = defaultDensity;
    cfg.restitution = defaultRestitution;
    cfg.friction = defaultFriction;
    cfg.is_static = false; // Initial state implies dynamic usually

    rods.push_back(createRod(cfg));
    ++dataRows;
  }

  if (skippedMalformed > 0) {
    std::cerr << "[init-csv] Skipped " << skippedMalformed
              << " malformed/short lines.\n";
  }
  if (!gQuiet) std::cerr << "[init-csv] Parsed rows=" << dataRows
            << " (malformed=" << skippedMalformed
            << ") header=" << (hadSchemaHeader ? "yes" : "no")
            << " fileLines=" << lineCount << "\n";
  if (rods.empty()) {
    std::cerr << "[init-csv] No rods created (expected 6 numeric columns: "
                 "x0 y0 z0 x1 y1 z1, with optional '#' metadata/header).\n";
  }
  return !rods.empty();
}

bool App::saveInitialConfigCSV(const std::string &path) {
  std::ofstream out(path);
  if (!out) {
    std::cerr << "[save-init] Failed to open for writing: " << path << "\n";
    return false;
  }
  out << std::fixed << std::setprecision(6);
  if (!rods.empty()) {
    out << "# rod_length=" << rods[0].cap.h * 2.0f << "\n";
    out << "# rod_diameter=" << rods[0].cap.r * 2.0f << "\n";
  }
  out << "# pbc=" << (usePBC ? "true" : "false") << "\n";
  if (usePBC) {
    out << "# box_size=" << (pbcMax.x - pbcMin.x) << "\n";
  }
  out << "x0,y0,z0,x1,y1,z1\n";
  for (const auto &rod : rods) {
    glm::vec3 c = rod.x;
    glm::vec3 dir = rod.q * glm::vec3(0, 1, 0);
    float halfL = rod.cap.h;
    glm::vec3 p0 = c - dir * halfL;
    glm::vec3 p1 = c + dir * halfL;
    out << p0.x << "," << p0.y << "," << p0.z << "," << p1.x << "," << p1.y
        << "," << p1.z << "\n";
  }
  std::cerr << "[save-init] Saved " << rods.size() << " rods to " << path
            << "\n";
  return true;
}

bool App::loadInitialStateCSV(const std::string &path) {
  std::filesystem::path p(path);
  std::ifstream in(p);
  if (!in) {
    std::cerr << "[init-state] Failed to open: " << path << "\n";
    return false;
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty())
      lines.push_back(line);
  }

  if (lines.empty())
    return false;

  // Parse header
  std::stringstream ss(lines[0]);
  std::string col;
  std::vector<std::string> headers;
  while (std::getline(ss, col, ',')) {
    // trim
    size_t first = col.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
      continue;
    size_t last = col.find_last_not_of(" \t\r\n");
    headers.push_back(col.substr(first, (last - first + 1)));
  }

  auto getColIdx = [&](const std::string &name) -> int {
    for (size_t i = 0; i < headers.size(); ++i)
      if (headers[i] == name)
        return i;
    return -1;
  };

  int idxFrame = getColIdx("frame");
  int idxPx = getColIdx("px");
  int idxPy = getColIdx("py");
  int idxPz = getColIdx("pz");
  int idxVx = getColIdx("vx");
  int idxVy = getColIdx("vy");
  int idxVz = getColIdx("vz");
  int idxWx = getColIdx("wx");
  int idxWy = getColIdx("wy");
  int idxWz = getColIdx("wz");
  int idxQw = getColIdx("qw");
  int idxQx = getColIdx("qx");
  int idxQy = getColIdx("qy");
  int idxQz = getColIdx("qz");

  if (idxFrame < 0 || idxPx < 0 || idxQw < 0) {
    std::cerr << "[init-state] Missing required columns in CSV (need frame, "
                 "px, qw, etc).\n";
    return false;
  }

  struct RodState {
    glm::vec3 p, v, w;
    glm::quat q;
  };
  std::map<uint64_t, std::vector<RodState>> frames;

  for (size_t i = 1; i < lines.size(); ++i) {
    std::stringstream ls(lines[i]);
    std::string valStr;
    std::vector<double> row;
    while (std::getline(ls, valStr, ',')) {
      try {
        row.push_back(std::stod(valStr));
      } catch (...) {
        row.push_back(0.0);
      }
    }

    if (row.size() < headers.size())
      continue;

    uint64_t f = (uint64_t)row[idxFrame];
    RodState s;
    s.p = glm::vec3(row[idxPx], row[idxPy], row[idxPz]);
    s.v = glm::vec3(row[idxVx], row[idxVy], row[idxVz]);
    s.w = glm::vec3(row[idxWx], row[idxWy], row[idxWz]);
    s.q = glm::quat(row[idxQw], row[idxQx], row[idxQy], row[idxQz]);

    frames[f].push_back(s);
  }

  if (frames.empty())
    return false;

  uint64_t lastF = frames.rbegin()->first;
  const auto &states = frames.rbegin()->second;

  std::cerr << "[init-state] Found " << frames.size()
            << " frames. Loading frame " << lastF << " with " << states.size()
            << " rods.\n";

  // Default properties
  float defLen = 0.5f, defRad = 0.025f, defDens = 1000.0f, defRest = 0.5f,
        defFric = 0.5f;
  if (settings.scene.populate.count > 0 ||
      settings.scene.populate.density > 0) {
    defRad = settings.scene.populate.radius;
    defDens = settings.scene.populate.density;
  }

  rods.clear();
  for (const auto &s : states) {
    RigidBody rb;
    if (settings.scene.populate.shape == "capsule") {
      rb = RigidBody::makeCapsule(s.p, s.q, defDens, defRad, defLen * 0.5f,
                                  defRest, defFric);
    } else {
      rb = RigidBody::makeSphere(s.p, defDens, defRad, defRest, defFric);
      rb.q = s.q;
    }
    rb.v = s.v;
    rb.w = s.w;
    rods.push_back(rb);
  }

  frameIndex = lastF;
  return true;
}

#ifndef HEADLESS_BUILD
void App::keyCB(GLFWwindow *window, int key, int, int action, int) {
  if (action != GLFW_PRESS)
    return;

  auto *self = static_cast<App *>(glfwGetWindowUserPointer(window));
  switch (key) {
  case GLFW_KEY_ESCAPE:
    glfwSetWindowShouldClose(window, 1);
    break;
  case GLFW_KEY_SPACE:
    self->paused = !self->paused;
    if (self->inPlaybackMode) {
      std::cout << "[Playback] " << (self->paused ? "Paused" : "Playing")
                << "\n";
    }
    break;
  case GLFW_KEY_S:
    self->paused = true;
    break;
  case GLFW_KEY_G:
    self->paused = false;
    break;
  case GLFW_KEY_RIGHT:
    if (self->inPlaybackMode) {
      if (self->currentPlaybackFrame < self->totalPlaybackFrames - 1) {
        self->currentPlaybackFrame++;
        self->loadPlaybackFrame(self->currentPlaybackFrame);
        std::cout << "[Playback] Frame " << self->currentPlaybackFrame << " / "
                  << self->totalPlaybackFrames << "\n";
      } else {
        std::cout << "[Playback] Already at last frame\n";
      }
    } else if (self->paused) {
      self->stepSingle = true;
    }
    break;
  case GLFW_KEY_LEFT:
    if (self->inPlaybackMode) {
      if (self->currentPlaybackFrame > 0) {
        self->currentPlaybackFrame--;
        self->loadPlaybackFrame(self->currentPlaybackFrame);
        std::cout << "[Playback] Frame " << self->currentPlaybackFrame << " / "
                  << self->totalPlaybackFrames << "\n";
      } else {
        std::cout << "[Playback] Already at first frame\n";
      }
    } else {
      std::cout
          << "[App] Backward stepping not supported in live simulation.\n";
    }
    break;
  case GLFW_KEY_R:
    self->resetScene();
    break;
  case GLFW_KEY_V:
    self->vsync = !self->vsync;
    glfwSwapInterval(self->vsync ? 1 : 0);
    break;
  case GLFW_KEY_K:
    self->showContactForces = !self->showContactForces;
    std::cout << "[Viz] Contact forces: "
              << (self->showContactForces ? "ON" : "OFF") << "\n";
    break;
  case GLFW_KEY_F:
    self->forceFadingEnabled = !self->forceFadingEnabled;
    std::cout << "[Viz] Force Fading: "
              << (self->forceFadingEnabled ? "ON" : "OFF") << "\n";
    break;
  case GLFW_KEY_LEFT_BRACKET:
    if (self->rods.empty())
      break;
    if (self->viewRodIndex <= -1)
      self->viewRodIndex = int(self->rods.size()) - 1;
    else
      self->viewRodIndex--;
    if (self->viewRodIndex < -1)
      self->viewRodIndex = int(self->rods.size()) - 1;
    std::cout << "[Viz] Viewing rod: "
              << (self->viewRodIndex == -1 ? "ALL"
                                           : std::to_string(self->viewRodIndex))
              << "\n";
    break;
  case GLFW_KEY_RIGHT_BRACKET:
    if (self->rods.empty())
      break;
    self->viewRodIndex++;
    if (self->viewRodIndex >= (int)self->rods.size())
      self->viewRodIndex = -1;
    std::cout << "[Viz] Viewing rod: "
              << (self->viewRodIndex == -1 ? "ALL"
                                           : std::to_string(self->viewRodIndex))
              << "\n";
    break;
  case GLFW_KEY_BACKSLASH:
    self->viewRodIndex = -1;
    std::cout << "[Viz] Viewing rod: ALL\n";
    break;
  case GLFW_KEY_HOME:
    if (self->inPlaybackMode && self->totalPlaybackFrames > 0) {
      self->currentPlaybackFrame = 0;
      self->loadPlaybackFrame(self->currentPlaybackFrame);
      std::cout << "[Playback] Jumped to first frame (0 / "
                << self->totalPlaybackFrames << ")\n";
    }
    break;
  case GLFW_KEY_END:
    if (self->inPlaybackMode && self->totalPlaybackFrames > 0) {
      self->currentPlaybackFrame = self->totalPlaybackFrames - 1;
      self->loadPlaybackFrame(self->currentPlaybackFrame);
      std::cout << "[Playback] Jumped to last frame ("
                << self->currentPlaybackFrame << " / "
                << self->totalPlaybackFrames << ")\n";
    }
    break;
  case GLFW_KEY_EQUAL: // '+' key (increase speed)
    if (self->inPlaybackMode) {
      self->playbackSpeedMultiplier =
          std::min(4.0f, self->playbackSpeedMultiplier * 2.0f);
      std::cout << "[Playback] Speed: " << self->playbackSpeedMultiplier
                << "x\n";
    }
    break;
  case GLFW_KEY_MINUS: // '-' key (decrease speed)
    if (self->inPlaybackMode) {
      self->playbackSpeedMultiplier =
          std::max(0.25f, self->playbackSpeedMultiplier * 0.5f);
      std::cout << "[Playback] Speed: " << self->playbackSpeedMultiplier
                << "x\n";
    }
    break;
  default:
    break;
  }
}

void App::cursorCB(GLFWwindow *window, double x, double y) {
  auto *self = static_cast<App *>(glfwGetWindowUserPointer(window));
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
  if (self->cam.pitch < -1.2f)
    self->cam.pitch = -1.2f;
  if (self->cam.pitch > +1.2f)
    self->cam.pitch = +1.2f;
}

void App::mouseCB(GLFWwindow *window, int button, int action, int) {
  auto *self = static_cast<App *>(glfwGetWindowUserPointer(window));
  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    self->dragging = (action == GLFW_PRESS);
  }
}

void App::scrollCB(GLFWwindow *window, double, double dy) {
  auto *self = static_cast<App *>(glfwGetWindowUserPointer(window));
  self->cam.dist *= std::exp(-0.1f * float(dy));

  // Clamp camera distance
  if (self->cam.dist < 2.0f)
    self->cam.dist = 2.0f;
  if (self->cam.dist > 30.0f)
    self->cam.dist = 30.0f;
}
#endif

#ifndef HEADLESS_BUILD
void App::maybeUpdateWindowTitle() {
  if (!profilingEnabled || !window)
    return;
  using clock = std::chrono::high_resolution_clock;
  auto now = clock::now();
  if (lastTitleUpdate.time_since_epoch().count() == 0)
    lastTitleUpdate = now;
  sumTimes += curTimes;
  sumFrames++;
  curTimes.reset();
  double sec = std::chrono::duration<double>(now - lastTitleUpdate).count();
  if (sec < 0.5)
    return;
  double invF = sumFrames > 0 ? 1.0 / double(sumFrames) : 0.0;
  double fps = sec > 0 ? double(sumFrames) / sec : 0.0;
  double bp = sumTimes.broadphase * invF;
  double sv = (sumTimes.solve + sumTimes.floorSolve) * invF;
  double rd = sumTimes.render * invF;
  double bpPairs = sumTimes.bpPairs * invF;
  std::ostringstream ss;
  ss.setf(std::ios::fixed);
  ss.precision(1);
  ss << "Frame " << frameIndex << " | Rods: " << rods.size()
     << " | KE: " << std::setprecision(6) << lastKE << " J"
     << " | FPS " << std::setprecision(0) << fps << " | BP "
     << std::setprecision(1) << bp << " ms (pairs " << bpPairs << ")"
     << " | Solve " << sv << " ms | Render " << rd << " ms";
  glfwSetWindowTitle(window, ss.str().c_str());
  sumTimes = Times{};
  sumFrames = 0;
  lastTitleUpdate = now;
}
#endif

void App::logCsvFrame() {
  if (!csvEnabled || !csvStream)
    return;
  static bool debugPrinted = false;
  if (!debugPrinted) {
    if (!gQuiet) std::cerr << "[Debug] profilingEnabled=" << profilingEnabled << " softOrHM="
              << (settings.physics.soft_contact.enabled ||
                  settings.physics.hertz_mindlin.enabled)
              << " use_mujoco=" << settings.physics.use_mujoco_contact
              << " integrate=" << curTimes.integrate
              << " broadphase=" << curTimes.broadphase << "\n";
    debugPrinted = true;
  }
  if (!shouldLogThisFrame())
    return;
  // Choose a slim header when soft-contact (or Hertz-Mindlin) is enabled
  // to avoid zero columns
  const bool softOrHM = settings.physics.soft_contact.enabled ||
                        settings.physics.hertz_mindlin.enabled;
  if (!csvHeaderWritten) {
    if (softOrHM) {
      csvStream << "frame,rods,integrate_ms,sleep_ms,broadphase_ms,bpCount_ms,"
                   "bpPrefix_ms,bpFill_ms,bpPairs_ms,solve_ms,pbcWrap_ms,"
                   "render_ms,"
                   "contacts,KE,soft_PE,gyration_sq,reldisp_sq,ent_pairs,ent_"
                   "sum,step_ms\n";
    } else {
      csvStream << "frame,rods,integrate_ms,sleep_ms,broadphase_ms,bpCount_ms,"
                   "bpPrefix_ms,bpFill_ms,bpPairs_ms,bpLongLong_ms,warmstart_"
                   "ms,buildIslands_ms,solve_ms,floorSolve_ms,posCorrect_ms,"
                   "pbcWrap_ms,render_ms,contacts,islands,KE,KE_after_"
                   "integrate,KE_after_warmstart,KE_after_solve,KE_after_"
                   "posCorrect,KE_after_pbcWrap,soft_PE,gyration_sq,reldisp_sq,"
                   "jn_sum,jt_sum,nsc_residual,ent_pairs,ent_sum,step_ms\n";
    }
    csvHeaderWritten = true;
  }
  double gyr_sq = computeGyrationSq();
  double reldisp_sq = computeRelativeMotionSq();
  if (softOrHM) {
    csvStream << frameIndex << ',' << rods.size() << ',' << curTimes.integrate
              << ',' << curTimes.sleepUpdate << ',' << curTimes.broadphase
              << ',' << curTimes.bpCount << ',' << curTimes.bpPrefix << ','
              << curTimes.bpFill << ',' << curTimes.bpPairs << ','
              << curTimes.solve << ',' << curTimes.pbcWrap << ','
              << curTimes.render << ',' << lastHitCount << ',' << lastKE << ','
              << lastSoftPotentialEnergy << ',' << gyr_sq << ',' << reldisp_sq
              << ',' << lastEntanglementPairs << ',' << lastEntanglementSum
              << ',' << lastStepWallMs << '\n';
  } else {
    csvStream << frameIndex << ',' << rods.size() << ',' << curTimes.integrate
              << ',' << curTimes.sleepUpdate << ',' << curTimes.broadphase
              << ',' << curTimes.bpCount << ',' << curTimes.bpPrefix << ','
              << curTimes.bpFill << ',' << curTimes.bpPairs << ','
              << curTimes.bpLongLong << ',' << curTimes.warmstart << ','
              << curTimes.buildIslands << ',' << curTimes.solve << ','
              << curTimes.floorSolve << ',' << curTimes.posCorrect << ','
              << curTimes.pbcWrap << ',' << curTimes.render << ','
              << lastHitCount << ',' << lastIslandCount << ',' << lastKE << ','
              << keAfterIntegrate << ',' << keAfterWarmstart << ','
              << keAfterSolve << ',' << keAfterPosCorrect << ','
              << keAfterPBCWrap << ',' << lastSoftPotentialEnergy << ','
              << gyr_sq << ',' << reldisp_sq << ',';
    // Compute NSC impulse sums from manifolds.
    {
      double jnSum = 0.0, jtSum = 0.0;
      for (const auto& m : nscSolver.getManifolds()) {
        jnSum += std::abs(m.lambda_n);
        jtSum += std::abs(m.lambda_t1) + std::abs(m.lambda_t2);
      }
      csvStream << jnSum << ',' << jtSum << ','
                << nscSolver.getLastResidual();
    }
    csvStream << ',' << lastEntanglementPairs << ',' << lastEntanglementSum
              << ',' << lastStepWallMs << '\n';
  }
  if ((frameIndex & 0x3F) == 0)
    csvStream.flush();
}

double App::totalKE() const {
  double KE = 0.0;
  for (const auto &rb : rods) {
    double v2 = glm::dot(rb.v, rb.v);
    KE += 0.5 * double(rb.mass) * v2;
    glm::mat3 Iw = rb.R() * rb.I_body * glm::transpose(rb.R());
    glm::vec3 Iw_w = Iw * rb.w;
    KE += 0.5 * double(glm::dot(rb.w, Iw_w));
  }
  return KE;
}

// Per-rod logging implementation
void App::enablePerRod(const std::string &path, int maxFrames) {
  perRodPath = path.empty() ? std::string("perrod.csv") : path;
  perRodStream.open(perRodPath, std::ios::out | std::ios::trunc);
  if (!perRodStream) {
    std::cerr << "Failed to open per-rod CSV file: " << perRodPath << "\n";
    perRodEnabled = false;
    return;
  }
  perRodEnabled = true;
  perRodHeaderWritten = false;
  perRodMaxFrames = std::max(1, maxFrames);
  perRodWrittenFrames = 0;
  // Compute sampling skip when running headless (approximate total frames
  // known), unless explicit stride set
  if (!explicitPerRodStride) {
    perRodSkip = 1;
    if (headless && headlessSteps > 0)
      perRodSkip = std::max(1, headlessSteps / perRodMaxFrames);
  }
}

void App::enableTestRodEndpoints(const std::string &path) {
  testRodEndpointsPath =
      path.empty() ? std::string("test_rod_endpoints.csv") : path;
  testRodEndpointsStream.open(testRodEndpointsPath,
                              std::ios::out | std::ios::trunc);
  if (!testRodEndpointsStream) {
    std::cerr << "Failed to open test-rod endpoints CSV file: "
              << testRodEndpointsPath << "\n";
    testRodEndpointsEnabled = false;
    return;
  }
  testRodEndpointsEnabled = true;
  testRodEndpointsHeaderWritten = false;
  testRodEndpointsWrittenFrames = 0;
  if (!explicitTestRodEndpointsStride && headless && headlessSteps > 0 &&
      testRodEndpointsMaxFrames > 0) {
    testRodEndpointsStride =
        std::max(1, headlessSteps / testRodEndpointsMaxFrames);
  }
}

void App::logPerRodFrame() {
  if (!perRodEnabled || !perRodStream)
    return;
  if (!perRodHeaderWritten) {
    if (rods.empty())
      return;
    const auto &rb = rods[0];
    float r = (rb.type == ShapeType::Capsule) ? rb.cap.r : rb.sphere.r;
    float l = (rb.type == ShapeType::Capsule) ? rb.cap.h * 2.0f : 0.0f;
    perRodStream << "# rod_radius=" << r << "\n";
    perRodStream << "# rod_length=" << l << "\n";
    perRodStream << "frame,rod,px,py,pz,vx,vy,vz,wx,wy,wz,qw,qx,qy,qz,KE_lin,"
                    "KE_rot,KE_total\n";
    perRodStream.flush();
    perRodHeaderWritten = true;
  }
  if (perRodWrittenFrames >= perRodMaxFrames)
    return;
  // Respect square wave or snapshot-only logging if configured
  if (!shouldLogThisFrame())
    return;
  if ((frameIndex % perRodSkip) != 0)
    return;
  for (size_t i = 0; i < rods.size(); ++i) {
    const auto &rb = rods[i];
    double ke_lin = 0.5 * double(rb.mass) * double(glm::dot(rb.v, rb.v));
    glm::mat3 Iw = rb.R() * rb.I_body * glm::transpose(rb.R());
    glm::vec3 Iw_w = Iw * rb.w;
    double ke_rot = 0.5 * double(glm::dot(Iw_w, rb.w));
    double ke_total = ke_lin + ke_rot;
    perRodStream << frameIndex << ',' << i << ',' << rb.x.x << ',' << rb.x.y
                 << ',' << rb.x.z << ',' << rb.v.x << ',' << rb.v.y << ','
                 << rb.v.z << ',' << rb.w.x << ',' << rb.w.y << ',' << rb.w.z
                 << ',' << rb.q.w << ',' << rb.q.x << ',' << rb.q.y << ','
                 << rb.q.z << ',' << ke_lin << ',' << ke_rot << ',' << ke_total
                 << '\n';
  }
  ++perRodWrittenFrames;
  if ((frameIndex & 0x3F) == 0)
    perRodStream.flush();
}

void App::logTestRodEndpointsFrame() {
  if (!testRodEndpointsEnabled || !testRodEndpointsStream)
    return;
  if (!shouldLogThisFrame())
    return;
  if ((frameIndex % testRodEndpointsStride) != 0)
    return;
  if (testRodEndpointsWrittenFrames >= testRodEndpointsMaxFrames)
    return;

  int trackedIdx = (testRodIndex >= 0) ? testRodIndex : settings.scene.fixEveryExcept;
  if (trackedIdx < 0) {
    std::cerr << "[test-rod-endpoints] No rod selected. Use --fix-every-except "
                 "or set --test-rod-id. Disabling logger.\n";
    testRodEndpointsEnabled = false;
    return;
  }
  if (trackedIdx >= (int)rods.size()) {
    std::cerr << "[test-rod-endpoints] Rod index " << trackedIdx
              << " out of range (num_rods=" << rods.size()
              << "). Disabling logger.\n";
    testRodEndpointsEnabled = false;
    return;
  }

  if (!testRodEndpointsHeaderWritten) {
    testRodEndpointsStream << "frame,time,rod,x0,y0,z0,x1,y1,z1\n";
    testRodEndpointsHeaderWritten = true;
  }

  const auto &rb = rods[trackedIdx];
  glm::vec3 a = rb.x;
  glm::vec3 b = rb.x;
  if (rb.type == ShapeType::Capsule) {
    rb.capsuleEndpoints(a, b);
  }
  const double simTime = double(frameIndex) * double(settings.physics.dt);
  testRodEndpointsStream << frameIndex << ',' << simTime << ',' << trackedIdx
                         << ',' << a.x << ',' << a.y << ',' << a.z << ','
                         << b.x << ',' << b.y << ',' << b.z << '\n';
  ++testRodEndpointsWrittenFrames;
  if ((frameIndex & 0x3F) == 0)
    testRodEndpointsStream.flush();
}

// Center-of-mass computation with PBC handling
glm::vec3 App::computeCOM() const {
  if (rods.empty())
    return glm::vec3(0);

  double totalMass = 0.0;
  glm::dvec3 com(0.0);

  if (usePBC) {
    // Use circular/ring method for COM in periodic box
    // Map coordinates to angles on a circle, average using complex
    // numbers This correctly handles particles spread throughout the
    // periodic domain

    const glm::vec3 boxSize = pbcMax - pbcMin;
    const double pi = 3.14159265358979323846;

    // For each dimension, compute angle-averaged position
    for (int k = 0; k < 3; ++k) {
      if (boxSize[k] <= 0.0f) {
        // Non-periodic dimension
        double sum = 0.0;
        totalMass = 0.0;
        for (const auto &rb : rods) {
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

        for (const auto &rb : rods) {
          double m = double(rb.mass);
          totalMass += m;

          // Normalize position to [0, 1] within box
          double normalized =
              (double(rb.x[k]) - double(pbcMin[k])) / double(boxSize[k]);
          double theta = 2.0 * pi * normalized;

          cos_sum += m * std::cos(theta);
          sin_sum += m * std::sin(theta);
        }

        // Average angle
        double avg_theta = std::atan2(sin_sum / totalMass, cos_sum / totalMass);

        // Ensure avg_theta is in [0, 2*pi)
        if (avg_theta < 0.0)
          avg_theta += 2.0 * pi;

        // Convert back to position
        com[k] =
            double(pbcMin[k]) + (avg_theta / (2.0 * pi)) * double(boxSize[k]);
      }
    }

    return glm::vec3(com);

  } else {
    // Simple COM for non-periodic case
    for (const auto &rb : rods) {
      double m = double(rb.mass);
      totalMass += m;
      com += m * glm::dvec3(rb.x);
    }
    return glm::vec3(com / totalMass);
  }
}

void App::logCOMFrame() {
  if (!comEnabled || !comStream)
    return;
  if (!comHeaderWritten) {
    comStream << "frame,com_x,com_y,com_z,total_mass,num_rods\n";
    comHeaderWritten = true;
    std::cerr << "[COM] Header written\n";
  }

  glm::vec3 com = computeCOM();
  double totalMass = 0.0;
  for (const auto &rb : rods)
    totalMass += double(rb.mass);

  comStream << frameIndex << ',' << com.x << ',' << com.y << ',' << com.z << ','
            << totalMass << ',' << rods.size() << '\n';

  if ((frameIndex & 0x3F) == 0)
    comStream.flush();
}

void App::logRelDispFrame() {
  if (!relDispEnabled || !relDispStream)
    return;
  if (!shouldLogThisFrame())
    return;
  if (!relDispHeaderWritten) {
    relDispStream << "frame,rod,dx,dy,dz,l2\n";
    relDispHeaderWritten = true;
  }
  glm::vec3 rc = computeCOM();
  glm::vec3 boxSize = pbcMax - pbcMin;
  for (size_t i = 0; i < rods.size(); ++i) {
    glm::vec3 d = rods[i].x - rc;
    // Fix: Apply minimum image convention for PBC distance
    if (usePBC) {
      for (int k = 0; k < 3; ++k) {
        if (boxSize[k] > 0.0f) {
          d[k] -= boxSize[k] * std::floor(d[k] / boxSize[k] + 0.5f);
        }
      }
    }
    double l2 = glm::dot(d, d);
    relDispStream << frameIndex << ',' << i << ',' << d.x << ',' << d.y << ','
                  << d.z << ',' << l2 << '\n';
  }
  if ((frameIndex & 0x3F) == 0)
    relDispStream.flush();
}

void App::logNetworkFrame() {
  if (!networkEnabled || !networkStream) {
    return;
  }
  // Use user-defined stride (shared with CSV stride or frame stride)
  if ((frameIndex % networkStride) != 0)
    return;
  if (networkMaxFrames > 0 && networkWrittenFrames >= networkMaxFrames)
    return;

  if (!shouldLogThisFrame())
    return;

  const bool nscMode = settings.physics.nsc.enabled;

  if (!networkHeaderWritten) {
    if (nscMode) {
      // Per-contact impulses and pre/post-solve normal relative velocities:
      // the data needed to study relative velocity across collisions.
      networkStream
          << "frame,rod_i,rod_j,contact_x,contact_y,contact_z,normal_x,"
             "normal_y,normal_z,phi,lambda_n,lambda_t1,lambda_t2,"
             "vn_pre,vn_post\n";
    } else {
      networkStream
          << "frame,rod_i,rod_j,contact_x,contact_y,contact_z,normal_x,"
             "normal_"
             "y,"
             "normal_z,distance,"
          << "force_a_x,force_a_y,force_a_z,force_b_x,force_b_y,force_b_z,"
          << "friction_a_x,friction_a_y,friction_a_z,friction_b_x,friction_b_"
             "y,"
             "friction_b_z\n";
    }
    networkHeaderWritten = true;
  }

  size_t rowsWrittenThisFrame = 0;

  // Branch priority must match physicsStep: NSC wins over soft contact.
  if (!nscMode && settings.physics.soft_contact.enabled) {
    if (settings.physics.use_mujoco_contact) {
      // MuJoCo soft contacts (no force data available yet)
      const auto &contacts = mjContactSolver.getContacts();
      for (const auto &c : contacts) {
        glm::vec3 midpoint = 0.5f * (c.pA + c.pB);
        networkStream << frameIndex << ',' << c.a << ',' << c.b << ','
                      << midpoint.x << ',' << midpoint.y << ',' << midpoint.z
                      << ',' << c.n.x << ',' << c.n.y << ',' << c.n.z << ','
                      << c.dist << ','
                      << "0,0,0,0,0,0,0,0,0,0,0,0\n"; // Placeholder zeros
                                                      // for forces
        ++rowsWrittenThisFrame;
      }
    } else {
      // Standard soft contacts - with force data
      const auto &contacts = softContactSolver.getContacts();
      for (const auto &c : contacts) {
        glm::vec3 midpoint = 0.5f * (c.point_a + c.point_b);
        networkStream << frameIndex << ',' << c.body_a << ',' << c.body_b << ','
                      << midpoint.x << ',' << midpoint.y << ',' << midpoint.z
                      << ',' << c.normal.x << ',' << c.normal.y << ','
                      << c.normal.z << ',' << c.distance << ',' << c.force_a.x
                      << ',' << c.force_a.y << ',' << c.force_a.z << ','
                      << c.force_b.x << ',' << c.force_b.y << ',' << c.force_b.z
                      << ',' << c.friction_a.x << ',' << c.friction_a.y << ','
                      << c.friction_a.z << ',' << c.friction_b.x << ','
                      << c.friction_b.y << ',' << c.friction_b.z << '\n';
        ++rowsWrittenThisFrame;
      }
    }
  } else if (nscMode) {
    // NSC contacts: manifolds from the last solve carry accumulated
    // impulses (lambda) and the pre-solve normal velocity; the post-solve
    // normal velocity is computed from current body velocities.
    for (const auto &m : nscSolver.getManifolds()) {
      if (m.isWall)
        continue; // Only rod-rod pairs in the network export
      const RigidBody &A = rods[m.body_a];
      const RigidBody &B = rods[m.body_b];
      const glm::vec3 point = A.x + m.r_a;
      const glm::vec3 v_rel_post =
          (B.v + glm::cross(B.w, m.r_b)) - (A.v + glm::cross(A.w, m.r_a));
      const float vn_post = glm::dot(m.normal, v_rel_post);

      networkStream << frameIndex << ',' << m.body_a << ',' << m.body_b << ','
                    << point.x << ',' << point.y << ',' << point.z << ','
                    << m.normal.x << ',' << m.normal.y << ',' << m.normal.z
                    << ',' << m.phi << ',' << m.lambda_n << ',' << m.lambda_t1
                    << ',' << m.lambda_t2 << ',' << m.v_n_pre << ','
                    << vn_post << '\n';
      ++rowsWrittenThisFrame;
    }
  }

  if (networkEmitEmptyFrames && rowsWrittenThisFrame == 0) {
    if (nscMode)
      networkStream << frameIndex << ",-1,-1,0,0,0,0,0,0,0,0,0,0,0,0\n";
    else
      networkStream << frameIndex
                    << ",-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n";
  }

  if ((frameIndex & 0x3F) == 0)
    networkStream.flush();
  networkWrittenFrames++;
}

void App::dumpContactsCSV(const std::vector<Hit> &hits,
                          const char *stageLabel) {
  if (!contactDumpEnabled)
    return;
  // Apply the same skip factor as per-rod logging to reduce file size
  if ((frameIndex % perRodSkip) != 0)
    return;

  if (!contactDumpStream.is_open()) {
    contactDumpStream.open(contactDumpPath, std::ios::out | std::ios::app);
    if (!contactDumpStream) {
      std::cerr << "Failed to open contact dump file: " << contactDumpPath
                << "\n";
      contactDumpEnabled = false;
      return;
    }
  }
  if (!contactDumpHeaderWritten) {
    contactDumpStream << "frame,stage,idx,a,b,px,py,pz,nx,ny,nz,pen,shiftBx,"
                         "shiftBy,shiftBz,vn,vt\n";
    contactDumpHeaderWritten = true;
  }
  size_t idx = 0;
  for (const auto &h : hits) {
    const RigidBody &A = rods[h.a];
    const RigidBody &B = (h.b >= 0) ? rods[h.b] : floorRB;
    glm::vec3 rA = h.c.point - A.x;
    glm::vec3 rB = h.c.point - (B.x + h.c.shiftB);
    glm::vec3 vA = A.v + glm::cross(A.w, rA);
    glm::vec3 vB = B.v + glm::cross(B.w, rB);
    glm::vec3 rel = vB - vA;
    float vn = glm::dot(rel, h.c.normal);
    glm::vec3 t = rel - h.c.normal * vn;
    float vt = glm::length(t);
    contactDumpStream << frameIndex << ',' << stageLabel << ',' << idx++ << ','
                      << h.a << ',' << h.b << ',' << h.c.point.x << ','
                      << h.c.point.y << ',' << h.c.point.z << ','
                      << h.c.normal.x << ',' << h.c.normal.y << ','
                      << h.c.normal.z << ',' << h.c.penetration << ','
                      << h.c.shiftB.x << ',' << h.c.shiftB.y << ','
                      << h.c.shiftB.z << ',' << vn << ',' << vt << '\n';
  }
  if ((frameIndex & 0x3F) == 0)
    contactDumpStream.flush();
}

bool App::ensureEarlyPairContactStream() {
  if (!earlyPairDiagnostics.enabled)
    return false;
  if (earlyPairDiagnostics.contact_output_path.empty())
    return false;
  if (!earlyPairContactStream.is_open()) {
    earlyPairContactStream.open(earlyPairDiagnostics.contact_output_path,
                                std::ios::out | std::ios::trunc);
    if (!earlyPairContactStream) {
      std::cerr << "Failed to open early pair contact CSV file: "
                << earlyPairDiagnostics.contact_output_path << "\n";
      return false;
    }
  }
  if (!earlyPairContactHeaderWritten) {
    earlyPairContactStream
        << "frame,solver,body_a,body_b,point_ax,point_ay,point_az,point_bx,"
           "point_by,point_bz,normal_x,normal_y,normal_z,signed_gap,"
           "surface_limit,distance,v_rel_x,v_rel_y,v_rel_z,v_n,v_t\n";
    earlyPairContactHeaderWritten = true;
  }
  return true;
}

bool App::ensureEarlyPairDistanceStream() {
  if (!earlyPairDiagnostics.enabled)
    return false;
  if (earlyPairDiagnostics.pair_distance_output_path.empty())
    return false;
  if (!earlyPairDistanceStream.is_open()) {
    earlyPairDistanceStream.open(earlyPairDiagnostics.pair_distance_output_path,
                                 std::ios::out | std::ios::trunc);
    if (!earlyPairDistanceStream) {
      std::cerr << "Failed to open early pair distance CSV file: "
                << earlyPairDiagnostics.pair_distance_output_path << "\n";
      return false;
    }
  }
  if (!earlyPairDistanceHeaderWritten) {
    earlyPairDistanceStream
        << "frame,body_a,body_b,signed_gap,distance_metric,surface_limit,"
           "pair_type,point_ax,point_ay,point_az,point_bx,point_by,point_bz,"
           "normal_x,normal_y,normal_z,v_rel_x,v_rel_y,v_rel_z,v_rel_speed,"
           "v_n,v_t\n";
    earlyPairDistanceHeaderWritten = true;
  }
  return true;
}

bool App::ensureEarlyPairVelocitySummaryStream() {
  if (!earlyPairDiagnostics.enabled)
    return false;
  if (earlyPairDiagnostics.pair_velocity_summary_output_path.empty())
    return false;
  if (!earlyPairVelocitySummaryStream.is_open()) {
    earlyPairVelocitySummaryStream.open(
        earlyPairDiagnostics.pair_velocity_summary_output_path,
        std::ios::out | std::ios::trunc);
    if (!earlyPairVelocitySummaryStream) {
      std::cerr << "Failed to open early pair velocity summary CSV file: "
                << earlyPairDiagnostics.pair_velocity_summary_output_path
                << "\n";
      return false;
    }
  }
  if (!earlyPairVelocitySummaryHeaderWritten) {
    earlyPairVelocitySummaryStream
        << "frame,pair_count,mean_signed_gap,mean_distance_metric,"
           "mean_v_rel_x,mean_v_rel_y,mean_v_rel_z,mean_v_rel_speed,"
           "mean_v_n,mean_abs_v_n,mean_v_t\n";
    earlyPairVelocitySummaryHeaderWritten = true;
  }
  return true;
}

void App::logDetectedContactsFrame(
    const std::vector<CommonContactGeometry> &contacts, const char *solver,
    int sampleFrame) {
  if (!shouldSampleEarlyPairDiagnostics(sampleFrame))
    return;
  if (!ensureEarlyPairContactStream())
    return;

  for (const auto &contact : contacts) {
    if (contact.bodyA < 0 || contact.bodyB < 0)
      continue;
    if (contact.bodyA >= (int)rods.size() || contact.bodyB >= (int)rods.size())
      continue;

    const auto &bodyA = rods[contact.bodyA];
    const auto &bodyB = rods[contact.bodyB];
    const auto kinematics = computeContactKinematics(bodyA, bodyB, contact);

    earlyPairContactStream
        << sampleFrame << ',' << solver << ',' << contact.bodyA << ','
        << contact.bodyB << ',' << contact.pointA.x << ',' << contact.pointA.y
        << ',' << contact.pointA.z << ',' << contact.pointB.x << ','
        << contact.pointB.y << ',' << contact.pointB.z << ','
        << contact.normal.x << ',' << contact.normal.y << ','
        << contact.normal.z << ',' << contact.signedGap << ','
        << contact.surfaceLimit << ',' << contact.distance << ','
        << kinematics.vRel.x << ',' << kinematics.vRel.y << ','
        << kinematics.vRel.z << ',' << kinematics.vNormal << ','
        << kinematics.vTangentialMagnitude << '\n';
  }

  if ((sampleFrame & 0x3F) == 0)
    earlyPairContactStream.flush();
}

App::PairGapInfo App::computePairGapInfo(const RigidBody &A,
                                         const RigidBody &B) const {
  auto segseg_dist = [](const glm::vec3 &p0, const glm::vec3 &p1,
                        const glm::vec3 &q0, const glm::vec3 &q1) {
    glm::vec3 u = p1 - p0;
    glm::vec3 v = q1 - q0;
    glm::vec3 w0 = p0 - q0;
    float uu = glm::dot(u, u), vv = glm::dot(v, v), uv = glm::dot(u, v);
    float wu = glm::dot(w0, u), wv = glm::dot(w0, v);
    float D = uu * vv - uv * uv;
    float s, t;
    const float eps = 1e-12f;

    auto fixBound = [](float &x) {
      if (x < 0.0f)
        x = 0.0f;
      else if (x > 1.0f)
        x = 1.0f;
    };

    if (std::abs(D) < eps) {
      s = 0.0f;
      t = (vv > eps) ? (-wv / vv) : 0.0f;
      fixBound(t);
    } else {
      s = (uv * wv - vv * wu) / D;
      fixBound(s);
      t = (s * uv + wv) / (vv > eps ? vv : 1.0f);
      float tUnclamped = t;
      fixBound(t);
      if (std::abs(t - tUnclamped) > 1e-6f) {
        s = (t * uv - wu) / (uu > eps ? uu : 1.0f);
        fixBound(s);
      }
    }
    glm::vec3 d = (w0 + s * u) - t * v;
    return std::sqrt(std::max(0.0f, glm::dot(d, d)));
  };

  PairGapInfo out;
  if (A.type == ShapeType::Sphere && B.type == ShapeType::Sphere) {
    float center = glm::length(B.x - A.x);
    out.distanceMetric = center;
    out.surfaceLimit = A.sphere.r + B.sphere.r;
    out.signedGap = out.distanceMetric - out.surfaceLimit;
    out.pairType = "sphere_sphere";
    return out;
  }

  if (A.type == ShapeType::Capsule && B.type == ShapeType::Capsule) {
    glm::vec3 a0, a1, b0, b1;
    A.capsuleEndpoints(a0, a1);
    B.capsuleEndpoints(b0, b1);
    glm::vec3 centerDelta = B.x - A.x;
    if (usePBC) {
      glm::vec3 boxSize = pbcMax - pbcMin;
      for (int k = 0; k < 3; ++k) {
        float L = boxSize[k];
        if (L > 0.0f)
          centerDelta[k] -= L * std::floor(centerDelta[k] / L + 0.5f);
      }
    }
    glm::vec3 shift = centerDelta - (B.x - A.x);
    glm::vec3 b0Wrapped = b0 + shift;
    glm::vec3 b1Wrapped = b1 + shift;
    out.distanceMetric = segseg_dist(a0, a1, b0Wrapped, b1Wrapped);
    out.surfaceLimit = A.cap.r + B.cap.r;
    out.signedGap = out.distanceMetric - out.surfaceLimit;
    out.pairType = "capsule_capsule";
    return out;
  }

  const RigidBody *sphere = nullptr;
  const RigidBody *capsule = nullptr;
  if (A.type == ShapeType::Sphere && B.type == ShapeType::Capsule) {
    sphere = &A;
    capsule = &B;
  } else if (A.type == ShapeType::Capsule && B.type == ShapeType::Sphere) {
    sphere = &B;
    capsule = &A;
  }
  if (sphere && capsule) {
    glm::vec3 a0, a1;
    capsule->capsuleEndpoints(a0, a1);
    glm::vec3 u = a1 - a0;
    float L2 = glm::dot(u, u);
    float t = L2 > 0 ? glm::dot(sphere->x - a0, u) / L2 : 0.0f;
    t = glm::clamp(t, 0.0f, 1.0f);
    glm::vec3 closest = a0 + t * u;
    out.distanceMetric = glm::length(sphere->x - closest);
    out.surfaceLimit = sphere->sphere.r + capsule->cap.r;
    out.signedGap = out.distanceMetric - out.surfaceLimit;
    out.pairType = "sphere_capsule";
    return out;
  }

  out.distanceMetric = glm::length(B.x - A.x);
  out.surfaceLimit = 0.0;
  out.signedGap = out.distanceMetric;
  out.pairType = "other";
  return out;
}

App::PairKinematicsInfo App::computePairKinematicsInfo(const RigidBody &A,
                                                       const RigidBody &B) const {
  auto minimumImageShift = [&](const glm::vec3 &centerDelta) {
    glm::vec3 adjusted = centerDelta;
    if (usePBC) {
      const glm::vec3 boxSize = pbcMax - pbcMin;
      for (int k = 0; k < 3; ++k) {
        const float L = boxSize[k];
        if (L > 0.0f)
          adjusted[k] -= L * std::floor(adjusted[k] / L + 0.5f);
      }
    }
    return adjusted - centerDelta;
  };

  auto pointVelocity = [](const RigidBody &body, const glm::vec3 &point,
                          const glm::vec3 &bodyOrigin) {
    return body.v + glm::cross(body.w, point - bodyOrigin);
  };

  auto finalize = [&](PairKinematicsInfo &info, const glm::vec3 &bodyOriginA,
                      const glm::vec3 &bodyOriginB) {
    glm::vec3 separation = info.pointB - info.pointA;
    const float eps = 1e-8f;
    float distance = glm::length(separation);
    if (distance > eps) {
      info.normal = separation / distance;
    }
    glm::vec3 vA = pointVelocity(A, info.pointA, bodyOriginA);
    glm::vec3 vB = pointVelocity(B, info.pointB, bodyOriginB);
    info.vRel = vB - vA;
    info.vNormal = glm::dot(info.vRel, info.normal);
    glm::vec3 vT = info.vRel - float(info.vNormal) * info.normal;
    info.vTangentialMagnitude = glm::length(vT);
  };

  auto closestSegmentSegment = [](const glm::vec3 &p0, const glm::vec3 &p1,
                                  const glm::vec3 &q0, const glm::vec3 &q1,
                                  glm::vec3 &cp, glm::vec3 &cq) {
    const glm::vec3 u = p1 - p0;
    const glm::vec3 v = q1 - q0;
    const glm::vec3 w = p0 - q0;
    const float a = glm::dot(u, u);
    const float b = glm::dot(u, v);
    const float c = glm::dot(v, v);
    const float d = glm::dot(u, w);
    const float e = glm::dot(v, w);
    const float D = a * c - b * b;
    float sN, sD = D, tN, tD = D;
    const float eps = 1e-8f;

    if (D < eps) {
      sN = 0.0f;
      sD = 1.0f;
      tN = e;
      tD = c;
    } else {
      sN = (b * e - c * d);
      tN = (a * e - b * d);
      if (sN < 0.0f) {
        sN = 0.0f;
        tN = e;
        tD = c;
      } else if (sN > sD) {
        sN = sD;
        tN = e + b;
        tD = c;
      }
    }

    if (tN < 0.0f) {
      tN = 0.0f;
      if (-d < 0.0f)
        sN = 0.0f;
      else if (-d > a)
        sN = sD;
      else {
        sN = -d;
        sD = a;
      }
    } else if (tN > tD) {
      tN = tD;
      if ((-d + b) < 0.0f)
        sN = 0.0f;
      else if ((-d + b) > a)
        sN = sD;
      else {
        sN = (-d + b);
        sD = a;
      }
    }

    const float sc = (std::abs(sN) < eps ? 0.0f : sN / sD);
    const float tc = (std::abs(tN) < eps ? 0.0f : tN / tD);
    cp = p0 + sc * u;
    cq = q0 + tc * v;
  };

  PairKinematicsInfo out;
  out.gap = computePairGapInfo(A, B);

  if (A.type == ShapeType::Sphere && B.type == ShapeType::Sphere) {
    glm::vec3 shift = minimumImageShift(B.x - A.x);
    glm::vec3 centerB = B.x + shift;
    glm::vec3 delta = centerB - A.x;
    float dist = glm::length(delta);
    if (dist > 1e-8f)
      out.normal = delta / dist;
    out.pointA = A.x + out.normal * A.sphere.r;
    out.pointB = centerB - out.normal * B.sphere.r;
    finalize(out, A.x, centerB);
    return out;
  }

  if (A.type == ShapeType::Capsule && B.type == ShapeType::Capsule) {
    glm::vec3 a0, a1, b0, b1;
    A.capsuleEndpoints(a0, a1);
    B.capsuleEndpoints(b0, b1);
    glm::vec3 shift = minimumImageShift(B.x - A.x);
    glm::vec3 centerB = B.x + shift;
    glm::vec3 b0Wrapped = b0 + shift;
    glm::vec3 b1Wrapped = b1 + shift;
    glm::vec3 axisA, axisB;
    closestSegmentSegment(a0, a1, b0Wrapped, b1Wrapped, axisA, axisB);
    glm::vec3 delta = axisB - axisA;
    float dist = glm::length(delta);
    if (dist > 1e-8f) {
      out.normal = delta / dist;
    } else {
      glm::vec3 fallback = centerB - A.x;
      if (glm::length(fallback) > 1e-8f)
        out.normal = glm::normalize(fallback);
    }
    out.pointA = axisA + out.normal * A.cap.r;
    out.pointB = axisB - out.normal * B.cap.r;
    finalize(out, A.x, centerB);
    return out;
  }

  const RigidBody *sphere = nullptr;
  const RigidBody *capsule = nullptr;
  bool sphereIsA = false;
  if (A.type == ShapeType::Sphere && B.type == ShapeType::Capsule) {
    sphere = &A;
    capsule = &B;
    sphereIsA = true;
  } else if (A.type == ShapeType::Capsule && B.type == ShapeType::Sphere) {
    sphere = &B;
    capsule = &A;
    sphereIsA = false;
  }
  if (sphere && capsule) {
    glm::vec3 c0, c1;
    capsule->capsuleEndpoints(c0, c1);
    glm::vec3 shift = sphereIsA ? minimumImageShift(capsule->x - sphere->x)
                                : minimumImageShift(sphere->x - capsule->x);
    glm::vec3 sphereCenter = sphere->x;
    glm::vec3 capsuleCenter = capsule->x;
    if (sphereIsA) {
      c0 += shift;
      c1 += shift;
      capsuleCenter += shift;
    } else {
      sphereCenter += shift;
    }
    glm::vec3 u = c1 - c0;
    float L2 = glm::dot(u, u);
    float t = L2 > 0.0f ? glm::dot(sphereCenter - c0, u) / L2 : 0.0f;
    t = glm::clamp(t, 0.0f, 1.0f);
    glm::vec3 axisPoint = c0 + t * u;
    glm::vec3 delta = axisPoint - sphereCenter;
    float dist = glm::length(delta);
    if (dist > 1e-8f)
      out.normal = delta / dist;
    if (sphereIsA) {
      out.pointA = sphereCenter + out.normal * sphere->sphere.r;
      out.pointB = axisPoint - out.normal * capsule->cap.r;
      finalize(out, sphereCenter, capsuleCenter);
    } else {
      out.pointA = axisPoint + out.normal * capsule->cap.r;
      out.pointB = sphereCenter - out.normal * sphere->sphere.r;
      finalize(out, capsuleCenter, sphereCenter);
    }
    return out;
  }

  glm::vec3 shift = minimumImageShift(B.x - A.x);
  glm::vec3 centerB = B.x + shift;
  glm::vec3 delta = centerB - A.x;
  float dist = glm::length(delta);
  if (dist > 1e-8f)
    out.normal = delta / dist;
  out.pointA = A.x;
  out.pointB = centerB;
  finalize(out, A.x, centerB);
  return out;
}

void App::logAllPairDistancesFrame(int sampleFrame) {
  if (!shouldSampleEarlyPairDiagnostics(sampleFrame))
    return;
  if (!ensureEarlyPairDistanceStream())
    return;

  const bool writeVelocitySummary = ensureEarlyPairVelocitySummaryStream();
  double sumSignedGap = 0.0;
  double sumDistanceMetric = 0.0;
  glm::dvec3 sumVRel(0.0);
  double sumVRelSpeed = 0.0;
  double sumVNormal = 0.0;
  double sumAbsVNormal = 0.0;
  double sumVTangential = 0.0;
  size_t pairCount = 0;

  const size_t N = rods.size();
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      const auto info = computePairKinematicsInfo(rods[i], rods[j]);
      if (earlyPairDiagnostics.pair_distance_cutoff >= 0.0 &&
          info.gap.signedGap > earlyPairDiagnostics.pair_distance_cutoff) {
        continue;
      }
      const double vRelSpeed = glm::length(info.vRel);
      earlyPairDistanceStream << sampleFrame << ',' << i << ',' << j << ','
                              << info.gap.signedGap << ','
                              << info.gap.distanceMetric << ','
                              << info.gap.surfaceLimit << ','
                              << info.gap.pairType << ',' << info.pointA.x
                              << ',' << info.pointA.y << ',' << info.pointA.z
                              << ',' << info.pointB.x << ',' << info.pointB.y
                              << ',' << info.pointB.z << ',' << info.normal.x
                              << ',' << info.normal.y << ',' << info.normal.z
                              << ',' << info.vRel.x << ',' << info.vRel.y << ','
                              << info.vRel.z << ',' << vRelSpeed << ','
                              << info.vNormal << ','
                              << info.vTangentialMagnitude << '\n';
      ++pairCount;
      sumSignedGap += info.gap.signedGap;
      sumDistanceMetric += info.gap.distanceMetric;
      sumVRel += glm::dvec3(info.vRel);
      sumVRelSpeed += vRelSpeed;
      sumVNormal += info.vNormal;
      sumAbsVNormal += std::abs(info.vNormal);
      sumVTangential += info.vTangentialMagnitude;
    }
  }

  if (writeVelocitySummary && pairCount > 0) {
    const double invPairCount = 1.0 / double(pairCount);
    earlyPairVelocitySummaryStream
        << sampleFrame << ',' << pairCount << ','
        << (sumSignedGap * invPairCount) << ','
        << (sumDistanceMetric * invPairCount) << ','
        << (sumVRel.x * invPairCount) << ',' << (sumVRel.y * invPairCount)
        << ',' << (sumVRel.z * invPairCount) << ','
        << (sumVRelSpeed * invPairCount) << ','
        << (sumVNormal * invPairCount) << ','
        << (sumAbsVNormal * invPairCount) << ','
        << (sumVTangential * invPairCount) << '\n';
  }

  if ((sampleFrame & 0x3F) == 0)
    earlyPairDistanceStream.flush();
  if (writeVelocitySummary && (sampleFrame & 0x3F) == 0)
    earlyPairVelocitySummaryStream.flush();
}

// ---- Simulation ----

void App::logOutputFrame() {
  if (!outputEnabled || !outputStream)
    return;
  if ((frameIndex % outputStride) != 0)
    return;
  if (outputMaxFrames > 0 && outputWrittenFrames >= outputMaxFrames)
    return;

  if (!shouldLogThisFrame())
    return;
  if (!outputHeaderWritten) {
    outputStream << "frame,contacts,KE,max_overlap,gyration_sq,reldisp_sq,ent_"
                    "sum,ent_pairs\n";
    outputHeaderWritten = true;
  }
  double gyr_sq = computeGyrationSq();
  double reldisp_sq = computeRelativeMotionSq();
  double gap = minPairGap();
  double max_overlap = (gap < 0.0 ? -gap : 0.0); // positive overlap depth
  outputStream << frameIndex << ',' << lastHitCount << ',' << lastKE << ','
               << max_overlap << ',' << gyr_sq << ',' << reldisp_sq << ','
               << lastEntanglementSum << ',' << lastEntanglementPairs << '\n';
  if ((frameIndex & 0x3F) == 0)
    outputStream.flush();
  outputWrittenFrames++;
}

void App::computeEntanglement() {
  std::vector<std::array<double, 6>> segs;
  segs.reserve(rods.size());
  // Convert each rod to a segment [a,b] from capsule axis endpoints
  for (const auto &rb : rods) {
    if (rb.type != ShapeType::Capsule)
      continue;
    glm::vec3 a, b;
    rb.capsuleEndpoints(a, b);
    // If PBC enabled, map endpoints into primary box to keep segments
    // short
    if (usePBC) {
      auto wrap = [&](glm::vec3 &p) { wrapPos(p, pbcMin, pbcMax); };
      wrap(a);
      wrap(b);
    }
    segs.push_back({double(a.x), double(a.y), double(a.z), double(b.x),
                    double(b.y), double(b.z)});
  }
#ifdef _OPENMP
  int threads =
      (entanglementThreads > 0 ? entanglementThreads : omp_get_max_threads());
#else
  int threads = (entanglementThreads > 0 ? entanglementThreads : 1);
#endif
  long long pairs = 0;
  double sum_abs = pairwise_abs_linking_sum_with_cutoff(
      segs, entanglementCutoff, &pairs, threads);
  lastEntanglementSum = sum_abs;
  lastEntanglementPairs = pairs;

  // Print to CLI for quick feedback
  if (entanglementEnabled) {
    if (CLI_UNIFIED_PRINT) {
      printCliStatus("[Entanglement]");
    } else {
      std::cout << "[Entanglement] frame=" << frameIndex
                << " pairs=" << lastEntanglementPairs << " sum=" << std::fixed
                << std::setprecision(6) << lastEntanglementSum
                << std::defaultfloat << "\n";
    }
    // flush occasionally
    std::cout.flush();
  }
}

void App::physicsStep() {
  // fprintf(stderr, "[Debug] App::physicsStep start\n");
  // Debug prints removed
#ifdef TRACY_ENABLE
  ZoneScopedN("PhysicsStep");
#endif
  // Reset diagnostic accumulators before this step

  // Apply constant random forces (acceleration)
  if (useConstantRandomAccel && constantForces.size() == rods.size()) {
    for (size_t i = 0; i < rods.size(); ++i) {
      rods[i].f += constantForces[i];
    }
  }

  // Apply random forces if enabled
  if (useRandomForce) {
    // Generate Gaussian noise for each component
    for (auto &rb : rods) {
      // Force: fSigma * N(0,1) per component
      rb.forceRandom =
          glm::vec3(normal_f(genRandomForce), normal_f(genRandomForce),
                    normal_f(genRandomForce)) *
          fSigma;
      rb.f += rb.forceRandom;

      // Torque: tauMag * N(0,1) per component
      if (tauMag > 0.0f) {
        if (rb.type == ShapeType::Capsule) {
          // Apply noise in body frame, zeroing out the Y (long) axis
          // This prevents "hot spinning" while maintaining correct tumbling
          // temperature
          glm::vec3 tauBody(normal_f(genRandomForce), 0.0f,
                            normal_f(genRandomForce));
          tauBody *= tauMag;
          rb.torqueRandom = rb.q * tauBody; // Rotate to world frame
        } else {
          // Isotropic for spheres/others
          rb.torqueRandom =
              glm::vec3(normal_f(genRandomForce), normal_f(genRandomForce),
                        normal_f(genRandomForce)) *
              tauMag;
        }
        rb.tau += rb.torqueRandom;
      } else {
        rb.torqueRandom = glm::vec3(0.0f);
      }
    }
  }

  if (settings.physics.nsc.enabled) {
    // ===== NSC (Hard Contact) Semi-Implicit Euler for Rods =====
    // Follows Chrono ChTimestepperEulerImplicitProjected pattern:
    //   1) Free-flight velocity prediction
    //   2) Detect capsule contacts
    //   3) Velocity PSOR with friction cones
    //   4) Position update
    //   5) Position stabilization

    // 1) Free-flight velocity prediction: v += M⁻¹·f·dt
    {
#ifdef TRACY_ENABLE
      ZoneScopedN("NSC_FreeFlight");
#endif
#pragma omp parallel for schedule(static) if((int)rods.size() > 1000) num_threads(g_thread_limit > 0 ? g_thread_limit : omp_get_max_threads())
      for (int i = 0; i < (int)rods.size(); ++i) {
        if (sleeping[i]) continue;
        auto& rb = rods[i];
        if (rb.invMass > 0) {
          rb.v += (rb.f * rb.invMass + gravity) * dt;
          glm::mat3 Iinv = rb.IworldInv();
          glm::mat3 Iw = rb.R() * rb.I_body * glm::transpose(rb.R());
          glm::vec3 gyro = glm::cross(rb.w, Iw * rb.w);
          rb.w += Iinv * (rb.tau - gyro) * dt;
        }
      }
    }

    keAfterIntegrate = totalKE();

    // 2) Detect capsule-capsule contacts and build manifolds
    {
#ifdef TRACY_ENABLE
      ZoneScopedN("NSC_Detect");
#endif
      ScopedAccum tBroad(profilingEnabled ? &curTimes.broadphase : nullptr);
      nscSolver.detectAndBuildManifolds(rods);
      if (earlyPairDiagnostics.enabled) {
        std::vector<CommonContactGeometry> contacts;
        const auto &detected = nscSolver.getDetectedContacts();
        contacts.reserve(detected.size());
        for (const auto &contact : detected) {
          contacts.push_back(toCommonContactGeometry(contact));
        }
        logDetectedContactsFrame(contacts, "nsc", static_cast<int>(frameIndex + 1));
      }
    }

    // 2b) Cylinder wall contacts
    if (settings.scene.cylinder.enabled) {
      const float cylR = settings.scene.cylinder.radius;
      const glm::vec3 cylAxis = glm::normalize(settings.scene.cylinder.axis);
      std::vector<Contact> cylContacts;
      for (int i = 0; i < (int)rods.size(); ++i) {
        if (sleeping[i] || rods[i].invMass <= 0.0f) continue;
        
        cylContacts.clear();
        collideCapsuleInsideCylinder(rods[i], cylR, cylAxis, cylContacts);
        
        for (const auto& c : cylContacts) {
          ++reptWallHits;
          nscSolver.addWallContact(i, c, rods, rods[i].restitution);
        }
      }
    }

    lastHitCount = nscSolver.getNumContacts();

    // 3) Solve velocity constraints with friction (PSOR)
    {
#ifdef TRACY_ENABLE
      ZoneScopedN("NSC_VelSolve");
#endif
      ScopedAccum tSolve(profilingEnabled ? &curTimes.solve : nullptr);
      nscSolver.setCurrentFrame(static_cast<int>(frameIndex + 1));
      nscSolver.solveVelocities(rods, dt);
    }

    keAfterSolve = totalKE();

    // 4) Update positions + orientations
    {
#ifdef TRACY_ENABLE
      ZoneScopedN("NSC_PosUpdate");
#endif
      ScopedAccum tInteg(profilingEnabled ? &curTimes.integrate : nullptr);
#pragma omp parallel for schedule(static) if((int)rods.size() > 1000) num_threads(g_thread_limit > 0 ? g_thread_limit : omp_get_max_threads())
      for (int i = 0; i < (int)rods.size(); ++i) {
        if (sleeping[i]) continue;
        auto& rb = rods[i];
        rb.x += rb.v * dt;
        glm::quat wq(0.0f, rb.w);
        rb.q += 0.5f * dt * wq * rb.q;
        rb.q = glm::normalize(rb.q);
        // Damping
        rb.v *= (1.0f - g_lin_damp * dt);
        rb.w *= (1.0f - g_ang_damp * dt);
        // Clamp angular velocity
        float wLen = glm::length(rb.w);
        if (wLen > g_w_max) rb.w *= g_w_max / wLen;
      }
    }

    // 4b) Project out omega_parallel (spin about rod axis) inside cylinder
    if (settings.scene.cylinder.enabled) {
      for (int i = 0; i < (int)rods.size(); ++i) {
        if (sleeping[i]) continue;
        auto& rb = rods[i];
        glm::vec3 u = glm::normalize(rb.axisY());
        rb.w -= glm::dot(rb.w, u) * u;
      }
    }

    // 5) Position stabilization (normal-only PSOR)
    {
#ifdef TRACY_ENABLE
      ZoneScopedN("NSC_PosProject");
#endif
      ScopedAccum tPos(profilingEnabled ? &curTimes.posCorrect : nullptr);
      nscSolver.projectPositions(rods);
    }

    keAfterPosCorrect = totalKE();

    // 6) PBC wrapping
    if (usePBC) {
      ScopedAccum tWrap(profilingEnabled ? &curTimes.pbcWrap : nullptr);
      for (auto& rb : rods) {
        for (int ax = 0; ax < 3; ++ax) {
          float span = pbcMax[ax] - pbcMin[ax];
          while (rb.x[ax] < pbcMin[ax]) rb.x[ax] += span;
          while (rb.x[ax] >= pbcMax[ax]) rb.x[ax] -= span;
        }
      }
    }

    keAfterPBCWrap = totalKE();

    logAllPairDistancesFrame(static_cast<int>(frameIndex + 1));

    // 7) Clear forces for next step
    for (auto& rb : rods) {
      rb.f = glm::vec3(0);
      rb.tau = glm::vec3(0);
    }

  } else if (settings.physics.hertz_mindlin.enabled) {
    // ===== Hertz-Mindlin contact model for spheres =====
    // Uses Velocity Verlet with contact forces
    // 1) contacts & forces at time t
    {
      ScopedAccum tBroad(profilingEnabled ? &curTimes.broadphase : nullptr);
      hertzMindlinSolver.detectContacts(rods);
    }
    {
      ScopedAccum tSolve(profilingEnabled ? &curTimes.solve : nullptr);
      hertzMindlinSolver.computeForces(rods, dt);
    }
    lastSoftPotentialEnergy = hertzMindlinSolver.getLastPotentialEnergy();
    lastHitCount = hertzMindlinSolver.getNumContacts();

    // 2) half-step velocities + position/orientation advance
    {
#ifdef TRACY_ENABLE
      ZoneScopedN("IntegrateHalfPos");
#endif
      ScopedAccum tIntegrateHP(profilingEnabled ? &curTimes.integrate
                                                : nullptr);
#pragma omp parallel for schedule(static)
      for (int i = 0; i < (int)rods.size(); ++i) {
        if (!sleeping[i])
          integrateHalfPos(rods[i], gravity, dt);
      }
    }
// Clear forces before recompute at t+dt
#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)rods.size(); ++i) {
      rods[i].f = glm::vec3(0);
      rods[i].tau = glm::vec3(0);
      // Re-apply random forces (constant over the step)
      if (useRandomForce) {
        rods[i].f += rods[i].forceRandom;
        rods[i].tau += rods[i].torqueRandom;
      }
    }

    // 3) contacts & forces at time t+dt (updated positions)
    {
      ScopedAccum tBroad(profilingEnabled ? &curTimes.broadphase : nullptr);
      hertzMindlinSolver.detectContacts(rods);
    }
    {
      ScopedAccum tSolve(profilingEnabled ? &curTimes.solve : nullptr);
      hertzMindlinSolver.computeForces(rods, dt);
    }
    lastSoftPotentialEnergy = hertzMindlinSolver.getLastPotentialEnergy();
    lastHitCount = hertzMindlinSolver.getNumContacts();

    // 4) second half velocity update
    // Random forces are already in rb.f (re-applied after clear)
    {
#ifdef TRACY_ENABLE
      ZoneScopedN("IntegrateSecondHalf");
#endif
      ScopedAccum tIntegrateHV(profilingEnabled ? &curTimes.integrate
                                                : nullptr);
#pragma omp parallel for schedule(static)
      for (int i = 0; i < (int)rods.size(); ++i) {
        if (!sleeping[i]) {
          integrateSecondHalf(rods[i], gravity, dt);
        } else {
          std::cout << "Sleeping rod " << i << " not integrated\n";
        }
      }
    }
    if (useRandomForce) {
      for (auto &rb : rods) {
        glm::vec3 dirF = uniform_dir_s2(genRandomForce);
        float magF = fSigma * normal_f(genRandomForce);
        rb.f += dirF * magF;
        if (tauMag > 0.0f) {
          glm::vec3 dirT = uniform_dir_s2(genRandomForce);
          rb.tau += dirT * tauMag;
        }
      }
    }
    keAfterSolve = totalKE();
  } else if (settings.physics.soft_contact.enabled) {
    // ===== Full Velocity Verlet sequence for soft contacts =====
    // 1) contacts & forces at time t
    if (settings.physics.use_mujoco_contact) {
      {
        ScopedAccum tBroad(profilingEnabled ? &curTimes.broadphase : nullptr);
        mjContactSolver.detectContacts(rods);
        if (earlyPairDiagnostics.enabled) {
          std::vector<CommonContactGeometry> contacts;
          const auto &detected = mjContactSolver.getContacts();
          contacts.reserve(detected.size());
          for (const auto &contact : detected) {
            contacts.push_back(toCommonContactGeometry(contact));
          }
          logDetectedContactsFrame(contacts, "mujoco", static_cast<int>(frameIndex + 1));
        }
      }
      {
        ScopedAccum tSolve(profilingEnabled ? &curTimes.solve : nullptr);
        mjContactSolver.computeForces(rods, dt);
      }
      lastSoftPotentialEnergy =
          mjContactSolver.getLastPotentialEnergy(); // PE at configuration t
    } else {
      {
        ScopedAccum tBroad(profilingEnabled ? &curTimes.broadphase : nullptr);
        softContactSolver.detectContacts(rods);
        if (earlyPairDiagnostics.enabled) {
          std::vector<CommonContactGeometry> contacts;
          const auto &detected = softContactSolver.getContacts();
          contacts.reserve(detected.size());
          for (const auto &contact : detected) {
            contacts.push_back(toCommonContactGeometry(contact));
          }
          logDetectedContactsFrame(contacts, "soft", static_cast<int>(frameIndex + 1));
        }
        if (profilingEnabled) {
          const auto &s = softContactSolver.getStats();
          curTimes.bpCount += s.count_ms;
          curTimes.bpPrefix += s.prefix_ms;
          curTimes.bpFill += s.fill_ms;
          curTimes.bpPairs += s.sort_ms + s.detect_ms;
        }
      }
      {
        ScopedAccum tSolve(profilingEnabled ? &curTimes.solve : nullptr);
        softContactSolver.computeForces(rods, dt, gravity);
      }
      lastSoftPotentialEnergy =
          softContactSolver.getLastPotentialEnergy(); // PE at configuration t
    }
    // Cylinder wall: penalty spring + smooth Coulomb friction (Verlet, phase t)
    if (settings.scene.cylinder.enabled) {
      const float cylR    = settings.scene.cylinder.radius;
      const auto  cylAxis = glm::normalize(settings.scene.cylinder.axis);
      const float k_wall  = settings.physics.soft_contact.k_scaler;
      const float c_wall  = settings.physics.soft_contact.damping;
      const float mu_wall = settings.physics.soft_contact.mu;
      const float nu_wall = std::max((float)settings.physics.soft_contact.nu, 1e-10f);
      for (auto& rb : rods) {
        if (rb.invMass <= 0.0f) continue;
        const float maxD = cylR - rb.cap.r;
        glm::vec3 p_perp = rb.x - glm::dot(rb.x, cylAxis) * cylAxis;
        float d_perp = glm::length(p_perp);
        float pen = d_perp - maxD;
        if (pen <= 0.0f) continue;
        glm::vec3 n = (d_perp > 1e-8f) ? p_perp / d_perp
            : glm::normalize(glm::cross(cylAxis,
                  (std::abs(cylAxis.x) < 0.9f) ? glm::vec3(1, 0, 0)
                                                : glm::vec3(0, 1, 0)));
        float v_n    = glm::dot(rb.v, n);
        float Fn_mag = k_wall * pen + c_wall * std::max(v_n, 0.0f);
        rb.f -= Fn_mag * n;
        glm::vec3 v_t    = rb.v - v_n * n;
        float     v_t_len = glm::length(v_t);
        if (v_t_len > 1e-12f) {
          float smooth = std::tanh(v_t_len / nu_wall);
          rb.f -= (mu_wall * Fn_mag * smooth / v_t_len) * v_t;
        }
      }
    }
    // if (settings.physics.soft_contact.verbose && frameIndex % 200 == 0)
    // {
    //     std::cout << "[Verlet] frame=" << frameIndex << " contacts(t)="
    //     << softContactSolver.getNumContacts() << '\n';
    // }
    // 2) half-step velocities + position/orientation advance
    {
#ifdef TRACY_ENABLE
      ZoneScopedN("IntegrateHalfPos");
#endif
      ScopedAccum tIntegrateHP(profilingEnabled ? &curTimes.integrate
                                                : nullptr);
#pragma omp parallel for schedule(static)
      for (int i = 0; i < (int)rods.size(); ++i) {
        if (!sleeping[i])
          integrateHalfPos(rods[i], gravity, dt);
      }
    }
// Clear forces before recompute at t+dt
#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)rods.size(); ++i) {
      rods[i].f = glm::vec3(0);
      rods[i].tau = glm::vec3(0);
      // Re-apply random forces (constant over the step)
      if (useRandomForce) {
        rods[i].f += rods[i].forceRandom;
        rods[i].tau += rods[i].torqueRandom;
      }
    }
    // 3) contacts & forces at time t+dt (updated positions)
    if (settings.physics.use_mujoco_contact) {
      {
        ScopedAccum tBroad(profilingEnabled ? &curTimes.broadphase : nullptr);
        mjContactSolver.detectContacts(rods);
      }
      {
        ScopedAccum tSolve(profilingEnabled ? &curTimes.solve : nullptr);
        mjContactSolver.computeForces(rods, dt);
      }
      lastSoftPotentialEnergy =
          mjContactSolver.getLastPotentialEnergy(); // overwrite with PE at
                                                    // configuration t+dt
    } else {
      {
        ScopedAccum tBroad(profilingEnabled ? &curTimes.broadphase : nullptr);
        softContactSolver.detectContacts(rods);
        if (profilingEnabled) {
          const auto &s = softContactSolver.getStats();
          curTimes.bpCount += s.count_ms;
          curTimes.bpPrefix += s.prefix_ms;
          curTimes.bpFill += s.fill_ms;
          curTimes.bpPairs += s.sort_ms + s.detect_ms;
        }
      }
      {
        ScopedAccum tSolve(profilingEnabled ? &curTimes.solve : nullptr);
        softContactSolver.computeForces(rods, dt, gravity);
      }
      lastSoftPotentialEnergy =
          softContactSolver.getLastPotentialEnergy(); // overwrite with PE at
                                                      // configuration t+dt
      lastHitCount =
          softContactSolver
              .getNumContacts(); // Update contact count for CSV logging
    }
    // Cylinder wall: penalty spring + smooth Coulomb friction (Verlet, phase t+dt)
    if (settings.scene.cylinder.enabled) {
      const float cylR    = settings.scene.cylinder.radius;
      const auto  cylAxis = glm::normalize(settings.scene.cylinder.axis);
      const float k_wall  = settings.physics.soft_contact.k_scaler;
      const float c_wall  = settings.physics.soft_contact.damping;
      const float mu_wall = settings.physics.soft_contact.mu;
      const float nu_wall = std::max((float)settings.physics.soft_contact.nu, 1e-10f);
      for (auto& rb : rods) {
        if (rb.invMass <= 0.0f) continue;
        const float maxD = cylR - rb.cap.r;
        glm::vec3 p_perp = rb.x - glm::dot(rb.x, cylAxis) * cylAxis;
        float d_perp = glm::length(p_perp);
        float pen = d_perp - maxD;
        if (pen <= 0.0f) continue;
        ++reptWallHits;  // count contact steps (t+dt phase)
        glm::vec3 n = (d_perp > 1e-8f) ? p_perp / d_perp
            : glm::normalize(glm::cross(cylAxis,
                  (std::abs(cylAxis.x) < 0.9f) ? glm::vec3(1, 0, 0)
                                                : glm::vec3(0, 1, 0)));
        float v_n    = glm::dot(rb.v, n);
        float Fn_mag = k_wall * pen + c_wall * std::max(v_n, 0.0f);
        rb.f -= Fn_mag * n;
        glm::vec3 v_t    = rb.v - v_n * n;
        float     v_t_len = glm::length(v_t);
        if (v_t_len > 1e-12f) {
          float smooth = std::tanh(v_t_len / nu_wall);
          rb.f -= (mu_wall * Fn_mag * smooth / v_t_len) * v_t;
        }
      }
    }
    // if (settings.physics.soft_contact.verbose && frameIndex % 200 == 0)
    // {
    //     std::cout << "[Verlet] frame=" << frameIndex << "
    //     contacts(t+dt)="
    //     << softContactSolver.getNumContacts() << '\n';
    // }
    // 4) second half velocity update
    // Random forces are already in rb.f (re-applied after clear)
    {
#ifdef TRACY_ENABLE
      ZoneScopedN("IntegrateSecondHalf");
#endif
      ScopedAccum tIntegrateSH(profilingEnabled ? &curTimes.integrate
                                                : nullptr);
#pragma omp parallel for schedule(static)
      for (int i = 0; i < (int)rods.size(); ++i) {
        if (!sleeping[i])
          integrateSecondHalf(rods[i], gravity, dt);
      }
    }
    // KE after full Verlet integrate
    keAfterIntegrate = totalKE();
    logAllPairDistancesFrame(static_cast<int>(frameIndex + 1));
  }

  // Update sleeping state (after integration, before collision)
  {
#ifdef TRACY_ENABLE
    ZoneScopedN("SleepUpdate");
#endif
    ScopedAccum tSleep(profilingEnabled ? &curTimes.sleepUpdate : nullptr);
    for (size_t i = 0; i < rods.size(); ++i) {
      if (sleeping[i])
        continue;
      float vs = glm::length(rods[i].v);
      float ws = glm::length(rods[i].w);
      if (vs < sleepLinThresh && ws < sleepAngThresh) {
        sleepTimer[i] += dt;
        if (sleepTimer[i] > sleepTimeThresh) {
          sleeping[i] = 1;
          sleepTimer[i] = 0.f;
          rods[i].v = glm::vec3(0);
          rods[i].w = glm::vec3(0);
        }
      } else {
        sleepTimer[i] = 0.f;
      }
    }
  }
  // end sleep update

  // If using Hertz-Mindlin sphere model, skip hard-contact rod pipeline
  // entirely.
  if (settings.physics.hertz_mindlin.enabled) {
    // KE checkpoints already updated in Hertz-Mindlin branch.
    lastKE = totalKE(); // ensure KE reported for Hertz-Mindlin path
    return;
  }

  // Hard collision resolution & impulse solver (skip entirely when using
  // Hertz-Mindlin sphere model)

  // Update adaptive metrics
  lastFrameKEDelta = keAfterPBCWrap - prevFrameKE;
  prevFrameKE = keAfterPBCWrap;

  // After physics step bookkeeping
  lastKE = totalKE();
  // Per-frame logging and entanglement live in the run loops, not here:
  // physicsStep runs once per *substep*, so logging here emits duplicate
  // rows whenever substeps > 1.
}

#ifndef HEADLESS_BUILD
void App::renderFrame() {
#ifdef TRACY_ENABLE
  ZoneScopedN("RenderFrame");
#endif
  int width, height;
  glfwGetFramebufferSize(window, &width, &height);
  glViewport(0, 0, width, height);
  glClearColor(settings.render.bg.r, settings.render.bg.g, settings.render.bg.b,
               1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  ScopedAccum tRender(profilingEnabled ? &curTimes.render : nullptr);
  float aspect = (height > 0) ? float(width) / float(height) : 1.0f;
  glm::mat4 projection =
      glm::perspective(glm::radians(50.0f), aspect, 0.05f, 100.0f);
  glm::mat4 view = cam.view(camTarget);

  RenderUniforms uniforms;
  uniforms.P = projection;
  uniforms.V = view;
  uniforms.eye = cam.eye(camTarget);
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
      {0.90f, 0.30f, 0.30f}, // Red
      {0.30f, 0.90f, 0.90f}, // Cyan
      {1.00f, 0.60f, 0.80f}, // Pink
      {0.60f, 0.40f, 0.20f}, // Brown
      {0.50f, 0.50f, 0.50f}, // Gray
  };
  constexpr int numColors = sizeof(rodColors) / sizeof(rodColors[0]);

  // Identify neighbors if creating a single-rod view
  std::unordered_set<int> neighborIndices;
  if (viewRodIndex != -1) {
    const auto &contacts = softContactSolver.getContacts();
    for (const auto &c : contacts) {
      if (c.body_a == viewRodIndex)
        neighborIndices.insert(c.body_b);
      else if (c.body_b == viewRodIndex)
        neighborIndices.insert(c.body_a);
    }
  }

  // Draw rods: use instancing beyond a small threshold to reduce draw
  // calls
  const size_t N = rods.size();
  const size_t INST_THRESHOLD = 64;
  if (N > INST_THRESHOLD) {
    // Separate bodies by shape type for instanced rendering
    std::vector<glm::mat4> capsuleModels, sphereModels;
    std::vector<glm::vec4> capsuleColors, sphereColors;

    for (size_t i = 0; i < N; ++i) {
      bool isTarget = ((int)i == viewRodIndex);
      bool isNeighbor = (viewRodIndex != -1 && neighborIndices.count((int)i));

      if (viewRodIndex != -1 && !isTarget && !isNeighbor)
        continue;

      glm::mat4 model = rods[i].modelMatrix();
      glm::vec3 color3 = rodColors[i % numColors];
      glm::vec4 color(color3, isNeighbor ? 0.3f : 1.0f);
      if (isNeighbor) {
        // Force a standard ghost color to distinguish from the target?
        // Or keep original color but transparent. Let's keep original but
        // transparent.
      }

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
      RenderUniforms common = uniforms;
      common.useGrid = false;
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      rnd.drawInstances(cyl, capsuleModels.data(), capsuleColors.data(),
                        capsuleModels.size(), common);
      glDisable(GL_BLEND);
    }

    // Draw spheres
    if (!sphereModels.empty()) {
      RenderUniforms common = uniforms;
      common.useGrid = false;
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      rnd.drawInstances(sphere, sphereModels.data(), sphereColors.data(),
                        sphereModels.size(), common);
      glDisable(GL_BLEND);
    }
  } else {
    for (size_t i = 0; i < N; ++i) {
      bool isTarget = ((int)i == viewRodIndex);
      bool isNeighbor = (viewRodIndex != -1 && neighborIndices.count((int)i));

      if (viewRodIndex != -1 && !isTarget && !isNeighbor)
        continue;

      uniforms.M = rods[i].modelMatrix();
      uniforms.color = rodColors[i % numColors];

      // Handle alpha for non-instanced
      // Assuming RenderUniforms has 'alpha' or we rely on 'color' being vec3
      // and shader using 1.0 Actually RenderUniforms usually takes vec3 color.
      // Checking Renderer logic would be good, but we can try just setting
      // glBlend and passing opaque for now or if uniforms.color is vec3, we
      // can't easily pass alpha unless we modify shader/uniforms. Given
      // instancing path uses alpha, we should probably check if uniforms
      // supports it. Looking at line 3221: capsuleColors uses vec4. In line
      // 3279: uniforms.color = vec3. So non-instanced path might not support
      // alpha easily without changing struct. However, we can just skip
      // transparency for non-instanced or just update the loop logic to at
      // least SHOW the neighbors opaque. Let's at least show them.

      uniforms.useGrid = false;

      // Select mesh based on shape type
      const Mesh &mesh = (rods[i].type == ShapeType::Sphere) ? sphere : cyl;
      rnd.draw(mesh, uniforms);
    }
  }

  if (!usePBC && !disableFloorRender) {
    // Draw floor
    uniforms.M = floorRB.modelMatrix();
    uniforms.useGrid = true;
    uniforms.color = {1.0f, 1.0f, 1.0f};
    rnd.draw(cube, uniforms);
  }

  // Draw contact forces
  if (showContactForces) {
    if (!forceFadingEnabled) {
      fadingContacts.clear();
    }
    // 1. Collect new contacts
    if (settings.physics.soft_contact.enabled) {
      const auto &contacts = softContactSolver.getContacts();
      if (!contacts.empty() && (frameIndex % 60 == 0)) {
        std::cout << "[Viz] Active contacts: " << contacts.size()
                  << " | Sample Force: " << glm::length(contacts[0].force_a)
                  << "\n";
      }
      for (const auto &c : contacts) {
        fadingContacts.push_back({c.point_a,
                                  c.point_a + c.force_a * contactForceScale,
                                  forceFadeDuration, c.body_a, c.body_b});
      }
    }
    if (settings.physics.hertz_mindlin.enabled) {
      const auto &contacts = hertzMindlinSolver.getContacts();
      for (const auto &c : contacts) {
        // force_n is the force on body B; show the reaction acting on A at
        // point_a, matching the soft-contact branch above.
        fadingContacts.push_back({c.point_a,
                                  c.point_a - c.force_n * contactForceScale,
                                  forceFadeDuration, c.body_a, c.body_b});
      }
    }

    // 2. Update and prune contacts
    // Use wall clock dt for fading, or simulated dt? User said "like 1 second",
    // probably wall time is better for visual feel. But let's use a fixed small
    // dt for now if we don't have frame time easily accessible here. Ideally
    // pass frame delta time. settings.scene.dt is sim step, might be smaller
    // than frame time. Let's assume 60FPS -> 0.016s per frame roughly for
    // visualization update? Or use ImGui IO delta time if available? No ImGui
    // here. Let's use a rough Estimate: we update frame roughly.
    float dtViz = 1.0f / 60.0f; // Approx

    std::vector<VisualContact> surviving;
    surviving.reserve(fadingContacts.size());
    for (auto &c : fadingContacts) {
      c.timeLeft -= dtViz;
      if (c.timeLeft > 0) {
        surviving.push_back(c);
      }
    }
    fadingContacts = std::move(surviving);

    // 3. Prepare for rendering
    std::vector<glm::mat4> arrowModels;
    std::vector<glm::vec4> arrowColors;
    arrowModels.reserve(fadingContacts.size());
    arrowColors.reserve(fadingContacts.size());

    float arrowRadius = 0.02f; // Fixed thickness for now

    for (const auto &c : fadingContacts) {
      if (viewRodIndex != -1 && c.idxA != viewRodIndex &&
          c.idxB != viewRodIndex)
        continue;
      glm::vec3 d = c.p1 - c.p0;
      float len = glm::length(d);
      if (len < 1e-6f)
        continue;

      glm::vec3 dir = d / len;
      glm::vec3 center = (c.p0 + c.p1) * 0.5f;

      glm::mat4 M(1.0f);
      M = glm::translate(M, center);

      // Rotate from (0,1,0) to dir
      glm::vec3 up(0.0f, 1.0f, 0.0f);
      float dot = glm::dot(up, dir);
      if (std::abs(dot - 1.0f) < 1e-6f) {
        // aligned
      } else if (std::abs(dot + 1.0f) < 1e-6f) {
        M = glm::rotate(M, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
      } else {
        glm::vec3 axis = glm::normalize(glm::cross(up, dir));
        float angle = std::acos(dot);
        M = glm::rotate(M, angle, axis);
      }

      M = glm::scale(M, glm::vec3(arrowRadius, len * 0.5f, arrowRadius));
      arrowModels.push_back(M);

      // Alpha calculation
      float alpha = forceFadingEnabled
                        ? (c.timeLeft / forceFadeDuration)
                        : 1.0f; // Fix transparency when fading off
      arrowColors.push_back(glm::vec4(1.0f, 0.0f, 0.0f, alpha));
    }

    if (!arrowModels.empty()) {
      RenderUniforms arrowUniforms = uniforms;
      arrowUniforms.useGrid = false;
      // Color is per-instance now
      rnd.drawInstances(cyl, arrowModels.data(), arrowColors.data(),
                        arrowModels.size(), arrowUniforms);
    }
  }
}
#endif

void App::stepWithSubsteps() {
  // Determine substeps (adaptive if enabled)
  int baseSub = 1;
  int substeps = baseSub;
  if (adaptiveSubsteps) {
    bool heavyContacts = (lastHitCount >= (size_t)asHitThresh);
    bool keUp = (lastFrameKEDelta > asKEUpThresh);
    bool keDown = (lastFrameKEDelta < asKEDownThresh);
    if (heavyContacts || keUp || keDown)
      substeps = asMax;
    else
      substeps = std::max(asMin, baseSub);
  }
  float frameDt = dt;
  float subDt = frameDt / float(substeps);
  float saveDt = dt;
  const auto stepT0 = std::chrono::high_resolution_clock::now();
  for (int s = 0; s < substeps; ++s) {
    dt = subDt;
    physicsStep();
  }
  lastStepWallMs = std::chrono::duration<double, std::milli>(
                       std::chrono::high_resolution_clock::now() - stepT0)
                       .count();
  dt = saveDt;
}

int App::run() {
#ifdef HEADLESS_BUILD
  // Force headless mode when built without graphics
  headless = true;
  resetScene();
  if (headlessSteps <= 0)
    headlessSteps = 1000; // Default for headless
  if (!gQuiet) std::cout << "Running headless for " << headlessSteps << " steps..."
            << std::endl;

  // Log initial state (Frame 0)
  if (entanglementEnabled)
    computeEntanglement();
  if (perRodEnabled)
    logPerRodFrame();
  if (testRodEndpointsEnabled)
    logTestRodEndpointsFrame();
  logCsvFrame();
  logOutputFrame();
  logNetworkFrame();
  // Apply minimum moment of inertia workaround if requested
  if (g_minMoment > 0.0f) {
    for (auto &r : rods) {
      for (int k = 0; k < 3; ++k) {
        if (r.I_body[k][k] < g_minMoment) {
          r.I_body[k][k] = g_minMoment;
        }
      }
      r.I_body_inv = glm::inverse(r.I_body);
    }
    if (!gQuiet) std::cerr << "[Debug] Applied min-moment " << g_minMoment << " to "
              << rods.size() << " rods.\n";
  }

  if (snapshotEnabled && snapStride > 0 && snapStartFrame == 0)
    writeSnapshotLine();

  for (int step = 0; step < headlessSteps; ++step) {
    if (!paused) {
      stepWithSubsteps();
    }
    // Track reptation sliding accumulators
    reptAccumulate();
    ++frameIndex;
    if (entanglementEnabled && (frameIndex % entanglementEvery == 0)) {
      computeEntanglement();
    }
    if (perRodEnabled)
      logPerRodFrame();
    if (testRodEndpointsEnabled)
      logTestRodEndpointsFrame();
    // CSV logging if enabled
    if (frameIndex % csvStride == 0) {
      logCsvFrame();
    }
    logSoftPEFrame();
    logCOMFrame();
    logOutputFrame();
    logNetworkFrame();
    // Accumulate profiling then reset per-frame timers. Must run AFTER the
    // CSV logging above, which reads curTimes.
    if (profilingEnabled) {
      sumTimes += curTimes;
      curTimes.reset();
    }

    // Snapshot capture (HEADLESS_BUILD)
    if (snapshotEnabled && snapStride > 0 &&
        frameIndex >= (uint64_t)snapStartFrame &&
        ((frameIndex - snapStartFrame) % snapStride) == 0 &&
        snapshotCount < snapFrames) {
      writeSnapshotLine();
    }

    if (gHeadlessProgressEnabled && cliStatusStride > 0 &&
        (step % cliStatusStride) == 0) {
      printCliStatus("[Headless] ");
    }

    if (!paused && stopKEThreshold > 0.0 && (step + 1) >= stopKEMinSteps) {
      const double ke_now = totalKE();
      // rolling-average bookkeeping
      if ((int)stopKEBuffer.size() < stopKEAvgWindow)
        stopKEBuffer.push_back(ke_now);
      else
        stopKEBuffer[stopKEBufIdx % stopKEAvgWindow] = ke_now;
      ++stopKEBufIdx;
      double ke_avg = 0.0;
      for (auto v : stopKEBuffer) ke_avg += v;
      ke_avg /= (double)stopKEBuffer.size();
      if ((int)stopKEBuffer.size() >= stopKEAvgWindow && ke_avg < stopKEThreshold) {
        std::cout << "[Headless] Early stop at step=" << (step + 1)
                  << " frame=" << frameIndex << " KE_avg(" << stopKEAvgWindow
                  << ")=" << std::setprecision(8)
                  << ke_avg << " < " << stopKEThreshold << "\n";
        break;
      }
    }

    if (!paused && stopSlideVelThreshold > 0.0 &&
        (step + 1) >= stopSlideVelMinSteps) {
      int trackedIdx =
          (testRodIndex >= 0) ? testRodIndex : settings.scene.fixEveryExcept;
      if (trackedIdx >= 0 && trackedIdx < (int)rods.size()) {
        const glm::vec3 axis = rods[trackedIdx].axisY();
        const double slide_vel =
            std::abs((double)glm::dot(rods[trackedIdx].v, axis));
        if (slide_vel < stopSlideVelThreshold) {
          std::cout << "[Headless] Early stop at step=" << (step + 1)
                    << " frame=" << frameIndex
                    << " |v.dot(axis)|=" << std::setprecision(8) << slide_vel
                    << " < " << stopSlideVelThreshold << "\n";
          break;
        }
      }
    }
  }
  // Write reptation summary at end of headless run
  writeReptSummary();
  if (csvEnabled)
    csvStream.flush();
  if (perRodEnabled)
    perRodStream.flush();
  if (testRodEndpointsEnabled)
    testRodEndpointsStream.flush();
  // Headless profiling summary (HEADLESS_BUILD)
  if (profilingEnabled && frameIndex > 0) {
    double invF = 1.0 / double(frameIndex);
    std::cout << "[Profile] Frames=" << frameIndex
              << " | integrate=" << sumTimes.integrate * invF << " ms"
              << " | sleep=" << sumTimes.sleepUpdate * invF << " ms"
              << " | broadphase=" << sumTimes.broadphase * invF << " ms"
              << " (count=" << sumTimes.bpCount * invF
              << ", prefix=" << sumTimes.bpPrefix * invF
              << ", fill=" << sumTimes.bpFill * invF
              << ", pairs=" << sumTimes.bpPairs * invF
              << ", long=" << sumTimes.bpLongLong * invF << ")"
              << " | warmstart=" << sumTimes.warmstart * invF << " ms"
              << " | islands=" << sumTimes.buildIslands * invF << " ms"
              << " | solve=" << sumTimes.solve * invF << " ms"
              << " | floorSolve=" << sumTimes.floorSolve * invF << " ms"
              << " | posCorrect=" << sumTimes.posCorrect * invF << " ms"
              << " | pbcWrap=" << sumTimes.pbcWrap * invF << " ms"
              << "\n";
  }
  std::cout << "Headless run complete. Frames=" << frameIndex << "\n";
  return 0;
#else
  if (headless) {
    // Headless: don't initialize window/graphics, run tight physics loop
    resetScene();
#ifdef _OPENMP
  if (!gQuiet) logOpenMpStartupConfig();
#else
    if (!gQuiet) std::cout << "[Info] OpenMP NOT enabled.\n";
#endif
    if (headlessSteps <= 0)
      headlessSteps = 1000; // Default for headless
    if (!gQuiet) std::cout << "Running headless for " << headlessSteps << " steps...\n";

    // Log initial state (Frame 0)
    if (entanglementEnabled)
      computeEntanglement();
    if (perRodEnabled)
      logPerRodFrame();
    if (testRodEndpointsEnabled)
      logTestRodEndpointsFrame();
    logCsvFrame();
    logOutputFrame();
    logNetworkFrame();
    // Apply minimum moment of inertia workaround if requested
    if (g_minMoment > 0.0f) {
      for (auto &r : rods) {
        for (int k = 0; k < 3; ++k) {
          if (r.I_body[k][k] < g_minMoment) {
            r.I_body[k][k] = g_minMoment;
          }
        }
        r.I_body_inv = glm::inverse(r.I_body);
      }
    }
    if (snapshotEnabled && snapStride > 0 && snapStartFrame == 0)
      writeSnapshotLine();

    for (int step = 0; step < headlessSteps; ++step) {
      if (!paused) {
        stepWithSubsteps();
        ++frameIndex; // Increment frame index *after* step, so next log is
                      // frame N+1
      }
      reptAccumulate();
      if (entanglementEnabled && (frameIndex % entanglementEvery == 0)) {
        computeEntanglement();
      }
      if (perRodEnabled)
        logPerRodFrame();
      if (testRodEndpointsEnabled)
        logTestRodEndpointsFrame();
      logSoftPEFrame();
      logCOMFrame();
      // CSV logging if enabled
      if (frameIndex % csvStride == 0) {
        logCsvFrame();
        logOutputFrame();
        logNetworkFrame();
      }
      // Snapshot capture (runtime headless path)
      if (snapshotEnabled && snapStride > 0 &&
          frameIndex >= (uint64_t)snapStartFrame &&
          ((frameIndex - snapStartFrame) % snapStride) == 0 &&
          snapshotCount < snapFrames) {
        writeSnapshotLine();
      }
      if (profilingEnabled) {
        sumTimes += curTimes;
        curTimes.reset();
      }
      if (gHeadlessProgressEnabled && cliStatusStride > 0 &&
          (step % cliStatusStride) == 0) {
        printCliStatus("[Headless] ");
      }

      if (!paused && stopKEThreshold > 0.0 && (step + 1) >= stopKEMinSteps) {
        const double ke_now = totalKE();
        if ((int)stopKEBuffer.size() < stopKEAvgWindow)
          stopKEBuffer.push_back(ke_now);
        else
          stopKEBuffer[stopKEBufIdx % stopKEAvgWindow] = ke_now;
        ++stopKEBufIdx;
        double ke_avg = 0.0;
        for (auto v : stopKEBuffer) ke_avg += v;
        ke_avg /= (double)stopKEBuffer.size();
        if ((int)stopKEBuffer.size() >= stopKEAvgWindow && ke_avg < stopKEThreshold) {
          std::cout << "[Headless] Early stop at step=" << (step + 1)
                    << " frame=" << frameIndex << " KE_avg(" << stopKEAvgWindow
                    << ")=" << std::setprecision(8)
                    << ke_avg << " < " << stopKEThreshold << "\n";
          break;
        }
      }

      if (!paused && stopSlideVelThreshold > 0.0 &&
          (step + 1) >= stopSlideVelMinSteps) {
        int trackedIdx =
            (testRodIndex >= 0) ? testRodIndex : settings.scene.fixEveryExcept;
        if (trackedIdx >= 0 && trackedIdx < (int)rods.size()) {
          const glm::vec3 axis = rods[trackedIdx].axisY();
          const double slide_vel =
              std::abs((double)glm::dot(rods[trackedIdx].v, axis));
          if (slide_vel < stopSlideVelThreshold) {
            std::cout << "[Headless] Early stop at step=" << (step + 1)
                      << " frame=" << frameIndex
                      << " |v.dot(axis)|=" << std::setprecision(8) << slide_vel
                      << " < " << stopSlideVelThreshold << "\n";
            break;
          }
        }
      }
    }
    // ensure CSV flushed and closed
    writeReptSummary();
    if (csvEnabled)
      csvStream.flush();
    if (perRodEnabled)
      perRodStream.flush();
    if (testRodEndpointsEnabled)
      testRodEndpointsStream.flush();
    if (softPEEnabled)
      softPEStream.flush();
    if (comEnabled)
      comStream.flush();
    if (networkEnabled)
      networkStream.flush();
    if (profilingEnabled && frameIndex > 0) {
      double invF = 1.0 / double(frameIndex);
      std::cout << "[Profile] Frames=" << frameIndex
                << " | integrate=" << sumTimes.integrate * invF << " ms"
                << " | sleep=" << sumTimes.sleepUpdate * invF << " ms"
                << " | broadphase=" << sumTimes.broadphase * invF << " ms"
                << " (count=" << sumTimes.bpCount * invF
                << ", prefix=" << sumTimes.bpPrefix * invF
                << ", fill=" << sumTimes.bpFill * invF
                << ", pairs=" << sumTimes.bpPairs * invF
                << ", long=" << sumTimes.bpLongLong * invF << ")"
                << " | warmstart=" << sumTimes.warmstart * invF << " ms"
                << " | islands=" << sumTimes.buildIslands * invF << " ms"
                << " | solve=" << (sumTimes.solve + sumTimes.floorSolve) * invF
                << " ms"
                << " | posCorrect=" << sumTimes.posCorrect * invF << " ms"
                << " | pbcWrap=" << sumTimes.pbcWrap * invF << " ms"
                << "\n";
    }
    std::cout << "Headless run complete. Frames=" << frameIndex << "\n";
    return 0;
  }

  if (!initWindow())
    return -1;
  if (!initGraphics())
    return -1;
  resetScene();

  // In headful mode, headlessSteps acts as a limit if set (>0) via --steps
  // If not set (or default 1000 from init, unless we change default), verify
  // behavior. The default init was 1000. We should probably set it to -1 by
  // default in main variable declaration but since we are patching, let's
  // assume valid 'limit' only if explicitly requested or we handle it.
  // Actually, checking how it was initialized: "int headlessSteps = 1000;"
  // We should change that initialization to -1 to distinguish "not set" from
  // "1000". But for now let's assume if the user passed --steps it's what they
  // want. If they didn't, it's 1000. Wait, that breaks infinite run. logic: if
  // headlessSteps == 1000 (default) and NOT headless, we want infinite?
  // Impossible to know if user typed --steps 1000 or it's default.
  // I must change the variable initialization logic too.
  // For this chunk, I will just add the run loop variables.

  bool limitSteps = (headlessSteps > 0);

  lastTitleUpdate = std::chrono::high_resolution_clock::now();

  auto lastTime = std::chrono::high_resolution_clock::now();
  double accumulator = 0.0;
#ifdef TRACY_ENABLE
  tracy::SetThreadName("Main");
#endif

  // Log initial state (Frame 0) - only if we haven't run headless before (check
  // logic if needed, but safe to log frame 0)
  if (frameIndex == 0) {
    if (entanglementEnabled)
      computeEntanglement();
    if (perRodEnabled)
      logPerRodFrame();
    logCsvFrame();
    logOutputFrame();
    logNetworkFrame();
  }

  while (!glfwWindowShouldClose(window)) {
    if (renderStride > 1) {
      // Fast-forward mode: Run 'renderStride' physics steps, then render
      // once. This decouples simulation speed from wall-clock time and
      // aligns 'frameIndex' with physics steps.
      for (int i = 0; i < renderStride; ++i) {
        if (!paused || stepSingle) {
          stepWithSubsteps();
          stepSingle = false; // consume single step
          ++frameIndex; // Increment before logging so first step is Frame 1

          if (entanglementEnabled && (frameIndex % entanglementEvery == 0)) {
            computeEntanglement();
          }
          if (perRodEnabled)
            logPerRodFrame();
          if (frameIndex % csvStride == 0) {
            logCsvFrame();
          }
          logSoftPEFrame();
          logCOMFrame();
          logOutputFrame();
          logNetworkFrame();

          if (headlessSteps > 0 && frameIndex >= (uint64_t)headlessSteps) {
            std::cout << "Reached step limit (" << headlessSteps
                      << "). Exiting.\n";
            glfwSetWindowShouldClose(window, GL_TRUE);
          }
        } else {
          // If paused, we still need to break the loop or handle UI,
          // but here we just sleep/idle effectively by doing nothing in
          // the loop and rendering once. To avoid spinning 100% CPU while
          // paused in this loop, we might want to just render once per
          // outer loop.
          break;
        }
      }
      renderFrame();
      maybeUpdateWindowTitle();
      glfwSwapBuffers(window);
      glfwPollEvents();
    } else {
      auto currentTime = std::chrono::high_resolution_clock::now();
      double deltaTime =
          std::chrono::duration<double>(currentTime - lastTime).count();
      lastTime = currentTime;

      // Limit frame time to prevent spiral of death
      accumulator = std::min(accumulator + deltaTime, 1.0 / 15.0);

      while (accumulator >= dt) {
        if (!paused || stepSingle) {
          stepWithSubsteps();
          stepSingle = false; // consume single step
          ++frameIndex;

          // Log PER PHYSICS STEP to match headless
          if (entanglementEnabled && (frameIndex % entanglementEvery == 0)) {
            computeEntanglement();
          }
          if (perRodEnabled)
            logPerRodFrame();
          if (frameIndex % csvStride == 0) {
            logCsvFrame();
          }
          logSoftPEFrame();
          logCOMFrame();
          logOutputFrame();
          logNetworkFrame();

          if (headlessSteps > 0 && frameIndex >= (uint64_t)headlessSteps) {
            std::cout << "Reached step limit (" << headlessSteps
                      << "). Exiting.\n";
            glfwSetWindowShouldClose(window, GL_TRUE);
          }
          accumulator -= dt;
        }

        renderFrame();
        // Logging moved to physics loop for consistency
        // if (perRodEnabled)
        //   logPerRodFrame();
        // CSV logging uses current per-frame times before they are reset by
        // maybeUpdateWindowTitle
        // logCsvFrame();
        // logOutputFrame();
        maybeUpdateWindowTitle();
        glfwSwapBuffers(window);
        glfwPollEvents();
        // ++frameIndex; // incremented in physics loop
      }
#ifdef TRACY_ENABLE
      FrameMark;
#endif
    }
  }
  if (csvEnabled)
    csvStream.flush();
  if (perRodEnabled)
    perRodStream.flush();
  if (softPEEnabled)
    softPEStream.flush();
  if (comEnabled)
    comStream.flush();
  if (networkEnabled)
    networkStream.flush();
  // Destroy GL resources while the context is still alive, before
  // glfwTerminate(). Shader::~Shader() calls glDeleteProgram(), which would
  // segfault on a dead context if we let it run after glfwTerminate().
  cube.destroy();
  cyl.destroy();
  sphere.destroy();
  if (rnd.instanceVBO) {
    glDeleteBuffers(1, &rnd.instanceVBO);
    rnd.instanceVBO = 0;
  }
  if (rnd.shader.prog) { glDeleteProgram(rnd.shader.prog); rnd.shader.prog = 0; }
  if (rnd.instanced.prog) { glDeleteProgram(rnd.instanced.prog); rnd.instanced.prog = 0; }
  glfwDestroyWindow(window);
  window = nullptr;
  glfwTerminate();
  return 0;
#endif
}

#ifndef HEADLESS_BUILD
void App::loadPlaybackFrame(int frameIndex) {
  if (frameIndex < 0 || frameIndex >= totalPlaybackFrames) {
    std::cerr << "[Playback] Invalid frame index: " << frameIndex << "\n";
    return;
  }

  const std::string &rawLine = playbackFrameData[frameIndex];
  nlohmann::json j = nlohmann::json::parse(rawLine, nullptr, false);
  if (j.is_discarded()) {
    std::cerr << "[Playback] JSON parse error frame=" << frameIndex << "\n";
    return;
  }

  rods.clear();
  if (j.contains("bodies") && j["bodies"].is_array()) {
    for (const auto &jb : j["bodies"]) {
      std::string shape = jb.value("shape", "sphere");
      if (shape == "sphere") {
        auto pos = jb["pos"];
        float r = jb.value("radius", 0.05f);
        float density = 1000.0f;
        RigidBody rb = RigidBody::makeSphere(glm::vec3(pos[0], pos[1], pos[2]),
                                             density, r, 0.3f, 0.3f);
        rods.push_back(rb);
      } else if (shape == "capsule") {
        auto pos = jb["pos"];
        auto quat = jb["quat"];
        float r = jb.value("radius", 0.05f);
        float h = jb.value("halfHeight", 0.1f);
        float density = 1000.0f;
        glm::quat q(quat[0], quat[1], quat[2], quat[3]);
        RigidBody rb = RigidBody::makeCapsule(glm::vec3(pos[0], pos[1], pos[2]),
                                              q, density, r, h, 0.3f, 0.3f);
        rods.push_back(rb);
      } else if (shape == "box") {
        auto pos = jb["pos"];
        auto quat = jb["quat"];
        float hx = jb.value("hx", 0.1f);
        float hy = jb.value("hy", 0.1f);
        float hz = jb.value("hz", 0.1f);
        glm::quat q(quat[0], quat[1], quat[2], quat[3]);
        RigidBody rb = RigidBody::makeStaticFloor(
            glm::vec3(pos[0], pos[1], pos[2]), q, hx, hy, hz, 0.3f, 0.3f);
        rods.push_back(rb);
      }
    }
  }

  // Update contacts for visualization
  fadingContacts.clear();
  auto it = playbackContacts.find(frameIndex);
  if (it != playbackContacts.end()) {
    fadingContacts = it->second;
  }

  // Update window title to show current frame
#ifndef HEADLESS_BUILD
  if (window) {
    std::ostringstream title;
    title << "Playback: Frame " << frameIndex << " / " << totalPlaybackFrames;
    glfwSetWindowTitle(window, title.str().c_str());
  }
#endif
}

int App::runPlayback(const std::string &ndjsonPath, const std::string &dumpDir,
                     int playbackFps, bool orbit, float orbitSpeed,
                     bool camPosSet, const glm::vec3 &camPos, bool camTargetSet,
                     const glm::vec3 &camTarget, bool autoFrame, float scale,
                     float camScale, bool skipDupes, bool hideWindow,
                     bool noFloor, int exportStride,
                     const std::string &moviePath, bool autoExit) {
  this->autoExitAfterPlayback = autoExit;
  // Initialize minimal window/renderer
  if (!initWindow())
    return -1;
  if (!initGraphics())
    return -1;
  if (hideWindow) {
    glfwHideWindow(window); // Hide the playback window for frames-only mode
  }

  // Default to disabling floor render in playback as the app's floorRB is
  // uninitialized (default cube at origin) unless the scene/playback file
  // explicitly provides environment geometry.
  disableFloorRender = true;

  if (noFloor) {
    disableFloorRender = true; // suppress floor rendering for playback clarity
  }

  // Store contacts for visualization [frame -> contacts]
  std::unordered_map<int, std::vector<VisualContact>> playbackContacts;
  // Camera overrides
  if (camTargetSet) {
    this->camTarget = camTarget;
  }
  if (camPosSet) {
    glm::vec3 rel = camPos - this->camTarget;
    cam.dist = glm::length(rel);
    if (cam.dist < 1e-6f)
      cam.dist = 5.0f;
    glm::vec3 dir = glm::normalize(rel);
    cam.yaw = std::atan2(-dir.z, dir.x);
    cam.pitch = std::asin(glm::clamp(dir.y, -1.0f, 1.0f));
  }
  // Load all lines from NDJSON or CSV
  std::vector<std::string> lines;

  if ((ndjsonPath.size() > 4 &&
       ndjsonPath.substr(ndjsonPath.size() - 4) == ".csv") ||
      (ndjsonPath.size() > 4 &&
       ndjsonPath.substr(ndjsonPath.size() - 4) == ".txt")) {
    std::ifstream fin(ndjsonPath);
    if (!fin) {
      std::cerr << "[playback] Failed to open CSV file: " << ndjsonPath << "\n";
      return -1;
    }
    std::string line;
    // Determine default rod shape/dims from settings
    std::string defaultShape = "capsule";
    float defaultRadius = 0.05f;
    float defaultHalfHeight = 0.5f;

    if (!settings.scene.bodies.empty()) {
      const auto &b = settings.scene.bodies[0];
      defaultShape = b.shape;
      defaultRadius = (b.shape == "capsule") ? b.diameter * 0.5f : b.radius;
      defaultHalfHeight = b.length * 0.5f;
    } else if (settings.scene.populate.count > 0) {
      const auto &p = settings.scene.populate;
      defaultShape = p.shape;
      defaultRadius = p.radius;
      if (p.shape == "sphere")
        defaultRadius = p.radius;
    }

    // Skip metadata and parse overrides
    while (std::getline(fin, line)) {
      if (line.empty())
        continue;
      if (line[0] == '#') {
        auto eq = line.find('=');
        if (eq != std::string::npos) {
          std::string key = line.substr(1, eq - 1);
          std::string val = line.substr(eq + 1);
          // trim whitespace from key
          key.erase(0, key.find_first_not_of(" \t"));
          key.erase(key.find_last_not_of(" \t") + 1);

          if (key == "rod_radius")
            defaultRadius = std::stof(val);
          else if (key == "rod_length")
            defaultHalfHeight = std::stof(val) * 0.5f;
        }
        continue;
      }
      if (line.find("frame") != std::string::npos) {
        continue; // skip header
      }
      // If we got here, it's data
      break;
    }

    // Check format by counting columns in first data line
    // - 6 columns: initial configuration (static snapshot)
    // - 8 columns: endpoints format (frame,id,x1,y1,z1,x2,y2,z2)
    // - 15+ columns: perrod format
    // (frame,rod,px,py,pz,vx,vy,vz,wx,wy,wz,qw,qx,qy,qz,...)
    bool isInitialConfig = false;
    bool isEndpointsFormat = false;
    if (!line.empty()) {
      std::stringstream ss(line);
      std::string segment;
      int colCount = 0;
      char sep = (line.find(',') != std::string::npos) ? ',' : ' ';
      while (std::getline(ss, segment, sep)) {
        if (!segment.empty())
          colCount++;
      }
      if (colCount >= 6 && colCount < 8) {
        isInitialConfig = true;
      } else if (colCount == 8) {
        isEndpointsFormat = true;
        std::cout << "[playback] Detected endpoints format (8 columns)\n";
      }
    }

    if (isInitialConfig) {
      std::cout << "[playback] Detected initial configuration CSV format. "
                   "loading as static snapshot.\n";
      fin.close();
      if (this->loadInitialConfigCSV(ndjsonPath)) {
        nlohmann::json frameJson;
        std::vector<nlohmann::json> bodyList;
        for (const auto &rb : rods) {
          nlohmann::json b;
          b["pos"] = {rb.x.x, rb.x.y, rb.x.z};
          b["quat"] = {rb.q.w, rb.q.x, rb.q.y, rb.q.z};
          if (rb.type == ShapeType::Capsule) {
            b["shape"] = "capsule";
            b["radius"] = rb.cap.r;
            b["halfHeight"] = rb.cap.h;
          } else if (rb.type == ShapeType::Sphere) {
            b["shape"] = "sphere";
            b["radius"] = rb.sphere.r;
          } else if (rb.type == ShapeType::Box) {
            b["shape"] = "box";
            b["halfExtents"] = {rb.box.hx, rb.box.hy, rb.box.hz};
          }
          bodyList.push_back(b);
        }
        frameJson["bodies"] = bodyList;
        lines.push_back(frameJson.dump());
      } else {
        std::cerr << "[playback] Failed to load initial configuration from: "
                  << ndjsonPath << "\n";
        return -1;
      }
    } else if (isEndpointsFormat) {
      // Endpoints format: frame,id,x1,y1,z1,x2,y2,z2
      std::vector<nlohmann::json> currentFrameBodies;
      int currentFrameIdx = -1;

      auto processEndpointsLine = [&](const std::string &l) {
        if (l.empty())
          return;
        std::stringstream ss(l);
        std::string segment;
        std::vector<double> vals;
        vals.reserve(8);
        while (std::getline(ss, segment, ',')) {
          try {
            vals.push_back(std::stod(segment));
          } catch (...) {
            vals.push_back(0.0);
          }
        }
        if (vals.size() < 8)
          return;

        int frame = (int)vals[0];
        // vals[1] is rod id (not needed for rendering)
        double x1 = vals[2], y1 = vals[3], z1 = vals[4];
        double x2 = vals[5], y2 = vals[6], z2 = vals[7];

        if (frame != currentFrameIdx) {
          if (currentFrameIdx != -1 && !currentFrameBodies.empty()) {
            nlohmann::json frameJson;
            frameJson["bodies"] = currentFrameBodies;
            lines.push_back(frameJson.dump());
          }
          currentFrameIdx = frame;
          currentFrameBodies.clear();
        }

        // Calculate center and orientation from endpoints
        double cx = (x1 + x2) * 0.5;
        double cy = (y1 + y2) * 0.5;
        double cz = (z1 + z2) * 0.5;

        double dx = x2 - x1;
        double dy = y2 - y1;
        double dz = z2 - z1;
        double len = std::sqrt(dx * dx + dy * dy + dz * dz);

        if (len < 1e-9) {
          // Degenerate rod, skip
          return;
        }

        // Normalize direction
        dx /= len;
        dy /= len;
        dz /= len;

        // Compute quaternion to rotate (0,1,0) to (dx,dy,dz)
        glm::vec3 up(0, 1, 0);
        glm::vec3 dir(dx, dy, dz);
        glm::quat q;
        float d = glm::dot(up, dir);
        if (d > 0.999999f) {
          q = glm::quat(1, 0, 0, 0);
        } else if (d < -0.999999f) {
          q = glm::quat(0, 0, 0, 1); // 180 deg around Z
        } else {
          glm::vec3 c = glm::cross(up, dir);
          float s = std::sqrt((1.0f + d) * 2.0f);
          float invs = 1.0f / s;
          q = glm::quat(s * 0.5f, c.x * invs, c.y * invs, c.z * invs);
        }
        q = glm::normalize(q);

        nlohmann::json b;
        b["pos"] = {cx, cy, cz};
        b["quat"] = {q.w, q.x, q.y, q.z};
        b["shape"] = "capsule";
        b["radius"] = defaultRadius;
        b["halfHeight"] = defaultHalfHeight;
        currentFrameBodies.push_back(b);
      };

      if (!line.empty() && std::isdigit(line[0])) {
        processEndpointsLine(line);
      }
      while (std::getline(fin, line)) {
        processEndpointsLine(line);
      }
      // Push last frame
      if (!currentFrameBodies.empty()) {
        nlohmann::json frameJson;
        frameJson["bodies"] = currentFrameBodies;
        lines.push_back(frameJson.dump());
      }
    } else {
      std::vector<nlohmann::json> currentFrameBodies;
      int currentFrameIdx = -1;

      auto processLine = [&](const std::string &l) {
        if (l.empty())
          return;
        std::stringstream ss(l);
        std::string segment;
        std::vector<double> vals;
        vals.reserve(18);
        while (std::getline(ss, segment, ',')) {
          try {
            vals.push_back(std::stod(segment));
          } catch (...) {
            vals.push_back(0.0);
          }
        }
        if (vals.size() < 15)
          return;

        int frame = (int)vals[0];
        double px = vals[2], py = vals[3], pz = vals[4];
        double qw = vals[11], qx = vals[12], qy = vals[13], qz = vals[14];

        if (frame != currentFrameIdx) {
          if (currentFrameIdx != -1 && !currentFrameBodies.empty()) {
            nlohmann::json frameJson;
            frameJson["bodies"] = currentFrameBodies;
            lines.push_back(frameJson.dump());
          }
          currentFrameIdx = frame;
          currentFrameBodies.clear();
        }

        nlohmann::json b;
        b["pos"] = {px, py, pz};
        b["quat"] = {qw, qx, qy, qz};
        b["shape"] = defaultShape;
        b["radius"] = defaultRadius;
        b["halfHeight"] = defaultHalfHeight;
        currentFrameBodies.push_back(b);
      };

      if (!line.empty() && std::isdigit(line[0])) {
        processLine(line);
      }
      while (std::getline(fin, line)) {
        processLine(line);
      }
      // Push last frame
      if (!currentFrameBodies.empty()) {
        nlohmann::json frameJson;
        frameJson["bodies"] = currentFrameBodies;
        lines.push_back(frameJson.dump());
      }
    }
    std::cout << "[playback] Loaded " << lines.size() << " frames from CSV.\n";

    // Load network contacts if provided (populate existing map)
    if (!networkPath.empty()) {
      std::ifstream nfin(networkPath);
      if (nfin) {
        std::string nline;
        std::getline(nfin, nline); // skip header
        int nParsed = 0;
        while (std::getline(nfin, nline)) {
          if (nline.empty())
            continue;
          std::stringstream nss(nline);
          std::string segment;
          std::vector<double> vals;
          while (std::getline(nss, segment, ',')) {
            try {
              vals.push_back(std::stod(segment));
            } catch (...) {
              vals.push_back(0.0);
            }
          }
          // Expected cols: frame=0, contact=3,4,5, force_a=10,11,12
          if (vals.size() > 12) {
            int f = (int)vals[0];
            int ia = (int)vals[1];
            int ib = (int)vals[2];
            glm::vec3 c_pos(vals[3], vals[4], vals[5]);
            glm::vec3 force(vals[10], vals[11], vals[12]);
            float forceMag = glm::length(force);
            // Only add significant forces
            if (forceMag > 1e-9f) {
              float vizScale = contactForceScale; // Use existing member
              VisualContact vc;
              vc.p0 = c_pos;
              vc.p1 = c_pos + force * vizScale;
              vc.timeLeft =
                  1000.0f; // Persistent for the frame (cleared next frame)
              vc.idxA = ia;
              vc.idxB = ib;
              playbackContacts[f].push_back(vc);
              nParsed++;
            }
          }
        }
        std::cout << "[playback] Loaded " << nParsed
                  << " contacts from network CSV.\n";
      } else {
        std::cerr << "[playback] Failed to open network CSV: " << networkPath
                  << "\n";
      }
    }

  } else {
    // NDJSON Block
    std::ifstream fin(ndjsonPath);
    if (!fin) {
      std::cerr << "[playback] Failed to open snapshots file: " << ndjsonPath
                << "\n";
      return -1;
    }
    std::string line;
    lines.reserve(1024);
    while (std::getline(fin, line)) {
      if (!line.empty())
        lines.push_back(line);
    }
  }
  if (lines.empty()) {
    std::cerr << "[playback] No frames in file: " << ndjsonPath << "\n";
    return 0;
  }
  // Auto-frame using first snapshot AABB
  if (autoFrame) {
    nlohmann::json j0 = nlohmann::json::parse(lines[0], nullptr, false);
    if (!j0.is_discarded() && j0.contains("bodies")) {
      glm::vec3 bmin(std::numeric_limits<float>::max());
      glm::vec3 bmax(-std::numeric_limits<float>::max());
      for (const auto &jb : j0["bodies"]) {
        if (!jb.contains("pos"))
          continue;
        auto p = jb["pos"];
        glm::vec3 q(p[0], p[1], p[2]);
        bmin = glm::min(bmin, q);
        bmax = glm::max(bmax, q);
      }
      glm::vec3 center = 0.5f * (bmin + bmax);
      this->camTarget = center;
      glm::vec3 diag = bmax - bmin;
      float radius = 0.5f * glm::length(diag);
      float fovY = glm::radians(50.0f);
      float dist =
          (radius > 1e-4f) ? (radius / std::sin(fovY * 0.5f)) * 1.15f : 5.0f;
      cam.dist = dist * camScale;
      // Choose yaw/pitch based on diagonal orientation for slight
      // perspective
      cam.yaw = 0.8f;
      cam.pitch = 0.5f;
      std::cerr << "[playback] Auto-framed camera dist=" << dist
                << " center=" << center.x << "," << center.y << "," << center.z
                << " radius=" << radius << "\n";
    }
  }
  if (scale != 1.0f) {
    std::cerr << "[playback] Scaling frames by factor " << scale << "\n";
  }
  if (skipDupes) {
    std::cerr << "[playback] Duplicate frame skipping enabled\n";
  }
  if (!dumpDir.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(dumpDir, ec);
    if (ec)
      std::cerr << "[playback] Warning: couldn't create dump dir: " << dumpDir
                << " : " << ec.message() << "\n";
  }
  auto lastFrameTime = std::chrono::high_resolution_clock::now();
  std::cerr << "[playback] Frames=" << lines.size()
            << (dumpDir.empty() ? "" : " dumping enabled") << "\n";

  // Store frame data in member variables for interactive navigation
  playbackFrameData = std::move(lines);
  totalPlaybackFrames = static_cast<int>(playbackFrameData.size());
  currentPlaybackFrame = 0;
  inPlaybackMode = true;

  // Load initial frame
  if (totalPlaybackFrames > 0) {
    loadPlaybackFrame(currentPlaybackFrame);
  }

  std::string prevLine;
  size_t exportedCount = 0; // Counter for sequentially numbered export files

  // Enable continuous play by default
  // playbackFps == 0 means run as fast as possible (no frame throttle)
  const bool unlimitedPlayback = (playbackFps == 0);
  if (unlimitedPlayback)
    glfwSwapInterval(0); // disable vsync for maximum speed
  double playbackRate = (playbackFps > 0) ? playbackFps : 30.0;
  const double playbackDt = 1.0 / playbackRate;

  bool autoPlay = true; // Always enable auto-play (can pause with SPACE)
  int lastRenderedFrame = -1;

  // Interactive playback loop
  while (!glfwWindowShouldClose(window)) {
    // Auto-advance frame if in auto-play mode and not paused
    if (autoPlay && !paused && currentPlaybackFrame < totalPlaybackFrames - 1) {
    } else if (autoExitAfterPlayback && !paused &&
               currentPlaybackFrame >= totalPlaybackFrames - 1) {
      // Render one last time then exit
      if (currentPlaybackFrame == lastRenderedFrame) {
        glfwSetWindowShouldClose(window, 1);
      }
    }
    auto now = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(now - lastFrameTime).count();
    // Apply speed multiplier to playback rate
    double adjustedDt = playbackDt / playbackSpeedMultiplier;
    if (unlimitedPlayback || elapsed >= adjustedDt) {
      lastFrameTime = now;
      currentPlaybackFrame++;
      loadPlaybackFrame(currentPlaybackFrame);
    }

    // Only re-render if frame changed or camera moved (orbit)
    if (currentPlaybackFrame != lastRenderedFrame || orbit) {
      lastRenderedFrame = currentPlaybackFrame;

      // Simple orbit: rotate camera eye around center keeping dist
      if (orbit) {
        cam.yaw += orbitSpeed * 0.01f; // incremental yaw shift
      }

      renderFrame();
      // Ensure rendering finished before pixel read
      glFinish();

      if (!dumpDir.empty() && (currentPlaybackFrame % exportStride == 0)) {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        std::vector<unsigned char> pixels(width * height * 4);
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE,
                     pixels.data());
        // Optional scaling (nearest neighbor)
        int outW = width, outH = height;
        std::vector<unsigned char> scaled;
        if (scale != 1.0f) {
          outW = std::max(1, int(width * scale));
          outH = std::max(1, int(height * scale));
          scaled.resize(outW * outH * 4);
          for (int y = 0; y < outH; ++y) {
            int sy = std::min(height - 1, int(y / scale));
            for (int x = 0; x < outW; ++x) {
              int sx = std::min(width - 1, int(x / scale));
              for (int c = 0; c < 4; ++c)
                scaled[(y * outW + x) * 4 + c] =
                    pixels[(sy * width + sx) * 4 + c];
            }
          }
        }
        const unsigned char *srcPixels =
            (scale == 1.0f) ? pixels.data() : scaled.data();
        // Vertical flip
        std::vector<unsigned char> flipped(outW * outH * 4);
        for (int y = 0; y < outH; ++y) {
          int sy = outH - 1 - y;
          std::memcpy(&flipped[y * outW * 4], &srcPixels[sy * outW * 4],
                      outW * 4);
        }
        // Overlay frame index text (simple 5x7 digit font)
        auto putPx = [&](int x, int y, unsigned char r, unsigned char g,
                         unsigned char b) {
          if (x >= 0 && x < outW && y >= 0 && y < outH) {
            unsigned char *p = &flipped[(y * outW + x) * 4];
            p[0] = r;
            p[1] = g;
            p[2] = b;
            p[3] = 255;
          }
        };
        static const unsigned char font[10][7] = {
            {0x3E, 0x51, 0x49, 0x45, 0x3E, 0x00,
             0x00}, // 0 (packed rows 5 bits used)
            {0x00, 0x42, 0x7F, 0x40, 0x00, 0x00, 0x00}, // 1
            {0x42, 0x61, 0x51, 0x49, 0x46, 0x00, 0x00}, // 2
            {0x21, 0x41, 0x45, 0x4B, 0x31, 0x00, 0x00}, // 3
            {0x18, 0x14, 0x12, 0x7F, 0x10, 0x00, 0x00}, // 4
            {0x27, 0x45, 0x45, 0x45, 0x39, 0x00, 0x00}, // 5
            {0x3C, 0x4A, 0x49, 0x49, 0x30, 0x00, 0x00}, // 6
            {0x01, 0x71, 0x09, 0x05, 0x03, 0x00, 0x00}, // 7
            {0x36, 0x49, 0x49, 0x49, 0x36, 0x00, 0x00}, // 8
            {0x06, 0x49, 0x49, 0x29, 0x1E, 0x00, 0x00}  // 9
        };
        auto drawDigit = [&](int d, int ox, int oy) {
          if (d < 0 || d > 9)
            return;
          for (int row = 0; row < 5; ++row) {
            unsigned char bits = font[d][row];
            for (int col = 0; col < 7; ++col) {
              if (bits & (1 << (6 - col))) {
                putPx(ox + col, oy + row, 255, 255, 0);
              }
            }
          }
        };
        if (showLabel) {
          // Render frame number at top-left
          std::string label = std::to_string(currentPlaybackFrame);
          int cursor = 4;
          for (char c : label) {
            drawDigit(c - '0', cursor, 4);
            cursor += 8;
          }
        }
        std::vector<unsigned char> png;
        lodepng::encode(png, flipped.data(), (unsigned)outW, (unsigned)outH);
        char name[256];
        // Use exportedCount for sequential filenames (frame_00000.png,
        // frame_00001.png, ...) so ffmpeg detects a continuous sequence.
        std::snprintf(name, sizeof(name), "%s/frame_%05zu.png", dumpDir.c_str(),
                      exportedCount);
        lodepng::save_file(png, name);
        exportedCount++;
      }
    }

    glfwSwapBuffers(window);
    glfwPollEvents();

    // Exit loop if we're in auto-play mode and reached the end
    if (autoPlay && currentPlaybackFrame >= totalPlaybackFrames - 1 &&
        !dumpDir.empty()) {
      break; // Exit to generate movie if dumping
    }
  }

  // Cleanup and movie generation
  if (!dumpDir.empty()) {
    std::cerr << "[playback] Dump complete: " << dumpDir << "\n";
    if (!moviePath.empty()) {
      std::cerr << "[playback] Generating movie: " << moviePath << " ...\n";
      // Construct ffmpeg command
      // Assuming 60fps or playbackFps if set
      int fps = (playbackFps > 0) ? playbackFps : 60;
      std::stringstream cmd;
      cmd << "ffmpeg -y -framerate " << fps << " -i " << dumpDir
          << "/frame_%05d.png -c:v libx264 -pix_fmt yuv420p " << moviePath;
      std::cout << "[Exec] " << cmd.str() << "\n";
      int ret = std::system(cmd.str().c_str());
      if (ret == 0)
        std::cerr << "[playback] Movie generated successfully.\n";
      else
        std::cerr << "[playback] ffmpeg failed with code " << ret << "\n";
    } else {
      std::cerr << "[playback] Example ffmpeg: ffmpeg -framerate "
                << (playbackFps > 0 ? playbackFps : 60) << " -i " << dumpDir
                << "/frame_%05d.png -c:v libx264 -pix_fmt yuv420p movie.mp4\n";
    }
  }
  return 0;
}
#endif

void App::setConfig(const AppCfg &config) {
  settings = config;
  configureEarlyPairDiagnostics(settings.diagnostics.early_pairs);
  // If scene specifies an initial CSV, configure it here so resetScene
  // will load it.
  if (!settings.scene.initCsvPath.empty()) {
    initCsvPath = settings.scene.initCsvPath;
  }
}


namespace app {

App *createPythonApp(const std::string &scenePath,
                     const std::string &initCsvPath,
                     bool quiet) {
  gQuiet = quiet;

  AppCfg cfg = defaultAppCfg();
  if (!loadConfigFromFile(scenePath, cfg)) {
    throw std::runtime_error("failed to load scene config: " + scenePath);
  }

  App *appInstance = new App();
  appInstance->setConfig(cfg);
  appInstance->setHeadless(true);
  if (!initCsvPath.empty()) {
    appInstance->setInitCsvPath(initCsvPath);
  }
  appInstance->initializePythonSession();
  return appInstance;
}

void destroyPythonApp(App *appInstance) { delete appInstance; }

void stepPythonSession(App *appInstance, int steps) {
  if (!appInstance) {
    throw std::invalid_argument("appInstance must not be null");
  }
  appInstance->stepPythonSession(steps);
}

const std::vector<RigidBody> &pythonRods(const App *appInstance) {
  if (!appInstance) {
    throw std::invalid_argument("appInstance must not be null");
  }
  return appInstance->pythonRods();
}

uint64_t pythonFrameIndex(const App *appInstance) {
  if (!appInstance) {
    throw std::invalid_argument("appInstance must not be null");
  }
  return appInstance->pythonFrameIndex();
}

double pythonLastKE(const App *appInstance) {
  if (!appInstance) {
    throw std::invalid_argument("appInstance must not be null");
  }
  return appInstance->pythonLastKE();
}

size_t pythonLastHitCount(const App *appInstance) {
  if (!appInstance) {
    throw std::invalid_argument("appInstance must not be null");
  }
  return appInstance->pythonLastHitCount();
}

size_t pythonLastIslandCount(const App *appInstance) {
  if (!appInstance) {
    throw std::invalid_argument("appInstance must not be null");
  }
  return appInstance->pythonLastIslandCount();
}

float pythonDt(const App *appInstance) {
  if (!appInstance) {
    throw std::invalid_argument("appInstance must not be null");
  }
  return appInstance->pythonDt();
}

} // namespace app
void App::printCliStatus(const std::string &prefix) const {
  if (!CLI_UNIFIED_PRINT)
    return;
  // frame, bodies, contacts, KE, ent_pairs, ent_sum
  std::cout << prefix << "frame=" << frameIndex << " bodies=" << rods.size()
            << " contacts=" << lastHitCount
            << " KE=" << std::fixed << std::setprecision(6) << lastKE
            << " ent_pairs=" << lastEntanglementPairs
            << " ent_sum=" << std::fixed << std::setprecision(6)
            << lastEntanglementSum;
  if (!rods.empty()) {
    std::cout << " p=" << rods[0].x.x << "," << rods[0].x.y << "," << rods[0].x.z
              << " w=" << rods[0].w.x << "," << rods[0].w.y << "," << rods[0].w.z
              << " ay=" << rods[0].axisY().x << "," << rods[0].axisY().y << "," << rods[0].axisY().z;
  }
  std::cout << std::defaultfloat << "\n";
}

// ---- Main Function ----

#ifndef ROD_DYNAMICS_NO_CLI_MAIN
int main(int argc, char **argv) {
  std::string scenePath = std::string(ASSETS_DIR) + "/scenes/default_entangled.json";
  bool enableProfile = false;
  std::string csvPath;
  bool noCsv = false;
  bool headlessFlag = false;
  int headlessSteps = -1; // Default to -1 (infinite/unset)
  std::string perRodPath;
  int perRodMaxFrames = 1000;
  int cliPerrodStride = -1;
  int cliOutputStride = -1;
  int cliOutputMax = -1;
  int cliNetworkStride = -1;
  int cliNetworkMax = -1;

  // CLI overrides

  int cliSeed = 0; // 0 means no override

  int cliThreads = -1;
  float cliDt = -1.0f;
  double cliStopKEThreshold = -1.0;
  int cliStopKEMinSteps = 0;
  int cliStopKEAvgWindow = 1;
  double cliStopSlideVelThreshold = -1.0;
  int cliStopSlideVelMinSteps = 0;
  std::string cliSoftPEPath;   // optional soft potential energy output file
  std::string cliCOMPath;      // center-of-mass tracking
  std::string cliNetworkPath;  // contact network tracking
  std::string cliOutputPath;   // compact output CSV
  bool cliDebugMinGap = false; // enable minPairGap debug printing
  bool cliDebugNormalVelocity = false;
  std::string cliDebugNormalVelocityCsvPath;
  std::string cliEnergyBalanceCsvPath;
  int cliEarlyPairDiagnostics = -1;
  int cliEarlyPairStart = -1;
  int cliEarlyPairEnd = -1;
  int cliEarlyPairStride = -1;
  std::string cliEarlyPairScheduleMode;
  int cliEarlyPairGeomspaceSamples = -1;
  std::string cliEarlyPairContactCsvPath;
  std::string cliEarlyPairDistanceCsvPath;
  double cliEarlyPairDistanceCutoff = std::numeric_limits<double>::quiet_NaN();
  int cliEarlyPairBinaryDistance = -1;
  bool cliCheckInitNonpenetration =
      false; // run minPairGap once right after init
  // Soft contact CLI
  int cliSpatialHash = -1; // -1=unset, 0=false, 1=true
  int cliUseAABB = -1;     // -1=unset, 0=false, 1=true
  float cliCellSize = -1.0f;
  int cliVerboseSoft = -1; // -1=unset, 0=false, 1=true
  // Adaptive substeps CLI
  int cliAdaptive = -1; // -1 unset, 0 off, 1 on
  int cliAsMin = -1, cliAsMax = -1, cliAsHit = -1;
  double cliAsKEUp = std::numeric_limits<double>::quiet_NaN();
  double cliAsKEDown = std::numeric_limits<double>::quiet_NaN();
  // Stabilization CLI
  float cliBetaMin = std::numeric_limits<float>::quiet_NaN();
  int cliBetaHit = -1;
  float cliBetaScale = std::numeric_limits<float>::quiet_NaN();
  // Entanglement CLI
  bool cliEntanglement = false;
  double cliEntanglementCutoff = -1.0;
  int cliEntanglementPeriod = 60;
  int cliEntanglementThreads = 0;
  // Snapshot CLI
  int cliSnapStride = -1;
  int cliSnapFrames = -1;
  int cliSnapStart = 0;
  std::string cliSnapPath;
  // Playback CLI
  std::string cliPlaybackPath;  // NDJSON snapshots file
  std::string cliDumpFramesDir; // directory to write PNG frames
  std::string cliMoviePath;     // automatic movie output path
  int cliExportStride = 1;      // Export every N frames
  int cliPlaybackFps = 0;       // 0 => fastest
  bool cliOrbit = false;
  float cliOrbitSpeed = 0.5f;
  bool cliCamPosSet = false;
  glm::vec3 cliCamPos(0.0f);
  bool cliCamTargetSet = false;
  glm::vec3 cliCamTarget(0.0f);
  bool cliAutoFrame = false;
  float cliScale = 1.0f;
  float cliCamScale = 1.0f;
  bool cliSkipDupes = false;
  bool cliFramesOnly = false;      // hide window during playback frame dumping
  bool cliNoFloor = false;         // disable floor rendering in playback
  std::string cliInitCsvPath;      // initial configuration CSV (segments)
  std::string cliSaveInitPath;     // output path for initial configuration
  std::string cliInitStateCsvPath; // initial state CSV (per-rod format)
  std::string cliRelDispPath;      // relative displacement CSV
  std::string cliEarlyPairVelocitySummaryCsvPath;
  bool cliNetworkEmitEmpty = false;
  int cliRods = -1; // Override rod count
  int cliPerturbRod = -1;
  int cliFixedRods = -1;     // Number of rods to fix
  int cliFixEveryExcept = -1; // Fix all rods except this index

  // Specific velocity override
  bool cliSetVelEnabled = false;
  int cliSetVelId = -1;
  glm::vec3 cliSetVel(0.0f);

  // Specific angular velocity override
  bool cliSetAngVelEnabled = false;
  int cliSetAngVelId = -1;
  glm::vec3 cliSetAngVel(0.0f);

  int cliRenderStride = 1; // Render every N frames
  int cliCsvStride = 1;    // Log CSV every N frames
  int cliStatusStride = 1024; // Print CLI status every N steps
  bool cliHeadlessProgressEnabled = true;

  // Reptation summary
  std::string cliReptSummaryPath;
  int cliReptRodIdx = 0;

  // Wave logging
  int cliLogWavePeriod = 0;
  int cliLogWaveWidth = 0;
  float cliConstAccelSigma = 0.0f;
  bool cliUseConstantRandomAccel = false;

  bool cliPaused = false;
  bool cliWhiteBg = false;
  bool cliAutoReplay = false; // Automatically replay after headless run

  bool cliKarnopp = false;
  bool cliCundall = false;
  double cliKt = -1.0;
  double cliVelDeadband = -1.0;
  // NSC (hard contact) CLI
  bool cliNsc = false;
  std::string cliContactModel;
  int cliNscIters = -1;
  float cliNscBeta = -1.0f;
  float cliNscCfm = -1.0f;
  float cliNscOmega = -1.0f;
  float cliNscMu = -1.0f;
  int cliNscPosIters = -1;
  int cliNscPosPsor = -1;
  bool cliNoNscPos = false;
  bool cliNoWarmStart = false;

  bool cliNoLabel = false;
  float cliRodDiameter = -1.0f;
  bool cliAutoExit = false;
  std::string cliRunFolder;
  std::string cliTestRodEndpointsPath;
  int cliTestRodId = -1;
  int cliTestRodEndpointsStride = -1;
  int cliTestRodEndpointsMaxFrames = -1;

  // Parse command line arguments
  // Allow first positional argument (if doesn't start with --) to be scene path
  if (argc > 1 && argv[1][0] != '-') {
    scenePath = argv[1];
  }

  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
      std::cout << "Rod Dynamics 3D - Rigid Body Simulation\n\n";
      std::cout << "Usage: " << argv[0] << " [options]\n\n";
      std::cout << "Scene Configuration:\n";
      std::cout << "  --scene <path>              Load scene from JSON file "
                   "(default: assets/scenes/default_entangled.json)\n";
      std::cout << "  --run-folder <path>         Auto-configure from folder "
                   "(scene.json, x_relaxed.txt)\n\n";
      std::cout << "Execution Modes:\n";
      std::cout << "  --headless                  Run without graphics\n";
      std::cout << "  --steps <N>                 Number of steps for headless "
                   "mode (default: 1000)\n";
        std::cout << "  --stop-ke-threshold <E>     Stop headless early when total KE < E\n";
        std::cout << "  --stop-ke-avg-window <N>    Rolling avg window for stop-KE check (default: 1)\n";
        std::cout << "  --stop-ke-min-steps <N>     Minimum steps before KE stop check\n";
          std::cout << "  --stop-slide-vel-threshold <V> Stop headless early when |v·axis| < V\n";
          std::cout << "  --stop-slide-vel-min-steps <N> Minimum steps before sliding-vel stop check\n";
      std::cout
          << "  --perturb-rod <ID>          Apply random init velocity ONLY "
             "to this rod\n";
      std::cout << "  --fixed-rods <N>            Number of rods to fix (first "
                   "by method, rest random)\n";
      std::cout << "  --fix-every-except <ID>     Fix all rods except the one "
                   "with this index\n";
      std::cout << "  --set-velocity <ID> vx vy vz   Override initial linear velocity of rod\n";
      std::cout << "  --set-ang-velocity <ID> wx wy wz   Override initial angular velocity of rod\n";
      std::cout << "  --reptation-summary <path>  Write reptation summary CSV at end of run\n";
      std::cout << "  --reptation-rod <ID>        Rod index for reptation tracking (default: 0)\n";
      std::cout << "  --render-stride <N>         Render every N frames "
                   "(default: 1)\n\n";
      std::cout << "Output & Logging:\n";
      std::cout << "  --status-stride <N>         Print CLI status every N headless steps "
           "(default: 1024)\n";
       std::cout << "  --no-headless-progress      Suppress periodic [Headless] frame status lines\n";
      std::cout << "  --csv [path]                Enable CSV profile output "
                   "(default: profile.csv)\n";
      std::cout << "  --no-csv                    Disable CSV profile output completely\n";
      std::cout << "  --csv-stride <N>            Log CSV every N frames "
                   "(default: 1)\n";
      std::cout << "  --perrod [path]             Enable per-rod trajectory "
                   "CSV (default: perrod.csv)\n";
        std::cout
          << "  --test-rod-endpoints [path] Endpoint-only trajectory CSV "
           "for one rod (default: test_rod_endpoints.csv)\n";
        std::cout << "  --test-rod-id <ID>          Rod index for --test-rod-"
               "endpoints (default: --fix-every-except rod)\n";
          std::cout << "  --test-rod-endpoints-stride <N> Sample every N frames "
                 "for test rod endpoint logging (default: 1)\n";
          std::cout << "  --test-rod-endpoints-max <N> Max sampled frames to log "
               "for test rod endpoints (auto stride if headless)\n";
             std::cout << "  --test-rod-max <N>           Alias for --test-rod-"
               "endpoints-max\n";
      std::cout << "  --perrod-max <N>            Max frames to log in per-rod "
                   "CSV (default: 1000)\n";
      std::cout << "  --soft-pe <path>            Log soft contact potential "
                   "energy\n";
      std::cout << "  --com <path>                Track center-of-mass "
                   "(default: com.csv)\n";
      std::cout << "  --network <path>            Track contact network "
                   "(default: network.csv)\n";
      std::cout
          << "  --network-emit-empty        Emit sentinel rows for frames "
             "with zero contacts (rod_i=-1, rod_j=-1)\n";
      std::cout << "  --debug-normal-velocity     Print NSC normal relative velocities before/after solve\n";
      std::cout << "  --debug-normal-velocity-csv [path]  Log NSC pre/post normal+tangential speeds to CSV\n";
      std::cout << "  --energy-balance-csv <path>  Log per-contact energy balance diagnostics to CSV\n";
      std::cout << "  --early-pair-diagnostics    Enable early pair diagnostics scaffolding\n";
      std::cout << "  --no-early-pair-diagnostics Disable early pair diagnostics scaffolding\n";
      std::cout << "  --early-pair-start <N>      First step to sample (default: 100)\n";
      std::cout << "  --early-pair-end <N>        Last step to sample (default: 10000)\n";
      std::cout << "  --early-pair-stride <N>     Sample every N steps (default: 1)\n";
      std::cout << "  --early-pair-schedule <linear|geomspace>  Sampling schedule (default: linear)\n";
      std::cout << "  --early-pair-geom-count <N> Number of geomspace samples between start and end\n";
      std::cout << "  --early-pair-contact-csv <path>   Output path for active-contact diagnostics\n";
      std::cout << "  --early-pair-distance-csv <path>  Enable and set output path for all-pair distance diagnostics\n";
      std::cout << "  --early-pair-velocity-summary-csv <path>  Output path for per-frame averaged all-pair velocities\n";
      std::cout << "  --early-pair-distance-cutoff <d>  Optional signed-gap cutoff for dense pair output\n";
      std::cout << "  --early-pair-binary-distance       Use binary format for dense pair-distance output\n";
      std::cout << "Solver Configuration:\n";
      std::cout << "  --dt <float>                Timestep size\n";
      std::cout << "  --karnopp                   Enable Karnopp stick-slip "
                   "friction\n";
      std::cout
          << "  --cundall                   Enable Cundall-Strack friction\n";
      std::cout << "  --kt <float>                Tangential stiffness for "
                   "Cundall (N/m)\n";
      std::cout << "  --vel-deadband <float>      Velocity deadband for "
                   "Karnopp (m/s)\n";
      std::cout << "  --contact-model <name>      Select contact model: nsc, "
                   "harmonic, hertz-mindlin, mujoco\n";
      std::cout << "  --nsc                       Enable NSC (hard) contact "
                   "solver\n";
      std::cout << "  --nsc-iters <N>             PSOR velocity iterations "
                   "(default: 40)\n";
      std::cout << "  --nsc-beta <f>              Baumgarte factor "
                   "(default: 0.2)\n";
      std::cout << "  --nsc-cfm <f>               Constraint regularization "
                   "(default: 0.0)\n";
      std::cout << "  --nsc-omega <f>             SOR relaxation factor "
                   "(default: 1.0)\n";
      std::cout << "  --nsc-mu <f>                Friction coefficient "
                   "(default: 0.3)\n";
      std::cout << "  --nsc-pos-iters <N>         Position stabilization "
                   "outer iters (default: 5)\n";
      std::cout << "  --nsc-pos-psor <N>          Position stabilization "
                   "inner PSOR iters (default: 50)\n";
      std::cout << "  --no-nsc-pos                Disable position "
                   "stabilization\n";
      std::cout << "  --no-warm-start             Disable warm starting\n\n";
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
      std::cout << "  --beta-hit <N>              Hit count for beta "
                   "adjustment\n";
      std::cout << "  --beta-scale <float>        Beta scaling factor\n\n";
      std::cout << "Entanglement:\n";
      std::cout << "  --entanglement              Enable entanglement "
                   "computation\n";
      std::cout << "  --ent-cutoff <float>        Distance cutoff for linking "
                   "(default: 5.0)\n";
      std::cout << "  --ent-period <N>            Compute every N frames "
                   "(default: 60)\n";
      std::cout << "  --ent-threads <N>           Thread count (0=auto)\n\n";
      std::cout << "Other:\n";
      std::cout << "  --quiet, -q                 Suppress verbose output (keep frame summaries)\n";
      std::cout << "  --seed <N>                  Random seed\n";
      std::cout << "  --threads <N>               Thread limit (0=auto)\n";
      std::cout << "  --profile                   Enable profiling\n";
      std::cout << "\nPlayback / Visualization:\n";
      std::cout << "  --playback <snap.ndjson>    Playback snapshots (disables "
                   "physics)\n";
      std::cout << "  --dump-frames <dir>         Dump PNG frames to "
                   "directory\n";
      std::cout << "  --fps <N>                   Playback target FPS "
                   "(0=fast)\n";
      std::cout << "  --orbit                     Enable simple camera orbit\n";
      std::cout << "  --orbit-speed <float>       Orbit angular speed scale\n";
      std::cout << "  --cam-pos x y z             Override camera position\n";
      std::cout << "  --cam-target x y z          Override camera "
                   "target/center\n";
      std::cout << "  --auto-frame                Auto center/zoom based on "
                   "first snapshot\n";
      std::cout
          << "  --cam-scale <f>             Scale auto-framed camera distance "
             "(>1 = zoom out, <1 = zoom in)\n";
      std::cout << "  --scale <f>                 Downsample (f<1) or upsample "
                   "(f>1) before PNG write\n";
      std::cout << "  --skip-dupes                Skip writing PNG for "
                   "duplicate consecutive snapshots\n";
      std::cout << "  --frames-only               Hide window (offscreen) "
                   "while dumping playback frames\n";
      std::cout << "  --frames-only               Hide window (offscreen) "
                   "while dumping playback frames\n";
      std::cout << "  --export <dir>              Dump PNG frames to directory "
                   "(alias for --dump-frames)\n";
      std::cout << "  --export-stride <N>         Export every N frames "
                   "(default: 1)\n";
      std::cout << "  --movie <file.mp4>          Automatically run ffmpeg to "
                   "create movie after export\n";
      std::cout << "  --no-floor                  Disable floor rendering in "
                   "playback\n";
      std::cout << "\nInitial Configuration:\n";
      std::cout << "  --init-csv <path>          Load initial rods from CSV "
                   "with endpoints (x0..z1)\n";
      std::cout
          << "  --save-init <path>         Save initial rod configuration "
             "to CSV\n";
      std::cout << "\nDiagnostics / Metrics:\n";
      std::cout << "  --com <path>                Track center-of-mass "
                   "(default: com.csv)\n";
      std::cout << "  --debug-com                 Track center-of-mass to "
                   "com_debug.csv\n";
      std::cout << "  --reldisp <path>           Track ri-rc and L2 norm "
                   "(default: reldisp.csv)\n";
      std::cout << "  --paused                   Start the simulation paused\n";
      std::cout << "  --white-bg                 Use white background\n";
      std::cout << "  --help, -h                  Show this help message\n\n";
      std::cout << "Examples:\n";
      std::cout << "  " << argv[0] << " --scene my_scene.json\n";
      std::cout << "  " << argv[0]
                << " --headless --steps 10000 --csv --perrod\n";
      std::cout << "  " << argv[0]
                << " --scene confined.json --dt 0.001 --nsc-iters 20\n";
      return 0;
    } else if (std::string(argv[i]) == "--scene" && i + 1 < argc) {
      scenePath = argv[++i];
    } else if ((std::string(argv[i]) == "--run-folder" ||
                std::string(argv[i]) == "--run-rolder") &&
               i + 1 < argc) {
      cliRunFolder = argv[++i];
    } else if (std::string(argv[i]) == "--profile") {
      enableProfile = true;
    } else if (std::string(argv[i]) == "--csv") {
      // The profile CSV is useless without timing collection, so --csv
      // implies --profile.
      enableProfile = true;
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        csvPath = argv[++i];
      } else {
        csvPath = "profile.csv";
      }
    } else if (std::string(argv[i]) == "--no-csv") {
      noCsv = true;
    } else if (std::string(argv[i]) == "--quiet" || std::string(argv[i]) == "-q") {
      gQuiet = true;
    } else if (std::string(argv[i]) == "--headless") {
      headlessFlag = true;
    } else if (std::string(argv[i]) == "--steps" && i + 1 < argc) {
      headlessSteps = std::stoi(argv[++i]);
    } else if (std::string(argv[i]) == "--render-stride" && i + 1 < argc) {
      cliRenderStride = std::max(1, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--status-stride" && i + 1 < argc) {
      cliStatusStride = std::max(1, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--no-headless-progress") {
      cliHeadlessProgressEnabled = false;
    } else if (std::string(argv[i]) == "--perturb-rod" && i + 1 < argc) {
      cliPerturbRod = std::stoi(argv[++i]);
    } else if (std::string(argv[i]) == "--fixed-rods" && i + 1 < argc) {
      cliFixedRods = std::max(0, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--fix-every-except" && i + 1 < argc) {
      cliFixEveryExcept = std::stoi(argv[++i]);
    } else if (std::string(argv[i]) == "--csv-stride" && i + 1 < argc) {
      cliCsvStride = std::max(1, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--auto-replay") {
      cliAutoReplay = true;
    } else if (std::string(argv[i]) == "--paused") {
      cliPaused = true;
    } else if (std::string(argv[i]) == "--white-bg") {
      cliWhiteBg = true;
    } else if (std::string(argv[i]) == "--no-label") {
      cliNoLabel = true;
    } else if (std::string(argv[i]) == "--rod-diameter" && i + 1 < argc) {
      cliRodDiameter = std::stof(argv[++i]);
    } else if (std::string(argv[i]) == "--auto-exit") {
      cliAutoExit = true;
    } else if (std::string(argv[i]) == "--perrod") {
      if (i + 1 < argc && argv[i + 1][0] != '-')
        perRodPath = argv[++i];
      else
        perRodPath = "perrod.csv";
    } else if (std::string(argv[i]) == "--test-rod-endpoints") {
      if (i + 1 < argc && argv[i + 1][0] != '-')
        cliTestRodEndpointsPath = argv[++i];
      else
        cliTestRodEndpointsPath = "test_rod_endpoints.csv";
    } else if (std::string(argv[i]) == "--test-rod-id" && i + 1 < argc) {
      cliTestRodId = std::stoi(argv[++i]);
    } else if (std::string(argv[i]) == "--test-rod-endpoints-stride" &&
               i + 1 < argc) {
      cliTestRodEndpointsStride = std::max(1, std::stoi(argv[++i]));
    } else if ((std::string(argv[i]) == "--test-rod-endpoints-max" ||
                std::string(argv[i]) == "--test-rod-max") &&
               i + 1 < argc) {
      cliTestRodEndpointsMaxFrames = std::max(1, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--perrod-max" && i + 1 < argc) {
      perRodMaxFrames = std::max(1, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--perrod-stride" && i + 1 < argc) {
      cliPerrodStride = std::max(1, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--output-stride" && i + 1 < argc) {
      cliOutputStride = std::max(1, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--output-max" && i + 1 < argc) {
      cliOutputMax = std::max(0, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--network-stride" && i + 1 < argc) {
      cliNetworkStride = std::max(1, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--network-max" && i + 1 < argc) {
      cliNetworkMax = std::max(0, std::stoi(argv[++i]));

    } else if (std::string(argv[i]) == "--seed" && i + 1 < argc) {
      cliSeed = std::stoi(argv[++i]);
    } else if (std::string(argv[i]) == "--rods" && i + 1 < argc) {
      cliRods = std::max(1, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--min-moment" && i + 1 < argc) {
      // Workaround for FDT stability: enforce minimum inertia
      float minI = std::stof(argv[++i]);
      if (minI > 0.0f) {
        if (!gQuiet) std::cerr << "[Debug] Enforcing min moment of inertia: " << minI
                  << std::endl;
      }
      // Store in a global or apply immediately if rods exist?
      // Since rods are created later, we need to pass this to App or apply it
      // after init. We'll store it in a static/global or modify App to accept
      // it. For expedience, we'll apply it just before running.
      g_minMoment = minI;
    } else if (std::string(argv[i]) == "--threads" && i + 1 < argc) {
      cliThreads = std::max(0, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--dt" && i + 1 < argc) {
      cliDt = std::max(0.0f, std::stof(argv[++i]));
    } else if (std::string(argv[i]) == "--stop-ke-threshold" && i + 1 < argc) {
      cliStopKEThreshold = std::max(0.0, std::stod(argv[++i]));
    } else if (std::string(argv[i]) == "--stop-ke-min-steps" && i + 1 < argc) {
      cliStopKEMinSteps = std::max(0, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--stop-ke-avg-window" && i + 1 < argc) {
      cliStopKEAvgWindow = std::max(1, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--stop-slide-vel-threshold" && i + 1 < argc) {
      cliStopSlideVelThreshold = std::max(0.0, std::stod(argv[++i]));
    } else if (std::string(argv[i]) == "--stop-slide-vel-min-steps" && i + 1 < argc) {
      cliStopSlideVelMinSteps = std::max(0, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--soft-pe" && i + 1 < argc) {
      cliSoftPEPath = argv[++i];
    } else if (std::string(argv[i]) == "--com" && i + 1 < argc) {
      cliCOMPath = argv[++i];
    } else if (std::string(argv[i]) == "--debug-com") {
      cliCOMPath = "com_debug.csv";
    } else if (std::string(argv[i]) == "--network" && i + 1 < argc) {
      cliNetworkPath = argv[++i];
    } else if (std::string(argv[i]) == "--network-emit-empty") {
      cliNetworkEmitEmpty = true;
    } else if (std::string(argv[i]) == "--per-rod" && i + 1 < argc) {
      perRodPath = argv[++i];
    } else if (std::string(argv[i]) == "--limit-per-rod" && i + 1 < argc) {
      perRodMaxFrames = std::max(1, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--output") {
      // Optional compact output CSV; path argument optional
      if (i + 1 < argc && argv[i + 1][0] != '-')
        cliOutputPath = argv[++i];
      else
        cliOutputPath = "output.csv";
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
    } else if (std::string(argv[i]) == "--set-velocity" && i + 4 < argc) {
      cliSetVelEnabled = true;
      cliSetVelId = std::stoi(argv[++i]);
      float vx = std::stof(argv[++i]);
      float vy = std::stof(argv[++i]);
      float vz = std::stof(argv[++i]);
      cliSetVel = glm::vec3(vx, vy, vz);
    } else if (std::string(argv[i]) == "--set-ang-velocity" && i + 4 < argc) {
      cliSetAngVelEnabled = true;
      cliSetAngVelId = std::stoi(argv[++i]);
      float wx = std::stof(argv[++i]);
      float wy = std::stof(argv[++i]);
      float wz = std::stof(argv[++i]);
      cliSetAngVel = glm::vec3(wx, wy, wz);
    } else if (std::string(argv[i]) == "--beta-min" && i + 1 < argc) {
      cliBetaMin = std::stof(argv[++i]);
    } else if (std::string(argv[i]) == "--beta-hit" && i + 1 < argc) {
      cliBetaHit = std::max(0, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--beta-scale" && i + 1 < argc) {
      cliBetaScale = std::stof(argv[++i]);
    } else if (std::string(argv[i]) == "--entanglement") {
      cliEntanglement = true;
    } else if (std::string(argv[i]) == "--reptation-summary" && i + 1 < argc) {
      cliReptSummaryPath = argv[++i];
    } else if (std::string(argv[i]) == "--reptation-rod" && i + 1 < argc) {
      cliReptRodIdx = std::stoi(argv[++i]);
    } else if (std::string(argv[i]) == "--ent-cutoff" && i + 1 < argc) {
      cliEntanglementCutoff = std::stod(argv[++i]);
    } else if (std::string(argv[i]) == "--ent-period" && i + 1 < argc) {
      cliEntanglementPeriod = std::max(1, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--ent-threads" && i + 1 < argc) {
      cliEntanglementThreads = std::max(0, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--entanglement-cutoff" &&
               i + 1 < argc) {
      cliEntanglementCutoff = std::stod(argv[++i]);
    } else if (std::string(argv[i]) == "--entanglement-period" &&
               i + 1 < argc) {
      cliEntanglementPeriod = std::max(1, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--entanglement-threads" &&
               i + 1 < argc) {
      cliEntanglementThreads = std::max(0, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--snap-stride" && i + 1 < argc) {
      cliSnapStride = std::max(1, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--snap-frames" && i + 1 < argc) {
      cliSnapFrames = std::max(1, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--snap-start" && i + 1 < argc) {
      cliSnapStart = std::max(0, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--snap-path" && i + 1 < argc) {
      cliSnapPath = argv[++i];
    } else if (std::string(argv[i]) == "--playback" && i + 1 < argc) {
      cliPlaybackPath = argv[++i];
    } else if (std::string(argv[i]) == "--dump-frames" && i + 1 < argc) {
      cliDumpFramesDir = argv[++i];
    } else if (std::string(argv[i]) == "--export" && i + 1 < argc) {
      cliDumpFramesDir = argv[++i];
    } else if (std::string(argv[i]) == "--export-stride" && i + 1 < argc) {
      cliExportStride = std::max(1, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--movie" && i + 1 < argc) {
      cliMoviePath = argv[++i];
    } else if (std::string(argv[i]) == "--fps" && i + 1 < argc) {
      cliPlaybackFps = std::max(0, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--orbit") {
      cliOrbit = true;
    } else if (std::string(argv[i]) == "--orbit-speed" && i + 1 < argc) {
      cliOrbitSpeed = std::stof(argv[++i]);
    } else if (std::string(argv[i]) == "--cam-pos" && i + 3 < argc) {
      ++i;
      float px = std::stof(argv[i]);
      ++i;
      float py = std::stof(argv[i]);
      ++i;
      float pz = std::stof(argv[i]);
      cliCamPos = glm::vec3(px, py, pz);
      cliCamPosSet = true;
    } else if (std::string(argv[i]) == "--cam-target" && i + 3 < argc) {
      ++i;
      float tx = std::stof(argv[i]);
      ++i;
      float ty = std::stof(argv[i]);
      ++i;
      float tz = std::stof(argv[i]);
      cliCamTarget = glm::vec3(tx, ty, tz);
      cliCamTargetSet = true;
    } else if (std::string(argv[i]) == "--auto-frame") {
      cliAutoFrame = true;
    } else if (std::string(argv[i]) == "--cam-scale" && i + 1 < argc) {
      cliCamScale = std::stof(argv[++i]);
    } else if (std::string(argv[i]) == "--scale" && i + 1 < argc) {
      cliScale = std::max(0.01f, std::stof(argv[++i]));
    } else if (std::string(argv[i]) == "--skip-dupes") {
      cliSkipDupes = true;
    } else if (std::string(argv[i]) == "--frames-only") {
      cliFramesOnly = true;
    } else if (std::string(argv[i]) == "--no-floor") {
      cliNoFloor = true;
    } else if (std::string(argv[i]) == "--init-csv" && i + 1 < argc) {
      cliInitCsvPath = argv[++i];
    } else if (std::string(argv[i]) == "--save-init" && i + 1 < argc) {
      cliSaveInitPath = argv[++i];
    } else if (std::string(argv[i]) == "--init-state-csv" && i + 1 < argc) {
      cliInitStateCsvPath = argv[++i];
    } else if (std::string(argv[i]) == "--reldisp") {
      if (i + 1 < argc && argv[i + 1][0] != '-')
        cliRelDispPath = argv[++i];
      else
        cliRelDispPath = "reldisp.csv";
    } else if (std::string(argv[i]) == "--debug-min-gap") {
      cliDebugMinGap = true;
    } else if (std::string(argv[i]) == "--debug-normal-velocity") {
      cliDebugNormalVelocity = true;
    } else if (std::string(argv[i]) == "--debug-normal-velocity-csv") {
      if (i + 1 < argc && argv[i + 1][0] != '-')
        cliDebugNormalVelocityCsvPath = argv[++i];
      else
        cliDebugNormalVelocityCsvPath = "nsc_contact_velocities.csv";
    } else if (std::string(argv[i]) == "--energy-balance-csv" && i + 1 < argc) {
      cliEnergyBalanceCsvPath = argv[++i];
    } else if (std::string(argv[i]) == "--early-pair-diagnostics") {
      cliEarlyPairDiagnostics = 1;
    } else if (std::string(argv[i]) == "--no-early-pair-diagnostics") {
      cliEarlyPairDiagnostics = 0;
    } else if (std::string(argv[i]) == "--early-pair-start" && i + 1 < argc) {
      cliEarlyPairStart = std::max(0, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--early-pair-end" && i + 1 < argc) {
      cliEarlyPairEnd = std::stoi(argv[++i]);
    } else if (std::string(argv[i]) == "--early-pair-stride" && i + 1 < argc) {
      cliEarlyPairStride = std::max(1, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--early-pair-schedule" && i + 1 < argc) {
      cliEarlyPairScheduleMode = argv[++i];
    } else if (std::string(argv[i]) == "--early-pair-geom-count" && i + 1 < argc) {
      cliEarlyPairGeomspaceSamples = std::max(1, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--early-pair-contact-csv" && i + 1 < argc) {
      cliEarlyPairContactCsvPath = argv[++i];
    } else if (std::string(argv[i]) == "--early-pair-distance-csv" && i + 1 < argc) {
      cliEarlyPairDistanceCsvPath = argv[++i];
    } else if (std::string(argv[i]) == "--early-pair-velocity-summary-csv" && i + 1 < argc) {
      cliEarlyPairVelocitySummaryCsvPath = argv[++i];
    } else if (std::string(argv[i]) == "--early-pair-distance-cutoff" && i + 1 < argc) {
      cliEarlyPairDistanceCutoff = std::stod(argv[++i]);
    } else if (std::string(argv[i]) == "--early-pair-binary-distance") {
      cliEarlyPairBinaryDistance = 1;
    } else if (std::string(argv[i]) == "--check-init-nonpenetration") {
      cliCheckInitNonpenetration = true;
    } else if (std::string(argv[i]) == "--use-spatial-hash") {
      cliSpatialHash = 1;
    } else if (std::string(argv[i]) == "--no-spatial-hash") {
      cliSpatialHash = 0;
    } else if (std::string(argv[i]) == "--use-aabb") {
      cliUseAABB = 1;
    } else if (std::string(argv[i]) == "--no-aabb") {
      cliUseAABB = 0;
    } else if (std::string(argv[i]) == "--cell-size" && i + 1 < argc) {
      cliCellSize = std::stof(argv[++i]);
    } else if (std::string(argv[i]) == "--verbose-soft") {
      cliVerboseSoft = 1;
    } else if (std::string(argv[i]) == "--no-verbose-soft") {
      cliVerboseSoft = 0;
    } else if (std::string(argv[i]) == "--karnopp") {
      cliKarnopp = true;
    } else if (std::string(argv[i]) == "--cundall") {
      cliCundall = true;
    } else if (std::string(argv[i]) == "--kt" && i + 1 < argc) {
      cliKt = std::stof(argv[++i]);
    } else if (std::string(argv[i]) == "--vel-deadband" && i + 1 < argc) {
      cliVelDeadband = std::stof(argv[++i]);
    } else if (std::string(argv[i]) == "--contact-model" && i + 1 < argc) {
      cliContactModel = argv[++i];
    } else if (std::string(argv[i]) == "--nsc") {
      cliNsc = true;
    } else if (std::string(argv[i]) == "--nsc-iters" && i + 1 < argc) {
      cliNscIters = std::stoi(argv[++i]);
    } else if (std::string(argv[i]) == "--nsc-beta" && i + 1 < argc) {
      cliNscBeta = std::stof(argv[++i]);
    } else if (std::string(argv[i]) == "--nsc-cfm" && i + 1 < argc) {
      cliNscCfm = std::stof(argv[++i]);
    } else if (std::string(argv[i]) == "--nsc-omega" && i + 1 < argc) {
      cliNscOmega = std::stof(argv[++i]);
    } else if (std::string(argv[i]) == "--nsc-mu" && i + 1 < argc) {
      cliNscMu = std::stof(argv[++i]);
    } else if (std::string(argv[i]) == "--nsc-pos-iters" && i + 1 < argc) {
      cliNscPosIters = std::stoi(argv[++i]);
    } else if (std::string(argv[i]) == "--nsc-pos-psor" && i + 1 < argc) {
      cliNscPosPsor = std::stoi(argv[++i]);
    } else if (std::string(argv[i]) == "--no-nsc-pos") {
      cliNoNscPos = true;
    } else if (std::string(argv[i]) == "--no-warm-start" ||
               std::string(argv[i]) == "--no-warmstart") {
      cliNoWarmStart = true;
    } else if (std::string(argv[i]) == "--log-wave-period" && i + 1 < argc) {
      cliLogWavePeriod = std::max(1, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--log-wave-width" && i + 1 < argc) {
      cliLogWaveWidth = std::max(1, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--random-accel-sigma" && i + 1 < argc) {
      cliConstAccelSigma = std::stof(argv[++i]);
      cliUseConstantRandomAccel = true;
    } else if (argv[i][0] == '-') {
      // Unknown options were previously ignored silently, so a typo like
      // --nsc-itres ran a full simulation with defaults.
      std::cerr << "Error: unknown option '" << argv[i]
                << "' (see --help)\n";
      return 1;
    }
  }

  // Auto-configure from folder if requested
  if (!cliRunFolder.empty()) {
    std::filesystem::path folder(cliRunFolder);
    if (std::filesystem::exists(folder / "scene.json")) {
      scenePath = (folder / "scene.json").string();
    }
    if (std::filesystem::exists(folder / "x_relaxed.txt")) {
      cliInitCsvPath = (folder / "x_relaxed.txt").string();
    } else if (std::filesystem::exists(folder / "init_csv.csv")) {
      cliInitCsvPath = (folder / "init_csv.csv").string();
    }
    if (!gQuiet) std::cout << "[App] --run-folder: " << cliRunFolder << "\n"
              << "      scene=" << scenePath << "\n"
              << "      init=" << cliInitCsvPath << "\n";
  }

  AppCfg settings = defaultAppCfg();

  // Load scene configuration (keep defaults if load fails)
  if (!gQuiet) std::cout << "[App] Loading scene configuration from: " << scenePath << "\n";
  if (!loadConfigFromFile(scenePath, settings)) {
    std::cerr << "Warning: Could not load scene file '" << scenePath
              << "', using defaults.\n";
  } else {
    if (!gQuiet) std::cout << "[App] Successfully loaded scene: " << scenePath << "\n";
  }

  // Apply CLI overrides to settings
  if (cliRods > 0)
    settings.scene.populate.count = cliRods;

  if (cliFixedRods >= 0)
    settings.scene.numFixedRods = cliFixedRods;

  if (cliFixEveryExcept >= 0)
    settings.scene.fixEveryExcept = cliFixEveryExcept;

  if (cliSeed != 0) {
    settings.scene.populate.seed = cliSeed;
    settings.scene.randomInit.seed = cliSeed;
  }

  if (cliThreads >= 0) {
    g_thread_limit = cliThreads;
    if (cliThreads > 0)
      g_user_threads_set = true;
  }
  if (cliDt > 0.0f)
    settings.physics.dt = cliDt;

  if (cliSpatialHash != -1)
    settings.physics.soft_contact.use_spatial_hash = (cliSpatialHash != 0);
  if (cliUseAABB != -1)
    settings.physics.soft_contact.use_aabb = (cliUseAABB != 0);
  if (cliKarnopp)
    settings.physics.soft_contact.friction_karnopp = true;
  if (cliCundall)
    settings.physics.soft_contact.friction_cundall = true;
  if (cliKt > 0.0)
    settings.physics.soft_contact.kt = cliKt;
  if (cliVelDeadband > 0.0)
    settings.physics.soft_contact.vel_deadband = cliVelDeadband;
  if (cliCellSize > 0.0f)
    settings.physics.soft_contact.cell_size = cliCellSize;
  if (cliVerboseSoft != -1)
    settings.physics.soft_contact.verbose = (cliVerboseSoft != 0);

  // Contact-model selection (overrides scene JSON; --nsc etc. below can
  // still further-override for back-compat)
  if (!cliContactModel.empty()) {
    if (!applyContactModel(settings, cliContactModel))
      return 1;
  }

  // NSC CLI overrides
  if (cliNsc)
    settings.physics.nsc.enabled = true;
  if (cliNscIters > 0)
    settings.physics.nsc.velocity_iters = cliNscIters;
  if (cliNscBeta >= 0.0f)
    settings.physics.nsc.beta = cliNscBeta;
  if (cliNscCfm >= 0.0f)
    settings.physics.nsc.cfm = cliNscCfm;
  if (cliNscOmega > 0.0f)
    settings.physics.nsc.omega = cliNscOmega;
  if (cliNscMu >= 0.0f)
    settings.physics.nsc.mu = cliNscMu;
  if (cliNoWarmStart)
    settings.physics.nsc.enable_warm_start = false;
  if (cliNscPosIters > 0)
    settings.physics.nsc.position_iters = cliNscPosIters;
  if (cliNscPosPsor > 0)
    settings.physics.nsc.position_psor_iters = cliNscPosPsor;
  if (cliNoNscPos)
    settings.physics.nsc.position_stabilization = false;

  App a;
  gHeadlessProgressEnabled = cliHeadlessProgressEnabled;
  a.setHeadless(headlessFlag);
  a.setHeadlessSteps(headlessSteps);
  a.setStopKEThreshold(cliStopKEThreshold);
  a.setStopKEMinSteps(cliStopKEMinSteps);
  a.setStopKEAvgWindow(cliStopKEAvgWindow);
  a.setStopSlideVelThreshold(cliStopSlideVelThreshold);
  a.setStopSlideVelMinSteps(cliStopSlideVelMinSteps);
  a.setCsvStride(cliCsvStride);
  a.setCliStatusStride(cliStatusStride);

  // Set output/network default strides to match CSV stride if not specified
  if (cliOutputStride > 0)
    a.setOutputStride(cliOutputStride);
  else if (cliCsvStride > 1)
    a.setOutputStride(cliCsvStride);

  if (cliOutputMax >= 0)
    a.setOutputMax(cliOutputMax);

  if (cliNetworkStride > 0)
    a.setNetworkStride(cliNetworkStride);
  else if (cliCsvStride > 1)
    a.setNetworkStride(cliCsvStride);

  if (cliNetworkMax >= 0)
    a.setNetworkMax(cliNetworkMax);

  if (cliPerrodStride > 0)
    a.setPerRodStride(cliPerrodStride);

  // Default: enable CSV profile (KE, contact count, timings)
  if (!noCsv) {
    if (!csvPath.empty())
      a.enableCsv(csvPath);
    else
      a.enableCsv("profile.csv");
  }

  if (!perRodPath.empty())
    a.enablePerRod(perRodPath, perRodMaxFrames);
  // Set test rod index (used by stop-slide-vel and reptation tracking)
  if (cliTestRodId >= 0)
    a.setTestRodIndex(cliTestRodId);
  if (!cliTestRodEndpointsPath.empty()) {
    a.enableTestRodEndpoints(cliTestRodEndpointsPath);
    if (cliTestRodEndpointsStride > 0)
      a.setTestRodEndpointsStride(cliTestRodEndpointsStride);
    if (cliTestRodEndpointsMaxFrames > 0)
      a.setTestRodEndpointsMaxFrames(cliTestRodEndpointsMaxFrames);
    std::cerr << "[app] Test-rod endpoint tracking enabled: "
              << cliTestRodEndpointsPath;
    if (cliTestRodId >= 0)
      std::cerr << " (rod=" << cliTestRodId << ")";
    else
      std::cerr << " (rod=auto from --fix-every-except)";
    if (cliTestRodEndpointsStride > 0)
      std::cerr << " stride=" << cliTestRodEndpointsStride;
    else
      std::cerr << " stride=auto";
    if (cliTestRodEndpointsMaxFrames > 0)
      std::cerr << " max_frames=" << cliTestRodEndpointsMaxFrames;
    else
      std::cerr << " max_frames=unlimited";
    std::cerr << "\n";
  }
  if (!cliSoftPEPath.empty())
    a.enableSoftPE(cliSoftPEPath);
  if (!cliCOMPath.empty())
    a.enableCOM(cliCOMPath);
  // Relative displacement CSV is optional; enable only if --reldisp is
  // provided
  if (!cliNetworkPath.empty()) {
    a.enableNetwork(cliNetworkPath);
    a.setNetworkEmitEmptyFrames(cliNetworkEmitEmpty);
    if (!gQuiet) std::cerr << "[app] Contact network tracking enabled: " << cliNetworkPath
              << "\n";
  }
  if (!cliOutputPath.empty()) {
    a.enableOutput(cliOutputPath);
    if (!gQuiet) std::cerr << "[app] Compact output logging enabled: " << cliOutputPath
              << "\n";
  }
  a.setProfiling(enableProfile);
  if (cliAdaptive != -1)
    a.enableAdaptiveSubsteps(cliAdaptive == 1);
  if (cliAsMin > 0 || cliAsMax > 0 || cliAsHit >= 0 || !std::isnan(cliAsKEUp) ||
      !std::isnan(cliAsKEDown)) {
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
    float defBetaMin = 0.0f;
    int defBetaHit = INT32_MAX;
    float defBetaScale = 1.0f;
    a.setStabilization(std::isnan(cliBetaMin) ? defBetaMin : cliBetaMin,
                       cliBetaHit >= 0 ? cliBetaHit : defBetaHit,
                       std::isnan(cliBetaScale) ? defBetaScale : cliBetaScale);
  }
  // Entanglement options
  if (cliEntanglement) {
    a.setEntanglement(true, cliEntanglementCutoff, cliEntanglementPeriod,
                      cliEntanglementThreads);
  }
  if (cliEarlyPairDiagnostics >= 0) {
    settings.diagnostics.early_pairs.enabled = (cliEarlyPairDiagnostics == 1);
  }
  if (cliEarlyPairStart >= 0) {
    settings.diagnostics.early_pairs.start_step = cliEarlyPairStart;
  }
  if (cliEarlyPairEnd >= 0) {
    settings.diagnostics.early_pairs.end_step = cliEarlyPairEnd;
  }
  if (cliEarlyPairStride > 0) {
    settings.diagnostics.early_pairs.stride = cliEarlyPairStride;
  }
  if (!cliEarlyPairScheduleMode.empty()) {
    settings.diagnostics.early_pairs.schedule_mode = cliEarlyPairScheduleMode;
  }
  if (cliEarlyPairGeomspaceSamples > 0) {
    settings.diagnostics.early_pairs.geomspace_samples =
        cliEarlyPairGeomspaceSamples;
  }
  if (!cliEarlyPairContactCsvPath.empty()) {
    settings.diagnostics.early_pairs.contact_output_path =
        cliEarlyPairContactCsvPath;
  }
  if (!cliEarlyPairDistanceCsvPath.empty()) {
    settings.diagnostics.early_pairs.pair_distance_output_path =
        cliEarlyPairDistanceCsvPath;
  }
  if (!cliEarlyPairVelocitySummaryCsvPath.empty()) {
    settings.diagnostics.early_pairs.pair_velocity_summary_output_path =
        cliEarlyPairVelocitySummaryCsvPath;
  }
  if (!std::isnan(cliEarlyPairDistanceCutoff)) {
    settings.diagnostics.early_pairs.pair_distance_cutoff =
        cliEarlyPairDistanceCutoff;
  }
  if (cliEarlyPairBinaryDistance >= 0) {
    settings.diagnostics.early_pairs.binary_pair_distance_output =
        (cliEarlyPairBinaryDistance == 1);
  }
  a.setConfig(settings);
  if (cliDebugMinGap) {
    a.setDebugMinGap(true);
    if (!gQuiet) std::cerr << "[app] minPairGap debug enabled via --debug-min-gap\n";
  }
  if (cliDebugNormalVelocity) {
    a.setDebugNormalVelocity(true);
    if (!gQuiet) std::cerr << "[app] NSC normal-velocity debug enabled via --debug-normal-velocity\n";
  }
  if (!cliDebugNormalVelocityCsvPath.empty()) {
    a.setDebugNormalVelocityCsv(cliDebugNormalVelocityCsvPath);
    if (!gQuiet) std::cerr << "[app] NSC normal-velocity CSV logging enabled: "
                           << cliDebugNormalVelocityCsvPath << "\n";
  }
  if (!cliEnergyBalanceCsvPath.empty()) {
    a.setEnergyBalanceCsv(cliEnergyBalanceCsvPath);
    if (!gQuiet) std::cerr << "[app] Energy balance CSV logging enabled: "
                           << cliEnergyBalanceCsvPath << "\n";
  }
  if (cliCheckInitNonpenetration) {
    a.setCheckInitNonpenetration(true);
    if (!gQuiet) std::cerr << "[app] Initial nonpenetration check enabled via "
                 "--check-init-nonpenetration\n";
  }
  if (!cliInitCsvPath.empty()) {
    a.setInitCsvPath(cliInitCsvPath);
    if (!gQuiet) std::cerr << "[app] Initial CSV configured: " << cliInitCsvPath << "\n";
  }
  if (!cliSaveInitPath.empty()) {
    a.setSaveInitPath(cliSaveInitPath);
  }
  if (!cliInitStateCsvPath.empty()) {
    a.setInitStateCsvPath(cliInitStateCsvPath);
    if (!gQuiet) std::cerr << "[app] Initial State CSV configured: " << cliInitStateCsvPath
              << "\n";
  }
  if (!cliRelDispPath.empty()) {
    a.enableRelDisp(cliRelDispPath);
    if (!gQuiet) std::cerr << "[app] Relative displacement tracking enabled: "
              << cliRelDispPath << "\n";
  }
  if (cliSnapStride > 0 && cliSnapFrames > 0) {
    a.enableSnapshots(cliSnapStride, cliSnapFrames, cliSnapPath, cliSnapStart);
    a.setLogOnSnapshotOnly(true);
  }

  if (cliLogWavePeriod > 0 && cliLogWaveWidth > 0) {
    a.setLogWave(cliLogWavePeriod, cliLogWaveWidth);
  }

  if (cliUseConstantRandomAccel) {
    a.setConstantRandomAccel(true, cliConstAccelSigma);
    if (!gQuiet) std::cerr << "[app] Constant random acceleration enabled (sigma="
              << cliConstAccelSigma << ")\n";
  }

  if (cliPerturbRod >= 0) {
    a.setPerturbationRod(cliPerturbRod);
    if (!gQuiet) std::cerr << "[app] Targeting random initialization to rod "
              << cliPerturbRod << "\n";
  }

  // Auto-replay configuration
  if (cliAutoReplay) {
    headlessFlag = true; // Force headless for the first phase
    a.setHeadless(true);
    a.setHeadlessSteps(headlessSteps);

    // Ensure snapshotting is enabled if not already
    if (cliSnapStride <= 0)
      cliSnapStride = (cliRenderStride > 1) ? cliRenderStride : 1;
    if (cliSnapFrames <= 0)
      cliSnapFrames = headlessSteps / cliSnapStride;
    if (cliSnapPath.empty())
      cliSnapPath = "auto_replay.ndjson";

    a.enableSnapshots(cliSnapStride, cliSnapFrames, cliSnapPath, cliSnapStart);
    if (!gQuiet) std::cerr << "[app] Auto-replay enabled. Recording to " << cliSnapPath
              << "\n";
  }

  a.setRenderStride(cliRenderStride);
  a.setPaused(cliPaused);
  if (cliWhiteBg) {
    a.setBackgroundColor(glm::vec3(1.0f));
  }
  int result = 0;
  a.showLabel = !cliNoLabel;
  a.rodDiameterOverride = cliRodDiameter;
  // Playback path: run playback then exit (only if not headless build /
  // not headless flag)
#ifndef HEADLESS_BUILD
  if (!cliPlaybackPath.empty()) {
    if (headlessFlag) {
      std::cerr << "[playback] Ignoring --headless (playback requires "
                   "graphics).\n";
    }
    return a.runPlayback(cliPlaybackPath, cliDumpFramesDir, cliPlaybackFps,
                         cliOrbit, cliOrbitSpeed, cliCamPosSet, cliCamPos,
                         cliCamTargetSet, cliCamTarget, cliAutoFrame, cliScale,
                         cliCamScale, cliSkipDupes, cliFramesOnly, cliNoFloor,
                         cliExportStride, cliMoviePath, cliAutoExit);
  }
#endif

  // Override specific rod velocity if requested
  if (cliSetVelEnabled && cliSetVelId >= 0) {
    a.setOverrideVelocity(cliSetVelId, cliSetVel);
    if (!gQuiet) std::cout << "[App] Configured velocity override for rod " << cliSetVelId
              << " to " << cliSetVel.x << "," << cliSetVel.y << ","
              << cliSetVel.z << "\n";
  }

  // Override specific rod angular velocity if requested
  if (cliSetAngVelEnabled && cliSetAngVelId >= 0) {
    a.setOverrideAngVelocity(cliSetAngVelId, cliSetAngVel);
    if (!gQuiet) std::cout << "[App] Configured angular velocity override for rod " << cliSetAngVelId
              << " to " << cliSetAngVel.x << "," << cliSetAngVel.y << ","
              << cliSetAngVel.z << "\n";
  }

  // Enable reptation summary if requested
  if (!cliReptSummaryPath.empty()) {
    a.enableReptSummary(cliReptSummaryPath, cliReptRodIdx);
  }

  result = a.run();

#ifndef HEADLESS_BUILD
  if (cliAutoReplay && result == 0) {
    if (!gQuiet) std::cerr << "[app] Headless run complete. Starting playback...\n";
    return a.runPlayback(cliSnapPath, cliDumpFramesDir, cliPlaybackFps,
                         cliOrbit, cliOrbitSpeed, cliCamPosSet, cliCamPos,
                         cliCamTargetSet, cliCamTarget, cliAutoFrame, cliScale,
                         cliCamScale, cliSkipDupes, cliFramesOnly, cliNoFloor,
                         cliExportStride, cliMoviePath, cliAutoExit);
  }
#endif
  return result;
}
#endif
