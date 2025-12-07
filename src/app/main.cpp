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

#include <array>
#include <chrono>
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

#include "physics/collision.hpp"
#include "physics/hertz_mindlin.hpp"
#include "physics/integrator.hpp"
#include "physics/mujoco_contact.hpp"
#include "physics/rigid_body.hpp"
#include "physics/soft_contact.hpp"

#ifndef HEADLESS_BUILD
#include "gfx/camera.hpp"
#include "gfx/mesh.hpp"
#include "gfx/renderer.hpp"
#endif

#include "config/config.hpp"

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
  App() = default;
  ~App() = default;

  int run();
  // Playback a snapshots NDJSON file (no physics) with optional frame dumping
  int runPlayback(const std::string &ndjsonPath, const std::string &dumpDir,
                  int playbackFps, bool orbit, float orbitSpeed, bool camPosSet,
                  const glm::vec3 &camPos, bool camTargetSet,
                  const glm::vec3 &camTarget, bool autoFrame, float scale,
                  bool skipDupes, bool hideWindow, bool noFloor);
  void setConfig(const AppCfg &config);
  void setProfiling(bool enabled) {
    profilingEnabled = enabled;
    std::cerr << "[Debug] setProfiling: " << enabled << "\n";
  }
  void setInitCsvPath(const std::string &path) { initCsvPath = path; }
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
    if (perRodEnabled && perRodMaxFrames > 0)
      perRodSkip = std::max(1, headlessSteps / perRodMaxFrames);
  }
  // Render stride control
  void setRenderStride(int s) { renderStride = std::max(1, s); }
  // CSV stride control
  void setCsvStride(int s) { csvStride = std::max(1, s); }
  // Enable per-rod CSV output (path, maximum sampled frames)
  void enablePerRod(const std::string &path, int maxFrames);

  void enableAdaptiveSubsteps(bool on) { adaptiveSubsteps = on; }
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
  GLFWwindow *window = nullptr;
  bool vsync = true;
#endif
  bool headless = false;
  int headlessSteps = 1000;

  int renderStride = 1;
  int csvStride = 1;

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
  glm::vec3 gravity{0.0f, -10.0f, 0.0f};
  float dt = 1.0f / 600.0f;
  AppCfg settings{};

  SoftContactSolver softContactSolver{};
  MujocoContactSolver mjContactSolver{};
  HertzMindlinSolver hertzMindlinSolver{};

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
  std::string initStateCsvPath;

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
  // When true, selected logs (CSV, reldisp) are emitted only on snapshot frames
  bool logOnSnapshotOnly = false;
  inline bool shouldLogThisFrame() const {
    if (!logOnSnapshotOnly)
      return true;
    if (!snapshotEnabled || snapStride <= 0)
      return true; // fallback
    if (frameIndex < (uint64_t)snapStartFrame)
      return false;
    return ((frameIndex - snapStartFrame) % snapStride) == 0 &&
           snapshotCount < snapFrames;
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

private:
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
  void logNetworkFrame(); // Will detect contact mode automatically

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
  // Invoke optional logs after a frame is fully updated
  void logOptionalFrames() {
    logCsvFrame();
    logSoftPEFrame();
    logCOMFrame();
    logNetworkFrame();
    logPerRodFrame();
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
  float sleepLinThresh = 0.02f;  // m/s
  float sleepAngThresh = 0.05f;  // rad/s
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

  std::cout << "[Debug] Calling softContactSolver.setPBC..." << std::endl;
  // Pass PBC settings to soft contact solver
  softContactSolver.setPBC(usePBC, pbcMin, pbcMax);
  std::cout << "[Debug] setPBC done." << std::endl;

  // Random initialization for PBC study
  const bool useRandomInit = usePBC && settings.scene.randomInit.enabled;
  if (useRandomInit) {
    std::cout << "[Debug] Using random init..." << std::endl;
    gravity = glm::vec3(0.0f);
  } else {
    std::cout << "[Debug] Populating scene..." << std::endl;
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
  glm::quat qF(floorConfig.rot_quat.x, floorConfig.rot_quat.y,
               floorConfig.rot_quat.z, floorConfig.rot_quat.w);
  floorRB = RigidBody::makeStaticFloor(
      floorConfig.pos, qF, floorConfig.half_extents.x,
      floorConfig.half_extents.y, floorConfig.half_extents.z,
      floorConfig.restitution, floorConfig.friction);

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
      std::cerr << "[init-csv] Loaded initial configuration from "
                << initCsvPath << " (rods=" << rods.size() << ")\n";
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
      auto linIdx = [&](int ix, int iy, int iz) {
        return ix + n.x * (iy + n.y * iz);
      };
      auto wrapI = [&](int a, int dim) {
        int res = a % dim;
        if (res < 0)
          res += dim;
        return res;
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
      auto segseg_dist2 = [&](const glm::vec3 &p0, const glm::vec3 &p1,
                              const glm::vec3 &q0, const glm::vec3 &q1) {
        // robust segment-segment distance (no PBC inside; caller applies
        // min-image via shifting q by centroid-based shift)
        glm::vec3 u = p1 - p0;
        glm::vec3 v = q1 - q0;
        glm::vec3 w0 = p0 - q0;
        float uu = glm::dot(u, u), vv = glm::dot(v, v), uv = glm::dot(u, v);
        float wu = glm::dot(w0, u), wv = glm::dot(w0, v);
        float D = uu * vv - uv * uv;
        float s, t;
        const float eps = 1e-12f;
        if (std::abs(D) < eps) {
          s = 0.0f;
          t = (vv >= eps) ? (-wv / vv) : 0.0f;
        } else {
          s = (uv * wv - vv * wu) / D;
          t = (uu * wv - uv * wu) / D;
        }
        s = glm::clamp(s, 0.0f, 1.0f);
        t = (s * uv + wv) / (vv >= eps ? vv : 1.0f);
        t = glm::clamp(t, 0.0f, 1.0f);
        float su = (-wu + t * uv) / (uu >= eps ? uu : 1.0f);
        if (!(t > 1e-6f && t < 1.0f - 1e-6f)) {
          if (su < 0.0f)
            s = 0.0f;
          else if (su > 1.0f)
            s = su;
        }
        glm::vec3 d = (w0 + s * u) - t * v;
        return glm::dot(d, d);
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
                  // Shift previous rod to minimum image wrt this centroid
                  // (approx)
                  glm::vec3 cj = C[j];
                  glm::vec3 uj = U[j];
                  glm::vec3 cj_img = cj;
                  if (usePBC) {
                    glm::vec3 d = cj - c;
                    // Wrap d to [-L/2, L/2]
                    for (int k = 0; k < 3; ++k) {
                      const float L = boxSize[k];
                      if (L > 0.0f)
                        d[k] -= L * std::floor(d[k] / L + 0.5f);
                    }
                    cj_img = c + d;
                  }
                  glm::vec3 q0 = cj_img - uj * halfL;
                  glm::vec3 q1 = cj_img + uj * halfL;
                  float d2 = segseg_dist2(p0, p1, q0, q1);
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

    if (useRandomInit) {
      // Gaussian translational velocities, Uniform S2 direction with fixed
      // magnitude for angular
      std::random_device rd;
      std::mt19937 gen(settings.scene.randomInit.seed
                           ? settings.scene.randomInit.seed
                           : rd());
      std::uniform_real_distribution<float> uniform(
          -settings.scene.randomInit.vSigma, settings.scene.randomInit.vSigma);
      std::uniform_real_distribution<float> uni(0.0f, 1.0f);
      const float wSpeed = settings.scene.randomInit.wSpeed;

      auto uniform_dir_s2 = [&](std::mt19937 &g) {
        float u = 2.0f * uni(g) - 1.0f; // cos(theta) in [-1,1]
        float phi = 2.0f * float(M_PI) * uni(g);
        float s = std::sqrt(std::max(0.0f, 1.0f - u * u));
        return glm::vec3(s * std::cos(phi), u, s * std::sin(phi));
      };

      for (auto &rb : rods) {
        rb.v = {uniform(gen), uniform(gen), uniform(gen)};
        rb.w = wSpeed * uniform_dir_s2(gen);
      }
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

  // init sleeping arrays
  sleeping.assign(rods.size(), 0);
  sleepTimer.assign(rods.size(), 0.f);

  // Auto-tuning: for small N, switch to serial execution if user didn't force
  // parallel
  static bool autoSerialMode = false;
  if (g_thread_limit == 0 || autoSerialMode) {
    if (rods.size() > 0 && rods.size() < 256) {
      if (g_thread_limit != 1) {
        std::cout
            << "[App] Auto-switching to serial mode (threads=1) for small N="
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

  // Reset KE history for adaptive decisions
  lastKE = totalKE();
  prevFrameKE = lastKE;
  lastFrameKEDelta = 0.0;
}

// Load initial configuration from CSV with endpoints per rod:
// x0,y0,z0,x1,y1,z1
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

  // Parse optional metadata headers starting with '#'
  std::string line;
  bool sawHeader = false;
  size_t lineCount = 0;
  size_t dataRows = 0;
  size_t skippedMalformed = 0;
  while (std::getline(in, line)) {
    ++lineCount;
    if (line.empty())
      continue;
    if (line[0] == '#') {
      // Try to parse helpful overrides (metadata is authoritative over
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
          else if (key == "rod_diameter")
            defaultDiameter = std::stof(val);
          else if (key == "pbc") {
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
    // First non-# line expected to be the CSV header
    // Accept exactly: x0,y0,z0,x1,y1,z1 (comma-separated)
    // If it's not that, try to proceed anyway if it contains x0 and z1.
    if (!sawHeader) {
      sawHeader = true;
      // Just move on; next lines are data rows.
      continue;
    }
    // Data row
    std::stringstream ss(line);
    std::string tok;
    std::vector<double> vals;
    while (std::getline(ss, tok, ',')) {
      if (!tok.empty()) {
        try {
          vals.push_back(std::stod(tok));
        } catch (...) { /* skip */
        }
      }
    }
    if (vals.size() < 6) {
      ++skippedMalformed;
      continue;
    }
    glm::vec3 a{float(vals[0]), float(vals[1]), float(vals[2])};
    glm::vec3 b{float(vals[3]), float(vals[4]), float(vals[5])};
    // Derive center and orientation from endpoints
    glm::vec3 c = 0.5f * (a + b);
    glm::vec3 u = glm::normalize(b - a);
    float Lseg = glm::length(b - a);
    float L = (Lseg > 0.0f ? Lseg : defaultLength);
    float D = defaultDiameter;
    // Build quaternion that rotates +Y to direction u
    glm::vec3 y(0, 1, 0);
    float d = glm::clamp(glm::dot(y, u), -1.0f, 1.0f);
    float ang = std::acos(d);
    glm::quat q;
    if (ang < 1e-6f) {
      q = glm::quat(1, 0, 0, 0);
    } else if (std::abs(d + 1.0f) < 1e-6f) {
      // 180-deg: rotate around any axis orthogonal to Y, use X
      q = glm::angleAxis(float(M_PI), glm::vec3(1, 0, 0));
    } else {
      glm::vec3 axis = glm::normalize(glm::cross(y, u));
      q = glm::angleAxis(ang, axis);
    }
    BodyCfg cfg{};
    cfg.shape = "capsule";
    cfg.pos = c;
    cfg.rot_quat = glm::vec4(q.w, q.x, q.y, q.z);
    cfg.length = L;
    cfg.diameter = D;
    cfg.density = defaultDensity;
    cfg.restitution = defaultRestitution;
    cfg.friction = defaultFriction;
    rods.push_back(createRod(cfg));
    ++dataRows;
  }
  std::cerr << "[init-csv] Parsed rows=" << dataRows
            << " (malformed=" << skippedMalformed
            << ") header=" << (sawHeader ? "yes" : "no")
            << " fileLines=" << lineCount << "\n";
  if (rods.empty()) {
    std::cerr << "[init-csv] No rods created (check CSV format: expect header "
                 "x0,y0,z0,x1,y1,z1).\n";
  }
  return !rods.empty();
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
    std::cerr << "[Debug] profilingEnabled=" << profilingEnabled << " softOrHM="
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
                   "sum\n";
    } else {
      csvStream << "frame,rods,integrate_ms,sleep_ms,broadphase_ms,bpCount_ms,"
                   "bpPrefix_ms,bpFill_ms,bpPairs_ms,bpLongLong_ms,warmstart_"
                   "ms,buildIslands_ms,solve_ms,floorSolve_ms,posCorrect_ms,"
                   "pbcWrap_ms,render_ms,contacts,islands,KE,KE_after_"
                   "integrate,KE_after_warmstart,KE_after_solve,KE_after_"
                   "posCorrect,KE_after_pbcWrap,soft_PE,gyration_sq,reldisp_sq,"
                   "jn_sum,jt_sum,impulse_count,ent_pairs,ent_sum\n";
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
              << '\n';
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
              << gyr_sq << ',' << reldisp_sq << ',' << 0.0 << ',' << 0.0 << ','
              << 0 << ',' << lastEntanglementPairs << ',' << lastEntanglementSum
              << '\n';
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
  // known)
  perRodSkip = 1;
  if (headless && headlessSteps > 0)
    perRodSkip = std::max(1, headlessSteps / perRodMaxFrames);
}

void App::logPerRodFrame() {
  if (!perRodEnabled || !perRodStream)
    return;
  if (!perRodHeaderWritten) {
    perRodStream << "frame,rod,px,py,pz,vx,vy,vz,wx,wy,wz,qw,qx,qy,qz,KE_lin,"
                    "KE_rot,KE_total\n";
    perRodHeaderWritten = true;
  }
  if (perRodWrittenFrames >= perRodMaxFrames)
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
  if (!networkEnabled || !networkStream)
    return;
  if (!networkHeaderWritten) {
    networkStream
        << "frame,rod_i,rod_j,contact_x,contact_y,contact_z,normal_x,"
           "normal_"
           "y,"
           "normal_z,distance,"
        << "force_a_x,force_a_y,force_a_z,force_b_x,force_b_y,force_b_z,"
        << "friction_a_x,friction_a_y,friction_a_z,friction_b_x,friction_"
           "b.y,"
           "friction_b.z\n";
    networkHeaderWritten = true;
  }

  // Handle both soft and hard contact modes
  if (settings.physics.soft_contact.enabled) {
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
      }
    }
  } else {
    // Hard contacts - use hitsScratch from broadphase (no force data
    // available)
    for (const auto &hit : hitsScratch) {
      if (hit.b < 0)
        continue; // Skip floor contacts

      networkStream
          << frameIndex << ',' << hit.a << ',' << hit.b << ',' << hit.c.point.x
          << ',' << hit.c.point.y << ',' << hit.c.point.z << ','
          << hit.c.normal.x << ',' << hit.c.normal.y << ',' << hit.c.normal.z
          << ',' << -hit.c.penetration << ','
          << "0,0,0,0,0,0,0,0,0,0,0,0\n"; // Placeholder zeros for forces
    }
  }

  if ((frameIndex & 0x3F) == 0)
    networkStream.flush();
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

// ---- Simulation ----

void App::logOutputFrame() {
  if (!outputEnabled || !outputStream)
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
#ifdef TRACY_ENABLE
  ZoneScopedN("PhysicsStep");
#endif
  // Reset diagnostic accumulators before this step

  // Apply random forces if enabled
  if (useRandomForce) {
    // Match World::applyRandomForces semantics: random direction *
    // (fSigma * N(0,1))
    for (auto &rb : rods) {
      glm::vec3 dirF = uniform_dir_s2(genRandomForce);
      float magF = fSigma * normal_f(genRandomForce);
      rb.f += dirF * magF;
      if (tauMag > 0.0f) {
        glm::vec3 dirT = uniform_dir_s2(genRandomForce);
        rb.tau += dirT * tauMag; // torque magnitude fixed
      }
    }
  }

  if (settings.physics.hertz_mindlin.enabled) {
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
      for (size_t i = 0; i < rods.size(); ++i) {
        if (!sleeping[i])
          integrateHalfPos(rods[i], gravity, dt);
      }
    }
// Clear forces before recompute at t+dt
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rods.size(); ++i) {
      rods[i].f = glm::vec3(0);
      rods[i].tau = glm::vec3(0);
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
    // Re-inject random forces for second half-step so they act over full
    // dt (contact forces already accumulated).
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
    {
#ifdef TRACY_ENABLE
      ZoneScopedN("IntegrateSecondHalf");
#endif
      ScopedAccum tIntegrateHV(profilingEnabled ? &curTimes.integrate
                                                : nullptr);
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < rods.size(); ++i) {
        if (!sleeping[i])
          integrateSecondHalf(rods[i], gravity, dt);
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
        softContactSolver.computeForces(rods, dt);
      }
      lastSoftPotentialEnergy =
          softContactSolver.getLastPotentialEnergy(); // PE at configuration t
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
      for (size_t i = 0; i < rods.size(); ++i) {
        if (!sleeping[i])
          integrateHalfPos(rods[i], gravity, dt);
      }
    }
// Clear forces before recompute at t+dt
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rods.size(); ++i) {
      rods[i].f = glm::vec3(0);
      rods[i].tau = glm::vec3(0);
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
        softContactSolver.computeForces(rods, dt);
      }
      lastSoftPotentialEnergy =
          softContactSolver.getLastPotentialEnergy(); // overwrite with PE at
                                                      // configuration t+dt
      lastHitCount =
          softContactSolver
              .getNumContacts(); // Update contact count for CSV logging
    }
    // if (settings.physics.soft_contact.verbose && frameIndex % 200 == 0)
    // {
    //     std::cout << "[Verlet] frame=" << frameIndex << "
    //     contacts(t+dt)="
    //     << softContactSolver.getNumContacts() << '\n';
    // }
    // 4) second half velocity update
    {
#ifdef TRACY_ENABLE
      ZoneScopedN("IntegrateSecondHalf");
#endif
      ScopedAccum tIntegrateSH(profilingEnabled ? &curTimes.integrate
                                                : nullptr);
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < rods.size(); ++i) {
        if (!sleeping[i])
          integrateSecondHalf(rods[i], gravity, dt);
      }
    }
    // KE after full Verlet integrate
    keAfterIntegrate = totalKE();
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
  // Optionally compute entanglement every N frames
  if (entanglementEnabled && (frameIndex % entanglementEvery == 0)) {
    computeEntanglement();
  }
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
  };
  constexpr int numColors = sizeof(rodColors) / sizeof(rodColors[0]);

  // Draw rods: use instancing beyond a small threshold to reduce draw
  // calls
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
      RenderUniforms common = uniforms;
      common.useGrid = false;
      rnd.drawInstances(cyl, capsuleModels.data(), capsuleColors.data(),
                        capsuleModels.size(), common);
    }

    // Draw spheres
    if (!sphereModels.empty()) {
      RenderUniforms common = uniforms;
      common.useGrid = false;
      rnd.drawInstances(sphere, sphereModels.data(), sphereColors.data(),
                        sphereModels.size(), common);
    }
  } else {
    for (size_t i = 0; i < N; ++i) {
      uniforms.M = rods[i].modelMatrix();
      uniforms.color = rodColors[i % numColors];
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
  std::cout << "Running headless for " << headlessSteps << " steps..."
            << std::endl;
  for (int step = 0; step < headlessSteps; ++step) {
    if (!paused) {
      if (step % 100 == 0)
        std::cout << "[Headless] Step " << step << " begin" << std::endl;
      stepWithSubsteps();
      if (step % 100 == 0)
        std::cout << "[Headless] Step " << step << " end" << std::endl;
    }
    if (entanglementEnabled && (frameIndex % entanglementEvery == 0)) {
      computeEntanglement();
    }
    if (perRodEnabled)
      logPerRodFrame();
    // CSV logging if enabled
    if (frameIndex % csvStride == 0) {
      logCsvFrame();
      logOutputFrame();
    }
    // Snapshot capture (HEADLESS_BUILD)
    if (snapshotEnabled && snapStride > 0 &&
        frameIndex >= (uint64_t)snapStartFrame &&
        ((frameIndex - snapStartFrame) % snapStride) == 0 &&
        snapshotCount < snapFrames) {
      writeSnapshotLine();
    }
    // Accumulate profiling then reset per-frame timers for accurate
    // per-frame CSV
    if (profilingEnabled) {
      sumTimes += curTimes;
      curTimes.reset();
    }
    ++frameIndex;
    if ((step & 0x3FF) == 0) {
      printCliStatus("[Headless] ");
    }
  }
  if (csvEnabled)
    csvStream.flush();
  if (perRodEnabled)
    perRodStream.flush();
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
    std::cout << "[Info] OpenMP enabled. Max threads: " << omp_get_max_threads()
              << "\n";
#else
    std::cout << "[Info] OpenMP NOT enabled.\n";
#endif
    std::cout << "Running headless for " << headlessSteps << " steps...\n";
    for (int step = 0; step < headlessSteps; ++step) {
      if (!paused)
        stepWithSubsteps();
      if (entanglementEnabled && (frameIndex % entanglementEvery == 0)) {
        computeEntanglement();
      }
      if (perRodEnabled)
        logPerRodFrame();
      // CSV logging if enabled
      logCsvFrame();
      logOutputFrame();
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
      ++frameIndex;
      if ((step & 0x3FF) == 0) {
        printCliStatus("[Headless] ");
      }
    }
    // ensure CSV flushed and closed
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

  lastTitleUpdate = std::chrono::high_resolution_clock::now();

  auto lastTime = std::chrono::high_resolution_clock::now();
  double accumulator = 0.0;
#ifdef TRACY_ENABLE
  tracy::SetThreadName("Main");
#endif

  while (!glfwWindowShouldClose(window)) {
    if (renderStride > 1) {
      // Fast-forward mode: Run 'renderStride' physics steps, then render
      // once. This decouples simulation speed from wall-clock time and
      // aligns 'frameIndex' with physics steps.
      for (int i = 0; i < renderStride; ++i) {
        if (!paused) {
          stepWithSubsteps();
          if (perRodEnabled)
            logPerRodFrame();
          logCsvFrame();
          logOutputFrame();
          ++frameIndex;
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
        if (!paused) {
          stepWithSubsteps();
        }
        accumulator -= dt;
      }

      renderFrame();
      if (perRodEnabled)
        logPerRodFrame();
      // CSV logging uses current per-frame times before they are reset by
      // maybeUpdateWindowTitle
      logCsvFrame();
      logOutputFrame();
      maybeUpdateWindowTitle();
      glfwSwapBuffers(window);
      glfwPollEvents();
      ++frameIndex;
    }
#ifdef TRACY_ENABLE
    FrameMark;
#endif
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
  glfwTerminate();
  return 0;
#endif
}

#ifndef HEADLESS_BUILD
int App::runPlayback(const std::string &ndjsonPath, const std::string &dumpDir,
                     int playbackFps, bool orbit, float orbitSpeed,
                     bool camPosSet, const glm::vec3 &camPos, bool camTargetSet,
                     const glm::vec3 &camTarget, bool autoFrame, float scale,
                     bool skipDupes, bool hideWindow, bool noFloor) {
  // Initialize minimal window/renderer
  if (!initWindow())
    return -1;
  if (!initGraphics())
    return -1;
  if (hideWindow) {
    glfwHideWindow(window); // Hide the playback window for frames-only mode
  }
  if (noFloor) {
    disableFloorRender = true; // suppress floor rendering for playback clarity
  }
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
  // Load all lines from NDJSON
  std::vector<std::string> lines;
  {
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
      cam.dist = dist;
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
  const double targetDt = (playbackFps > 0) ? 1.0 / double(playbackFps) : 0.0;
  auto lastFrameTime = std::chrono::high_resolution_clock::now();
  std::cerr << "[playback] Frames=" << lines.size()
            << (dumpDir.empty() ? "" : " dumping enabled") << "\n";
  std::string prevLine;
  for (size_t fi = 0; fi < lines.size() && !glfwWindowShouldClose(window);
       ++fi) {
    if (playbackFps > 0) {
      while (true) {
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed =
            std::chrono::duration<double>(now - lastFrameTime).count();
        if (elapsed >= targetDt) {
          lastFrameTime = now;
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
    const std::string &rawLine = lines[fi];
    if (skipDupes && fi > 0 && rawLine == prevLine) {
      // Still advance render/orbit for continuity but skip PNG write
      if (orbit)
        cam.yaw += orbitSpeed * 0.01f;
      renderFrame();
      glfwSwapBuffers(window);
      glfwPollEvents();
      continue;
    }
    prevLine = rawLine;
    nlohmann::json j = nlohmann::json::parse(rawLine, nullptr, false);
    if (j.is_discarded()) {
      std::cerr << "[playback] JSON parse error frame=" << fi << "\n";
      continue;
    }
    rods.clear();
    if (j.contains("bodies") && j["bodies"].is_array()) {
      for (const auto &jb : j["bodies"]) {
        std::string shape = jb.value("shape", "sphere");
        if (shape == "sphere") {
          auto pos = jb["pos"];
          float r = jb.value("radius", 0.05f);
          float density = 1000.0f;
          RigidBody rb = RigidBody::makeSphere(
              glm::vec3(pos[0], pos[1], pos[2]), density, r, 0.3f, 0.3f);
          rods.push_back(rb);
        } else if (shape == "capsule") {
          auto pos = jb["pos"];
          auto quat = jb["quat"];
          float r = jb.value("radius", 0.05f);
          float h = jb.value("halfHeight", 0.1f);
          float density = 1000.0f;
          glm::quat q(quat[0], quat[1], quat[2], quat[3]);
          RigidBody rb = RigidBody::makeCapsule(
              glm::vec3(pos[0], pos[1], pos[2]), q, density, r, h, 0.3f, 0.3f);
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
    // Simple orbit: rotate camera eye around center keeping dist
    if (orbit) {
      cam.yaw += orbitSpeed * 0.01f; // incremental yaw shift
    }
    renderFrame();
    // Ensure rendering finished before pixel read
    glFinish();
    if (!dumpDir.empty()) {
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
      // Render frame number at top-left
      std::string label = std::to_string(fi);
      int cursor = 4;
      for (char c : label) {
        drawDigit(c - '0', cursor, 4);
        cursor += 8;
      }
      std::vector<unsigned char> png;
      lodepng::encode(png, flipped.data(), (unsigned)outW, (unsigned)outH);
      char name[256];
      std::snprintf(name, sizeof(name), "%s/frame_%05zu.png", dumpDir.c_str(),
                    fi);
      lodepng::save_file(png, name);
    }
    glfwSwapBuffers(window);
    glfwPollEvents();
  }
  if (!dumpDir.empty()) {
    std::cerr << "[playback] Dump complete: " << dumpDir << "\n";
    std::cerr << "[playback] Example ffmpeg: ffmpeg -framerate "
              << (playbackFps > 0 ? playbackFps : 60) << " -i " << dumpDir
              << "/frame_%05d.png -c:v libx264 -pix_fmt yuv420p movie.mp4\n";
  }
  glfwTerminate();
  return 0;
}
#endif

void App::setConfig(const AppCfg &config) {
  settings = config;
  // If scene specifies an initial CSV, configure it here so resetScene
  // will load it.
  if (!settings.scene.initCsvPath.empty()) {
    initCsvPath = settings.scene.initCsvPath;
  }
}

void App::printCliStatus(const std::string &prefix) const {
  if (!CLI_UNIFIED_PRINT)
    return;
  // frame, bodies, KE, ent_pairs, ent_sum
  std::cout << prefix << "frame=" << frameIndex << " bodies=" << rods.size()
            << " KE=" << std::fixed << std::setprecision(6) << lastKE
            << " ent_pairs=" << lastEntanglementPairs
            << " ent_sum=" << std::fixed << std::setprecision(6)
            << lastEntanglementSum << std::defaultfloat << "\n";
}

// ---- Main Function ----

int main(int argc, char **argv) {
  std::string scenePath = std::string(ASSETS_DIR) + "/scenes/default.json";
  bool enableProfile = false;
  std::string csvPath;
  bool headlessFlag = false;
  int headlessSteps = 1000;
  std::string perRodPath;
  int perRodMaxFrames = 1000;

  // CLI overrides

  int cliSeed = 0; // 0 means no override

  int cliThreads = -1;
  float cliDt = -1.0f;
  std::string cliContactDumpPath;

  std::string cliContactDumpTrig;
  std::string cliSoftPEPath;   // optional soft potential energy output file
  std::string cliCOMPath;      // center-of-mass tracking
  std::string cliNetworkPath;  // contact network tracking
  std::string cliOutputPath;   // compact output CSV
  bool cliDebugMinGap = false; // enable minPairGap debug printing
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
  int cliPlaybackFps = 0;       // 0 => fastest
  bool cliOrbit = false;
  float cliOrbitSpeed = 0.5f;
  bool cliCamPosSet = false;
  glm::vec3 cliCamPos(0.0f);
  bool cliCamTargetSet = false;
  glm::vec3 cliCamTarget(0.0f);
  bool cliAutoFrame = false;
  float cliScale = 1.0f;
  bool cliSkipDupes = false;
  bool cliFramesOnly = false;      // hide window during playback frame dumping
  bool cliNoFloor = false;         // disable floor rendering in playback
  std::string cliInitCsvPath;      // initial configuration CSV (segments)
  std::string cliInitStateCsvPath; // initial state CSV (per-rod format)
  std::string cliRelDispPath;      // relative displacement CSV
  int cliRods = -1;                // Override rod count

  int cliRenderStride = 1;    // Render every N frames
  int cliCsvStride = 1;       // Log CSV every N frames
  bool cliAutoReplay = false; // Automatically replay after headless run

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
      std::cout << "Rod Dynamics 3D - Rigid Body Simulation\n\n";
      std::cout << "Usage: " << argv[0] << " [options]\n\n";
      std::cout << "Scene Configuration:\n";
      std::cout << "  --scene <path>              Load scene from JSON file "
                   "(default: assets/scenes/default.json)\n\n";
      std::cout << "Execution Modes:\n";
      std::cout << "  --headless                  Run without graphics\n";
      std::cout << "  --steps <N>                 Number of steps for headless "
                   "mode (default: 1000)\n";
      std::cout << "  --render-stride <N>         Render every N frames "
                   "(default: 1)\n\n";
      std::cout << "Output & Logging:\n";
      std::cout << "  --csv [path]                Enable CSV profile output "
                   "(default: profile.csv)\n";
      std::cout << "  --csv-stride <N>            Log CSV every N frames "
                   "(default: 1)\n";
      std::cout << "  --perrod [path]             Enable per-rod trajectory "
                   "CSV (default: perrod.csv)\n";
      std::cout << "  --perrod-max <N>            Max frames to log in per-rod "
                   "CSV (default: 1000)\n";
      std::cout << "  --soft-pe <path>            Log soft contact potential "
                   "energy\n";
      std::cout << "  --com <path>                Track center-of-mass "
                   "(default: com.csv)\n";
      std::cout << "  --network <path>            Track contact network "
                   "(default: network.csv)\n";
      std::cout << "  --contact-dump <path>       Log contact details to CSV\n";
      std::cout << "  --contact-dump-thresh <T>   KE threshold for "
                   "contact dump\n";
      std::cout << "  --contact-dump-trigger <M>  Trigger mode: "
                   "any|up|down\n\n";
      std::cout << "Solver Configuration:\n";
      std::cout << "  --dt <float>                Timestep size\n";
      std::cout << "  --substeps <N>              Substeps per frame\n";
      std::cout << "  --velIters <N>              Velocity solver iterations\n";
      std::cout << "  --ngs-sweeps <N>            NGS sweeps for constraint "
                   "solver\n";
      std::cout << "  --ngs-vth <float>           NGS velocity threshold\n";
      std::cout << "  --split-impulse             Enable split impulse\n";
      std::cout << "  --no-split-impulse          Disable split impulse\n";
      std::cout << "  --split-orient              Enable split orientation "
                   "correction\n";
      std::cout << "  --no-split-orient           Disable split orientation "
                   "correction\n";
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
      std::cout << "  --scale <f>                 Downsample (f<1) or upsample "
                   "(f>1) before PNG write\n";
      std::cout << "  --skip-dupes                Skip writing PNG for "
                   "duplicate consecutive snapshots\n";
      std::cout << "  --frames-only               Hide window (offscreen) "
                   "while dumping playback frames\n";
      std::cout << "  --no-floor                  Disable floor rendering in "
                   "playback\n";
      std::cout << "\nInitial Configuration:\n";
      std::cout << "  --init-csv <path>          Load initial rods from CSV "
                   "with endpoints (x0..z1)\n";
      std::cout << "\nDiagnostics / Metrics:\n";
      std::cout << "  --com <path>                Track center-of-mass "
                   "(default: com.csv)\n";
      std::cout << "  --debug-com                 Track center-of-mass to "
                   "com_debug.csv\n";
      std::cout << "  --reldisp <path>           Track ri-rc and L2 norm "
                   "(default: reldisp.csv)\n";
      std::cout << "  --help, -h                  Show this help message\n\n";
      std::cout << "Examples:\n";
      std::cout << "  " << argv[0] << " --scene my_scene.json\n";
      std::cout << "  " << argv[0]
                << " --headless --steps 10000 --csv --perrod\n";
      std::cout << "  " << argv[0]
                << " --scene confined.json --dt 0.001 --velIters 20\n";
      return 0;
    } else if (std::string(argv[i]) == "--scene" && i + 1 < argc) {
      scenePath = argv[++i];
    } else if (std::string(argv[i]) == "--profile") {
      enableProfile = true;
    } else if (std::string(argv[i]) == "--csv") {
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        csvPath = argv[++i];
      } else {
        csvPath = "profile.csv";
      }
    } else if (std::string(argv[i]) == "--headless") {
      headlessFlag = true;
    } else if (std::string(argv[i]) == "--steps" && i + 1 < argc) {
      headlessSteps = std::stoi(argv[++i]);
    } else if (std::string(argv[i]) == "--render-stride" && i + 1 < argc) {
      cliRenderStride = std::max(1, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--csv-stride" && i + 1 < argc) {
      cliCsvStride = std::max(1, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--auto-replay") {
      cliAutoReplay = true;
    } else if (std::string(argv[i]) == "--perrod") {
      if (i + 1 < argc && argv[i + 1][0] != '-')
        perRodPath = argv[++i];
      else
        perRodPath = "perrod.csv";
    } else if (std::string(argv[i]) == "--perrod-max" && i + 1 < argc) {
      perRodMaxFrames = std::max(1, std::stoi(argv[++i]));

    } else if (std::string(argv[i]) == "--seed" && i + 1 < argc) {
      cliSeed = std::stoi(argv[++i]);
    } else if (std::string(argv[i]) == "--rods" && i + 1 < argc) {
      cliRods = std::max(1, std::stoi(argv[++i]));

    } else if (std::string(argv[i]) == "--threads" && i + 1 < argc) {
      cliThreads = std::max(0, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--dt" && i + 1 < argc) {
      cliDt = std::max(0.0f, std::stof(argv[++i]));
    } else if (std::string(argv[i]) == "--contact-dump" && i + 1 < argc) {
      cliContactDumpPath = argv[++i];

    } else if (std::string(argv[i]) == "--contact-dump-trigger" &&
               i + 1 < argc) {
      cliContactDumpTrig = argv[++i]; // any|up|down
    } else if (std::string(argv[i]) == "--soft-pe" && i + 1 < argc) {
      cliSoftPEPath = argv[++i];
    } else if (std::string(argv[i]) == "--com" && i + 1 < argc) {
      cliCOMPath = argv[++i];
    } else if (std::string(argv[i]) == "--debug-com") {
      cliCOMPath = "com_debug.csv";
    } else if (std::string(argv[i]) == "--network" && i + 1 < argc) {
      cliNetworkPath = argv[++i];
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
    } else if (std::string(argv[i]) == "--beta-min" && i + 1 < argc) {
      cliBetaMin = std::stof(argv[++i]);
    } else if (std::string(argv[i]) == "--beta-hit" && i + 1 < argc) {
      cliBetaHit = std::max(0, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--beta-scale" && i + 1 < argc) {
      cliBetaScale = std::stof(argv[++i]);
    } else if (std::string(argv[i]) == "--entanglement") {
      cliEntanglement = true;
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
    } else if (std::string(argv[i]) == "--init-state-csv" && i + 1 < argc) {
      cliInitStateCsvPath = argv[++i];
    } else if (std::string(argv[i]) == "--reldisp") {
      if (i + 1 < argc && argv[i + 1][0] != '-')
        cliRelDispPath = argv[++i];
      else
        cliRelDispPath = "reldisp.csv";
    } else if (std::string(argv[i]) == "--debug-min-gap") {
      cliDebugMinGap = true;
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
    }
  }

  AppCfg settings = defaultAppCfg();

  // Load scene configuration (keep defaults if load fails)
  if (!loadConfigFromFile(scenePath, settings)) {
    std::cerr << "Warning: Could not load scene file '" << scenePath
              << "', using defaults.\n";
  }

  // Apply CLI overrides to settings
  if (cliRods > 0)
    settings.scene.populate.count = cliRods;

  if (cliSeed != 0) {
    settings.scene.populate.seed = cliSeed;
    settings.scene.randomInit.seed = cliSeed;
  }

  if (cliThreads >= 0)
    g_thread_limit = cliThreads;
  if (cliDt > 0.0f)
    settings.physics.dt = cliDt;
  if (cliSpatialHash != -1)
    settings.physics.soft_contact.use_spatial_hash = (cliSpatialHash != 0);
  if (cliUseAABB != -1)
    settings.physics.soft_contact.use_aabb = (cliUseAABB != 0);
  if (cliCellSize > 0.0f)
    settings.physics.soft_contact.cell_size = cliCellSize;
  if (cliVerboseSoft != -1)
    settings.physics.soft_contact.verbose = (cliVerboseSoft != 0);

  App app;
  app.setConfig(settings);
  app.setProfiling(enableProfile);
  if (cliDebugMinGap) {
    app.setDebugMinGap(true);
  }
  if (!csvPath.empty())
    app.enableCsv(csvPath);
  if (headlessFlag) {
    // Provide default csv path if none given
    if (csvPath.empty())
      app.enableCsv("profile_headless.csv");
    app.setHeadless(true);
    app.setHeadlessSteps(headlessSteps);
  }
  // Enable per-rod logging if requested (do after headless/steps set so
  // sampling skip can be computed)
  if (!perRodPath.empty()) {
    app.enablePerRod(perRodPath, perRodMaxFrames);
  }
  if (!cliSoftPEPath.empty()) {
    app.enableSoftPE(cliSoftPEPath);
    std::cerr << "[app] Soft contact potential energy logging enabled: "
              << cliSoftPEPath << "\n";
  }
  if (!cliCOMPath.empty()) {
    app.enableCOM(cliCOMPath);
    std::cerr << "[app] Center-of-mass tracking enabled: " << cliCOMPath
              << "\n";
  }
  if (!cliNetworkPath.empty()) {
    app.enableNetwork(cliNetworkPath);
    std::cerr << "[app] Contact network tracking enabled: " << cliNetworkPath
              << "\n";
  }
  if (!cliOutputPath.empty()) {
    app.enableOutput(cliOutputPath);
    std::cerr << "[app] Compact output logging enabled: " << cliOutputPath
              << "\n";
  }
  // Global toggles for solver diagnostics/testing

  if (cliThreads >= 0) {
    std::cerr << "[app] Thread limit set to " << cliThreads
              << " via --threads\n";
  }
  if (cliDt > 0.0f) {
    std::cerr << "[app] Timestep set to " << cliDt << " via --dt\n";
  }
  // Apply adaptive substeps and stabilization config to app
  if (cliAdaptive != -1) {
    app.enableAdaptiveSubsteps(cliAdaptive != 0);
    std::cerr << "[app] Adaptive substeps "
              << ((cliAdaptive != 0) ? "on" : "off") << "\n";
  }
  if (cliAsMin > 0 || cliAsMax > 0 || cliAsHit >= 0 || !std::isnan(cliAsKEUp) ||
      !std::isnan(cliAsKEDown)) {
    app.setAdaptiveParams(cliAsMin > 0 ? cliAsMin : 1,
                          cliAsMax > 0 ? cliAsMax
                          : 1          ? 1
                                       : 1,
                          cliAsHit >= 0 ? cliAsHit : INT32_MAX,
                          !std::isnan(cliAsKEUp) ? cliAsKEUp : 1e300,
                          !std::isnan(cliAsKEDown) ? cliAsKEDown : -1e300);
  }
  if (!std::isnan(cliBetaMin) || cliBetaHit >= 0 || !std::isnan(cliBetaScale)) {
    app.setStabilization(!std::isnan(cliBetaMin) ? cliBetaMin : 0.0f,
                         cliBetaHit >= 0 ? cliBetaHit : INT32_MAX,
                         !std::isnan(cliBetaScale) ? cliBetaScale : 1.0f);
    std::cerr << "[app] Stabilization configured\n";
  }
  // Apply entanglement config
  if (cliEntanglement) {
    app.setEntanglement(true, cliEntanglementCutoff, cliEntanglementPeriod,
                        cliEntanglementThreads);
    std::cerr << "[app] Entanglement enabled with cutoff="
              << cliEntanglementCutoff << ", period=" << cliEntanglementPeriod
              << ", threads=" << cliEntanglementThreads << "\n";
  }
  if (cliSnapStride > 0 && cliSnapFrames > 0) {
    app.enableSnapshots(cliSnapStride, cliSnapFrames, cliSnapPath,
                        cliSnapStart);
  }

  App a;
  a.setHeadless(headlessFlag);
  a.setHeadlessSteps(headlessSteps);
  a.setCsvStride(cliCsvStride);
  // Default: enable CSV profile (KE, contact count, timings)
  if (!csvPath.empty())
    a.enableCsv(csvPath);
  else
    a.enableCsv("profile.csv");
  if (!perRodPath.empty())
    a.enablePerRod(perRodPath, perRodMaxFrames);
  if (!cliSoftPEPath.empty())
    a.enableSoftPE(cliSoftPEPath);
  if (!cliCOMPath.empty())
    a.enableCOM(cliCOMPath);
  // Relative displacement CSV is optional; enable only if --reldisp is
  // provided
  if (!cliNetworkPath.empty())
    a.enableNetwork(cliNetworkPath);
  if (!cliOutputPath.empty())
    a.enableOutput(cliOutputPath);
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
  a.setConfig(settings);
  if (cliDebugMinGap) {
    a.setDebugMinGap(true);
    std::cerr << "[app] minPairGap debug enabled via --debug-min-gap\n";
  }
  if (cliCheckInitNonpenetration) {
    a.setCheckInitNonpenetration(true);
    std::cerr << "[app] Initial nonpenetration check enabled via "
                 "--check-init-nonpenetration\n";
  }
  if (!cliInitCsvPath.empty()) {
    a.setInitCsvPath(cliInitCsvPath);
    std::cerr << "[app] Initial CSV configured: " << cliInitCsvPath << "\n";
  }
  if (!cliInitStateCsvPath.empty()) {
    a.setInitStateCsvPath(cliInitStateCsvPath);
    std::cerr << "[app] Initial State CSV configured: " << cliInitStateCsvPath
              << "\n";
  }
  if (!cliRelDispPath.empty()) {
    a.enableRelDisp(cliRelDispPath);
    std::cerr << "[app] Relative displacement tracking enabled: "
              << cliRelDispPath << "\n";
  }
  if (cliSnapStride > 0 && cliSnapFrames > 0) {
    a.enableSnapshots(cliSnapStride, cliSnapFrames, cliSnapPath, cliSnapStart);
    a.setLogOnSnapshotOnly(true);
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
    std::cerr << "[app] Auto-replay enabled. Recording to " << cliSnapPath
              << "\n";
  }

  a.setRenderStride(cliRenderStride);
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
                         cliSkipDupes, cliFramesOnly, cliNoFloor);
  }
#endif
  int result = a.run();

#ifndef HEADLESS_BUILD
  if (cliAutoReplay && result == 0) {
    std::cerr << "[app] Headless run complete. Starting playback...\n";
    // Re-use the app instance? Or create a new one?
    // App state (rods, etc.) is modified. runPlayback clears rods and
    // reloads from JSON. So it should be safe to reuse 'a'. However, 'a'
    // might have other state. Safest is to use 'a' since runPlayback
    // seems to re-init everything it needs.
    return a.runPlayback(cliSnapPath, cliDumpFramesDir, cliPlaybackFps,
                         cliOrbit, cliOrbitSpeed, cliCamPosSet, cliCamPos,
                         cliCamTargetSet, cliCamTarget, cliAutoFrame, cliScale,
                         cliSkipDupes, cliFramesOnly, cliNoFloor);
  }
#endif
  return result;
}
