#pragma once
#include <glm/glm.hpp>
#include <string>
#include <vector>
#include "physics/types.hpp"   // <- use the existing SolverConfig


// struct SolverConfig { float baumgarte=0.25f, allowedPen=0.003f; int velIters=30; };

struct SoftContactCfg {
  bool enabled = false;          // Use soft contact instead of hard impulse solver
  double delta = 0.005;          // Transition width for piecewise potential
  double k_scaler = 1000.0;      // Contact stiffness multiplier
  double mu = 0.;               // Friction coefficient
  double nu = 1e-5;              // Sticking velocity threshold (m/s)
  bool enable_friction = true;   // Enable friction in soft contact
  bool verbose = false;          // Print contact debug info
};

struct PhysicsCfg {
  float dt = 1.0f / 600.0f;
  glm::vec3 gravity{0.0f,-10.0f,0.0f};
  float lin_damp = 0.08f;
  float ang_damp = 0.12f;
  float w_max    = 80.0f;
  int substeps = 1;            // Integrator/solver substeps per frame (<=0 => adaptive)
  SolverConfig solver{};
  SoftContactCfg soft_contact{}; // Soft penalty-based contact configuration
  // Optional alternative contact model inspired by MuJoCo. When enabled,
  // this uses MujocoContactSolver instead of SoftContactSolver for
  // soft penalty contacts. The two paths are kept separate for easy A/B tests.
  bool use_mujoco_contact = false;
};

struct GridCfg {
  bool enabled = true;
  float scale = 1.0f;
  glm::vec3 c1{0.80f,0.82f,0.85f};
  glm::vec3 c2{0.65f,0.67f,0.70f};
};

struct RenderCfg {
  // orbit camera
  float yaw = 0.6f, pitch = 0.35f, dist = 6.0f;
  glm::vec3 lightDir{-0.4f,-1.0f,-0.3f};
  glm::vec3 bg{0.08f,0.09f,0.11f};
  GridCfg grid{};
  bool vsync = true;
  bool cull = false;
  int msaa_samples = 4;
};

struct FloorCfg {
  glm::vec3 pos{0,-0.8f,0};
  glm::vec4 rot_quat{1,0,0,0}; // wxyz
  glm::vec3 half_extents{10.0f,0.1f,10.0f};
  float restitution = 0.3f;
  float friction    = 0.9f;
};

// Periodic boundary configuration
struct PeriodicCfg {
  bool enabled = false;              // Enable periodic boundaries instead of floor
  glm::vec3 min{-3.0f, -1.0f, -3.0f}; // Box minimum corner
  glm::vec3 max{+3.0f, +3.0f, +3.0f}; // Box maximum corner
  float cellSize = 0.6f;             // Broadphase grid cell size
  int   longSpan = 4;                // Threshold: rods spanning > this many cells on any axis are treated as long
};

// Random initialization configuration (for PBC studies)
struct RandomInitCfg {
  bool enabled = false;   // If true and periodic is enabled, set gravity=0 and assign random velocities
  float vSigma = 0.3f;    // Stddev for translational velocity normal distribution
  float wSpeed = 1.5f;    // Constant angular speed magnitude (direction uniform over S2)
  unsigned int seed = 0;  // Optional seed; 0 => random_device
};

// Random force injection configuration
struct RandomForceCfg {
  bool enabled = false;   // If true, apply random forces/torques each step
  float fSigma = 0.0f;    // Stddev for translational force Gaussian noise
  float tauMag = 0.0f;    // Magnitude for rotational torque (direction uniform over S2)
  unsigned int seed = 0;  // Optional seed; 0 => random_device
};

// Procedural population for large-N runs
struct PopulateCfg {
  int count = 0;            // Number of rods to generate; if >0, overrides scene.bodies
  bool grid = false;        // Back-compat: grid arrangement (vs uniform)
  float spacingMul = 1.6f;  // Spacing multiplier relative to diameter
  unsigned int seed = 0;    // RNG seed; 0 => random_device
  // New: populate mode: "grid", "uniform", "nonoverlap"
  std::string mode{"uniform"};
  int maxAttempts = 200000; // Max attempts per rod for nonoverlap sampling
};

struct BodyCfg {
    // existing fields:
    glm::vec3 pos{0};
    // rotation options previously present:
    glm::vec3 rot_axis{0,1,0};
    float rot_deg{0.0f};
    glm::vec4 rot_quat{1,0,0,0}; // we treat this as wxyz by default
    // NEW:
    glm::vec3 euler_deg{0,0,0};   // [yaw, pitch, roll] or whatever you prefer—see below
    std::string rot_quat_order{"wxyz"}; // "wxyz" (GLM default) or "xyzw"

    // shape/material
    std::string shape{"capsule"};  // "capsule" or "sphere"
    float length{0.5f};            // Used for capsule
    float diameter{0.1f};          // Used for capsule
    float radius{0.1f};            // Used for sphere
    float density{1000.0f};
    float restitution{0.2f};
    float friction{0.7f};
    // New advanced friction
    float friction_s{-1.0f};      // static friction (<=0 => use 'friction')
    float friction_d{-1.0f};      // dynamic friction (<=0 => use 'friction')
    float rolling_friction{0.0f}; // optional, not yet used in solver

    glm::vec3 v_lin{0};
    glm::vec3 v_ang{0};
};


struct SceneCfg {
  FloorCfg floor{};
  std::vector<BodyCfg> bodies;
  PeriodicCfg periodic{}; // optional periodic box
  RandomInitCfg randomInit{}; // optional random initialization for PBC
  RandomForceCfg randomForce{}; // optional random force injection
  PopulateCfg populate{}; // optional large-N population
};

struct AppCfg {
  PhysicsCfg physics{};
  RenderCfg  render{};
  SceneCfg   scene{};
};

// Returns false if file missing or invalid; cfg is still filled with defaults.
bool loadConfigFromFile(const std::string& path, AppCfg& cfg);

// Handy fallback (used when load fails)
AppCfg defaultAppCfg();

