#pragma once
#include <glm/glm.hpp>
#include <string>
#include <vector>

struct SoftContactCfg {
  bool enabled = false;     // Use soft contact instead of hard impulse solver
  double delta = 0.005;     // Transition width for piecewise potential
  float k_scaler = 1000.0f; // Stiffness scaler
  float damping = 0.0f;     // Viscous damping coefficient
  float mu = 0.5f;          // Friction coefficient
  double mu_static =
      0.;           // Static friction coefficient (if > mu, enables stick-slip)
  double nu = 1e-5; // Sticking velocity threshold (m/s)
  bool enable_friction = true;  // Enable friction in soft contact
  bool verbose = false;         // Print contact debug info
  bool use_spatial_hash = true; // Enable spatial hash broadphase
  bool use_cuda = false;        // Enable GPU (CUDA) naive O(N^2) broadphase
  bool use_aabb = true;         // Enable AABB pre-check
  double cell_size = -1.0;      // Spatial hash cell size (<=0 => auto)
  // Karnopp friction settings
  bool friction_karnopp = false; // Use Karnopp stick-slip friction model
  // Cundall-Strack Friction (Incremental history-dependent)
  bool friction_cundall = false; // Enable Cundall-Strack model
  double kt = 1000.0;            // Tangential stiffness for Cundall-Strack
  double vel_deadband = 1e-3;    // Velocity deadband for Karnopp (m/s)
};

struct HertzMindlinCfg {
  bool enabled = false; // Use Hertz-Mindlin model (for spheres only)
  double youngs_modulus =
      7e10; // Young's modulus E (Pa) [default: glass ~70 GPa]
  double poisson_ratio = 0.25;    // Poisson's ratio ν [typical: 0.2-0.3]
  double restitution_coeff = 0.9; // Coefficient of restitution e [0-1]
  double friction_coeff = 0.3;    // Kinetic friction coefficient μ_k
  double friction_static_coeff =
      0.5; // Static friction coefficient μ_s [typically > μ_k]
  double friction_transition_vel =
      1e-3; // Velocity threshold v_c for μ_s→μ_k transition (m/s)
  double rolling_friction_coeff = 0.01; // Rolling friction μ_r
  bool enable_tangential = true;        // Enable Mindlin tangential forces
  bool enable_rolling = true;           // Enable rolling friction
  bool use_uniform_grid = true;         // Enable sphere broadphase grid
  double broadphase_cell_size =
      -1.0; // Optional override for grid cell size (<=0 => auto)
  int broadphase_min_bodies = 12; // Minimum sphere count before grid kicks in
  bool verbose = false;           // Debug output

  // Damping coefficients (computed from restitution)
  mutable double normal_damping = 0.0;     // γ_n (computed lazily)
  mutable double tangential_damping = 0.0; // γ_t (computed lazily)

  // Compute damping coefficients from restitution (Silbert et al. 2001)
  void computeDamping() const {
    if (restitution_coeff >= 1.0) {
      normal_damping = 0.0;
      tangential_damping = 0.0;
    } else {
      double ln_e = std::log(restitution_coeff);
      normal_damping = -ln_e / std::sqrt(M_PI * M_PI + ln_e * ln_e);
      tangential_damping = normal_damping;
    }
  }
};

struct NscContactCfg {
  bool enabled = false;           // Enable NSC (hard) contact solver

  // Solver parameters
  float mu = 0.3f;                // Coulomb friction coefficient
  float beta = 0.2f;              // Baumgarte stabilization factor
  float cfm = 0.0f;               // Constraint Force Mixing (regularization)
  float omega = 1.0f;             // SOR relaxation (1.0 = Gauss-Seidel)
  int velocity_iters = 40;        // PSOR iterations for velocity solve

  // Position stabilization
  bool position_stabilization = true;
  int position_iters = 5;         // Outer loops of position projection
  int position_psor_iters = 50;   // Inner PSOR per position loop
  float slop = 1e-4f;             // Allowed penetration before correction

  // Broadphase (reuses SoftContactSolver infrastructure)
  bool use_spatial_hash = true;
  double cell_size = -1.0;        // Auto if <=0
  bool use_aabb = true;
};

struct PhysicsCfg {
  float dt = 1.0f / 600.0f;
  glm::vec3 gravity{0.0f, -10.0f, 0.0f};
  float lin_damp = 0.08f;
  float ang_damp = 0.12f;
  float w_max = 80.0f;

  SoftContactCfg soft_contact{};   // Soft penalty-based contact configuration
  HertzMindlinCfg hertz_mindlin{}; // Hertz-Mindlin granular contact model
  NscContactCfg nsc{};             // NSC impulse-based contact (hard contacts)
  // Optional alternative contact model inspired by MuJoCo. When enabled,
  // this uses MujocoContactSolver instead of SoftContactSolver for
  // soft penalty contacts. The two paths are kept separate for easy A/B tests.
  bool use_mujoco_contact = false;
};

struct GridCfg {
  bool enabled = true;
  float scale = 1.0f;
  glm::vec3 c1{0.80f, 0.82f, 0.85f};
  glm::vec3 c2{0.65f, 0.67f, 0.70f};
};

struct RenderCfg {
  // orbit camera
  float yaw = 0.6f, pitch = 0.35f, dist = 6.0f;
  glm::vec3 lightDir{-0.4f, -1.0f, -0.3f};
  glm::vec3 bg{0.08f, 0.09f, 0.11f};
  GridCfg grid{};
  bool vsync = true;
  bool cull = false;
  int msaa_samples = 4;
};

struct FloorCfg {
  bool enabled = false; // Enable implicit floor
  glm::vec3 pos{0, -0.8f, 0};
  glm::vec4 rot_quat{1, 0, 0, 0}; // wxyz
  glm::vec3 half_extents{10.0f, 0.1f, 10.0f};
  float restitution = 0.3f;
  float friction = 0.9f;
};

// Periodic boundary configuration
struct PeriodicCfg {
  bool enabled = false; // Enable periodic boundaries instead of floor
  glm::vec3 min{-3.0f, -1.0f, -3.0f}; // Box minimum corner
  glm::vec3 max{+3.0f, +3.0f, +3.0f}; // Box maximum corner
  float cellSize = 0.6f;              // Broadphase grid cell size
  int longSpan = 4; // Threshold: rods spanning > this many cells on any axis
                    // are treated as long
};

// Random initialization configuration (for PBC studies)
struct RandomInitCfg {
  bool enabled = false; // If true and periodic is enabled, set gravity=0 and
                        // assign random velocities
  float vSigma = 0.3f;  // Stddev for translational velocity normal distribution
  float wSpeed =
      1.5f; // Constant angular speed magnitude (direction uniform over S2)
  unsigned int seed = 0; // Optional seed; 0 => random_device
};

// Random force injection configuration
struct RandomForceCfg {
  bool enabled = false; // If true, apply random forces/torques each step
  float fSigma = 0.0f;  // Stddev for translational force Gaussian noise
  float tauMag =
      0.0f; // Magnitude for rotational torque (direction uniform over S2)
  unsigned int seed = 0; // Optional seed; 0 => random_device
};

// Procedural population for large-N runs
struct PopulateCfg {
  int count = 0; // Number of bodies to generate; if >0, overrides scene.bodies
  bool grid = false;       // Back-compat: grid arrangement (vs uniform)
  float spacingMul = 1.6f; // Spacing multiplier relative to diameter
  unsigned int seed = 0;   // RNG seed; 0 => random_device
  // New: populate mode: "grid", "uniform", "nonoverlap"
  std::string mode{"uniform"};
  int maxAttempts = 200000; // Max attempts per body for nonoverlap sampling
  // Shape specification for populate
  std::string shape{"capsule"}; // "capsule" (rod) or "sphere"
  float radius = 0.05f;         // Sphere radius (when shape="sphere")
  float density = 2500.0f;      // Density for populate bodies
};

struct BodyCfg {
  // existing fields:
  glm::vec3 pos{0};
  // rotation options previously present:
  glm::vec3 rot_axis{0, 1, 0};
  float rot_deg{0.0f};
  glm::vec4 rot_quat{1, 0, 0, 0}; // we treat this as wxyz by default
  // NEW:
  glm::vec3 euler_deg{0, 0,
                      0}; // [yaw, pitch, roll] or whatever you prefer—see below
  std::string rot_quat_order{"wxyz"}; // "wxyz" (GLM default) or "xyzw"

  // shape/material
  std::string shape{"capsule"}; // "capsule" or "sphere"
  float length{0.5f};           // Used for capsule
  float diameter{0.1f};         // Used for capsule
  float radius{0.1f};           // Used for sphere
  float density{1000.0f};
  float restitution{0.2f};
  float friction{0.7f};
  // New advanced friction
  float friction_s{-1.0f};      // static friction (<=0 => use 'friction')
  float friction_d{-1.0f};      // dynamic friction (<=0 => use 'friction')
  float rolling_friction{0.0f}; // optional, not yet used in solver
  bool is_static{false}; // If true, body has infinite mass/inertia and is fixed

  glm::vec3 v_lin{0};
  glm::vec3 v_ang{0};
};

struct SceneCfg {
  FloorCfg floor{};

  PeriodicCfg periodic{};       // optional periodic box
  RandomInitCfg randomInit{};   // optional random initialization for PBC
  RandomForceCfg randomForce{}; // optional random force injection
  PopulateCfg populate{};       // optional large-N population
  // Optional initial configuration for rods from CSV (endpoints per row)
  // Path to a CSV file with header x0,y0,z0,x1,y1,z1 and optional '#' metadata
  // lines.
  std::vector<BodyCfg> bodies;
  std::string initCsvPath;

  bool fixCentroidRod = false;
  std::string fixedRodSelectionMethod = "centroid"; // "centroid" or "horizontal"
  int numFixedRods = 1; // Number of rods to fix (first uses selection method, rest random)
  int fixEveryExcept = -1; // If >= 0, fix all rods except this index (-1 = disabled)
};

struct AppCfg {
  PhysicsCfg physics{};
  RenderCfg render{};
  SceneCfg scene{};
};

// Returns false if file missing or invalid; cfg is still filled with defaults.
bool loadConfigFromFile(const std::string &path, AppCfg &cfg);

// Handy fallback (used when load fails)
AppCfg defaultAppCfg();
