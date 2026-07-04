/**
 * @file nsc_contact.hpp
 * @brief Non-Smooth Contact (NSC) solver for rigid rod (capsule) collisions.
 *
 * Implements an impulse-based PSOR (Projected Successive Over-Relaxation)
 * solver with Coulomb friction, following the Chrono
 * ChTimestepperEulerImplicitProjected pattern:
 *
 *   1. Free-flight velocity prediction
 *   2. Detect capsule-capsule contacts
 *   3. Velocity-level PSOR with friction cone projection
 *   4. Position update
 *   5. Position stabilization (normal-only PSOR)
 *
 * This operates on the same ContactPrimitive data produced by
 * the shared contact detector, but resolves contacts
 * as hard constraints (impulses) rather than penalty forces.
 */

#pragma once
#include "config/config.hpp"
#include "physics/contact_detector.hpp"
#include "physics/types.hpp"
#include <glm/glm.hpp>
#include <string>
#include <unordered_map>
#include <vector>

struct RigidBody;

/**
 * @brief Internal manifold for a single NSC contact constraint.
 *
 * Stores the contact geometry, cached diagonal values, and
 * accumulated impulses for the PSOR solver.
 */
struct NscManifold {
  int body_a, body_b;
  glm::vec3 normal;     ///< Contact normal (A→B)
  glm::vec3 t1, t2;     ///< Tangent basis on the contact plane
  glm::vec3 r_a, r_b;   ///< Lever arms: contact_point − body COM

  float phi;             ///< Signed gap (negative = penetrating)
  glm::vec3 v_rel_pre{0.0f}; ///< Pre-solve relative contact velocity
  float v_n_pre;         ///< Pre-solve normal relative velocity (for restitution)
  float restitution;     ///< Combined coefficient of restitution for this pair

  // Cached diagonal of J·M⁻¹·Jᵀ per direction
  float g_n;             ///< Normal
  float g_t1, g_t2;     ///< Tangent directions

  // Pre-cached Iinv * (r × d) terms for impulse application
  glm::vec3 IinvA_rAxn, IinvA_rAxt1, IinvA_rAxt2;
  glm::vec3 IinvB_rBxn, IinvB_rBxt1, IinvB_rBxt2;
  float invMassA, invMassB;

  // Accumulated impulses
  float lambda_n  = 0.0f; ///< Normal impulse (≥ 0)
  float lambda_t1 = 0.0f; ///< Tangent impulse direction 1
  float lambda_t2 = 0.0f; ///< Tangent impulse direction 2

  bool isWall = false;     ///< True for wall contacts (no body_b impulse)
};

/**
 * @brief NSC (hard contact) solver for capsule rigid bodies.
 *
 * Reuses ContactDetector for broadphase/narrowphase contact detection,
 * then resolves contacts via impulse-based PSOR with Coulomb friction.
 */
class NscContactSolver {
public:
  NscContactSolver() = default;

  /// Configure from NscContactCfg (called once at scene reset).
  void setConfig(const NscContactCfg& cfg);

  /// Configure PBC pass-through to internal detector.
  void setPBC(bool enabled, const glm::vec3& min, const glm::vec3& max);

  /**
   * @brief Detect capsule-capsule contacts and build constraint manifolds.
   *
   * Delegates broadphase + narrowphase to SoftContactSolver, then converts
   * ContactPrimitive → NscManifold with Jacobian diagonals and tangent basis.
   * Warm-starts impulses from cached values of the previous frame.
   */
  void detectAndBuildManifolds(const std::vector<RigidBody>& bodies);

  /**
   * @brief Solve velocity-level constraints with friction (PSOR).
   *
   * Modifies body velocities (v, w) in-place.
   */
  void solveVelocities(std::vector<RigidBody>& bodies, float dt);

  /**
   * @brief Position stabilization pass (normal-only PSOR).
   *
   * Re-detects contacts and projects positions to remove residual penetration.
   * Modifies body positions (x, q) in-place.
   */
  void projectPositions(std::vector<RigidBody>& bodies);

  /// Number of active contact manifolds after last detectAndBuildManifolds().
  size_t getNumContacts() const { return manifolds_.size(); }

  /// Number of manifolds that were impacting (approaching, v_n_pre < -tol)
  /// at the last solve — "collisions this step" as opposed to persistent
  /// resting contacts. Used for the collision-count time series (paper
  /// Fig. S4).
  int countImpacts(float tol = 1e-6f) const {
    int n = 0;
    for (const auto& m : manifolds_)
      if (m.v_n_pre < -tol) ++n;
    return n;
  }

  /// Add an externally-built manifold (e.g. cylinder wall contacts).
  void addManifold(const NscManifold& m) { manifolds_.push_back(m); }

  /// Build and add a wall-contact manifold for a single body.
  /// The wall has infinite mass and zero velocity.
  /// @param bodyIdx  Index of the body in the bodies array.
  /// @param contact  Contact geometry (normal points from body toward wall).
  /// @param bodies   The bodies array (needed for velocity/inertia precompute).
  /// @param restitution  Coefficient of restitution for the wall contact.
  void addWallContact(int bodyIdx, const Contact& contact,
                      const std::vector<RigidBody>& bodies,
                      float restitution = 1.0f);

  /// Access manifolds (for diagnostics / visualization).
  const std::vector<NscManifold>& getManifolds() const { return manifolds_; }

  /// Access raw detected contact geometry before manifold-specific response.
  const std::vector<ContactPrimitive>& getDetectedContacts() const {
    return detector_.getContacts();
  }

  /// Max constraint residual from the last solveVelocities() call.
  float getLastResidual() const { return lastResidual_; }

  /// Enable verbose printing of pre/post normal relative velocity per contact.
  void setDebugNormalVelocity(bool enabled) { debugNormalVelocity_ = enabled; }

  /// Configure CSV logging for pre/post contact relative velocities.
  void setDebugNormalVelocityCsvPath(const std::string& path) {
    debugNormalVelocityCsvPath_ = path;
  }

  /// Configure CSV logging for per-contact energy balance diagnostics.
  void setEnergyBalanceCsvPath(const std::string& path) {
    energyBalanceCsvPath_ = path;
  }

  /// Set current frame index for energy balance CSV logging.
  void setCurrentFrame(int frame) { currentFrame_ = frame; }

private:
  NscContactCfg cfg_{};
  ContactDetector detector_; ///< Reused for broadphase + narrowphase
  std::vector<NscManifold> manifolds_;
  float lastResidual_ = 0.0f;
  bool debugNormalVelocity_ = false;
  std::string debugNormalVelocityCsvPath_;
  std::string energyBalanceCsvPath_;
  int currentFrame_ = 0;

  // PBC settings (mirrored from detector_ for inline narrowphase)
  bool pbcEnabled_ = false;
  glm::vec3 pbcSize_{0.0f};

  // Persistent scratch buffers to avoid per-frame allocation
  std::vector<glm::vec3> posDx_, posDtheta_;
  std::vector<glm::mat3> posIinv_;

  // Warm-starting cache: keyed by ordered body pair → (λ_n, λ_t1, λ_t2).
  struct WarmKey {
    int lo, hi;
    bool operator==(const WarmKey& o) const { return lo == o.lo && hi == o.hi; }
  };
  struct WarmKeyHash {
    size_t operator()(const WarmKey& k) const {
      return std::hash<int>()(k.lo) ^ (std::hash<int>()(k.hi) << 16);
    }
  };
  struct WarmData { float n, t1, t2; };
  std::unordered_map<WarmKey, WarmData, WarmKeyHash> warmCache_;
};
