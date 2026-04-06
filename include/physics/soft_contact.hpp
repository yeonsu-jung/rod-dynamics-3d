/**
 * @file soft_contact.hpp
 * @brief Soft penalty-based contact solver for rod collisions
 *
 * Implements smooth potential energy-based contacts inspired by DisMech.
 * Uses penalty forces instead of impulses for better energy conservation.
 */

#pragma once
#include "config/config.hpp" // For SoftContactCfg
#include "physics/contact_detector.hpp"
#include <cmath>
#include <glm/glm.hpp>
#include <vector>

#include <unordered_map>

struct RigidBody;

/**
 * @brief Soft penalty-based contact solver
 *
 * Computes smooth contact forces from potential energy gradients.
 * Allows small controlled penetrations for stability and energy conservation.
 */
class SoftContactSolver {
public:
  explicit SoftContactSolver(const SoftContactCfg &config = SoftContactCfg());

  /**
   * @brief Configure Periodic Boundary Conditions
   */
  void setPBC(bool enabled, const glm::vec3 &min, const glm::vec3 &max);

  /**
   * @brief Detect contacts between all capsule pairs
   * @param bodies Vector of rigid bodies (capsules)
   */
  void detectContacts(const std::vector<RigidBody> &bodies);

  /**
   * @brief Compute contact forces and apply to bodies
   * @param bodies Vector of rigid bodies to apply forces to
   * @param dt Time step size
   * @param gravity Gravity vector (needed for Karnopp friction)
   */
  void computeForces(std::vector<RigidBody> &bodies, double dt,
                     const glm::vec3 &gravity = glm::vec3(0, -9.81, 0));

  /**
   * @brief Get current detected contacts (for visualization/debugging)
   */
  const std::vector<ContactPrimitive> &getContacts() const {
    return detector_.getContacts();
  }

  /**
   * @brief Get number of active contacts
   */
  size_t getNumContacts() const { return detector_.getNumContacts(); }

  /**
   * @brief Last accumulated total soft-contact potential energy (J) from most
   * recent computeForces call.
   *
   * The potential energy reported is the sum over all active contacts of the
   * underlying piecewise barrier/penalty potential scaled by k_scaler. This
   * lets the application monitor energy exchange between kinetic and stored
   * elastic energy in the soft model.
   */
  double getLastPotentialEnergy() const { return lastPotentialEnergy_; }

  const ContactBroadphaseStats &getStats() const { return detector_.getStats(); }

  /**
   * @brief Update configuration at runtime
   */
  void setConfig(const SoftContactCfg &config);

  /**
   * @brief Purge old contact history entries
   */
  void pruneContactHistory();

private:
  SoftContactCfg config_;
  ContactDetector detector_;

  double K1_; ///< Stiffness parameter = 15/delta
  double K2_; ///< Friction smoothness = 15/nu
  double lastPotentialEnergy_ =
      0.0; ///< Accumulated potential energy of contacts (J) after latest
           ///< computeForces()

  // Cundall-Strack History
  struct CSEntry {
    glm::vec3 tangential_force{0.0f}; ///< Accumulated tangential force (F_t)
    uint64_t last_frame = 0;          ///< For efficient pruning
  };
  std::unordered_map<uint64_t, CSEntry> contactHistory_;
  uint64_t frameCounter_ = 0;
  static constexpr uint64_t kHistoryRetainFrames = 2; // Keep for small gaps

  static uint64_t pairKey(int a, int b) {
    if (a > b)
      std::swap(a, b);
    return (uint64_t(a) << 32) | uint64_t(b);
  }

  // Force computation for each contact type
  void computeP2PForce(ContactPrimitive &contact);
  void computeE2PForce(ContactPrimitive &contact);
  void computeE2EForce(ContactPrimitive &contact);

  // Friction computation
  void computeFriction(ContactPrimitive &contact, const RigidBody &body_a,
                       const RigidBody &body_b, double dt,
                       const glm::vec3 &gravity);

  // Potential energy gradient helpers
  double potentialGradient(double distance, double h) const;
  double potentialEnergy(double distance, double h)
      const; ///< Unscaled base potential (before k_scaler)
};
