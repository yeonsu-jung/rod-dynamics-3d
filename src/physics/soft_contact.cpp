/**
 * @file soft_contact.cpp
 * @brief Implementation of soft penalty-based contact solver
 */

#include "physics/soft_contact.hpp"
#include "physics/rigid_body.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>

#ifdef USE_CUDA
#include "physics/cuda_broadphase.hpp"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

extern int g_thread_limit;
extern bool gQuiet;

namespace {
ContactDetectionCfg makeDetectionCfg(const SoftContactCfg &config) {
  ContactDetectionCfg detCfg;
  detCfg.delta = config.delta;
  detCfg.verbose = config.verbose;
  detCfg.use_spatial_hash = config.use_spatial_hash;
  detCfg.use_cuda = config.use_cuda;
  detCfg.use_aabb = config.use_aabb;
  detCfg.cell_size = config.cell_size;
  return detCfg;
}

#ifdef _OPENMP
void reportOpenMpTeamSizeOnce(const char *label) {
  static std::unordered_map<std::string, bool> reported;
  if (gQuiet) {
    return;
  }

#pragma omp critical(openmp_team_report)
  {
    bool &alreadyReported = reported[label];
    if (!alreadyReported) {
      alreadyReported = true;
      std::cout << "[Info] OpenMP actual team size for " << label << ": "
                << omp_get_num_threads() << "\n";
    }
  }
}
#endif
} // namespace

SoftContactSolver::SoftContactSolver(const SoftContactCfg &config)
    : config_(config), detector_(makeDetectionCfg(config)) {
  K1_ = 15.0 / config_.delta;
  K2_ = 15.0 / config_.nu;
  lastPotentialEnergy_ = 0.0;
}

void SoftContactSolver::setConfig(const SoftContactCfg &config) {
  config_ = config;
  detector_.setConfig(makeDetectionCfg(config));
  K1_ = 15.0 / config_.delta;
  K2_ = 15.0 / config_.nu;
  // Reset accumulator on config change (avoids stale reporting)
  lastPotentialEnergy_ = 0.0;
}
void SoftContactSolver::setPBC(bool enabled, const glm::vec3 &min,
                               const glm::vec3 &max) {
  detector_.setPBC(enabled, min, max);
}

void SoftContactSolver::detectContacts(const std::vector<RigidBody> &bodies) {
  detector_.detectContacts(bodies);
}

void SoftContactSolver::computeForces(std::vector<RigidBody> &bodies, double dt,
                                      const glm::vec3 &gravity) {
  auto &contacts = detector_.accessContacts();

#ifdef _OPENMP
  if (g_thread_limit > 0) {
    omp_set_num_threads(g_thread_limit);
  }
#endif
  // Process each contact and accumulate potential energy
  double pe_sum = 0.0;

  // Update frame counter for history tracking
  frameCounter_++;

#pragma omp parallel for reduction(+:pe_sum) if(contacts.size() > 64 && g_thread_limit != 1)
  for (int i = 0; i < (int)contacts.size(); ++i) {
    if (i == 0) {
      reportOpenMpTeamSizeOnce("contact force computation");
    }
    auto &contact = contacts[i];

    // Compute normal forces based on contact type
    switch (contact.type) {
    case ContactType::POINT_TO_POINT:
      computeP2PForce(contact);
      break;
    case ContactType::EDGE_TO_POINT:
      computeE2PForce(contact);
      break;
    case ContactType::EDGE_TO_EDGE:
      computeE2EForce(contact);
      break;
    }

    // std::cout << "damping coeff: " << config_.damping << std::endl;
    // Damping force (if enabled)
    if (config_.damping > 1e-9) {
      const auto &body_a = bodies[contact.body_a];
      const auto &body_b = bodies[contact.body_b];
      glm::vec3 r_a = contact.point_a - body_a.x;
      glm::vec3 r_b = contact.point_b - (body_b.x + contact.shift_b);
      glm::vec3 v_a = body_a.v + glm::cross(body_a.w, r_a);
      glm::vec3 v_b = body_b.v + glm::cross(body_b.w, r_b);
      glm::vec3 v_rel = v_a - v_b;
      float vn = glm::dot(v_rel, contact.normal);
      glm::vec3 f_damp = -config_.damping * vn * contact.normal;

      contact.force_a += f_damp;
      contact.force_b -= f_damp;
    }

    // Compute friction if enabled
    if (config_.enable_friction) {
      computeFriction(contact, bodies[contact.body_a], bodies[contact.body_b],
                      dt, gravity);
    } else {
      contact.friction_a = glm::vec3(0);
      contact.friction_b = glm::vec3(0);
    }

    // Apply forces to bodies
    auto &body_a = bodies[contact.body_a];
    auto &body_b = bodies[contact.body_b];

    // Total force = normal + friction
    glm::vec3 total_force_a = contact.force_a + contact.friction_a;
    glm::vec3 total_force_b = contact.force_b + contact.friction_b;

    // Compute torques: τ = r × F
    glm::vec3 r_a = contact.point_a - body_a.x;
    // For B, the contact point is in the shifted frame (near A).
    // The lever arm must be relative to the shifted center of B.
    // r_b = point_b - (body_b.x + shift_b)
    glm::vec3 r_b = contact.point_b - (body_b.x + contact.shift_b);

    glm::vec3 torque_a = glm::cross(r_a, total_force_a);
    glm::vec3 torque_b = glm::cross(r_b, total_force_b);

// Apply forces at contact points (generates torques automatically)
// Use atomics for thread safety
#pragma omp atomic
    body_a.f.x += total_force_a.x;
#pragma omp atomic
    body_a.f.y += total_force_a.y;
#pragma omp atomic
    body_a.f.z += total_force_a.z;

#pragma omp atomic
    body_b.f.x += total_force_b.x;
#pragma omp atomic
    body_b.f.y += total_force_b.y;
#pragma omp atomic
    body_b.f.z += total_force_b.z;

#pragma omp atomic
    body_a.tau.x += torque_a.x;
#pragma omp atomic
    body_a.tau.y += torque_a.y;
#pragma omp atomic
    body_a.tau.z += torque_a.z;

#pragma omp atomic
    body_b.tau.x += torque_b.x;
#pragma omp atomic
    body_b.tau.y += torque_b.y;
#pragma omp atomic
    body_b.tau.z += torque_b.z;

    // Accumulate potential energy for this contact (scaled by k_scaler)
    pe_sum += config_.k_scaler *
              potentialEnergy(contact.distance, contact.surface_limit);
  }
  lastPotentialEnergy_ = pe_sum;

  // if (config_.verbose && contacts.size() > 0) {
  //     std::cout << "[SoftContact] Detected " << contacts.size() << "
  //     contacts\n"; for (const auto& c : contacts_) {
  //         std::cout << "  Contact " << c.body_a << "-" << c.body_b
  //                   << ": dist=" << c.distance << " limit=" <<
  //                   c.surface_limit << "\n"
  //                   << "    point_a=" << c.point_a.x << "," << c.point_a.y <<
  //                   "," << c.point_a.z << "\n"
  //                   << "    point_b=" << c.point_b.x << "," << c.point_b.y <<
  //                   "," << c.point_b.z << "\n"
  //                   << "    normal=" << c.normal.x << "," << c.normal.y <<
  //                   "," << c.normal.z << "\n"
  //                   << "    force_a=" << c.force_a.x << "," << c.force_a.y <<
  //                   "," << c.force_a.z << "\n"
  //                   << "    force_b=" << c.force_b.x << "," << c.force_b.y <<
  //                   "," << c.force_b.z << "\n";
  //     }
  // }
}

double SoftContactSolver::potentialGradient(double distance, double h) const {
  const double delta = config_.delta;
  const double d = distance;

  // Piecewise gradient of potential energy
  if (d > h - delta) {
    // Non-penetrated regime: smooth log-barrier
    // U = (1/K1 * log(1 + exp(K1*(h-d))))²
    // dU/dd = -2/(K1²) * log(1+exp(arg)) * exp(arg)/(1+exp(arg))

    const double arg = K1_ * (h - d);
    const double exp_arg = std::exp(arg);
    const double log_term = std::log(1.0 + exp_arg);

    return -2.0 / (K1_ * K1_) * log_term * exp_arg / (1.0 + exp_arg);
  } else {
    // Penetrated regime: quadratic penalty
    // U = (h - d)²
    // dU/dd = -2*(h - d)

    return -2.0 * (h - d);
  }
}

double SoftContactSolver::potentialEnergy(double distance, double h) const {
  const double delta = config_.delta;
  const double d = distance;
  if (d > h - delta) {
    // Non-penetrated (barrier) regime
    const double arg = K1_ * (h - d);
    const double log_term = std::log(1.0 + std::exp(arg));
    double base = (log_term / K1_);
    return base * base; // (1/K1*log(1+exp(K1*(h-d))))^2
  } else {
    // Penetrated (quadratic) regime
    double pen = (h - d);
    return pen * pen; // (h-d)^2
  }
}

void SoftContactSolver::computeP2PForce(ContactPrimitive &contact) {
  // Point-to-point: simplest case
  // Force acts along line connecting the two points

  const double grad =
      potentialGradient(contact.distance, contact.surface_limit);
  const float force_mag =
      static_cast<float>(-config_.k_scaler * grad); // F = -k * dU/dd

  // Force direction: gradient points from A to B
  // dU/dx_b = (dU/dd) * (dd/dx_b) = grad * normal
  // dU/dx_a = (dU/dd) * (dd/dx_a) = grad * (-normal)

  contact.force_a = -force_mag * contact.normal; // Push A away from B
  contact.force_b = force_mag * contact.normal;  // Push B away from A
}

void SoftContactSolver::computeE2PForce(ContactPrimitive &contact) {
  // Edge-to-point: one body at endpoint, other on edge
  // For now, use same as P2P (simplified)
  // TODO: Implement proper E2P gradient distribution along edge

  computeP2PForce(contact);
}

void SoftContactSolver::computeE2EForce(ContactPrimitive &contact) {
  // Edge-to-edge: both bodies on interior of segments
  // For now, use same as P2P (simplified)
  // TODO: Implement proper E2E gradient with distributed forces

  computeP2PForce(contact);
}

void SoftContactSolver::computeFriction(ContactPrimitive &contact,
                                        const RigidBody &body_a,
                                        const RigidBody &body_b, double dt,
                                        const glm::vec3 &gravity) {
  // Compute velocities at contact points
  glm::vec3 r_a = contact.point_a - body_a.x;
  glm::vec3 r_b = contact.point_b - (body_b.x + contact.shift_b);

  glm::vec3 v_a = body_a.v + glm::cross(body_a.w, r_a);
  glm::vec3 v_b = body_b.v + glm::cross(body_b.w, r_b);

  glm::vec3 v_rel = v_a - v_b;

  // Normal force magnitude (available from earlier compute*Force calls)
  double fn_mag = glm::length(contact.force_a);

  // Project out normal component to get tangential velocity
  glm::vec3 v_tan = v_rel - glm::dot(v_rel, contact.normal) * contact.normal;
  float v_tan_mag = glm::length(v_tan);

  if (v_tan_mag < 1e-10) {
    // No tangential motion → no friction
    contact.friction_a = glm::vec3(0);
    contact.friction_b = glm::vec3(0);
    return;
  }

  // Karnopp Stick-Slip Model
  // --- Friction Models ---
  if (config_.friction_cundall) {
    // Cundall-Strack: Incremental history-dependent friction
    // 1. Get/Create history entry
    uint64_t key = pairKey(contact.body_a, contact.body_b);

    // Check if we already have this contact in history in a thread-safe way?
    // Wait, this is called inside an OMP loop. unordered_map access is NOT
    // thread safe. We MUST use a critical section or lock.

    glm::vec3 ft_accum(0.0f);

#pragma omp critical(cundall_history)
    {
      CSEntry &entry = contactHistory_[key];
      // If new or re-connected, we might start with previous value or 0.
      // Cundall assumes persistence.
      ft_accum = entry.tangential_force;

      // Update tangential spring
      // dF_t = -k_t * v_tan * dt
      ft_accum -= static_cast<float>(config_.kt * dt) * v_tan;

      // Enforce orthogonality to current normal (project onto tangent plane)
      // F_t_new = F_t_old - (F_t_old . n) * n
      // This accounts for the contact plane rotating.
      float normal_comp = glm::dot(ft_accum, contact.normal);
      ft_accum -= normal_comp * contact.normal;

      // Coulomb Limit
      float ft_mag = glm::length(ft_accum);
      float limit = static_cast<float>(config_.mu * fn_mag);
      if (ft_mag > limit) {
        if (ft_mag > 1e-8f) {
          ft_accum *= (limit / ft_mag);
        } else {
          ft_accum = glm::vec3(0.0f);
        }
      }

      // Write back and touch frame
      entry.tangential_force = ft_accum;
      entry.last_frame = frameCounter_;
    }

    // Apply the computed spring force
    contact.friction_a = ft_accum;
    contact.friction_b = -ft_accum;

    // Only used for logging if needed?
    return;
  }

  if (config_.friction_karnopp) {
    // If below velocity deadband, we are in "stick" or "pre-stick" state
    if (v_tan_mag < config_.vel_deadband) {
      // Need to find force required to stop the relative tangential motion in
      // one step.
      // 1. Force to kill current velocity: F_damping = -m * v_tan / dt
      // 2. Force to cancel external forces: F_static = -F_ext_tan
      //
      // Estimation of F_ext_tan:
      // We assume Gravity is the main external driver in common stick-slip
      // cases. We also check the body's accumulated force so far (which
      // includes random forces and previous contacts).
      // Note: body_a.f is the accumulated force on A *so far* in this step.
      // Threading note: this read is racy if we are updating body_a.f in
      // parallel, but typically we read specific fields or accept the
      // approximation. However, standard penalty methods accumulate.
      // A better stable approximation for F_ext is mostly Gravity + GravityB
      // (relative?)
      // Stick relies on relative motion.

      // Effective mass for contact point (simplified as reduced mass or just
      // local body masses) For simplicity, use body A's mass if A is dynamic,
      // B's if B is dynamic. Reduced mass m* = (ma*mb)/(ma+mb).
      float ma = body_a.invMass > 0 ? body_a.mass : 0.0f;
      float mb = body_b.invMass > 0 ? body_b.mass : 0.0f;
      float m_eff = 0.0f;
      if (ma > 0 && mb > 0)
        m_eff = (ma * mb) / (ma + mb);
      else if (ma > 0)
        m_eff = ma;
      else if (mb > 0)
        m_eff = mb;

      // Force to stop velocity:
      glm::vec3 F_stop = -(float)(m_eff / dt) * v_tan;

      // Estimate external force tangential component
      // F_ext_rel = F_ext_a/ma - F_ext_b/mb (acceleration difference) * m_eff
      // Simplified: Just use Gravity projected on tangent.
      // (Gravity is the same for both, so it tends to cancel in relative acc
      // UNLESS one is on a slope and the other is floor. But floor has 0 grav
      // effect if static?)
      // If one body is static floor (B), relative acc from gravity is g
      // (down). Tangent component of g drives sliding.
      // F_drive = m_eff * g_tan.
      // We want to oppose F_drive.

      glm::vec3 g_vec = gravity;
      glm::vec3 g_tan =
          g_vec - glm::dot(g_vec, contact.normal) * contact.normal;
      glm::vec3 F_drive = m_eff * g_tan;

      // Total required friction to HOLD: F_req = -F_drive + F_stop
      // (F_stop handles the inertial drift, -F_drive handles the gravity push)
      glm::vec3 F_req = -F_drive + F_stop;

      float F_req_mag = glm::length(F_req);
      // float fn_mag = glm::length(contact.force_a); // Already defined
      float max_static_friction = config_.mu_static * fn_mag;

      if (F_req_mag <= max_static_friction) {
        // STICK: We can generate enough force to hold/stop.
        // Apply exactly F_req.
        contact.friction_a = F_req;
        contact.friction_b = -F_req;
        return;
      } else {
        // SLIP (Breakaway): Cannot hold.
        // Apply max static friction opposing the required direction?
        // Or transition to dynamic?
        // Karnopp usually caps at static limit during the breakaway frame.
        if (F_req_mag > 1e-12) {
          contact.friction_a = F_req * (max_static_friction / F_req_mag);
          contact.friction_b = -contact.friction_a;
        }
        return;
      }
    } else {
      // Nominal Sliding (Slip state)
      // Use kinetic friction
      // float fn_mag = glm::length(contact.force_a); // Already defined
      glm::vec3 dir = -v_tan / v_tan_mag; // Oppose motion
      float fric_mag = config_.mu * fn_mag;
      contact.friction_a = dir * fric_mag;
      contact.friction_b = -contact.friction_a;
      return;
    }
  }

  // Smooth friction coefficient (sticking-to-sliding transition)
  double gamma;
  if (v_tan_mag > config_.nu) {
    gamma = 1.0; // Sliding regime
  } else {
    // Sticking regime: smooth transition using tanh-like function
    gamma = 2.0 / (1.0 + std::exp(-K2_ * v_tan_mag)) - 1.0;
  }

  // Effective friction coefficient (Stribeck effect for stick-slip)
  double mu_eff = config_.mu;
  if (config_.mu_static > config_.mu) {
    // Blend from mu_static (at v=0) to mu_dynamic (at v>>nu)
    // Decay scale: we want the static peak to be effective within the sticking
    // range. Using nu as the decay constant ensures smooth transition.
    double decay = std::exp(-v_tan_mag / config_.nu);
    mu_eff = config_.mu + (config_.mu_static - config_.mu) * decay;
  }

  // Friction force magnitude: μ_eff * γ * |F_normal|
  // const float fn_mag = glm::length(contact.force_a); // Already defined
  const glm::vec3 friction_dir = -v_tan / v_tan_mag; // Oppose tangential motion

  const float friction_mag = static_cast<float>(mu_eff * gamma) * fn_mag;
  const glm::vec3 friction_force = friction_mag * friction_dir;

  contact.friction_a = friction_force;
  contact.friction_b = -friction_force;
}

