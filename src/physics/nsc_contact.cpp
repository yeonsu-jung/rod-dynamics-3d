/**
 * @file nsc_contact.cpp
 * @brief NSC (Non-Smooth Contact) impulse-based PSOR solver.
 *
 * Implements velocity-level PSOR with Coulomb box friction for capsule rods,
 * position stabilization via normal-only PSOR, and warm-starting of impulses.
 */

#include "physics/nsc_contact.hpp"
#include "physics/rigid_body.hpp"
#include <algorithm>
#include <cmath>

// ─── Configuration ──────────────────────────────────────────────────────────

void NscContactSolver::setConfig(const NscContactCfg& cfg) {
  cfg_ = cfg;

  // Forward broadphase settings to the internal detector.
  SoftContactCfg detCfg;
  detCfg.enabled = true;            // Always on when NSC is active
  detCfg.use_spatial_hash = cfg.use_spatial_hash;
  detCfg.cell_size = cfg.cell_size;
  detCfg.use_aabb = cfg.use_aabb;
  detCfg.enable_friction = false;   // Friction handled by PSOR, not penalty
  detCfg.k_scaler = 0.0f;          // No penalty forces
  detector_.setConfig(detCfg);
  warmCache_.clear();
}

void NscContactSolver::setPBC(bool enabled,
                              const glm::vec3& min,
                              const glm::vec3& max) {
  detector_.setPBC(enabled, min, max);
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Build an orthonormal tangent basis {t1, t2} on the plane perpendicular to n.
static void buildTangentBasis(const glm::vec3& n,
                              glm::vec3& t1, glm::vec3& t2) {
  // Pick the axis least aligned with n to avoid near-zero cross products.
  glm::vec3 ref = (std::abs(n.x) < 0.9f) ? glm::vec3(1, 0, 0)
                                           : glm::vec3(0, 1, 0);
  t1 = glm::normalize(glm::cross(n, ref));
  t2 = glm::cross(n, t1);  // Already unit length since n, t1 are orthonormal.
}

/// Compute diagonal of J·M⁻¹·Jᵀ for constraint direction d.
///   g = 1/m_a + 1/m_b + (r_a×d)·Iinv_a·(r_a×d) + (r_b×d)·Iinv_b·(r_b×d) + cfm
static float computeDiag(float invMassA, float invMassB,
                         const glm::mat3& IinvA, const glm::mat3& IinvB,
                         const glm::vec3& rA, const glm::vec3& rB,
                         const glm::vec3& d, float cfm) {
  glm::vec3 rAxd = glm::cross(rA, d);
  glm::vec3 rBxd = glm::cross(rB, d);
  float g = invMassA + invMassB
          + glm::dot(rAxd, IinvA * rAxd)
          + glm::dot(rBxd, IinvB * rBxd)
          + cfm;
  return std::max(g, 1e-12f);
}

// ─── Contact Detection → Manifold Build ─────────────────────────────────────

void NscContactSolver::detectAndBuildManifolds(
    const std::vector<RigidBody>& bodies) {
  manifolds_.clear();

  // Delegate broadphase + narrowphase to existing SoftContactSolver.
  detector_.detectContacts(bodies);
  const auto& contacts = detector_.getContacts();
  manifolds_.reserve(contacts.size());

  for (const auto& cp : contacts) {
    // Signed gap: negative = penetrating.
    float phi = static_cast<float>(cp.distance - cp.surface_limit);

    NscManifold m;
    m.body_a = cp.body_a;
    m.body_b = cp.body_b;
    m.normal = cp.normal;
    m.phi    = phi;

    buildTangentBasis(m.normal, m.t1, m.t2);

    // Lever arms: contact point relative to each body's COM.
    // point_b is in the PBC-shifted frame, so subtract the shift.
    m.r_a = cp.point_a - bodies[cp.body_a].x;
    m.r_b = cp.point_b - (bodies[cp.body_b].x + cp.shift_b);

    // Cache world-space inverse inertia tensors (once per manifold).
    const auto& A = bodies[cp.body_a];
    const auto& B = bodies[cp.body_b];
    glm::mat3 IinvA = A.IworldInv();
    glm::mat3 IinvB = B.IworldInv();

    // Diagonal of J·M⁻¹·Jᵀ for each constraint direction.
    m.g_n  = computeDiag(A.invMass, B.invMass, IinvA, IinvB,
                         m.r_a, m.r_b, m.normal, cfg_.cfm);
    m.g_t1 = computeDiag(A.invMass, B.invMass, IinvA, IinvB,
                         m.r_a, m.r_b, m.t1, cfg_.cfm);
    m.g_t2 = computeDiag(A.invMass, B.invMass, IinvA, IinvB,
                         m.r_a, m.r_b, m.t2, cfg_.cfm);

    // Warm-start: look up cached impulses from the previous frame.
    int lo = std::min(cp.body_a, cp.body_b);
    int hi = std::max(cp.body_a, cp.body_b);
    auto it = warmCache_.find({lo, hi});
    if (it != warmCache_.end()) {
      m.lambda_n  = std::max(0.0f, it->second.n);
      m.lambda_t1 = it->second.t1;
      m.lambda_t2 = it->second.t2;
    } else {
      m.lambda_n  = 0.0f;
      m.lambda_t1 = 0.0f;
      m.lambda_t2 = 0.0f;
    }

    manifolds_.push_back(m);
  }
}

// ─── Velocity PSOR Solve ────────────────────────────────────────────────────

void NscContactSolver::solveVelocities(std::vector<RigidBody>& bodies,
                                       float dt) {
  lastResidual_ = 0.0f;
  if (manifolds_.empty()) return;

  const float omega = cfg_.omega;
  const float cfm   = cfg_.cfm;
  const float mu    = cfg_.mu;
  const float beta  = cfg_.beta;

  // Cache world-space inverse inertia tensors (orientations are constant
  // during the velocity solve — only v, w change).
  std::vector<glm::mat3> Iinv(bodies.size());
  for (size_t i = 0; i < bodies.size(); ++i)
    Iinv[i] = bodies[i].IworldInv();

  // Apply warm-start impulses before iterating.
  for (const auto& m : manifolds_) {
    auto& A = bodies[m.body_a];
    auto& B = bodies[m.body_b];
    const glm::mat3& IinvA = Iinv[m.body_a];
    const glm::mat3& IinvB = Iinv[m.body_b];

    // Normal warm-start
    if (m.lambda_n != 0.0f) {
      glm::vec3 rAxn = glm::cross(m.r_a, m.normal);
      glm::vec3 rBxn = glm::cross(m.r_b, m.normal);
      A.v -= m.normal * (m.lambda_n * A.invMass);
      A.w -= IinvA * rAxn * m.lambda_n;
      B.v += m.normal * (m.lambda_n * B.invMass);
      B.w += IinvB * rBxn * m.lambda_n;
    }
    // Tangent-1 warm-start
    if (m.lambda_t1 != 0.0f) {
      glm::vec3 rAxt1 = glm::cross(m.r_a, m.t1);
      glm::vec3 rBxt1 = glm::cross(m.r_b, m.t1);
      A.v -= m.t1 * (m.lambda_t1 * A.invMass);
      A.w -= IinvA * rAxt1 * m.lambda_t1;
      B.v += m.t1 * (m.lambda_t1 * B.invMass);
      B.w += IinvB * rBxt1 * m.lambda_t1;
    }
    // Tangent-2 warm-start
    if (m.lambda_t2 != 0.0f) {
      glm::vec3 rAxt2 = glm::cross(m.r_a, m.t2);
      glm::vec3 rBxt2 = glm::cross(m.r_b, m.t2);
      A.v -= m.t2 * (m.lambda_t2 * A.invMass);
      A.w -= IinvA * rAxt2 * m.lambda_t2;
      B.v += m.t2 * (m.lambda_t2 * B.invMass);
      B.w += IinvB * rBxt2 * m.lambda_t2;
    }
  }

  // PSOR iterations.
  float maxResidual = 0.0f;
  for (int iter = 0; iter < cfg_.velocity_iters; ++iter) {
    float iterResidual = 0.0f;
    for (auto& m : manifolds_) {
      auto& A = bodies[m.body_a];
      auto& B = bodies[m.body_b];
      const glm::mat3& IinvA = Iinv[m.body_a];
      const glm::mat3& IinvB = Iinv[m.body_b];

      // ── Normal constraint (unilateral: λ_n ≥ 0) ──

      // Relative velocity at contact: v_rel = v_B_contact − v_A_contact
      glm::vec3 v_rel = (B.v + glm::cross(B.w, m.r_b))
                       - (A.v + glm::cross(A.w, m.r_a));

      // Baumgarte bias: pushes penetrating contacts apart.
      float b_n = beta * m.phi / dt;
      float w_n = glm::dot(m.normal, v_rel) + b_n + cfm * m.lambda_n;
      float delta_n = -(omega / m.g_n) * w_n;
      float old_n = m.lambda_n;
      m.lambda_n = std::max(0.0f, old_n + delta_n);
      float dn = m.lambda_n - old_n;

      if (dn != 0.0f) {
        glm::vec3 rAxn = glm::cross(m.r_a, m.normal);
        glm::vec3 rBxn = glm::cross(m.r_b, m.normal);
        A.v -= m.normal * (dn * A.invMass);
        A.w -= IinvA * rAxn * dn;
        B.v += m.normal * (dn * B.invMass);
        B.w += IinvB * rBxn * dn;
      }

      // ── Tangent-1 friction (box: |λ_t1| ≤ μ·λ_n) ──

      v_rel = (B.v + glm::cross(B.w, m.r_b))
            - (A.v + glm::cross(A.w, m.r_a));
      float max_fric = mu * m.lambda_n;

      float w_t1 = glm::dot(m.t1, v_rel) + cfm * m.lambda_t1;
      float delta_t1 = -(omega / m.g_t1) * w_t1;
      float old_t1 = m.lambda_t1;
      m.lambda_t1 = std::clamp(old_t1 + delta_t1, -max_fric, max_fric);
      float dt1 = m.lambda_t1 - old_t1;

      if (dt1 != 0.0f) {
        glm::vec3 rAxt1 = glm::cross(m.r_a, m.t1);
        glm::vec3 rBxt1 = glm::cross(m.r_b, m.t1);
        A.v -= m.t1 * (dt1 * A.invMass);
        A.w -= IinvA * rAxt1 * dt1;
        B.v += m.t1 * (dt1 * B.invMass);
        B.w += IinvB * rBxt1 * dt1;
      }

      // ── Tangent-2 friction (box: |λ_t2| ≤ μ·λ_n) ──

      v_rel = (B.v + glm::cross(B.w, m.r_b))
            - (A.v + glm::cross(A.w, m.r_a));

      float w_t2 = glm::dot(m.t2, v_rel) + cfm * m.lambda_t2;
      float delta_t2 = -(omega / m.g_t2) * w_t2;
      float old_t2 = m.lambda_t2;
      m.lambda_t2 = std::clamp(old_t2 + delta_t2, -max_fric, max_fric);
      float dt2 = m.lambda_t2 - old_t2;

      if (dt2 != 0.0f) {
        glm::vec3 rAxt2 = glm::cross(m.r_a, m.t2);
        glm::vec3 rBxt2 = glm::cross(m.r_b, m.t2);
        A.v -= m.t2 * (dt2 * A.invMass);
        A.w -= IinvA * rAxt2 * dt2;
        B.v += m.t2 * (dt2 * B.invMass);
        B.w += IinvB * rBxt2 * dt2;
      }

      // Track per-constraint residual (sum of absolute deltas).
      iterResidual = std::max(iterResidual,
          std::abs(dn) + std::abs(dt1) + std::abs(dt2));
    }
    maxResidual = iterResidual;
  }
  lastResidual_ = maxResidual;

  // Update warm-start cache for next frame.
  warmCache_.clear();
  for (const auto& m : manifolds_) {
    int lo = std::min(m.body_a, m.body_b);
    int hi = std::max(m.body_a, m.body_b);
    warmCache_[{lo, hi}] = {m.lambda_n, m.lambda_t1, m.lambda_t2};
  }
}

// ─── Position Stabilization ─────────────────────────────────────────────────

void NscContactSolver::projectPositions(std::vector<RigidBody>& bodies) {
  if (!cfg_.position_stabilization) return;

  const float omega = cfg_.omega;
  const float cfm   = cfg_.cfm;
  const float slop  = cfg_.slop;

  for (int outer = 0; outer < cfg_.position_iters; ++outer) {
    // Re-detect contacts at current positions.
    detector_.detectContacts(bodies);
    const auto& contacts = detector_.getContacts();

    // Build normal-only manifolds for penetrating contacts beyond slop.
    struct PosManifold {
      int body_a, body_b;
      glm::vec3 normal;
      glm::vec3 r_a, r_b;
      float phi;       // gap + slop (still negative if beyond slop)
      float g_n;       // diagonal
      float lambda_n;  // accumulated correction impulse
    };

    std::vector<PosManifold> pm;
    pm.reserve(contacts.size());

    for (const auto& cp : contacts) {
      float phi = static_cast<float>(cp.distance - cp.surface_limit);
      if (phi >= -slop) continue; // Not penetrating beyond slop.

      PosManifold p;
      p.body_a  = cp.body_a;
      p.body_b  = cp.body_b;
      p.normal  = cp.normal;
      p.phi     = phi + slop; // Shift so correction targets slop boundary.

      p.r_a = cp.point_a - bodies[cp.body_a].x;
      p.r_b = cp.point_b - (bodies[cp.body_b].x + cp.shift_b);

      const auto& A = bodies[cp.body_a];
      const auto& B = bodies[cp.body_b];
      p.g_n = computeDiag(A.invMass, B.invMass,
                          A.IworldInv(), B.IworldInv(),
                          p.r_a, p.r_b, p.normal, cfm);
      p.lambda_n = 0.0f;
      pm.push_back(p);
    }

    if (pm.empty()) return; // No significant penetration; done.

    // Accumulate position (and orientation) corrections via normal-only PSOR.
    std::vector<glm::vec3> dx(bodies.size(), glm::vec3(0.0f));
    std::vector<glm::vec3> dtheta(bodies.size(), glm::vec3(0.0f));

    // Cache inverse inertia tensors (orientations unchanged within this pass).
    std::vector<glm::mat3> Iinv(bodies.size());
    for (size_t i = 0; i < bodies.size(); ++i)
      Iinv[i] = bodies[i].IworldInv();

    for (int inner = 0; inner < cfg_.position_psor_iters; ++inner) {
      for (auto& p : pm) {
        // Residual: projection of accumulated correction onto normal + gap.
        float corr_rel = glm::dot(p.normal,
                           dx[p.body_b] + glm::cross(dtheta[p.body_b], p.r_b)
                         - dx[p.body_a] - glm::cross(dtheta[p.body_a], p.r_a));
        float residual = corr_rel + p.phi + cfm * p.lambda_n;

        float delta = -(omega / p.g_n) * residual;
        float old_lambda = p.lambda_n;
        p.lambda_n = std::max(0.0f, old_lambda + delta);
        float dl = p.lambda_n - old_lambda;

        if (dl != 0.0f) {
          glm::vec3 rAxn = glm::cross(p.r_a, p.normal);
          glm::vec3 rBxn = glm::cross(p.r_b, p.normal);

          dx[p.body_a]     -= p.normal * (dl * bodies[p.body_a].invMass);
          dtheta[p.body_a] -= Iinv[p.body_a] * rAxn * dl;
          dx[p.body_b]     += p.normal * (dl * bodies[p.body_b].invMass);
          dtheta[p.body_b] += Iinv[p.body_b] * rBxn * dl;
        }
      }
    }

    // Apply accumulated position + orientation corrections.
    for (size_t i = 0; i < bodies.size(); ++i) {
      if (bodies[i].invMass <= 0.0f) continue;
      bodies[i].x += dx[i];

      float angle = glm::length(dtheta[i]);
      if (angle > 1e-8f) {
        glm::vec3 axis = dtheta[i] / angle;
        glm::quat dq = glm::angleAxis(angle, axis);
        bodies[i].q = glm::normalize(dq * bodies[i].q);
      }
    }
  }
}
