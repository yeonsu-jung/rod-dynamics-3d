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
  pbcEnabled_ = enabled;
  pbcSize_ = max - min;
  detector_.setPBC(enabled, min, max);
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Closest points between two line segments (parameters s,t in [0,1]).
static void closestPointsSegSeg(const glm::vec3& a1, const glm::vec3& a2,
                                const glm::vec3& b1, const glm::vec3& b2,
                                double& s, double& t) {
  const glm::vec3 e1 = a2 - a1;
  const glm::vec3 e2 = b2 - b1;
  const glm::vec3 e12 = b1 - a1;
  const double D1 = glm::dot(e1, e1);
  const double D2 = glm::dot(e2, e2);
  const double S1 = glm::dot(e1, e12);
  const double S2 = glm::dot(e2, e12);
  const double R  = glm::dot(e1, e2);
  const double den = D1 * D2 - R * R;

  auto fixBound = [](double& x) -> bool {
    if (x > 1.0) { x = 1.0; return true; }
    if (x < 0.0) { x = 0.0; return true; }
    return false;
  };

  double uf;
  if (den == 0.0) {
    s = 0.0;  t = -S2 / D2;  uf = t;
    fixBound(uf);
    if (uf != t) { s = (uf * R + S1) / D1; fixBound(s); t = uf; }
  } else {
    s = (S1 * D2 - S2 * R) / den;
    fixBound(s);
    t = (s * R - S2) / D2;  uf = t;
    fixBound(uf);
    if (uf != t) { s = (uf * R + S1) / D1; fixBound(s); t = uf; }
  }
}

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

/// Compute diagonal of J·M⁻¹·Jᵀ for a single-body wall constraint.
///   g = 1/m_a + (r_a×d)·Iinv_a·(r_a×d) + cfm
static float computeDiagWall(float invMassA, const glm::mat3& IinvA,
                             const glm::vec3& rA, const glm::vec3& d,
                             float cfm) {
  glm::vec3 rAxd = glm::cross(rA, d);
  float g = invMassA + glm::dot(rAxd, IinvA * rAxd) + cfm;
  return std::max(g, 1e-12f);
}

// ─── Wall Contact Manifold Builder ──────────────────────────────────────────

void NscContactSolver::addWallContact(int bodyIdx, const Contact& contact,
                                      const std::vector<RigidBody>& bodies,
                                      float restitution) {
  const auto& A = bodies[bodyIdx];
  NscManifold m;
  m.body_a = bodyIdx;
  m.body_b = bodyIdx;  // unused for wall contacts
  m.isWall = true;
  m.normal = contact.normal;
  m.phi    = -contact.penetration;  // negative = penetrating

  buildTangentBasis(m.normal, m.t1, m.t2);

  m.r_a = contact.point - A.x;
  m.r_b = glm::vec3(0.0f);  // wall has no lever arm

  glm::mat3 IinvA = A.IworldInv();
  glm::mat3 zeroMat(0.0f);

  m.g_n  = computeDiagWall(A.invMass, IinvA, m.r_a, m.normal, cfg_.cfm);
  m.g_t1 = computeDiagWall(A.invMass, IinvA, m.r_a, m.t1, cfg_.cfm);
  m.g_t2 = computeDiagWall(A.invMass, IinvA, m.r_a, m.t2, cfg_.cfm);

  // Pre-cache Iinv * (r × d) terms for wall contact.
  m.invMassA = A.invMass;
  m.invMassB = 0.0f;
  m.IinvA_rAxn  = IinvA * glm::cross(m.r_a, m.normal);
  m.IinvA_rAxt1 = IinvA * glm::cross(m.r_a, m.t1);
  m.IinvA_rAxt2 = IinvA * glm::cross(m.r_a, m.t2);
  m.IinvB_rBxn  = glm::vec3(0.0f);
  m.IinvB_rBxt1 = glm::vec3(0.0f);
  m.IinvB_rBxt2 = glm::vec3(0.0f);

  // Pre-solve normal relative velocity (wall is stationary: v_wall = 0)
  glm::vec3 v_contact = A.v + glm::cross(A.w, m.r_a);
  m.v_n_pre = -glm::dot(m.normal, v_contact);  // relative: wall − body

  m.restitution = restitution;

  m.lambda_n  = 0.0f;
  m.lambda_t1 = 0.0f;
  m.lambda_t2 = 0.0f;

  manifolds_.push_back(m);
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

    // Pre-cache Iinv * (r × d) for impulse application in PSOR loop.
    m.invMassA = A.invMass;
    m.invMassB = B.invMass;
    m.IinvA_rAxn  = IinvA * glm::cross(m.r_a, m.normal);
    m.IinvA_rAxt1 = IinvA * glm::cross(m.r_a, m.t1);
    m.IinvA_rAxt2 = IinvA * glm::cross(m.r_a, m.t2);
    m.IinvB_rBxn  = IinvB * glm::cross(m.r_b, m.normal);
    m.IinvB_rBxt1 = IinvB * glm::cross(m.r_b, m.t1);
    m.IinvB_rBxt2 = IinvB * glm::cross(m.r_b, m.t2);

    // Pre-solve normal relative velocity (before any impulse).
    // v_rel = v_B_contact − v_A_contact; v_n = dot(n, v_rel).
    glm::vec3 v_rel_pre;
    if (m.isWall) {
      v_rel_pre = -(A.v + glm::cross(A.w, m.r_a));
    } else {
      v_rel_pre = (B.v + glm::cross(B.w, m.r_b))
                - (A.v + glm::cross(A.w, m.r_a));
    }
    m.v_n_pre = glm::dot(m.normal, v_rel_pre);

    // Combined restitution: use minimum (conservative) of the two bodies.
    m.restitution = std::min(A.restitution, B.restitution);

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

  // When position stabilization is active, use a very small Baumgarte bias
  // (1% of nominal) so that penetrating contacts still seed a nonzero normal
  // impulse — and hence a nonzero friction cone.  The bias is deliberately
  // tiny to avoid the energy-injection instability that occurs with the
  // standard Baumgarte + friction coupling at high mu.
  const float beta  = cfg_.position_stabilization
                        ? cfg_.beta * 0.01f
                        : cfg_.beta;

  // Apply warm-start impulses before iterating (uses pre-cached terms).
  if (cfg_.enable_warm_start) {
    for (const auto& m : manifolds_) {
      auto& A = bodies[m.body_a];
      const bool wall = m.isWall;

      // Normal warm-start
      if (m.lambda_n != 0.0f) {
        A.v -= m.normal * (m.lambda_n * m.invMassA);
        A.w -= m.IinvA_rAxn * m.lambda_n;
        if (!wall) {
          auto& B = bodies[m.body_b];
          B.v += m.normal * (m.lambda_n * m.invMassB);
          B.w += m.IinvB_rBxn * m.lambda_n;
        }
      }
      // Tangent-1 warm-start
      if (m.lambda_t1 != 0.0f) {
        A.v -= m.t1 * (m.lambda_t1 * m.invMassA);
        A.w -= m.IinvA_rAxt1 * m.lambda_t1;
        if (!wall) {
          auto& B = bodies[m.body_b];
          B.v += m.t1 * (m.lambda_t1 * m.invMassB);
          B.w += m.IinvB_rBxt1 * m.lambda_t1;
        }
      }
      // Tangent-2 warm-start
      if (m.lambda_t2 != 0.0f) {
        A.v -= m.t2 * (m.lambda_t2 * m.invMassA);
        A.w -= m.IinvA_rAxt2 * m.lambda_t2;
        if (!wall) {
          auto& B = bodies[m.body_b];
          B.v += m.t2 * (m.lambda_t2 * m.invMassB);
          B.w += m.IinvB_rBxt2 * m.lambda_t2;
        }
      }
    }
  }

  // PSOR iterations with early termination.
  float maxResidual = 0.0f;
  constexpr float convergenceTol = 1e-6f;
  for (int iter = 0; iter < cfg_.velocity_iters; ++iter) {
    float iterResidual = 0.0f;
    for (auto& m : manifolds_) {
      auto& A = bodies[m.body_a];
      const bool wall = m.isWall;

      // Compute v_rel once for all three constraint directions.
      glm::vec3 v_rel = wall
          ? -(A.v + glm::cross(A.w, m.r_a))
          : (bodies[m.body_b].v + glm::cross(bodies[m.body_b].w, m.r_b))
            - (A.v + glm::cross(A.w, m.r_a));

      // ── Normal constraint (unilateral: λ_n ≥ 0) ──
      // Baumgarte bias only for penetrating contacts (phi < 0).
      // For separated contacts (phi > 0), bias = 0 so the solver still
      // generates a normal impulse whenever v_n < 0 (approaching).
      float b_n = (m.phi < 0.0f) ? beta * m.phi / dt : 0.0f;
      float b_rest = (m.v_n_pre < 0.0f) ? -m.restitution * m.v_n_pre : 0.0f;

      float w_n = glm::dot(m.normal, v_rel) + b_n + b_rest + cfm * m.lambda_n;
      float delta_n = -(omega / m.g_n) * w_n;
      float old_n = m.lambda_n;
      m.lambda_n = std::max(0.0f, old_n + delta_n);
      float dn = m.lambda_n - old_n;

      if (dn != 0.0f) {
        // Update velocities using pre-cached Iinv*(r×d) terms.
        glm::vec3 dvA = m.normal * (dn * m.invMassA);
        A.v -= dvA;
        A.w -= m.IinvA_rAxn * dn;
        // Update v_rel incrementally: v_rel += dvA + cross(dwA, r_a)
        v_rel += dvA + glm::cross(m.IinvA_rAxn * dn, m.r_a);
        if (!wall) {
          auto& B = bodies[m.body_b];
          glm::vec3 dvB = m.normal * (dn * m.invMassB);
          B.v += dvB;
          B.w += m.IinvB_rBxn * dn;
          v_rel += dvB + glm::cross(m.IinvB_rBxn * dn, m.r_b);
        }
      }

      // ── Tangent-1 friction (coupled bounds: |λ_t| ≤ μ·λ_n) ──
      float max_fric = mu * m.lambda_n;
      float limit_t1 = std::sqrt(std::max(0.0f, max_fric * max_fric - m.lambda_t2 * m.lambda_t2));

      float w_t1 = glm::dot(m.t1, v_rel) + cfm * m.lambda_t1;
      float delta_t1 = -(omega / m.g_t1) * w_t1;
      float old_t1 = m.lambda_t1;
      m.lambda_t1 = std::clamp(old_t1 + delta_t1, -limit_t1, limit_t1);
      float dt1 = m.lambda_t1 - old_t1;

      if (dt1 != 0.0f) {
        glm::vec3 dvA = m.t1 * (dt1 * m.invMassA);
        A.v -= dvA;
        A.w -= m.IinvA_rAxt1 * dt1;
        v_rel += dvA + glm::cross(m.IinvA_rAxt1 * dt1, m.r_a);
        if (!wall) {
          auto& B = bodies[m.body_b];
          glm::vec3 dvB = m.t1 * (dt1 * m.invMassB);
          B.v += dvB;
          B.w += m.IinvB_rBxt1 * dt1;
          v_rel += dvB + glm::cross(m.IinvB_rBxt1 * dt1, m.r_b);
        }
      }

      // ── Tangent-2 friction (coupled bounds: |λ_t| ≤ μ·λ_n) ──
      float limit_t2 = std::sqrt(std::max(0.0f, max_fric * max_fric - m.lambda_t1 * m.lambda_t1));

      float w_t2 = glm::dot(m.t2, v_rel) + cfm * m.lambda_t2;
      float delta_t2 = -(omega / m.g_t2) * w_t2;
      float old_t2 = m.lambda_t2;
      m.lambda_t2 = std::clamp(old_t2 + delta_t2, -limit_t2, limit_t2);
      float dt2 = m.lambda_t2 - old_t2;

      if (dt2 != 0.0f) {
        A.v -= m.t2 * (dt2 * m.invMassA);
        A.w -= m.IinvA_rAxt2 * dt2;
        if (!wall) {
          auto& B = bodies[m.body_b];
          B.v += m.t2 * (dt2 * m.invMassB);
          B.w += m.IinvB_rBxt2 * dt2;
        }
      }

      // Track per-constraint residual (sum of absolute deltas).
      iterResidual = std::max(iterResidual,
          std::abs(dn) + std::abs(dt1) + std::abs(dt2));
    }
    maxResidual = iterResidual;
    if (iterResidual < convergenceTol) { break; }
  }
  lastResidual_ = maxResidual;

  // Update warm-start cache for next frame.
  warmCache_.clear();
  if (cfg_.enable_warm_start) {
    for (const auto& m : manifolds_) {
      int lo = std::min(m.body_a, m.body_b);
      int hi = std::max(m.body_a, m.body_b);
      warmCache_[{lo, hi}] = {m.lambda_n, m.lambda_t1, m.lambda_t2};
    }
  }
}

// ─── Position Stabilization ─────────────────────────────────────────────────

void NscContactSolver::projectPositions(std::vector<RigidBody>& bodies) {

  if (!cfg_.position_stabilization) { return; }

  const float omega = cfg_.omega;
  const float cfm   = cfg_.cfm;
  const float slop  = cfg_.slop;

  // Reuse persistent scratch buffers to avoid per-frame allocation.
  const size_t N = bodies.size();
  posDx_.resize(N);
  posDtheta_.resize(N);
  posIinv_.resize(N);

  struct PosManifold {
    int body_a, body_b;
    bool isWall = false;
    glm::vec3 normal;
    glm::vec3 r_a, r_b;
    float phi;
    float g_n;
    float lambda_n;
  };
  std::vector<PosManifold> pm;

  // Cache body pairs from first iteration's broadphase.
  // Subsequent iterations recompute narrowphase only for these pairs,
  // skipping the O(N²) broadphase which dominates detection cost.
  struct CachedPair {
    int body_a, body_b;
    double surface_limit;  // r_a + r_b
  };
  std::vector<CachedPair> cachedPairs;

  for (int outer = 0; outer < cfg_.position_iters; ++outer) {

    pm.clear();

    if (outer == 0) {
      // First iteration: full broadphase + narrowphase.
      detector_.detectContacts(bodies);
      const auto& contacts = detector_.getContacts();

      // Cache all detected pairs (not just penetrating ones) for subsequent
      // iterations, since corrections might push a near-miss pair into contact.
      cachedPairs.clear();
      cachedPairs.reserve(contacts.size());
      for (const auto& cp : contacts) {
        cachedPairs.push_back({cp.body_a, cp.body_b, cp.surface_limit});
      }

      pm.reserve(contacts.size());
      for (const auto& cp : contacts) {
        float phi = static_cast<float>(cp.distance - cp.surface_limit);
        if (phi >= -slop) continue;

        PosManifold p;
        p.body_a  = cp.body_a;
        p.body_b  = cp.body_b;
        p.normal  = cp.normal;
        p.phi     = phi + slop;
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
    } else {
      // Subsequent iterations: narrowphase only for cached pairs.
      pm.reserve(cachedPairs.size());

      for (const auto& cp : cachedPairs) {
        const auto& A = bodies[cp.body_a];
        const auto& B = bodies[cp.body_b];

        // Recompute capsule endpoints.
        glm::vec3 axA = A.axisY();
        glm::vec3 axB = B.axisY();
        glm::vec3 a1 = A.x - axA * A.cap.h;
        glm::vec3 a2 = A.x + axA * A.cap.h;
        glm::vec3 b1 = B.x - axB * B.cap.h;
        glm::vec3 b2 = B.x + axB * B.cap.h;
        glm::vec3 shift_b(0.0f);

        if (pbcEnabled_) {
          glm::vec3 delta = B.x - A.x;
          for (int k = 0; k < 3; ++k) {
            if (pbcSize_[k] > 0.0f) {
              float n = std::floor(delta[k] / pbcSize_[k] + 0.5f);
              shift_b[k] = -n * pbcSize_[k];
            }
          }
          b1 += shift_b;
          b2 += shift_b;
        }

        double s, t;
        closestPointsSegSeg(a1, a2, b1, b2, s, t);

        glm::vec3 ptA = a1 + float(s) * (a2 - a1);
        glm::vec3 ptB = b1 + float(t) * (b2 - b1);
        glm::vec3 diff = ptB - ptA;
        float dist = glm::length(diff);
        float phi = dist - float(cp.surface_limit);

        if (phi >= -slop) continue;

        PosManifold p;
        p.body_a = cp.body_a;
        p.body_b = cp.body_b;
        p.normal = (dist > 1e-10f) ? diff / dist : glm::vec3(1, 0, 0);
        p.phi    = phi + slop;
        p.r_a    = ptA - A.x;
        p.r_b    = ptB - (B.x + shift_b);
        p.g_n    = computeDiag(A.invMass, B.invMass,
                               A.IworldInv(), B.IworldInv(),
                               p.r_a, p.r_b, p.normal, cfm);
        p.lambda_n = 0.0f;
        pm.push_back(p);
      }
    }

    if (pm.empty()) { return; } // No significant penetration; done.

    // Zero scratch buffers.
    std::fill(posDx_.begin(), posDx_.end(), glm::vec3(0.0f));
    std::fill(posDtheta_.begin(), posDtheta_.end(), glm::vec3(0.0f));

    // Cache inverse inertia tensors (orientations unchanged within this pass).
    for (size_t i = 0; i < N; ++i)
      posIinv_[i] = bodies[i].IworldInv();

    constexpr float posTol = 1e-8f;
    for (int inner = 0; inner < cfg_.position_psor_iters; ++inner) {
      float posResidual = 0.0f;
      for (auto& p : pm) {
        // Residual: projection of accumulated correction onto normal + gap.
        float corr_rel = glm::dot(p.normal,
               (p.isWall ? glm::vec3(0.0f) : (posDx_[p.body_b] + glm::cross(posDtheta_[p.body_b], p.r_b)))
             - posDx_[p.body_a] - glm::cross(posDtheta_[p.body_a], p.r_a));
        float residual = corr_rel + p.phi + cfm * p.lambda_n;

        float delta = -(omega / p.g_n) * residual;
        float old_lambda = p.lambda_n;
        p.lambda_n = std::max(0.0f, old_lambda + delta);
        float dl = p.lambda_n - old_lambda;

        if (dl != 0.0f) {
          glm::vec3 rAxn = glm::cross(p.r_a, p.normal);

          posDx_[p.body_a]     -= p.normal * (dl * bodies[p.body_a].invMass);
          posDtheta_[p.body_a] -= posIinv_[p.body_a] * rAxn * dl;

          if (!p.isWall) {
            glm::vec3 rBxn = glm::cross(p.r_b, p.normal);
            posDx_[p.body_b]     += p.normal * (dl * bodies[p.body_b].invMass);
            posDtheta_[p.body_b] += posIinv_[p.body_b] * rBxn * dl;
          }
        }
        posResidual = std::max(posResidual, std::abs(dl));
      }
      if (posResidual < posTol) break; // Converged early
    }

    // Apply accumulated position + orientation corrections.
    for (size_t i = 0; i < N; ++i) {
      if (bodies[i].invMass <= 0.0f) continue;
      bodies[i].x += posDx_[i];

      float angle = glm::length(posDtheta_[i]);
      if (angle > 1e-8f) {
        glm::vec3 axis = posDtheta_[i] / angle;
        glm::quat dq = glm::angleAxis(angle, axis);
        bodies[i].q = glm::normalize(dq * bodies[i].q);
      }
    }
  }
}
