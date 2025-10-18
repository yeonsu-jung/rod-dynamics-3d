/**
 * @file solver.cpp
 * @brief Physics constraint solver for impulse-based resolution
 */

#include "physics/solver.hpp"
#include "physics/types.hpp"
#include <glm/gtc/quaternion.hpp>
#include <algorithm>
#include <cmath>

namespace {
    constexpr float BOUNCE_THRESHOLD = 0.0f; ///< Honor restitution at all impact speeds
}

// Diagnostic accumulators
double g_diag_jn_sum = 0.0;
double g_diag_jt_sum = 0.0;
int    g_diag_impulse_count = 0;

bool g_energy_safeguard = false;
static bool g_disable_warmstart = false;

void resetFrameImpulseAccumulators() {
    g_diag_jn_sum = 0.0;
    g_diag_jt_sum = 0.0;
    g_diag_impulse_count = 0;
}

void setEnergySafeguard(bool enabled) {
    g_energy_safeguard = enabled;
}

void setWarmstartEnabled(bool enabled) {
    g_disable_warmstart = !enabled;
}

// Helper to optionally scale an impulse J applied at points rA/rB to avoid increasing KE
static void applyImpulseWithSafeguard(RigidBody& bodyA, RigidBody& bodyB, const glm::vec3& J, const glm::vec3& rA, const glm::vec3& rB) {
    if (!g_energy_safeguard) {
        bodyA.v -= J * bodyA.invMass;
        bodyB.v += J * bodyB.invMass;
        bodyA.w -= bodyA.IworldInv() * glm::cross(rA, J);
        bodyB.w += bodyB.IworldInv() * glm::cross(rB, J);
        return;
    }
    // Compute pre-impulse KE for bodies (linear + rotational)
    double KE_before = 0.0;
    double va2 = double(glm::dot(bodyA.v, bodyA.v));
    KE_before += 0.5 * double(bodyA.mass) * va2;
    glm::mat3 Ia = bodyA.R() * bodyA.I_body * glm::transpose(bodyA.R());
    glm::vec3 Ia_w = Ia * bodyA.w; KE_before += 0.5 * double(glm::dot(bodyA.w, Ia_w));

    double vb2 = double(glm::dot(bodyB.v, bodyB.v));
    KE_before += 0.5 * double(bodyB.mass) * vb2;
    glm::mat3 Ib = bodyB.R() * bodyB.I_body * glm::transpose(bodyB.R());
    glm::vec3 Ib_w = Ib * bodyB.w; KE_before += 0.5 * double(glm::dot(bodyB.w, Ib_w));

    // Apply impulse tentatively
    RigidBody copyA = bodyA; RigidBody copyB = bodyB;
    copyA.v -= J * copyA.invMass;
    copyB.v += J * copyB.invMass;
    copyA.w -= copyA.IworldInv() * glm::cross(rA, J);
    copyB.w += copyB.IworldInv() * glm::cross(rB, J);

    double KE_after = 0.0;
    double va2a = double(glm::dot(copyA.v, copyA.v));
    KE_after += 0.5 * double(copyA.mass) * va2a;
    glm::mat3 Ia2 = copyA.R() * copyA.I_body * glm::transpose(copyA.R());
    glm::vec3 Ia_w2 = Ia2 * copyA.w; KE_after += 0.5 * double(glm::dot(copyA.w, Ia_w2));

    double vb2b = double(glm::dot(copyB.v, copyB.v));
    KE_after += 0.5 * double(copyB.mass) * vb2b;
    glm::mat3 Ib2 = copyB.R() * copyB.I_body * glm::transpose(copyB.R());
    glm::vec3 Ib_w2 = Ib2 * copyB.w; KE_after += 0.5 * double(glm::dot(copyB.w, Ib_w2));

    if (KE_after <= KE_before + 1e-9) {
        // safe to apply
        bodyA = copyA; bodyB = copyB; return;
    }

    // Otherwise scale down J via bisection until KE_after <= KE_before (or small tolerance)
    glm::vec3 J0 = J; float lo = 0.0f, hi = 1.0f;
    for (int it = 0; it < 12; ++it) {
        float mid = 0.5f * (lo + hi);
        glm::vec3 Jc = mid * J0;
        RigidBody ca = bodyA; RigidBody cb = bodyB;
        ca.v -= Jc * ca.invMass; cb.v += Jc * cb.invMass;
        ca.w -= ca.IworldInv() * glm::cross(rA, Jc); cb.w += cb.IworldInv() * glm::cross(rB, Jc);
        double KEc = 0.0;
        KEc += 0.5 * double(ca.mass) * double(glm::dot(ca.v, ca.v));
        glm::mat3 Ia3 = ca.R() * ca.I_body * glm::transpose(ca.R()); glm::vec3 Ia_w3 = Ia3 * ca.w; KEc += 0.5 * double(glm::dot(ca.w, Ia_w3));
        KEc += 0.5 * double(cb.mass) * double(glm::dot(cb.v, cb.v));
        glm::mat3 Ib3 = cb.R() * cb.I_body * glm::transpose(cb.R()); glm::vec3 Ib_w3 = Ib3 * cb.w; KEc += 0.5 * double(glm::dot(cb.w, Ib_w3));
        if (KEc <= KE_before + 1e-9) { lo = mid; } else { hi = mid; }
    }
    glm::vec3 Jc = 0.5f * (lo + hi) * J0;
    bodyA.v -= Jc * bodyA.invMass; bodyB.v += Jc * bodyB.invMass;
    bodyA.w -= bodyA.IworldInv() * glm::cross(rA, Jc);
    bodyB.w += bodyB.IworldInv() * glm::cross(rB, Jc);
}

void applyImpulse(RigidBody& bodyA, RigidBody& bodyB, const Contact& contact, AppliedImpulse* out) {
    glm::vec3 rA = contact.point - bodyA.x;
    glm::vec3 rB = contact.point - (bodyB.x + contact.shiftB);

    // Calculate relative velocity at contact point
    glm::vec3 vA = bodyA.v + glm::cross(bodyA.w, rA);
    glm::vec3 vB = bodyB.v + glm::cross(bodyB.w, rB);
    glm::vec3 relativeVelocity = vB - vA;

    float normalVelocity = glm::dot(relativeVelocity, contact.normal);
    if (normalVelocity > 0.0f) { if (out) *out = {}; return; }

    // Calculate restitution (bouncing) - disable for gentle impacts
    float restitution = std::min(bodyA.restitution, bodyB.restitution);
    if (std::abs(normalVelocity) < BOUNCE_THRESHOLD) {
        restitution = 0.0f;
    }

    glm::mat3 invInertiaA = bodyA.IworldInv();
    glm::mat3 invInertiaB = bodyB.IworldInv();

    // Lambda function to compute effective mass in a given direction
    auto computeEffectiveMass = [&](const glm::vec3& direction) -> float {
        glm::vec3 crossA = glm::cross(rA, direction);
        glm::vec3 crossB = glm::cross(rB, direction);
        float effectiveMass = bodyA.invMass + bodyB.invMass
                            + glm::dot(direction, glm::cross(invInertiaA * crossA, rA))
                            + glm::dot(direction, glm::cross(invInertiaB * crossB, rB));
        return (effectiveMass > 1e-8f) ? effectiveMass : 1e-8f;
    };

    // Calculate normal impulse
    float normalEffectiveMass = computeEffectiveMass(contact.normal);
    float normalImpulseMagnitude = -(1.0f + restitution) * normalVelocity / normalEffectiveMass;
    glm::vec3 normalImpulse = normalImpulseMagnitude * contact.normal;

    // Apply normal impulse
    applyImpulseWithSafeguard(bodyA, bodyB, normalImpulse, rA, rB);

    // Accumulate diagnostics for normal
    g_diag_jn_sum += std::abs(normalImpulseMagnitude);
    g_diag_impulse_count += 1;

    // Recompute relative velocity for friction calculation
    vA = bodyA.v + glm::cross(bodyA.w, rA);
    vB = bodyB.v + glm::cross(bodyB.w, rB);
    relativeVelocity = vB - vA;

    // Calculate friction (Coulomb model with optional static/dynamic params)
    glm::vec3 tangent = relativeVelocity - contact.normal * glm::dot(relativeVelocity, contact.normal);
    float vt = glm::length(tangent);

    float mu_d_A = (bodyA.frictionD > 0.0f) ? bodyA.frictionD : bodyA.friction;
    float mu_d_B = (bodyB.frictionD > 0.0f) ? bodyB.frictionD : bodyB.friction;
    float mu_s_A = (bodyA.frictionS > 0.0f) ? bodyA.frictionS : mu_d_A;
    float mu_s_B = (bodyB.frictionS > 0.0f) ? bodyB.frictionS : mu_d_B;

    float mu_d = 0.5f * (mu_d_A + mu_d_B);
    float mu_s = 0.5f * (mu_s_A + mu_s_B);

    float tangentImpulseMagnitude = 0.0f;
    if (vt > 1e-5f) {
        tangent /= vt;
        float tangentEffectiveMass = computeEffectiveMass(tangent);
        float jt_free = -glm::dot(relativeVelocity, tangent) / tangentEffectiveMass;

        // Static vs dynamic: allow up to mu_s*jn if near-sticking, else clamp to mu_d*jn
        float mu_cap = mu_d;
        if (std::abs(jt_free) < mu_d * normalImpulseMagnitude * 1.1f) {
            mu_cap = mu_s;
        }
        tangentImpulseMagnitude = glm::clamp(jt_free, -mu_cap * normalImpulseMagnitude, mu_cap * normalImpulseMagnitude);

        glm::vec3 frictionImpulse = tangentImpulseMagnitude * tangent;
        applyImpulseWithSafeguard(bodyA, bodyB, frictionImpulse, rA, rB);

        // Accumulate diagnostics for tangent
        g_diag_jt_sum += std::abs(tangentImpulseMagnitude);
    }

    if (out) {
        out->jn = normalImpulseMagnitude;
        out->jt = tangentImpulseMagnitude;
        out->tangent = (vt > 1e-5f) ? tangent : glm::vec3(0);
    }
}

void applyNormalImpulse(RigidBody& bodyA, RigidBody& bodyB, const Contact& contact, AppliedImpulse* out) {
    glm::vec3 rA = contact.point - bodyA.x;
    glm::vec3 rB = contact.point - (bodyB.x + contact.shiftB);

    glm::vec3 vA = bodyA.v + glm::cross(bodyA.w, rA);
    glm::vec3 vB = bodyB.v + glm::cross(bodyB.w, rB);
    glm::vec3 relativeVelocity = vB - vA;

    float vn = glm::dot(relativeVelocity, contact.normal);
    if (vn > 0.0f) { if (out) *out = {}; return; }

    float restitution = std::min(bodyA.restitution, bodyB.restitution);
    if (std::abs(vn) < BOUNCE_THRESHOLD) restitution = 0.0f;

    auto computeEffectiveMass = [&](const glm::vec3& dir){
        glm::vec3 crossA = glm::cross(rA, dir);
        glm::vec3 crossB = glm::cross(rB, dir);
        float K = bodyA.invMass + bodyB.invMass
                + glm::dot(dir, glm::cross(bodyA.IworldInv()*crossA, rA))
                + glm::dot(dir, glm::cross(bodyB.IworldInv()*crossB, rB));
        return (K > 1e-8f) ? K : 1e-8f;
    };

    float Kn = computeEffectiveMass(contact.normal);
    float jn = -(1.0f + restitution) * vn / Kn;
    glm::vec3 J = jn * contact.normal;
    applyImpulseWithSafeguard(bodyA, bodyB, J, rA, rB);

    g_diag_jn_sum += std::abs(jn);
    g_diag_impulse_count += 1;

    if (out) { out->jn = jn; out->jt = 0.0f; out->tangent = glm::vec3(0); }
}

// Targeted restitution-accurate sweeps on high-speed impacts (normal-only)
void ngsRestitutionSweeps(std::vector<PairContact>& contacts, std::vector<RigidBody>& rods, const SolverConfig& cfg) {
    if (cfg.ngsNormalSweeps <= 0) return;
    if (contacts.empty()) return;
    const float vth = std::max(0.0f, cfg.ngsHighVThresh);

    // Filter high-approach contacts once (approaching speed along normal >= vth)
    std::vector<size_t> hot;
    hot.reserve(contacts.size());
    for (size_t i = 0; i < contacts.size(); ++i) {
        const auto& ac = contacts[i];
        if (ac.b < 0) continue;
        const Contact& c = ac.c;
        const RigidBody& A = rods[ac.a];
        const RigidBody& B = rods[ac.b];
        glm::vec3 rA = c.point - A.x;
        glm::vec3 rB = c.point - (B.x + c.shiftB);
        glm::vec3 vA = A.v + glm::cross(A.w, rA);
        glm::vec3 vB = B.v + glm::cross(B.w, rB);
        float vn = glm::dot(vB - vA, c.normal);
        if (-vn >= vth) hot.push_back(i);
    }
    if (hot.empty()) return;

    for (int it = 0; it < cfg.ngsNormalSweeps; ++it) {
        for (size_t idx : hot) {
            auto& ac = contacts[idx];
            if (ac.b < 0) continue;
            applyNormalImpulse(rods[ac.a], rods[ac.b], ac.c, nullptr);
        }
    }
}

void positionalCorrection(RigidBody& bodyA, RigidBody& bodyB, const Contact& contact, const SolverConfig& config) {
    float correctionAmount = std::max(0.0f, contact.penetration - config.allowedPen);
    if (correctionAmount <= 0) return;

    const glm::vec3 n = contact.normal;
    const float beta = config.baumgarte;
    const float d = beta * correctionAmount;

    if (!config.splitImpulse) {
        // Simple positional projection along normal (mass-weighted)
        glm::vec3 correction = d * n / (bodyA.invMass + bodyB.invMass + 1e-8f);
        bodyA.x -= bodyA.invMass * correction;
        bodyB.x += bodyB.invMass * correction;
        return;
    }

    // Split-impulse positional correction using effective mass including inertia
    glm::vec3 rA = contact.point - bodyA.x;
    glm::vec3 rB = contact.point - (bodyB.x + contact.shiftB);
    glm::mat3 invIA = bodyA.IworldInv();
    glm::mat3 invIB = bodyB.IworldInv();

    glm::vec3 raXn = glm::cross(rA, n);
    glm::vec3 rbXn = glm::cross(rB, n);
    float K = bodyA.invMass + bodyB.invMass
            + glm::dot(n, glm::cross(invIA * raXn, rA))
            + glm::dot(n, glm::cross(invIB * rbXn, rB));
    if (K < 1e-8f) K = 1e-8f;

    float P = d / K; // positional-correction scalar along n
    glm::vec3 Pn = P * n;

    // Apply translation part
    bodyA.x -= bodyA.invMass * Pn;
    bodyB.x += bodyB.invMass * Pn;

    // Apply small-angle orientation correction derived from positional split impulse
    glm::vec3 dThetaA = invIA * glm::cross(rA, Pn);
    glm::vec3 dThetaB = invIB * glm::cross(rB, Pn);

    auto applySmallRot = [](RigidBody& rb, const glm::vec3& dth){
        float ang = glm::length(dth);
        if (ang < 1e-6f) return;
        float maxAng = 0.2f; // clamp to avoid instability
        float sc = std::min(1.0f, maxAng / ang);
        glm::vec3 axis = dth * sc / ang;
        glm::quat dq = glm::angleAxis(ang * sc, axis);
        rb.q = glm::normalize(dq * rb.q);
    };
    if (config.splitOrient) {
        applySmallRot(bodyA, -dThetaA);
        applySmallRot(bodyB, +dThetaB);
    }
}

void applyWarmStart(RigidBody& bodyA, RigidBody& bodyB, const Contact& c, float jn, float jt, const glm::vec3& tangent) {
    if (g_disable_warmstart) return;
    glm::vec3 rA = c.point - bodyA.x;
    glm::vec3 rB = c.point - (bodyB.x + c.shiftB);

    glm::vec3 J = jn * c.normal + jt * tangent;

    applyImpulseWithSafeguard(bodyA, bodyB, J, rA, rB);

    // Accumulate diagnostics for warm-start as well
    g_diag_jn_sum += std::abs(jn);
    g_diag_jt_sum += std::abs(jt);
    if (std::abs(jn) > 0.0f || std::abs(jt) > 0.0f) ++g_diag_impulse_count;
}
