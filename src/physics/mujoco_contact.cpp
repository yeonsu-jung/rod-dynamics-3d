/**
 * @file mujoco_contact.cpp
 * @brief MuJoCo-inspired penalty contact model (separate from SoftContactSolver).
 */

#include "physics/mujoco_contact.hpp"

#include <algorithm>
#include <cmath>

MujocoContactSolver::MujocoContactSolver(const MujocoContactCfg& cfg)
    : cfg_(cfg)
{
}

void MujocoContactSolver::setConfig(const MujocoContactCfg& cfg) {
    cfg_ = cfg;
    lastPotentialEnergy_ = 0.0;
}

void MujocoContactSolver::detectContacts(const std::vector<RigidBody>& bodies) {
    contacts_.clear();
    const size_t N = bodies.size();
    if (!cfg_.enabled || N < 2) return;

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            detectCapsuleCapsule(bodies[i], bodies[j], static_cast<int>(i), static_cast<int>(j));
        }
    }
}

void MujocoContactSolver::detectCapsuleCapsule(const RigidBody& a, const RigidBody& b,
                                               int ia, int ib) {
    // Treat each rod as a capsule aligned with its local +Y axis.
    const glm::vec3 axisA = a.axisY();
    const glm::vec3 axisB = b.axisY();

    const glm::vec3 a1 = a.x - axisA * a.cap.h;
    const glm::vec3 a2 = a.x + axisA * a.cap.h;
    const glm::vec3 b1 = b.x - axisB * b.cap.h;
    const glm::vec3 b2 = b.x + axisB * b.cap.h;

    const double rA = a.cap.r;
    const double rB = b.cap.r;
    const double h = rA + rB;  // surface separation at contact

    double s = 0.0, t = 0.0;
    closestPointsSegmentSegment(a1, a2, b1, b2, s, t);

    const glm::vec3 pA = a1 + static_cast<float>(s) * (a2 - a1);
    const glm::vec3 pB = b1 + static_cast<float>(t) * (b2 - b1);
    glm::vec3 diff = pB - pA;
    double dist = glm::length(diff);

    // Only consider contacts up to some activation distance beyond h
    const double activation_dist = h * 1.5; // simple heuristic buffer
    if (dist > activation_dist) return;

    Contact c;
    c.a = ia;
    c.b = ib;
    c.pA = pA;
    c.pB = pB;
    c.dist = dist;
    c.surface_limit = h;
    if (dist > 1e-9) {
        c.n = diff / static_cast<float>(dist);
    } else {
        // Degenerate: pick arbitrary normal
        c.n = glm::vec3(1.0f, 0.0f, 0.0f);
    }
    contacts_.push_back(c);
}

void MujocoContactSolver::computeForces(std::vector<RigidBody>& bodies, double dt) {
    if (!cfg_.enabled || contacts_.empty()) {
        lastPotentialEnergy_ = 0.0;
        return;
    }

    double pe_sum = 0.0;

    for (const auto& c : contacts_) {
        RigidBody& A = bodies[c.a];
        RigidBody& B = bodies[c.b];

        // Effective penetration depth (positive when overlapping)
        const double d = std::max(0.0, c.surface_limit - c.dist);
        if (d <= 0.0) continue;

        // Relative velocity at contact
        const glm::vec3 rA = c.pA - A.x;
        const glm::vec3 rB = c.pB - B.x;
        const glm::vec3 vA = A.v + glm::cross(A.w, rA);
        const glm::vec3 vB = B.v + glm::cross(B.w, rB);
        const glm::vec3 vRel = vB - vA;

        const float vn = glm::dot(vRel, c.n);
        const glm::vec3 vN = vn * c.n;
        const glm::vec3 vT = vRel - vN;
        const float vT_mag = glm::length(vT);

        // --- Normal force (simple linear spring-damper) ---
        // f_n = k * d - c_damp * vn   (note vn>0 means separating)
        const double k_n = cfg_.normal_k;
        const double c_n = cfg_.normal_damping;
        double fn = k_n * d - c_n * static_cast<double>(vn);
        if (fn < 0.0) fn = 0.0;  // no attraction

        const glm::vec3 fN = static_cast<float>(fn) * c.n;

        // --- Tangential (friction) force ---
        glm::vec3 fT(0.0f);
        if (cfg_.friction_mu > 0.0 && vT_mag > 0.0f) {
            // Regularized slip speed
            const double vt_eff = std::sqrt(static_cast<double>(vT_mag) * static_cast<double>(vT_mag)
                                            + cfg_.vel_eps * cfg_.vel_eps);
            // Viscous-like friction scaled by normal force
            const double ft_raw_mag = cfg_.friction_mu * fn * (static_cast<double>(vT_mag) / vt_eff);
            glm::vec3 t_dir = -vT / vT_mag; // oppose motion
            fT = static_cast<float>(ft_raw_mag) * t_dir;

            // Clamp to Coulomb cone (|fT| <= mu * fn)
            const float ft_mag = glm::length(fT);
            const float ft_max = static_cast<float>(cfg_.friction_mu * fn);
            if (ft_mag > ft_max && ft_mag > 0.0f) {
                fT *= (ft_max / ft_mag);
            }
        }

        const glm::vec3 fTotal = fN + fT;

        // Apply to bodies A and B
        A.f -= fTotal;
        B.f += fTotal;
        A.tau -= glm::cross(rA, fTotal);
        B.tau += glm::cross(rB, fTotal);

        // Potential energy diagnostic: 0.5 * k * d^2
        pe_sum += 0.5 * k_n * d * d;
    }

    lastPotentialEnergy_ = pe_sum;
}

// Geometric helper: closest points between two segments
void MujocoContactSolver::closestPointsSegmentSegment(
    const glm::vec3& a1, const glm::vec3& a2,
    const glm::vec3& b1, const glm::vec3& b2,
    double& s, double& t)
{
    const glm::vec3 e1 = a2 - a1;
    const glm::vec3 e2 = b2 - b1;
    const glm::vec3 e12 = b1 - a1;

    const double D1 = glm::dot(e1, e1);
    const double D2 = glm::dot(e2, e2);
    const double S1 = glm::dot(e1, e12);
    const double S2 = glm::dot(e2, e12);
    const double R = glm::dot(e1, e2);

    const double den = D1 * D2 - R * R;

    auto clamp01 = [](double& x) {
        if (x < 0.0) x = 0.0;
        else if (x > 1.0) x = 1.0;
    };

    if (std::abs(den) < 1e-12) {
        // Parallel segments: project one endpoint and clamp
        s = 0.0;
        t = (D2 > 0.0) ? -S2 / D2 : 0.0;
        clamp01(t);
    } else {
        s = (S1 * D2 - S2 * R) / den;
        clamp01(s);
        t = (s * R - S2) / D2;
        clamp01(t);
    }
}
