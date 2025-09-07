#include "physics/solver.hpp"
#include "physics/types.hpp"
#include <algorithm>
#include <cmath>

void applyImpulse(RigidBody& A, RigidBody& B, const Contact& c)
{
    glm::vec3 rA = c.point - A.x;
    glm::vec3 rB = c.point - B.x;

    glm::vec3 vA = A.v + glm::cross(A.w, rA);
    glm::vec3 vB = B.v + glm::cross(B.w, rB);
    glm::vec3 rv = vB - vA;

    float rvn = glm::dot(rv, c.normal);
    if (rvn > 0.0f) return;

    // gentle impacts don't bounce
    constexpr float bounceThreshold = 0.4f;
    float e = std::min(A.restitution, B.restitution);
    if (std::abs(rvn) < bounceThreshold) e = 0.0f;

    glm::mat3 IA = A.IworldInv();
    glm::mat3 IB = B.IworldInv();

    auto K_scalar = [&](const glm::vec3& n){
        glm::vec3 rnA = glm::cross(rA, n);
        glm::vec3 rnB = glm::cross(rB, n);
        float k = A.invMass + B.invMass
                + glm::dot(n, glm::cross(IA * rnA, rA))
                + glm::dot(n, glm::cross(IB * rnB, rB));
        return (k > 1e-8f) ? k : 1e-8f;
    };

    // normal impulse
    float kN = K_scalar(c.normal);
    float j = -(1.0f + e) * rvn / kN;
    glm::vec3 impulseN = j * c.normal;

    A.v -= impulseN * A.invMass;  B.v += impulseN * B.invMass;
    A.w -= IA * glm::cross(rA, impulseN);
    B.w += IB * glm::cross(rB, impulseN);

    // recompute relative velocity
    vA = A.v + glm::cross(A.w, rA);
    vB = B.v + glm::cross(B.w, rB);
    rv = vB - vA;

    // friction (Coulomb)
    glm::vec3 t = rv - c.normal * glm::dot(rv, c.normal);
    float tlen = glm::length(t);

    // tiny slip? approximate static friction snap-to-rest
    constexpr float slipEps = 1e-3f;
    if (tlen < slipEps) return;

    t /= tlen;
    float kT = K_scalar(t);
    float jt = -glm::dot(rv, t) / kT;

    float mu = 0.5f * (A.friction + B.friction);
    jt = glm::clamp(jt, -mu*j, mu*j);

    glm::vec3 impulseT = jt * t;
    A.v -= impulseT * A.invMass;  B.v += impulseT * B.invMass;
    A.w -= IA * glm::cross(rA, impulseT);
    B.w += IB * glm::cross(rB, impulseT);
}

void positionalCorrection(RigidBody& A, RigidBody& B, const Contact& c, const SolverConfig& cfg)
{
    float k = std::max(0.0f, c.penetration - cfg.allowedPen);
    if (k <= 0) return;
    glm::vec3 corr = (cfg.baumgarte * k) * c.normal / (A.invMass + B.invMass + 1e-8f);
    A.x -= A.invMass * corr;
    B.x += B.invMass * corr;
}
