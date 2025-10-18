#pragma once
#include "rigid_body.hpp"
#include "types.hpp"

struct AppliedImpulse {
    float jn = 0.0f;         // normal impulse magnitude
    float jt = 0.0f;         // friction impulse magnitude (signed along tangent)
    glm::vec3 tangent{0.0f}; // tangent direction used
};

void applyImpulse(RigidBody& A, RigidBody& B, const Contact& c, AppliedImpulse* out = nullptr);
void positionalCorrection(RigidBody& A, RigidBody& B, const Contact& c, const SolverConfig& cfg);

// Apply a precomputed impulse (warm start): J = jn * n + jt * t
void applyWarmStart(RigidBody& A, RigidBody& B, const Contact& c, float jn, float jt, const glm::vec3& tangent);
