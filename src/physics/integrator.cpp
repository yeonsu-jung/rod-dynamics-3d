/**
 * @file integrator.cpp
 * @brief Physics integration with damping and constraints
 */

#include "physics/integrator.hpp"
#include "physics/types.hpp"
#include "physics/rigid_body.hpp"
#include <glm/gtc/quaternion.hpp>
#include <cmath>

// Global damping parameters (can be modified externally)
float g_lin_damp = 0.08f;  // Linear damping coefficient (s^-1)
float g_ang_damp = 0.12f;  // Angular damping coefficient (s^-1)
float g_w_max = 80.0f;     // Maximum angular velocity (rad/s)

// Periodic boundary globals (configured by app)
bool g_pbc_enabled = false;
glm::vec3 g_pbc_min{-3.0f, -1.0f, -3.0f};
glm::vec3 g_pbc_max{+3.0f, +3.0f, +3.0f};

static inline void wrap_position(glm::vec3& p, const glm::vec3& bmin, const glm::vec3& bmax) {
    const glm::vec3 size = bmax - bmin;
    // Avoid division by zero if dimensions are degenerate
    for (int i = 0; i < 3; ++i) {
        if (size[i] <= 0.0f) continue;
        while (p[i] < bmin[i]) p[i] += size[i];
        while (p[i] >= bmax[i]) p[i] -= size[i];
    }
}

// Compute angular acceleration including gyroscopic term in world frame:
//   I_w wdot + w x (I_w w) = tau  =>  wdot = I_w^{-1} (tau - w x (I_w w))
static inline glm::vec3 angularAccelGyro(const RigidBody& body) {
    // Treat static bodies as having zero acceleration
    if (body.invMass <= 0.0f) {
        return glm::vec3(0.0f);
    }

    // Rotation matrix from body to world
    const glm::mat3 R = body.R();
    const glm::mat3 Iw = R * body.I_body * glm::transpose(R);
    const glm::mat3 Iw_inv = R * body.I_body_inv * glm::transpose(R);

    const glm::vec3 Iw_w = Iw * body.w;
    const glm::vec3 gyro = glm::cross(body.w, Iw_w);

    return Iw_inv * (body.tau - gyro);
}

void integrate(RigidBody& body, const glm::vec3& gravity, float deltaTime) {
    // Skip integration for static bodies
    if (body.invMass <= 0.0f) return;
    
    // ====== SEMI-IMPLICIT (SYMPLECTIC) EULER INTEGRATION ======
    // We previously attempted a single-step Velocity Verlet half-step update, but because
    // forces are only computed once per frame (prior to this call) we never performed the
    // second half velocity update, causing artificial energy decay. We revert to the classic
    // symplectic Euler: first apply full acceleration to velocity, then advance position.
    // This is stable for our small dt (1/12000) and preserves energy better with penalty forces.
    const glm::vec3 acc_lin = (body.f / body.mass) + gravity;
    const glm::vec3 acc_ang = body.I_body_inv * body.tau;

    // Full-step velocity update
    body.v += acc_lin * deltaTime;
    body.w += acc_ang * deltaTime;

    // Position update using updated velocity
    body.x += body.v * deltaTime;
    
    // Periodic wrapping of position
    if (g_pbc_enabled) {
        wrap_position(body.x, g_pbc_min, g_pbc_max);
    }
    
    // Integrate orientation using half-step angular velocity
    float wLen = glm::length(body.w);
    if (wLen > 1e-8f) {
        float angle = wLen * deltaTime;
        glm::vec3 axis = body.w / wLen;
        glm::quat dq = glm::angleAxis(angle, axis);
        body.q = glm::normalize(dq * body.q);
    }
    
    // Apply damping (if configured) to post-update velocities
    if (g_lin_damp > 0.0f) body.v *= std::exp(-g_lin_damp * deltaTime);
    if (g_ang_damp > 0.0f) body.w *= std::exp(-g_ang_damp * deltaTime);
    
    // Clamp angular velocity
    wLen = glm::length(body.w);
    if (wLen > g_w_max) {
        body.w *= (g_w_max / wLen);
    }
    
    // Clear accumulated forces/torques for next frame
    body.f = glm::vec3(0.0f);
    body.tau = glm::vec3(0.0f);
}

// ===== Velocity Verlet split implementation (for soft contact path) =====
// See docs/rigid_body_verlet.md for the derivation of the translational and
// rotational Velocity–Verlet scheme (including gyroscopic angular dynamics).
// Phase 1: half velocity + position/orientation advance using v(t+dt/2)
void integrateHalfPos(RigidBody& body, const glm::vec3& gravity, float deltaTime) {
    if (body.invMass <= 0.0f) return;
    const glm::vec3 acc_lin = (body.f / body.mass) + gravity;
    const glm::vec3 acc_ang = angularAccelGyro(body);
    // Half-step velocities
    body.v += acc_lin * (0.5f * deltaTime);
    body.w += acc_ang * (0.5f * deltaTime);
    // Position advance with half-step linear velocity
    body.x += body.v * deltaTime;
    if (g_pbc_enabled) wrap_position(body.x, g_pbc_min, g_pbc_max);
    // Orientation advance with half-step angular velocity
    float wLen = glm::length(body.w);
    if (wLen > 1e-8f) {
        float angle = wLen * deltaTime;
        glm::vec3 axis = body.w / wLen;
        glm::quat dq = glm::angleAxis(angle, axis);
        body.q = glm::normalize(dq * body.q);
    }
    // Do NOT clear forces here; we need them recomputed after position update.
}

// Phase 2: recompute forces externally, then call to finish velocities.
void integrateSecondHalf(RigidBody& body, const glm::vec3& gravity, float deltaTime) {
    if (body.invMass <= 0.0f) return;
    const glm::vec3 acc_lin = (body.f / body.mass) + gravity;
    const glm::vec3 acc_ang = angularAccelGyro(body);
    // Second half-step velocity update
    body.v += acc_lin * (0.5f * deltaTime);
    body.w += acc_ang * (0.5f * deltaTime);
    // Apply damping AFTER full velocity is known
    if (g_lin_damp > 0.0f) body.v *= std::exp(-g_lin_damp * deltaTime);
    if (g_ang_damp > 0.0f) body.w *= std::exp(-g_ang_damp * deltaTime);
    // Clamp angular speed
    float wLen = glm::length(body.w);
    if (wLen > g_w_max) body.w *= (g_w_max / wLen);
    // Clear forces for next frame accumulation
    body.f = glm::vec3(0.0f);
    body.tau = glm::vec3(0.0f);
}
