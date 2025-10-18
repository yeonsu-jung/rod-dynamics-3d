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

void integrate(RigidBody& body, const glm::vec3& gravity, float deltaTime) {
    // Skip integration for static bodies
    if (body.invMass <= 0.0f) return;
    
    // Semi-implicit (symplectic) Euler for linear motion
    // v_{t+dt} = v_t + a*dt, then x_{t+dt} = x_t + v_{t+dt}*dt
    body.v += gravity * deltaTime;
    // Exponential damping (stable, frame-rate independent)
    if (g_lin_damp > 0.0f) body.v *= std::exp(-g_lin_damp * deltaTime);
    body.x += body.v * deltaTime;

    // Periodic wrapping of position
    if (g_pbc_enabled) {
        wrap_position(body.x, g_pbc_min, g_pbc_max);
    }

    // Angular: apply damping and clamp magnitude, then integrate quaternion
    if (g_ang_damp > 0.0f) body.w *= std::exp(-g_ang_damp * deltaTime);

    float wLen = glm::length(body.w);
    if (wLen > g_w_max) {
        body.w *= (g_w_max / wLen);
        wLen = g_w_max;
    }

    // Integrate orientation using quaternion exponential map for better stability
    // q_{t+dt} = normalize( exp(0.5*omega*dt) * q_t )
    if (wLen > 1e-8f) {
        float angle = wLen * deltaTime; // total rotation in radians this step
        glm::vec3 axis = body.w / wLen;
        glm::quat dq = glm::angleAxis(angle, axis);
        body.q = glm::normalize(dq * body.q);
    }
}
