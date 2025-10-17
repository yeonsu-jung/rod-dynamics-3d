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

void integrate(RigidBody& body, const glm::vec3& gravity, float deltaTime) {
    // Skip integration for static bodies
    if (body.invMass <= 0.0f) return;
    
    // Linear integration with damping
    body.v += gravity * deltaTime;
    body.v *= std::exp(-g_lin_damp * deltaTime);
    body.x += body.v * deltaTime;

    // Angular integration with damping and velocity clamping
    body.w *= std::exp(-g_ang_damp * deltaTime);
    
    float angularSpeed = glm::length(body.w);
    if (angularSpeed > g_w_max) {
        body.w *= (g_w_max / angularSpeed);
    }

    // Quaternion integration
    if (angularSpeed > 1e-6f) {
        glm::quat deltaQ(0.0f, body.w.x, body.w.y, body.w.z);
        body.q = glm::normalize(body.q + 0.5f * deltaQ * body.q * deltaTime);
    }
}
