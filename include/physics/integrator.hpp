/**
 * @file integrator.hpp
 * @brief Physics integration with damping
 */

#pragma once
#include <glm/glm.hpp>

struct RigidBody;

// Global damping parameters (modifiable at runtime)
extern float g_lin_damp;  ///< Linear damping coefficient (s^-1)
extern float g_ang_damp;  ///< Angular damping coefficient (s^-1)
extern float g_w_max;     ///< Maximum angular velocity (rad/s)

// Periodic boundary configuration (set by the app)
extern bool g_pbc_enabled;      ///< Enable periodic boundary conditions
extern glm::vec3 g_pbc_min;     ///< Minimum corner of periodic box
extern glm::vec3 g_pbc_max;     ///< Maximum corner of periodic box

/**
 * @brief Legacy single-step symplectic Euler integration (used for hard contacts path)
 */
void integrate(RigidBody& body, const glm::vec3& gravity, float deltaTime);

// ===== Full Velocity Verlet support (split phases) =====
// Usage for soft contact penalty forces:
//   1) contacts+forces at t
//   2) integrateHalfPos(body, gravity, dt)  // v,w half-step; x,q full step
//   3) clear forces; recompute contacts+forces at t+dt
//   4) integrateSecondHalf(body, gravity, dt) // complete v,w; apply damping & clear forces

void integrateHalfPos(RigidBody& body, const glm::vec3& gravity, float deltaTime);
void integrateSecondHalf(RigidBody& body, const glm::vec3& gravity, float deltaTime);
