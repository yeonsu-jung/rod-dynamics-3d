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
 * @brief Integrate rigid body physics for one timestep
 * @param body Rigid body to integrate
 * @param gravity Gravity vector
 * @param deltaTime Time step size
 */
void integrate(RigidBody& body, const glm::vec3& gravity, float deltaTime);
