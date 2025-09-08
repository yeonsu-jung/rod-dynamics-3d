#pragma once
#include <glm/glm.hpp>
struct RigidBody;
extern float g_lin_damp, g_ang_damp, g_w_max;   // <— add
void integrate(RigidBody& b, const glm::vec3& g, float dt);
