#include "physics/integrator.hpp"
#include "physics/types.hpp"
#include <glm/gtc/quaternion.hpp>
#include <cmath>
#include "physics/rigid_body.hpp"   // ⬅️ add this line to get the full struct


// defaults (matched to your previous constants)
float g_lin_damp = 0.08f;
float g_ang_damp = 0.12f;
float g_w_max    = 80.0f;

void integrate(RigidBody& b, const glm::vec3& g, float dt)
{
    b.v += g * dt;
    b.v *= std::exp(-g_lin_damp * dt);
    b.x += b.v * dt;

    b.w *= std::exp(-g_ang_damp * dt);
    float wlen = glm::length(b.w);
    if (wlen > g_w_max) b.w *= (g_w_max / wlen);

    glm::quat dq(0.0f, b.w.x, b.w.y, b.w.z);
    b.q = glm::normalize(b.q + 0.5f * dq * b.q * dt);
}
