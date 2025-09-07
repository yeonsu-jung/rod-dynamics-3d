#include "physics/integrator.hpp"
#include "physics/types.hpp"
#include <glm/gtc/quaternion.hpp>
#include <cmath>

void integrate(RigidBody& b, const glm::vec3& g, float dt)
{
    // linear
    b.v += g * dt;
    b.v *= std::exp(-LIN_DAMP * dt);
    b.x += b.v * dt;

    // angular
    b.w *= std::exp(-ANG_DAMP * dt);
    float wlen = glm::length(b.w);
    if (wlen > W_MAX) b.w *= (W_MAX / wlen);

    glm::quat dq(0.0f, b.w.x, b.w.y, b.w.z);
    b.q = glm::normalize(b.q + 0.5f * dq * b.q * dt);
}
