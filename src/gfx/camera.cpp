#include "gfx/camera.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>

glm::mat4 OrbitCamera::view(const glm::vec3& target) const {
    float cp = std::cos(pitch), sp = std::sin(pitch);
    float cy = std::cos(yaw),   sy = std::sin(yaw);
    glm::vec3 eye = target + glm::vec3(cp*cy, sp, cp*sy) * dist;
    return glm::lookAt(eye, target, glm::vec3(0,1,0));
}

glm::vec3 OrbitCamera::eye(const glm::vec3& target) const {
    float cp = std::cos(pitch), sp = std::sin(pitch);
    float cy = std::cos(yaw),   sy = std::sin(yaw);
    return target + glm::vec3(cp*cy, sp, cp*sy) * dist;
}
