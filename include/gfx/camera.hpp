#pragma once
#include <glm/glm.hpp>

struct OrbitCamera {
  float yaw=0.6f, pitch=0.35f, dist=6.0f;
  glm::mat4 view(const glm::vec3& target = glm::vec3(0)) const;
  glm::vec3 eye(const glm::vec3& target = glm::vec3(0)) const;
};
