#pragma once
#include <glm/glm.hpp>

inline float length2(const glm::vec3& v){ return glm::dot(v,v); }

struct Contact{
  bool hit{false};
  glm::vec3 normal{0};  // from A to B
  float penetration{0};
  glm::vec3 point{0};
};

struct SolverConfig{
  float baumgarte = 0.25f;
  float allowedPen = 0.003f;
  int   velIters = 30;
};

// Tunables
constexpr float LIN_DAMP = 0.08f;  // s^-1
constexpr float ANG_DAMP = 0.12f;  // s^-1
constexpr float W_MAX    = 80.0f;  // rad/s
