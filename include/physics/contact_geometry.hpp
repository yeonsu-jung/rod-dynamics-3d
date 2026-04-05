#pragma once

#include <glm/glm.hpp>

struct RigidBody;

struct CommonContactGeometry {
  int bodyA = -1;
  int bodyB = -1;

  glm::vec3 pointA{0.0f};
  glm::vec3 pointB{0.0f};
  glm::vec3 normal{0.0f};

  double distance = 0.0;
  double surfaceLimit = 0.0;
  double signedGap = 0.0;

  glm::vec3 shiftB{0.0f};
  bool isWall = false;
};

struct ContactKinematics {
  glm::vec3 vRel{0.0f};
  double vNormal = 0.0;
  glm::vec3 vTangential{0.0f};
  double vTangentialMagnitude = 0.0;
};

ContactKinematics computeContactKinematics(const RigidBody& bodyA,
                                           const RigidBody& bodyB,
                                           const CommonContactGeometry& contact);
