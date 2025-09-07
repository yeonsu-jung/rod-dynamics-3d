#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "shape.hpp"

struct RigidBody {
  // state
  glm::vec3 x{0};
  glm::quat q{1,0,0,0};
  glm::vec3 v{0};
  glm::vec3 w{0};

  // props
  float mass{1}, invMass{1};
  glm::mat3 I_body{1.0f}, I_body_inv{1.0f};
  float restitution{0.25f}, friction{0.7f};

  // shape
  ShapeType type{ShapeType::Capsule};
  Box box{};
  Capsule cap{};

  static RigidBody makeCapsule(const glm::vec3& pos, const glm::quat& q,
                               float density, float r, float h,
                               float restitution=0.25f, float friction=0.7f);

  static RigidBody makeRodLD(const glm::vec3& pos, const glm::quat& q,
                             float density, float L, float D,
                             float restitution=0.25f, float friction=0.7f);

  static RigidBody makeStaticFloor(const glm::vec3& pos, const glm::quat& q,
                                   float hx, float hy, float hz,
                                   float restitution=0.3f, float friction=0.9f);

  glm::mat3 R() const;
  glm::mat3 IworldInv() const;
  glm::mat4 modelMatrix() const;     // unit cylinder [-1,1] y, radius 1; box is [-1,1]^3
  glm::vec3 axisY() const;           // local +Y in world
};
