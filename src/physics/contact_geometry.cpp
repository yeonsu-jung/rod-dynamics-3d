#include "physics/contact_geometry.hpp"

#include <glm/geometric.hpp>

#include "physics/rigid_body.hpp"

ContactKinematics computeContactKinematics(const RigidBody& bodyA,
                                           const RigidBody& bodyB,
                                           const CommonContactGeometry& contact) {
  glm::vec3 rA = contact.pointA - bodyA.x;
  glm::vec3 rB = contact.pointB - (bodyB.x + contact.shiftB);

  glm::vec3 vA = bodyA.v + glm::cross(bodyA.w, rA);
  glm::vec3 vB = bodyB.v + glm::cross(bodyB.w, rB);
  glm::vec3 vRel = vB - vA;

  double vNormal = glm::dot(contact.normal, vRel);
  glm::vec3 vTangential = vRel - static_cast<float>(vNormal) * contact.normal;

  ContactKinematics out;
  out.vRel = vRel;
  out.vNormal = vNormal;
  out.vTangential = vTangential;
  out.vTangentialMagnitude = glm::length(vTangential);
  return out;
}
