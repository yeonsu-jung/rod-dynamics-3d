#pragma once
#include "rigid_body.hpp"
#include "types.hpp"
#include <vector>

void closestPtSegmentSegment(const glm::vec3& p1, const glm::vec3& q1,
                             const glm::vec3& p2, const glm::vec3& q2,
                             glm::vec3& c1, glm::vec3& c2);

Contact collideCapsuleCapsule(const RigidBody& A, const RigidBody& B);

// Capsule vs floor (top plane of box G)
Contact collideCapsuleFloor(const RigidBody& C, const RigidBody& G);

// Capsule vs inner wall of an infinite cylindrical tube.
// Returns up to 3 contacts (endpoints + midpoint) via the output vector.
void collideCapsuleInsideCylinder(const RigidBody& capsule,
                                  float cylRadius,
                                  const glm::vec3& cylAxis,
                                  std::vector<Contact>& contacts);

// Capsule vs the half-space bounded by an infinite plane.
//
// planeNormal must point INTO the allowed region (away from the wall, toward
// the rods). A point p penetrates when dot(p - planePoint, planeNormal) < r.
//
// Emits one contact per penetrating endpoint, so a rod lying flat against a
// plate yields two and picks up the correct torque, while a rod touching
// end-on yields one. The deepest point of a capsule against a plane is always
// an endpoint, so endpoints suffice.
//
// Contact convention matches the routines above: `normal` points from the
// capsule toward the wall, i.e. -planeNormal.
void collideCapsulePlane(const RigidBody& capsule,
                         const glm::vec3& planePoint,
                         const glm::vec3& planeNormal,
                         std::vector<Contact>& contacts);
