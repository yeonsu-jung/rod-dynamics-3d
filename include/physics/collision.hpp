#pragma once
#include "rigid_body.hpp"
#include "types.hpp"

void closestPtSegmentSegment(const glm::vec3& p1, const glm::vec3& q1,
                             const glm::vec3& p2, const glm::vec3& q2,
                             glm::vec3& c1, glm::vec3& c2);

Contact collideCapsuleCapsule(const RigidBody& A, const RigidBody& B);

// Capsule vs floor (top plane of box G)
Contact collideCapsuleFloor(const RigidBody& C, const RigidBody& G);
