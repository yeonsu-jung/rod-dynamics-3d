/**
 * @file collision.cpp
 * @brief Collision detection between rigid bodies
 */

#include "physics/collision.hpp"
#include "physics/types.hpp"
#include <algorithm>
#include <cmath>

// PBC globals declared in integrator.cpp; declare externs here to avoid including headers
extern bool g_pbc_enabled;
extern glm::vec3 g_pbc_min;
extern glm::vec3 g_pbc_max;

namespace {
    constexpr float EPSILON = 1e-8f;

    inline glm::vec3 pbc_shift_vec(const glm::vec3& delta, const glm::vec3& bmin, const glm::vec3& bmax) {
        glm::vec3 size = bmax - bmin;
        glm::vec3 s(0.0f);
        for (int i = 0; i < 3; ++i) {
            if (size[i] <= 0.0f) { s[i] = 0.0f; continue; }
            float n = std::floor(delta[i] / size[i] + 0.5f); // nearest image count
            s[i] = -n * size[i];
        }
        return s;
    }
}

void closestPtSegmentSegment(const glm::vec3& p1, const glm::vec3& q1,
                            const glm::vec3& p2, const glm::vec3& q2,
                            glm::vec3& c1, glm::vec3& c2) {
    const glm::vec3 u = q1 - p1;
    const glm::vec3 v = q2 - p2;
    const glm::vec3 w0 = p1 - p2;
    
    float a = glm::dot(u, u);
    float b = glm::dot(u, v);
    float c = glm::dot(v, v);
    float d = glm::dot(u, w0);
    float e = glm::dot(v, w0);
    float D = a * c - b * b;
    
    float sN, sD = D;
    float tN, tD = D;

    if (D < EPSILON) { // Almost parallel segments
        sN = 0.0f; sD = 1.0f;
        tN = e; tD = c;
    } else {
        sN = (b * e - c * d);
        tN = (a * e - b * d);
        if (sN < 0) { 
            sN = 0; tN = e; tD = c; 
        } else if (sN > sD) { 
            sN = sD; tN = e + b; tD = c; 
        }
    }

    if (tN < 0){
        tN = 0;
        if (-d < 0) sN = 0;
        else if (-d > a) sN = sD;
        else { sN = -d; sD = a; }
    } else if (tN > tD){
        tN = tD;
        if ((-d + b) < 0) sN = 0;
        else if ((-d + b) > a) sN = sD;
        else { sN = (-d + b); sD = a; }
    }

    float sc = (std::abs(sN) < EPSILON) ? 0.0f : (sN / sD);
    float tc = (std::abs(tN) < EPSILON) ? 0.0f : (tN / tD);

    c1 = p1 + sc * u;
    c2 = p2 + tc * v;
}

namespace {
    inline float clampf(float x, float a, float b) { 
        return std::max(a, std::min(b, x)); 
    }
}

Contact collideCapsuleCapsule(const RigidBody& A, const RigidBody& B) {
    Contact contact;
    
    const glm::vec3 axisA = glm::normalize(A.axisY());
    const glm::vec3 axisB = glm::normalize(B.axisY());

    // Calculate segment endpoints for both capsules
    const glm::vec3 A0 = A.x - axisA * A.cap.h;
    const glm::vec3 A1 = A.x + axisA * A.cap.h;
    glm::vec3 B0 = B.x - axisB * B.cap.h;
    glm::vec3 B1 = B.x + axisB * B.cap.h;

    // Apply minimum-image shift for B if PBC is enabled
    glm::vec3 shiftB(0.0f);
    if (g_pbc_enabled) {
        shiftB = pbc_shift_vec(B.x - A.x, g_pbc_min, g_pbc_max);
        B0 += shiftB;
        B1 += shiftB;
    }

    glm::vec3 closestA, closestB;
    closestPtSegmentSegment(A0, A1, B0, B1, closestA, closestB);

    glm::vec3 separation = closestB - closestA;
    float distance = glm::length(separation);
    float radiusSum = A.cap.r + B.cap.r;
    
    if (distance >= radiusSum) return contact; // No collision

    contact.hit = true;
    contact.penetration = radiusSum - distance;
    
    if (distance > 1e-6f) {
        contact.normal = separation / distance;
    } else {
        // Handle degenerate case where capsules are coincident
        glm::vec3 fallback = (B.x + shiftB) - A.x;
        if (glm::dot(fallback, fallback) < 1e-8f) {
            fallback = glm::cross(axisA, glm::vec3(1, 0, 0));
        }
        if (glm::dot(fallback, fallback) < 1e-8f) {
            fallback = glm::vec3(0, 1, 0);
        }
        contact.normal = glm::normalize(fallback);
    }
    contact.point = 0.5f * (closestA + closestB);
    contact.shiftB = shiftB; // record periodic image shift used
    return contact;
}

Contact collideCapsuleFloor(const RigidBody& capsule, const RigidBody& floor) {
    Contact contact;
    
    // Floor top normal (ensure it points upward)
    glm::vec3 normal = glm::normalize(floor.R()[1]);
    if (glm::dot(normal, glm::vec3(0, 1, 0)) < 0) {
        normal = -normal;
    }
    const glm::vec3 floorPoint = floor.x + normal * floor.box.hy;

    const glm::vec3 capsuleAxis = capsule.axisY(); // Not normalized
    const float halfHeight = capsule.cap.h;
    const float radius = capsule.cap.r;

    const float denominator = glm::dot(capsuleAxis, normal);
    glm::vec3 closestPointOnAxis;

    if (std::abs(denominator) < 1e-6f) {
        // Capsule axis is nearly parallel to floor plane: use the lower endpoint
        const glm::vec3 endpoint0 = capsule.x - capsuleAxis * halfHeight;
        const glm::vec3 endpoint1 = capsule.x + capsuleAxis * halfHeight;
        float distance0 = glm::dot(endpoint0 - floorPoint, normal);
        float distance1 = glm::dot(endpoint1 - floorPoint, normal);
        closestPointOnAxis = (distance0 < distance1) ? endpoint0 : endpoint1;
    } else {
        // Find closest point on capsule axis to floor plane, clamped to segment
        float t = glm::dot(floorPoint - capsule.x, normal) / denominator;
        t = clampf(t, -halfHeight, +halfHeight);
        closestPointOnAxis = capsule.x + capsuleAxis * t;
    }

    // Calculate signed distance (negative means penetration)
    float signedDistance = glm::dot(closestPointOnAxis - floorPoint, normal) - radius;
    if (signedDistance >= 0.0f) return contact; // No collision

    contact.hit = true;
    contact.normal = -normal; // From capsule to floor (important for impulse direction)
    contact.penetration = -signedDistance;
    contact.point = closestPointOnAxis - radius * normal;
    return contact;
}
