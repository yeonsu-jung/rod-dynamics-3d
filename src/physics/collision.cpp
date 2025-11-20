/**
 * @file collision.cpp
 * @brief Collision detection between rigid bodies
 * Uses Lumelsky algorithm from DisMech for robust segment-segment distance
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
    
    inline bool fixBound(float& x) {
        if (x > 1.0f) {
            x = 1.0f;
            return true;
        } else if (x < 0.0f) {
            x = 0.0f;
            return true;
        }
        return false;
    }
}

/**
 * @brief Lumelsky algorithm for closest points between two line segments
 * Based on DisMech implementation - more robust than previous closestPtSegmentSegment
 * 
 * @param p1 Start of segment 1
 * @param q1 End of segment 1
 * @param p2 Start of segment 2
 * @param q2 End of segment 2
 * @param c1 Output: closest point on segment 1
 * @param c2 Output: closest point on segment 2
 */
void closestPtSegmentSegment(const glm::vec3& p1, const glm::vec3& q1,
                            const glm::vec3& p2, const glm::vec3& q2,
                            glm::vec3& c1, glm::vec3& c2) {
    // DisMech Lumelsky algorithm implementation
    const glm::vec3 e1 = q1 - p1;  // Direction vector of segment 1
    const glm::vec3 e2 = q2 - p2;  // Direction vector of segment 2
    const glm::vec3 e12 = p2 - p1; // Vector from start of seg1 to start of seg2
    
    const float D1 = glm::dot(e1, e1);
    const float D2 = glm::dot(e2, e2);
    const float S1 = glm::dot(e1, e12);
    const float S2 = glm::dot(e2, e12);
    const float R = glm::dot(e1, e2);
    
    const float den = D1 * D2 - R * R;
    
    float t = 0.0f;
    float u = 0.0f;
    float uf = 0.0f;
    
    if (den == 0.0f) {
        // Segments are parallel
        t = 0.0f;
        u = -S2 / D2;
        uf = u;
        fixBound(uf);
        
        if (uf != u) {
            t = (uf * R + S1) / D1;
            fixBound(t);
            u = uf;
        }
    } else {
        // General case
        t = (S1 * D2 - S2 * R) / den;
        fixBound(t);
        u = (t * R - S2) / D2;
        uf = u;
        fixBound(uf);
        
        if (uf != u) {
            t = (uf * R + S1) / D1;
            fixBound(t);
            u = uf;
        }
    }
    
    c1 = p1 + t * e1;
    c2 = p2 + u * e2;
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
        t = std::max(-halfHeight, std::min(halfHeight, t));
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
