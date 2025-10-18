/**
 * @file solver.cpp
 * @brief Physics constraint solver for impulse-based resolution
 */

#include "physics/solver.hpp"
#include "physics/types.hpp"
#include <algorithm>
#include <cmath>

namespace {
    constexpr float BOUNCE_THRESHOLD = 0.4f; ///< Velocity below which objects don't bounce
}

void applyImpulse(RigidBody& bodyA, RigidBody& bodyB, const Contact& contact, AppliedImpulse* out) {
    glm::vec3 rA = contact.point - bodyA.x;
    glm::vec3 rB = contact.point - (bodyB.x + contact.shiftB);

    // Calculate relative velocity at contact point
    glm::vec3 vA = bodyA.v + glm::cross(bodyA.w, rA);
    glm::vec3 vB = bodyB.v + glm::cross(bodyB.w, rB);
    glm::vec3 relativeVelocity = vB - vA;

    float normalVelocity = glm::dot(relativeVelocity, contact.normal);
    if (normalVelocity > 0.0f) { if (out) *out = {}; return; }

    // Calculate restitution (bouncing) - disable for gentle impacts
    float restitution = std::min(bodyA.restitution, bodyB.restitution);
    if (std::abs(normalVelocity) < BOUNCE_THRESHOLD) {
        restitution = 0.0f;
    }

    glm::mat3 invInertiaA = bodyA.IworldInv();
    glm::mat3 invInertiaB = bodyB.IworldInv();

    // Lambda function to compute effective mass in a given direction
    auto computeEffectiveMass = [&](const glm::vec3& direction) -> float {
        glm::vec3 crossA = glm::cross(rA, direction);
        glm::vec3 crossB = glm::cross(rB, direction);
        float effectiveMass = bodyA.invMass + bodyB.invMass
                            + glm::dot(direction, glm::cross(invInertiaA * crossA, rA))
                            + glm::dot(direction, glm::cross(invInertiaB * crossB, rB));
        return (effectiveMass > 1e-8f) ? effectiveMass : 1e-8f;
    };

    // Calculate normal impulse
    float normalEffectiveMass = computeEffectiveMass(contact.normal);
    float normalImpulseMagnitude = -(1.0f + restitution) * normalVelocity / normalEffectiveMass;
    glm::vec3 normalImpulse = normalImpulseMagnitude * contact.normal;

    // Apply normal impulse
    bodyA.v -= normalImpulse * bodyA.invMass;  
    bodyB.v += normalImpulse * bodyB.invMass;
    bodyA.w -= invInertiaA * glm::cross(rA, normalImpulse);
    bodyB.w += invInertiaB * glm::cross(rB, normalImpulse);

    // Recompute relative velocity for friction calculation
    vA = bodyA.v + glm::cross(bodyA.w, rA);
    vB = bodyB.v + glm::cross(bodyB.w, rB);
    relativeVelocity = vB - vA;

    // Calculate friction (Coulomb model)
    glm::vec3 tangent = relativeVelocity - contact.normal * glm::dot(relativeVelocity, contact.normal);
    float tangentLength = glm::length(tangent);

    // Handle static friction for very small relative motion
    float tangentImpulseMagnitude = 0.0f;
    if (tangentLength > 1e-3f) {
        tangent /= tangentLength;
        float tangentEffectiveMass = computeEffectiveMass(tangent);
        tangentImpulseMagnitude = -glm::dot(relativeVelocity, tangent) / tangentEffectiveMass;

        // Apply Coulomb friction constraint
        float combinedFriction = 0.5f * (bodyA.friction + bodyB.friction);
        tangentImpulseMagnitude = glm::clamp(tangentImpulseMagnitude, 
                                            -combinedFriction * normalImpulseMagnitude, 
                                             combinedFriction * normalImpulseMagnitude);

        glm::vec3 frictionImpulse = tangentImpulseMagnitude * tangent;
        bodyA.v -= frictionImpulse * bodyA.invMass;  
        bodyB.v += frictionImpulse * bodyB.invMass;
        bodyA.w -= invInertiaA * glm::cross(rA, frictionImpulse);
        bodyB.w += invInertiaB * glm::cross(rB, frictionImpulse);
    }

    if (out) {
        out->jn = normalImpulseMagnitude;
        out->jt = tangentImpulseMagnitude;
        out->tangent = (tangentLength > 1e-3f) ? tangent : glm::vec3(0);
    }
}

void positionalCorrection(RigidBody& bodyA, RigidBody& bodyB, const Contact& contact, const SolverConfig& config) {
    float correctionAmount = std::max(0.0f, contact.penetration - config.allowedPen);
    if (correctionAmount <= 0) return;
    
    glm::vec3 correction = (config.baumgarte * correctionAmount) * contact.normal / 
                          (bodyA.invMass + bodyB.invMass + 1e-8f);
    bodyA.x -= bodyA.invMass * correction;
    bodyB.x += bodyB.invMass * correction;
}

void applyWarmStart(RigidBody& bodyA, RigidBody& bodyB, const Contact& c, float jn, float jt, const glm::vec3& tangent) {
    glm::vec3 rA = c.point - bodyA.x;
    glm::vec3 rB = c.point - (bodyB.x + c.shiftB);

    glm::vec3 J = jn * c.normal + jt * tangent;
    glm::mat3 invInertiaA = bodyA.IworldInv();
    glm::mat3 invInertiaB = bodyB.IworldInv();

    bodyA.v -= J * bodyA.invMass;  
    bodyB.v += J * bodyB.invMass;
    bodyA.w -= invInertiaA * glm::cross(rA, J);
    bodyB.w += invInertiaB * glm::cross(rB, J);
}
