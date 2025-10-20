/**
 * @file rigid_body.hpp
 * @brief Rigid body representation for physics simulation
 */

#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "shape.hpp"

/**
 * @brief Rigid body with position, orientation, velocity, and material properties
 */
struct RigidBody {
    // Physical state
    glm::vec3 x{0};           ///< Position
    glm::quat q{1, 0, 0, 0};  ///< Orientation quaternion (w, x, y, z)
    glm::vec3 v{0};           ///< Linear velocity
    glm::vec3 w{0};           ///< Angular velocity

    // Accumulated forces/torques (reset each step)
    glm::vec3 f{0};           ///< Accumulated linear force
    glm::vec3 tau{0};         ///< Accumulated torque

    // Mass properties
    float mass{1.0f};         ///< Mass (kg)
    float invMass{1.0f};      ///< Inverse mass (1/kg)
    glm::mat3 I_body{1.0f};   ///< Body-space inertia tensor
    glm::mat3 I_body_inv{1.0f}; ///< Inverse body-space inertia tensor

    // Material properties
    float restitution{0.25f}; ///< Coefficient of restitution (0-1)
    float friction{0.7f};     ///< Legacy single friction coefficient (used as default)
    // Advanced friction (optional)
    float frictionS{-1.0f};   ///< Static friction (<=0 => use 'friction')
    float frictionD{-1.0f};   ///< Dynamic friction (<=0 => use 'friction')
    float rollingFriction{0.0f}; ///< Rolling friction (not used yet)

    // Shape representation
    ShapeType type{ShapeType::Capsule};
    Box box{};
    Capsule cap{};

    // Factory methods
    static RigidBody makeCapsule(const glm::vec3& pos, const glm::quat& orientation,
                                float density, float radius, float halfHeight,
                                float restitution = 0.25f, float friction = 0.7f);

    static RigidBody makeRodLD(const glm::vec3& pos, const glm::quat& orientation,
                              float density, float length, float diameter,
                              float restitution = 0.25f, float friction = 0.7f);

    static RigidBody makeStaticFloor(const glm::vec3& pos, const glm::quat& orientation,
                                     float halfX, float halfY, float halfZ,
                                     float restitution = 0.3f, float friction = 0.9f);

    // Utility methods
    glm::mat3 R() const;              ///< Get rotation matrix from quaternion
    glm::mat3 IworldInv() const;      ///< Get world-space inverse inertia tensor
    glm::mat4 modelMatrix() const;    ///< Get model matrix for rendering
    glm::vec3 axisY() const;          ///< Get local Y-axis in world space           // local +Y in world
    // Compute segment endpoints for capsule axis in world space
    inline void capsuleEndpoints(glm::vec3& a, glm::vec3& b) const {
        const glm::vec3 u = axisY();
        a = x - u * cap.h;
        b = x + u * cap.h;
    }
};
