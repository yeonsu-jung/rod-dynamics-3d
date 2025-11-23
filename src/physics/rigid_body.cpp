/**
 * @file rigid_body.cpp
 * @brief Implementation of rigid body physics for capsule and box shapes
 */

#include "physics/rigid_body.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>

RigidBody RigidBody::makeCapsule(const glm::vec3& pos, const glm::quat& orientation,
                                float density, float radius, float halfHeight,
                                float restitution, float friction) {
    RigidBody body;
    body.type = ShapeType::Capsule;
    body.x = pos;
    body.q = glm::normalize(orientation);
    body.cap = Capsule{radius, halfHeight};

    // Calculate volume (approximate as cylinder, ignoring hemispherical ends)
    const float totalHeight = 2.0f * halfHeight;
    const float volume = static_cast<float>(M_PI) * radius * radius * totalHeight;
    body.mass = std::max(1e-6f, density * volume);
    body.invMass = 1.0f / body.mass;

    // Solid cylinder inertia tensor (axis along local +Y)
    const float Ixx = body.mass * (3.0f * radius * radius + totalHeight * totalHeight) / 12.0f;
    const float Iyy = body.mass * (radius * radius) / 2.0f;

    body.I_body = glm::mat3(0.0f);
    body.I_body[0][0] = Ixx; 
    body.I_body[1][1] = Iyy; 
    body.I_body[2][2] = Ixx;

    body.I_body_inv = glm::mat3(0.0f);
    body.I_body_inv[0][0] = (Ixx > 0) ? 1.0f / Ixx : 0.0f;
    body.I_body_inv[1][1] = (Iyy > 0) ? 1.0f / Iyy : 0.0f;
    body.I_body_inv[2][2] = (Ixx > 0) ? 1.0f / Ixx : 0.0f;

    body.restitution = restitution;
    body.friction = friction;
    // Advanced friction defaults fall back to single coefficient unless provided later
    body.frictionS = -1.0f;
    body.frictionD = -1.0f;
    body.rollingFriction = 0.0f;
    return body;
}

RigidBody RigidBody::makeRodLD(const glm::vec3& pos, const glm::quat& orientation,
                              float density, float length, float diameter,
                              float restitution, float friction) {
    const float radius = 0.5f * diameter;
    const float halfHeight = 0.5f * length;
    return makeCapsule(pos, orientation, density, radius, halfHeight, restitution, friction);
}

RigidBody RigidBody::makeStaticFloor(const glm::vec3& pos, const glm::quat& orientation,
                                    float halfX, float halfY, float halfZ,
                                    float restitution, float friction) {
    RigidBody body;
    body.type = ShapeType::Box;
    body.x = pos;
    body.q = glm::normalize(orientation);
    body.box = Box{halfX, halfY, halfZ};

    // Static body (infinite mass)
    body.mass = 0.0f;
    body.invMass = 0.0f;
    body.I_body = glm::mat3(0.0f);
    body.I_body_inv = glm::mat3(0.0f);

    body.restitution = restitution;
    body.friction = friction;
    return body;
}

RigidBody RigidBody::makeSphere(const glm::vec3& pos, float density, float radius,
                                float restitution, float friction) {
    RigidBody body;
    body.type = ShapeType::Sphere;
    body.x = pos;
    body.q = glm::quat(1, 0, 0, 0);  // Identity rotation (spheres are isotropic)
    body.sphere = Sphere{radius};

    // Calculate mass: m = (4/3) * π * r³ * ρ
    const float volume = (4.0f / 3.0f) * static_cast<float>(M_PI) * radius * radius * radius;
    body.mass = std::max(1e-6f, density * volume);
    body.invMass = 1.0f / body.mass;

    // Solid sphere inertia tensor: I = (2/5) * m * r²
    const float I_diag = 0.4f * body.mass * radius * radius;
    
    body.I_body = glm::mat3(0.0f);
    body.I_body[0][0] = I_diag;
    body.I_body[1][1] = I_diag;
    body.I_body[2][2] = I_diag;

    body.I_body_inv = glm::mat3(0.0f);
    body.I_body_inv[0][0] = (I_diag > 0) ? 1.0f / I_diag : 0.0f;
    body.I_body_inv[1][1] = (I_diag > 0) ? 1.0f / I_diag : 0.0f;
    body.I_body_inv[2][2] = (I_diag > 0) ? 1.0f / I_diag : 0.0f;

    body.restitution = restitution;
    body.friction = friction;
    body.frictionS = -1.0f;
    body.frictionD = -1.0f;
    body.rollingFriction = 0.0f;
    
    return body;
}

glm::mat3 RigidBody::R() const { 
    return glm::mat3_cast(q); 
}

glm::mat3 RigidBody::IworldInv() const {
    if (invMass <= 0.0f) return glm::mat3(0.0f);
    
    glm::mat3 rotationMatrix = R();
    return rotationMatrix * I_body_inv * glm::transpose(rotationMatrix);
}

glm::mat4 RigidBody::modelMatrix() const {
    glm::mat4 transform = glm::translate(glm::mat4(1.0f), x) * glm::mat4_cast(q);
    
    if (type == ShapeType::Capsule) {
        // Unit cylinder is radius=1, y in [-1,1]
        return glm::scale(transform, glm::vec3(cap.r, cap.h, cap.r));
    } else {
        return glm::scale(transform, glm::vec3(box.hx, box.hy, box.hz));
    }
}

glm::vec3 RigidBody::axisY() const { 
    return R()[1]; 
}
