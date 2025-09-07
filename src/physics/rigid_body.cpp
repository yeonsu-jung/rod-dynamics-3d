#include "physics/rigid_body.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>

static inline float sq(float x){ return x*x; }

RigidBody RigidBody::makeCapsule(const glm::vec3& pos, const glm::quat& q_,
                                 float density, float r, float h,
                                 float restitution, float friction)
{
    RigidBody b;
    b.type = ShapeType::Capsule;
    b.x = pos;
    b.q = glm::normalize(q_);
    b.cap = Capsule{r,h};

    // approximate volume by cylinder section (ignoring hemispherical ends)
    const float H = 2.0f*h;
    const float volume = float(M_PI) * r*r * H;
    b.mass    = std::max(1e-6f, density * volume);
    b.invMass = 1.0f / b.mass;

    // solid cylinder inertia (axis along local +Y)
    const float Ix = b.mass * (3.f*r*r + H*H) / 12.f;
    const float Iy = b.mass * (r*r) / 2.f;

    b.I_body = glm::mat3(0.0f);
    b.I_body[0][0] = Ix; b.I_body[1][1] = Iy; b.I_body[2][2] = Ix;

    b.I_body_inv = glm::mat3(0.0f);
    b.I_body_inv[0][0] = (Ix>0)? 1.f/Ix : 0.f;
    b.I_body_inv[1][1] = (Iy>0)? 1.f/Iy : 0.f;
    b.I_body_inv[2][2] = (Ix>0)? 1.f/Ix : 0.f;

    b.restitution = restitution;
    b.friction    = friction;
    return b;
}

RigidBody RigidBody::makeRodLD(const glm::vec3& pos, const glm::quat& q_,
                               float density, float L, float D,
                               float restitution, float friction)
{
    const float r = 0.5f*D;
    const float h = 0.5f*L;
    return makeCapsule(pos, q_, density, r, h, restitution, friction);
}

RigidBody RigidBody::makeStaticFloor(const glm::vec3& pos, const glm::quat& q_,
                                     float hx, float hy, float hz,
                                     float restitution, float friction)
{
    RigidBody b;
    b.type = ShapeType::Box;
    b.x = pos;
    b.q = glm::normalize(q_);
    b.box = Box{hx,hy,hz};

    b.mass = 0.0f;
    b.invMass = 0.0f;
    b.I_body = glm::mat3(0.0f);
    b.I_body_inv = glm::mat3(0.0f);

    b.restitution = restitution;
    b.friction    = friction;
    return b;
}

glm::mat3 RigidBody::R() const { return glm::mat3_cast(q); }

glm::mat3 RigidBody::IworldInv() const {
    if (invMass <= 0.0f) return glm::mat3(0.0f);
    glm::mat3 Rm = R();
    return Rm * I_body_inv * glm::transpose(Rm);
}

glm::mat4 RigidBody::modelMatrix() const {
    glm::mat4 M = glm::translate(glm::mat4(1.0f), x) * glm::mat4_cast(q);
    if (type == ShapeType::Capsule){
        // unit cylinder is radius=1, y in [-1,1]
        return glm::scale(M, glm::vec3(cap.r, cap.h, cap.r));
    } else {
        return glm::scale(M, glm::vec3(box.hx, box.hy, box.hz));
    }
}

glm::vec3 RigidBody::axisY() const { return R()[1]; }
