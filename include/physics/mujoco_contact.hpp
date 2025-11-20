#pragma once

#include <vector>
#include <glm/glm.hpp>

#include "physics/rigid_body.hpp"

struct MujocoContactCfg {
    bool enabled = false;           // master switch for this contact path
    double normal_k = 1e4;          // normal stiffness (N/m)
    double normal_damping = 1.0;    // damping coefficient (N·s/m) or scaled factor
    double friction_mu = 0.5;       // Coulomb friction coefficient
    double vel_eps = 1e-3;          // regularization velocity for friction (m/s)
};

// Simple MuJoCo-inspired penalty contact model.
// This is intentionally independent from SoftContactSolver so that we can
// A/B test behavior and parameterization without entangling the two paths.
class MujocoContactSolver {
public:
    explicit MujocoContactSolver(const MujocoContactCfg& cfg = {});

    void setConfig(const MujocoContactCfg& cfg);

    // Detect capsule-capsule contacts for the current rods.
    // For now we do a simple O(N^2) pair loop; later we can hook into
    // the main broadphase if needed.
    void detectContacts(const std::vector<RigidBody>& bodies);

    // Compute and apply contact forces/torques to bodies based on the
    // detected contacts and the current configuration.
    void computeForces(std::vector<RigidBody>& bodies, double dt);

    double getLastPotentialEnergy() const { return lastPotentialEnergy_; }

private:
    struct Contact {
        int a = -1, b = -1;        // body indices
        glm::vec3 pA{0.0f};        // contact point on A (world)
        glm::vec3 pB{0.0f};        // contact point on B (world)
        glm::vec3 n{1.0f,0.0f,0.0f}; // normal from A to B
        double dist = 0.0;         // centerline distance between capsule axes
        double surface_limit = 0.0; // sum of radii (h)
    };

    MujocoContactCfg cfg_{};
    std::vector<Contact> contacts_;
    double lastPotentialEnergy_ = 0.0; // simple 0.5*k*d^2 sum

    void detectCapsuleCapsule(const RigidBody& a, const RigidBody& b,
                              int ia, int ib);
    static void closestPointsSegmentSegment(
        const glm::vec3& a1, const glm::vec3& a2,
        const glm::vec3& b1, const glm::vec3& b2,
        double& s, double& t);
};
