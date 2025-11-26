/**
 * @file soft_contact.cpp
 * @brief Implementation of soft penalty-based contact solver
 */

#include "physics/soft_contact.hpp"
#include "physics/rigid_body.hpp"
#include <algorithm>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

SoftContactSolver::SoftContactSolver(const SoftContactCfg& config)
    : config_(config)
{
    K1_ = 15.0 / config_.delta;
    K2_ = 15.0 / config_.nu;
    lastPotentialEnergy_ = 0.0;
}

void SoftContactSolver::setConfig(const SoftContactCfg& config) {
    config_ = config;
    K1_ = 15.0 / config_.delta;
    K2_ = 15.0 / config_.nu;
    // Reset accumulator on config change (avoids stale reporting)
    lastPotentialEnergy_ = 0.0;
}

void SoftContactSolver::detectContacts(const std::vector<RigidBody>& bodies) {
    contacts_.clear();
    
    // Broadphase: All pairs (O(n²) - fine for small number of objects)
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<ContactPrimitive>> thread_contacts(num_threads);
    // Reserve some space to avoid frequent reallocations
    for(auto& v : thread_contacts) v.reserve(100);

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < bodies.size(); ++i) {
        int tid = omp_get_thread_num();
        for (size_t j = i + 1; j < bodies.size(); ++j) {
            const RigidBody& a = bodies[i];
            const RigidBody& b = bodies[j];
            
            // Dispatch based on shape types
            if (a.type == ShapeType::Capsule && b.type == ShapeType::Capsule) {
                detectCapsuleCapsule(a, b, i, j, thread_contacts[tid]);
            } else if (a.type == ShapeType::Sphere && b.type == ShapeType::Sphere) {
                detectSphereSphere(a, b, i, j, thread_contacts[tid]);
            }
            // TODO: Add sphere-capsule detection if needed
        }
    }

    // Merge results
    for (const auto& tc : thread_contacts) {
        contacts_.insert(contacts_.end(), tc.begin(), tc.end());
    }
#else
    for (size_t i = 0; i < bodies.size(); ++i) {
        for (size_t j = i + 1; j < bodies.size(); ++j) {
            const RigidBody& a = bodies[i];
            const RigidBody& b = bodies[j];
            
            // Dispatch based on shape types
            if (a.type == ShapeType::Capsule && b.type == ShapeType::Capsule) {
                detectCapsuleCapsule(a, b, i, j, contacts_);
            } else if (a.type == ShapeType::Sphere && b.type == ShapeType::Sphere) {
                detectSphereSphere(a, b, i, j, contacts_);
            }
            // TODO: Add sphere-capsule detection if needed
        }
    }
#endif
    
    // Verbose summary: total contacts only (user request)
    if (config_.verbose) {
        std::cout << "[SoftContact] contacts=" << contacts_.size() << "\n";
    }

    // if (config_.verbose && contacts_.size() > 0) {
    //     std::cout << "[SoftContact] Detected " << contacts_.size() << " contacts\n";
    //     for (const auto& c : contacts_) {
    //         std::cout << "  Contact " << c.body_a << "-" << c.body_b 
    //                   << ": dist=" << c.distance << " limit=" << c.surface_limit << "\n"
    //                   << "    point_a=" << c.point_a.x << "," << c.point_a.y << "," << c.point_a.z << "\n"
    //                   << "    point_b=" << c.point_b.x << "," << c.point_b.y << "," << c.point_b.z << "\n"
    //                   << "    normal=" << c.normal.x << "," << c.normal.y << "," << c.normal.z << "\n"
    //                   << "    force_a=" << c.force_a.x << "," << c.force_a.y << "," << c.force_a.z << "\n"
    //                   << "    force_b=" << c.force_b.x << "," << c.force_b.y << "," << c.force_b.z << "\n";
    //     }
    // }
}

void SoftContactSolver::detectCapsuleCapsule(const RigidBody& a, const RigidBody& b,
                                              int idx_a, int idx_b, std::vector<ContactPrimitive>& out_contacts) {
    // Get capsule endpoints
    const glm::vec3 axis_a = a.axisY();
    const glm::vec3 axis_b = b.axisY();
    
    const glm::vec3 a1 = a.x - axis_a * a.cap.h;
    const glm::vec3 a2 = a.x + axis_a * a.cap.h;
    const glm::vec3 b1 = b.x - axis_b * b.cap.h;
    const glm::vec3 b2 = b.x + axis_b * b.cap.h;
    
    const double r_a = a.cap.r;
    const double r_b = b.cap.r;
    const double h = r_a + r_b;  // Surface limit
    
    // Find closest points between two line segments
    double s, t;
    closestPointsSegmentSegment(a1, a2, b1, b2, s, t);
    
    glm::vec3 point_a = a1 + static_cast<float>(s) * (a2 - a1);
    glm::vec3 point_b = b1 + static_cast<float>(t) * (b2 - b1);
    
    glm::vec3 diff = point_b - point_a;
    double distance = glm::length(diff);
    
    // Only create contact if within activation distance
    const double activation_dist = h + config_.delta;
    
    // Debug: Print first detection attempt
    static int debug_count = 0;
    if (config_.verbose && debug_count < 5) {
        std::cout << "[SoftContact::detect] dist=" << distance 
                  << " activation=" << activation_dist 
                  << " h=" << h << " delta=" << config_.delta << std::endl;
        debug_count++;
    }
    
    if (distance < activation_dist) {
        ContactPrimitive contact;
        
        // Classify contact type based on parameters
        const double eps = 1e-6;
        if (s < eps && t < eps) {
            contact.type = ContactType::POINT_TO_POINT;  // Both at start
        } else if (s > 1.0 - eps && t > 1.0 - eps) {
            contact.type = ContactType::POINT_TO_POINT;  // Both at end
        } else if (s < eps || s > 1.0 - eps) {
            contact.type = ContactType::EDGE_TO_POINT;   // A is point, B is edge
        } else if (t < eps || t > 1.0 - eps) {
            contact.type = ContactType::EDGE_TO_POINT;   // B is point, A is edge
        } else {
            contact.type = ContactType::EDGE_TO_EDGE;    // Both on edges
        }
        
        contact.body_a = idx_a;
        contact.body_b = idx_b;
        contact.point_a = point_a;
        contact.point_b = point_b;
        contact.distance = distance;
        contact.surface_limit = h;
        
        if (distance > 1e-10) {
            contact.normal = diff / static_cast<float>(distance);
        } else {
            // Fallback for coincident points
            contact.normal = glm::vec3(1, 0, 0);
        }
        
        out_contacts.push_back(contact);
    }
}

void SoftContactSolver::detectSphereSphere(const RigidBody& a, const RigidBody& b,
                                           int idx_a, int idx_b, std::vector<ContactPrimitive>& out_contacts) {
    // Get sphere centers and radii
    const glm::vec3& center_a = a.x;
    const glm::vec3& center_b = b.x;
    const double r_a = a.sphere.r;
    const double r_b = b.sphere.r;
    const double h = r_a + r_b;  // Surface limit
    
    // Compute distance between centers
    glm::vec3 diff = center_b - center_a;
    double distance = glm::length(diff);
    
    // Only create contact if within activation distance
    const double activation_dist = h + config_.delta;
    
    if (distance < activation_dist) {
        ContactPrimitive contact;
        
        // Sphere-sphere is always point-to-point (centers don't matter, contact is on surface)
        contact.type = ContactType::POINT_TO_POINT;
        contact.body_a = idx_a;
        contact.body_b = idx_b;
        
        // Contact points on each sphere surface
        if (distance > 1e-10) {
            glm::vec3 normal = diff / static_cast<float>(distance);
            contact.point_a = center_a + normal * static_cast<float>(r_a);
            contact.point_b = center_b - normal * static_cast<float>(r_b);
            contact.normal = normal;
        } else {
            // Spheres are at same position - use arbitrary normal
            contact.normal = glm::vec3(1, 0, 0);
            contact.point_a = center_a + contact.normal * static_cast<float>(r_a);
            contact.point_b = center_b - contact.normal * static_cast<float>(r_b);
        }
        
        contact.distance = distance;
        contact.surface_limit = h;
        
        out_contacts.push_back(contact);
    }
}

void SoftContactSolver::computeForces(std::vector<RigidBody>& bodies, double dt) {
    // Process each contact and accumulate potential energy
    double pe_sum = 0.0;
    for (auto& contact : contacts_) {
        // Compute normal forces based on contact type
        switch (contact.type) {
            case ContactType::POINT_TO_POINT:
                computeP2PForce(contact);
                break;
            case ContactType::EDGE_TO_POINT:
                computeE2PForce(contact);
                break;
            case ContactType::EDGE_TO_EDGE:
                computeE2EForce(contact);
                break;
        }
        
        // Compute friction if enabled
        if (config_.enable_friction) {
            computeFriction(contact, bodies[contact.body_a], bodies[contact.body_b], dt);
        } else {
            contact.friction_a = glm::vec3(0);
            contact.friction_b = glm::vec3(0);
        }
        
        // Apply forces to bodies
        auto& body_a = bodies[contact.body_a];
        auto& body_b = bodies[contact.body_b];
        
        // Total force = normal + friction
        glm::vec3 total_force_a = contact.force_a + contact.friction_a;
        glm::vec3 total_force_b = contact.force_b + contact.friction_b;
        
        // Apply forces at contact points (generates torques automatically)
        body_a.f += total_force_a;
        body_b.f += total_force_b;
        
        // Compute torques: τ = r × F
        glm::vec3 r_a = contact.point_a - body_a.x;
        glm::vec3 r_b = contact.point_b - body_b.x;
        
        body_a.tau += glm::cross(r_a, total_force_a);
        body_b.tau += glm::cross(r_b, total_force_b);

        // Accumulate potential energy for this contact (scaled by k_scaler)
        pe_sum += config_.k_scaler * potentialEnergy(contact.distance, contact.surface_limit);
    }
    lastPotentialEnergy_ = pe_sum;

    // if (config_.verbose && contacts_.size() > 0) {
    //     std::cout << "[SoftContact] Detected " << contacts_.size() << " contacts\n";
    //     for (const auto& c : contacts_) {
    //         std::cout << "  Contact " << c.body_a << "-" << c.body_b 
    //                   << ": dist=" << c.distance << " limit=" << c.surface_limit << "\n"
    //                   << "    point_a=" << c.point_a.x << "," << c.point_a.y << "," << c.point_a.z << "\n"
    //                   << "    point_b=" << c.point_b.x << "," << c.point_b.y << "," << c.point_b.z << "\n"
    //                   << "    normal=" << c.normal.x << "," << c.normal.y << "," << c.normal.z << "\n"
    //                   << "    force_a=" << c.force_a.x << "," << c.force_a.y << "," << c.force_a.z << "\n"
    //                   << "    force_b=" << c.force_b.x << "," << c.force_b.y << "," << c.force_b.z << "\n";
    //     }
    // }
}

double SoftContactSolver::potentialGradient(double distance, double h) const {
    const double delta = config_.delta;
    const double d = distance;
    
    // Piecewise gradient of potential energy
    if (d > h - delta) {
        // Non-penetrated regime: smooth log-barrier
        // U = (1/K1 * log(1 + exp(K1*(h-d))))²
        // dU/dd = -2/(K1²) * log(1+exp(arg)) * exp(arg)/(1+exp(arg))
        
        const double arg = K1_ * (h - d);
        const double exp_arg = std::exp(arg);
        const double log_term = std::log(1.0 + exp_arg);
        
        return -2.0 / (K1_ * K1_) * log_term * exp_arg / (1.0 + exp_arg);
    } else {
        // Penetrated regime: quadratic penalty
        // U = (h - d)²
        // dU/dd = -2*(h - d)
        
        return -2.0 * (h - d);
    }
}

double SoftContactSolver::potentialEnergy(double distance, double h) const {
    const double delta = config_.delta;
    const double d = distance;
    if (d > h - delta) {
        // Non-penetrated (barrier) regime
        const double arg = K1_ * (h - d);
        const double log_term = std::log(1.0 + std::exp(arg));
        double base = (log_term / K1_);
        return base * base; // (1/K1*log(1+exp(K1*(h-d))))^2
    } else {
        // Penetrated (quadratic) regime
        double pen = (h - d);
        return pen * pen; // (h-d)^2
    }
}

void SoftContactSolver::computeP2PForce(ContactPrimitive& contact) {
    // Point-to-point: simplest case
    // Force acts along line connecting the two points
    
    const double grad = potentialGradient(contact.distance, contact.surface_limit);
    const float force_mag = static_cast<float>(-config_.k_scaler * grad);  // F = -k * dU/dd
    
    // Force direction: gradient points from A to B
    // dU/dx_b = (dU/dd) * (dd/dx_b) = grad * normal
    // dU/dx_a = (dU/dd) * (dd/dx_a) = grad * (-normal)
    
    contact.force_a = -force_mag * contact.normal;  // Push A away from B
    contact.force_b = force_mag * contact.normal;   // Push B away from A
}

void SoftContactSolver::computeE2PForce(ContactPrimitive& contact) {
    // Edge-to-point: one body at endpoint, other on edge
    // For now, use same as P2P (simplified)
    // TODO: Implement proper E2P gradient distribution along edge
    
    computeP2PForce(contact);
}

void SoftContactSolver::computeE2EForce(ContactPrimitive& contact) {
    // Edge-to-edge: both bodies on interior of segments
    // For now, use same as P2P (simplified)
    // TODO: Implement proper E2E gradient with distributed forces
    
    computeP2PForce(contact);
}

void SoftContactSolver::computeFriction(ContactPrimitive& contact,
                                        const RigidBody& body_a,
                                        const RigidBody& body_b,
                                        double dt) {
    // Compute velocities at contact points
    glm::vec3 r_a = contact.point_a - body_a.x;
    glm::vec3 r_b = contact.point_b - body_b.x;
    
    glm::vec3 v_a = body_a.v + glm::cross(body_a.w, r_a);
    glm::vec3 v_b = body_b.v + glm::cross(body_b.w, r_b);
    
    glm::vec3 v_rel = v_a - v_b;
    
    // Project out normal component to get tangential velocity
    glm::vec3 v_tan = v_rel - glm::dot(v_rel, contact.normal) * contact.normal;
    float v_tan_mag = glm::length(v_tan);
    
    if (v_tan_mag < 1e-10) {
        // No tangential motion → no friction
        contact.friction_a = glm::vec3(0);
        contact.friction_b = glm::vec3(0);
        return;
    }
    
    // Smooth friction coefficient (sticking-to-sliding transition)
    double gamma;
    if (v_tan_mag > config_.nu) {
        gamma = 1.0;  // Sliding regime
    } else {
        // Sticking regime: smooth transition using tanh-like function
        gamma = 2.0 / (1.0 + std::exp(-K2_ * v_tan_mag)) - 1.0;
    }
    
    // Friction force magnitude: μ * γ * |F_normal|
    const float fn_mag = glm::length(contact.force_a);
    const glm::vec3 friction_dir = -v_tan / v_tan_mag;  // Oppose tangential motion
    
    const float friction_mag = static_cast<float>(config_.mu * gamma) * fn_mag;
    const glm::vec3 friction_force = friction_mag * friction_dir;
    
    contact.friction_a = friction_force;
    contact.friction_b = -friction_force;
}

// Geometric helper: Closest points on two line segments (Lumelsky algorithm from DisMech)
void SoftContactSolver::closestPointsSegmentSegment(
    const glm::vec3& a1, const glm::vec3& a2,
    const glm::vec3& b1, const glm::vec3& b2,
    double& s, double& t)
{
    // Same algorithm as DisMech for consistency
    const glm::vec3 e1 = a2 - a1;
    const glm::vec3 e2 = b2 - b1;
    const glm::vec3 e12 = b1 - a1;
    
    const double D1 = glm::dot(e1, e1);
    const double D2 = glm::dot(e2, e2);
    const double S1 = glm::dot(e1, e12);
    const double S2 = glm::dot(e2, e12);
    const double R = glm::dot(e1, e2);
    
    const double den = D1 * D2 - R * R;
    
    double uf = 0.0;
    
    auto fixBound = [](double& x) -> bool {
        if (x > 1.0) {
            x = 1.0;
            return true;
        } else if (x < 0.0) {
            x = 0.0;
            return true;
        }
        return false;
    };
    
    if (den == 0.0) {
        // Parallel segments
        s = 0.0;
        t = -S2 / D2;
        uf = t;
        fixBound(uf);
        
        if (uf != t) {
            s = (uf * R + S1) / D1;
            fixBound(s);
            t = uf;
        }
    } else {
        // General case
        s = (S1 * D2 - S2 * R) / den;
        fixBound(s);
        t = (s * R - S2) / D2;
        uf = t;
        fixBound(uf);
        
        if (uf != t) {
            s = (uf * R + S1) / D1;
            fixBound(s);
            t = uf;
        }
    }
}

double SoftContactSolver::distancePointToSegment(
    const glm::vec3& point,
    const glm::vec3& seg_start, const glm::vec3& seg_end,
    double& t)
{
    const glm::vec3 d = seg_end - seg_start;
    const glm::vec3 r = point - seg_start;
    
    const double len_sq = glm::dot(d, d);
    if (len_sq < 1e-10) {
        t = 0.0;
        return glm::length(r);
    }
    
    t = glm::clamp(glm::dot(r, d) / len_sq, 0.0, 1.0);
    const glm::vec3 closest = seg_start + static_cast<float>(t) * d;
    
    return glm::length(point - closest);
}
