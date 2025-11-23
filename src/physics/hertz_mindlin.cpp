/**
 * @file hertz_mindlin.cpp
 * @brief Implementation of Hertz-Mindlin contact model
 */

#include "physics/hertz_mindlin.hpp"
#include "physics/rigid_body.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

HertzMindlinSolver::HertzMindlinSolver(const HertzMindlinCfg& config)
    : config_(config), last_potential_energy_(0.0)
{
}

void HertzMindlinSolver::setConfig(const HertzMindlinCfg& config) {
    config_ = config;
}

void HertzMindlinSolver::clearHistory() {
    contact_history_.clear();
}

void HertzMindlinSolver::detectContacts(const std::vector<RigidBody>& bodies) {
    contacts_.clear();
    
    // Detect all sphere-sphere contacts
    for (size_t i = 0; i < bodies.size(); ++i) {
        for (size_t j = i + 1; j < bodies.size(); ++j) {
            const RigidBody& a = bodies[i];
            const RigidBody& b = bodies[j];
            
            // Only handle sphere-sphere for now
            if (a.type == ShapeType::Sphere && b.type == ShapeType::Sphere) {
                detectSphereSphere(a, b, i, j);
            }
        }
    }
}

void HertzMindlinSolver::detectSphereSphere(const RigidBody& a, const RigidBody& b,
                                            int idx_a, int idx_b) {
    const glm::vec3& center_a = a.x;
    const glm::vec3& center_b = b.x;
    const double r_a = a.sphere.r;
    const double r_b = b.sphere.r;
    
    glm::vec3 diff = center_b - center_a;
    double distance = glm::length(diff);
    
    // Check for overlap
    double sum_radii = r_a + r_b;
    if (distance < sum_radii) {
        HMContact contact;
        contact.body_a = idx_a;
        contact.body_b = idx_b;
        contact.overlap = sum_radii - distance;
        
        // Compute contact geometry
        if (distance > 1e-10) {
            contact.normal = diff / static_cast<float>(distance);
        } else {
            // Coincident spheres - arbitrary normal
            contact.normal = glm::vec3(1, 0, 0);
        }
        
        // Contact points on sphere surfaces
        contact.point_a = center_a + contact.normal * static_cast<float>(r_a);
        contact.point_b = center_b - contact.normal * static_cast<float>(r_b);
        
        // Compute effective properties
        contact.effective_radius = (r_a * r_b) / (r_a + r_b);
        contact.effective_mass = (a.mass * b.mass) / (a.mass + b.mass);
        
        // Effective Young's modulus: E* = E/(2(1-ν²))
        double nu = config_.poisson_ratio;
        contact.effective_E = config_.youngs_modulus / (2.0 * (1.0 - nu * nu));
        
        // Effective shear modulus: G* = E/(2(1+ν))
        contact.effective_G = config_.youngs_modulus / (2.0 * (1.0 + nu));
        
        contacts_.push_back(contact);
    }
}

void HertzMindlinSolver::computeForces(std::vector<RigidBody>& bodies, double dt) {
    last_potential_energy_ = 0.0;
    
    for (auto& contact : contacts_) {
        const RigidBody& body_a = bodies[contact.body_a];
        const RigidBody& body_b = bodies[contact.body_b];
        
        // Get or create contact state
        uint64_t key = pairKey(contact.body_a, contact.body_b);
        HMContactState& state = contact_history_[key];
        
        // Compute normal force (Hertz)
        computeHertzForce(contact, state, body_a, body_b, dt);
        
        // Compute tangential force (Mindlin) if enabled
        if (config_.enable_tangential) {
            computeMindlinForce(contact, state, body_a, body_b, dt);
        } else {
            contact.force_t = glm::vec3(0.0f);
        }
        
        // Compute rolling friction if enabled
        if (config_.enable_rolling) {
            computeRollingFriction(contact, body_a, body_b);
        } else {
            contact.torque_a = glm::vec3(0.0f);
            contact.torque_b = glm::vec3(0.0f);
        }
        
        // Apply forces and torques to bodies
        bodies[contact.body_a].f += contact.force_n + contact.force_t;
        bodies[contact.body_b].f -= contact.force_n + contact.force_t;
        
        // Torques from tangential force
        glm::vec3 r_a = contact.point_a - body_a.x;
        glm::vec3 r_b = contact.point_b - body_b.x;
        
        bodies[contact.body_a].tau += glm::cross(r_a, contact.force_t) + contact.torque_a;
        bodies[contact.body_b].tau += glm::cross(r_b, -contact.force_t) + contact.torque_b;
        
        // Accumulate potential energy (Hertzian part only)
        // U = (2/5)k_n·δ^(5/2) = (8/15)E*√R*·δ^(5/2)
        double delta = contact.overlap;
        double R_star = contact.effective_radius;
        double E_star = contact.effective_E;
        double U = (8.0 / 15.0) * E_star * std::sqrt(R_star) * std::pow(delta, 2.5);
        last_potential_energy_ += U;
        
        // Update state
        state.prev_overlap = contact.overlap;
    }
    
    // Clean up history for contacts that no longer exist
    // Keep history for recently broken contacts (for re-contact)
    std::vector<uint64_t> active_keys;
    active_keys.reserve(contacts_.size());
    for (const auto& c : contacts_) {
        active_keys.push_back(pairKey(c.body_a, c.body_b));
    }
    
    // Remove very old contacts (simple cleanup)
    if (contact_history_.size() > active_keys.size() * 2) {
        std::unordered_map<uint64_t, HMContactState> new_history;
        for (uint64_t key : active_keys) {
            new_history[key] = contact_history_[key];
        }
        contact_history_ = std::move(new_history);
    }
}

void HertzMindlinSolver::computeHertzForce(HMContact& contact, HMContactState& state,
                                           const RigidBody& body_a, const RigidBody& body_b,
                                           double dt) {
    double delta = contact.overlap;
    double R_star = contact.effective_radius;
    double E_star = contact.effective_E;
    double m_star = contact.effective_mass;
    
    // Hertzian normal stiffness: k_n = (4/3)E*√(R*·δ)
    double k_n = (4.0 / 3.0) * E_star * std::sqrt(R_star * delta);
    
    // Normal elastic force: F_n = k_n·δ^(3/2)
    double F_elastic = k_n * std::pow(delta, 1.5);
    
    // Normal damping force: F_d = -γ_n·√(m*k_n)·v_n
    // Relative velocity at contact point
    glm::vec3 v_a = body_a.v + glm::cross(body_a.w, contact.point_a - body_a.x);
    glm::vec3 v_b = body_b.v + glm::cross(body_b.w, contact.point_b - body_b.x);
    glm::vec3 v_rel = v_b - v_a;
    
    double v_n = glm::dot(v_rel, contact.normal);  // Normal component
    
    // Damping coefficient: C_n = -γ_n·2√(m*k_n)
    double C_n = config_.normal_damping * 2.0 * std::sqrt(m_star * k_n);
    double F_damping = C_n * v_n;
    
    // Total normal force (repulsive)
    double F_n_magnitude = F_elastic + F_damping;
    
    // Ensure non-attractive (only compressive contacts)
    F_n_magnitude = std::max(0.0, F_n_magnitude);
    
    contact.force_n = contact.normal * static_cast<float>(F_n_magnitude);
}

void HertzMindlinSolver::computeMindlinForce(HMContact& contact, HMContactState& state,
                                             const RigidBody& body_a, const RigidBody& body_b,
                                             double dt) {
    double delta = contact.overlap;
    double R_star = contact.effective_radius;
    double G_star = contact.effective_G;
    double m_star = contact.effective_mass;
    
    // Tangential stiffness: k_t = 8G*√(R*·δ)
    double k_t = 8.0 * G_star * std::sqrt(R_star * delta);
    
    // Relative velocity at contact
    glm::vec3 v_a = body_a.v + glm::cross(body_a.w, contact.point_a - body_a.x);
    glm::vec3 v_b = body_b.v + glm::cross(body_b.w, contact.point_b - body_b.x);
    glm::vec3 v_rel = v_b - v_a;
    
    // Tangential component (perpendicular to normal)
    double v_n = glm::dot(v_rel, contact.normal);
    glm::vec3 v_t = v_rel - contact.normal * static_cast<float>(v_n);
    
    // Incremental tangential displacement
    glm::vec3 delta_xi_t = v_t * static_cast<float>(dt);
    
    // Rotate previous displacement into current contact frame if needed
    // (For sphere-sphere, this is simplified - just accumulate)
    state.tangential_displacement += delta_xi_t;
    
    // Elastic tangential force
    glm::vec3 F_t_elastic = -static_cast<float>(k_t) * state.tangential_displacement;
    
    // Tangential damping: C_t = -γ_t·2√(m*k_t)
    double C_t = config_.tangential_damping * 2.0 * std::sqrt(m_star * k_t);
    glm::vec3 F_t_damping = -static_cast<float>(C_t) * v_t;
    
    // Total tangential force
    glm::vec3 F_t = F_t_elastic + F_t_damping;
    
    // Coulomb friction limit: |F_t| ≤ μ|F_n|
    double F_n_mag = glm::length(contact.force_n);
    double F_t_mag = glm::length(F_t);
    double F_t_max = config_.friction_coeff * F_n_mag;
    
    if (F_t_mag > F_t_max && F_t_mag > 1e-10) {
        // Sliding: truncate to Coulomb limit and reset spring
        F_t = F_t * static_cast<float>(F_t_max / F_t_mag);
        
        // Reset tangential displacement to match Coulomb limit
        if (k_t > 1e-10) {
            state.tangential_displacement = -F_t / static_cast<float>(k_t);
        }
    }
    
    contact.force_t = F_t;
}

void HertzMindlinSolver::computeRollingFriction(HMContact& contact,
                                               const RigidBody& body_a, const RigidBody& body_b) {
    // Rolling friction torque: τ_r = -μ_r·R·|F_n|·ω̂_rel
    // Where ω_rel is the relative angular velocity
    
    glm::vec3 omega_rel = body_a.w - body_b.w;
    double omega_mag = glm::length(omega_rel);
    
    if (omega_mag < 1e-10) {
        contact.torque_a = glm::vec3(0.0f);
        contact.torque_b = glm::vec3(0.0f);
        return;
    }
    
    glm::vec3 omega_hat = omega_rel / static_cast<float>(omega_mag);
    
    // Rolling friction magnitude
    double F_n_mag = glm::length(contact.force_n);
    double R_eff = contact.effective_radius;
    double tau_r_mag = config_.rolling_friction_coeff * R_eff * F_n_mag;
    
    // Apply torque opposite to relative rotation
    glm::vec3 tau_r = -static_cast<float>(tau_r_mag) * omega_hat;
    
    // Distribute torque between bodies (inverse to inertia, simplified: equal)
    contact.torque_a = tau_r * 0.5f;
    contact.torque_b = -tau_r * 0.5f;
}

