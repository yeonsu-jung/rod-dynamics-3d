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
    // Compute damping coefficients from restitution
    config_.computeDamping();
}

void HertzMindlinSolver::setConfig(const HertzMindlinCfg& config) {
    config_ = config;
    config_.computeDamping();
}

void HertzMindlinSolver::clearHistory() {
    contact_history_.clear();
    sphere_indices_.clear();
    grid_cells_.clear();
}

void HertzMindlinSolver::detectContacts(const std::vector<RigidBody>& bodies) {
    ++frame_counter_;
    contacts_.clear();
    sphere_indices_.clear();
    sphere_indices_.reserve(bodies.size());
    int num_free = 0;
    int free_idx = -1;
    for (size_t i = 0; i < bodies.size(); ++i) {
        if (bodies[i].type == ShapeType::Sphere) {
            sphere_indices_.push_back(static_cast<int>(i));
            if (bodies[i].invMass > 0.0f) {
                num_free++;
                free_idx = static_cast<int>(i);
            }
        }
    }
    
    if (sphere_indices_.size() < 2) {
        return;
    }
    
    if (num_free == 1 && free_idx >= 0) {
        const RigidBody& a = bodies[free_idx];
        for (int b_idx : sphere_indices_) {
            if (b_idx == free_idx) continue;
            detectSphereSphere(a, bodies[b_idx], free_idx, b_idx);
        }
        return;
    }
    
    const size_t minBodiesForGrid = static_cast<size_t>(std::max(2, config_.broadphase_min_bodies));
    const bool useGrid = config_.use_uniform_grid && sphere_indices_.size() >= minBodiesForGrid;
    
    if (!useGrid) {
        detectContactsNaive(bodies, sphere_indices_);
        return;
    }
    
    double cell_size = config_.broadphase_cell_size;
    if (cell_size <= 0.0) {
        cell_size = computeAdaptiveCellSize(bodies, sphere_indices_);
    }
    if (cell_size <= 0.0) {
        detectContactsNaive(bodies, sphere_indices_);
        return;
    }
    
    grid_cells_.clear();
    grid_cells_.reserve(sphere_indices_.size() * 2);
    
    for (int idx : sphere_indices_) {
        const RigidBody& a = bodies[idx];
        GridKey baseKey = cellKey(a.x, cell_size);
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    GridKey key{baseKey.x + dx, baseKey.y + dy, baseKey.z + dz};
                    auto it = grid_cells_.find(key);
                    if (it == grid_cells_.end()) continue;
                    for (int otherIdx : it->second) {
                        if (otherIdx == idx) continue;
                        const RigidBody& b = bodies[otherIdx];
                        detectSphereSphere(a, b, idx, otherIdx);
                    }
                }
            }
        }
        grid_cells_[baseKey].push_back(idx);
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

void HertzMindlinSolver::detectContactsNaive(const std::vector<RigidBody>& bodies,
                                             const std::vector<int>& sphere_indices) {
    for (size_t i = 0; i + 1 < sphere_indices.size(); ++i) {
        for (size_t j = i + 1; j < sphere_indices.size(); ++j) {
            int idx_a = sphere_indices[i];
            int idx_b = sphere_indices[j];
            detectSphereSphere(bodies[idx_a], bodies[idx_b], idx_a, idx_b);
        }
    }
}

double HertzMindlinSolver::computeAdaptiveCellSize(const std::vector<RigidBody>& bodies,
                                                   const std::vector<int>& sphere_indices) const {
    if (sphere_indices.empty()) {
        return 0.0;
    }
    double sumDiameter = 0.0;
    double maxDiameter = 0.0;
    for (int idx : sphere_indices) {
        double diameter = 2.0 * bodies[idx].sphere.r;
        sumDiameter += diameter;
        maxDiameter = std::max(maxDiameter, diameter);
    }
    double avgDiameter = sumDiameter / static_cast<double>(sphere_indices.size());
    double cell = std::max(maxDiameter, avgDiameter * 1.25);
    return cell;
}

HertzMindlinSolver::GridKey HertzMindlinSolver::cellKey(const glm::vec3& pos, double cell_size) const {
    const double inv = 1.0 / cell_size;
    return GridKey{
        static_cast<int>(std::floor(pos.x * inv)),
        static_cast<int>(std::floor(pos.y * inv)),
        static_cast<int>(std::floor(pos.z * inv))
    };
}

void HertzMindlinSolver::computeForces(std::vector<RigidBody>& bodies, double dt) {
    last_potential_energy_ = 0.0;
    if (contacts_.empty()) {
        pruneContactHistory();
        return;
    }

    const size_t target_capacity = contact_history_.size() + contacts_.size() * 2;
    if (target_capacity > contact_history_.size()) {
        contact_history_.reserve(target_capacity);
    }
    
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
        state.last_frame = frame_counter_;
    }
    
    pruneContactHistory();
}

void HertzMindlinSolver::computeHertzForce(HMContact& contact, HMContactState& state,
                                           const RigidBody& body_a, const RigidBody& body_b,
                                           double dt) {
    double delta = contact.overlap;
    double R_star = contact.effective_radius;
    double E_star = contact.effective_E;
    double m_star = contact.effective_mass;
    
    // Hertzian normal stiffness: k_n = (4/3)E*√(R*·δ)
    double safe_delta = std::max(delta, 0.0);
    double sqrt_delta = std::sqrt(safe_delta);
    double sqrt_R = std::sqrt(std::max(R_star, 0.0));
    double k_n = (4.0 / 3.0) * E_star * sqrt_R * sqrt_delta;
    
    // Normal elastic force: F_n = k_n·δ^(3/2)
    double F_elastic = k_n * safe_delta * sqrt_delta;
    
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
    double safe_delta = std::max(delta, 0.0);
    double sqrt_delta = std::sqrt(safe_delta);
    double sqrt_R = std::sqrt(std::max(R_star, 0.0));
    double k_t = 8.0 * G_star * sqrt_R * sqrt_delta;
    
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
    
    // Velocity-dependent friction coefficient: μ(v) = μ_k + (μ_s - μ_k)·exp(-|v_t|/v_c)
    // This provides smooth transition from static (μ_s) to kinetic (μ_k) friction
    double v_t_mag = glm::length(v_t);
    double mu_effective = config_.friction_coeff; // μ_k (kinetic)
    if (config_.friction_static_coeff > config_.friction_coeff && config_.friction_transition_vel > 0.0) {
        double transition_factor = std::exp(-v_t_mag / config_.friction_transition_vel);
        mu_effective += (config_.friction_static_coeff - config_.friction_coeff) * transition_factor;
    }
    
    // Coulomb friction limit: |F_t| ≤ μ(v)·|F_n|
    double F_n_mag = glm::length(contact.force_n);
    double F_t_mag = glm::length(F_t);
    double F_t_max = mu_effective * F_n_mag;
    
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

void HertzMindlinSolver::pruneContactHistory() {
    for (auto it = contact_history_.begin(); it != contact_history_.end(); ) {
        if (frame_counter_ - it->second.last_frame > kHistoryRetainFrames || it->second.last_frame == 0) {
            it = contact_history_.erase(it);
        } else {
            ++it;
        }
    }
}

