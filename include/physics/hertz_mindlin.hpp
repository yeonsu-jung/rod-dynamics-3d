/**
 * @file hertz_mindlin.hpp
 * @brief Hertz-Mindlin contact model for granular spheres
 * 
 * Implements physically-based Hertzian normal contact and Mindlin tangential friction
 * for sphere-sphere interactions. More accurate than penalty methods for granular materials.
 * 
 * References:
 * - Mindlin & Deresiewicz (1953): Elastic spheres in contact
 * - Cundall & Strack (1979): DEM formulation
 * - Silbert et al. (2001): Granular flow simulations
 * - Makse et al. (1999, 2000): Jammed sphere packings
 */

#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <unordered_map>
#include "config/config.hpp"

struct RigidBody;

/**
 * @brief Contact state for tracking tangential displacement history
 * 
 * Hertz-Mindlin requires tracking incremental tangential displacement
 * between particles to compute elastic tangential forces.
 */
struct HMContactState {
    glm::vec3 tangential_displacement{0.0f};  ///< Accumulated tangential spring displacement
    glm::vec3 rolling_displacement{0.0f};     ///< Accumulated rolling displacement for rolling friction
    double prev_overlap{0.0};                 ///< Previous overlap for damping calculation
};

/**
 * @brief Hertz-Mindlin contact between two spheres
 */
struct HMContact {
    int body_a, body_b;                       ///< Sphere indices
    glm::vec3 point_a, point_b;               ///< Contact points on each sphere surface
    glm::vec3 normal;                         ///< Normal direction (from A to B)
    double overlap;                           ///< Overlap depth δ = r_a + r_b - distance
    
    // Computed forces/torques
    glm::vec3 force_n;                        ///< Normal force
    glm::vec3 force_t;                        ///< Tangential (friction) force
    glm::vec3 torque_a, torque_b;             ///< Rolling friction torques
    
    // Contact properties
    double effective_radius;                  ///< R* = (r_a·r_b)/(r_a + r_b)
    double effective_mass;                    ///< m* = (m_a·m_b)/(m_a + m_b)
    double effective_E;                       ///< E* = E/(2(1-ν²))
    double effective_G;                       ///< G* = E/(2(1+ν)) for shear
};

/**
 * @brief Configuration for Hertz-Mindlin contact model
 */
struct HertzMindlinCfg {
    // Material properties (typical values for glass beads)
    double youngs_modulus{7e10};              ///< Young's modulus E (Pa) [default: glass ~70 GPa]
    double poisson_ratio{0.25};               ///< Poisson's ratio ν (dimensionless) [typical: 0.2-0.3]
    double restitution_coeff{0.9};            ///< Coefficient of restitution e [0 = inelastic, 1 = elastic]
    double friction_coeff{0.3};               ///< Friction coefficient μ [typical: 0.3-0.5]
    double rolling_friction_coeff{0.01};      ///< Rolling friction coefficient μ_r [typically << μ]
    
    // Damping parameters (derived from restitution)
    double normal_damping{0.0};               ///< Normal damping coefficient γ_n (computed from e)
    double tangential_damping{0.0};           ///< Tangential damping coefficient γ_t
    
    // Simulation parameters
    bool enable_tangential{true};             ///< Enable tangential forces (Mindlin)
    bool enable_rolling{true};                ///< Enable rolling friction torque
    bool verbose{false};                      ///< Debug output
    
    HertzMindlinCfg() {
        // Compute damping from restitution coefficient
        // See Silbert et al. (2001) for derivation
        computeDamping();
    }
    
    void computeDamping() {
        if (restitution_coeff >= 1.0) {
            normal_damping = 0.0;
            tangential_damping = 0.0;
        } else {
            // Normal damping: γ_n = -2√(m*k_n) ln(e) / √(π² + ln²(e))
            double ln_e = std::log(restitution_coeff);
            normal_damping = -ln_e / std::sqrt(M_PI * M_PI + ln_e * ln_e);
            
            // Tangential damping (typically same as normal)
            tangential_damping = normal_damping;
        }
    }
    
    void setRestitution(double e) {
        restitution_coeff = e;
        computeDamping();
    }
};

/**
 * @brief Hertz-Mindlin contact solver for sphere-sphere interactions
 * 
 * Implements:
 * - Hertzian normal force: F_n = k_n·δ^(3/2) - γ_n·v_n
 * - Mindlin tangential force: F_t = -k_t·ξ_t - γ_t·v_t (with Coulomb limit)
 * - Rolling friction: τ_r = -μ_r·R·|F_n|·ω̂
 * 
 * Where:
 * - k_n = (4/3)E*√(R*δ) is the nonlinear normal stiffness
 * - k_t = 8G*√(R*δ) is the tangential stiffness
 * - ξ_t is the tangential displacement spring
 */
class HertzMindlinSolver {
public:
    explicit HertzMindlinSolver(const HertzMindlinCfg& config = HertzMindlinCfg());
    
    /**
     * @brief Detect contacts between all sphere pairs
     * @param bodies Vector of rigid bodies (must be spheres)
     */
    void detectContacts(const std::vector<RigidBody>& bodies);
    
    /**
     * @brief Compute Hertz-Mindlin forces and apply to bodies
     * @param bodies Vector of rigid bodies to apply forces to
     * @param dt Time step size
     */
    void computeForces(std::vector<RigidBody>& bodies, double dt);
    
    /**
     * @brief Get current detected contacts
     */
    const std::vector<HMContact>& getContacts() const { return contacts_; }
    
    /**
     * @brief Get number of active contacts
     */
    size_t getNumContacts() const { return contacts_.size(); }
    
    /**
     * @brief Get total potential energy stored in contacts
     */
    double getLastPotentialEnergy() const { return last_potential_energy_; }
    
    /**
     * @brief Update configuration at runtime
     */
    void setConfig(const HertzMindlinCfg& config);
    
    /**
     * @brief Clear contact history (e.g., when resetting simulation)
     */
    void clearHistory();
    
private:
    HertzMindlinCfg config_;
    std::vector<HMContact> contacts_;
    
    // Contact state history (key = pair hash)
    std::unordered_map<uint64_t, HMContactState> contact_history_;
    
    double last_potential_energy_{0.0};
    
    // Contact detection
    void detectSphereSphere(const RigidBody& a, const RigidBody& b,
                           int idx_a, int idx_b);
    
    // Force computation
    void computeHertzForce(HMContact& contact, HMContactState& state,
                          const RigidBody& body_a, const RigidBody& body_b,
                          double dt);
    
    void computeMindlinForce(HMContact& contact, HMContactState& state,
                            const RigidBody& body_a, const RigidBody& body_b,
                            double dt);
    
    void computeRollingFriction(HMContact& contact,
                               const RigidBody& body_a, const RigidBody& body_b);
    
    // Utility
    static uint64_t pairKey(int a, int b) {
        if (a > b) std::swap(a, b);
        return (uint64_t(a) << 32) | uint64_t(b);
    }
};

