/**
 * @file soft_contact.hpp
 * @brief Soft penalty-based contact solver for rod collisions
 * 
 * Implements smooth potential energy-based contacts inspired by DisMech.
 * Uses penalty forces instead of impulses for better energy conservation.
 */

#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <cmath>
#include "config/config.hpp"  // For SoftContactCfg

#include <unordered_map>

struct RigidBody;

/**
 * @brief Contact primitive types
 */
enum class ContactType {
    POINT_TO_POINT,  ///< Endpoint to endpoint
    EDGE_TO_POINT,   ///< Edge to endpoint
    EDGE_TO_EDGE     ///< Edge to edge (most common for rods)
};

/**
 * @brief Detected contact between two capsules
 */
struct ContactPrimitive {
    ContactType type;
    int body_a, body_b;              ///< Rod indices
    glm::vec3 point_a, point_b;      ///< Contact points on each body
    glm::vec3 normal;                ///< Normal from A to B
    double distance;                 ///< Signed distance (negative = penetration)
    double surface_limit;            ///< Sum of radii (h = r_a + r_b)
    
    // Forces computed (output)
    glm::vec3 force_a, force_b;      ///< Normal forces
    glm::vec3 friction_a, friction_b; ///< Friction forces (if enabled)
};

/**
 * @brief Soft penalty-based contact solver
 * 
 * Computes smooth contact forces from potential energy gradients.
 * Allows small controlled penetrations for stability and energy conservation.
 */
class SoftContactSolver {
public:
    explicit SoftContactSolver(const SoftContactCfg& config = SoftContactCfg());
    
    /**
     * @brief Detect contacts between all capsule pairs
     * @param bodies Vector of rigid bodies (capsules)
     */
    void detectContacts(const std::vector<RigidBody>& bodies);
    
    /**
     * @brief Compute contact forces and apply to bodies
     * @param bodies Vector of rigid bodies to apply forces to
     * @param dt Time step size
     */
    void computeForces(std::vector<RigidBody>& bodies, double dt);
    
    /**
     * @brief Get current detected contacts (for visualization/debugging)
     */
    const std::vector<ContactPrimitive>& getContacts() const { return contacts_; }
    
    /**
     * @brief Get number of active contacts
     */
    size_t getNumContacts() const { return contacts_.size(); }

    /**
     * @brief Last accumulated total soft-contact potential energy (J) from most recent computeForces call.
     *
     * The potential energy reported is the sum over all active contacts of the underlying
     * piecewise barrier/penalty potential scaled by k_scaler. This lets the application
     * monitor energy exchange between kinetic and stored elastic energy in the soft model.
     */
    double getLastPotentialEnergy() const { return lastPotentialEnergy_; }

    struct BroadphaseStats {
        double count_ms = 0.0;
        double prefix_ms = 0.0;
        double fill_ms = 0.0;
        double sort_ms = 0.0;
        double detect_ms = 0.0;
    };
    
    const BroadphaseStats& getStats() const { return stats_; }
    
    /**
     * @brief Update configuration at runtime
     */
    void setConfig(const SoftContactCfg& config);
    
private:
    SoftContactCfg config_;
    std::vector<ContactPrimitive> contacts_;
    BroadphaseStats stats_;
    
    double K1_;  ///< Stiffness parameter = 15/delta
    double K2_;  ///< Friction smoothness = 15/nu
    double lastPotentialEnergy_ = 0.0; ///< Accumulated potential energy of contacts (J) after latest computeForces()
    
    // Spatial Hash Grid
    struct GridKey {
        int x, y, z;
        bool operator==(const GridKey& other) const {
            return x == other.x && y == other.y && z == other.z;
        }
    };
    struct GridKeyHash {
        std::size_t operator()(const GridKey& k) const {
            // Simple hash combining x,y,z
            return ((std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1)) >> 1) ^ (std::hash<int>()(k.z) << 1);
        }
    };
    
    // Map from grid cell to list of body indices
    // Note: For parallel access, we might need a different structure or thread-local grids
    // For now, we'll build it serially or use thread-local vectors and merge
    using GridMap = std::unordered_map<GridKey, std::vector<int>, GridKeyHash>;
    
    void detectContactsSpatialHash(const std::vector<RigidBody>& bodies);
    void detectContactsNaive(const std::vector<RigidBody>& bodies);
    
    double computeAdaptiveCellSize(const std::vector<RigidBody>& bodies) const;
    void insertBodyIntoGrid(int bodyIdx, const RigidBody& body, double cellSize, GridMap& grid);
    
    // Contact detection helpers
    void detectCapsuleCapsule(const RigidBody& a, const RigidBody& b,
                              int idx_a, int idx_b, std::vector<ContactPrimitive>& out_contacts);
    void detectSphereSphere(const RigidBody& a, const RigidBody& b,
                            int idx_a, int idx_b, std::vector<ContactPrimitive>& out_contacts);
    
    // Force computation for each contact type
    void computeP2PForce(ContactPrimitive& contact);
    void computeE2PForce(ContactPrimitive& contact);
    void computeE2EForce(ContactPrimitive& contact);
    
    // Friction computation
    void computeFriction(ContactPrimitive& contact,
                        const RigidBody& body_a,
                        const RigidBody& body_b,
                        double dt);
    
    // Potential energy gradient helpers
    double potentialGradient(double distance, double h) const;
    double potentialEnergy(double distance, double h) const; ///< Unscaled base potential (before k_scaler)
    
    // Geometry helpers
    void getAABB(const RigidBody& b, glm::vec3& min_pt, glm::vec3& max_pt) const;
    bool checkAABBOverlap(const RigidBody& a, const RigidBody& b) const;

    static void closestPointsSegmentSegment(
        const glm::vec3& a1, const glm::vec3& a2,
        const glm::vec3& b1, const glm::vec3& b2,
        double& s, double& t);
    
    static double distancePointToSegment(
        const glm::vec3& point,
        const glm::vec3& seg_start, const glm::vec3& seg_end,
        double& t);
};

