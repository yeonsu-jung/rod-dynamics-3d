/**
 * @file World.hpp
 * @brief Core physics simulation world - owns state and stepping logic
 */

#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <random>
#include <cstdint>
#include <unordered_map>

#include "physics/rigid_body.hpp"
#include "physics/collision.hpp"
#include "physics/solver.hpp"
#include "physics/soft_contact.hpp"

namespace core {

/**
 * @brief World encapsulates the core physics simulation state and stepping.
 * 
 * Responsibilities:
 * - Owns rigid bodies (rods), floor
 * - Owns solver configuration and soft contact solver
 * - Executes physics step (integrate, collide, solve, wrap)
 * - Computes derived metrics (KE, entanglement)
 * - Applies random forces when configured
 * 
 * Does NOT own:
 * - Logging (delegated to observers/logger)
 * - Rendering (handled by environment)
 * - Window/input management (handled by environment)
 */
class World {
public:
    World();
    ~World() = default;

    // ---- Configuration ----
    void setGravity(const glm::vec3& g) { gravity = g; }
    void setDt(float delta) { dt = delta; }
    float getDt() const { return dt; }
    void setSolver(const SolverConfig& cfg) { solver = cfg; }
    void setSoftContactSolver(const SoftContactSolver& scs) { softContactSolver = scs; }
    void setSoftContactEnabled(bool enabled) { softContactEnabled = enabled; }
    
    // Periodic boundary conditions
    void setPBC(bool enabled, const glm::vec3& min, const glm::vec3& max, float cellSize);
    
    // Random force field
    void setRandomForce(bool enabled, float fSigma, float tauMag, unsigned int seed = 0);
    
    // Sleeping parameters
    void setSleepThresholds(float linThresh, float angThresh, float timeThresh);

    // ---- Body management ----
    void addRod(const RigidBody& rb);
    void setFloor(const RigidBody& floor);
    void clearRods();
    size_t numRods() const { return rods.size(); }
    const std::vector<RigidBody>& getRods() const { return rods; }
    std::vector<RigidBody>& getRodsMutable() { return rods; }
    const RigidBody& getFloor() const { return floorRB; }

    // ---- Simulation stepping ----
    void step();
    void stepWithSubsteps(int substeps);
    
    // ---- Queries ----
    double totalKE() const;
    uint64_t getFrameIndex() const { return frameIndex; }
    void setFrameIndex(uint64_t idx) { frameIndex = idx; }
    
    // Access last step diagnostics
    size_t getLastHitCount() const { return lastHitCount; }
    size_t getLastIslandCount() const { return lastIslandCount; }
    double getLastSoftPE() const { return lastSoftPotentialEnergy; }
    
    // KE checkpoints (for debugging energy evolution)
    double getKEAfterIntegrate() const { return keAfterIntegrate; }
    double getKEAfterWarmstart() const { return keAfterWarmstart; }
    double getKEAfterSolve() const { return keAfterSolve; }
    double getKEAfterPosCorrect() const { return keAfterPosCorrect; }
    double getKEAfterPBCWrap() const { return keAfterPBCWrap; }

    // Sleeping state
    bool isAsleep(size_t i) const { return i < sleeping.size() && sleeping[i]; }
    void wake(size_t i);
    void wakeAll();

    // Access to contacts for diagnostics/logging
    struct Hit {
        int a = -1, b = -1;
        Contact c{};
    };
    const std::vector<Hit>& getLastContacts() const { return hitsScratch; }

private:
    // ---- Simulation state ----
    std::vector<RigidBody> rods;
    RigidBody floorRB;
    uint64_t frameIndex = 0;

    // ---- Physics parameters ----
    glm::vec3 gravity{0.0f, -10.0f, 0.0f};
    float dt = 1.0f / 600.0f;
    SolverConfig solver{};
    SoftContactSolver softContactSolver{};
    bool softContactEnabled = false;

    // ---- Periodic boundary conditions ----
    bool usePBC = false;
    glm::vec3 pbcMin{-3,-1,-3}, pbcMax{3,3,3};
    float cellSize = 0.6f;

    // ---- Sleeping ----
    float sleepLinThresh = 0.02f;
    float sleepAngThresh = 0.05f;
    float sleepTimeThresh = 0.6f;
    std::vector<float> sleepTimer;
    std::vector<uint8_t> sleeping;

    // ---- Random forces ----
    bool useRandomForce = false;
    std::mt19937 genRandomForce;
    std::normal_distribution<float> normal_f{0.0f, 1.0f};
    std::uniform_real_distribution<float> uni_u{-1.0f, 1.0f};
    std::uniform_real_distribution<float> uni_phi{0.0f, 2.0f * float(M_PI)};
    float tauMag = 0.1f;
    float fSigma = 0.0f;
    glm::vec3 uniform_dir_s2();

    // ---- Diagnostics (last frame) ----
    size_t lastHitCount = 0;
    size_t lastIslandCount = 0;
    double lastSoftPotentialEnergy = 0.0;
    double keAfterIntegrate = 0.0;
    double keAfterWarmstart = 0.0;
    double keAfterSolve = 0.0;
    double keAfterPosCorrect = 0.0;
    double keAfterPBCWrap = 0.0;

    // ---- Broadphase scratch (reused) ----
    glm::ivec3 gridN{0};
    std::vector<uint32_t> gridCounts;
    std::vector<uint32_t> gridOffsets;
    std::vector<uint32_t> gridWrite;
    std::vector<int> gridItems;
    std::vector<std::vector<Hit>> thHitsScratch;
    std::vector<std::vector<int>> thSeenAt;
    std::vector<std::vector<int>> thCellSeenAt;
    std::vector<Hit> hitsScratch;

    // Warm-start cache
    std::unordered_map<uint64_t, AppliedImpulse> warmCache;
    std::vector<uint64_t> hitKeysScratch;

    // ---- Internal helpers ----
    void updateSleeping();
    void applyRandomForces();
    void integrateRods();
    void performCollisionResolution();
    void wrapPBC();
    
    static inline glm::ivec3 gridDims(const glm::vec3& bmin, const glm::vec3& bmax, float cs);
    static inline glm::ivec3 cellIndex(const glm::vec3& p, const glm::vec3& bmin, const glm::vec3& bmax, const glm::ivec3& n);
    static inline int64_t packKey(const glm::ivec3& i, const glm::ivec3& n);
    static inline size_t linearIndex(const glm::ivec3& i, const glm::ivec3& n);
    static inline uint64_t pairKey(int a, int b);
    static inline void wrapPos(glm::vec3& p, const glm::vec3& bmin, const glm::vec3& bmax);
};

} // namespace core
