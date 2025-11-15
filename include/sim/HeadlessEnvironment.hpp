/**
 * @file HeadlessEnvironment.hpp
 * @brief Headless simulation environment for deterministic runs
 */

#pragma once

#include "sim/ISimulationEnvironment.hpp"
#include <memory>
#include <string>
#include <vector>

// Forward declarations
namespace core { class World; }
namespace logging { class WorldLogger; }

namespace sim {

class IFrameObserver;

/**
 * @brief Headless simulation environment
 * 
 * Runs a fixed number of steps without rendering.
 * Optionally logs data via WorldLogger.
 * Used for:
 * - Batch parametric studies
 * - Regression tests
 * - Long production runs
 * - CI/CD validation
 */
class HeadlessEnvironment : public ISimulationEnvironment {
public:
    /**
     * @brief Construct headless environment
     * @param world World instance to simulate (non-owning reference)
     * @param logger Optional logger for output (non-owning pointer, can be nullptr)
     * @param steps Number of simulation steps to execute
     * @param substeps Number of substeps per frame (if >1, calls World::stepWithSubsteps)
     */
    HeadlessEnvironment(core::World& world, logging::WorldLogger* logger, int steps, int substeps = 1);
    
    ~HeadlessEnvironment() override = default;
    
    /**
     * @brief Run headless simulation
     * @return 0 on success
     */
    int run() override;
    
    /**
     * @brief Add an observer to be notified on frames
     * @param observer Observer to register (non-owning pointer)
     */
    void addObserver(IFrameObserver* observer);
    
    /**
     * @brief Enable CLI status printing
     * @param enabled Print unified status line each frame
     */
    void setCliStatus(bool enabled) { cliStatusEnabled_ = enabled; }
    
private:
    core::World& world_;
    logging::WorldLogger* logger_;
    int steps_;
    int substeps_;
    
    // Observer pattern
    std::vector<IFrameObserver*> observers_;
    
    // CLI output
    bool cliStatusEnabled_ = false;
    
    void notifyObservers(int frameIdx, double time);
    void printCliStatus(const std::string& prefix = "") const;
};

} // namespace sim
