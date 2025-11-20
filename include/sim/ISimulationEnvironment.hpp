/**
 * @file ISimulationEnvironment.hpp
 * @brief Abstract interface for simulation environments
 * 
 * Inspired by dismech's derSimulationEnvironment pattern.
 * Separates run loop orchestration from simulation logic.
 */

#pragma once

namespace sim {

/**
 * @brief Abstract simulation environment interface
 * 
 * Implementations:
 * - HeadlessEnvironment: Deterministic run with optional logging
 * - OpenGLEnvironment: Interactive visualization with input handling
 * - TestEnvironment: Minimal harness for unit/integration tests
 * 
 * Each environment owns:
 * - Run loop control (step count, timing, exit conditions)
 * - User interaction (if applicable)
 * - Output/logging configuration
 * 
 * Each environment uses but does not own:
 * - World (simulation state and stepping)
 * - Logger (optional output)
 */
class ISimulationEnvironment {
public:
    virtual ~ISimulationEnvironment() = default;
    
    /**
     * @brief Execute the simulation
     * @return Exit code (0 = success)
     */
    virtual int run() = 0;
};

} // namespace sim
