/**
 * @file HeadlessEnvironment.cpp
 * @brief Implementation of HeadlessEnvironment with observer pattern
 */

#include "sim/HeadlessEnvironment.hpp"
#include "sim/IFrameObserver.hpp"
#include "core/World.hpp"
#include "logging/WorldLogger.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>

namespace sim {

HeadlessEnvironment::HeadlessEnvironment(core::World& world, 
                                         logging::WorldLogger* logger, 
                                         int steps, 
                                         int substeps)
    : world_(world)
    , logger_(logger)
    , steps_(steps)
    , substeps_(substeps)
{
}

void HeadlessEnvironment::addObserver(IFrameObserver* observer) {
    if (observer) {
        observers_.push_back(observer);
    }
}

void HeadlessEnvironment::notifyObservers(int frameIdx, double time) {
    for (auto* observer : observers_) {
        if (observer && observer->shouldObserve(frameIdx)) {
            observer->onFrame(world_, frameIdx, time);
        }
    }
}

void HeadlessEnvironment::printCliStatus(const std::string& prefix) const {
    std::cout << prefix
              << "frame=" << world_.getFrameIndex()
              << " rods=" << world_.numRods()
              << " KE=" << std::fixed << std::setprecision(6) << world_.totalKE()
              << std::defaultfloat << "\n";
}

int HeadlessEnvironment::run() {
    std::cout << "[HeadlessEnvironment] Starting simulation: " << steps_ << " steps";
    if (substeps_ > 1) std::cout << " (" << substeps_ << " substeps/frame)";
    std::cout << "\n";
    
    auto tStart = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < steps_; ++i) {
        // Step the world
        if (substeps_ > 1) {
            world_.stepWithSubsteps(substeps_);
        } else {
            world_.step();
        }
        
        const double currentTime = world_.getFrameIndex() * world_.getDt();
        
        // Notify all observers
        notifyObservers(world_.getFrameIndex(), currentTime);
        
        // Log frame if logger provided
        if (logger_) {
            logging::FrameTimes times{};  // For now, empty times
            logger_->logFrame(world_, times, 0, 0.0);  // Entanglement moved to observer
        }
        
        // Print CLI status periodically
        if (cliStatusEnabled_ && (i % 100 == 0 || i == steps_ - 1)) {
            printCliStatus("[Progress] ");
        }
    }
    
    auto tEnd = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(tEnd - tStart).count();
    
    std::cout << "[HeadlessEnvironment] Completed " << steps_ << " steps in " 
              << std::fixed << std::setprecision(3) << elapsed << " s\n";
    std::cout << "[HeadlessEnvironment] Final KE: " 
              << std::setprecision(6) << world_.totalKE() << " J\n";
    
    if (logger_) {
        logger_->flush();
    }
    
    return 0;
}

} // namespace sim

