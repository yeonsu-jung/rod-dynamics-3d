#pragma once

#include "IFrameObserver.hpp"
#include <string>

namespace sim {

/**
 * @brief Observer that computes and logs rod entanglement.
 * 
 * Computes the sum of pairwise linking numbers between all rod pairs
 * using a cutoff distance to avoid spurious linkages from distant rods.
 */
class EntanglementObserver : public IFrameObserver {
public:
    /**
     * @brief Construct an entanglement observer.
     * @param outputPath Path to CSV file for entanglement output
     * @param frequency Compute entanglement every N frames (0 = disabled)
     * @param cutoffDistance Distance cutoff for pairwise linking computation
     */
    EntanglementObserver(const std::string& outputPath, 
                         int frequency = 0, 
                         double cutoffDistance = 5.0);

    ~EntanglementObserver() override;

    void onFrame(const core::World& world, int frameIdx, double time) override;
    bool shouldObserve(int frameIdx) const override;

    void setFrequency(int freq) { frequency_ = freq; }
    void setCutoffDistance(double cutoff) { cutoffDistance_ = cutoff; }
    void setEnabled(bool enabled) { enabled_ = enabled; }

private:
    void computeAndLogEntanglement(const core::World& world, int frameIdx, double time);
    void writeHeader();

    std::string outputPath_;
    int frequency_;
    double cutoffDistance_;
    bool enabled_;
    bool headerWritten_;
    void* fileHandle_; // FILE* to avoid including <cstdio> in header
};

} // namespace sim
