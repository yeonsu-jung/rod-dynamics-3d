#include "sim/EntanglementObserver.hpp"
#include "core/World.hpp"
#include "physics/rigid_body.hpp"
#include <cstdio>
#include <vector>
#include <iostream>

// External entanglement computation function
extern "C" double pairwise_abs_linking_sum_with_cutoff(
    const double* segments1, int n1,
    const double* segments2, int n2,
    double cutoff
);

namespace sim {

EntanglementObserver::EntanglementObserver(const std::string& outputPath, 
                                           int frequency, 
                                           double cutoffDistance)
    : outputPath_(outputPath)
    , frequency_(frequency)
    , cutoffDistance_(cutoffDistance)
    , enabled_(frequency > 0)
    , headerWritten_(false)
    , fileHandle_(nullptr)
{
}

EntanglementObserver::~EntanglementObserver() {
    if (fileHandle_) {
        fclose(static_cast<FILE*>(fileHandle_));
    }
}

bool EntanglementObserver::shouldObserve(int frameIdx) const {
    return enabled_ && frequency_ > 0 && (frameIdx % frequency_ == 0);
}

void EntanglementObserver::onFrame(const core::World& world, int frameIdx, double time) {
    if (!enabled_ || frequency_ <= 0) {
        return;
    }

    if (!fileHandle_) {
        fileHandle_ = fopen(outputPath_.c_str(), "w");
        if (!fileHandle_) {
            std::cerr << "Warning: Could not open entanglement output file: " 
                      << outputPath_ << std::endl;
            enabled_ = false;
            return;
        }
    }

    if (!headerWritten_) {
        writeHeader();
        headerWritten_ = true;
    }

    computeAndLogEntanglement(world, frameIdx, time);
}

void EntanglementObserver::writeHeader() {
    FILE* fp = static_cast<FILE*>(fileHandle_);
    fprintf(fp, "frame,time,total_entanglement\n");
    fflush(fp);
}

void EntanglementObserver::computeAndLogEntanglement(const core::World& world, 
                                                     int frameIdx, 
                                                     double time) {
    const auto& rods = world.getRods();
    const int nRods = static_cast<int>(rods.size());

    if (nRods < 2) {
        FILE* fp = static_cast<FILE*>(fileHandle_);
        fprintf(fp, "%d,%.6f,0.0\n", frameIdx, time);
        fflush(fp);
        return;
    }

    // Build segment arrays for each rod (each rod is a single capsule)
    std::vector<std::vector<double>> rodSegments(nRods);
    for (int i = 0; i < nRods; ++i) {
        const auto& rod = rods[i];
        
        // Capsule axis along y direction
        glm::vec3 u = rod.axisY();
        glm::vec3 a = rod.x - u * rod.cap.h;
        glm::vec3 b = rod.x + u * rod.cap.h;
        
        // Single segment: x0,y0,z0,x1,y1,z1
        rodSegments[i].resize(6);
        rodSegments[i][0] = a.x;
        rodSegments[i][1] = a.y;
        rodSegments[i][2] = a.z;
        rodSegments[i][3] = b.x;
        rodSegments[i][4] = b.y;
        rodSegments[i][5] = b.z;
    }

    // Compute pairwise entanglement
    double totalEntanglement = 0.0;
    for (int i = 0; i < nRods; ++i) {
        for (int j = i + 1; j < nRods; ++j) {
            // Each rod is a single capsule segment
            const int nSeg_i = 1;
            const int nSeg_j = 1;
            
            double linking = pairwise_abs_linking_sum_with_cutoff(
                rodSegments[i].data(), nSeg_i,
                rodSegments[j].data(), nSeg_j,
                cutoffDistance_
            );
            
            totalEntanglement += std::abs(linking);
        }
    }

    FILE* fp = static_cast<FILE*>(fileHandle_);
    fprintf(fp, "%d,%.6f,%.6f\n", frameIdx, time, totalEntanglement);
    fflush(fp);
}

} // namespace sim
