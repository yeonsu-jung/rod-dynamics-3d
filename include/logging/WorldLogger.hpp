/**
 * @file WorldLogger.hpp
 * @brief Logging subsystem for World simulation data
 * 
 * Separates data serialization from simulation logic.
 * Supports multiple output streams: profile CSV, per-rod CSV, soft PE, contact dumps.
 */

#pragma once

#include <string>
#include <fstream>
#include <cstdint>
#include <vector>

// Forward declarations
namespace core { class World; }
struct Contact;

namespace logging {

/**
 * @brief Frame timing and diagnostics data
 */
struct FrameTimes {
    double integrate = 0.0;
    double sleepUpdate = 0.0;
    double broadphase = 0.0;
    double bpCount = 0.0;
    double bpPrefix = 0.0;
    double bpFill = 0.0;
    double bpPairs = 0.0;
    double bpLongLong = 0.0;
    double warmstart = 0.0;
    double buildIslands = 0.0;
    double solve = 0.0;
    double floorSolve = 0.0;
    double posCorrect = 0.0;
    double pbcWrap = 0.0;
    double render = 0.0;
};

/**
 * @brief Contact dump record for debugging contact dynamics
 */
struct ContactRecord {
    int bodyA;
    int bodyB;
    float normalX, normalY, normalZ;
    float penetration;
    float pointX, pointY, pointZ;
    
    ContactRecord(int a, int b, float nx, float ny, float nz, 
                  float pen, float px, float py, float pz)
        : bodyA(a), bodyB(b), normalX(nx), normalY(ny), normalZ(nz),
          penetration(pen), pointX(px), pointY(py), pointZ(pz) {}
};

/**
 * @brief Logger for World simulation data
 * 
 * Responsibilities:
 * - Profile CSV: frame-level metrics (timing, hit counts, energies)
 * - Per-rod CSV: detailed trajectory data (position, velocity, orientation per rod)
 * - Soft PE CSV: potential energy time series
 * - Contact dump CSV: diagnostic contact data when triggered
 * 
 * Usage:
 *   WorldLogger logger;
 *   logger.enableProfile("profile.csv");
 *   logger.enablePerRod("perrod.csv", 1000);
 *   // ... in simulation loop:
 *   logger.logFrame(world, times, impulseStats);
 */
class WorldLogger {
public:
    WorldLogger() = default;
    ~WorldLogger();

    // ---- Enable/disable specific outputs ----
    
    /**
     * @brief Enable profile CSV output
     * @param path Output file path
     */
    void enableProfile(const std::string& path);
    
    /**
     * @brief Enable per-rod trajectory CSV output
     * @param path Output file path
     * @param maxFrames Maximum number of frames to sample
     * @param skip Sample every N frames (computed automatically for headless runs)
     */
    void enablePerRod(const std::string& path, int maxFrames = 1000, int skip = 1);
    
    /**
     * @brief Enable soft contact potential energy CSV output
     * @param path Output file path
     */
    void enableSoftPE(const std::string& path);
    
    /**
     * @brief Enable contact dump diagnostics
     * @param path Output file path
     * @param keDeltaThreshold Absolute KE change threshold to trigger dump (Joules)
     * @param trigger 0=any change, +1=increase only, -1=decrease only
     */
    void enableContactDump(const std::string& path, double keDeltaThreshold = 0.0, int trigger = 0);
    
    /**
     * @brief Set per-rod sampling skip (computed from headless step count)
     * @param totalSteps Total number of simulation steps expected
     */
    void setPerRodSkipFromSteps(int totalSteps);
    
    // ---- Logging methods ----
    
    /**
     * @brief Log a complete frame of data
     * @param world World instance to query state from
     * @param times Frame timing diagnostics
     * @param entanglementPairs Number of rod pairs analyzed for entanglement
     * @param entanglementSum Sum of absolute linking numbers
     */
    void logFrame(const core::World& world, const FrameTimes& times, 
                  long long entanglementPairs, double entanglementSum);
    
    /**
     * @brief Log contacts for current frame (if dump enabled and triggered)
     * @param frameIndex Current frame number
     * @param contacts Vector of contact hits
     * @param stageLabel Stage identifier (e.g., "pre-solve", "post-solve")
     * @param keDelta Change in kinetic energy since last frame
     */
    void logContacts(uint64_t frameIndex, const std::vector<ContactRecord>& contacts,
                     const char* stageLabel, double keDelta);
    
    /**
     * @brief Flush all output streams
     */
    void flush();
    
    /**
     * @brief Close all output streams
     */
    void close();

private:
    // Profile CSV
    bool profileEnabled = false;
    std::ofstream profileStream;
    std::string profilePath;
    bool profileHeaderWritten = false;
    
    // Per-rod CSV
    bool perRodEnabled = false;
    std::ofstream perRodStream;
    std::string perRodPath;
    bool perRodHeaderWritten = false;
    int perRodMaxFrames = 1000;
    int perRodSkip = 1;
    int perRodWrittenFrames = 0;
    
    // Soft PE CSV
    bool softPEEnabled = false;
    std::ofstream softPEStream;
    std::string softPEPath;
    bool softPEHeaderWritten = false;
    
    // Contact dump CSV
    bool contactDumpEnabled = false;
    std::ofstream contactDumpStream;
    std::string contactDumpPath;
    bool contactDumpHeaderWritten = false;
    double contactDumpThreshold = 0.0;
    int contactDumpTrigger = 0;  // 0=any, +1=up, -1=down
    
    // Internal helpers
    void writeProfileHeader();
    void writePerRodHeader();
    void writeSoftPEHeader();
    void writeContactDumpHeader();
    
    bool shouldTriggerContactDump(double keDelta) const;
};

} // namespace logging
