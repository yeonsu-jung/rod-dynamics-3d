/**
 * @file WorldLogger.cpp
 * @brief Implementation of WorldLogger
 */

#include "logging/WorldLogger.hpp"
#include "core/World.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

// External globals for impulse diagnostics (will be refactored in later PRs)
extern double g_diag_jn_sum;
extern double g_diag_jt_sum;
extern int g_diag_impulse_count;

namespace logging {

WorldLogger::~WorldLogger() {
    close();
}

void WorldLogger::enableProfile(const std::string& path) {
    profilePath = path.empty() ? "profile.csv" : path;
    profileStream.open(profilePath, std::ios::out | std::ios::trunc);
    if (!profileStream) {
        std::cerr << "[WorldLogger] Failed to open profile CSV: " << profilePath << "\n";
        return;
    }
    profileEnabled = true;
    profileHeaderWritten = false;
    std::cout << "[WorldLogger] Profile CSV enabled: " << profilePath << "\n";
}

void WorldLogger::enablePerRod(const std::string& path, int maxFrames, int skip) {
    perRodPath = path.empty() ? "perrod.csv" : path;
    perRodStream.open(perRodPath, std::ios::out | std::ios::trunc);
    if (!perRodStream) {
        std::cerr << "[WorldLogger] Failed to open per-rod CSV: " << perRodPath << "\n";
        return;
    }
    perRodEnabled = true;
    perRodHeaderWritten = false;
    perRodMaxFrames = std::max(1, maxFrames);
    perRodSkip = std::max(1, skip);
    perRodWrittenFrames = 0;
    std::cout << "[WorldLogger] Per-rod CSV enabled: " << perRodPath 
              << " (max_frames=" << perRodMaxFrames << ", skip=" << perRodSkip << ")\n";
}

void WorldLogger::enableSoftPE(const std::string& path) {
    softPEPath = path.empty() ? "soft_pe.csv" : path;
    softPEStream.open(softPEPath, std::ios::out | std::ios::trunc);
    if (!softPEStream) {
        std::cerr << "[WorldLogger] Failed to open soft PE CSV: " << softPEPath << "\n";
        return;
    }
    softPEEnabled = true;
    softPEHeaderWritten = false;
    std::cout << "[WorldLogger] Soft PE CSV enabled: " << softPEPath << "\n";
}

void WorldLogger::enableContactDump(const std::string& path, double keDeltaThreshold, int trigger) {
    contactDumpPath = path;
    contactDumpThreshold = keDeltaThreshold;
    contactDumpTrigger = trigger;
    contactDumpEnabled = true;
    std::cout << "[WorldLogger] Contact dump enabled: " << path 
              << " (threshold=" << keDeltaThreshold << ", trigger=" << trigger << ")\n";
}

void WorldLogger::setPerRodSkipFromSteps(int totalSteps) {
    if (perRodEnabled && totalSteps > 0 && perRodMaxFrames > 0) {
        perRodSkip = std::max(1, totalSteps / perRodMaxFrames);
        std::cout << "[WorldLogger] Per-rod skip computed: " << perRodSkip 
                  << " (totalSteps=" << totalSteps << ")\n";
    }
}

void WorldLogger::writeProfileHeader() {
    profileStream << "frame,num_rods,"
                  << "t_integrate,t_sleep,t_broadphase,"
                  << "t_bp_count,t_bp_prefix,t_bp_fill,t_bp_pairs,t_bp_longlong,"
                  << "t_warmstart,t_islands,t_solve,t_floor,t_posCorrect,t_pbcWrap,t_render,"
                  << "hits,islands,"
                  << "KE_total,KE_afterIntegrate,KE_afterWarmstart,KE_afterSolve,KE_afterPosCorrect,KE_afterPBCWrap,"
                  << "soft_PE,"
                  << "impulse_jn_sum,impulse_jt_sum,impulse_count,"
                  << "ent_pairs,ent_sum\n";
    profileHeaderWritten = true;
}

void WorldLogger::writePerRodHeader() {
    perRodStream << "frame,rod,px,py,pz,vx,vy,vz,wx,wy,wz,qw,qx,qy,qz,"
                 << "KE_lin,KE_rot,KE_total\n";
    perRodHeaderWritten = true;
}

void WorldLogger::writeSoftPEHeader() {
    softPEStream << "frame,soft_PE\n";
    softPEHeaderWritten = true;
}

void WorldLogger::writeContactDumpHeader() {
    contactDumpStream << "frame,stage,contact_idx,bodyA,bodyB,"
                      << "normal_x,normal_y,normal_z,penetration,"
                      << "point_x,point_y,point_z\n";
    contactDumpHeaderWritten = true;
}

void WorldLogger::logFrame(const core::World& world, const FrameTimes& times,
                            long long entanglementPairs, double entanglementSum) {
    uint64_t frameIndex = world.getFrameIndex();
    
    // Profile CSV
    if (profileEnabled && profileStream) {
        if (!profileHeaderWritten) writeProfileHeader();
        
        profileStream 
            << frameIndex << ',' << world.numRods() << ','
            << times.integrate << ',' << times.sleepUpdate << ',' << times.broadphase << ','
            << times.bpCount << ',' << times.bpPrefix << ',' << times.bpFill << ',' 
            << times.bpPairs << ',' << times.bpLongLong << ','
            << times.warmstart << ',' << times.buildIslands << ',' << times.solve << ','
            << times.floorSolve << ',' << times.posCorrect << ',' << times.pbcWrap << ',' 
            << times.render << ','
            << world.getLastHitCount() << ',' << world.getLastIslandCount() << ','
            << world.totalKE() << ','
            << world.getKEAfterIntegrate() << ',' << world.getKEAfterWarmstart() << ','
            << world.getKEAfterSolve() << ',' << world.getKEAfterPosCorrect() << ','
            << world.getKEAfterPBCWrap() << ','
            << world.getLastSoftPE() << ','
            << g_diag_jn_sum << ',' << g_diag_jt_sum << ',' << g_diag_impulse_count << ','
            << entanglementPairs << ',' << entanglementSum << '\n';
        
        // Periodic flush
        if ((frameIndex & 0x3F) == 0) profileStream.flush();
    }
    
    // Per-rod CSV
    if (perRodEnabled && perRodStream) {
        if (!perRodHeaderWritten) writePerRodHeader();
        if (perRodWrittenFrames < perRodMaxFrames && (frameIndex % perRodSkip) == 0) {
            const auto& rods = world.getRods();
            for (size_t i = 0; i < rods.size(); ++i) {
                const auto& rb = rods[i];
                double m = rb.mass;
                double vSq = glm::dot(rb.v, rb.v);
                double KElin = 0.5 * m * vSq;
                glm::vec3 Llocal = rb.I_body * rb.w;
                double KErot = 0.5 * glm::dot(rb.w, Llocal);
                double KEtot = KElin + KErot;
                
                perRodStream 
                    << frameIndex << ',' << i << ','
                    << rb.x.x << ',' << rb.x.y << ',' << rb.x.z << ','
                    << rb.v.x << ',' << rb.v.y << ',' << rb.v.z << ','
                    << rb.w.x << ',' << rb.w.y << ',' << rb.w.z << ','
                    << rb.q.w << ',' << rb.q.x << ',' << rb.q.y << ',' << rb.q.z << ','
                    << KElin << ',' << KErot << ',' << KEtot << '\n';
            }
            ++perRodWrittenFrames;
            if ((frameIndex & 0x3F) == 0) perRodStream.flush();
        }
    }
    
    // Soft PE CSV
    if (softPEEnabled && softPEStream) {
        if (!softPEHeaderWritten) writeSoftPEHeader();
        softPEStream << frameIndex << ',' << world.getLastSoftPE() << '\n';
        if ((frameIndex & 0x3F) == 0) softPEStream.flush();
    }
}

bool WorldLogger::shouldTriggerContactDump(double keDelta) const {
    if (!contactDumpEnabled) return false;
    if (std::abs(keDelta) < contactDumpThreshold) return false;
    
    if (contactDumpTrigger == 0) return true;  // any change
    if (contactDumpTrigger > 0 && keDelta > 0) return true;  // increase only
    if (contactDumpTrigger < 0 && keDelta < 0) return true;  // decrease only
    
    return false;
}

void WorldLogger::logContacts(uint64_t frameIndex, const std::vector<ContactRecord>& contacts,
                               const char* stageLabel, double keDelta) {
    if (!shouldTriggerContactDump(keDelta)) return;
    
    // Lazy open
    if (!contactDumpStream.is_open()) {
        contactDumpStream.open(contactDumpPath, std::ios::out | std::ios::trunc);
        if (!contactDumpStream) {
            std::cerr << "[WorldLogger] Failed to open contact dump: " << contactDumpPath << "\n";
            contactDumpEnabled = false;
            return;
        }
    }
    
    if (!contactDumpHeaderWritten) writeContactDumpHeader();
    
    for (size_t i = 0; i < contacts.size(); ++i) {
        const auto& c = contacts[i];
        contactDumpStream 
            << frameIndex << ',' << stageLabel << ',' << i << ','
            << c.bodyA << ',' << c.bodyB << ','
            << c.normalX << ',' << c.normalY << ',' << c.normalZ << ','
            << c.penetration << ','
            << c.pointX << ',' << c.pointY << ',' << c.pointZ << '\n';
    }
    
    if ((frameIndex & 0x3F) == 0) contactDumpStream.flush();
}

void WorldLogger::flush() {
    if (profileStream) profileStream.flush();
    if (perRodStream) perRodStream.flush();
    if (softPEStream) softPEStream.flush();
    if (contactDumpStream) contactDumpStream.flush();
}

void WorldLogger::close() {
    if (profileStream) { profileStream.close(); profileEnabled = false; }
    if (perRodStream) { perRodStream.close(); perRodEnabled = false; }
    if (softPEStream) { softPEStream.close(); softPEEnabled = false; }
    if (contactDumpStream) { contactDumpStream.close(); contactDumpEnabled = false; }
}

} // namespace logging
