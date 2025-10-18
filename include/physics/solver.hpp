#pragma once
#include <vector>
#include "rigid_body.hpp"
#include "types.hpp"

struct AppliedImpulse {
    float jn = 0.0f;         // normal impulse magnitude
    float jt = 0.0f;         // friction impulse magnitude (signed along tangent)
    glm::vec3 tangent{0.0f}; // tangent direction used
};

// Lightweight pair-contact used by solver utilities
struct PairContact { int a = -1; int b = -1; Contact c; };

void applyImpulse(RigidBody& A, RigidBody& B, const Contact& c, AppliedImpulse* out = nullptr);
// Apply only the normal component (no friction). Useful for targeted high-speed sweeps.
void applyNormalImpulse(RigidBody& A, RigidBody& B, const Contact& c, AppliedImpulse* out = nullptr);
void positionalCorrection(RigidBody& A, RigidBody& B, const Contact& c, const SolverConfig& cfg);

// Apply a precomputed impulse (warm start): J = jn * n + jt * t
void applyWarmStart(RigidBody& A, RigidBody& B, const Contact& c, float jn, float jt, const glm::vec3& tangent);

// Diagnostic accumulators (summed magnitudes per physics step)
extern double g_diag_jn_sum; // sum of |normal impulse| applied during current frame
extern double g_diag_jt_sum; // sum of |tangent impulse| applied during current frame
extern int    g_diag_impulse_count; // number of impulses applied

// Reset per-frame diagnostic accumulators (call before solve)
void resetFrameImpulseAccumulators();

// Energy safeguard: when enabled, applied impulses are scaled down to avoid increasing total KE due to numerical issues
extern bool g_energy_safeguard;
void setEnergySafeguard(bool enabled);

// Enable/disable warm-start globally
void setWarmstartEnabled(bool enabled);

// Targeted restitution-accurate sweeps on high-speed impacts (normal-only)
void ngsRestitutionSweeps(std::vector<PairContact>& contacts, std::vector<RigidBody>& rods, const SolverConfig& cfg);
