/**
 * @file types.hpp
 * @brief Common physics types and utilities
 */

#pragma once
#include <glm/glm.hpp>

/**
 * @brief Utility function to compute squared length of a vector
 * @param v Input vector
 * @return Squared length
 */
inline float length2(const glm::vec3& v) { 
    return glm::dot(v, v); 
}

/**
 * @brief Contact information between two rigid bodies
 */
struct Contact {
    bool hit{false};              ///< Whether a collision occurred
    glm::vec3 normal{0};          ///< Contact normal (from A to B)
    float penetration{0};         ///< Penetration depth
    glm::vec3 point{0};           ///< Contact point in world space
    glm::vec3 shiftB{0};          ///< Periodic image shift applied to body B when evaluating this contact
};

/**
 * @brief Configuration for constraint solver
 */
struct SolverConfig {
    float baumgarte = 0.25f;      ///< Baumgarte stabilization parameter
    float allowedPen = 0.003f;    ///< Allowed penetration before correction
    int velIters = 30;            ///< Number of velocity solver iterations
    bool splitImpulse = false;    ///< If true, positional correction uses split impulse (no direct velocity change)
    bool splitOrient = true;      ///< If false and splitImpulse=true, apply translation-only correction (no orientation tweak)
    int ngsNormalSweeps = 0;      ///< Extra normal-only GS sweeps for high-speed impacts
    float ngsHighVThresh = 0.5f;  ///< Threshold on approaching normal speed to include in NGS (m/s)
};

// Physics tuning constants (deprecated - use global variables instead)
constexpr float LIN_DAMP = 0.08f;  // s^-1
constexpr float ANG_DAMP = 0.12f;  // s^-1  
constexpr float W_MAX = 80.0f;     // rad/s
