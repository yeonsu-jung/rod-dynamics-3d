/**
 * @file camera.hpp
 * @brief Orbit camera for 3D viewing
 */

#pragma once
#include <glm/glm.hpp>

/**
 * @brief Orbit camera that rotates around a target point
 */
struct OrbitCamera {
    float yaw = 0.6f;     ///< Horizontal rotation (radians)
    float pitch = 0.35f;  ///< Vertical rotation (radians)
    float dist = 6.0f;    ///< Distance from target

    /**
     * @brief Get view matrix
     * @param target Target point to orbit around
     * @return View matrix for rendering
     */
    glm::mat4 view(const glm::vec3& target = glm::vec3(0)) const;

    /**
     * @brief Get camera eye position
     * @param target Target point to orbit around
     * @return Camera position in world space
     */
    glm::vec3 eye(const glm::vec3& target = glm::vec3(0)) const;
};
