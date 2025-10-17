/**
 * @file shape.hpp
 * @brief Shape definitions for collision detection
 */

#pragma once

/**
 * @brief Box shape defined by half-extents
 */
struct Box { 
    float hx{10.0f};  ///< Half-extent in X direction
    float hy{0.1f};   ///< Half-extent in Y direction  
    float hz{10.0f};  ///< Half-extent in Z direction
};

/**
 * @brief Capsule shape (cylinder with hemispherical ends)
 */
struct Capsule { 
    float r{0.05f};   ///< Radius
    float h{0.5f};    ///< Half-length of cylindrical section
};

/**
 * @brief Available shape types for rigid bodies
 */
enum class ShapeType { 
    Box,      ///< Box/cuboid shape
    Capsule   ///< Capsule/rod shape
};
