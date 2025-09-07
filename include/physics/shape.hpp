#pragma once
struct Box     { float hx{10.f}, hy{0.1f}, hz{10.f}; };
struct Capsule { float r{0.05f}, h{0.5f}; }; // r = radius, h = half-length
enum class ShapeType { Box, Capsule };
