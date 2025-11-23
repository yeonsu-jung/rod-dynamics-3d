/**
 * @file debug_segseg.cpp
 * @brief Debug segment-segment distance calculation
 */

#include <iostream>
#include <glm/glm.hpp>
#include <cmath>

float segmentSegmentDistance(const glm::vec3& p0, const glm::vec3& p1,
                              const glm::vec3& q0, const glm::vec3& q1) {
    std::cout << "  p0: (" << p0.x << ", " << p0.y << ", " << p0.z << ")\n";
    std::cout << "  p1: (" << p1.x << ", " << p1.y << ", " << p1.z << ")\n";
    std::cout << "  q0: (" << q0.x << ", " << q0.y << ", " << q0.z << ")\n";
    std::cout << "  q1: (" << q1.x << ", " << q1.y << ", " << q1.z << ")\n";
    
    glm::vec3 u = p1 - p0;
    glm::vec3 v = q1 - q0;
    glm::vec3 w0 = p0 - q0;
    
    std::cout << "  u: (" << u.x << ", " << u.y << ", " << u.z << ")\n";
    std::cout << "  v: (" << v.x << ", " << v.y << ", " << v.z << ")\n";
    std::cout << "  w0: (" << w0.x << ", " << w0.y << ", " << w0.z << ")\n";
    
    float uu = glm::dot(u, u);
    float vv = glm::dot(v, v);
    float uv = glm::dot(u, v);
    float wu = glm::dot(w0, u);
    float wv = glm::dot(w0, v);
    float D = uu * vv - uv * uv;
    
    std::cout << "  uu: " << uu << ", vv: " << vv << ", uv: " << uv << "\n";
    std::cout << "  wu: " << wu << ", wv: " << wv << ", D: " << D << "\n";
    
    float s, t;
    const float eps = 1e-12f;
    
    if (std::abs(D) < eps) {
        // Parallel segments
        std::cout << "  Segments are parallel\n";
        s = 0.0f;
        t = (vv >= eps) ? (-wv / vv) : 0.0f;
    } else {
        s = (uv * wv - vv * wu) / D;
        t = (uu * wv - uv * wu) / D;
        std::cout << "  Initial s: " << s << ", t: " << t << "\n";
    }
    
    s = glm::clamp(s, 0.0f, 1.0f);
    std::cout << "  After first clamp s: " << s << "\n";
    
    t = (s * uv + wv) / (vv >= eps ? vv : 1.0f);
    std::cout << "  Recomputed t: " << t << "\n";
    t = glm::clamp(t, 0.0f, 1.0f);
    std::cout << "  After clamp t: " << t << "\n";
    
    float su = (-wu + t * uv) / (uu >= eps ? uu : 1.0f);
    std::cout << "  Recomputed su: " << su << "\n";
    std::cout << "  t check: t > 1e-6f = " << (t > 1e-6f) << ", t < 1.0f-1e-6f = " << (t < 1.0f-1e-6f) << "\n";
    
    if (!(t > 1e-6f && t < 1.0f - 1e-6f)) {
        std::cout << "  Adjusting s based on su\n";
        if (su < 0.0f) s = 0.0f;
        else if (su > 1.0f) s = 1.0f;
        else s = su;
        std::cout << "  Final adjusted s: " << s << "\n";
    }
    
    glm::vec3 cp = p0 + s * u;
    glm::vec3 cq = q0 + t * v;
    std::cout << "  Closest point on p: (" << cp.x << ", " << cp.y << ", " << cp.z << ")\n";
    std::cout << "  Closest point on q: (" << cq.x << ", " << cq.y << ", " << cq.z << ")\n";
    
    glm::vec3 d = (w0 + s * u) - t * v;
    std::cout << "  Distance vector d: (" << d.x << ", " << d.y << ", " << d.z << ")\n";
    float dist = glm::length(d);
    std::cout << "  Distance: " << dist << "\n";
    
    return dist;
}

int main() {
    std::cout << "Test: Perpendicular capsules\n";
    std::cout << "Cap1: center (0,0,0), axis (1,0,0), halfLen 0.25, radius 0.05\n";
    std::cout << "Cap2: center (0,0.15,0), axis (0,1,0), halfLen 0.25, radius 0.05\n\n";
    
    glm::vec3 a0(-0.25f, 0, 0);
    glm::vec3 a1(0.25f, 0, 0);
    glm::vec3 b0(0, -0.25f + 0.15f, 0);
    glm::vec3 b1(0, 0.25f + 0.15f, 0);
    
    float dist = segmentSegmentDistance(a0, a1, b0, b1);
    
    std::cout << "\nMinimum required separation: 0.1 (2*radius)\n";
    std::cout << "Actual separation: " << dist << "\n";
    std::cout << "Should overlap: " << (dist < 0.1f ? "YES" : "NO") << "\n";
    
    return 0;
}
