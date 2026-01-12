#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <glm/glm.hpp>

// Constants
const float PI = 3.14159265358979323846f;

struct Rod {
    int id;
    glm::vec3 p1;
    glm::vec3 p2;
};

struct FrameData {
    int frame;
    std::vector<Rod> rods;
};

// Event for angular sweep
struct Event {
    float angle;
    int change; // +1 or -1

    bool operator<(const Event& other) const {
        return angle < other.angle;
    }
};

// Helper: split string
std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Compute min depth (crossing number) for a specific rod against all others
int compute_rod_crossing_number(const Rod& rod, const std::vector<Rod>& all_rods) {
    if (all_rods.size() <= 1) return 0;

    glm::vec3 axis = rod.p2 - rod.p1;
    float len = glm::length(axis);
    if (len < 1e-6f) return 0;
    axis /= len;

    // Basis (u, v) orthogonal to axis
    glm::vec3 ref = (std::abs(axis.x) < 0.9f) ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
    glm::vec3 u = glm::normalize(glm::cross(axis, ref));
    glm::vec3 v = glm::normalize(glm::cross(axis, u));

    std::vector<Event> events;
    events.reserve(all_rods.size() * 2);

    for (const auto& other : all_rods) {
        if (other.id == rod.id) continue;

        // Relative positions projected onto (u, v) plane
        glm::vec3 ps = other.p1 - rod.p1;
        glm::vec3 pe = other.p2 - rod.p1;

        float us = glm::dot(ps, u);
        float vs = glm::dot(ps, v);
        float ue = glm::dot(pe, u);
        float ve = glm::dot(pe, v);

        float ang1 = std::atan2(vs, us);
        float ang2 = std::atan2(ve, ue);

        // Normalize order
        float a_start = ang1;
        float a_end = ang2;

        // Ensure we take the shorter arc (assuming no collision with origin)
        // Logic replicated from python script analysis
        if (a_start > a_end) std::swap(a_start, a_end);
        
        float diff = a_end - a_start;
        
        if (diff < PI) {
            events.push_back({a_start, 1});
            events.push_back({a_end, -1});
        } else {
            // Wrap around cases
            events.push_back({a_end, 1});
            events.push_back({PI, -1});
            
            events.push_back({-PI, 1});
            events.push_back({a_start, -1});
        }
    }

    if (events.empty()) return 0;

    std::sort(events.begin(), events.end());

    int min_depth = std::numeric_limits<int>::max();
    int current_depth = 0;
    float last_angle = -PI;

    for (const auto& e : events) {
        if (e.angle > last_angle + 1e-7f) {
             if (current_depth < min_depth) min_depth = current_depth;
        }
        current_depth += e.change;
        last_angle = e.angle;
    }
    
    // Check end of interval
    if (PI > last_angle + 1e-7f) {
         if (current_depth < min_depth) min_depth = current_depth;
    }
    
    // Safety for initial state if starts inside
    // Actually the logic above for wrapping handles circularity if we consider the full circle.
    // However, classical sweep line on circle usually needs to set initial depth.
    // But here we split wrapping intervals into [-PI, a] and [b, PI].
    // So current_depth starts at 0 (or whatever accumulated from -PI if we had events exactly at -PI).
    // The splitting logic implies we start "fresh" from -PI with whatever is covering -PI.
    // Wait, if an interval wraps, we added {-PI, 1}. So at -PI, depth becomes 1.
    // So initial depth 0 is correct if we process events at -PI first.
    
    // One edge case: if min_depth is still huge (no gaps?), it is at least 0.
    if (min_depth == std::numeric_limits<int>::max()) return 0;

    return min_depth;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <endpoints.csv> <output.csv>" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];

    std::ifstream infile(input_path);
    if (!infile.is_open()) {
        std::cerr << "Error opening " << input_path << std::endl;
        return 1;
    }

    std::ofstream outfile(output_path);
    if (!outfile.is_open()) {
        std::cerr << "Error opening " << output_path << std::endl;
        return 1;
    }

    // Write header
    outfile << "frame,min_crossing\n";

    std::string line;
    std::vector<Rod> current_rods;
    int current_frame = -1;
    
    // Skip header
    if (infile.good()) {
        std::getline(infile, line);
    }

    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        
        std::vector<std::string> tokens = split(line, ',');
        if (tokens.size() < 8) continue;

        try {
            int f = std::stoi(tokens[0]);
            int r_id = std::stoi(tokens[1]);
            float x1 = std::stof(tokens[2]);
            float y1 = std::stof(tokens[3]);
            float z1 = std::stof(tokens[4]);
            float x2 = std::stof(tokens[5]);
            float y2 = std::stof(tokens[6]);
            float z2 = std::stof(tokens[7]);

            if (f != current_frame) {
                if (current_frame != -1) {
                    // Process previous frame
                    if (!current_rods.empty()) {
                         int packing_min = std::numeric_limits<int>::max();
                         
                         // Simple optimization: if we find a 0, we can stop?
                         // Yes, min crossing cannot be less than 0. 
                         // But we want the exact min. 
                         // If any rod has 0, the min is 0.
                         
                         // To speed up: parallelize?
                         // For now, serial is fine, C++ is fast.
                         for (const auto& r : current_rods) {
                             int mc = compute_rod_crossing_number(r, current_rods);
                             if (mc < packing_min) packing_min = mc;
                             if (packing_min == 0) break; 
                         }
                         outfile << current_frame << "," << packing_min << "\n";
                    }
                }
                current_frame = f;
                current_rods.clear();
            }
            current_rods.push_back({r_id, {x1, y1, z1}, {x2, y2, z2}});

        } catch (...) {
            continue;
        }
    }

    // Process last frame
    if (current_frame != -1 && !current_rods.empty()) {
         int packing_min = std::numeric_limits<int>::max();
         for (const auto& r : current_rods) {
             int mc = compute_rod_crossing_number(r, current_rods);
             if (mc < packing_min) packing_min = mc;
             if (packing_min == 0) break;
         }
         outfile << current_frame << "," << packing_min << "\n";
    }

    infile.close();
    outfile.close();
    return 0;
}
