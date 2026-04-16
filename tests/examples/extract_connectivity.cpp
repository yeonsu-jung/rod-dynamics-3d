/**
 * @file extract_connectivity.cpp
 * @brief C++ tool to extract connectivity network from rod packing time data.
 *
 * Usage: ./extract_connectivity <input_csv> <output_csv>
 */

#include "geometry.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#ifdef USE_OPENMP
#include <omp.h>
#endif

using namespace entanglement;

// ---------------------------------------------------------
// Data Structures
// ---------------------------------------------------------

struct FrameData {
    int frame_id;
    std::vector<Rod> rods;
    std::vector<int> rod_ids; // If rod IDs are not sequential/0-indexed in file
};

struct Edge {
    int frame;
    int source;
    int target;
    
    // For sorting/unique
    bool operator<(const Edge& other) const {
        if (frame != other.frame) return frame < other.frame;
        if (source != other.source) return source < other.source;
        return target < other.target;
    }
};

// ---------------------------------------------------------
// CSV Parsing
// ---------------------------------------------------------

std::map<int, FrameData> load_frames(const std::string& filename) {
    std::map<int, FrameData> frames;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        exit(1);
    }
    
    std::string line;
    // Skip header Check
    // We assume header exists if first char is not digit
    char c = file.peek();
    if (!isdigit(c) && c != '-') {
        std::getline(file, line);
    }
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::stringstream ss(line);
        std::string segment;
        std::vector<std::string> parts;
        
        while(std::getline(ss, segment, ',')) {
            parts.push_back(segment);
        }
        
        if (parts.size() < 8) continue;
        
        try {
            int frame = std::stoi(parts[0]);
            int rod_id = std::stoi(parts[1]);
            double x1 = std::stod(parts[2]);
            double y1 = std::stod(parts[3]);
            double z1 = std::stod(parts[4]);
            double x2 = std::stod(parts[5]);
            double y2 = std::stod(parts[6]);
            double z2 = std::stod(parts[7]);
            
            Vec3 p1(x1, y1, z1);
            Vec3 p2(x2, y2, z2);
            Vec3 center = (p1 + p2) * 0.5;
            Vec3 axis = p2 - p1;
            double len = axis.norm();
            
            // Build rod
            // We need phi/theta for Rod struct but extracting directions is enough for this task
            // The Rod struct calculates direction FROM phi/theta.
            // So we must convert back.
            double phi = 0, theta = 0;
            if (len > 1e-9) {
                Vec3 dir = axis / len;
                phi = std::acos(std::clamp(dir.z, -1.0, 1.0));
                theta = std::atan2(dir.y, dir.x);
            }
            
            Rod r(center, phi, theta, len);
            
            frames[frame].frame_id = frame;
            frames[frame].rods.push_back(r);
            frames[frame].rod_ids.push_back(rod_id);
            
        } catch (...) {
            continue;
        }
    }
    return frames;
}

// ---------------------------------------------------------
// Ray Casting Logic
// ---------------------------------------------------------

std::vector<Edge> process_frame(const FrameData& data, int n_rays = 360) {
    std::vector<Edge> edges;
    int N = data.rods.size();
    
    // Per-rod basis construction and ray casting
    // Prepare rays in 2D
    std::vector<double> ray_dx(n_rays), ray_dy(n_rays);
    double d_theta = 2.0 * M_PI / n_rays;
    for(int i=0; i<n_rays; ++i) {
        double t = -M_PI + i * d_theta; // -pi to pi
        ray_dx[i] = std::cos(t);
        ray_dy[i] = std::sin(t);
    }
    
    // Thread-local storage for edges to avoid contention
    #pragma omp parallel 
    {
        std::vector<Edge> local_edges;
        
        #pragma omp for
        for (int i = 0; i < N; ++i) {
            const Rod& r = data.rods[i];
            int src_id = data.rod_ids[i];
            
            Vec3 axis = r.direction();
            // Basis
            Vec3 ref = (std::abs(axis.x) < 0.9) ? Vec3(1,0,0) : Vec3(0,1,0);
            Vec3 u_vec = axis.cross(ref).normalized();
            Vec3 v_vec = axis.cross(u_vec).normalized();
            
            // Project all OTHER rods
            // We store projection segments: (u1, v1) -> (u2, v2)
            struct Seg { double u1, v1, du, dv; int target_idx; };
            std::vector<Seg> segments;
            segments.reserve(N-1);
            
            Vec3 center_i = r.center;
            
            for (int j = 0; j < N; ++j) {
                if (i == j) continue;
                
                const Rod& other = data.rods[j];
                auto [p1, p2] = other.endpoints();
                
                Vec3 p1_rel = p1 - center_i;
                Vec3 p2_rel = p2 - center_i;
                
                double u1 = p1_rel.dot(u_vec);
                double v1 = p1_rel.dot(v_vec);
                double u2 = p2_rel.dot(u_vec);
                double v2 = p2_rel.dot(v_vec);
                
                segments.push_back({u1, v1, u2-u1, v2-v1, j});
            }
            
            // Ray casting
            std::set<int> neighbors;
            
            for (int r_idx = 0; r_idx < n_rays; ++r_idx) {
                double rdx = ray_dx[r_idx];
                double rdy = ray_dy[r_idx];
                
                double min_k = std::numeric_limits<double>::infinity();
                int hit_idx = -1;
                
                // Vectorizable inner loop?
                // Compiler might auto-vectorize this loop over segments
                for (const auto& seg : segments) {
                    // Intersection of Ray (0,0) -> (rdx, rdy) * k
                    // and Segment (u1, v1) + (du, dv) * t
                    // k * rdx = u1 + t * du
                    // k * rdy = v1 + t * dv
                    
                    // Det = rdy * du - rdx * dv
                    double det = rdy * seg.du - rdx * seg.dv;
                    
                    if (std::abs(det) < 1e-9) continue;
                    
                    // Numerators
                    // num_k = v1 * du - u1 * dv
                    // num_t = rdx * v1 - rdy * u1
                    
                    double num_k = seg.v1 * seg.du - seg.u1 * seg.dv;
                    // k = num_k / det
                    double k = num_k / det;
                    
                    if (k <= 0) continue; // Behind or at origin
                    if (k >= min_k) continue; // Further than closest
                    
                    double num_t = rdx * seg.v1 - rdy * seg.u1;
                    double t = num_t / det;
                    
                    if (t >= 0.0 && t <= 1.0) {
                        min_k = k;
                        hit_idx = seg.target_idx;
                    }
                }
                
                if (hit_idx != -1) {
                    neighbors.insert(data.rod_ids[hit_idx]);
                }
            }
            
            for (int tgt : neighbors) {
                local_edges.push_back({data.frame_id, src_id, tgt});
            }
        }
        
        #pragma omp critical
        edges.insert(edges.end(), local_edges.begin(), local_edges.end());
    }
    
    return edges;
}

// ---------------------------------------------------------
// Main
// ---------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_csv> <output_csv>\n";
        return 1;
    }
    
    std::string input_path = argv[1];
    std::string output_path = argv[2];
    
    std::cout << "Loading " << input_path << "..." << std::endl;
    auto frames_map = load_frames(input_path);
    std::cout << "Loaded " << frames_map.size() << " frames." << std::endl;
    
    // Sort frames to process in order
    std::vector<int> frame_indices;
    for(auto const& [k, v] : frames_map) {
        frame_indices.push_back(k);
    }
    std::sort(frame_indices.begin(), frame_indices.end());
    
    std::vector<Edge> all_edges;
    
    std::cout << "Processing frames..." << std::endl;
    int count = 0;
    
    // Can process frames in parallel too, but might saturate
    // process_frame has internal parallelism.
    // Better to process frames serially (for progress tracking) and parallelize internal loops
    
    for (int f_id : frame_indices) {
        auto frame_edges = process_frame(frames_map[f_id]);
        all_edges.insert(all_edges.end(), frame_edges.begin(), frame_edges.end());
        
        count++;
        if (count % 10 == 0) {
            std::cout << "Processed " << count << "/" << frame_indices.size() << " frames.\r" << std::flush;
        }
    }
    std::cout << std::endl;
    
    std::cout << "Saving " << all_edges.size() << " edges to " << output_path << "..." << std::endl;
    
    std::ofstream out(output_path);
    // Header
    out << "frame,source,target\n";
    for (const auto& e : all_edges) {
        out << e.frame << "," << e.source << "," << e.target << "\n";
    }
    
    std::cout << "Done." << std::endl;
    
    return 0;
}
