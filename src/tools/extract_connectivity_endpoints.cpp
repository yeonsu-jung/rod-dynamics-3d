/**
 * @file extract_connectivity_endpoints.cpp
 * @brief Standalone C++ tool to extract connectivity from endpoints CSV files.
 *        Handles frame,id,x1,y1,z1,x2,y2,z2 format.
 *
 * Usage: ./extract_connectivity_endpoints <input_endpoints.csv> <output_connectivity.csv>
 */

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
#include <limits>

// Geometry Structs
struct Vec3 {
    double x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}
    Vec3 operator+(const Vec3& o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
    Vec3 operator-(const Vec3& o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
    Vec3 operator*(double s) const { return Vec3(x * s, y * s, z * s); }
    double dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return Vec3(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x);
    }
    Vec3 normalized() const {
        double n = std::sqrt(x * x + y * y + z * z);
        if (n < 1e-9) return Vec3(0, 0, 0);
        return Vec3(x/n, y/n, z/n);
    }
};

struct Rod {
    int id;
    Vec3 p1, p2;
    Vec3 center() const { return (p1 + p2) * 0.5; }
    Vec3 direction() const { return (p2 - p1).normalized(); }
    double length() const { return std::sqrt((p2-p1).dot(p2-p1)); }
};

struct FrameData {
    int frame_id;
    std::vector<Rod> rods;
};

struct Edge {
    int frame, source, target;
    bool operator<(const Edge& other) const {
        if (frame != other.frame) return frame < other.frame;
        if (source != other.source) return source < other.source;
        return target < other.target;
    }
};

// CSV Parsing
std::map<int, FrameData> load_frames(const std::string& filename) {
    std::map<int, FrameData> frames;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        exit(1);
    }
    
    std::string line;
    // Skip comments and header
    while (file.peek() == '#') {
        std::getline(file, line);
    }
    
    char c = file.peek();
    if (!isdigit(c) && c != '-') {
        std::getline(file, line); // Skip header
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
            
            Rod r;
            r.id = rod_id;
            r.p1 = Vec3(x1, y1, z1);
            r.p2 = Vec3(x2, y2, z2);
            
            frames[frame].frame_id = frame;
            frames[frame].rods.push_back(r);
            
        } catch (...) {
            continue;
        }
    }
    return frames;
}

// Ray Casting Logic
std::vector<Edge> process_frame(const FrameData& data, int n_rays = 360) {
    std::vector<Edge> edges;
    int N = data.rods.size();

    std::vector<double> ray_dx(n_rays), ray_dy(n_rays);
    double d_theta = 2.0 * M_PI / n_rays;
    for (int i = 0; i < n_rays; ++i) {
        double t = -M_PI + i * d_theta;
        ray_dx[i] = std::cos(t);
        ray_dy[i] = std::sin(t);
    }

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; ++i) {
        const Rod& r = data.rods[i];
        int src_id = r.id;

        Vec3 axis = r.direction();
        Vec3 ref = (std::abs(axis.x) < 0.9) ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
        Vec3 u_vec = axis.cross(ref).normalized();
        Vec3 v_vec = axis.cross(u_vec).normalized();

        struct Seg { double u1, v1, du, dv; int target_idx; };
        std::vector<Seg> segments;
        segments.reserve(N - 1);

        Vec3 center_i = r.center();

        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            const Rod& other = data.rods[j];
            
            Vec3 p1_rel = other.p1 - center_i;
            Vec3 p2_rel = other.p2 - center_i;

            double u1 = p1_rel.dot(u_vec);
            double v1 = p1_rel.dot(v_vec);
            double u2 = p2_rel.dot(u_vec);
            double v2 = p2_rel.dot(v_vec);

            segments.push_back({u1, v1, u2 - u1, v2 - v1, j});
        }

        std::set<int> neighbors;

        for (int r_idx = 0; r_idx < n_rays; ++r_idx) {
            double rdx = ray_dx[r_idx];
            double rdy = ray_dy[r_idx];

            double min_k = 1e30;
            int hit_idx = -1;

            for (const auto& seg : segments) {
                double det = rdy * seg.du - rdx * seg.dv;
                if (std::abs(det) < 1e-9) continue;

                double num_k = seg.v1 * seg.du - seg.u1 * seg.dv;
                double k = num_k / det;

                if (k <= 0 || k >= min_k) continue;

                double num_t = rdx * seg.v1 - rdy * seg.u1;
                double t = num_t / det;

                if (t >= 0.0 && t <= 1.0) {
                    min_k = k;
                    hit_idx = seg.target_idx;
                }
            }

            if (hit_idx != -1) {
                neighbors.insert(data.rods[hit_idx].id);
            }
        }

        #pragma omp critical
        for (int tgt : neighbors) {
            edges.push_back({data.frame_id, src_id, tgt});
        }
    }

    return edges;
}

// Main
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_endpoints.csv> <output_connectivity.csv>\n";
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];

    std::cout << "Loading " << input_path << "..." << std::endl;
    auto frames_map = load_frames(input_path);
    std::cout << "Loaded " << frames_map.size() << " frames." << std::endl;

    if (frames_map.empty()) {
        std::cerr << "No frames loaded. Check input format." << std::endl;
        return 1;
    }

    std::vector<int> frame_indices;
    for (auto const& [k, v] : frames_map) {
        frame_indices.push_back(k);
    }
    std::sort(frame_indices.begin(), frame_indices.end());

    std::vector<Edge> all_edges;
    int count = 0;

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
    out << "frame,source,target\n";
    for (const auto& e : all_edges) {
        out << e.frame << "," << e.source << "," << e.target << "\n";
    }

    std::cout << "Done." << std::endl;
    return 0;
}
