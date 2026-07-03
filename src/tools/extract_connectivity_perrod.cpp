/**
 * @file extract_connectivity_perrod.cpp
 * @brief Standalone C++ tool to extract connectivity from perrod.csv files.
 *        Handles Quaternions -> Endpoints conversion and Ray-Casting.
 *
 * Usage: ./extract_connectivity_perrod <input_perrod.csv> <output_connectivity.csv>
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

// ---------------------------------------------------------
// Geometry Structs (Embedded to avoid external deps)
// ---------------------------------------------------------

struct Vec3 {
    double x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}

    Vec3 operator+(const Vec3& o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
    Vec3 operator-(const Vec3& o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
    Vec3 operator*(double s) const { return Vec3(x * s, y * s, z * s); }
    Vec3 operator/(double s) const { return Vec3(x / s, y / s, z / s); }

    double dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }

    Vec3 cross(const Vec3& o) const {
        return Vec3(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x);
    }

    double norm() const { return std::sqrt(x * x + y * y + z * z); } // Bug fixed in norm calculation below

    Vec3 normalized() const {
        double n = std::sqrt(x * x + y * y + z * z);
        if (n < 1e-9) return Vec3(0, 0, 0);
        return *this / n;
    }
};

struct Quaternion {
    double w, x, y, z;

    // Rotate vector [0,1,0] by this quaternion. The rod axis is local +Y
    // (see RigidBody::axisY / capsuleEndpoints and convert_perrod_endpoints).
    Vec3 rotateY() const {
        return Vec3(
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x)
        );
    }
};

struct Rod {
    int id;
    Vec3 center;
    Quaternion q;
    double length;

    Vec3 direction() const {
        return q.rotateY().normalized();
    }

    std::pair<Vec3, Vec3> endpoints() const {
        Vec3 dir = direction();
        Vec3 p1 = center - dir * (length * 0.5);
        Vec3 p2 = center + dir * (length * 0.5);
        return {p1, p2};
    }
};

struct FrameData {
    int frame_id;
    std::vector<Rod> rods;
};

struct Edge {
    int frame;
    int source;
    int target;

    bool operator<(const Edge& other) const {
        if (frame != other.frame) return frame < other.frame;
        if (source != other.source) return source < other.source;
        return target < other.target;
    }
};

// ---------------------------------------------------------
// CSV Parsing
// ---------------------------------------------------------

std::map<int, FrameData> load_perrod(const std::string& filename) {
    std::map<int, FrameData> frames;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        exit(1);
    }

    double global_rod_length = 1.0; // Default
    std::string line;

    // Parse Header for Metadata
    // Look for # rod_length=...
    while (file.peek() == '#') {
        std::getline(file, line);
        if (line.find("rod_length=") != std::string::npos) {
            size_t pos = line.find("rod_length=");
            std::string val = line.substr(pos + 11);
            try {
                global_rod_length = std::stod(val);
                std::cout << "Found rod_length=" << global_rod_length << std::endl;
            } catch (...) {}
        }
    }

    // Header line: frame,rod,px,py,pz,vx,vy,vz,wx,wy,wz,qw,qx,qy,qz,...
    // We expect header line after comments, skip it if it starts with 'frame' or non-digit
    char c = file.peek();
    if (!isdigit(c) && c != '-') {
        std::getline(file, line); // Skip header
    }

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string segment;
        std::vector<std::string> parts;
        while (std::getline(ss, segment, ',')) {
            parts.push_back(segment);
        }

        // Needs at least up to qz (15th column, 0-indexed)
        if (parts.size() < 15) continue;

        try {
            int frame = std::stoi(parts[0]);
            int id = std::stoi(parts[1]);
            
            // px,py,pz at 2,3,4
            double px = std::stod(parts[2]);
            double py = std::stod(parts[3]);
            double pz = std::stod(parts[4]);
            
            // qw,qx,qy,qz at 11,12,13,14
            double qw = std::stod(parts[11]);
            double qx = std::stod(parts[12]);
            double qy = std::stod(parts[13]);
            double qz = std::stod(parts[14]);
            
            Rod r;
            r.id = id;
            r.center = Vec3(px, py, pz);
            r.q = {qw, qx, qy, qz};
            r.length = global_rod_length;
            
            frames[frame].frame_id = frame;
            frames[frame].rods.push_back(r);
            
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

    // Pre-sort rods by ID to ensure consistency? (Already usually sorted by ID in map)
    // Actually data.rods might be out of order if parsed randomly, but CSV is usually ordered.
    // Let's rely on data.rods indices for inner loops but use r.id for output.

    std::vector<double> ray_dx(n_rays), ray_dy(n_rays);
    double d_theta = 2.0 * 3.14159265359 / n_rays;
    for (int i = 0; i < n_rays; ++i) {
        double t = -3.14159265359 + i * d_theta;
        ray_dx[i] = std::cos(t);
        ray_dy[i] = std::sin(t);
    }

    // OpenMP could be used here if compiled with -fopenmp
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; ++i) {
        const Rod& r = data.rods[i];
        int src_id = r.id;

        Vec3 axis = r.direction();
        // Basis
        Vec3 ref = (std::abs(axis.x) < 0.9) ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
        Vec3 u_vec = axis.cross(ref).normalized();
        Vec3 v_vec = axis.cross(u_vec).normalized();

        struct Seg { double u1, v1, du, dv; int target_idx; };
        std::vector<Seg> segments;
        segments.reserve(N - 1);

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

            segments.push_back({u1, v1, u2 - u1, v2 - v1, j});
        }

        std::set<int> neighbors;

        for (int r_idx = 0; r_idx < n_rays; ++r_idx) {
            double rdx = ray_dx[r_idx];
            double rdy = ray_dy[r_idx];

            double min_k = 1e30; // Infinity
            int hit_idx = -1;

            for (const auto& seg : segments) {
                // k * r = u1 + t * du
                // Det = rdy * du - rdx * dv
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

// ---------------------------------------------------------
// Main
// ---------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_perrod.csv> <output_connectivity.csv>\n";
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];

    std::cout << "Loading " << input_path << "..." << std::endl;
    auto frames_map = load_perrod(input_path);
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
