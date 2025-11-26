#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace std;

constexpr double PI = 3.141592653589793238462643383279502884;

struct Vec3 {
    double x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(double X,double Y,double Z):x(X),y(Y),z(Z){}
    inline Vec3 operator+(const Vec3& o) const { return {x+o.x,y+o.y,z+o.z}; }
    inline Vec3 operator-(const Vec3& o) const { return {x-o.x,y-o.y,z-o.z}; }
    inline Vec3 operator*(double s) const { return {x*s,y*s,z*s}; }
};

static inline double dot(const Vec3&a,const Vec3&b){return a.x*b.x+a.y*b.y+a.z*b.z;}

// Minimum-image displacement for a component given box length L (sim's convention)
static inline double min_image(double d, double L){
    if (L <= 0.0) return d;
    d -= L * std::floor(d / L + 0.5);
    return d;
}

// Robust squared distance between 3D segments under PBC, using centroid-based shift (sim-aligned).
static inline double dist2_seg_pbc(const Vec3& p0, const Vec3& p1,
                                   const Vec3& q0, const Vec3& q1,
                                   double L)
{
    // Compute centroids
    Vec3 cp{0.5*(p0.x + p1.x), 0.5*(p0.y + p1.y), 0.5*(p0.z + p1.z)};
    Vec3 cq{0.5*(q0.x + q1.x), 0.5*(q0.y + q1.y), 0.5*(q0.z + q1.z)};

    // Minimum-image shift to bring q near p
    Vec3 dC{cq.x - cp.x, cq.y - cp.y, cq.z - cp.z};
    dC.x = min_image(dC.x, L);
    dC.y = min_image(dC.y, L);
    dC.z = min_image(dC.z, L);
    Vec3 shift{dC.x - (cq.x - cp.x), dC.y - (cq.y - cp.y), dC.z - (cq.z - cp.z)};

    Vec3 q0w{q0.x + shift.x, q0.y + shift.y, q0.z + shift.z};
    Vec3 q1w{q1.x + shift.x, q1.y + shift.y, q1.z + shift.z};

    Vec3 u{p1.x - p0.x, p1.y - p0.y, p1.z - p0.z};
    Vec3 v{q1w.x - q0w.x, q1w.y - q0w.y, q1w.z - q0w.z};
    Vec3 w0{p0.x - q0w.x, p0.y - q0w.y, p0.z - q0w.z};

    const double eps = 1e-12;
    double uu = dot(u,u), vv = dot(v,v), uv = dot(u,v);
    double wu = dot(w0,u), wv = dot(w0,v);
    double D = uu*vv - uv*uv;

    double s, t;
    if (fabs(D) < eps) {
        s = 0.0;
        t = (vv >= eps) ? (-wv / vv) : 0.0;
    } else {
        s = (uv*wv - vv*wu) / D;
        t = (uu*wv - uv*wu) / D;
    }

    if (s < 0.0) s = 0.0; else if (s > 1.0) s = 1.0;
    t = (s*uv + wv) / (vv >= eps ? vv : 1.0);
    if (t < 0.0) t = 0.0; else if (t > 1.0) t = 1.0;

    double su = (-wu + t*uv) / (uu >= eps ? uu : 1.0);
    if (!(t > 1e-6 && t < 1-1e-6)) {
        if (su < 0.0) s = 0.0;
        else if (su > 1.0) s = 1.0;
        else s = su;
    }

    double dx = w0.x + s*u.x - t*v.x;
    double dy = w0.y + s*u.y - t*v.y;
    double dz = w0.z + s*u.z - t*v.z;
    return dx*dx + dy*dy + dz*dz;
}

struct RodAngles {
    double cx, cy, cz;
    double phi, theta;
};

// random unit vector from spherical angles
static inline Vec3 dir_from_phi_theta(double phi, double theta){
    double s = sin(phi);
    return { s*cos(theta), s*sin(theta), cos(phi) };
}

// Brute-force PBC rod packing
vector<RodAngles> generate_rods_bruteforce_pbc(
    int N,
    double C,
    double rod_length,
    double rod_diameter,
    uint64_t seed,
    int max_attempts_per_rod = 1000000)
{
    vector<RodAngles> rods;
    rods.reserve(N);

    const double R = 0.5 * rod_diameter;
    const double D = 2.0 * R;
    const double D2 = D * D;
    const double halfL = 0.5 * rod_length;
    const double box_L = 2.0 * C;

    mt19937_64 rng(seed);
    uniform_real_distribution<double> U01(0.0, 1.0);
    auto U = [&](double a, double b){ return a + (b-a)*U01(rng); };

    for (int i = 0; i < N; ++i) {
        bool placed = false;
        for (int attempt = 0; attempt < max_attempts_per_rod && !placed; ++attempt) {
            double cx = U(-C, C);
            double cy = U(-C, C);
            double cz = U(-C, C);

            double phi   = U(0.0, PI);
            double theta = U(0.0, 2.0 * PI);
            Vec3 u = dir_from_phi_theta(phi, theta);

            Vec3 c{cx, cy, cz};
            Vec3 p0 = c - u * halfL;
            Vec3 p1 = c + u * halfL;

            bool collide = false;
            for (int j = 0; j < (int)rods.size(); ++j) {
                const auto &rj = rods[j];
                Vec3 cj{rj.cx, rj.cy, rj.cz};
                Vec3 uj = dir_from_phi_theta(rj.phi, rj.theta);
                Vec3 q0 = cj - uj * halfL;
                Vec3 q1 = cj + uj * halfL;

                double d2 = dist2_seg_pbc(p0, p1, q0, q1, box_L);
                if (d2 < D2) {
                    collide = true;
                    break;
                }
            }

            if (!collide) {
                rods.push_back({cx, cy, cz, phi, theta});
                placed = true;
            }
        }

        if (!placed) {
            cerr << "Failed to place rod " << i
                 << " without overlap after " << max_attempts_per_rod
                 << " attempts\n";
            break;
        }
    }

    return rods;
}

int main(int argc, char** argv){
    double C = 0.55;
    double rod_length = 1.0;
    double alpha = 100.0;
    int    N = 932;
    uint64_t seed = 12345;
    string output_path = "rods932_bruteforce.csv";

    for(int i=1;i<argc;++i){
        string a = argv[i];
        auto need = [&](const char* name){
            if(i+1>=argc){ cerr << "Missing value for " << name << "\n"; exit(1); }
            return string(argv[++i]);
        };
        if(a=="--C") C = stod(need("--C"));
        else if(a=="--rod_length") rod_length = stod(need("--rod_length"));
        else if(a=="--alpha") alpha = stod(need("--alpha"));
        else if(a=="--N") N = stoi(need("--N"));
        else if(a=="--seed") seed = stoull(need("--seed"));
        else if(a=="--output" || a=="-o") output_path = need("--output");
        else {
            cerr << "Unknown arg: " << a << "\n";
            return 2;
        }
    }

    double rod_diam = rod_length / alpha;

    auto t0 = chrono::high_resolution_clock::now();
    vector<RodAngles> rods = generate_rods_bruteforce_pbc(N, C, rod_length, rod_diam, seed);
    auto t1 = chrono::high_resolution_clock::now();
    double dt = chrono::duration<double>(t1-t0).count();

    int placed = (int)rods.size();
    cout << "Placed " << placed << "/" << N << " rods in " << dt << " s\n";

    // All-pairs PBC min-gap verification (capsule surface-to-surface)
    const double R = 0.5 * rod_diam;
    const double sumR = 2.0 * R;
    const double box_L = 2.0 * C;
    const double halfL = 0.5 * rod_length;

    double minGap = numeric_limits<double>::infinity();
    int minI = -1, minJ = -1;

    for(int i = 0; i < placed; ++i){
        const auto &ri = rods[i];
        Vec3 ci{ri.cx, ri.cy, ri.cz};
        Vec3 ui = dir_from_phi_theta(ri.phi, ri.theta);
        Vec3 p0i = ci - ui * halfL;
        Vec3 p1i = ci + ui * halfL;
        for(int j = i+1; j < placed; ++j){
            const auto &rj = rods[j];
            Vec3 cj{rj.cx, rj.cy, rj.cz};
            Vec3 uj = dir_from_phi_theta(rj.phi, rj.theta);
            Vec3 p0j = cj - uj * halfL;
            Vec3 p1j = cj + uj * halfL;

            double d2 = dist2_seg_pbc(p0i, p1i, p0j, p1j, box_L);
            double d = sqrt(d2);
            double gap = d - sumR;
            if(gap < minGap){
                minGap = gap;
                minI = i;
                minJ = j;
            }
        }
    }

    cout.setf(std::ios::scientific);
    cout << "All-pairs PBC min gap: " << minGap
         << " (i=" << minI << ", j=" << minJ << ")\n";

    if(minGap < 0.0){
        cerr << "FATAL: Generated configuration has overlap (minGap="
             << minGap << ")\n";
        return 1;
    }

    // Write endpoints CSV
    ofstream eout(output_path, ios::out | ios::trunc);
    if(!eout.good()){
        cerr << "Failed to open output file: " << output_path << "\n";
        return 1;
    }

    eout << "# rod_length=" << rod_length << "\n";
    eout << "# rod_diameter=" << rod_diam << "\n";
    eout << "# alpha=" << alpha << "\n";
    eout << "# container_C=" << C << "\n";
    eout << "# box_size=" << (2*C) << "\n";
    eout << "# pbc=true" << "\n";
    eout << "# seed=" << seed << "\n";
    eout << "# placed=" << placed << "\n";
    eout << "x0,y0,z0,x1,y1,z1\n";
    eout << scientific << setprecision(12);
    for(int i=0;i<placed;++i){
        const auto &r = rods[i];
        Vec3 c{r.cx, r.cy, r.cz};
        Vec3 u = dir_from_phi_theta(r.phi, r.theta);
        Vec3 p0 = c - u * halfL;
        Vec3 p1 = c + u * halfL;
        eout << p0.x << "," << p0.y << "," << p0.z << ","
             << p1.x << "," << p1.y << "," << p1.z << "\n";
    }

    eout.close();
    cout << "Wrote endpoints to: " << output_path << "\n";

    return 0;
}
