#include <vector>
#include <array>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "linking_number.h" // Vector3D, compute_linking_number_cartesian

// Local copy of segment-segment distance from entanglement-cpp
static double segment_segment_distance_local(const Vector3D& p1, const Vector3D& q1,
                                             const Vector3D& p2, const Vector3D& q2) {
    const double EPS = 1e-12;
    Vector3D u = q1 - p1;
    Vector3D v = q2 - p2;
    Vector3D w = p1 - p2;
    double a = u.dot(u);
    double b = u.dot(v);
    double c = v.dot(v);
    double d = u.dot(w);
    double e = v.dot(w);
    double D = a * c - b * b;

    double sN, sD = D;
    double tN, tD = D;

    if (D < EPS) {
        sN = 0.0; sD = 1.0;
        tN = e;  tD = c;
    } else {
        sN = (b * e - c * d);
        tN = (a * e - b * d);
        if (sN < 0.0) { sN = 0.0; tN = e; tD = c; }
        else if (sN > sD) { sN = sD; tN = e + b; tD = c; }
    }

    if (tN < 0.0) {
        tN = 0.0;
        if (-d < 0.0) { sN = 0.0; sD = 1.0; }
        else if (-d > a) { sN = sD; }
        else { sN = -d; sD = a; }
    } else if (tN > tD) {
        tN = tD;
        if ((-d + b) < 0.0) { sN = 0.0; sD = 1.0; }
        else if ((-d + b) > a) { sN = sD; }
        else { sN = (-d + b); sD = a; }
    }

    double sc = (std::abs(sN) < EPS ? 0.0 : sN / sD);
    double tc = (std::abs(tN) < EPS ? 0.0 : tN / tD);

    Vector3D dP = w + u * sc - v * tc;
    return dP.norm();
}

double pairwise_abs_linking_sum_with_cutoff(
    const std::vector<std::array<double,6>>& rods_array,
    double cutoff,
    long long* out_pairs,
    int num_threads
) {
    const int N = static_cast<int>(rods_array.size());

    auto get_seg = [&](int i, Vector3D& a, Vector3D& b){
        a = Vector3D{rods_array[i][0], rods_array[i][1], rods_array[i][2]};
        b = Vector3D{rods_array[i][3], rods_array[i][4], rods_array[i][5]};
    };

    // Fast path: no cutoff
    if (!(cutoff > 0.0)) {
        long long pairs = 0;
        double total_abs_linking = 0.0;
        #pragma omp parallel for schedule(dynamic,64) reduction(+: total_abs_linking, pairs) if(num_threads>1)
        for (int i = 0; i < N; ++i) {
            Vector3D ai, bi; get_seg(i, ai, bi);
            for (int j = i + 1; j < N; ++j) {
                Vector3D aj, bj; get_seg(j, aj, bj);
                double lk = compute_linking_number_cartesian(ai, bi, aj, bj);
                total_abs_linking += std::abs(lk);
                ++pairs;
            }
        }
        if (out_pairs) *out_pairs = pairs;
        return total_abs_linking;
    }

    // With cutoff: prune using center distance lower bound, then segment distance
    std::vector<Vector3D> centers; centers.reserve(N);
    for (int i = 0; i < N; ++i) {
        centers.emplace_back(
            0.5 * (rods_array[i][0] + rods_array[i][3]),
            0.5 * (rods_array[i][1] + rods_array[i][4]),
            0.5 * (rods_array[i][2] + rods_array[i][5])
        );
    }

    long long pairs = 0;
    double total_abs_linking = 0.0;

    #pragma omp parallel for schedule(dynamic,64) reduction(+: total_abs_linking, pairs) if(num_threads>1)
    for (int i = 0; i < N; ++i) {
        Vector3D ai, bi; get_seg(i, ai, bi);
        for (int j = i + 1; j < N; ++j) {
            double centerLower = std::max(0.0, (centers[i] - centers[j]).norm() - 1.0);
            if (centerLower > cutoff) continue;
            Vector3D aj, bj; get_seg(j, aj, bj);
            double sdist = segment_segment_distance_local(ai, bi, aj, bj);
            if (sdist > cutoff) continue;
            double lk = compute_linking_number_cartesian(ai, bi, aj, bj);
            total_abs_linking += std::abs(lk);
            ++pairs;
        }
    }

    if (out_pairs) *out_pairs = pairs;
    return total_abs_linking;
}
