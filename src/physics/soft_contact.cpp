/**
 * @file soft_contact.cpp
 * @brief Implementation of soft penalty-based contact solver
 */

#include "physics/soft_contact.hpp"
#include "physics/rigid_body.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>

#ifdef USE_CUDA
#include "physics/cuda_broadphase.hpp"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

extern int g_thread_limit;
extern bool gQuiet;

#ifdef _OPENMP
namespace {
void reportOpenMpTeamSizeOnce(const char *label) {
  static std::unordered_map<std::string, bool> reported;
  if (gQuiet) {
    return;
  }

#pragma omp critical(openmp_team_report)
  {
    bool &alreadyReported = reported[label];
    if (!alreadyReported) {
      alreadyReported = true;
      std::cout << "[Info] OpenMP actual team size for " << label << ": "
                << omp_get_num_threads() << "\n";
    }
  }
}
} // namespace
#endif

// Aligned buffer to prevent false sharing between threads
struct alignas(64) ThreadContactBuffer {
  std::vector<ContactPrimitive> contacts;
};

SoftContactSolver::SoftContactSolver(const SoftContactCfg &config)
    : config_(config) {
  K1_ = 15.0 / config_.delta;
  K2_ = 15.0 / config_.nu;
  lastPotentialEnergy_ = 0.0;
}

void SoftContactSolver::setConfig(const SoftContactCfg &config) {
  config_ = config;
  K1_ = 15.0 / config_.delta;
  K2_ = 15.0 / config_.nu;
  // Reset accumulator on config change (avoids stale reporting)
  lastPotentialEnergy_ = 0.0;
}
void SoftContactSolver::setPBC(bool enabled, const glm::vec3 &min,
                               const glm::vec3 &max) {
  pbcEnabled_ = enabled;
  pbcMin_ = min;
  pbcMax_ = max;
  pbcSize_ = max - min;
}

void SoftContactSolver::detectContacts(const std::vector<RigidBody> &bodies) {
  contacts_.clear();

  int num_free = 0;
  int free_idx = -1;
  for (size_t i = 0; i < bodies.size(); ++i) {
    if (bodies[i].invMass > 0.0f) {
      num_free++;
      free_idx = static_cast<int>(i);
    }
  }

  if (num_free == 1 && free_idx >= 0) {
    auto t0 = std::chrono::high_resolution_clock::now();
    const RigidBody &a = bodies[free_idx];
    const double R_a = (a.type == ShapeType::Capsule ? (a.cap.h + a.cap.r) : a.sphere.r);

#ifdef _OPENMP
    int num_threads = (g_thread_limit > 0) ? g_thread_limit : omp_get_max_threads();
    // Re-use or resize persistent buffers
    if ((int)threadBuffers_.size() != num_threads) {
      threadBuffers_.assign(num_threads, std::vector<ContactPrimitive>());
      for (auto &v : threadBuffers_) v.reserve(128);
    }
    for (auto &v : threadBuffers_) v.clear();

    // Parallelize single-free-rod path for all reasonable N.
    bool use_omp = (num_threads > 1 && bodies.size() > 50);
#else
    bool use_omp = false;
#endif

    if (use_omp) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(num_threads)
      for (int i = 0; i < (int)bodies.size(); ++i) {
        if (i == free_idx) continue;

        const RigidBody &b = bodies[i];
        const double R_b = (b.type == ShapeType::Capsule ? (b.cap.h + b.cap.r) : b.sphere.r);
        const double max_dist = R_a + R_b + config_.delta;

        glm::vec3 delta_pos = b.x - a.x;
        if (pbcEnabled_) {
          for (int k = 0; k < 3; ++k) {
            if (pbcSize_[k] > 0.0f) {
              float n = std::floor(delta_pos[k] / pbcSize_[k] + 0.5f);
              delta_pos[k] -= n * pbcSize_[k];
            }
          }
        }
        
        if (glm::dot(delta_pos, delta_pos) > (max_dist * max_dist)) {
          continue;
        }

        int tid = omp_get_thread_num();
        auto &local_contacts = threadBuffers_[tid];
        if (a.type == ShapeType::Capsule && b.type == ShapeType::Capsule) {
          detectCapsuleCapsule(a, b, free_idx, i, local_contacts);
        } else if (a.type == ShapeType::Sphere && b.type == ShapeType::Sphere) {
          detectSphereSphere(a, b, free_idx, i, local_contacts);
        } else if (a.type == ShapeType::Sphere && b.type == ShapeType::Capsule) {
          detectSphereCapsule(a, b, free_idx, i, local_contacts);
        } else if (a.type == ShapeType::Capsule && b.type == ShapeType::Sphere) {
          detectSphereCapsule(b, a, i, free_idx, local_contacts);
        }
      }

      for (const auto &tc : threadBuffers_) {
        contacts_.insert(contacts_.end(), tc.begin(), tc.end());
      }
#endif
    } else {
      // SERIAL PATH (much faster for N=2000 single-free-rod due to zero overhead)
      for (int i = 0; i < (int)bodies.size(); ++i) {
        if (i == free_idx) continue;

        const RigidBody &b = bodies[i];
        const double R_b = (b.type == ShapeType::Capsule ? (b.cap.h + b.cap.r) : b.sphere.r);
        const double max_dist = R_a + R_b + config_.delta;

        glm::vec3 delta_pos = b.x - a.x;
        if (pbcEnabled_) {
          for (int k = 0; k < 3; ++k) {
            if (pbcSize_[k] > 0.0f) {
              float n = std::floor(delta_pos[k] / pbcSize_[k] + 0.5f);
              delta_pos[k] -= n * pbcSize_[k];
            }
          }
        }
        
        if (glm::dot(delta_pos, delta_pos) > (max_dist * max_dist)) {
          continue;
        }

        if (a.type == ShapeType::Capsule && b.type == ShapeType::Capsule) {
          detectCapsuleCapsule(a, b, free_idx, i, contacts_);
        } else if (a.type == ShapeType::Sphere && b.type == ShapeType::Sphere) {
          detectSphereSphere(a, b, free_idx, i, contacts_);
        } else if (a.type == ShapeType::Sphere && b.type == ShapeType::Capsule) {
          detectSphereCapsule(a, b, free_idx, i, contacts_);
        } else if (a.type == ShapeType::Capsule && b.type == ShapeType::Sphere) {
          detectSphereCapsule(b, a, i, free_idx, contacts_);
        }
      }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    stats_.cuda_kernel_ms = 0.0;
    stats_.cuda_pack_ms = 0.0;
    stats_.cuda_download_ms = 0.0;
    stats_.cuda_contacts = 0;
    stats_.count_ms = 0.0;
    stats_.prefix_ms = 0.0;
    stats_.fill_ms = 0.0;
    stats_.sort_ms = 0.0;
    stats_.detect_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (config_.verbose) {
      std::cout << "[SoftContact] contacts=" << contacts_.size() << " (single free rod optimized)\n";
    }
    return;
  }

#ifdef USE_CUDA
  if (config_.use_cuda) {
    detectContactsCuda(bodies);
  } else
#endif
  if (config_.use_spatial_hash) {
    detectContactsSpatialHash(bodies);
  } else {
    detectContactsNaive(bodies);
  }

  // Verbose summary: total contacts only (user request)
  if (config_.verbose) {
    std::cout << "[SoftContact] contacts=" << contacts_.size() << "\n";
  }
}

void SoftContactSolver::detectContactsNaive(
    const std::vector<RigidBody> &bodies) {
  // Broadphase: All pairs (O(n²) - fine for small number of objects)
#ifdef _OPENMP
  int num_threads = (g_thread_limit > 0) ? g_thread_limit : omp_get_max_threads();
  // Re-use persistent buffers
  if ((int)threadBuffers_.size() != num_threads) {
    threadBuffers_.assign(num_threads, std::vector<ContactPrimitive>());
    for (auto &v : threadBuffers_) v.reserve(100);
  }
  for (auto &v : threadBuffers_) v.clear();

#pragma omp parallel for schedule(dynamic, 4) num_threads(num_threads)
  for (int i = 0; i < (int)bodies.size(); ++i) {
    if (i == 0) {
      reportOpenMpTeamSizeOnce("contact detection");
    }
    int tid = omp_get_thread_num();
    auto &local_contacts = threadBuffers_[tid];
    for (int j = i + 1; j < (int)bodies.size(); ++j) {
      const RigidBody &a = bodies[i];
      const RigidBody &b = bodies[j];

      // Skip pairs where both bodies are fixed (invMass == 0); contacts
      // between two static bodies cannot affect dynamics.
      if (a.invMass <= 0.0f && b.invMass <= 0.0f)
        continue;

      // Dispatch based on shape types
      if (a.type == ShapeType::Capsule && b.type == ShapeType::Capsule) {
        detectCapsuleCapsule(a, b, i, j, local_contacts);
      } else if (a.type == ShapeType::Sphere && b.type == ShapeType::Sphere) {
        detectSphereSphere(a, b, i, j, local_contacts);
      } else if (a.type == ShapeType::Sphere && b.type == ShapeType::Capsule) {
        detectSphereCapsule(a, b, i, j, local_contacts);
      } else if (a.type == ShapeType::Capsule && b.type == ShapeType::Sphere) {
        detectSphereCapsule(b, a, j, i, local_contacts);
      }
    }
  }

  // Merge results
  for (const auto &tc : threadBuffers_) {
    contacts_.insert(contacts_.end(), tc.begin(), tc.end());
  }
#else
  for (size_t i = 0; i < bodies.size(); ++i) {
    for (size_t j = i + 1; j < bodies.size(); ++j) {
      const RigidBody &a = bodies[i];
      const RigidBody &b = bodies[j];

      // Skip pairs where both bodies are fixed.
      if (a.invMass <= 0.0f && b.invMass <= 0.0f)
        continue;

      // Dispatch based on shape types
      if (a.type == ShapeType::Capsule && b.type == ShapeType::Capsule) {
        detectCapsuleCapsule(a, b, i, j, contacts_);
      } else if (a.type == ShapeType::Sphere && b.type == ShapeType::Sphere) {
        detectSphereSphere(a, b, i, j, contacts_);
      } else if (a.type == ShapeType::Capsule && b.type == ShapeType::Sphere) {
        detectSphereCapsule(b, a, j, i, contacts_);
      } else if (a.type == ShapeType::Sphere && b.type == ShapeType::Capsule) {
        detectSphereCapsule(a, b, i, j, contacts_);
      }
    }
  }
#endif
}

// ---------------------------------------------------------------------------
// CUDA broadphase: GPU-accelerated naive O(N^2) capsule-capsule detection
// ---------------------------------------------------------------------------
#ifdef USE_CUDA
void SoftContactSolver::detectContactsCuda(
    const std::vector<RigidBody> &bodies) {

  using Clock = std::chrono::high_resolution_clock;

  // 1. GPU broadphase: pack SoA, upload, kernel, download
  const auto t0 = Clock::now();
  std::vector<GpuContactRaw> raw;
  raw.reserve(64);
  cudaDetectCapsulePairsTwoPass(
      bodies,
      static_cast<float>(config_.delta),
      pbcEnabled_,
      pbcSize_.x, pbcSize_.y, pbcSize_.z,
      raw);
  const auto t1 = Clock::now();

  stats_.cuda_kernel_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  stats_.cuda_contacts = static_cast<int>(raw.size());

  // 2. Convert GpuContactRaw → ContactPrimitive
  contacts_.reserve(contacts_.size() + raw.size());
  constexpr double eps = 1e-6;
  for (const GpuContactRaw &r : raw) {
    ContactPrimitive cp;
    cp.body_a = r.a;
    cp.body_b = r.b;
    cp.point_a = glm::vec3(r.px_a, r.py_a, r.pz_a);
    cp.point_b = glm::vec3(r.px_b, r.py_b, r.pz_b);
    cp.normal  = glm::vec3(r.nx,   r.ny,   r.nz);
    cp.distance      = static_cast<double>(r.dist);
    cp.surface_limit = static_cast<double>(r.surface_limit);
    cp.shift_b = glm::vec3(r.shift_bx, r.shift_by, r.shift_bz);
    // Zero out forces (filled by computeForces)
    cp.force_a = cp.force_b = glm::vec3(0.0f);
    cp.friction_a = cp.friction_b = glm::vec3(0.0f);

    // Classify contact type from Lumelsky parameters
    const double s = r.s, t = r.t;
    if (s < eps && t < eps) {
      cp.type = ContactType::POINT_TO_POINT;
    } else if (s > 1.0 - eps && t > 1.0 - eps) {
      cp.type = ContactType::POINT_TO_POINT;
    } else if (s < eps || s > 1.0 - eps) {
      cp.type = ContactType::EDGE_TO_POINT;
    } else if (t < eps || t > 1.0 - eps) {
      cp.type = ContactType::EDGE_TO_POINT;
    } else {
      cp.type = ContactType::EDGE_TO_EDGE;
    }

    contacts_.push_back(cp);
  }

  // 3. Handle non-capsule contacts on CPU (sphere-sphere, sphere-capsule)
  //    These are rare/absent in pure-rod simulations, but kept correct.
  bool has_spheres = false;
  for (const auto &b : bodies) {
    if (b.type == ShapeType::Sphere) { has_spheres = true; break; }
  }
  if (has_spheres) {
    for (size_t i = 0; i < bodies.size(); ++i) {
      for (size_t j = i + 1; j < bodies.size(); ++j) {
        const RigidBody &a = bodies[i];
        const RigidBody &b = bodies[j];
        if (a.type == ShapeType::Sphere && b.type == ShapeType::Sphere) {
          detectSphereSphere(a, b, i, j, contacts_);
        } else if (a.type == ShapeType::Sphere && b.type == ShapeType::Capsule) {
          detectSphereCapsule(a, b, i, j, contacts_);
        } else if (a.type == ShapeType::Capsule && b.type == ShapeType::Sphere) {
          detectSphereCapsule(b, a, j, i, contacts_);
        }
      }
    }
  }
}
#endif // USE_CUDA

double SoftContactSolver::computeAdaptiveCellSize(
    const std::vector<RigidBody> &bodies) const {
  if (bodies.empty())
    return 1.0;

  double max_dim = 0.0;
  double sum_dim = 0.0;
  int count = 0;

  // Sample a subset if too many bodies
  int step = (bodies.size() > 1000) ? (bodies.size() / 100) : 1;

  for (size_t i = 0; i < bodies.size(); i += step) {
    const auto &b = bodies[i];
    double dim = 0.0;
    if (b.type == ShapeType::Capsule) {
      // For capsule, use length + diameter
      dim = b.cap.h * 2.0 + b.cap.r * 2.0;
    } else if (b.type == ShapeType::Sphere) {
      dim = b.sphere.r * 2.0;
    }
    max_dim = std::max(max_dim, dim);
    sum_dim += dim;
    count++;
  }

  double avg_dim = (count > 0) ? (sum_dim / count) : 0.0;

  // Heuristic: cell size should be larger than average object, but not too
  // large If objects are very different in size, max_dim is safer to avoid
  // checking too many neighbors But for rods, they are long. A cell size of
  // ~length is good.
  return std::max(max_dim, avg_dim * 1.5);
}

void SoftContactSolver::getAABB(const RigidBody &b, glm::vec3 &min_pt,
                                glm::vec3 &max_pt) const {
  double margin = config_.delta; // Include interaction margin
  if (b.type == ShapeType::Capsule) {
    glm::vec3 axis = b.axisY();
    glm::vec3 p1 = b.x - axis * b.cap.h;
    glm::vec3 p2 = b.x + axis * b.cap.h;
    double r = b.cap.r + margin;
    min_pt = glm::min(p1, p2) - glm::vec3(r);
    max_pt = glm::max(p1, p2) + glm::vec3(r);
  } else { // Sphere
    double r = b.sphere.r + margin;
    min_pt = b.x - glm::vec3(r);
    max_pt = b.x + glm::vec3(r);
  }
}

bool SoftContactSolver::checkAABBOverlap(const RigidBody &a,
                                         const RigidBody &b) const {
  glm::vec3 min_a, max_a;
  getAABB(a, min_a, max_a);

  glm::vec3 min_b, max_b;
  getAABB(b, min_b, max_b);

  if (max_a.x < min_b.x || min_a.x > max_b.x)
    return false;
  if (max_a.y < min_b.y || min_a.y > max_b.y)
    return false;
  if (max_a.z < min_b.z || min_a.z > max_b.z)
    return false;

  return true;
}

void SoftContactSolver::insertBodyIntoGrid(int bodyIdx, const RigidBody &body,
                                           double cellSize, GridMap &grid) {
  // Compute AABB
  glm::vec3 min_pt, max_pt;
  getAABB(body, min_pt, max_pt);

  // Determine grid cell range
  int min_x = static_cast<int>(std::floor(min_pt.x / cellSize));
  int min_y = static_cast<int>(std::floor(min_pt.y / cellSize));
  int min_z = static_cast<int>(std::floor(min_pt.z / cellSize));

  int max_x = static_cast<int>(std::floor(max_pt.x / cellSize));
  int max_y = static_cast<int>(std::floor(max_pt.y / cellSize));
  int max_z = static_cast<int>(std::floor(max_pt.z / cellSize));

  // Insert into all overlapping cells
  for (int x = min_x; x <= max_x; ++x) {
    for (int y = min_y; y <= max_y; ++y) {
      for (int z = min_z; z <= max_z; ++z) {
        grid[{x, y, z}].push_back(bodyIdx);
      }
    }
  }
}

// Helper struct for parallel sort
struct SpatialEntry {
  uint64_t key;
  int bodyIdx;
  bool operator<(const SpatialEntry &other) const {
    if (key != other.key)
      return key < other.key;
    return bodyIdx < other.bodyIdx;
  }
};

// Packed 64-bit key to avoid hash collisions
inline uint64_t hashPos64(int x, int y, int z) {
  // Pack 21 bits each. Offset to handle negatives.
  // 2^20 = 1,048,576. Range [-500k, +500k]. Sufficient.
  uint64_t ux = (uint64_t)(x + 1000000);
  uint64_t uy = (uint64_t)(y + 1000000);
  uint64_t uz = (uint64_t)(z + 1000000);
  return (ux << 42) | (uy << 21) | uz;
}

struct BodyCellRange {
  int min_x, min_y, min_z;
  int max_x, max_y, max_z;
};

void SoftContactSolver::detectContactsSpatialHash(
    const std::vector<RigidBody> &bodies) {
#ifdef _OPENMP
  if (g_thread_limit > 0) {
    omp_set_num_threads(g_thread_limit);
  }
#endif
  using Clock = std::chrono::high_resolution_clock;
  if (bodies.empty())
    return;

  double cellSize = config_.cell_size;
  if (cellSize <= 0.0) {
    cellSize = computeAdaptiveCellSize(bodies);
    static bool printed = false;
    if (!printed && config_.verbose) {
      std::cout << "[SoftContact] Adaptive cell size computed: " << cellSize
                << "\n";
      printed = true;
    }
  }

  int numBodies = (int)bodies.size();

  // Scratch buffers (static to avoid reallocation)
  static std::vector<int> counts;
  static std::vector<int> offsets;
  static std::vector<SpatialEntry> entries;
  static std::vector<BodyCellRange> ranges;
  static std::vector<glm::vec3> aabb_min;
  static std::vector<glm::vec3> aabb_max;
  // Aligned AABBs for cache-friendly access during detect
  static std::vector<glm::vec3> aligned_aabb_min;
  static std::vector<glm::vec3> aligned_aabb_max;

  if (counts.size() < (size_t)numBodies)
    counts.resize(numBodies);
  if (offsets.size() < (size_t)numBodies + 1)
    offsets.resize(numBodies + 1);
  if (ranges.size() < (size_t)numBodies)
    ranges.resize(numBodies);
  if (aabb_min.size() < (size_t)numBodies)
    aabb_min.resize(numBodies);
  if (aabb_max.size() < (size_t)numBodies)
    aabb_max.resize(numBodies);

  auto t1 = Clock::now();

// 1. Count phase & Compute Ranges
#pragma omp parallel for schedule(static) if (g_thread_limit != 1)
  for (int i = 0; i < numBodies; ++i) {
    glm::vec3 min_pt, max_pt;
    getAABB(bodies[i], min_pt, max_pt);
    aabb_min[i] = min_pt;
    aabb_max[i] = max_pt;
    int min_x = (int)std::floor(min_pt.x / cellSize);
    int min_y = (int)std::floor(min_pt.y / cellSize);
    int min_z = (int)std::floor(min_pt.z / cellSize);
    int max_x = (int)std::floor(max_pt.x / cellSize);
    int max_y = (int)std::floor(max_pt.y / cellSize);
    int max_z = (int)std::floor(max_pt.z / cellSize);

    ranges[i] = {min_x, min_y, min_z, max_x, max_y, max_z};
    counts[i] = (max_x - min_x + 1) * (max_y - min_y + 1) * (max_z - min_z + 1);
  }

  // Actually, for PBC, we need to map absolute cell index to wrapped cell
  // index. Let's compute grid dimensions if PBC is enabled.
  int gridDimX = 0, gridDimY = 0, gridDimZ = 0;
  int minGridX = 0, minGridY = 0, minGridZ = 0;

  if (pbcEnabled_) {
    minGridX = (int)std::floor(pbcMin_.x / cellSize);
    minGridY = (int)std::floor(pbcMin_.y / cellSize);
    minGridZ = (int)std::floor(pbcMin_.z / cellSize);

    int maxGridX = (int)std::floor(pbcMax_.x / cellSize);
    int maxGridY = (int)std::floor(pbcMax_.y / cellSize);
    int maxGridZ = (int)std::floor(pbcMax_.z / cellSize);

    gridDimX = maxGridX - minGridX;
    gridDimY = maxGridY - minGridY;
    gridDimZ = maxGridZ - minGridZ;

    // Ensure at least 1 cell
    if (gridDimX < 1)
      gridDimX = 1;
    if (gridDimY < 1)
      gridDimY = 1;
    if (gridDimZ < 1)
      gridDimZ = 1;
  }

  auto t2 = Clock::now();

  // 2. Prefix sum
  offsets[0] = 0;
  for (int i = 0; i < numBodies; ++i) {
    offsets[i + 1] = offsets[i] + counts[i];
  }
  int totalEntries = offsets[numBodies];

  if (entries.size() < (size_t)totalEntries)
    entries.resize(totalEntries);
  if (aligned_aabb_min.size() < (size_t)totalEntries)
    aligned_aabb_min.resize(totalEntries);
  if (aligned_aabb_max.size() < (size_t)totalEntries)
    aligned_aabb_max.resize(totalEntries);

  auto t3 = Clock::now();

// 3. Fill phase
#pragma omp parallel for schedule(static) if (g_thread_limit != 1)
  for (int i = 0; i < numBodies; ++i) {
    const auto &r = ranges[i];
    int offset = offsets[i];
    for (int x = r.min_x; x <= r.max_x; ++x) {
      for (int y = r.min_y; y <= r.max_y; ++y) {
        for (int z = r.min_z; z <= r.max_z; ++z) {
          int kx = x, ky = y, kz = z;
          if (pbcEnabled_) {
            // Wrap indices to [minGrid, minGrid + dim)
            kx = ((x - minGridX) % gridDimX);
            if (kx < 0)
              kx += gridDimX;
            kx += minGridX;

            ky = ((y - minGridY) % gridDimY);
            if (ky < 0)
              ky += gridDimY;
            ky += minGridY;

            kz = ((z - minGridZ) % gridDimZ);
            if (kz < 0)
              kz += gridDimZ;
            kz += minGridZ;
          }
          entries[offset] = {hashPos64(kx, ky, kz), i};
          // Store AABB in aligned arrays
          aligned_aabb_min[offset] = aabb_min[i];
          aligned_aabb_max[offset] = aabb_max[i];
          offset++;
        }
      }
    }
  }

  auto t4 = Clock::now();

  // 4. Sort
  std::sort(entries.begin(), entries.begin() + totalEntries);

// Re-populate aligned AABBs in sorted order
#pragma omp parallel for schedule(static) if (g_thread_limit != 1)
  for (int i = 0; i < totalEntries; ++i) {
    int bodyIdx = entries[i].bodyIdx;
    aligned_aabb_min[i] = aabb_min[bodyIdx];
    aligned_aabb_max[i] = aabb_max[bodyIdx];
  }

  auto t5 = Clock::now();

  // 5. Detect
  // Identify cells (runs of same key)
  static std::vector<std::pair<int, int>> cells;
  cells.clear();
  if (totalEntries > 0) {
    int start = 0;
    for (int i = 1; i < totalEntries; ++i) {
      if (entries[i].key != entries[start].key) {
        if (i - start > 1)
          cells.push_back({start, i});
        start = i;
      }
    }
    if (totalEntries - start > 1)
      cells.push_back({start, totalEntries});
  }

#ifdef _OPENMP
  int num_threads = omp_get_max_threads();
  static std::vector<std::vector<ContactPrimitive>> thread_contacts;
  if (thread_contacts.size() < (size_t)num_threads)
    thread_contacts.resize(num_threads);
  for (auto &v : thread_contacts)
    v.clear();

#pragma omp parallel for schedule(dynamic) if (g_thread_limit != 1)
  for (int c = 0; c < (int)cells.size(); ++c) {
    int tid = omp_get_thread_num();
    int start = cells[c].first;
    int end = cells[c].second;

    for (int i = start; i < end; ++i) {
      for (int j = i + 1; j < end; ++j) {
        int idx_a = entries[i].bodyIdx;
        int idx_b = entries[j].bodyIdx;

        if (idx_a == idx_b)
          continue;
        if (idx_a > idx_b)
          std::swap(idx_a, idx_b);

        const RigidBody &a = bodies[idx_a];
        const RigidBody &b = bodies[idx_b];

        // Skip pairs where both bodies are fixed.
        if (a.invMass <= 0.0f && b.invMass <= 0.0f)
          continue;

        // Generic AABB check using aligned arrays
        if (config_.use_aabb && !pbcEnabled_) {
          const glm::vec3 &min_a = aligned_aabb_min[i];
          const glm::vec3 &max_a = aligned_aabb_max[i];
          const glm::vec3 &min_b = aligned_aabb_min[j];
          const glm::vec3 &max_b = aligned_aabb_max[j];
          if (max_a.x < min_b.x || min_a.x > max_b.x || max_a.y < min_b.y ||
              min_a.y > max_b.y || max_a.z < min_b.z || min_a.z > max_b.z) {
            continue;
          }
        }

        if (a.type == ShapeType::Capsule && b.type == ShapeType::Capsule) {
          detectCapsuleCapsule(a, b, idx_a, idx_b, thread_contacts[tid]);
        } else if (a.type == ShapeType::Sphere && b.type == ShapeType::Sphere) {
          detectSphereSphere(a, b, idx_a, idx_b, thread_contacts[tid]);
        } else if (a.type == ShapeType::Sphere &&
                   b.type == ShapeType::Capsule) {
          detectSphereCapsule(a, b, idx_a, idx_b, thread_contacts[tid]);
        } else if (a.type == ShapeType::Capsule &&
                   b.type == ShapeType::Sphere) {
          detectSphereCapsule(b, a, idx_b, idx_a, thread_contacts[tid]);
        }
      }
    }
  }

  // Merge
  size_t total_est = 0;
  for (const auto &tc : thread_contacts)
    total_est += tc.size();
  contacts_.reserve(total_est);
  for (const auto &tc : thread_contacts) {
    contacts_.insert(contacts_.end(), tc.begin(), tc.end());
  }
#else
  // Serial fallback
  for (size_t c = 0; c < cells.size(); ++c) {
    int start = cells[c].first;
    int end = cells[c].second;
    for (int i = start; i < end; ++i) {
      for (int j = i + 1; j < end; ++j) {
        int idx_a = entries[i].bodyIdx;
        int idx_b = entries[j].bodyIdx;
        if (idx_a == idx_b)
          continue;
        if (idx_a > idx_b)
          std::swap(idx_a, idx_b);

        const RigidBody &a = bodies[idx_a];
        const RigidBody &b = bodies[idx_b];

        // Generic AABB check
        if (config_.use_aabb) {
          const glm::vec3 &min_a = aabb_min[idx_a];
          const glm::vec3 &max_a = aabb_max[idx_a];
          const glm::vec3 &min_b = aabb_min[idx_b];
          const glm::vec3 &max_b = aabb_max[idx_b];
          if (max_a.x < min_b.x || min_a.x > max_b.x || max_a.y < min_b.y ||
              min_a.y > max_b.y || max_a.z < min_b.z || min_a.z > max_b.z) {
            continue;
          }
        }

        if (a.type == ShapeType::Capsule && b.type == ShapeType::Capsule) {
          detectCapsuleCapsule(a, b, idx_a, idx_b, contacts_);
        } else if (a.type == ShapeType::Sphere && b.type == ShapeType::Sphere) {
          detectSphereSphere(a, b, idx_a, idx_b, contacts_);
        } else if (a.type == ShapeType::Sphere &&
                   b.type == ShapeType::Capsule) {
          detectSphereCapsule(a, b, idx_a, idx_b, contacts_);
        } else if (a.type == ShapeType::Capsule &&
                   b.type == ShapeType::Sphere) {
          detectSphereCapsule(b, a, idx_b, idx_a, contacts_);
        }
      }
    }
  }
#endif

  // Sort and Unique
  if (!contacts_.empty()) {
    std::sort(contacts_.begin(), contacts_.end(),
              [](const ContactPrimitive &a, const ContactPrimitive &b) {
                if (a.body_a != b.body_a)
                  return a.body_a < b.body_a;
                return a.body_b < b.body_b;
              });
    auto last =
        std::unique(contacts_.begin(), contacts_.end(),
                    [](const ContactPrimitive &a, const ContactPrimitive &b) {
                      return a.body_a == b.body_a && a.body_b == b.body_b;
                    });
    contacts_.erase(last, contacts_.end());
  }

  auto t6 = Clock::now();

  stats_.count_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
  stats_.prefix_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
  stats_.fill_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
  stats_.sort_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
  stats_.detect_ms = std::chrono::duration<double, std::milli>(t6 - t5).count();

  if (config_.verbose) {
    std::cout << "[SoftContact] Breakdown: count=" << stats_.count_ms
              << " prefix=" << stats_.prefix_ms << " fill=" << stats_.fill_ms
              << " sort=" << stats_.sort_ms << " detect=" << stats_.detect_ms
              << " cells=" << cells.size() << "\n";
  }
}

void SoftContactSolver::detectCapsuleCapsule(
    const RigidBody &a, const RigidBody &b, int idx_a, int idx_b,
    std::vector<ContactPrimitive> &out_contacts) {
  // Get capsule endpoints
  const glm::vec3 axis_a = a.axisY();
  const glm::vec3 axis_b = b.axisY();

  const glm::vec3 a1 = a.x - axis_a * a.cap.h;
  const glm::vec3 a2 = a.x + axis_a * a.cap.h;
  const glm::vec3 b1_raw = b.x - axis_b * b.cap.h;
  const glm::vec3 b2_raw = b.x + axis_b * b.cap.h;

  glm::vec3 b1 = b1_raw;
  glm::vec3 b2 = b2_raw;
  glm::vec3 shift_b(0.0f);

  if (pbcEnabled_) {
    // Apply Minimum Image Convention
    // Shift B to be closest to A
    glm::vec3 delta = b.x - a.x;
    for (int k = 0; k < 3; ++k) {
      if (pbcSize_[k] > 0.0f) {
        float n = std::floor(delta[k] / pbcSize_[k] + 0.5f);
        shift_b[k] = -n * pbcSize_[k];
      }
    }
    b1 += shift_b;
    b2 += shift_b;
  }

  const double r_a = a.cap.r;
  const double r_b = b.cap.r;
  const double h = r_a + r_b; // Surface limit

  // Find closest points between two line segments
  double s, t;
  closestPointsSegmentSegment(a1, a2, b1, b2, s, t);

  glm::vec3 point_a = a1 + static_cast<float>(s) * (a2 - a1);
  glm::vec3 point_b = b1 + static_cast<float>(t) * (b2 - b1);

  glm::vec3 diff = point_b - point_a;
  double dist_sq = glm::dot(diff, diff);

  // Only create contact if within activation distance
  const double activation_dist = h + config_.delta;
  const double activation_dist_sq = activation_dist * activation_dist;

  // Debug: Print first detection attempt
  static int debug_count = 0;
  if (config_.verbose && debug_count < 5) {
    double distance = std::sqrt(dist_sq);
    std::cout << "[SoftContact::detect] dist=" << distance
              << " activation=" << activation_dist << " h=" << h
              << " delta=" << config_.delta << std::endl;
    debug_count++;
  }

  if (dist_sq < activation_dist_sq) {
    double distance = std::sqrt(dist_sq);
    if (config_.verbose) {
      std::cout << "[SoftContact::detect] CONTACT FOUND: i=" << idx_a
                << " j=" << idx_b << " dist=" << distance
                << " activation=" << activation_dist
                << " delta=" << config_.delta << " sumR=" << h << "\n";
    }
    ContactPrimitive contact;

    // Classify contact type based on parameters
    const double eps = 1e-6;
    if (s < eps && t < eps) {
      contact.type = ContactType::POINT_TO_POINT; // Both at start
    } else if (s > 1.0 - eps && t > 1.0 - eps) {
      contact.type = ContactType::POINT_TO_POINT; // Both at end
    } else if (s < eps || s > 1.0 - eps) {
      contact.type = ContactType::EDGE_TO_POINT; // A is point, B is edge
    } else if (t < eps || t > 1.0 - eps) {
      contact.type = ContactType::EDGE_TO_POINT; // B is point, A is edge
    } else {
      contact.type = ContactType::EDGE_TO_EDGE; // Both on edges
    }

    contact.body_a = idx_a;
    contact.body_b = idx_b;
    contact.point_a = point_a;
    contact.point_b = point_b;
    contact.distance = distance;
    contact.surface_limit = h;
    contact.shift_b = shift_b;

    if (distance > 1e-10) {
      contact.normal = diff / static_cast<float>(distance);
    } else {
      // Fallback for coincident points
      contact.normal = glm::vec3(1, 0, 0);
    }

    out_contacts.push_back(contact);
  }
}

void SoftContactSolver::detectSphereSphere(
    const RigidBody &a, const RigidBody &b, int idx_a, int idx_b,
    std::vector<ContactPrimitive> &out_contacts) {
  // Get sphere centers and radii
  const glm::vec3 &center_a = a.x;
  const double r_a = a.sphere.r;

  glm::vec3 center_b_raw = b.x;
  const double r_b = b.sphere.r;
  const double h = r_a + r_b; // Surface limit

  glm::vec3 center_b = center_b_raw;
  glm::vec3 shift_b(0.0f);

  if (pbcEnabled_) {
    glm::vec3 delta = b.x - a.x;
    for (int k = 0; k < 3; ++k) {
      if (pbcSize_[k] > 0.0f) {
        float n = std::floor(delta[k] / pbcSize_[k] + 0.5f);
        shift_b[k] = -n * pbcSize_[k];
      }
    }
    center_b += shift_b;
  }

  // Compute distance between centers
  glm::vec3 diff = center_b - center_a;
  double dist_sq = glm::dot(diff, diff);

  // Only create contact if within activation distance
  const double activation_dist = h + config_.delta;
  const double activation_dist_sq = activation_dist * activation_dist;

  if (dist_sq < activation_dist_sq) {
    double distance = std::sqrt(dist_sq);
    ContactPrimitive contact;

    // Sphere-sphere is always point-to-point (centers don't matter, contact is
    // on surface)
    contact.type = ContactType::POINT_TO_POINT;
    contact.body_a = idx_a;
    contact.body_b = idx_b;
    contact.shift_b = shift_b;

    // Contact points on each sphere surface
    if (distance > 1e-10) {
      glm::vec3 normal = diff / static_cast<float>(distance);
      contact.point_a = center_a + normal * static_cast<float>(r_a);
      contact.point_b = center_b - normal * static_cast<float>(r_b);
      contact.normal = normal;
    } else {
      // Spheres are at same position - use arbitrary normal
      contact.normal = glm::vec3(1, 0, 0);
      contact.point_a = center_a + contact.normal * static_cast<float>(r_a);
      contact.point_b = center_b - contact.normal * static_cast<float>(r_b);
    }

    contact.distance = distance;
    contact.surface_limit = h;

    out_contacts.push_back(contact);
  }
}

void SoftContactSolver::computeForces(std::vector<RigidBody> &bodies, double dt,
                                      const glm::vec3 &gravity) {
#ifdef _OPENMP
  if (g_thread_limit > 0) {
    omp_set_num_threads(g_thread_limit);
  }
#endif
  // Process each contact and accumulate potential energy
  double pe_sum = 0.0;

  // Update frame counter for history tracking
  frameCounter_++;

#pragma omp parallel for reduction(+:pe_sum) if(contacts_.size() > 64 && g_thread_limit != 1)
  for (int i = 0; i < (int)contacts_.size(); ++i) {
    if (i == 0) {
      reportOpenMpTeamSizeOnce("contact force computation");
    }
    auto &contact = contacts_[i];

    // Compute normal forces based on contact type
    switch (contact.type) {
    case ContactType::POINT_TO_POINT:
      computeP2PForce(contact);
      break;
    case ContactType::EDGE_TO_POINT:
      computeE2PForce(contact);
      break;
    case ContactType::EDGE_TO_EDGE:
      computeE2EForce(contact);
      break;
    }

    // std::cout << "damping coeff: " << config_.damping << std::endl;
    // Damping force (if enabled)
    if (config_.damping > 1e-9) {
      const auto &body_a = bodies[contact.body_a];
      const auto &body_b = bodies[contact.body_b];
      glm::vec3 r_a = contact.point_a - body_a.x;
      glm::vec3 r_b = contact.point_b - (body_b.x + contact.shift_b);
      glm::vec3 v_a = body_a.v + glm::cross(body_a.w, r_a);
      glm::vec3 v_b = body_b.v + glm::cross(body_b.w, r_b);
      glm::vec3 v_rel = v_a - v_b;
      float vn = glm::dot(v_rel, contact.normal);
      glm::vec3 f_damp = -config_.damping * vn * contact.normal;

      contact.force_a += f_damp;
      contact.force_b -= f_damp;
    }

    // Compute friction if enabled
    if (config_.enable_friction) {
      computeFriction(contact, bodies[contact.body_a], bodies[contact.body_b],
                      dt, gravity);
    } else {
      contact.friction_a = glm::vec3(0);
      contact.friction_b = glm::vec3(0);
    }

    // Apply forces to bodies
    auto &body_a = bodies[contact.body_a];
    auto &body_b = bodies[contact.body_b];

    // Total force = normal + friction
    glm::vec3 total_force_a = contact.force_a + contact.friction_a;
    glm::vec3 total_force_b = contact.force_b + contact.friction_b;

    // Compute torques: τ = r × F
    glm::vec3 r_a = contact.point_a - body_a.x;
    // For B, the contact point is in the shifted frame (near A).
    // The lever arm must be relative to the shifted center of B.
    // r_b = point_b - (body_b.x + shift_b)
    glm::vec3 r_b = contact.point_b - (body_b.x + contact.shift_b);

    glm::vec3 torque_a = glm::cross(r_a, total_force_a);
    glm::vec3 torque_b = glm::cross(r_b, total_force_b);

// Apply forces at contact points (generates torques automatically)
// Use atomics for thread safety
#pragma omp atomic
    body_a.f.x += total_force_a.x;
#pragma omp atomic
    body_a.f.y += total_force_a.y;
#pragma omp atomic
    body_a.f.z += total_force_a.z;

#pragma omp atomic
    body_b.f.x += total_force_b.x;
#pragma omp atomic
    body_b.f.y += total_force_b.y;
#pragma omp atomic
    body_b.f.z += total_force_b.z;

#pragma omp atomic
    body_a.tau.x += torque_a.x;
#pragma omp atomic
    body_a.tau.y += torque_a.y;
#pragma omp atomic
    body_a.tau.z += torque_a.z;

#pragma omp atomic
    body_b.tau.x += torque_b.x;
#pragma omp atomic
    body_b.tau.y += torque_b.y;
#pragma omp atomic
    body_b.tau.z += torque_b.z;

    // Accumulate potential energy for this contact (scaled by k_scaler)
    pe_sum += config_.k_scaler *
              potentialEnergy(contact.distance, contact.surface_limit);
  }
  lastPotentialEnergy_ = pe_sum;

  // if (config_.verbose && contacts_.size() > 0) {
  //     std::cout << "[SoftContact] Detected " << contacts_.size() << "
  //     contacts\n"; for (const auto& c : contacts_) {
  //         std::cout << "  Contact " << c.body_a << "-" << c.body_b
  //                   << ": dist=" << c.distance << " limit=" <<
  //                   c.surface_limit << "\n"
  //                   << "    point_a=" << c.point_a.x << "," << c.point_a.y <<
  //                   "," << c.point_a.z << "\n"
  //                   << "    point_b=" << c.point_b.x << "," << c.point_b.y <<
  //                   "," << c.point_b.z << "\n"
  //                   << "    normal=" << c.normal.x << "," << c.normal.y <<
  //                   "," << c.normal.z << "\n"
  //                   << "    force_a=" << c.force_a.x << "," << c.force_a.y <<
  //                   "," << c.force_a.z << "\n"
  //                   << "    force_b=" << c.force_b.x << "," << c.force_b.y <<
  //                   "," << c.force_b.z << "\n";
  //     }
  // }
}

double SoftContactSolver::potentialGradient(double distance, double h) const {
  const double delta = config_.delta;
  const double d = distance;

  // Piecewise gradient of potential energy
  if (d > h - delta) {
    // Non-penetrated regime: smooth log-barrier
    // U = (1/K1 * log(1 + exp(K1*(h-d))))²
    // dU/dd = -2/(K1²) * log(1+exp(arg)) * exp(arg)/(1+exp(arg))

    const double arg = K1_ * (h - d);
    const double exp_arg = std::exp(arg);
    const double log_term = std::log(1.0 + exp_arg);

    return -2.0 / (K1_ * K1_) * log_term * exp_arg / (1.0 + exp_arg);
  } else {
    // Penetrated regime: quadratic penalty
    // U = (h - d)²
    // dU/dd = -2*(h - d)

    return -2.0 * (h - d);
  }
}

double SoftContactSolver::potentialEnergy(double distance, double h) const {
  const double delta = config_.delta;
  const double d = distance;
  if (d > h - delta) {
    // Non-penetrated (barrier) regime
    const double arg = K1_ * (h - d);
    const double log_term = std::log(1.0 + std::exp(arg));
    double base = (log_term / K1_);
    return base * base; // (1/K1*log(1+exp(K1*(h-d))))^2
  } else {
    // Penetrated (quadratic) regime
    double pen = (h - d);
    return pen * pen; // (h-d)^2
  }
}

void SoftContactSolver::computeP2PForce(ContactPrimitive &contact) {
  // Point-to-point: simplest case
  // Force acts along line connecting the two points

  const double grad =
      potentialGradient(contact.distance, contact.surface_limit);
  const float force_mag =
      static_cast<float>(-config_.k_scaler * grad); // F = -k * dU/dd

  // Force direction: gradient points from A to B
  // dU/dx_b = (dU/dd) * (dd/dx_b) = grad * normal
  // dU/dx_a = (dU/dd) * (dd/dx_a) = grad * (-normal)

  contact.force_a = -force_mag * contact.normal; // Push A away from B
  contact.force_b = force_mag * contact.normal;  // Push B away from A
}

void SoftContactSolver::computeE2PForce(ContactPrimitive &contact) {
  // Edge-to-point: one body at endpoint, other on edge
  // For now, use same as P2P (simplified)
  // TODO: Implement proper E2P gradient distribution along edge

  computeP2PForce(contact);
}

void SoftContactSolver::computeE2EForce(ContactPrimitive &contact) {
  // Edge-to-edge: both bodies on interior of segments
  // For now, use same as P2P (simplified)
  // TODO: Implement proper E2E gradient with distributed forces

  computeP2PForce(contact);
}

void SoftContactSolver::computeFriction(ContactPrimitive &contact,
                                        const RigidBody &body_a,
                                        const RigidBody &body_b, double dt,
                                        const glm::vec3 &gravity) {
  // Compute velocities at contact points
  glm::vec3 r_a = contact.point_a - body_a.x;
  glm::vec3 r_b = contact.point_b - (body_b.x + contact.shift_b);

  glm::vec3 v_a = body_a.v + glm::cross(body_a.w, r_a);
  glm::vec3 v_b = body_b.v + glm::cross(body_b.w, r_b);

  glm::vec3 v_rel = v_a - v_b;

  // Normal force magnitude (available from earlier compute*Force calls)
  double fn_mag = glm::length(contact.force_a);

  // Project out normal component to get tangential velocity
  glm::vec3 v_tan = v_rel - glm::dot(v_rel, contact.normal) * contact.normal;
  float v_tan_mag = glm::length(v_tan);

  if (v_tan_mag < 1e-10) {
    // No tangential motion → no friction
    contact.friction_a = glm::vec3(0);
    contact.friction_b = glm::vec3(0);
    return;
  }

  // Karnopp Stick-Slip Model
  // --- Friction Models ---
  if (config_.friction_cundall) {
    // Cundall-Strack: Incremental history-dependent friction
    // 1. Get/Create history entry
    uint64_t key = pairKey(contact.body_a, contact.body_b);

    // Check if we already have this contact in history in a thread-safe way?
    // Wait, this is called inside an OMP loop. unordered_map access is NOT
    // thread safe. We MUST use a critical section or lock.

    glm::vec3 ft_accum(0.0f);

#pragma omp critical(cundall_history)
    {
      CSEntry &entry = contactHistory_[key];
      // If new or re-connected, we might start with previous value or 0.
      // Cundall assumes persistence.
      ft_accum = entry.tangential_force;

      // Update tangential spring
      // dF_t = -k_t * v_tan * dt
      ft_accum -= static_cast<float>(config_.kt * dt) * v_tan;

      // Enforce orthogonality to current normal (project onto tangent plane)
      // F_t_new = F_t_old - (F_t_old . n) * n
      // This accounts for the contact plane rotating.
      float normal_comp = glm::dot(ft_accum, contact.normal);
      ft_accum -= normal_comp * contact.normal;

      // Coulomb Limit
      float ft_mag = glm::length(ft_accum);
      float limit = static_cast<float>(config_.mu * fn_mag);
      if (ft_mag > limit) {
        if (ft_mag > 1e-8f) {
          ft_accum *= (limit / ft_mag);
        } else {
          ft_accum = glm::vec3(0.0f);
        }
      }

      // Write back and touch frame
      entry.tangential_force = ft_accum;
      entry.last_frame = frameCounter_;
    }

    // Apply the computed spring force
    contact.friction_a = ft_accum;
    contact.friction_b = -ft_accum;

    // Only used for logging if needed?
    return;
  }

  if (config_.friction_karnopp) {
    // If below velocity deadband, we are in "stick" or "pre-stick" state
    if (v_tan_mag < config_.vel_deadband) {
      // Need to find force required to stop the relative tangential motion in
      // one step.
      // 1. Force to kill current velocity: F_damping = -m * v_tan / dt
      // 2. Force to cancel external forces: F_static = -F_ext_tan
      //
      // Estimation of F_ext_tan:
      // We assume Gravity is the main external driver in common stick-slip
      // cases. We also check the body's accumulated force so far (which
      // includes random forces and previous contacts).
      // Note: body_a.f is the accumulated force on A *so far* in this step.
      // Threading note: this read is racy if we are updating body_a.f in
      // parallel, but typically we read specific fields or accept the
      // approximation. However, standard penalty methods accumulate.
      // A better stable approximation for F_ext is mostly Gravity + GravityB
      // (relative?)
      // Stick relies on relative motion.

      // Effective mass for contact point (simplified as reduced mass or just
      // local body masses) For simplicity, use body A's mass if A is dynamic,
      // B's if B is dynamic. Reduced mass m* = (ma*mb)/(ma+mb).
      float ma = body_a.invMass > 0 ? body_a.mass : 0.0f;
      float mb = body_b.invMass > 0 ? body_b.mass : 0.0f;
      float m_eff = 0.0f;
      if (ma > 0 && mb > 0)
        m_eff = (ma * mb) / (ma + mb);
      else if (ma > 0)
        m_eff = ma;
      else if (mb > 0)
        m_eff = mb;

      // Force to stop velocity:
      glm::vec3 F_stop = -(float)(m_eff / dt) * v_tan;

      // Estimate external force tangential component
      // F_ext_rel = F_ext_a/ma - F_ext_b/mb (acceleration difference) * m_eff
      // Simplified: Just use Gravity projected on tangent.
      // (Gravity is the same for both, so it tends to cancel in relative acc
      // UNLESS one is on a slope and the other is floor. But floor has 0 grav
      // effect if static?)
      // If one body is static floor (B), relative acc from gravity is g
      // (down). Tangent component of g drives sliding.
      // F_drive = m_eff * g_tan.
      // We want to oppose F_drive.

      glm::vec3 g_vec = gravity;
      glm::vec3 g_tan =
          g_vec - glm::dot(g_vec, contact.normal) * contact.normal;
      glm::vec3 F_drive = m_eff * g_tan;

      // Total required friction to HOLD: F_req = -F_drive + F_stop
      // (F_stop handles the inertial drift, -F_drive handles the gravity push)
      glm::vec3 F_req = -F_drive + F_stop;

      float F_req_mag = glm::length(F_req);
      // float fn_mag = glm::length(contact.force_a); // Already defined
      float max_static_friction = config_.mu_static * fn_mag;

      if (F_req_mag <= max_static_friction) {
        // STICK: We can generate enough force to hold/stop.
        // Apply exactly F_req.
        contact.friction_a = F_req;
        contact.friction_b = -F_req;
        return;
      } else {
        // SLIP (Breakaway): Cannot hold.
        // Apply max static friction opposing the required direction?
        // Or transition to dynamic?
        // Karnopp usually caps at static limit during the breakaway frame.
        if (F_req_mag > 1e-12) {
          contact.friction_a = F_req * (max_static_friction / F_req_mag);
          contact.friction_b = -contact.friction_a;
        }
        return;
      }
    } else {
      // Nominal Sliding (Slip state)
      // Use kinetic friction
      // float fn_mag = glm::length(contact.force_a); // Already defined
      glm::vec3 dir = -v_tan / v_tan_mag; // Oppose motion
      float fric_mag = config_.mu * fn_mag;
      contact.friction_a = dir * fric_mag;
      contact.friction_b = -contact.friction_a;
      return;
    }
  }

  // Smooth friction coefficient (sticking-to-sliding transition)
  double gamma;
  if (v_tan_mag > config_.nu) {
    gamma = 1.0; // Sliding regime
  } else {
    // Sticking regime: smooth transition using tanh-like function
    gamma = 2.0 / (1.0 + std::exp(-K2_ * v_tan_mag)) - 1.0;
  }

  // Effective friction coefficient (Stribeck effect for stick-slip)
  double mu_eff = config_.mu;
  if (config_.mu_static > config_.mu) {
    // Blend from mu_static (at v=0) to mu_dynamic (at v>>nu)
    // Decay scale: we want the static peak to be effective within the sticking
    // range. Using nu as the decay constant ensures smooth transition.
    double decay = std::exp(-v_tan_mag / config_.nu);
    mu_eff = config_.mu + (config_.mu_static - config_.mu) * decay;
  }

  // Friction force magnitude: μ_eff * γ * |F_normal|
  // const float fn_mag = glm::length(contact.force_a); // Already defined
  const glm::vec3 friction_dir = -v_tan / v_tan_mag; // Oppose tangential motion

  const float friction_mag = static_cast<float>(mu_eff * gamma) * fn_mag;
  const glm::vec3 friction_force = friction_mag * friction_dir;

  contact.friction_a = friction_force;
  contact.friction_b = -friction_force;
}

// Geometric helper: Closest points on two line segments (Lumelsky algorithm
// from DisMech)
void SoftContactSolver::closestPointsSegmentSegment(const glm::vec3 &a1,
                                                    const glm::vec3 &a2,
                                                    const glm::vec3 &b1,
                                                    const glm::vec3 &b2,
                                                    double &s, double &t) {
  // Same algorithm as DisMech for consistency
  const glm::vec3 e1 = a2 - a1;
  const glm::vec3 e2 = b2 - b1;
  const glm::vec3 e12 = b1 - a1;

  const double D1 = glm::dot(e1, e1);
  const double D2 = glm::dot(e2, e2);
  const double S1 = glm::dot(e1, e12);
  const double S2 = glm::dot(e2, e12);
  const double R = glm::dot(e1, e2);

  const double den = D1 * D2 - R * R;

  double uf = 0.0;

  auto fixBound = [](double &x) -> bool {
    if (x > 1.0) {
      x = 1.0;
      return true;
    } else if (x < 0.0) {
      x = 0.0;
      return true;
    }
    return false;
  };

  if (den == 0.0) {
    // Parallel segments
    s = 0.0;
    t = -S2 / D2;
    uf = t;
    fixBound(uf);

    if (uf != t) {
      s = (uf * R + S1) / D1;
      fixBound(s);
      t = uf;
    }
  } else {
    // General case
    s = (S1 * D2 - S2 * R) / den;
    fixBound(s);
    t = (s * R - S2) / D2;
    uf = t;
    fixBound(uf);

    if (uf != t) {
      s = (uf * R + S1) / D1;
      fixBound(s);
      t = uf;
    }
  }
}

double SoftContactSolver::distanceSqPointToSegment(const glm::vec3 &point,
                                                   const glm::vec3 &seg_start,
                                                   const glm::vec3 &seg_end,
                                                   double &t) {
  const glm::vec3 d = seg_end - seg_start;
  const glm::vec3 r = point - seg_start;

  const double len_sq = glm::dot(d, d);
  if (len_sq < 1e-10) {
    t = 0.0;
    return glm::dot(r, r);
  }

  t = glm::clamp(glm::dot(r, d) / len_sq, 0.0, 1.0);
  const glm::vec3 closest = seg_start + static_cast<float>(t) * d;

  const glm::vec3 diff = point - closest;
  return glm::dot(diff, diff);
}

void SoftContactSolver::detectSphereCapsule(
    const RigidBody &sphere, const RigidBody &capsule, int idx_sphere,
    int idx_capsule, std::vector<ContactPrimitive> &out_contacts) {
  // Sphere properties
  const glm::vec3 &center = sphere.x;
  const double r_s = sphere.sphere.r;

  // Capsule properties
  const glm::vec3 axis = capsule.axisY();
  const glm::vec3 p1 = capsule.x - axis * capsule.cap.h;
  const glm::vec3 p2 = capsule.x + axis * capsule.cap.h;
  const double r_c = capsule.cap.r;

  const double h = r_s + r_c;

  // Find closest point on capsule segment to sphere center
  double t;
  double dist_sq = distanceSqPointToSegment(center, p1, p2, t);

  const double activation_dist = h + config_.delta;
  const double activation_sq = activation_dist * activation_dist;

  if (dist_sq < activation_sq) {
    ContactPrimitive contact;
    double distance = std::sqrt(dist_sq);

    // Determine contact type
    const double eps = 1e-6;
    if (t < eps || t > 1.0 - eps) {
      contact.type = ContactType::POINT_TO_POINT; // Hitting end-cap
    } else {
      contact.type = ContactType::EDGE_TO_POINT; // Hitting cylinder side
    }

    contact.body_a = idx_sphere;
    contact.body_b = idx_capsule;

    glm::vec3 closest_on_seg = p1 + static_cast<float>(t) * (p2 - p1);

    if (distance > 1e-10) {
      contact.normal = (center - closest_on_seg) / static_cast<float>(distance);
    } else {
      contact.normal = glm::vec3(1, 0, 0); // Arbitrary fallback
    }

    // Point on sphere (A)
    contact.point_a = center - contact.normal * static_cast<float>(r_s);

    // Point on capsule (B)
    contact.point_b = closest_on_seg + contact.normal * static_cast<float>(r_c);

    contact.distance = distance;
    contact.surface_limit = h;

    out_contacts.push_back(contact);
  }
}
