/**
 * @file soft_contact.cpp
 * @brief Implementation of soft penalty-based contact solver
 */

#include "physics/soft_contact.hpp"
#include "physics/rigid_body.hpp"
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

extern int g_thread_limit;

// Aligned buffer to prevent false sharing between threads
struct alignas(64) ThreadContactBuffer {
    std::vector<ContactPrimitive> contacts;
};

SoftContactSolver::SoftContactSolver(const SoftContactCfg& config)
    : config_(config)
{
    K1_ = 15.0 / config_.delta;
    K2_ = 15.0 / config_.nu;
    lastPotentialEnergy_ = 0.0;
}

void SoftContactSolver::setConfig(const SoftContactCfg& config) {
    config_ = config;
    K1_ = 15.0 / config_.delta;
    K2_ = 15.0 / config_.nu;
    // Reset accumulator on config change (avoids stale reporting)
    lastPotentialEnergy_ = 0.0;
}

void SoftContactSolver::detectContacts(const std::vector<RigidBody>& bodies) {
    contacts_.clear();
    
    // Debug: check which solver is used
    static bool first = true;
    if (first) {
        // fprintf(stderr, "[Debug] use_spatial_hash=%d\n", config_.use_spatial_hash);
        first = false;
    }

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

void SoftContactSolver::detectContactsNaive(const std::vector<RigidBody>& bodies) {
    // Broadphase: All pairs (O(n²) - fine for small number of objects)
#ifdef _OPENMP
    int num_threads = (g_thread_limit > 0) ? g_thread_limit : omp_get_max_threads();
    std::vector<ThreadContactBuffer> thread_contacts(num_threads);
    // Reserve some space to avoid frequent reallocations
    for(auto& v : thread_contacts) v.contacts.reserve(100);

    #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (size_t i = 0; i < bodies.size(); ++i) {
        int tid = omp_get_thread_num();
        for (size_t j = i + 1; j < bodies.size(); ++j) {
            const RigidBody& a = bodies[i];
            const RigidBody& b = bodies[j];
            
            // Dispatch based on shape types
            if (a.type == ShapeType::Capsule && b.type == ShapeType::Capsule) {
                detectCapsuleCapsule(a, b, i, j, thread_contacts[tid].contacts);
            } else if (a.type == ShapeType::Sphere && b.type == ShapeType::Sphere) {
                detectSphereSphere(a, b, i, j, thread_contacts[tid].contacts);
            } else if (a.type == ShapeType::Sphere && b.type == ShapeType::Capsule) {
                detectSphereCapsule(a, b, i, j, thread_contacts[tid].contacts);
            } else if (a.type == ShapeType::Capsule && b.type == ShapeType::Sphere) {
                detectSphereCapsule(b, a, j, i, thread_contacts[tid].contacts);
            }
        }
    }

    // Merge results
    for (const auto& tc : thread_contacts) {
        contacts_.insert(contacts_.end(), tc.contacts.begin(), tc.contacts.end());
    }
#else
    for (size_t i = 0; i < bodies.size(); ++i) {
        for (size_t j = i + 1; j < bodies.size(); ++j) {
            const RigidBody& a = bodies[i];
            const RigidBody& b = bodies[j];
            
            // Dispatch based on shape types
            if (a.type == ShapeType::Capsule && b.type == ShapeType::Capsule) {
                detectCapsuleCapsule(a, b, i, j, contacts_);
            } else if (a.type == ShapeType::Sphere && b.type == ShapeType::Sphere) {
                detectSphereSphere(a, b, i, j, contacts_);
            } else if (a.type == ShapeType::Sphere && b.type == ShapeType::Capsule) {
                detectSphereCapsule(a, b, i, j, contacts_);
            } else if (a.type == ShapeType::Capsule && b.type == ShapeType::Sphere) {
                detectSphereCapsule(b, a, j, i, contacts_);
            }
        }
    }
#endif
}

double SoftContactSolver::computeAdaptiveCellSize(const std::vector<RigidBody>& bodies) const {
    if (bodies.empty()) return 1.0;
    
    double max_dim = 0.0;
    double sum_dim = 0.0;
    int count = 0;
    
    // Sample a subset if too many bodies
    int step = (bodies.size() > 1000) ? (bodies.size() / 100) : 1;
    
    for (size_t i = 0; i < bodies.size(); i += step) {
        const auto& b = bodies[i];
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
    
    // Heuristic: cell size should be larger than average object, but not too large
    // If objects are very different in size, max_dim is safer to avoid checking too many neighbors
    // But for rods, they are long. A cell size of ~length is good.
    return std::max(max_dim, avg_dim * 1.5);
}

void SoftContactSolver::getAABB(const RigidBody& b, glm::vec3& min_pt, glm::vec3& max_pt) const {
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

bool SoftContactSolver::checkAABBOverlap(const RigidBody& a, const RigidBody& b) const {
    glm::vec3 min_a, max_a;
    getAABB(a, min_a, max_a);
    
    glm::vec3 min_b, max_b;
    getAABB(b, min_b, max_b);
    
    if (max_a.x < min_b.x || min_a.x > max_b.x) return false;
    if (max_a.y < min_b.y || min_a.y > max_b.y) return false;
    if (max_a.z < min_b.z || min_a.z > max_b.z) return false;
    
    return true;
}

void SoftContactSolver::insertBodyIntoGrid(int bodyIdx, const RigidBody& body, double cellSize, GridMap& grid) {
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
    bool operator<(const SpatialEntry& other) const {
        if (key != other.key) return key < other.key;
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

void SoftContactSolver::detectContactsSpatialHash(const std::vector<RigidBody>& bodies) {
#ifdef _OPENMP
    int num_threads = (g_thread_limit > 0) ? g_thread_limit : omp_get_max_threads();
    static bool printed = false;
    if (!printed) {
        // std::cout << "[Debug] OpenMP enabled. Max threads: " << omp_get_max_threads() 
        //           << " Limit: " << g_thread_limit 
        //           << " Using: " << num_threads << "\n";
        printed = true;
    }
#else
    static bool printed = false;
    if (!printed) {
        // std::cout << "[Debug] OpenMP DISABLED.\n";
        printed = true;
    }
#endif
    if (bodies.empty()) return;
    
    double cellSize = config_.cell_size;
    if (cellSize <= 0.0) {
        cellSize = computeAdaptiveCellSize(bodies);
    }
    
    int numBodies = (int)bodies.size();
    
    // Scratch buffers (static to avoid reallocation)
    static std::vector<int> counts;
    static std::vector<int> offsets;
    static std::vector<SpatialEntry> entries;
    static std::vector<BodyCellRange> ranges;
    static std::vector<glm::vec3> aabb_min;
    static std::vector<glm::vec3> aabb_max;
    
    if (counts.size() < (size_t)numBodies) counts.resize(numBodies);
    if (offsets.size() < (size_t)numBodies + 1) offsets.resize(numBodies + 1);
    if (ranges.size() < (size_t)numBodies) ranges.resize(numBodies);
    if (aabb_min.size() < (size_t)numBodies) aabb_min.resize(numBodies);
    if (aabb_max.size() < (size_t)numBodies) aabb_max.resize(numBodies);

    // Task generation structures
    struct ContactTask {
        int start;
        int end;
        int i_idx; 
    };
    static std::vector<ContactTask> tasks;
    static std::vector<std::pair<int, int>> cells;
    static std::vector<ThreadContactBuffer> thread_contacts;

#ifdef _OPENMP
    #pragma omp parallel num_threads(num_threads)
    {
        // 1. Count phase & Compute Ranges
        #pragma omp for schedule(static)
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

        #pragma omp single
        {
            // 2. Prefix sum
            offsets[0] = 0;
            for (int i = 0; i < numBodies; ++i) {
                offsets[i+1] = offsets[i] + counts[i];
            }
            int totalEntries = offsets[numBodies];
            if (entries.size() < (size_t)totalEntries) entries.resize(totalEntries);
        }

        // 3. Fill phase
        #pragma omp for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            const auto& r = ranges[i];
            int offset = offsets[i];
            for (int x = r.min_x; x <= r.max_x; ++x) {
                for (int y = r.min_y; y <= r.max_y; ++y) {
                    for (int z = r.min_z; z <= r.max_z; ++z) {
                        entries[offset++] = { hashPos64(x,y,z), i };
                    }
                }
            }
        }

        #pragma omp single
        {
            // 4. Sort
            int totalEntries = offsets[numBodies];
            std::sort(entries.begin(), entries.begin() + totalEntries);

            // Identify cells
            cells.clear();
            if (totalEntries > 0) {
                int start = 0;
                for (int i = 1; i < totalEntries; ++i) {
                    if (entries[i].key != entries[start].key) {
                        if (i - start > 1) cells.push_back({start, i});
                        start = i;
                    }
                }
                if (totalEntries - start > 1) cells.push_back({start, totalEntries});
            }

            // Generate tasks
            tasks.clear();
            const int SPLIT_THRESHOLD = 50; 
            for (const auto& cell : cells) {
                int start = cell.first;
                int end = cell.second;
                int count = end - start;
                if (count > SPLIT_THRESHOLD) {
                    for (int i = start; i < end; ++i) tasks.push_back({start, end, i});
                } else {
                    tasks.push_back({start, end, -1});
                }
            }
            
            // Resize thread buffers
            if (thread_contacts.size() < (size_t)num_threads) {
                thread_contacts.resize(num_threads);
            }
        }

        // Clear thread buffers
        int tid = omp_get_thread_num();
        if (tid < (int)thread_contacts.size()) {
            thread_contacts[tid].contacts.clear();
            if (thread_contacts[tid].contacts.capacity() < 50000) {
                 thread_contacts[tid].contacts.reserve(50000);
            }
        }

        // 5. Detect
        #pragma omp for schedule(dynamic, 10)
        for (size_t t = 0; t < tasks.size(); ++t) {
            int tid = omp_get_thread_num();
            if (tid >= (int)thread_contacts.size()) continue;

            const auto& task = tasks[t];
            int start = task.start;
            int end = task.end;
            
            int i_start, i_end;
            if (task.i_idx == -1) {
                i_start = start;
                i_end = end;
            } else {
                i_start = task.i_idx;
                i_end = task.i_idx + 1;
            }

            for (int i = i_start; i < i_end; ++i) {
                for (int j = i + 1; j < end; ++j) {
                    int idx_a = entries[i].bodyIdx;
                    int idx_b = entries[j].bodyIdx;
                    
                    if (idx_a == idx_b) continue;
                    if (idx_a > idx_b) std::swap(idx_a, idx_b);
                    
                    const RigidBody& a = bodies[idx_a];
                    const RigidBody& b = bodies[idx_b];
                    
                    if (config_.use_aabb) {
                        const glm::vec3& min_a = aabb_min[idx_a];
                        const glm::vec3& max_a = aabb_max[idx_a];
                        const glm::vec3& min_b = aabb_min[idx_b];
                        const glm::vec3& max_b = aabb_max[idx_b];
                        if (max_a.x < min_b.x || min_a.x > max_b.x ||
                            max_a.y < min_b.y || min_a.y > max_b.y ||
                            max_a.z < min_b.z || min_a.z > max_b.z) {
                            continue;
                        }
                    }
                    
                    if (a.type == ShapeType::Capsule && b.type == ShapeType::Capsule) {
                        detectCapsuleCapsule(a, b, idx_a, idx_b, thread_contacts[tid].contacts);
                    } else if (a.type == ShapeType::Sphere && b.type == ShapeType::Sphere) {
                        detectSphereSphere(a, b, idx_a, idx_b, thread_contacts[tid].contacts);
                    } else if (a.type == ShapeType::Sphere && b.type == ShapeType::Capsule) {
                        detectSphereCapsule(a, b, idx_a, idx_b, thread_contacts[tid].contacts);
                    } else if (a.type == ShapeType::Capsule && b.type == ShapeType::Sphere) {
                        detectSphereCapsule(b, a, idx_b, idx_a, thread_contacts[tid].contacts);
                    }
                }
            }
        }
        
        #pragma omp single
        {
            // t_detect_end = Clock::now();
        }
    } // End parallel

    // Merge
    size_t total_est = 0;
    for (const auto& tc : thread_contacts) total_est += tc.contacts.size();
    contacts_.reserve(total_est);
    for (const auto& tc : thread_contacts) {
        contacts_.insert(contacts_.end(), tc.contacts.begin(), tc.contacts.end());
    }

#else
    // Serial fallback
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
    
    offsets[0] = 0;
    for (int i = 0; i < numBodies; ++i) offsets[i+1] = offsets[i] + counts[i];
    int totalEntries = offsets[numBodies];
    if (entries.size() < (size_t)totalEntries) entries.resize(totalEntries);
    
    for (int i = 0; i < numBodies; ++i) {
        const auto& r = ranges[i];
        int offset = offsets[i];
        for (int x = r.min_x; x <= r.max_x; ++x) {
            for (int y = r.min_y; y <= r.max_y; ++y) {
                for (int z = r.min_z; z <= r.max_z; ++z) {
                    entries[offset++] = { hashPos64(x,y,z), i };
                }
            }
        }
    }
    
    std::sort(entries.begin(), entries.begin() + totalEntries);
    
    cells.clear();
    if (totalEntries > 0) {
        int start = 0;
        for (int i = 1; i < totalEntries; ++i) {
            if (entries[i].key != entries[start].key) {
                if (i - start > 1) cells.push_back({start, i});
                start = i;
            }
        }
        if (totalEntries - start > 1) cells.push_back({start, totalEntries});
    }
    
    for (const auto& cell : cells) {
        int start = cell.first;
        int end = cell.second;
        for (int i = start; i < end; ++i) {
            for (int j = i + 1; j < end; ++j) {
                int idx_a = entries[i].bodyIdx;
                int idx_b = entries[j].bodyIdx;
                if (idx_a == idx_b) continue;
                if (idx_a > idx_b) std::swap(idx_a, idx_b);
                const RigidBody& a = bodies[idx_a];
                const RigidBody& b = bodies[idx_b];
                
                if (config_.use_aabb) {
                    const glm::vec3& min_a = aabb_min[idx_a];
                    const glm::vec3& max_a = aabb_max[idx_a];
                    const glm::vec3& min_b = aabb_min[idx_b];
                    const glm::vec3& max_b = aabb_max[idx_b];
                    if (max_a.x < min_b.x || min_a.x > max_b.x ||
                        max_a.y < min_b.y || min_a.y > max_b.y ||
                        max_a.z < min_b.z || min_a.z > max_b.z) continue;
                }
                
                if (a.type == ShapeType::Capsule && b.type == ShapeType::Capsule) {
                    detectCapsuleCapsule(a, b, idx_a, idx_b, contacts_);
                } else if (a.type == ShapeType::Sphere && b.type == ShapeType::Sphere) {
                    detectSphereSphere(a, b, idx_a, idx_b, contacts_);
                } else if (a.type == ShapeType::Sphere && b.type == ShapeType::Capsule) {
                    detectSphereCapsule(a, b, idx_a, idx_b, contacts_);
                } else if (a.type == ShapeType::Capsule && b.type == ShapeType::Sphere) {
                    detectSphereCapsule(b, a, idx_b, idx_a, contacts_);
                }
            }
        }
    }
#endif

    // Sort and Unique
    if (!contacts_.empty()) {
        std::sort(contacts_.begin(), contacts_.end(), [](const ContactPrimitive& a, const ContactPrimitive& b) {
            if (a.body_a != b.body_a) return a.body_a < b.body_a;
            return a.body_b < b.body_b;
        });
        auto last = std::unique(contacts_.begin(), contacts_.end(), [](const ContactPrimitive& a, const ContactPrimitive& b) {
            return a.body_a == b.body_a && a.body_b == b.body_b;
        });
        contacts_.erase(last, contacts_.end());
    }
}

void SoftContactSolver::detectCapsuleCapsule(const RigidBody& a, const RigidBody& b,
                                              int idx_a, int idx_b, std::vector<ContactPrimitive>& out_contacts) {
    // Get capsule endpoints
    const glm::vec3 axis_a = a.axisY();
    const glm::vec3 axis_b = b.axisY();
    
    const glm::vec3 a1 = a.x - axis_a * a.cap.h;
    const glm::vec3 a2 = a.x + axis_a * a.cap.h;
    const glm::vec3 b1 = b.x - axis_b * b.cap.h;
    const glm::vec3 b2 = b.x + axis_b * b.cap.h;
    
    const double r_a = a.cap.r;
    const double r_b = b.cap.r;
    const double h = r_a + r_b;  // Surface limit
    
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
                  << " activation=" << activation_dist 
                  << " h=" << h << " delta=" << config_.delta << std::endl;
        debug_count++;
    }
    
    if (dist_sq < activation_dist_sq) {
        double distance = std::sqrt(dist_sq);
        if (config_.verbose) {
             std::cout << "[SoftContact::detect] CONTACT FOUND: i=" << idx_a << " j=" << idx_b 
                       << " dist=" << distance << " activation=" << activation_dist 
                       << " delta=" << config_.delta << " sumR=" << h << "\n";
        }
        ContactPrimitive contact;
        
        // Classify contact type based on parameters
        const double eps = 1e-6;
        if (s < eps && t < eps) {
            contact.type = ContactType::POINT_TO_POINT;  // Both at start
        } else if (s > 1.0 - eps && t > 1.0 - eps) {
            contact.type = ContactType::POINT_TO_POINT;  // Both at end
        } else if (s < eps || s > 1.0 - eps) {
            contact.type = ContactType::EDGE_TO_POINT;   // A is point, B is edge
        } else if (t < eps || t > 1.0 - eps) {
            contact.type = ContactType::EDGE_TO_POINT;   // B is point, A is edge
        } else {
            contact.type = ContactType::EDGE_TO_EDGE;    // Both on edges
        }
        
        contact.body_a = idx_a;
        contact.body_b = idx_b;
        contact.point_a = point_a;
        contact.point_b = point_b;
        contact.distance = distance;
        contact.surface_limit = h;
        
        if (distance > 1e-10) {
            contact.normal = diff / static_cast<float>(distance);
        } else {
            // Fallback for coincident points
            contact.normal = glm::vec3(1, 0, 0);
        }
        
        out_contacts.push_back(contact);
    }
}

void SoftContactSolver::detectSphereSphere(const RigidBody& a, const RigidBody& b,
                                           int idx_a, int idx_b, std::vector<ContactPrimitive>& out_contacts) {
    // Get sphere centers and radii
    const glm::vec3& center_a = a.x;
    const glm::vec3& center_b = b.x;
    const double r_a = a.sphere.r;
    const double r_b = b.sphere.r;
    const double h = r_a + r_b;  // Surface limit
    
    // Compute distance between centers
    glm::vec3 diff = center_b - center_a;
    double dist_sq = glm::dot(diff, diff);
    
    // Only create contact if within activation distance
    const double activation_dist = h + config_.delta;
    const double activation_dist_sq = activation_dist * activation_dist;
    
    if (dist_sq < activation_dist_sq) {
        double distance = std::sqrt(dist_sq);
        ContactPrimitive contact;
        
        // Sphere-sphere is always point-to-point (centers don't matter, contact is on surface)
        contact.type = ContactType::POINT_TO_POINT;
        contact.body_a = idx_a;
        contact.body_b = idx_b;
        
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

void SoftContactSolver::computeForces(std::vector<RigidBody>& bodies, double dt) {
    struct BodyForce {
        glm::vec3 f;
        glm::vec3 tau;
    };
    
    int num_threads = 1;
#ifdef _OPENMP
    // if (g_thread_limit > 0) omp_set_num_threads(g_thread_limit);
    num_threads = omp_get_max_threads();
    if (g_thread_limit > 0) num_threads = g_thread_limit;
#endif

    // Thread-local accumulators
    static std::vector<std::vector<BodyForce>> thread_forces;
    
    if (thread_forces.size() < (size_t)num_threads) {
        thread_forces.resize(num_threads);
    }
    
    // Resize and clear buffers
    #pragma omp parallel num_threads(num_threads)
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        if (tid < (int)thread_forces.size()) {
            if (thread_forces[tid].size() < bodies.size()) {
                thread_forces[tid].resize(bodies.size());
            }
            // Clear forces for this step
            std::memset(thread_forces[tid].data(), 0, bodies.size() * sizeof(BodyForce));
        }
    }

    // Process each contact and accumulate potential energy
    double pe_sum = 0.0;
    
    #pragma omp parallel for reduction(+:pe_sum) schedule(static) num_threads(num_threads)
    for (size_t i = 0; i < contacts_.size(); ++i) {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        if (tid >= (int)thread_forces.size()) continue;

        auto& contact = contacts_[i];
        
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
        
        // Compute friction if enabled
        if (config_.enable_friction) {
            computeFriction(contact, bodies[contact.body_a], bodies[contact.body_b], dt);
        } else {
            contact.friction_a = glm::vec3(0);
            contact.friction_b = glm::vec3(0);
        }
        
        // Total force = normal + friction
        glm::vec3 total_force_a = contact.force_a + contact.friction_a;
        glm::vec3 total_force_b = contact.force_b + contact.friction_b;
        
        // Compute torques: τ = r × F
        glm::vec3 r_a = contact.point_a - bodies[contact.body_a].x;
        glm::vec3 r_b = contact.point_b - bodies[contact.body_b].x;
        
        glm::vec3 torque_a = glm::cross(r_a, total_force_a);
        glm::vec3 torque_b = glm::cross(r_b, total_force_b);

        // Accumulate to thread-local storage (no atomics!)
        auto& bf_a = thread_forces[tid][contact.body_a];
        auto& bf_b = thread_forces[tid][contact.body_b];
        
        bf_a.f += total_force_a;
        bf_a.tau += torque_a;
        bf_b.f += total_force_b;
        bf_b.tau += torque_b;

        // Accumulate potential energy for this contact (scaled by k_scaler)
        pe_sum += config_.k_scaler * potentialEnergy(contact.distance, contact.surface_limit);
    }
    lastPotentialEnergy_ = pe_sum;
    
    // Merge thread-local forces into global bodies
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (size_t i = 0; i < bodies.size(); ++i) {
        glm::vec3 f_sum(0);
        glm::vec3 tau_sum(0);
        for (int t = 0; t < num_threads; ++t) {
            if (i < thread_forces[t].size()) {
                f_sum += thread_forces[t][i].f;
                tau_sum += thread_forces[t][i].tau;
            }
        }
        bodies[i].f += f_sum;
        bodies[i].tau += tau_sum;
    }
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

void SoftContactSolver::computeP2PForce(ContactPrimitive& contact) {
    // Point-to-point: simplest case
    // Force acts along line connecting the two points
    
    const double grad = potentialGradient(contact.distance, contact.surface_limit);
    const float force_mag = static_cast<float>(-config_.k_scaler * grad);  // F = -k * dU/dd
    
    // Force direction: gradient points from A to B
    // dU/dx_b = (dU/dd) * (dd/dx_b) = grad * normal
    // dU/dx_a = (dU/dd) * (dd/dx_a) = grad * (-normal)
    
    contact.force_a = -force_mag * contact.normal;  // Push A away from B
    contact.force_b = force_mag * contact.normal;   // Push B away from A
}

void SoftContactSolver::computeE2PForce(ContactPrimitive& contact) {
    // Edge-to-point: one body at endpoint, other on edge
    // For now, use same as P2P (simplified)
    // TODO: Implement proper E2P gradient distribution along edge
    
    computeP2PForce(contact);
}

void SoftContactSolver::computeE2EForce(ContactPrimitive& contact) {
    // Edge-to-edge: both bodies on interior of segments
    // For now, use same as P2P (simplified)
    // TODO: Implement proper E2E gradient with distributed forces
    
    computeP2PForce(contact);
}

void SoftContactSolver::computeFriction(ContactPrimitive& contact,
                                        const RigidBody& body_a,
                                        const RigidBody& body_b,
                                        double dt) {
    // Compute velocities at contact points
    glm::vec3 r_a = contact.point_a - body_a.x;
    glm::vec3 r_b = contact.point_b - body_b.x;
    
    glm::vec3 v_a = body_a.v + glm::cross(body_a.w, r_a);
    glm::vec3 v_b = body_b.v + glm::cross(body_b.w, r_b);
    
    glm::vec3 v_rel = v_a - v_b;
    
    // Project out normal component to get tangential velocity
    glm::vec3 v_tan = v_rel - glm::dot(v_rel, contact.normal) * contact.normal;
    float v_tan_mag = glm::length(v_tan);
    
    if (v_tan_mag < 1e-10) {
        // No tangential motion → no friction
        contact.friction_a = glm::vec3(0);
        contact.friction_b = glm::vec3(0);
        return;
    }
    
    // Smooth friction coefficient (sticking-to-sliding transition)
    double gamma;
    if (v_tan_mag > config_.nu) {
        gamma = 1.0;  // Sliding regime
    } else {
        // Sticking regime: smooth transition using tanh-like function
        gamma = 2.0 / (1.0 + std::exp(-K2_ * v_tan_mag)) - 1.0;
    }
    
    // Effective friction coefficient (Stribeck effect for stick-slip)
    double mu_eff = config_.mu;
    if (config_.mu_static > config_.mu) {
        // Blend from mu_static (at v=0) to mu_dynamic (at v>>nu)
        // Decay scale: we want the static peak to be effective within the sticking range.
        // Using nu as the decay constant ensures smooth transition.
        double decay = std::exp(-v_tan_mag / config_.nu);
        mu_eff = config_.mu + (config_.mu_static - config_.mu) * decay;
    }

    // Friction force magnitude: μ_eff * γ * |F_normal|
    const float fn_mag = glm::length(contact.force_a);
    const glm::vec3 friction_dir = -v_tan / v_tan_mag;  // Oppose tangential motion
    
    const float friction_mag = static_cast<float>(mu_eff * gamma) * fn_mag;
    const glm::vec3 friction_force = friction_mag * friction_dir;
    
    contact.friction_a = friction_force;
    contact.friction_b = -friction_force;
}

// Geometric helper: Closest points on two line segments (Lumelsky algorithm from DisMech)
void SoftContactSolver::closestPointsSegmentSegment(
    const glm::vec3& a1, const glm::vec3& a2,
    const glm::vec3& b1, const glm::vec3& b2,
    double& s, double& t)
{
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
    
    auto fixBound = [](double& x) -> bool {
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

double SoftContactSolver::distanceSqPointToSegment(
    const glm::vec3& point,
    const glm::vec3& seg_start, const glm::vec3& seg_end,
    double& t)
{
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

void SoftContactSolver::detectSphereCapsule(const RigidBody& sphere, const RigidBody& capsule,
                                            int idx_sphere, int idx_capsule, std::vector<ContactPrimitive>& out_contacts) {
    // Sphere properties
    const glm::vec3& center = sphere.x;
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
            contact.type = ContactType::EDGE_TO_POINT;  // Hitting cylinder side
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
