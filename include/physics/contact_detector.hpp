/**
 * @file contact_detector.hpp
 * @brief Shared broadphase and narrowphase contact detector.
 */

#pragma once

#include <glm/glm.hpp>
#include <unordered_map>
#include <vector>

#include "config/config.hpp"

struct RigidBody;

/**
 * @brief Contact primitive types
 */
enum class ContactType {
  POINT_TO_POINT,
  EDGE_TO_POINT,
  EDGE_TO_EDGE
};

/**
 * @brief Detected contact between two shapes.
 */
struct ContactPrimitive {
  ContactType type;
  int body_a, body_b;
  glm::vec3 point_a, point_b;
  glm::vec3 normal;
  double distance;
  double surface_limit;

  glm::vec3 force_a, force_b;
  glm::vec3 friction_a, friction_b;
  glm::vec3 shift_b;
};

struct ContactDetectionCfg {
  double delta = 0.005;
  bool verbose = false;
  bool use_spatial_hash = false;
  bool use_cuda = false;
  bool use_aabb = true;
  double cell_size = -1.0;
};

struct ContactBroadphaseStats {
  double count_ms = 0.0;
  double prefix_ms = 0.0;
  double fill_ms = 0.0;
  double sort_ms = 0.0;
  double detect_ms = 0.0;
  double cuda_pack_ms = 0.0;
  double cuda_kernel_ms = 0.0;
  double cuda_download_ms = 0.0;
  int cuda_contacts = 0;
};

class ContactDetector {
public:
  explicit ContactDetector(const ContactDetectionCfg &config = ContactDetectionCfg());

  void setConfig(const ContactDetectionCfg &config);
  void setPBC(bool enabled, const glm::vec3 &min, const glm::vec3 &max);
  void detectContacts(const std::vector<RigidBody> &bodies);

  const std::vector<ContactPrimitive> &getContacts() const { return contacts_; }
  std::vector<ContactPrimitive> &accessContacts() { return contacts_; }
  size_t getNumContacts() const { return contacts_.size(); }
  const ContactBroadphaseStats &getStats() const { return stats_; }

private:
  ContactDetectionCfg config_;
  std::vector<ContactPrimitive> contacts_;
  ContactBroadphaseStats stats_;

  bool pbcEnabled_ = false;
  glm::vec3 pbcMin_{0.0f};
  glm::vec3 pbcMax_{0.0f};
  glm::vec3 pbcSize_{0.0f};

  std::vector<std::vector<ContactPrimitive>> threadBuffers_;

  struct GridKey {
    int x, y, z;
    bool operator==(const GridKey &other) const {
      return x == other.x && y == other.y && z == other.z;
    }
  };

  struct GridKeyHash {
    std::size_t operator()(const GridKey &k) const {
      return ((std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1)) >> 1) ^
             (std::hash<int>()(k.z) << 1);
    }
  };

  using GridMap = std::unordered_map<GridKey, std::vector<int>, GridKeyHash>;

  void detectContactsNaive(const std::vector<RigidBody> &bodies);
#ifdef USE_CUDA
  void detectContactsCuda(const std::vector<RigidBody> &bodies);
#endif
  void detectContactsSpatialHash(const std::vector<RigidBody> &bodies);

  double computeAdaptiveCellSize(const std::vector<RigidBody> &bodies) const;
  void insertBodyIntoGrid(int bodyIdx, const RigidBody &body, double cellSize,
                          GridMap &grid);

  void detectCapsuleCapsule(const RigidBody &a, const RigidBody &b, int idx_a,
                            int idx_b,
                            std::vector<ContactPrimitive> &out_contacts);
  void detectSphereSphere(const RigidBody &a, const RigidBody &b, int idx_a,
                          int idx_b,
                          std::vector<ContactPrimitive> &out_contacts);
  void detectSphereCapsule(const RigidBody &sphere, const RigidBody &capsule,
                           int idx_sphere, int idx_capsule,
                           std::vector<ContactPrimitive> &out_contacts);

  void getAABB(const RigidBody &b, glm::vec3 &min_pt, glm::vec3 &max_pt) const;
  bool checkAABBOverlap(const RigidBody &a, const RigidBody &b) const;

  static void closestPointsSegmentSegment(const glm::vec3 &a1,
                                          const glm::vec3 &a2,
                                          const glm::vec3 &b1,
                                          const glm::vec3 &b2, double &s,
                                          double &t);

  static double distanceSqPointToSegment(const glm::vec3 &point,
                                         const glm::vec3 &seg_start,
                                         const glm::vec3 &seg_end, double &t);
};
