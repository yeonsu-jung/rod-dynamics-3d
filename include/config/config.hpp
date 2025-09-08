#pragma once
#include <glm/glm.hpp>
#include <string>
#include <vector>
#include "physics/types.hpp"   // <- use the existing SolverConfig


// struct SolverConfig { float baumgarte=0.25f, allowedPen=0.003f; int velIters=30; };

struct PhysicsCfg {
  float dt = 1.0f / 600.0f;
  glm::vec3 gravity{0.0f,-10.0f,0.0f};
  float lin_damp = 0.08f;
  float ang_damp = 0.12f;
  float w_max    = 80.0f;
  SolverConfig solver{};
};

struct GridCfg {
  bool enabled = true;
  float scale = 1.0f;
  glm::vec3 c1{0.80f,0.82f,0.85f};
  glm::vec3 c2{0.65f,0.67f,0.70f};
};

struct RenderCfg {
  // orbit camera
  float yaw = 0.6f, pitch = 0.35f, dist = 6.0f;
  glm::vec3 lightDir{-0.4f,-1.0f,-0.3f};
  glm::vec3 bg{0.08f,0.09f,0.11f};
  GridCfg grid{};
  bool vsync = true;
  bool cull = false;
  int msaa_samples = 4;
};

struct FloorCfg {
  glm::vec3 pos{0,-0.8f,0};
  glm::vec4 rot_quat{1,0,0,0}; // wxyz
  glm::vec3 half_extents{10.0f,0.1f,10.0f};
  float restitution = 0.3f;
  float friction    = 0.9f;
};

// ... existing includes and structs ...

struct BodyCfg {
    // existing fields:
    glm::vec3 pos{0};
    // rotation options previously present:
    glm::vec3 rot_axis{0,1,0};
    float rot_deg{0.0f};
    glm::vec4 rot_quat{1,0,0,0}; // we treat this as wxyz by default
    // NEW:
    glm::vec3 euler_deg{0,0,0};   // [yaw, pitch, roll] or whatever you prefer—see below
    std::string rot_quat_order{"wxyz"}; // "wxyz" (GLM default) or "xyzw"

    // shape/material (unchanged)
    float length{0.5f};
    float diameter{0.1f};
    float density{1000.0f};
    float restitution{0.2f};
    float friction{0.7f};
    glm::vec3 v_lin{0};
    glm::vec3 v_ang{0};
};


struct SceneCfg {
  FloorCfg floor{};
  std::vector<BodyCfg> bodies;
};

struct AppCfg {
  PhysicsCfg physics{};
  RenderCfg  render{};
  SceneCfg   scene{};
};

// Returns false if file missing or invalid; cfg is still filled with defaults.
bool loadConfigFromFile(const std::string& path, AppCfg& cfg);

// Handy fallback (used when load fails)
AppCfg defaultAppCfg();

