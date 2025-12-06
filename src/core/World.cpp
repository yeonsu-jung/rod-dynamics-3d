/**
 * @file World.cpp
 * @brief Implementation of World class - core physics stepping
 */

#include "core/World.hpp"
#include "physics/integrator.hpp"
#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

// Thread limit (will be moved to World config)
extern int g_thread_limit;

// Helper for parallel loops (copied from main.cpp, will be refactored)
template <class F> static void parallel_for(size_t begin, size_t end, F fn) {
#ifdef _OPENMP
  int nthreads = omp_get_max_threads();
  if (g_thread_limit > 0 && nthreads > g_thread_limit) {
    omp_set_num_threads(g_thread_limit);
    nthreads = g_thread_limit;
  }
  if (end - begin < 100 || nthreads == 1) {
    for (size_t i = begin; i < end; ++i)
      fn(i);
  } else {
#pragma omp parallel for schedule(static)
    for (size_t i = begin; i < end; ++i)
      fn(i);
  }
#else
  for (size_t i = begin; i < end; ++i)
    fn(i);
#endif
}

namespace core {

World::World() { genRandomForce.seed(std::random_device{}()); }

void World::setPBC(bool enabled, const glm::vec3 &min, const glm::vec3 &max,
                   float cs) {
  usePBC = enabled;
  pbcMin = min;
  pbcMax = max;
  cellSize = cs;
  // Sync with global PBC state (temporary until globals removed)
  g_pbc_enabled = enabled;
  g_pbc_min = min;
  g_pbc_max = max;
}

void World::setRandomForce(bool enabled, float fSig, float tau,
                           unsigned int seed) {
  useRandomForce = enabled;
  fSigma = fSig;
  tauMag = tau;
  if (seed != 0) {
    genRandomForce.seed(seed);
  }
}

void World::setSleepThresholds(float linThresh, float angThresh,
                               float timeThresh) {
  sleepLinThresh = linThresh;
  sleepAngThresh = angThresh;
  sleepTimeThresh = timeThresh;
}

void World::addRod(const RigidBody &rb) {
  rods.push_back(rb);
  sleeping.push_back(0);
  sleepTimer.push_back(0.0f);
}

void World::setFloor(const RigidBody &floor) { floorRB = floor; }

void World::clearRods() {
  rods.clear();
  sleeping.clear();
  sleepTimer.clear();
  frameIndex = 0;
}

void World::wake(size_t i) {
  if (i >= rods.size())
    return;
  sleeping[i] = 0;
  sleepTimer[i] = 0.0f;
}

void World::wakeAll() {
  std::fill(sleeping.begin(), sleeping.end(), 0);
  std::fill(sleepTimer.begin(), sleepTimer.end(), 0.0f);
}

double World::totalKE() const {
  double KE = 0.0;
  for (const auto &rb : rods) {
    double m = rb.mass;
    double vSq = glm::dot(rb.v, rb.v);
    double KElin = 0.5 * m * vSq;
    glm::vec3 Llocal = rb.I_body * rb.w;
    double KErot = 0.5 * glm::dot(rb.w, Llocal);
    KE += KElin + KErot;
  }
  return KE;
}

glm::vec3 World::uniform_dir_s2() {
  float u = uni_u(genRandomForce);
  float phi = uni_phi(genRandomForce);
  float s = std::sqrt(std::max(0.0f, 1.0f - u * u));
  return glm::vec3(s * std::cos(phi), u, s * std::sin(phi));
}

void World::applyRandomForces() {
  if (!useRandomForce)
    return;
  for (auto &rb : rods) {
    glm::vec3 rndF = uniform_dir_s2() * (fSigma * normal_f(genRandomForce));
    glm::vec3 rndT = uniform_dir_s2() * tauMag;
    rb.f += rndF;
    rb.tau += rndT;
  }
}

void World::updateSleeping() {
  for (size_t i = 0; i < rods.size(); ++i) {
    const auto &rb = rods[i];
    float vMag = glm::length(rb.v);
    float wMag = glm::length(rb.w);
    if (vMag < sleepLinThresh && wMag < sleepAngThresh) {
      sleepTimer[i] += dt;
      if (sleepTimer[i] >= sleepTimeThresh) {
        sleeping[i] = 1;
      }
    } else {
      sleepTimer[i] = 0.0f;
      sleeping[i] = 0;
    }
  }
}

void World::integrateRods() {
  applyRandomForces();

  if (softContactEnabled) {
    // Soft contact uses force-based approach with Velocity Verlet
    for (auto &rb : rods) {
      rb.f += gravity * rb.mass;
    }
    parallel_for(0, rods.size(),
                 [&](size_t i) { integrateHalfPos(rods[i], gravity, dt); });
  }
  keAfterIntegrate = totalKE();
}

void World::wrapPBC() {
  if (!usePBC)
    return;
  const glm::vec3 size = pbcMax - pbcMin;
  for (auto &rb : rods) {
    for (int k = 0; k < 3; ++k) {
      float &coord = rb.x[k];
      if (coord < pbcMin[k])
        coord += size[k];
      else if (coord >= pbcMax[k])
        coord -= size[k];
    }
  }
  keAfterPBCWrap = totalKE();
}

// Grid helpers (static inline implementations)
glm::ivec3 World::gridDims(const glm::vec3 &bmin, const glm::vec3 &bmax,
                           float cs) {
  glm::vec3 size = bmax - bmin;
  return glm::max(glm::ivec3(1), glm::ivec3(glm::floor(size / cs)));
}

glm::ivec3 World::cellIndex(const glm::vec3 &p, const glm::vec3 &bmin,
                            const glm::vec3 &bmax, const glm::ivec3 &n) {
  glm::vec3 size = bmax - bmin;
  glm::vec3 rel = (p - bmin) / size;
  return glm::clamp(glm::ivec3(rel * glm::vec3(n)), glm::ivec3(0), n - 1);
}

int64_t World::packKey(const glm::ivec3 &i, const glm::ivec3 &n) {
  return (int64_t(i.x) << 42) ^ (int64_t(i.y) << 21) ^ int64_t(i.z);
}

size_t World::linearIndex(const glm::ivec3 &i, const glm::ivec3 &n) {
  return size_t(i.x) + size_t(n.x) * (size_t(i.y) + size_t(n.y) * size_t(i.z));
}

uint64_t World::pairKey(int a, int b) {
  if (b < a)
    std::swap(a, b);
  return (uint64_t(uint32_t(a)) << 32) | uint64_t(uint32_t(b));
}

void World::wrapPos(glm::vec3 &p, const glm::vec3 &bmin,
                    const glm::vec3 &bmax) {
  const glm::vec3 size = bmax - bmin;
  for (int k = 0; k < 3; ++k) {
    if (p[k] < bmin[k])
      p[k] += size[k];
    else if (p[k] >= bmax[k])
      p[k] -= size[k];
  }
}

void World::performCollisionResolution() {
  // This is the large collision detection + resolution block from
  // App::physicsStep For now, we'll implement a simplified version and expand
  // it in the next step
  // TODO: Full collision resolution logic will be moved here
  hitsScratch.clear();
  lastHitCount = 0;
  lastIslandCount = 0;
  keAfterWarmstart = totalKE();
  keAfterSolve = totalKE();
  keAfterPosCorrect = totalKE();
}

void World::step() {
  integrateRods();
  updateSleeping();

  if (!softContactEnabled) {
    performCollisionResolution();
  }

  wrapPBC();
  frameIndex++;
}

void World::stepWithSubsteps(int substeps) {
  float frameDt = dt;
  float subDt = frameDt / float(substeps);
  float saveDt = dt;

  for (int s = 0; s < substeps; ++s) {
    dt = subDt;
    step();
  }

  dt = saveDt;
}

} // namespace core
