// src/config/config.cpp
#include "config/config.hpp"

#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp> // glm::quat, angleAxis
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

// Declared in main.cpp
extern bool gQuiet;

using nlohmann::json;

/* -----------------------------
   JSON <-> GLM adapters
   ----------------------------- */
namespace nlohmann {

template <> struct adl_serializer<glm::vec3> {
  static void to_json(json &j, const glm::vec3 &v) {
    j = json::array({v.x, v.y, v.z});
  }
  static void from_json(const json &j, glm::vec3 &v) {
    if (j.is_array() && j.size() == 3) {
      v = glm::vec3(j[0].get<float>(), j[1].get<float>(), j[2].get<float>());
    } else if (j.is_object()) {
      v.x = j.value("x", 0.0f);
      v.y = j.value("y", 0.0f);
      v.z = j.value("z", 0.0f);
    } else {
      // Throwing here is fine; our jget<T> below catches and returns defaults.
      throw nlohmann::json::type_error::create(
          302, "glm::vec3 must be array[3] or object{x,y,z}", nullptr);
    }
  }
};

template <> struct adl_serializer<glm::vec4> {
  // Quaternions are stored component-order [w,x,y,z] in the vec4's
  // (x,y,z,w) slots — App::createRod reads quat(v.x, v.y, v.z, v.w) with
  // glm's (w,x,y,z) constructor. Serialize the slots in declaration order
  // so JSON [w,x,y,z] round-trips.
  static void to_json(json &j, const glm::vec4 &v) {
    j = json::array({v.x, v.y, v.z, v.w});
  }
  static void from_json(const json &j, glm::vec4 &v) {
    if (j.is_array() && j.size() == 4) {
      v = glm::vec4(j[0].get<float>(), j[1].get<float>(), j[2].get<float>(),
                    j[3].get<float>());
    } else if (j.is_object()) {
      // Object form {w,x,y,z}: JSON w goes into slot .x etc., matching the
      // storage convention above.
      v.x = j.value("w", 1.0f);
      v.y = j.value("x", 0.0f);
      v.z = j.value("y", 0.0f);
      v.w = j.value("z", 0.0f);
    } else {
      throw nlohmann::json::type_error::create(
          302, "glm::vec4 must be array[4] or object{w,x,y,z}", nullptr);
    }
  }
};

} // namespace nlohmann

/* -----------------------------
   Helpers
   ----------------------------- */

// Read j[key] as T, or return def if missing. A present-but-wrong-typed
// value is a config error the user should see, not a silent default.
template <class T> static T jget(const json &j, const char *key, const T &def) {
  if (!j.contains(key))
    return def;
  try {
    return j.at(key).get<T>();
  } catch (const std::exception &e) {
    std::cerr << "[config] WARNING: key \"" << key
              << "\" has unexpected type (" << e.what()
              << "); using default\n";
    return def;
  }
}

// Warn about JSON keys the parser does not handle, so typos don't silently
// run with defaults. Keys named name/description/notes or starting with '_'
// are treated as annotations and skipped.
static void warnUnknownKeys(const json &j,
                            std::initializer_list<const char *> known,
                            const char *ctx) {
  if (!j.is_object())
    return;
  for (const auto &item : j.items()) {
    const std::string &k = item.key();
    if (!k.empty() && k[0] == '_')
      continue;
    if (k == "name" || k == "description" || k == "notes")
      continue;
    bool found = false;
    for (const char *kn : known) {
      if (k == kn) {
        found = true;
        break;
      }
    }
    if (!found)
      std::cerr << "[config] WARNING: unknown key \"" << k << "\" in " << ctx
                << " (ignored)\n";
  }
}

// Old flat scene format (pre-2026) put physics and scene keys at the JSON
// root; the current parser would silently load an empty scene. Migrate the
// known keys under "physics"/"scene" so those files keep working.
static void migrateFlatFormat(json &j) {
  if (j.contains("scene") || j.contains("physics"))
    return;
  static const char *kPhysicsKeys[] = {
      "dt",  "gravity",       "lin_damp", "ang_damp",
      "w_max", "soft_contact", "hertz_mindlin", "nsc", "use_soft_contact"};
  static const char *kSceneKeys[] = {
      "bodies",      "periodic",    "floor",       "cylinder",
      "randomInit",  "randomForce", "populate",    "initCsvPath",
      "initCsv",     "fixCentroidRod", "fixedRodSelectionMethod",
      "numFixedRods", "fixEveryExcept"};
  bool any = false;
  for (const char *k : kPhysicsKeys)
    any = any || j.contains(k);
  for (const char *k : kSceneKeys)
    any = any || j.contains(k);
  if (!any)
    return;
  std::cerr << "[config] NOTE: legacy flat scene format detected; migrating "
               "keys under \"physics\"/\"scene\". Please update the file.\n";
  for (const char *k : kPhysicsKeys)
    if (j.contains(k)) {
      j["physics"][k] = j[k];
      j.erase(k);
    }
  for (const char *k : kSceneKeys)
    if (j.contains(k)) {
      j["scene"][k] = j[k];
      j.erase(k);
    }
}

/* -----------------------------
   Contact model selection
   ----------------------------- */
bool applyContactModel(AppCfg &cfg, const std::string &model) {
  if (model == "nsc") {
    cfg.physics.nsc.enabled = true;
    cfg.physics.soft_contact.enabled = false;
    cfg.physics.hertz_mindlin.enabled = false;
    cfg.physics.use_mujoco_contact = false;
  } else if (model == "harmonic") {
    cfg.physics.nsc.enabled = false;
    cfg.physics.soft_contact.enabled = true;
    cfg.physics.hertz_mindlin.enabled = false;
    cfg.physics.use_mujoco_contact = false;
  } else if (model == "hertz-mindlin") {
    cfg.physics.nsc.enabled = false;
    cfg.physics.soft_contact.enabled = false;
    cfg.physics.hertz_mindlin.enabled = true;
    cfg.physics.use_mujoco_contact = false;
  } else if (model == "mujoco") {
    cfg.physics.nsc.enabled = false;
    cfg.physics.soft_contact.enabled = true;
    cfg.physics.hertz_mindlin.enabled = false;
    cfg.physics.use_mujoco_contact = true;
    std::cerr << "[config] NOTE: the built-in mujoco-like model is "
                 "experimental (no PBC, sphere handling incomplete); for "
                 "validation prefer exporting the scene to real MuJoCo.\n";
  } else {
    std::cerr << "[config] ERROR: unknown contact_model \"" << model
              << "\" (valid: nsc, harmonic, hertz-mindlin, mujoco)\n";
    return false;
  }
  cfg.physics.contact_model = model;
  return true;
}

/* -----------------------------
   Defaults
   ----------------------------- */
AppCfg defaultAppCfg() {
  AppCfg c{};

  // Render
  c.render.bg = {0.08f, 0.09f, 0.11f};
  c.render.vsync = true;
  c.render.cull = false;
  c.render.msaa_samples = 4;
  c.render.yaw = 0.6f;
  c.render.pitch = 0.35f;
  c.render.dist = 6.0f;
  c.render.lightDir = {-0.4f, -1.0f, -0.3f};
  c.render.grid.enabled = true;
  c.render.grid.scale = 1.0f;
  c.render.grid.c1 = {0.80f, 0.82f, 0.85f};
  c.render.grid.c2 = {0.65f, 0.67f, 0.70f};

  // Physics
  c.physics.dt = 1.0f / 600.0f;
  c.physics.gravity = {0.0f, -10.0f, 0.0f};

  c.physics.lin_damp = 0.08f;
  c.physics.ang_damp = 0.12f;
  c.physics.w_max = 80.0f;

  // Floor
  c.scene.floor.pos = {0.0f, -0.8f, 0.0f};
  c.scene.floor.half_extents = {10.0f, 0.1f, 10.0f};
  c.scene.floor.rot_quat = {1, 0, 0, 0}; // w,x,y,z
  c.scene.floor.restitution = 0.3f;
  c.scene.floor.friction = 0.9f;

  // Periodic defaults (disabled)
  c.scene.periodic.enabled = false;
  c.scene.periodic.min = {-3.0f, -1.0f, -3.0f};
  c.scene.periodic.max = {+3.0f, +3.0f, +3.0f};
  c.scene.periodic.cellSize = 0.6f;

  // Random init defaults (disabled)
  c.scene.randomInit.enabled = false;
  c.scene.randomInit.mode = "thermal";
  c.scene.randomInit.vSigma = 0.0f;
  c.scene.randomInit.wSpeed = 0.0f;
  c.scene.randomInit.wSigma = 0.0f;
  c.scene.randomInit.kBT = 1.0f;
  c.scene.randomInit.kBTTrans = -1.0f;
  c.scene.randomInit.kBTRot = -1.0f;
  c.scene.randomInit.seed = 0;
  c.scene.randomInit.projectParallelSpin = true;

  // Random force defaults (disabled)
  c.scene.randomForce.enabled = false;
  c.scene.randomForce.fSigma = 0.0f;
  c.scene.randomForce.tauMag = 0.0f;
  c.scene.randomForce.seed = 0;

  // Populate defaults
  c.scene.populate.count = 0;
  c.scene.populate.grid = false;
  c.scene.populate.spacingMul = 1.0f;
  c.scene.populate.seed = 0;
  c.scene.populate.shape = "capsule";
  c.scene.populate.radius = 0.05f;
  c.scene.populate.density = 2500.0f;

  return c;
}

/* -----------------------------
   Load from JSON
   ----------------------------- */
bool loadConfigFromFile(const std::string &path, AppCfg &out) {
  std::ifstream f(path);
  if (!f) {
    std::cerr << "[config] Could not open: " << path << "\n";
    return false;
  }

  json j;
  try {
    f >> j;
  } catch (const std::exception &e) {
    std::cerr << "[config] JSON parse error in " << path << ": " << e.what()
              << "\n";
    return false;
  }

  // Start with defaults, then apply overrides
  AppCfg cfg = defaultAppCfg();

  migrateFlatFormat(j);
  warnUnknownKeys(j, {"render", "physics", "scene", "diagnostics"}, "root");

  // ---- render ----
  if (j.contains("render")) {
    const auto &jr = j["render"];
    warnUnknownKeys(jr,
                    {"bg", "vsync", "cull", "msaa", "yaw", "pitch", "dist",
                     "lightDir", "grid"},
                    "render");
    cfg.render.bg = jget(jr, "bg", cfg.render.bg);
    cfg.render.vsync = jget(jr, "vsync", cfg.render.vsync);
    cfg.render.cull = jget(jr, "cull", cfg.render.cull);
    cfg.render.msaa_samples = jget(jr, "msaa", cfg.render.msaa_samples);
    cfg.render.yaw = jget(jr, "yaw", cfg.render.yaw);
    cfg.render.pitch = jget(jr, "pitch", cfg.render.pitch);
    cfg.render.dist = jget(jr, "dist", cfg.render.dist);
    cfg.render.lightDir = jget(jr, "lightDir", cfg.render.lightDir);

    if (jr.contains("grid")) {
      const auto &jg = jr["grid"];
      cfg.render.grid.enabled = jget(jg, "enabled", cfg.render.grid.enabled);
      cfg.render.grid.scale = jget(jg, "scale", cfg.render.grid.scale);
      cfg.render.grid.c1 = jget(jg, "c1", cfg.render.grid.c1);
      cfg.render.grid.c2 = jget(jg, "c2", cfg.render.grid.c2);
    }
  }

  // ---- physics ----
  if (j.contains("physics")) {
    const auto &jp = j["physics"];
    warnUnknownKeys(jp,
                    {"dt", "gravity", "lin_damp", "ang_damp", "w_max",
                     "soft_contact", "hertz_mindlin", "nsc",
                     "use_soft_contact", "contact_model"},
                    "physics");
    cfg.physics.dt = jget(jp, "dt", cfg.physics.dt);
    cfg.physics.gravity = jget(jp, "gravity", cfg.physics.gravity);
    cfg.physics.lin_damp = jget(jp, "lin_damp", cfg.physics.lin_damp);
    cfg.physics.ang_damp = jget(jp, "ang_damp", cfg.physics.ang_damp);
    cfg.physics.w_max = jget(jp, "w_max", cfg.physics.w_max);

    // soft_contact
    if (jp.contains("soft_contact")) {
      const auto &jsc = jp["soft_contact"];
      warnUnknownKeys(jsc,
                      {"enabled", "delta", "k_scaler", "damping", "mu",
                       "mu_static", "nu", "enable_friction",
                       "friction_karnopp", "friction_cundall", "kt",
                       "vel_deadband", "verbose", "use_spatial_hash",
                       "use_cuda", "use_aabb", "cell_size"},
                      "physics.soft_contact");
      cfg.physics.soft_contact.enabled =
          jget(jsc, "enabled", cfg.physics.soft_contact.enabled);
      cfg.physics.soft_contact.delta =
          jget(jsc, "delta", cfg.physics.soft_contact.delta);
      cfg.physics.soft_contact.k_scaler =
          jget(jsc, "k_scaler", cfg.physics.soft_contact.k_scaler);
      cfg.physics.soft_contact.damping =
          jget(jsc, "damping", cfg.physics.soft_contact.damping);
      cfg.physics.soft_contact.mu =
          jget(jsc, "mu", cfg.physics.soft_contact.mu);
      cfg.physics.soft_contact.mu_static =
          jget(jsc, "mu_static", cfg.physics.soft_contact.mu_static);
      cfg.physics.soft_contact.nu =
          jget(jsc, "nu", cfg.physics.soft_contact.nu);
      cfg.physics.soft_contact.enable_friction = jget(
          jsc, "enable_friction", cfg.physics.soft_contact.enable_friction);
      cfg.physics.soft_contact.friction_karnopp = jget(
          jsc, "friction_karnopp", cfg.physics.soft_contact.friction_karnopp);
      cfg.physics.soft_contact.friction_cundall = jget(
          jsc, "friction_cundall", cfg.physics.soft_contact.friction_cundall);
      cfg.physics.soft_contact.kt =
          jget(jsc, "kt", cfg.physics.soft_contact.kt);
      cfg.physics.soft_contact.vel_deadband =
          jget(jsc, "vel_deadband", cfg.physics.soft_contact.vel_deadband);
      cfg.physics.soft_contact.verbose =
          jget(jsc, "verbose", cfg.physics.soft_contact.verbose);
      cfg.physics.soft_contact.use_spatial_hash = jget(
          jsc, "use_spatial_hash", cfg.physics.soft_contact.use_spatial_hash);
      cfg.physics.soft_contact.use_cuda =
          jget(jsc, "use_cuda", cfg.physics.soft_contact.use_cuda);
      cfg.physics.soft_contact.use_aabb =
          jget(jsc, "use_aabb", cfg.physics.soft_contact.use_aabb);
      cfg.physics.soft_contact.cell_size =
          jget(jsc, "cell_size", cfg.physics.soft_contact.cell_size);
    }

    // hertz_mindlin
    if (jp.contains("hertz_mindlin")) {
      const auto &jhm = jp["hertz_mindlin"];
      warnUnknownKeys(jhm,
                      {"enabled", "youngs_modulus", "poisson_ratio",
                       "restitution_coeff", "friction_coeff",
                       "friction_static_coeff", "friction_transition_vel",
                       "rolling_friction_coeff", "enable_tangential",
                       "enable_rolling", "use_uniform_grid",
                       "broadphase_cell_size", "broadphase_min_bodies",
                       "verbose"},
                      "physics.hertz_mindlin");
      cfg.physics.hertz_mindlin.enabled =
          jget(jhm, "enabled", cfg.physics.hertz_mindlin.enabled);
      cfg.physics.hertz_mindlin.youngs_modulus =
          jget(jhm, "youngs_modulus", cfg.physics.hertz_mindlin.youngs_modulus);
      cfg.physics.hertz_mindlin.poisson_ratio =
          jget(jhm, "poisson_ratio", cfg.physics.hertz_mindlin.poisson_ratio);
      cfg.physics.hertz_mindlin.restitution_coeff =
          jget(jhm, "restitution_coeff",
               cfg.physics.hertz_mindlin.restitution_coeff);
      cfg.physics.hertz_mindlin.friction_coeff =
          jget(jhm, "friction_coeff", cfg.physics.hertz_mindlin.friction_coeff);
      cfg.physics.hertz_mindlin.friction_static_coeff =
          jget(jhm, "friction_static_coeff",
               cfg.physics.hertz_mindlin.friction_static_coeff);
      cfg.physics.hertz_mindlin.friction_transition_vel =
          jget(jhm, "friction_transition_vel",
               cfg.physics.hertz_mindlin.friction_transition_vel);
      cfg.physics.hertz_mindlin.rolling_friction_coeff =
          jget(jhm, "rolling_friction_coeff",
               cfg.physics.hertz_mindlin.rolling_friction_coeff);
      cfg.physics.hertz_mindlin.enable_tangential =
          jget(jhm, "enable_tangential",
               cfg.physics.hertz_mindlin.enable_tangential);
      cfg.physics.hertz_mindlin.enable_rolling =
          jget(jhm, "enable_rolling", cfg.physics.hertz_mindlin.enable_rolling);
      cfg.physics.hertz_mindlin.use_uniform_grid = jget(
          jhm, "use_uniform_grid", cfg.physics.hertz_mindlin.use_uniform_grid);
      cfg.physics.hertz_mindlin.broadphase_cell_size =
          jget(jhm, "broadphase_cell_size",
               cfg.physics.hertz_mindlin.broadphase_cell_size);
      cfg.physics.hertz_mindlin.broadphase_min_bodies =
          jget(jhm, "broadphase_min_bodies",
               cfg.physics.hertz_mindlin.broadphase_min_bodies);
      cfg.physics.hertz_mindlin.verbose =
          jget(jhm, "verbose", cfg.physics.hertz_mindlin.verbose);
    }

    // nsc (Non-Smooth Contact)
    if (jp.contains("nsc")) {
      const auto &jnsc = jp["nsc"];
      warnUnknownKeys(jnsc,
                      {"enabled", "mu", "beta", "cfm", "omega",
                       "velocity_iters", "position_stabilization",
                       "position_iters", "position_psor_iters", "slop",
                       "use_spatial_hash", "cell_size", "use_aabb",
                       "use_cuda", "enable_warm_start"},
                      "physics.nsc");
      cfg.physics.nsc.enabled =
          jget(jnsc, "enabled", cfg.physics.nsc.enabled);
      cfg.physics.nsc.mu =
          jget(jnsc, "mu", cfg.physics.nsc.mu);
      cfg.physics.nsc.beta =
          jget(jnsc, "beta", cfg.physics.nsc.beta);
      cfg.physics.nsc.cfm =
          jget(jnsc, "cfm", cfg.physics.nsc.cfm);
      cfg.physics.nsc.omega =
          jget(jnsc, "omega", cfg.physics.nsc.omega);
      cfg.physics.nsc.velocity_iters =
          jget(jnsc, "velocity_iters", cfg.physics.nsc.velocity_iters);
      cfg.physics.nsc.position_stabilization =
          jget(jnsc, "position_stabilization", cfg.physics.nsc.position_stabilization);
      cfg.physics.nsc.position_iters =
          jget(jnsc, "position_iters", cfg.physics.nsc.position_iters);
      cfg.physics.nsc.position_psor_iters =
          jget(jnsc, "position_psor_iters", cfg.physics.nsc.position_psor_iters);
      cfg.physics.nsc.slop =
          jget(jnsc, "slop", cfg.physics.nsc.slop);
      cfg.physics.nsc.use_spatial_hash =
          jget(jnsc, "use_spatial_hash", cfg.physics.nsc.use_spatial_hash);
      cfg.physics.nsc.cell_size =
          jget(jnsc, "cell_size", cfg.physics.nsc.cell_size);
      cfg.physics.nsc.use_aabb =
          jget(jnsc, "use_aabb", cfg.physics.nsc.use_aabb);
      cfg.physics.nsc.use_cuda =
          jget(jnsc, "use_cuda", cfg.physics.nsc.use_cuda);
      cfg.physics.nsc.enable_warm_start =
          jget(jnsc, "enable_warm_start", cfg.physics.nsc.enable_warm_start);
    }

    // Legacy: allow top-level "use_soft_contact" boolean
    if (jp.contains("use_soft_contact")) {
      cfg.physics.soft_contact.enabled =
          jget(jp, "use_soft_contact", cfg.physics.soft_contact.enabled);
    }

    // Contact model selector — overrides the per-model enabled flags so a
    // single key switches models with everything else held fixed.
    if (jp.contains("contact_model")) {
      cfg.physics.contact_model =
          jget(jp, "contact_model", cfg.physics.contact_model);
      if (!applyContactModel(cfg, cfg.physics.contact_model))
        return false;
    }
  }

  // ---- scene ----
  if (j.contains("scene")) {
    const auto &jsn = j["scene"];
    warnUnknownKeys(jsn,
                    {"floor", "periodic", "cylinder", "randomInit",
                     "randomForce", "populate", "initCsvPath", "initCsv",
                     "fixCentroidRod", "fixedRodSelectionMethod",
                     "numFixedRods", "fixEveryExcept", "bodies"},
                    "scene");

    // floor
    if (jsn.contains("floor")) {
      const auto &jf = jsn["floor"];
      warnUnknownKeys(jf,
                      {"enabled", "pos", "half_extents", "rot_quat",
                       "restitution", "friction"},
                      "scene.floor");
      cfg.scene.floor.enabled =
          jget(jf, "enabled", cfg.scene.floor.enabled); // Load enabled
      cfg.scene.floor.pos = jget(jf, "pos", cfg.scene.floor.pos);
      cfg.scene.floor.half_extents =
          jget(jf, "half_extents", cfg.scene.floor.half_extents);
      cfg.scene.floor.rot_quat =
          jget(jf, "rot_quat", cfg.scene.floor.rot_quat); // (w,x,y,z)
      cfg.scene.floor.restitution =
          jget(jf, "restitution", cfg.scene.floor.restitution);
      cfg.scene.floor.friction = jget(jf, "friction", cfg.scene.floor.friction);
    }

    // periodic
    if (jsn.contains("periodic")) {
      const auto &jpbc = jsn["periodic"];
      warnUnknownKeys(jpbc, {"enabled", "min", "max", "cellSize", "longSpan"},
                      "scene.periodic");
      cfg.scene.periodic.enabled =
          jget(jpbc, "enabled", cfg.scene.periodic.enabled);
      cfg.scene.periodic.min = jget(jpbc, "min", cfg.scene.periodic.min);
      cfg.scene.periodic.max = jget(jpbc, "max", cfg.scene.periodic.max);
      cfg.scene.periodic.cellSize =
          jget(jpbc, "cellSize", cfg.scene.periodic.cellSize);
      cfg.scene.periodic.longSpan =
          jget(jpbc, "longSpan", cfg.scene.periodic.longSpan);
    }

    // cylinder (infinite tube boundary for reptation)
    if (jsn.contains("cylinder")) {
      const auto &jcyl = jsn["cylinder"];
      cfg.scene.cylinder.enabled =
          jget(jcyl, "enabled", cfg.scene.cylinder.enabled);
      cfg.scene.cylinder.axis =
          jget(jcyl, "axis", cfg.scene.cylinder.axis);
      cfg.scene.cylinder.radius =
          jget(jcyl, "radius", cfg.scene.cylinder.radius);
    }

    // random init
    if (jsn.contains("randomInit")) {
      const auto &jr = jsn["randomInit"];
      warnUnknownKeys(jr,
                      {"enabled", "mode", "vSigma", "wSpeed", "wSigma", "kBT",
                       "kBTTrans", "kBTRot", "seed", "projectParallelSpin"},
                      "scene.randomInit");
      cfg.scene.randomInit.enabled =
          jget(jr, "enabled", cfg.scene.randomInit.enabled);
      cfg.scene.randomInit.mode =
          jget(jr, "mode", cfg.scene.randomInit.mode);
      cfg.scene.randomInit.vSigma =
          jget(jr, "vSigma", cfg.scene.randomInit.vSigma);
      cfg.scene.randomInit.wSpeed =
          jget(jr, "wSpeed", cfg.scene.randomInit.wSpeed);
      cfg.scene.randomInit.wSigma =
          jget(jr, "wSigma", cfg.scene.randomInit.wSigma);
      cfg.scene.randomInit.kBT =
          jget(jr, "kBT", cfg.scene.randomInit.kBT);
        cfg.scene.randomInit.kBTTrans =
          jget(jr, "kBTTrans", cfg.scene.randomInit.kBTTrans);
        cfg.scene.randomInit.kBTRot =
          jget(jr, "kBTRot", cfg.scene.randomInit.kBTRot);
      cfg.scene.randomInit.seed = jget(jr, "seed", cfg.scene.randomInit.seed);
      cfg.scene.randomInit.projectParallelSpin =
          jget(jr, "projectParallelSpin", cfg.scene.randomInit.projectParallelSpin);
    }

    // random force
    if (jsn.contains("randomForce")) {
      if (!gQuiet) std::cout << "[config] Loading randomForce settings from JSON.\n";
      const auto &jrf = jsn["randomForce"];
      cfg.scene.randomForce.enabled =
          jget(jrf, "enabled", cfg.scene.randomForce.enabled);
      cfg.scene.randomForce.fSigma =
          jget(jrf, "fSigma", cfg.scene.randomForce.fSigma);
      cfg.scene.randomForce.tauMag =
          jget(jrf, "tauMag", cfg.scene.randomForce.tauMag);
      cfg.scene.randomForce.seed =
          jget(jrf, "seed", cfg.scene.randomForce.seed);
    }

    // populate
    if (jsn.contains("populate")) {
      const auto &jp = jsn["populate"];
      warnUnknownKeys(jp,
                      {"count", "grid", "spacingMul", "seed", "mode",
                       "maxAttempts", "shape", "radius", "density"},
                      "scene.populate");
      cfg.scene.populate.count = jget(jp, "count", cfg.scene.populate.count);
      cfg.scene.populate.grid = jget(jp, "grid", cfg.scene.populate.grid);
      cfg.scene.populate.spacingMul =
          jget(jp, "spacingMul", cfg.scene.populate.spacingMul);
      cfg.scene.populate.seed = jget(jp, "seed", cfg.scene.populate.seed);
      cfg.scene.populate.mode = jget(jp, "mode", cfg.scene.populate.mode);
      cfg.scene.populate.maxAttempts =
          jget(jp, "maxAttempts", cfg.scene.populate.maxAttempts);
      cfg.scene.populate.shape = jget(jp, "shape", cfg.scene.populate.shape);
      cfg.scene.populate.radius = jget(jp, "radius", cfg.scene.populate.radius);
      cfg.scene.populate.density =
          jget(jp, "density", cfg.scene.populate.density);
      // Back-compat: if mode not explicitly set, infer from 'grid'
      if (!jp.contains("mode")) {
        cfg.scene.populate.mode = cfg.scene.populate.grid
                                      ? std::string("grid")
                                      : std::string("uniform");
      }
    }

    // initial CSV configuration path (optional)
    if (jsn.contains("initCsvPath")) {
      try {
        cfg.scene.initCsvPath = jsn.at("initCsvPath").get<std::string>();
      } catch (...) {
        // ignore type errors
      }
    } else if (jsn.contains("initCsv")) {
      try {
        cfg.scene.initCsvPath = jsn.at("initCsv").get<std::string>();
      } catch (...) {
        // ignore type errors
      }
    }

    // Fix inner-most rod around centroid
    cfg.scene.fixCentroidRod = jget(jsn, "fixCentroidRod", cfg.scene.fixCentroidRod);
    cfg.scene.fixedRodSelectionMethod = jget(jsn, "fixedRodSelectionMethod", cfg.scene.fixedRodSelectionMethod);
    cfg.scene.numFixedRods = jget(jsn, "numFixedRods", cfg.scene.numFixedRods);
    cfg.scene.fixEveryExcept = jget(jsn, "fixEveryExcept", cfg.scene.fixEveryExcept);

    // bodies
    if (jsn.contains("bodies")) {
      const auto &jbodies = jsn["bodies"];
      if (jbodies.is_array()) {
        for (const auto &jb : jbodies) {
          BodyCfg b;
          warnUnknownKeys(jb,
                          {"shape", "radius", "length", "diameter", "density",
                           "restitution", "friction", "friction_s",
                           "friction_d", "is_static", "pos", "rot_quat",
                           "v_lin", "v_ang"},
                          "scene.bodies[]");
          b.shape = jget(jb, "shape", b.shape);
          b.radius = jget(jb, "radius", b.radius);
          b.length = jget(jb, "length", b.length);
          b.diameter = jget(jb, "diameter", b.diameter);
          // Capsules are built from `diameter` only. If the user gave a
          // radius but no diameter, honor it instead of silently using the
          // 0.1 default.
          if (b.shape != "sphere" && jb.contains("radius") &&
              !jb.contains("diameter")) {
            b.diameter = 2.0f * b.radius;
          }
          b.density = jget(jb, "density", b.density);
          b.restitution = jget(jb, "restitution", b.restitution);
          b.friction = jget(jb, "friction", b.friction);
          b.friction_s = jget(jb, "friction_s", b.friction_s);
          b.friction_d = jget(jb, "friction_d", b.friction_d);
          // could parse other fields if needed
          b.is_static = jget(jb, "is_static", b.is_static);
          b.pos = jget(jb, "pos", b.pos);
          b.rot_quat = jget(jb, "rot_quat", b.rot_quat);
          b.v_lin = jget(jb, "v_lin", b.v_lin);
          b.v_ang = jget(jb, "v_ang", b.v_ang);
          cfg.scene.bodies.push_back(b);
        }
      }
    }
  }

  // ---- diagnostics ----
  if (j.contains("diagnostics")) {
    const auto &jd = j["diagnostics"];
    if (jd.contains("early_pairs")) {
      const auto &jep = jd["early_pairs"];
      cfg.diagnostics.early_pairs.enabled =
          jget(jep, "enabled", cfg.diagnostics.early_pairs.enabled);
      cfg.diagnostics.early_pairs.start_step =
          jget(jep, "start_step", cfg.diagnostics.early_pairs.start_step);
      cfg.diagnostics.early_pairs.end_step =
          jget(jep, "end_step", cfg.diagnostics.early_pairs.end_step);
      cfg.diagnostics.early_pairs.stride =
          jget(jep, "stride", cfg.diagnostics.early_pairs.stride);
        cfg.diagnostics.early_pairs.schedule_mode =
          jget(jep, "schedule_mode", cfg.diagnostics.early_pairs.schedule_mode);
        cfg.diagnostics.early_pairs.geomspace_samples =
          jget(jep, "geomspace_samples",
             cfg.diagnostics.early_pairs.geomspace_samples);
      cfg.diagnostics.early_pairs.contact_output_path =
          jget(jep, "contact_output_path",
               cfg.diagnostics.early_pairs.contact_output_path);
      cfg.diagnostics.early_pairs.pair_distance_output_path =
          jget(jep, "pair_distance_output_path",
               cfg.diagnostics.early_pairs.pair_distance_output_path);
        cfg.diagnostics.early_pairs.pair_velocity_summary_output_path =
          jget(jep, "pair_velocity_summary_output_path",
             cfg.diagnostics.early_pairs.pair_velocity_summary_output_path);
      cfg.diagnostics.early_pairs.pair_distance_cutoff =
          jget(jep, "pair_distance_cutoff",
               cfg.diagnostics.early_pairs.pair_distance_cutoff);
      cfg.diagnostics.early_pairs.binary_pair_distance_output =
          jget(jep, "binary_pair_distance_output",
               cfg.diagnostics.early_pairs.binary_pair_distance_output);
    }
  }

  // If randomInit requested under PBC and gravity was NOT explicitly set in
  // the JSON, zero the default gravity. An explicit physics.gravity value is
  // always respected, even if it equals the default.
  if (cfg.scene.periodic.enabled && cfg.scene.randomInit.enabled) {
    const bool gravityExplicit =
        j.contains("physics") && j["physics"].contains("gravity");
    if (!gravityExplicit &&
        cfg.physics.gravity == glm::vec3(0.0f, -10.0f, 0.0f)) {
      if (!gQuiet)
        std::cout << "[config] periodic+randomInit: zeroing default gravity "
                     "(set physics.gravity explicitly to keep it)\n";
      cfg.physics.gravity = glm::vec3(0.0f);
    }
  }

  out = cfg;
  if (!gQuiet) std::cout << "[config] Loaded " << path
            << " | bodies=" << out.scene.bodies.size()
            << " | dt=" << out.physics.dt
            << " | periodic=" << (out.scene.periodic.enabled ? "on" : "off")
            << (out.scene.initCsvPath.empty() ? "" : " | initCsv=")
            << (out.scene.initCsvPath.empty() ? "" : out.scene.initCsvPath)
            << "\n";
  return true;
}
