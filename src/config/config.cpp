// src/config/config.cpp
#include "config/config.hpp"

#include <nlohmann/json.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>   // glm::quat, angleAxis
#include <fstream>
#include <iostream>
#include <string>

using nlohmann::json;

/* -----------------------------
   JSON <-> GLM adapters
   ----------------------------- */
namespace nlohmann {

template<> struct adl_serializer<glm::vec3> {
    static void to_json(json& j, const glm::vec3& v) {
        j = json::array({v.x, v.y, v.z});
    }
    static void from_json(const json& j, glm::vec3& v) {
        if (j.is_array() && j.size() == 3) {
            v = glm::vec3(j[0].get<float>(), j[1].get<float>(), j[2].get<float>());
        } else if (j.is_object()) {
            v.x = j.value("x", 0.0f);
            v.y = j.value("y", 0.0f);
            v.z = j.value("z", 0.0f);
        } else {
            // Throwing here is fine; our jget<T> below catches and returns defaults.
            throw nlohmann::json::type_error::create(
                302, "glm::vec3 must be array[3] or object{x,y,z}", nullptr
            );
        }
    }
};

template<> struct adl_serializer<glm::vec4> {
    // We serialize as [w,x,y,z] to match your quaternion storage order.
    static void to_json(json& j, const glm::vec4& v) {
        j = json::array({v.w, v.x, v.y, v.z});
    }
    static void from_json(const json& j, glm::vec4& v) {
        if (j.is_array() && j.size() == 4) {
            v = glm::vec4(j[0].get<float>(), j[1].get<float>(), j[2].get<float>(), j[3].get<float>());
        } else if (j.is_object()) {
            v.w = j.value("w", 1.0f);
            v.x = j.value("x", 0.0f);
            v.y = j.value("y", 0.0f);
            v.z = j.value("z", 0.0f);
        } else {
            throw nlohmann::json::type_error::create(
                302, "glm::vec4 must be array[4] or object{w,x,y,z}", nullptr
            );
        }
    }
};

} // namespace nlohmann

/* -----------------------------
   Helpers
   ----------------------------- */

// Read j[key] as T, or return def if missing / wrong type
template <class T>
static T jget(const json& j, const char* key, const T& def) {
    if (!j.contains(key)) return def;
    try { return j.at(key).get<T>(); }
    catch (...) { return def; }
}

/* -----------------------------
   Defaults
   ----------------------------- */
AppCfg defaultAppCfg() {
    AppCfg c{};

    // Render
    c.render.bg = {0.08f, 0.09f, 0.11f};
    c.render.vsync = true;
    c.render.cull  = false;
    c.render.msaa_samples = 4;
    c.render.yaw   = 0.6f;
    c.render.pitch = 0.35f;
    c.render.dist  = 6.0f;
    c.render.lightDir = {-0.4f, -1.0f, -0.3f};
    c.render.grid.enabled = true;
    c.render.grid.scale   = 1.0f;
    c.render.grid.c1 = {0.80f, 0.82f, 0.85f};
    c.render.grid.c2 = {0.65f, 0.67f, 0.70f};

    // Physics
    c.physics.dt = 1.0f/600.0f;
    c.physics.gravity = {0.0f, -10.0f, 0.0f};
    c.physics.solver.velIters   = 30;
    c.physics.solver.baumgarte  = 0.25f;
    c.physics.solver.allowedPen = 0.003f;
    c.physics.substeps = 1;
    c.physics.solver.splitImpulse = true;   // Enable split impulse (more stable, minimal cost)
    c.physics.solver.splitOrient  = true;
    c.physics.solver.ngsNormalSweeps = 0;
    c.physics.solver.ngsHighVThresh = 0.5f;
    c.physics.lin_damp = 0.08f;
    c.physics.ang_damp = 0.12f;
    c.physics.w_max    = 80.0f;

    // Floor
    c.scene.floor.pos = {0.0f, -0.8f, 0.0f};
    c.scene.floor.half_extents = {10.0f, 0.1f, 10.0f};
    c.scene.floor.rot_quat = {1,0,0,0}; // w,x,y,z
    c.scene.floor.restitution = 0.3f;
    c.scene.floor.friction    = 0.9f;

    // Periodic defaults (disabled)
    c.scene.periodic.enabled = false;
    c.scene.periodic.min = {-3.0f, -1.0f, -3.0f};
    c.scene.periodic.max = {+3.0f, +3.0f, +3.0f};
    c.scene.periodic.cellSize = 0.6f;

    // Random init defaults (disabled)
    c.scene.randomInit.enabled = false;
    c.scene.randomInit.vSigma  = 0.0f;
    c.scene.randomInit.wSpeed  = 0.0f;
    c.scene.randomInit.seed    = 0;

    // Random force defaults (disabled)
    c.scene.randomForce.enabled = false;
    c.scene.randomForce.fSigma  = 0.0f;
    c.scene.randomForce.tauMag  = 0.0f;
    c.scene.randomForce.seed    = 0;

    // Populate defaults
    c.scene.populate.count = 0;
    c.scene.populate.grid = false;
    c.scene.populate.spacingMul = 1.0f;
    c.scene.populate.seed = 0;
    c.scene.populate.shape = "capsule";
    c.scene.populate.radius = 0.05f;
    c.scene.populate.density = 2500.0f;

    // Two default rods
    BodyCfg a{}, b{};
    a.pos = {-1.6f, 0.6f, 0.0f};
    a.rot_quat = {1,0,0,0};
    a.density=1000.0f; a.length=0.5f; a.diameter=0.10f;
    a.restitution=0.15f; a.friction=0.6f; a.v_lin={+2.2f,0,0};

    b.pos = {+1.2f, 1.0f, 0.2f};
    b.rot_quat = {1,0,0,0};
    b.density=1000.0f; b.length=0.5f; b.diameter=0.10f;
    b.restitution=0.15f; b.friction=0.6f; b.v_lin={-1.0f,0,0};

    c.scene.bodies = {a,b};
    return c;
}

/* -----------------------------
   Load from JSON
   ----------------------------- */
bool loadConfigFromFile(const std::string& path, AppCfg& out) {
    std::ifstream f(path);
    if (!f) {
        std::cerr << "[config] Could not open: " << path << "\n";
        return false;
    }

    json j;
    try { f >> j; }
    catch (const std::exception& e) {
        std::cerr << "[config] JSON parse error in " << path << ": " << e.what() << "\n";
        return false;
    }

    // Start with defaults, then apply overrides
    AppCfg cfg = defaultAppCfg();

    // ---- render ----
    if (j.contains("render")) {
        const auto& jr = j["render"];
        cfg.render.bg = jget(jr, "bg", cfg.render.bg);
        cfg.render.vsync = jget(jr, "vsync", cfg.render.vsync);
        cfg.render.cull  = jget(jr, "cull", cfg.render.cull);
        cfg.render.msaa_samples = jget(jr, "msaa", cfg.render.msaa_samples);
        cfg.render.yaw   = jget(jr, "yaw",   cfg.render.yaw);
        cfg.render.pitch = jget(jr, "pitch", cfg.render.pitch);
        cfg.render.dist  = jget(jr, "dist",  cfg.render.dist);
        cfg.render.lightDir = jget(jr, "lightDir", cfg.render.lightDir);

        if (jr.contains("grid")) {
            const auto& jg = jr["grid"];
            cfg.render.grid.enabled = jget(jg, "enabled", cfg.render.grid.enabled);
            cfg.render.grid.scale   = jget(jg, "scale",   cfg.render.grid.scale);
            cfg.render.grid.c1      = jget(jg, "c1",      cfg.render.grid.c1);
            cfg.render.grid.c2      = jget(jg, "c2",      cfg.render.grid.c2);
        }
    }

    // ---- physics ----
    if (j.contains("physics")) {
        const auto& jp = j["physics"];
        cfg.physics.dt       = jget(jp, "dt",       cfg.physics.dt);
        cfg.physics.gravity  = jget(jp, "gravity",  cfg.physics.gravity);
        cfg.physics.lin_damp = jget(jp, "lin_damp", cfg.physics.lin_damp);
        cfg.physics.ang_damp = jget(jp, "ang_damp", cfg.physics.ang_damp);
        cfg.physics.w_max    = jget(jp, "w_max",    cfg.physics.w_max);
        cfg.physics.substeps = jget(jp, "substeps", cfg.physics.substeps);

        if (jp.contains("solver")) {
            const auto& js = jp["solver"];
            cfg.physics.solver.velIters   = jget(js, "velIters",   cfg.physics.solver.velIters);
            cfg.physics.solver.baumgarte  = jget(js, "baumgarte",  cfg.physics.solver.baumgarte);
            cfg.physics.solver.allowedPen = jget(js, "allowedPen", cfg.physics.solver.allowedPen);
            cfg.physics.solver.splitImpulse = jget(js, "splitImpulse", cfg.physics.solver.splitImpulse);
            cfg.physics.solver.splitOrient  = jget(js, "splitOrient",  cfg.physics.solver.splitOrient);
            cfg.physics.solver.ngsNormalSweeps = jget(js, "ngsNormalSweeps", cfg.physics.solver.ngsNormalSweeps);
            cfg.physics.solver.ngsHighVThresh = jget(js, "ngsHighVThresh", cfg.physics.solver.ngsHighVThresh);
        }
        
        // soft_contact
        if (jp.contains("soft_contact")) {
            const auto& jsc = jp["soft_contact"];
            cfg.physics.soft_contact.enabled = jget(jsc, "enabled", cfg.physics.soft_contact.enabled);
            cfg.physics.soft_contact.delta = jget(jsc, "delta", cfg.physics.soft_contact.delta);
            cfg.physics.soft_contact.k_scaler = jget(jsc, "k_scaler", cfg.physics.soft_contact.k_scaler);
            cfg.physics.soft_contact.mu = jget(jsc, "mu", cfg.physics.soft_contact.mu);
            cfg.physics.soft_contact.nu = jget(jsc, "nu", cfg.physics.soft_contact.nu);
            cfg.physics.soft_contact.enable_friction = jget(jsc, "enable_friction", cfg.physics.soft_contact.enable_friction);
            cfg.physics.soft_contact.verbose = jget(jsc, "verbose", cfg.physics.soft_contact.verbose);
        }
        
        // hertz_mindlin
        if (jp.contains("hertz_mindlin")) {
            const auto& jhm = jp["hertz_mindlin"];
            cfg.physics.hertz_mindlin.enabled = jget(jhm, "enabled", cfg.physics.hertz_mindlin.enabled);
            cfg.physics.hertz_mindlin.youngs_modulus = jget(jhm, "youngs_modulus", cfg.physics.hertz_mindlin.youngs_modulus);
            cfg.physics.hertz_mindlin.poisson_ratio = jget(jhm, "poisson_ratio", cfg.physics.hertz_mindlin.poisson_ratio);
            cfg.physics.hertz_mindlin.restitution_coeff = jget(jhm, "restitution_coeff", cfg.physics.hertz_mindlin.restitution_coeff);
            cfg.physics.hertz_mindlin.friction_coeff = jget(jhm, "friction_coeff", cfg.physics.hertz_mindlin.friction_coeff);
            cfg.physics.hertz_mindlin.friction_static_coeff = jget(jhm, "friction_static_coeff", cfg.physics.hertz_mindlin.friction_static_coeff);
            cfg.physics.hertz_mindlin.friction_transition_vel = jget(jhm, "friction_transition_vel", cfg.physics.hertz_mindlin.friction_transition_vel);
            cfg.physics.hertz_mindlin.rolling_friction_coeff = jget(jhm, "rolling_friction_coeff", cfg.physics.hertz_mindlin.rolling_friction_coeff);
            cfg.physics.hertz_mindlin.enable_tangential = jget(jhm, "enable_tangential", cfg.physics.hertz_mindlin.enable_tangential);
            cfg.physics.hertz_mindlin.enable_rolling = jget(jhm, "enable_rolling", cfg.physics.hertz_mindlin.enable_rolling);
            cfg.physics.hertz_mindlin.use_uniform_grid = jget(jhm, "use_uniform_grid", cfg.physics.hertz_mindlin.use_uniform_grid);
            cfg.physics.hertz_mindlin.broadphase_cell_size = jget(jhm, "broadphase_cell_size", cfg.physics.hertz_mindlin.broadphase_cell_size);
            cfg.physics.hertz_mindlin.broadphase_min_bodies = jget(jhm, "broadphase_min_bodies", cfg.physics.hertz_mindlin.broadphase_min_bodies);
            cfg.physics.hertz_mindlin.verbose = jget(jhm, "verbose", cfg.physics.hertz_mindlin.verbose);
        }
        
        // Legacy: allow top-level "use_soft_contact" boolean
        if (jp.contains("use_soft_contact")) {
            cfg.physics.soft_contact.enabled = jget(jp, "use_soft_contact", cfg.physics.soft_contact.enabled);
        }
    }

    // ---- scene ----
    if (j.contains("scene")) {
        const auto& jsn = j["scene"];

        // floor
        if (jsn.contains("floor")) {
            const auto& jf = jsn["floor"];
            cfg.scene.floor.pos          = jget(jf, "pos",          cfg.scene.floor.pos);
            cfg.scene.floor.half_extents = jget(jf, "half_extents", cfg.scene.floor.half_extents);
            cfg.scene.floor.rot_quat     = jget(jf, "rot_quat",     cfg.scene.floor.rot_quat); // (w,x,y,z)
            cfg.scene.floor.restitution  = jget(jf, "restitution",  cfg.scene.floor.restitution);
            cfg.scene.floor.friction     = jget(jf, "friction",     cfg.scene.floor.friction);
        }

        // periodic
        if (jsn.contains("periodic")) {
            const auto& jpbc = jsn["periodic"];
            cfg.scene.periodic.enabled = jget(jpbc, "enabled", cfg.scene.periodic.enabled);
            cfg.scene.periodic.min     = jget(jpbc, "min",     cfg.scene.periodic.min);
            cfg.scene.periodic.max     = jget(jpbc, "max",     cfg.scene.periodic.max);
            cfg.scene.periodic.cellSize= jget(jpbc, "cellSize",cfg.scene.periodic.cellSize);
            cfg.scene.periodic.longSpan= jget(jpbc, "longSpan",cfg.scene.periodic.longSpan);
        }

        // random init
        if (jsn.contains("randomInit")) {
            const auto& jr = jsn["randomInit"];
            cfg.scene.randomInit.enabled = jget(jr, "enabled", cfg.scene.randomInit.enabled);
            cfg.scene.randomInit.vSigma  = jget(jr, "vSigma",  cfg.scene.randomInit.vSigma);
            cfg.scene.randomInit.wSpeed  = jget(jr, "wSpeed",  cfg.scene.randomInit.wSpeed);
            cfg.scene.randomInit.seed    = jget(jr, "seed",    cfg.scene.randomInit.seed);
        }

        // random force
        if (jsn.contains("randomForce")) {
            std::cout << "[config] Loading randomForce settings from JSON.\n";
            const auto& jrf = jsn["randomForce"];
            cfg.scene.randomForce.enabled = jget(jrf, "enabled", cfg.scene.randomForce.enabled);
            cfg.scene.randomForce.fSigma  = jget(jrf, "fSigma",  cfg.scene.randomForce.fSigma);
            cfg.scene.randomForce.tauMag  = jget(jrf, "tauMag",  cfg.scene.randomForce.tauMag);
            cfg.scene.randomForce.seed    = jget(jrf, "seed",    cfg.scene.randomForce.seed);
        }

        // populate
        if (jsn.contains("populate")) {
            const auto& jp = jsn["populate"];
            cfg.scene.populate.count      = jget(jp, "count",      cfg.scene.populate.count);
            cfg.scene.populate.grid       = jget(jp, "grid",       cfg.scene.populate.grid);
            cfg.scene.populate.spacingMul = jget(jp, "spacingMul", cfg.scene.populate.spacingMul);
            cfg.scene.populate.seed       = jget(jp, "seed",       cfg.scene.populate.seed);
            cfg.scene.populate.mode       = jget(jp, "mode",       cfg.scene.populate.mode);
            cfg.scene.populate.maxAttempts= jget(jp, "maxAttempts",cfg.scene.populate.maxAttempts);
            cfg.scene.populate.shape      = jget(jp, "shape",      cfg.scene.populate.shape);
            cfg.scene.populate.radius     = jget(jp, "radius",     cfg.scene.populate.radius);
            cfg.scene.populate.density    = jget(jp, "density",    cfg.scene.populate.density);
            // Back-compat: if mode not explicitly set, infer from 'grid'
            if (!jp.contains("mode")) {
                cfg.scene.populate.mode = cfg.scene.populate.grid ? std::string("grid") : std::string("uniform");
            }
        }

        // bodies
        if (jsn.contains("bodies") && jsn["bodies"].is_array()) {
            cfg.scene.bodies.clear();

            for (const auto& jb : jsn["bodies"]) {
                BodyCfg bc{};

                // basics
                bc.pos        = jget(jb, "pos", glm::vec3(0));
                bc.density    = jget(jb, "density",  1000.0f);
                bc.shape      = jget(jb, "shape", std::string("capsule"));
                bc.length     = jget(jb, "length",   0.5f);
                bc.diameter   = jget(jb, "diameter", 0.10f);
                bc.radius     = jget(jb, "radius",   0.1f);
                bc.restitution= jget(jb, "restitution", 0.15f);
                bc.friction   = jget(jb, "friction",    0.6f);
                bc.friction_s = jget(jb, "friction_s",  bc.friction_s);
                bc.friction_d = jget(jb, "friction_d",  bc.friction_d);
                bc.rolling_friction = jget(jb, "rolling_friction", bc.rolling_friction);
                bc.v_lin      = jget(jb, "v_lin", glm::vec3(0));
                bc.v_ang      = jget(jb, "v_ang", glm::vec3(0));

                // orientation inputs (optional)
                bc.rot_quat       = jget(jb, "rot_quat", glm::vec4(1,0,0,0)); // (w,x,y,z)
                bc.euler_deg      = jget(jb, "euler_deg", bc.euler_deg);
                bc.rot_axis       = jget(jb, "rot_axis",  glm::vec3(0));
                bc.rot_deg        = jget(jb, "rot_deg",   0.0f);
                bc.rot_quat_order = jget(jb, "rot_quat_order", bc.rot_quat_order);

                // Orientation precedence: rot_quat > euler_deg > (rot_axis + rot_deg)
                if (jb.contains("rot_quat")) {
                    bc.rot_quat = jb.at("rot_quat").get<glm::vec4>(); // already (w,x,y,z)
                } else if (jb.contains("euler_deg")) {
                    glm::vec3 deg = jb.at("euler_deg").get<glm::vec3>();
                    glm::vec3 r   = glm::radians(deg);

                    // Construct quaternion from yaw(Y), pitch(X), roll(Z) (adjust if you prefer)
                    glm::quat qy = glm::angleAxis(r.y, glm::vec3(0,1,0));
                    glm::quat qx = glm::angleAxis(r.x, glm::vec3(1,0,0));
                    glm::quat qz = glm::angleAxis(r.z, glm::vec3(0,0,1));
                    glm::quat q  = qy * qx * qz; // NOTE: order matters

                    bc.rot_quat  = glm::vec4(q.w, q.x, q.y, q.z);
                } else if (glm::length(bc.rot_axis) > 0.0f && bc.rot_deg != 0.0f) {
                    glm::quat q = glm::angleAxis(glm::radians(bc.rot_deg), glm::normalize(bc.rot_axis));
                    bc.rot_quat = glm::vec4(q.w, q.x, q.y, q.z);
                }

                cfg.scene.bodies.push_back(bc);
            }
        }
    }

    // If randomInit requested under PBC, zero gravity unless user already set different gravity
    if (cfg.scene.periodic.enabled && cfg.scene.randomInit.enabled) {
        // respect user-provided physics.gravity if non-zero was provided in JSON physics
        if (cfg.physics.gravity == glm::vec3(0.0f, -10.0f, 0.0f)) {
            cfg.physics.gravity = glm::vec3(0.0f);
        }
    }

    out = cfg;
    std::cout << "[config] Loaded " << path
              << " | bodies=" << out.scene.bodies.size()
              << " | dt=" << out.physics.dt
              << " | periodic=" << (out.scene.periodic.enabled?"on":"off")
              << "\n";
    return true;
}
