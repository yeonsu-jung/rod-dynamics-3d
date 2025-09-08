// src/app/main.cpp
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "physics/rigid_body.hpp"
#include "physics/collision.hpp"
#include "physics/solver.hpp"
#include "physics/integrator.hpp"  // g_lin_damp, g_ang_damp, g_w_max

#include "gfx/renderer.hpp"
#include "gfx/mesh.hpp"
#include "gfx/camera.hpp"

#include "config/config.hpp"

#ifndef ASSETS_DIR
#define ASSETS_DIR "."
#endif

#ifdef GLAD_GL_KHR_debug
static void GLAPIENTRY glDebugCallback(GLenum, GLenum, GLuint, GLenum sev, GLsizei, const GLchar* msg, const void*)
{
    if (sev == GL_DEBUG_SEVERITY_NOTIFICATION) return;
    std::cerr << "[GL] " << msg << "\n";
}
#endif

struct App {
    // ---- window / GL
    GLFWwindow* window = nullptr;
    bool vsync = true;

    // ---- renderer & meshes (no gfx:: namespace)
    Renderer rnd;
    Mesh     cube, cyl;

    // ---- camera
    OrbitCamera cam;
    bool   dragging = false;
    double lastX = 0.0, lastY = 0.0;

    // ---- sim
    bool         paused = false;
    glm::vec3    gravity{0.0f, -10.0f, 0.0f};
    float        dt = 1.0f / 600.0f;
    AppCfg       settings{};
    SolverConfig solver{};

    // N rods + floor
    std::vector<RigidBody> rods;
    RigidBody              floorRB;

    bool initWindow(int W=1200, int H=800, const char* title="Rigid Bodies – Rods (Capsules)") {
        if (!glfwInit()) { std::cerr << "GLFW init failed\n"; return false; }
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_SAMPLES, std::max(1, settings.render.msaa_samples));
    #ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    #endif

        window = glfwCreateWindow(W, H, title, nullptr, nullptr);
        if (!window) { std::cerr << "GLFW window creation failed\n"; glfwTerminate(); return false; }
        glfwMakeContextCurrent(window);

        vsync = settings.render.vsync;
        glfwSwapInterval(vsync ? 1 : 0);

        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { std::cerr << "GLAD load failed\n"; return false; }

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_MULTISAMPLE);
        if (settings.render.cull) { glEnable(GL_CULL_FACE); glCullFace(GL_BACK); }
        else                      { glDisable(GL_CULL_FACE); }
        glDisable(GL_BLEND);

    #ifdef GLAD_GL_KHR_debug
        if (GLAD_GL_KHR_debug) { glEnable(GL_DEBUG_OUTPUT); glDebugMessageCallback(glDebugCallback, nullptr); }
    #endif

        glfwSetWindowUserPointer(window, this);
        glfwSetKeyCallback(window, &App::keyCB);
        glfwSetCursorPosCallback(window, &App::cursorCB);
        glfwSetMouseButtonCallback(window, &App::mouseCB);
        glfwSetScrollCallback(window, &App::scrollCB);
        return true;
    }

    bool initGfx() {
        if (!rnd.init(ASSETS_DIR)) {
            std::cerr << "Renderer init failed (check " << ASSETS_DIR << "/shaders).\n";
            return false;
        }
        cube = makeCubeMesh();
        cyl  = makeCappedCylinderMesh(40);
        return true;
    }

    static RigidBody mkRod(const BodyCfg& Bc){
        // rot_quat is glm::vec4 {w,x,y,z} (resolved by config)
        glm::quat q(Bc.rot_quat.x, Bc.rot_quat.y, Bc.rot_quat.z, Bc.rot_quat.w);
        RigidBody rb = RigidBody::makeRodLD(Bc.pos, q, Bc.density, Bc.length, Bc.diameter,
                                            Bc.restitution, Bc.friction);
        rb.v = Bc.v_lin; rb.w = Bc.v_ang;
        return rb;
    }

    void resetScene(){
        dt       = settings.physics.dt;
        gravity  = settings.physics.gravity;
        solver   = settings.physics.solver;

        g_lin_damp = settings.physics.lin_damp;
        g_ang_damp = settings.physics.ang_damp;
        g_w_max    = settings.physics.w_max;

        const auto& F = settings.scene.floor;
        glm::quat qF(F.rot_quat.x, F.rot_quat.y, F.rot_quat.z, F.rot_quat.w);
        floorRB = RigidBody::makeStaticFloor(
            F.pos, qF, F.half_extents.x, F.half_extents.y, F.half_extents.z,
            F.restitution, F.friction
        );

        rods.clear();
        if (!settings.scene.bodies.empty()){
            rods.reserve(settings.scene.bodies.size());
            for (const auto& Bc : settings.scene.bodies) rods.push_back(mkRod(Bc));
        } else {
            // fallback: two defaults if scene is empty
            BodyCfg a{}, b{};
            a.pos = {-1.6f,0.6f,0.0f}; a.rot_quat = {1,0,0,0};
            a.density=1000.0f; a.length=0.5f; a.diameter=0.10f; a.restitution=0.15f; a.friction=0.6f; a.v_lin={+2.2f,0,0};
            b.pos = {+1.2f,1.0f,0.2f}; b.rot_quat = {1,0,0,0};
            b.density=1000.0f; b.length=0.5f; b.diameter=0.10f; b.restitution=0.15f; b.friction=0.6f; b.v_lin={-1.0f,0,0};
            rods.push_back(mkRod(a));
            rods.push_back(mkRod(b));
        }
    }

    // ---- callbacks
    static void keyCB(GLFWwindow* w, int key, int, int action, int) {
        if (action != GLFW_PRESS) return;
        auto* self = static_cast<App*>(glfwGetWindowUserPointer(w));
        switch (key) {
            case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(w, 1); break;
            case GLFW_KEY_SPACE:  self->paused = !self->paused;   break;
            case GLFW_KEY_R:      self->resetScene();             break;
            case GLFW_KEY_V:
                self->vsync = !self->vsync;
                glfwSwapInterval(self->vsync ? 1 : 0);
                break;
            default: break;
        }
    }
    static void cursorCB(GLFWwindow* w, double x, double y) {
        auto* s = static_cast<App*>(glfwGetWindowUserPointer(w));
        if (!s->dragging) { s->lastX=x; s->lastY=y; return; }
        float dx = float(x - s->lastX), dy = float(y - s->lastY);
        s->lastX = x; s->lastY = y;
        s->cam.yaw   -= dx * 0.005f;
        s->cam.pitch -= dy * 0.005f;
        if (s->cam.pitch < -1.2f) s->cam.pitch = -1.2f;
        if (s->cam.pitch > +1.2f) s->cam.pitch = +1.2f;
    }
    static void mouseCB(GLFWwindow* w, int button, int action, int) {
        auto* s = static_cast<App*>(glfwGetWindowUserPointer(w));
        if (button == GLFW_MOUSE_BUTTON_LEFT) s->dragging = (action == GLFW_PRESS);
    }
    static void scrollCB(GLFWwindow* w, double, double dy) {
        auto* s = static_cast<App*>(glfwGetWindowUserPointer(w));
        s->cam.dist *= std::exp(-0.1f * float(dy));
        if (s->cam.dist < 2.0f)  s->cam.dist = 2.0f;
        if (s->cam.dist > 30.0f) s->cam.dist = 30.0f;
    }

    // ---- sim
    struct Hit { int a=-1, b=-1; Contact c{}; }; // b=-1 means floor

    void physicsStep() {
        // integrate all rods
        for (auto& r : rods) integrate(r, gravity, dt);

        // collect contacts
        std::vector<Hit> hits;
        hits.reserve(rods.size()*2);

        const int N = (int)rods.size();
        // rod-rod
        for (int i=0;i<N;i++)
            for (int j=i+1;j<N;j++)
                if (Contact c = collideCapsuleCapsule(rods[i], rods[j]); c.hit)
                    hits.push_back({i,j,c});
        // rod-floor
        for (int i=0;i<N;i++)
            if (Contact c = collideCapsuleFloor(rods[i], floorRB); c.hit)
                hits.push_back({i,-1,c});

        // velocity solve
        for (int it=0; it<solver.velIters; ++it)
            for (auto& h : hits)
                (h.b>=0) ? applyImpulse(rods[h.a], rods[h.b], h.c)
                         : applyImpulse(rods[h.a], floorRB,   h.c);

        // positional correction
        for (auto& h : hits)
            (h.b>=0) ? positionalCorrection(rods[h.a], rods[h.b], h.c, solver)
                     : positionalCorrection(rods[h.a], floorRB,   h.c, solver);
    }

    void renderFrame() {
        int w,h; glfwGetFramebufferSize(window, &w, &h);
        glViewport(0,0,w,h);
        glClearColor(settings.render.bg.r, settings.render.bg.g, settings.render.bg.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        float aspect = (h>0) ? float(w)/float(h) : 1.0f;
        glm::mat4 P = glm::perspective(glm::radians(50.0f), aspect, 0.05f, 100.0f);
        glm::mat4 V = cam.view();

        RenderUniforms U;
        U.P = P; U.V = V;
        U.eye      = cam.eye();
        U.lightDir = glm::normalize(settings.render.lightDir);
        U.useGrid  = settings.render.grid.enabled;
        U.gridScale= settings.render.grid.scale;
        U.gridC1   = settings.render.grid.c1;
        U.gridC2   = settings.render.grid.c2;

        // palette
        static const glm::vec3 kPalette[] = {
            {0.30f,0.70f,1.00f},
            {1.00f,0.55f,0.25f},
            {0.60f,0.90f,0.40f},
            {0.90f,0.40f,0.80f},
            {0.95f,0.85f,0.30f},
        };
        const int K = int(sizeof(kPalette)/sizeof(kPalette[0]));

        // draw rods
        for (size_t i=0;i<rods.size();++i){
            U.M = rods[i].modelMatrix();
            U.color = kPalette[i % K];
            U.useGrid = false;
            rnd.draw(cyl, U);
        }

        // draw floor
        U.M = floorRB.modelMatrix(); U.useGrid = true; U.color = {1,1,1};
        rnd.draw(cube, U);
    }

    int run() {
        if (!initWindow()) return -1;
        if (!initGfx())    return -1;
        resetScene();

        auto last = std::chrono::high_resolution_clock::now();
        double acc = 0.0;
        while (!glfwWindowShouldClose(window)) {
            auto now = std::chrono::high_resolution_clock::now();
            double dtReal = std::chrono::duration<double>(now - last).count();
            last = now;

            acc = std::min(acc + dtReal, 1.0/15.0);
            while (acc >= dt) {
                if (!paused) physicsStep();
                acc -= dt;
            }

            renderFrame();
            glfwSwapBuffers(window);
            glfwPollEvents();
        }
        glfwDestroyWindow(window);
        glfwTerminate();
        return 0;
    }
};

int main(int argc, char** argv){
    std::string scenePath = std::string(ASSETS_DIR) + "/scenes/default.json";
    for (int i=1;i<argc;i++) if (std::string(argv[i]) == "--scene" && i+1<argc) scenePath = argv[++i];

    AppCfg settings = defaultAppCfg();
    (void)loadConfigFromFile(scenePath, settings); // keep defaults if load fails

    App app;
    app.settings = settings;
    app.cam.yaw   = settings.render.yaw;
    app.cam.pitch = settings.render.pitch;
    app.cam.dist  = settings.render.dist;
    return app.run();
}
