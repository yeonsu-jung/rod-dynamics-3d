// src/app/main.cpp
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>
#include <iostream>
#include <cmath>

#include "physics/rigid_body.hpp"
#include "physics/collision.hpp"
#include "physics/solver.hpp"
#include "physics/integrator.hpp"

#include "gfx/renderer.hpp"
#include "gfx/mesh.hpp"
#include "gfx/camera.hpp"

#ifndef ASSETS_DIR
#define ASSETS_DIR "."
#endif

// Optional GL debug callback (works if KHR_debug is present)
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

    // ---- renderer
    Renderer rnd;
    Mesh cube, cyl;

    // ---- camera
    OrbitCamera cam;
    bool dragging = false;
    double lastX = 0.0, lastY = 0.0;

    // ---- sim
    bool paused = false;
    glm::vec3 gravity{0.0f, -10.0f, 0.0f};
    float dt = 1.0f / 600.0f;            // fixed substep
    SolverConfig cfg{0.25f, 0.003f, 30};

    // rods & floor
    RigidBody A, B, G;

    // ---- lifecycle
    bool initWindow(int W=1200, int H=800, const char* title="Rigid Bodies – Rods (Capsules)") {
        if (!glfwInit()) {
            std::cerr << "GLFW init failed\n";
            return false;
        }
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    #ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    #endif
        window = glfwCreateWindow(W, H, title, nullptr, nullptr);
        if (!window) {
            std::cerr << "GLFW window creation failed\n";
            glfwTerminate();
            return false;
        }
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            std::cerr << "GLAD load failed\n";
            return false;
        }

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_MULTISAMPLE);
        glDisable(GL_CULL_FACE);
        glDisable(GL_BLEND);

    #ifdef GLAD_GL_KHR_debug
        if (GLAD_GL_KHR_debug) {
            glEnable(GL_DEBUG_OUTPUT);
            glDebugMessageCallback(glDebugCallback, nullptr);
        }
    #endif

        // input callbacks
        glfwSetWindowUserPointer(window, this);
        glfwSetKeyCallback(window, &App::keyCB);
        glfwSetCursorPosCallback(window, &App::cursorCB);
        glfwSetMouseButtonCallback(window, &App::mouseCB);
        glfwSetScrollCallback(window, &App::scrollCB);
        return true;
    }

    bool initGfx() {
        if (!rnd.init(ASSETS_DIR)) {
            std::cerr << "Renderer init failed (check shaders path).\n";
            return false;
        }
        cube = makeCubeMesh();
        cyl  = makeCappedCylinderMesh(40);
        return true;
    }

    void resetScene() {
        // Length & diameter (meters)
        const float L = 1.0f;
        const float D = 0.10f;
        const float density = 1000.0f; // kg/m^3 (plastic-ish)

        A = RigidBody::makeRodLD({-1.6f, 0.6f, 0.0f},
                                 glm::angleAxis(+0.35f, glm::normalize(glm::vec3(0,1,1))),
                                 density, L, D, 0.15f, 0.6f);
        B = RigidBody::makeRodLD({ +1.2f, 1.0f, 0.2f},
                                 glm::angleAxis(-0.25f, glm::vec3(1,0,0)),
                                 density, 0.8f*L, D, 0.15f, 0.6f);
        G = RigidBody::makeStaticFloor({0.0f, -0.8f, 0.0f},
                                       glm::quat(1,0,0,0),
                                       /*hx*/10.0f, /*hy*/0.1f, /*hz*/10.0f,
                                       0.3f, 0.9f);

        A.v = {+2.2f, 0.0f, 0.0f};
        B.v = {-1.0f, 0.0f, 0.0f};
        A.w = B.w = glm::vec3(0.0f);
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
    void physicsStep() {
        integrate(A, gravity, dt);
        integrate(B, gravity, dt);

        if (Contact cAB = collideCapsuleCapsule(A,B); cAB.hit) {
            for (int i=0; i<cfg.velIters; ++i) applyImpulse(A,B,cAB);
            positionalCorrection(A,B,cAB,cfg);
        }
        if (Contact cAG = collideCapsuleFloor(A,G); cAG.hit) {
            for (int i=0; i<cfg.velIters; ++i) applyImpulse(A,G,cAG);
            positionalCorrection(A,G,cAG,cfg);
        }
        if (Contact cBG = collideCapsuleFloor(B,G); cBG.hit) {
            for (int i=0; i<cfg.velIters; ++i) applyImpulse(B,G,cBG);
            positionalCorrection(B,G,cBG,cfg);
        }
    }

    // ---- draw
    void renderFrame() {
        int w,h; glfwGetFramebufferSize(window, &w, &h);
        glViewport(0,0,w,h);
        glClearColor(0.08f, 0.09f, 0.11f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        float aspect = (h>0) ? float(w)/float(h) : 1.0f;
        glm::mat4 P = glm::perspective(glm::radians(50.0f), aspect, 0.05f, 100.0f);
        glm::mat4 V = cam.view();

        RenderUniforms U;
        U.P = P; U.V = V;
        U.eye = cam.eye();
        U.lightDir = glm::normalize(glm::vec3(-0.4f, -1.0f, -0.3f));

        // rods
        U.M = A.modelMatrix(); U.color = {0.30f,0.70f,1.00f}; U.useGrid=false;
        rnd.draw(cyl, U);
        U.M = B.modelMatrix(); U.color = {1.00f,0.55f,0.25f}; U.useGrid=false;
        rnd.draw(cyl, U);

        // floor (checker)
        U.M = G.modelMatrix(); U.useGrid = true; U.color = {1,1,1};
        U.gridScale = 1.0f;
        U.gridC1 = {0.80f,0.82f,0.85f};
        U.gridC2 = {0.65f,0.67f,0.70f};
        rnd.draw(cube, U);
    }

    int run() {
        if (!initWindow()) return -1;
        if (!initGfx())    return -1;
        resetScene();

        // fixed-timestep accumulator
        auto last = std::chrono::high_resolution_clock::now();
        double acc = 0.0;

        while (!glfwWindowShouldClose(window)) {
            auto now = std::chrono::high_resolution_clock::now();
            double dtReal = std::chrono::duration<double>(now - last).count();
            last = now;

            // avoid spiral of death
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

int main() {
    App app;
    return app.run();
}
