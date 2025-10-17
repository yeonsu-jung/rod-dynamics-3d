/**
 * @file main.cpp
 * @brief 3D Rod Dynamics Simulation - Main Application
 */

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "physics/rigid_body.hpp"
#include "physics/collision.hpp"
#include "physics/solver.hpp"
#include "physics/integrator.hpp"

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

class App {
public:
    App() = default;
    ~App() = default;

    int run();
    void setConfig(const AppCfg& config);

private:
    // ---- Window and OpenGL ----
    GLFWwindow* window = nullptr;
    bool vsync = true;

    // ---- Renderer and meshes ----
    Renderer rnd;
    Mesh cube, cyl;

    // ---- Camera ----
    OrbitCamera cam;
    bool dragging = false;
    double lastX = 0.0, lastY = 0.0;

    // ---- Simulation ----
    bool paused = false;
    glm::vec3 gravity{0.0f, -10.0f, 0.0f};
    float dt = 1.0f / 600.0f;
    AppCfg settings{};
    SolverConfig solver{};

    // ---- Physics objects ----
    std::vector<RigidBody> rods;
    RigidBody floorRB;

    // ---- Initialization ----
    bool initWindow(int width = 1200, int height = 800, 
                   const char* title = "Rigid Bodies – Rods (Capsules)");
    bool initGraphics();

    // ---- Scene management ----
    static RigidBody createRod(const BodyCfg& config);
    void resetScene();

    // ---- Event callbacks ----
    static void keyCB(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void cursorCB(GLFWwindow* window, double x, double y);
    static void mouseCB(GLFWwindow* window, int button, int action, int mods);
    static void scrollCB(GLFWwindow* window, double xoffset, double yoffset);

    // ---- Simulation ----
    struct Hit { 
        int a = -1, b = -1; 
        Contact c{}; 
    };
    
    void physicsStep();
    void renderFrame();
};

// ---- Implementation ----

bool App::initWindow(int width, int height, const char* title) {
    if (!glfwInit()) { 
        std::cerr << "GLFW init failed\n"; 
        return false; 
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, std::max(1, settings.render.msaa_samples));
    
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window) { 
        std::cerr << "GLFW window creation failed\n"; 
        glfwTerminate(); 
        return false; 
    }
    
    glfwMakeContextCurrent(window);

    vsync = settings.render.vsync;
    glfwSwapInterval(vsync ? 1 : 0);

    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) { 
        std::cerr << "GLAD load failed\n"; 
        return false; 
    }

    // OpenGL state setup
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    
    if (settings.render.cull) { 
        glEnable(GL_CULL_FACE); 
        glCullFace(GL_BACK); 
    } else { 
        glDisable(GL_CULL_FACE); 
    }
    
    glDisable(GL_BLEND);

#ifdef GLAD_GL_KHR_debug
    if (GLAD_GL_KHR_debug) { 
        glEnable(GL_DEBUG_OUTPUT); 
        glDebugMessageCallback(glDebugCallback, nullptr); 
    }
#endif

    // Set up event callbacks
    glfwSetWindowUserPointer(window, this);
    glfwSetKeyCallback(window, &App::keyCB);
    glfwSetCursorPosCallback(window, &App::cursorCB);
    glfwSetMouseButtonCallback(window, &App::mouseCB);
    glfwSetScrollCallback(window, &App::scrollCB);
    
    return true;
}

bool App::initGraphics() {
    if (!rnd.init(ASSETS_DIR)) {
        std::cerr << "Renderer init failed (check " << ASSETS_DIR << "/shaders).\n";
        return false;
    }
    cube = makeCubeMesh();
    cyl = makeCappedCylinderMesh(40);
    return true;
}

void App::setConfig(const AppCfg& config) {
    settings = config;
    cam.yaw = settings.render.yaw;
    cam.pitch = settings.render.pitch;
    cam.dist = settings.render.dist;
}

RigidBody App::createRod(const BodyCfg& config) {
    // rot_quat is glm::vec4 {w,x,y,z} (resolved by config)
    glm::quat q(config.rot_quat.x, config.rot_quat.y, config.rot_quat.z, config.rot_quat.w);
    RigidBody rb = RigidBody::makeRodLD(config.pos, q, config.density, config.length, 
                                       config.diameter, config.restitution, config.friction);
    rb.v = config.v_lin; 
    rb.w = config.v_ang;
    return rb;
}

void App::resetScene() {
    dt = settings.physics.dt;
    gravity = settings.physics.gravity;
    solver = settings.physics.solver;

    g_lin_damp = settings.physics.lin_damp;
    g_ang_damp = settings.physics.ang_damp;
    g_w_max = settings.physics.w_max;

    const auto& floorConfig = settings.scene.floor;
    glm::quat qF(floorConfig.rot_quat.x, floorConfig.rot_quat.y, 
                 floorConfig.rot_quat.z, floorConfig.rot_quat.w);
    floorRB = RigidBody::makeStaticFloor(
        floorConfig.pos, qF, 
        floorConfig.half_extents.x, floorConfig.half_extents.y, floorConfig.half_extents.z,
            floorConfig.restitution, floorConfig.friction
    );

    rods.clear();
    if (!settings.scene.bodies.empty()) {
        rods.reserve(settings.scene.bodies.size());
        for (const auto& bodyConfig : settings.scene.bodies) {
            rods.push_back(createRod(bodyConfig));
        }
    } else {
        // Fallback: two default rods if scene is empty
        BodyCfg rodA{}, rodB{};
        
        rodA.pos = {-1.6f, 0.6f, 0.0f}; 
        rodA.rot_quat = {1, 0, 0, 0};
        rodA.density = 1000.0f; 
        rodA.length = 0.5f; 
        rodA.diameter = 0.10f; 
        rodA.restitution = 0.15f; 
        rodA.friction = 0.6f; 
        rodA.v_lin = {+2.2f, 0, 0};
        
        rodB.pos = {+1.2f, 1.0f, 0.2f}; 
        rodB.rot_quat = {1, 0, 0, 0};
        rodB.density = 1000.0f; 
        rodB.length = 0.5f; 
        rodB.diameter = 0.10f; 
        rodB.restitution = 0.15f; 
        rodB.friction = 0.6f; 
        rodB.v_lin = {-1.0f, 0, 0};
        
        rods.push_back(createRod(rodA));
        rods.push_back(createRod(rodB));
    }
}

// ---- Event Callbacks ----

void App::keyCB(GLFWwindow* window, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    
    auto* self = static_cast<App*>(glfwGetWindowUserPointer(window));
    switch (key) {
        case GLFW_KEY_ESCAPE: 
            glfwSetWindowShouldClose(window, 1); 
            break;
        case GLFW_KEY_SPACE:  
            self->paused = !self->paused;   
            break;
        case GLFW_KEY_R:      
            self->resetScene();             
            break;
        case GLFW_KEY_V:
            self->vsync = !self->vsync;
            glfwSwapInterval(self->vsync ? 1 : 0);
            break;
        default: 
            break;
    }
}

void App::cursorCB(GLFWwindow* window, double x, double y) {
    auto* self = static_cast<App*>(glfwGetWindowUserPointer(window));
    if (!self->dragging) { 
        self->lastX = x; 
        self->lastY = y; 
        return; 
    }
    
    float dx = float(x - self->lastX);
    float dy = float(y - self->lastY);
    self->lastX = x; 
    self->lastY = y;
    
    self->cam.yaw -= dx * 0.005f;
    self->cam.pitch -= dy * 0.005f;
    
    // Clamp pitch to prevent over-rotation
    if (self->cam.pitch < -1.2f) self->cam.pitch = -1.2f;
    if (self->cam.pitch > +1.2f) self->cam.pitch = +1.2f;
}

void App::mouseCB(GLFWwindow* window, int button, int action, int) {
    auto* self = static_cast<App*>(glfwGetWindowUserPointer(window));
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        self->dragging = (action == GLFW_PRESS);
    }
}

void App::scrollCB(GLFWwindow* window, double, double dy) {
    auto* self = static_cast<App*>(glfwGetWindowUserPointer(window));
    self->cam.dist *= std::exp(-0.1f * float(dy));
    
    // Clamp camera distance
    if (self->cam.dist < 2.0f)  self->cam.dist = 2.0f;
    if (self->cam.dist > 30.0f) self->cam.dist = 30.0f;
}

// ---- Simulation ----

void App::physicsStep() {
    // Integrate all rods
    for (auto& rod : rods) {
        integrate(rod, gravity, dt);
    }

    // Collect contacts
    std::vector<Hit> hits;
    hits.reserve(rods.size() * 2);

    const int numRods = static_cast<int>(rods.size());
    
    // Rod-rod collisions
    for (int i = 0; i < numRods; i++) {
        for (int j = i + 1; j < numRods; j++) {
            if (Contact contact = collideCapsuleCapsule(rods[i], rods[j]); contact.hit) {
                hits.push_back({i, j, contact});
            }
        }
    }
    
    // Rod-floor collisions
    for (int i = 0; i < numRods; i++) {
        if (Contact contact = collideCapsuleFloor(rods[i], floorRB); contact.hit) {
            hits.push_back({i, -1, contact}); // -1 indicates floor collision
        }
    }

    // Velocity solving iterations
    for (int iteration = 0; iteration < solver.velIters; ++iteration) {
        for (auto& hit : hits) {
            if (hit.b >= 0) {
                applyImpulse(rods[hit.a], rods[hit.b], hit.c);
            } else {
                applyImpulse(rods[hit.a], floorRB, hit.c);
            }
        }
    }

    // Positional correction
    for (auto& hit : hits) {
        if (hit.b >= 0) {
            positionalCorrection(rods[hit.a], rods[hit.b], hit.c, solver);
        } else {
            positionalCorrection(rods[hit.a], floorRB, hit.c, solver);
        }
    }
}

void App::renderFrame() {
    int width, height; 
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);
    glClearColor(settings.render.bg.r, settings.render.bg.g, settings.render.bg.b, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    float aspect = (height > 0) ? float(width) / float(height) : 1.0f;
    glm::mat4 projection = glm::perspective(glm::radians(50.0f), aspect, 0.05f, 100.0f);
    glm::mat4 view = cam.view();

    RenderUniforms uniforms;
    uniforms.P = projection; 
    uniforms.V = view;
    uniforms.eye = cam.eye();
    uniforms.lightDir = glm::normalize(settings.render.lightDir);
    uniforms.useGrid = settings.render.grid.enabled;
    uniforms.gridScale = settings.render.grid.scale;
    uniforms.gridC1 = settings.render.grid.c1;
    uniforms.gridC2 = settings.render.grid.c2;

    // Color palette for different rods
    static const glm::vec3 rodColors[] = {
        {0.30f, 0.70f, 1.00f}, // Blue
        {1.00f, 0.55f, 0.25f}, // Orange
        {0.60f, 0.90f, 0.40f}, // Green
        {0.90f, 0.40f, 0.80f}, // Purple
        {0.95f, 0.85f, 0.30f}, // Yellow
    };
    constexpr int numColors = sizeof(rodColors) / sizeof(rodColors[0]);

    // Draw rods
    for (size_t i = 0; i < rods.size(); ++i) {
        uniforms.M = rods[i].modelMatrix();
        uniforms.color = rodColors[i % numColors];
        uniforms.useGrid = false;
        rnd.draw(cyl, uniforms);
    }

    // Draw floor
    uniforms.M = floorRB.modelMatrix(); 
    uniforms.useGrid = true; 
    uniforms.color = {1.0f, 1.0f, 1.0f};
    rnd.draw(cube, uniforms);
}

int App::run() {
    if (!initWindow()) return -1;
    if (!initGraphics()) return -1;
    resetScene();

    auto lastTime = std::chrono::high_resolution_clock::now();
    double accumulator = 0.0;
    
    while (!glfwWindowShouldClose(window)) {
        auto currentTime = std::chrono::high_resolution_clock::now();
        double deltaTime = std::chrono::duration<double>(currentTime - lastTime).count();
        lastTime = currentTime;

        // Limit frame time to prevent spiral of death
        accumulator = std::min(accumulator + deltaTime, 1.0 / 15.0);
        
        while (accumulator >= dt) {
            if (!paused) {
                physicsStep();
            }
            accumulator -= dt;
        }

        renderFrame();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

// ---- Main Function ----

int main(int argc, char** argv) {
    std::string scenePath = std::string(ASSETS_DIR) + "/scenes/default.json";
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--scene" && i + 1 < argc) {
            scenePath = argv[++i];
        }
    }

    AppCfg settings = defaultAppCfg();
    
    // Load scene configuration (keep defaults if load fails)
    if (!loadConfigFromFile(scenePath, settings)) {
        std::cerr << "Warning: Could not load scene file '" << scenePath 
                  << "', using defaults.\n";
    }

    App app;
    app.setConfig(settings);
    
    return app.run();
}
