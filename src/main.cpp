#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
// #include <glm/gtx/quaternion.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>

/* ===============================
   3D Rigid Body & Collision (OBB)
   =============================== */

struct Box { float hx{0.5f}, hy{0.3f}, hz{0.4f}; };

struct RigidBody {
    // State
    glm::vec3  x{0.0f};        // position
    glm::quat  q{1,0,0,0};     // orientation (w,x,y,z)
    glm::vec3  v{0.0f};        // linear velocity
    glm::vec3  w{0.0f};        // angular velocity (world)

    // Properties
    float      mass{1.0f};
    float      invMass{1.0f};
    glm::mat3  I_body{1.0f};       // inertia in body frame
    glm::mat3  I_body_inv{1.0f};   // inverse inertia in body
    float      restitution{0.35f};
    float      friction{0.6f};
    Box        shape;

    static RigidBody makeBox(const glm::vec3& pos, const glm::quat& q,
                             float density, float hx, float hy, float hz,
                             float restitution=0.35f, float friction=0.6f)
    {
        RigidBody b;
        b.x = pos;
        b.q = glm::normalize(q);
        b.shape.hx = hx; b.shape.hy = hy; b.shape.hz = hz;

        const float W = 2*hx, H = 2*hy, D = 2*hz;
        const float volume = W*H*D;
        b.mass    = std::max(1e-6f, density*volume);
        b.invMass = 1.0f / b.mass;

        // Box inertia about center
        const float ix = b.mass * (H*H + D*D) / 12.0f;
        const float iy = b.mass * (W*W + D*D) / 12.0f;
        const float iz = b.mass * (W*W + H*H) / 12.0f;
        b.I_body     = glm::mat3(1.0f);
        b.I_body[0][0] = ix; b.I_body[1][1] = iy; b.I_body[2][2] = iz;
        b.I_body_inv = glm::mat3(0.0f);
        b.I_body_inv[0][0] = 1.0f/ix; b.I_body_inv[1][1] = 1.0f/iy; b.I_body_inv[2][2] = 1.0f/iz;

        b.restitution = restitution;
        b.friction    = friction;
        return b;
    }

    glm::mat3 R() const { return glm::mat3_cast(q); }

    // World inverse inertia: R * I_body_inv * R^T
    glm::mat3 IworldInv() const {
        glm::mat3 Rm = R();
        return Rm * I_body_inv * glm::transpose(Rm);
    }

    glm::mat4 modelMatrix() const {
        glm::mat4 M = glm::translate(glm::mat4(1.0f), x) * glm::mat4_cast(q);
        return glm::scale(M, glm::vec3(shape.hx, shape.hy, shape.hz));
    }
};

// support point (extreme vertex) of OBB along direction d
static glm::vec3 supportOBB(const RigidBody& b, const glm::vec3& dWorld)
{
    glm::mat3 R = b.R();
    // local axes in world
    glm::vec3 ax = R[0]; // column 0
    glm::vec3 ay = R[1];
    glm::vec3 az = R[2];

    glm::vec3 s = glm::vec3(
        (glm::dot(ax, dWorld) >= 0.0f) ? 1.0f : -1.0f,
        (glm::dot(ay, dWorld) >= 0.0f) ? 1.0f : -1.0f,
        (glm::dot(az, dWorld) >= 0.0f) ? 1.0f : -1.0f
    );
    return b.x + ax * (s.x * b.shape.hx) + ay * (s.y * b.shape.hy) + az * (s.z * b.shape.hz);
}

struct Contact {
    bool       hit{false};
    glm::vec3  normal{0};    // from A to B
    float      penetration{0};
    glm::vec3  point{0};
};

// Project OBB onto axis L (unit) → radius
static float projectRadius(const RigidBody& B, const glm::vec3& L)
{
    glm::mat3 R = B.R();
    float r =
        B.shape.hx * std::abs(glm::dot(L, R[0])) +
        B.shape.hy * std::abs(glm::dot(L, R[1])) +
        B.shape.hz * std::abs(glm::dot(L, R[2]));
    return r;
}

static bool axisTest(const RigidBody& A, const RigidBody& B, const glm::vec3& L,
                     const glm::vec3& t, float& minOverlap, glm::vec3& bestAxis)
{
    float len2 = glm::dot(L, L);
    if (len2 < 1e-10f) return true; // ignore near-zero axis (parallel)
    glm::vec3 axis = glm::normalize(L);
    float rA = projectRadius(A, axis);
    float rB = projectRadius(B, axis);
    float d  = std::abs(glm::dot(t, axis));
    float overlap = rA + rB - d;
    if (overlap < 0.0f) return false;
    if (overlap < minOverlap){
        minOverlap = overlap;
        bestAxis = axis;
    }
    return true;
}

// OBB vs OBB (SAT with 15 axes)
static Contact collideOBB(const RigidBody& A, const RigidBody& B)
{
    Contact c;
    glm::mat3 Ra = A.R();
    glm::mat3 Rb = B.R();
    glm::vec3 t  = B.x - A.x;

    float minOv = 1e30f;
    glm::vec3 bestAxis(0);

    // A face normals (Ra columns)
    for (int i=0;i<3;++i) if (!axisTest(A,B, Ra[i], t, minOv, bestAxis)) return c;
    // B face normals
    for (int i=0;i<3;++i) if (!axisTest(A,B, Rb[i], t, minOv, bestAxis)) return c;
    // 9 cross axes
    for (int i=0;i<3;++i) for (int j=0;j<3;++j){
        glm::vec3 L = glm::cross(Ra[i], Rb[j]);
        if (!axisTest(A,B, L, t, minOv, bestAxis)) return c;
    }

    // Orient normal from A → B
    if (glm::dot(bestAxis, t) < 0.0f) bestAxis = -bestAxis;

    // Single contact point via support points (simple & stable)
    glm::vec3 pA = supportOBB(A,  bestAxis);
    glm::vec3 pB = supportOBB(B, -bestAxis);
    c.hit = true;
    c.normal = bestAxis;
    c.penetration = minOv;
    c.point = 0.5f * (pA + pB);
    return c;
}

// Impulse with restitution + Coulomb friction (tangent along instantaneous slip)
static void applyImpulse(RigidBody& A, RigidBody& B, const Contact& c)
{
    glm::vec3 rA = c.point - A.x;
    glm::vec3 rB = c.point - B.x;

    glm::vec3 vA = A.v + glm::cross(A.w, rA);
    glm::vec3 vB = B.v + glm::cross(B.w, rB);
    glm::vec3 rv = vB - vA;

    float rvn = glm::dot(rv, c.normal);
    if (rvn > 0.0f) return;

    float e = std::min(A.restitution, B.restitution);

    glm::mat3 IA = A.IworldInv();
    glm::mat3 IB = B.IworldInv();

    auto K_scalar = [&](const glm::vec3& n){
        glm::vec3 rnA = glm::cross(rA, n);
        glm::vec3 rnB = glm::cross(rB, n);
        float k = A.invMass + B.invMass
                + glm::dot(n, glm::cross(IA * rnA, rA))
                + glm::dot(n, glm::cross(IB * rnB, rB));
        return k;
    };

    float kN = K_scalar(c.normal);
    float j = -(1.0f + e) * rvn / kN;
    glm::vec3 impulseN = j * c.normal;

    A.v -= impulseN * A.invMass;  B.v += impulseN * B.invMass;
    A.w -= IA * glm::cross(rA, impulseN);
    B.w += IB * glm::cross(rB, impulseN);

    // recompute relative velocity post normal impulse
    vA = A.v + glm::cross(A.w, rA);
    vB = B.v + glm::cross(B.w, rB);
    rv = vB - vA;

    // tangent dir = slip direction
    glm::vec3 t = rv - c.normal * glm::dot(rv, c.normal);
    float tlen = glm::length(t);
    if (tlen < 1e-8f) return;
    t /= tlen;

    float kT = K_scalar(t);
    float jt = -glm::dot(rv, t) / kT;

    float mu = 0.5f*(A.friction + B.friction);
    jt = glm::clamp(jt, -mu*j, mu*j);

    glm::vec3 impulseT = jt * t;
    A.v -= impulseT * A.invMass;  B.v += impulseT * B.invMass;
    A.w -= IA * glm::cross(rA, impulseT);
    B.w += IB * glm::cross(rB, impulseT);
}

struct SolverConfig{ float baumgarte=0.2f, allowedPen=0.002f; int velIters=30; };

static void positionalCorrection(RigidBody& A, RigidBody& B, const Contact& c, const SolverConfig& cfg)
{
    float k = std::max(0.0f, c.penetration - cfg.allowedPen);
    if (k <= 0) return;
    glm::vec3 corr = (cfg.baumgarte * k) * c.normal / (A.invMass + B.invMass);
    A.x -= A.invMass * corr;
    B.x += B.invMass * corr;
}

static void integrate(RigidBody& b, const glm::vec3& g, float dt)
{
    // linear
    b.v += g * dt;
    b.x += b.v * dt;

    // angular (world w), quaternion derivative: dq/dt = 0.5 * (0,w) * q
    glm::quat dq(0.0f, b.w.x, b.w.y, b.w.z);
    b.q = glm::normalize(b.q + 0.5f * dq * b.q * dt);
}

/* ==============
   OpenGL Viewer
   ============== */

#ifdef GLAD_GL_KHR_debug
static void GLAPIENTRY glDebugCallback(GLenum, GLenum, GLuint, GLenum sev, GLsizei, const GLchar* msg, const void*)
{
    if (sev == GL_DEBUG_SEVERITY_NOTIFICATION) return;
    std::fprintf(stderr, "[GL] %s\n", msg);
}
#endif

static const char* kVS = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNor;

uniform mat4 uProj, uView, uModel;

out VS_OUT {
    vec3 posW;
    vec3 norW;
} vs_out;

void main(){
    mat3 normalMat = transpose(inverse(mat3(uModel)));
    vs_out.posW = vec3(uModel * vec4(aPos,1.0));
    vs_out.norW = normalize(normalMat * aNor);
    gl_Position = uProj * uView * vec4(vs_out.posW,1.0);
}
)";


static const char* kFS = R"(
#version 330 core
out vec4 FragColor;

in VS_OUT { vec3 posW; vec3 norW; } fs_in;

uniform vec3 uColor;      // base albedo for normal objects
uniform vec3 uLightDir;   // direction light (direction of travel)
uniform vec3 uEyePos;     // camera position

// Checker controls (used only for the floor)
uniform int   uUseGrid;        // 0 = off, 1 = on
uniform float uGridScale;      // tiles per world unit (e.g., 1.0 → 1x1)
uniform vec3  uGridColor1;     // colors for checker
uniform vec3  uGridColor2;

void main(){
    vec3 N = normalize(fs_in.norW);
    vec3 L = normalize(-uLightDir);
    vec3 V = normalize(uEyePos - fs_in.posW);
    vec3 H = normalize(L + V);

    float NdotL = max(dot(N,L), 0.0);
    float spec  = pow(max(dot(N,H), 0.0), 32.0);

    vec3 baseColor = uColor;

    if (uUseGrid == 1) {
        // Procedural checker in XZ
        vec2 uv = fs_in.posW.xz * uGridScale;
        float cell = mod(floor(uv.x) + floor(uv.y), 2.0);
        baseColor = mix(uGridColor1, uGridColor2, cell);
    }

    vec3 ambient  = 0.20 * baseColor;
    vec3 diffuse  = 0.70 * NdotL * baseColor;
    vec3 specular = 0.30 * spec * vec3(1.0);

    FragColor = vec4(ambient + diffuse + specular, 1.0);
}
)";



struct App {
    GLFWwindow* win{nullptr};
    GLuint prog=0, vao=0, vbo=0, ebo=0;
    GLint uProj=-1, uView=-1, uModel=-1, uColor=-1;
    
    GLint uUseGrid = -1, uGridScale = -1, uGridColor1 = -1, uGridColor2 = -1;

    GLint uLightDir = -1;
    GLint uEyePos   = -1;

    RigidBody G;  // ground

    // camera (orbit)
    float yaw = 0.6f, pitch = 0.35f, dist = 6.0f;
    double lastX=0, lastY=0; bool dragging=false;
    bool paused=false, vsync=true;

    // world
    glm::vec3 gravity{0.0f, -10.0f, 0.0f};
    float dt = 1.0f/600.0f;
    SolverConfig solver{0.25f, 0.003f, 30};

    RigidBody A = RigidBody::makeBox({-1.4f, 0.2f,  0.0f}, glm::angleAxis(+0.25f, glm::vec3(0,1,0)), 1.0f, 0.6f,0.3f,0.4f, 0.35f,0.6f);
    RigidBody B = RigidBody::makeBox({ +1.2f, 0.4f, 0.1f}, glm::angleAxis(-0.15f, glm::vec3(1,0,0)), 1.0f, 0.5f,0.5f,0.5f, 0.35f,0.6f);

    void reset(){
        A = RigidBody::makeBox({-1.4f, 0.2f,  0.0f}, glm::angleAxis(+0.25f, glm::vec3(0,1,0)), 1.0f, 0.6f,0.3f,0.4f, 0.35f,0.6f);
        B = RigidBody::makeBox({ +1.2f, 0.4f, 0.1f}, glm::angleAxis(-0.15f, glm::vec3(1,0,0)), 1.0f, 0.5f,0.5f,0.5f, 0.35f,0.6f);
        A.v = {+2.0f, 0.0f, 0.0f};  B.v = {-1.0f, 0.0f, 0.0f};
        A.w = {0.0f, 0.0f, 0.0f};   B.w = {0.0f, 0.0f, 0.0f};

        // Big, thin box centered slightly below origin
        G = RigidBody::makeBox({0.0f, -0.8f, 0.0f}, glm::quat(1,0,0,0),
                            /*density*/1.0f, /*hx*/10.0f, /*hy*/0.1f, /*hz*/10.0f,
                            /*restitution*/0.30f, /*friction*/0.8f);
        // Make it static
        G.invMass = 0.0f;
        G.I_body_inv = glm::mat3(0.0f);
        G.v = glm::vec3(0.0f);
        G.w = glm::vec3(0.0f);

    }

    static void keyCB(GLFWwindow* w, int key, int, int action, int){
        if (action != GLFW_PRESS) return;
        App* self = (App*)glfwGetWindowUserPointer(w);
        if (key==GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(w,1);
        if (key==GLFW_KEY_SPACE)  self->paused = !self->paused;
        if (key==GLFW_KEY_R)      self->reset();
        if (key==GLFW_KEY_V){ self->vsync=!self->vsync; glfwSwapInterval(self->vsync?1:0); }
    }
    static void cursorCB(GLFWwindow* w, double x, double y){
        App* s = (App*)glfwGetWindowUserPointer(w);
        if (!s->dragging){ s->lastX=x; s->lastY=y; return; }
        float dx = float(x - s->lastX), dy = float(y - s->lastY);
        s->lastX=x; s->lastY=y;
        s->yaw   -= dx * 0.005f;
        s->pitch -= dy * 0.005f;
        s->pitch = glm::clamp(s->pitch, -1.2f, 1.2f);
    }
    static void mouseCB(GLFWwindow* w, int button, int action, int){
        App* s = (App*)glfwGetWindowUserPointer(w);
        if (button==GLFW_MOUSE_BUTTON_LEFT) s->dragging = (action==GLFW_PRESS);
    }
    static void scrollCB(GLFWwindow* w, double , double dy){
        App* s = (App*)glfwGetWindowUserPointer(w);
        s->dist *= std::exp(-0.1f * float(dy));
        s->dist = glm::clamp(s->dist, 2.0f, 30.0f);
    }

    bool initGL(){
        if(!glfwInit()) return false;
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    #ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    #endif
        win = glfwCreateWindow(1200, 800, "3D Rigid Bodies – OBB Collision", nullptr, nullptr);
        if(!win){ glfwTerminate(); return false; }
        glfwMakeContextCurrent(win);
        glfwSwapInterval(1);
        if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) return false;

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_MULTISAMPLE);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_BLEND);

    #ifdef GLAD_GL_KHR_debug
        if (GLAD_GL_KHR_debug){
            glEnable(GL_DEBUG_OUTPUT);
            glDebugMessageCallback(glDebugCallback, nullptr);
        }
    #endif

        glfwSetWindowUserPointer(win, this);
        glfwSetKeyCallback(win, keyCB);
        glfwSetCursorPosCallback(win, cursorCB);
        glfwSetMouseButtonCallback(win, mouseCB);
        glfwSetScrollCallback(win, scrollCB);
        return true;
    }

    GLuint makeShader(GLenum type, const char* src){
        GLuint s = glCreateShader(type);
        glShaderSource(s,1,&src,nullptr);
        glCompileShader(s);
        GLint ok=0; glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
        if(!ok){ char log[4096]; glGetShaderInfoLog(s,4096,nullptr,log); std::cerr<<"Shader: "<<log<<"\n"; }
        return s;
    }
    void initGfx(){
        GLuint vs = makeShader(GL_VERTEX_SHADER,   kVS);
        GLuint fs = makeShader(GL_FRAGMENT_SHADER, kFS);
        prog = glCreateProgram(); glAttachShader(prog,vs); glAttachShader(prog,fs); glLinkProgram(prog);
        GLint ok=0; glGetProgramiv(prog,GL_LINK_STATUS,&ok);
        if(!ok){ char log[4096]; glGetProgramInfoLog(prog,4096,nullptr,log); std::cerr<<"Link: "<<log<<"\n"; }
        glDeleteShader(vs); glDeleteShader(fs);
        // uProj  = glGetUniformLocation(prog,"uProj");
        // uView  = glGetUniformLocation(prog,"uView");
        // uModel = glGetUniformLocation(prog,"uModel");
        // uColor = glGetUniformLocation(prog,"uColor");

        uProj      = glGetUniformLocation(prog,"uProj");
        uView      = glGetUniformLocation(prog,"uView");
        uModel     = glGetUniformLocation(prog,"uModel");
        uColor     = glGetUniformLocation(prog,"uColor");
        uLightDir  = glGetUniformLocation(prog,"uLightDir");
        uEyePos    = glGetUniformLocation(prog,"uEyePos");
        uUseGrid   = glGetUniformLocation(prog,"uUseGrid");
        uGridScale = glGetUniformLocation(prog,"uGridScale");
        uGridColor1= glGetUniformLocation(prog,"uGridColor1");
        uGridColor2= glGetUniformLocation(prog,"uGridColor2");



        glDisable(GL_CULL_FACE);
        glEnable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glCullFace(GL_BACK);

        // glEnable(GL_DEPTH_TEST);
        // glEnable(GL_CULL_FACE);
        // glCullFace(GL_BACK);
        // glFrontFace(GL_CCW);  // default, but set it explicitly



        // Unit cube [-1,1]^3 (positions only)
        const float P[] = {
            -1,-1,-1,  1,-1,-1,  1, 1,-1, -1, 1,-1,  // back
            -1,-1, 1,  1,-1, 1,  1, 1, 1, -1, 1, 1   // front
        };
        const unsigned I[] = {
            0,1,2, 0,2,3,  // back
            4,6,5, 4,7,6,  // front
            0,4,5, 0,5,1,  // bottom
            3,2,6, 3,6,7,  // top
            0,3,7, 0,7,4,  // left
            1,5,6, 1,6,2   // right
        };
        

        struct Vertex { float px,py,pz, nx,ny,nz; };
        static const Vertex CUBE[] = {
            // back (-Z)
            {-1,-1,-1,  0,0,-1}, { 1,-1,-1,  0,0,-1}, { 1, 1,-1,  0,0,-1}, {-1, 1,-1,  0,0,-1},
            // front (+Z)
            {-1,-1, 1,  0,0, 1}, { 1,-1, 1,  0,0, 1}, { 1, 1, 1,  0,0, 1}, {-1, 1, 1,  0,0, 1},
            // bottom (-Y)
            {-1,-1,-1,  0,-1,0}, { 1,-1,-1,  0,-1,0}, { 1,-1, 1,  0,-1,0}, {-1,-1, 1,  0,-1,0},
            // top (+Y)
            {-1, 1,-1,  0, 1,0}, { 1, 1,-1,  0, 1,0}, { 1, 1, 1,  0, 1,0}, {-1, 1, 1,  0, 1,0},
            // left (-X)
            {-1,-1,-1, -1,0,0}, {-1, 1,-1, -1,0,0}, {-1, 1, 1, -1,0,0}, {-1,-1, 1, -1,0,0},
            // right (+X)
            { 1,-1,-1,  1,0,0}, { 1, 1,-1,  1,0,0}, { 1, 1, 1,  1,0,0}, { 1,-1, 1,  1,0,0},
        };
        static const unsigned IDX[] = {
            0,1,2, 0,2,3,     4,5,6, 4,6,7,
            8,9,10, 8,10,11, 12,13,14, 12,14,15,
            16,17,18, 16,18,19, 20,21,22, 20,22,23
        };

        glGenVertexArrays(1,&vao);
        glGenBuffers(1,&vbo);
        glGenBuffers(1,&ebo);

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(CUBE), CUBE, GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(IDX), IDX, GL_STATIC_DRAW);

        // aPos (location=0)
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);

        // aNor (location=1)
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(3*sizeof(float)));

        glBindVertexArray(0);
    }

    void physicsStep(){
        integrate(A, gravity, dt);
        integrate(B, gravity, dt);

        Contact cAB = collideOBB(A,B);
        if (cAB.hit){
            for(int it=0; it<solver.velIters; ++it) applyImpulse(A,B,cAB);
            positionalCorrection(A,B,cAB,solver);
        }

        Contact cAG = collideOBB(A,G);
        if (cAG.hit){
            for(int it=0; it<solver.velIters; ++it) applyImpulse(A,G,cAG);
            positionalCorrection(A,G,cAG,solver);
        }

        Contact cBG = collideOBB(B,G);
        if (cBG.hit){
            for(int it=0; it<solver.velIters; ++it) applyImpulse(B,G,cBG);
            positionalCorrection(B,G,cBG,solver);
        }

    }

    glm::mat4 viewMatrix(){
        // orbit camera around origin
        glm::vec3 target(0.0f);
        float cp = std::cos(pitch), sp = std::sin(pitch);
        float cy = std::cos(yaw),   sy = std::sin(yaw);
        glm::vec3 eye = target + glm::vec3(cp*cy, sp, cp*sy) * dist;
        return glm::lookAt(eye, target, glm::vec3(0,1,0));
    }

    void drawBox(const RigidBody& b, const glm::vec3& color,
                const glm::mat4& P, const glm::mat4& V)
    {
        glUseProgram(prog);
        glUniformMatrix4fv(uProj,  1, GL_FALSE, &P[0][0]);
        glUniformMatrix4fv(uView,  1, GL_FALSE, &V[0][0]);
        glm::mat4 M = b.modelMatrix();
        glUniformMatrix4fv(uModel, 1, GL_FALSE, &M[0][0]);
        glUniform3fv(uColor, 1, &color[0]);

        glUniform1i(uUseGrid, 0); // normal shading

        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }

    void drawFloor(const RigidBody& floorBox,
                const glm::mat4& P, const glm::mat4& V)
    {
        glUseProgram(prog);
        glUniformMatrix4fv(uProj,  1, GL_FALSE, &P[0][0]);
        glUniformMatrix4fv(uView,  1, GL_FALSE, &V[0][0]);
        glm::mat4 M = floorBox.modelMatrix();
        glUniformMatrix4fv(uModel, 1, GL_FALSE, &M[0][0]);

        // Checker colors and scale
        const glm::vec3 c1(0.80f, 0.82f, 0.85f);
        const glm::vec3 c2(0.65f, 0.67f, 0.70f);
        glUniform3fv(uGridColor1, 1, &c1[0]);
        glUniform3fv(uGridColor2, 1, &c2[0]);
        glUniform1f(uGridScale, 1.0f);     // 1 tile per world unit
        glUniform1i(uUseGrid, 1);          // enable grid
        glUniform3f(uColor, 1.0f,1.0f,1.0f); // ignored when uUseGrid=1

        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        glUniform1i(uUseGrid, 0); // restore default
    }


    glm::vec3 eyePos() const {
        float cp = std::cos(pitch), sp = std::sin(pitch);
        float cy = std::cos(yaw),   sy = std::sin(yaw);
        return glm::vec3(cp*cy, sp, cp*sy) * dist; // target at origin
    }

    int run(){
        if(!initGL()) return -1;
        initGfx();
        reset();

        auto last = std::chrono::high_resolution_clock::now();
        double acc = 0.0;

        while(!glfwWindowShouldClose(win)){
            auto now = std::chrono::high_resolution_clock::now();
            double dtReal = std::chrono::duration<double>(now-last).count();
            last = now;
            acc = std::min(acc + dtReal, 1.0/15.0);
            while(acc >= dt){
                if(!paused) physicsStep();
                acc -= dt;
            }

            int w,h; glfwGetFramebufferSize(win,&w,&h);
            glViewport(0,0,w,h);
            glClearColor(0.08f,0.09f,0.11f,1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            float aspect = (h>0)? float(w)/float(h) : 1.0f;
            glm::mat4 P = glm::perspective(glm::radians(50.0f), aspect, 0.05f, 100.0f);
            glm::mat4 V = viewMatrix();

            glm::vec3 eye = eyePos();
            glm::vec3 lightDir = glm::normalize(glm::vec3(-0.4f, -1.0f, -0.3f));
            glUseProgram(prog);
            glUniform3fv(uLightDir, 1, &lightDir[0]);
            glUniform3fv(uEyePos,   1, &eye[0]);

            // draw dynamic boxes
            drawBox(A, glm::vec3(0.30f,0.70f,1.00f), P, V);
            drawBox(B, glm::vec3(1.00f,0.55f,0.25f), P, V);

            // draw checker floor last (or first; depth test handles it)
            drawFloor(G, P, V);
            

            drawBox(A, glm::vec3(0.30f,0.70f,1.00f), P, V);
            drawBox(B, glm::vec3(1.00f,0.55f,0.25f), P, V);

            glfwSwapBuffers(win);
            glfwPollEvents();
        }
        glfwDestroyWindow(win);
        glfwTerminate();
        return 0;
    }
};

int main(){ App app; return app.run(); }
