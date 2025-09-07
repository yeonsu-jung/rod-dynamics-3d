// src/main.cpp
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>

/* ===============================
   Rigid Body (Capsules) + Floor
   =============================== */

inline float length2(const glm::vec3& v){ return glm::dot(v,v); }
constexpr float PI = 3.14159265358979323846f;

// Frame-rate independent damping
constexpr float LIN_DAMP = 0.08f;  // s^-1
constexpr float ANG_DAMP = 0.12f;  // s^-1
constexpr float W_MAX    = 80.0f;  // rad/s guardrail

// Choose your rod size here:
constexpr float ROD_LENGTH_L = 1.0f;   // full length (meters)
constexpr float ROD_DIAMETER = 0.10f;  // diameter (meters)

struct Box { float hx{10.f}, hy{0.1f}, hz{10.f}; };  // only for the floor
struct Capsule { float r{0.01f}, h{1.0f}; };         // radius, half-length (cylinder half-height)

enum class ShapeType { Box, Capsule };

struct RigidBody {
    // State
    glm::vec3  x{0.0f};        // position (center)
    glm::quat  q{1,0,0,0};     // orientation (w,x,y,z)
    glm::vec3  v{0.0f};        // linear velocity
    glm::vec3  w{0.0f};        // angular velocity (world)

    // Properties
    float      mass{1.0f};
    float      invMass{1.0f};
    glm::mat3  I_body{1.0f};       // inertia in body frame
    glm::mat3  I_body_inv{1.0f};   // inverse inertia in body
    float      restitution{0.25f};
    float      friction{0.7f};

    // Shape
    ShapeType  type{ShapeType::Capsule};
    Box        box{};
    Capsule    cap{};

    static RigidBody makeCapsule(const glm::vec3& pos, const glm::quat& q,
                                 float density, float r, float h,
                                 float restitution=0.25f, float friction=0.7f)
    {
        RigidBody b;
        b.type = ShapeType::Capsule;
        b.x = pos;
        b.q = glm::normalize(q);
        b.cap = Capsule{r,h};

        // Approx mass from solid cylinder (ignoring rounded ends)
        const float H = 2*h;
        const float volume = PI * r*r * H;
        b.mass    = std::max(1e-6f, density*volume);
        b.invMass = 1.0f / b.mass;

        // Inertia tensor for a solid cylinder aligned with local Y
        // Ix = Iz = (1/12) m (3r^2 + H^2),  Iy = (1/2) m r^2
        const float Ix = b.mass * (3*r*r + H*H) / 12.0f;
        const float Iy = b.mass * (r*r) / 2.0f;
        b.I_body = glm::mat3(0.0f);
        b.I_body[0][0] = Ix; b.I_body[1][1] = Iy; b.I_body[2][2] = Ix;

        b.I_body_inv = glm::mat3(0.0f);
        b.I_body_inv[0][0] = (Ix>0)? 1.0f/Ix : 0.0f;
        b.I_body_inv[1][1] = (Iy>0)? 1.0f/Iy : 0.0f;
        b.I_body_inv[2][2] = (Ix>0)? 1.0f/Ix : 0.0f;

        b.restitution = restitution;
        b.friction    = friction;
        return b;
    }

    // Convenience: specify full Length L and Diameter D directly
    static RigidBody makeRodLD(const glm::vec3& pos, const glm::quat& q,
                               float density, float L, float D,
                               float restitution=0.25f, float friction=0.7f)
    {
        return makeCapsule(pos, q, density, /*r*/0.5f*D, /*h*/0.5f*L, restitution, friction);
    }

    static RigidBody makeStaticFloor(const glm::vec3& pos, const glm::quat& q,
                                     float hx, float hy, float hz,
                                     float restitution=0.3f, float friction=0.9f)
    {
        RigidBody b;
        b.type = ShapeType::Box;
        b.x = pos;
        b.q = glm::normalize(q);
        b.box = Box{hx,hy,hz};

        b.mass = 0.0f;
        b.invMass = 0.0f;
        b.I_body = glm::mat3(0.0f);
        b.I_body_inv = glm::mat3(0.0f);

        b.restitution = restitution;
        b.friction    = friction;
        return b;
    }

    glm::mat3 R() const { return glm::mat3_cast(q); }

    glm::mat3 IworldInv() const {
        if (invMass <= 0.0f) return glm::mat3(0.0f);
        glm::mat3 Rm = R();
        return Rm * I_body_inv * glm::transpose(Rm);
    }

    glm::mat4 modelMatrix() const {
        glm::mat4 M = glm::translate(glm::mat4(1.0f), x) * glm::mat4_cast(q);
        if (type == ShapeType::Capsule) {
            // unit capped cylinder mesh is radius=1, half-height=1 → scale to (r, h, r)
            return glm::scale(M, glm::vec3(cap.r, cap.h, cap.r));
        } else {
            return glm::scale(M, glm::vec3(box.hx, box.hy, box.hz));
        }
    }

    // Local Y axis in world (capsule axis)
    glm::vec3 axisY() const { return R()[1]; }
};

struct Contact {
    bool       hit{false};
    glm::vec3  normal{0};    // from A to B
    float      penetration{0};
    glm::vec3  point{0};
};

/* ----------------------------
   Capsule-Capsule collision
   Closest points between axes (segments)
   ---------------------------- */

static void closestPtSegmentSegment(const glm::vec3& p1, const glm::vec3& q1,
                                    const glm::vec3& p2, const glm::vec3& q2,
                                    glm::vec3& c1, glm::vec3& c2)
{
    const glm::vec3 u = q1 - p1;
    const glm::vec3 v = q2 - p2;
    const glm::vec3 w0 = p1 - p2;
    float a = glm::dot(u,u);
    float b = glm::dot(u,v);
    float c = glm::dot(v,v);
    float d = glm::dot(u,w0);
    float e = glm::dot(v,w0);
    float D = a*c - b*b;
    float sc, sN, sD = D;
    float tc, tN, tD = D;

    const float EPS = 1e-8f;

    if (D < EPS) { // parallel
        sN = 0.0f; sD = 1.0f;
        tN = e;    tD = c;
    } else {
        sN = (b*e - c*d);
        tN = (a*e - b*d);
        if (sN < 0){ sN = 0; tN = e; tD = c; }
        else if (sN > sD){ sN = sD; tN = e + b; tD = c; }
    }

    if (tN < 0) {
        tN = 0;
        if (-d < 0) sN = 0;
        else if (-d > a) sN = sD;
        else { sN = -d; sD = a; }
    } else if (tN > tD) {
        tN = tD;
        if ((-d + b) < 0) sN = 0;
        else if ((-d + b) > a) sN = sD;
        else { sN = (-d + b); sD = a; }
    }

    sc = (std::abs(sN) < EPS ? 0.0f : sN / sD);
    tc = (std::abs(tN) < EPS ? 0.0f : tN / tD);

    c1 = p1 + sc * u;
    c2 = p2 + tc * v;
}

static Contact collideCapsuleCapsule(const RigidBody& A, const RigidBody& B)
{
    Contact c;
    const glm::vec3 a = glm::normalize(A.axisY());
    const glm::vec3 b = glm::normalize(B.axisY());

    const glm::vec3 A0 = A.x - a * A.cap.h;
    const glm::vec3 A1 = A.x + a * A.cap.h;
    const glm::vec3 B0 = B.x - b * B.cap.h;
    const glm::vec3 B1 = B.x + b * B.cap.h;

    glm::vec3 pA, pB;
    closestPtSegmentSegment(A0, A1, B0, B1, pA, pB);

    glm::vec3 d = pB - pA;
    float dist = glm::length(d);
    float rsum = A.cap.r + B.cap.r;

    if (dist >= rsum) return c;

    c.hit = true;
    c.penetration = rsum - dist;
    if (dist > 1e-6f) c.normal = d / dist;
    else {
        // fallback normal if centers are nearly coincident
        glm::vec3 tmp = (B.x - A.x);
        if (glm::dot(tmp, tmp) < 1e-8f) tmp = glm::cross(a, glm::vec3(1,0,0));
        if (glm::dot(tmp, tmp) < 1e-8f) tmp = glm::vec3(0,1,0);
        c.normal = glm::normalize(tmp);
    }
    c.point = 0.5f * (pA + pB);
    return c;
}

/* ----------------------------
   Capsule - Floor (top plane of box G)
   Use closest point on segment to plane along plane normal.
   ---------------------------- */
static Contact collideCapsuleFloor(const RigidBody& C, const RigidBody& G)
{
    Contact c;
    // floor +Y (top face) normal in world
    glm::vec3 n = glm::normalize(G.R()[1]);
    if (glm::dot(n, glm::vec3(0,1,0)) < 0) n = -n;

    // plane point on top face
    glm::vec3 p0 = G.x + n * G.box.hy;

    // capsule axis and best point on its center segment
    glm::vec3 a  = C.axisY();              // unit (from rotation)
    float an     = glm::dot(a, n);

    float t; // param in [-h, h] giving point x + t*a
    if (std::abs(an) > 1e-6f) {
        // point that minimizes plane distance, clamped to segment
        t = glm::clamp(-glm::dot(C.x - p0, n) / an, -C.cap.h, C.cap.h);
    } else {
        // axis ~ parallel to floor → any t same distance; choose center
        t = 0.0f;
    }

    glm::vec3 cseg = C.x + t * a;          // closest segment point to plane along n
    float d = glm::dot(cseg - p0, n) - C.cap.r;  // signed (negative => penetration)
    if (d >= 0.0f) return c;

    c.hit         = true;
    c.normal      = -n;                     // body -> floor
    c.penetration = -d;
    c.point       = cseg - C.cap.r * n;     // surface point
    return c;
}

/* ----------------------------
   Impulses & Integration
   ---------------------------- */

static void applyImpulse(RigidBody& A, RigidBody& B, const Contact& c)
{
    glm::vec3 rA = c.point - A.x;
    glm::vec3 rB = c.point - B.x;

    glm::vec3 vA = A.v + glm::cross(A.w, rA);
    glm::vec3 vB = B.v + glm::cross(B.w, rB);
    glm::vec3 rv = vB - vA;

    float rvn = glm::dot(rv, c.normal);
    if (rvn > 0.0f) return;

    constexpr float bounceThreshold = 0.4f; // m/s
    float e = std::min(A.restitution, B.restitution);
    if (std::abs(rvn) < bounceThreshold) e = 0.0f;

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
    if (kN < 1e-8f) return;
    float j = -(1.0f + e) * rvn / kN;
    glm::vec3 impulseN = j * c.normal;

    A.v -= impulseN * A.invMass;  B.v += impulseN * B.invMass;
    A.w -= IA * glm::cross(rA, impulseN);
    B.w += IB * glm::cross(rB, impulseN);

    // Tangential friction (dynamic)
    vA = A.v + glm::cross(A.w, rA);
    vB = B.v + glm::cross(B.w, rB);
    rv = vB - vA;

    glm::vec3 t = rv - c.normal * glm::dot(rv, c.normal);
    float tlen = glm::length(t);

    // static-friction snap: ignore tiny slip
    const float slipEps = 1e-3f;
    if (tlen < slipEps) return;

    t /= tlen;

    float kT = K_scalar(t);
    if (kT < 1e-8f) return;
    float jt = -glm::dot(rv, t) / kT;

    float mu = 0.5f * (A.friction + B.friction);
    jt = glm::clamp(jt, -mu * j, mu * j);

    glm::vec3 impulseT = jt * t;
    A.v -= impulseT * A.invMass;  B.v += impulseT * B.invMass;
    A.w -= IA * glm::cross(rA, impulseT);
    B.w += IB * glm::cross(rB, impulseT);
}

struct SolverConfig{ float baumgarte=0.25f, allowedPen=0.003f; int velIters=30; };

static void positionalCorrection(RigidBody& A, RigidBody& B, const Contact& c, const SolverConfig& cfg)
{
    float k = std::max(0.0f, c.penetration - cfg.allowedPen);
    if (k <= 0) return;
    glm::vec3 corr = (cfg.baumgarte * k) * c.normal / (A.invMass + B.invMass + 1e-8f);
    A.x -= A.invMass * corr;
    B.x += B.invMass * corr;
}

static void integrate(RigidBody& b, const glm::vec3& g, float dt)
{
    // linear
    b.v += g * dt;
    float ld = std::exp(-LIN_DAMP * dt);
    b.v *= ld;
    b.x += b.v * dt;

    // angular
    float ad = std::exp(-ANG_DAMP * dt);
    b.w *= ad;

    float wlen = glm::length(b.w);
    if (wlen > W_MAX) b.w *= (W_MAX / wlen);

    glm::quat dq(0.0f, b.w.x, b.w.y, b.w.z);
    b.q = glm::normalize(b.q + 0.5f * dq * b.q * dt);
}

/* ==============
   OpenGL bits
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

uniform vec3 uColor;
uniform vec3 uLightDir;   // direction of light travel
uniform vec3 uEyePos;

// Checker controls (used for floor)
uniform int   uUseGrid;
uniform float uGridScale;
uniform vec3  uGridColor1;
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

/* Mesh helpers: cube (for floor) + unit capped cylinder (radius=1, y in [-1,1]) */

struct Mesh {
    GLuint vao=0, vbo=0, ebo=0;
    GLsizei indexCount=0;
};

static Mesh makeCubeMesh()
{
    Mesh m;
    struct V { float px,py,pz, nx,ny,nz; };
    const V CUBE[] = {
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
    const unsigned IDX[] = {
        0,1,2, 0,2,3,     4,5,6, 4,6,7,
        8,9,10, 8,10,11, 12,13,14, 12,14,15,
        16,17,18, 16,18,19, 20,21,22, 20,22,23
    };

    glGenVertexArrays(1,&m.vao);
    glGenBuffers(1,&m.vbo);
    glGenBuffers(1,&m.ebo);

    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(CUBE), CUBE, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(IDX), IDX, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(V),(void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,sizeof(V),(void*)(3*sizeof(float)));

    glBindVertexArray(0);
    m.indexCount = 36;
    return m;
}

static Mesh makeCappedCylinderMesh(int seg=32)
{
    Mesh m;
    struct V { float px,py,pz, nx,ny,nz; };
    std::vector<V> verts;
    std::vector<unsigned> idx;

    // side rings at y = -1, +1
    verts.reserve(seg*2 + 2 + seg*2);
    for (int i=0;i<seg;i++){
        float a = (2.0f * PI * i) / seg;
        float x = std::cos(a), z = std::sin(a);
        // bottom ring
        verts.push_back({x,-1,z,  x,0,z});
        // top ring
        verts.push_back({x, 1,z,  x,0,z});
    }
    // indices for sides
    for (int i=0;i<seg;i++){
        int i0 = 2*i;
        int i1 = (2*((i+1)%seg));
        // quad -> two triangles: (i0,i1,i1+1) (i0,i1+1,i0+1)
        idx.push_back(i0); idx.push_back(i1); idx.push_back(i1+1);
        idx.push_back(i0); idx.push_back(i1+1); idx.push_back(i0+1);
    }

    // top cap center + rim
    int baseTop = (int)verts.size();
    verts.push_back({0,1,0,  0,1,0}); // center
    for (int i=0;i<seg;i++){
        float a = (2.0f * PI * i) / seg;
        float x = std::cos(a), z = std::sin(a);
        verts.push_back({x,1,z,  0,1,0});
    }
    for (int i=0;i<seg;i++){
        int a0 = baseTop + 1 + i;
        int a1 = baseTop + 1 + ((i+1)%seg);
        idx.push_back(baseTop); idx.push_back(a0); idx.push_back(a1);
    }

    // bottom cap center + rim
    int baseBot = (int)verts.size();
    verts.push_back({0,-1,0,  0,-1,0}); // center
    for (int i=0;i<seg;i++){
        float a = (2.0f * PI * i) / seg;
        float x = std::cos(a), z = std::sin(a);
        verts.push_back({x,-1,z,  0,-1,0});
    }
    for (int i=0;i<seg;i++){
        int a0 = baseBot + 1 + ((i+1)%seg);
        int a1 = baseBot + 1 + i;
        idx.push_back(baseBot); idx.push_back(a0); idx.push_back(a1);
    }

    glGenVertexArrays(1,&m.vao);
    glGenBuffers(1,&m.vbo);
    glGenBuffers(1,&m.ebo);

    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size()*sizeof(V), verts.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size()*sizeof(unsigned), idx.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(V),(void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,sizeof(V),(void*)(3*sizeof(float)));

    glBindVertexArray(0);
    m.indexCount = (GLsizei)idx.size();
    return m;
}

/* App */

struct App {
    GLFWwindow* win{nullptr};
    GLuint prog=0;
    GLint uProj=-1, uView=-1, uModel=-1, uColor=-1;
    GLint uUseGrid=-1, uGridScale=-1, uGridColor1=-1, uGridColor2=-1;
    GLint uLightDir=-1, uEyePos=-1;

    Mesh cube{}, cyl{};

    // camera (orbit)
    float yaw = 0.6f, pitch = 0.35f, dist = 6.0f;
    double lastX=0, lastY=0; bool dragging=false;
    bool paused=false, vsync=true;

    // world
    glm::vec3 gravity{0.0f, -10.0f, 0.0f};
    float dt = 1.0f/600.0f;
    SolverConfig solver{0.25f, 0.003f, 30};

    // bodies (use makeRodLD to set Length & Diameter)
    RigidBody A = RigidBody::makeRodLD({-1.6f, 0.6f, 0.0f}, glm::angleAxis(+0.35f, glm::vec3(0,1,1)),
                                       /*density*/1000.0f, /*L*/ROD_LENGTH_L, /*D*/ROD_DIAMETER, 0.15f, 0.6f);
    RigidBody B = RigidBody::makeRodLD({ +1.2f, 1.0f, 0.2f}, glm::angleAxis(-0.25f, glm::vec3(1,0,0)),
                                       1000.0f, ROD_LENGTH_L*0.8f, ROD_DIAMETER, 0.15f, 0.6f);
    RigidBody G = RigidBody::makeStaticFloor({0.0f, -0.8f, 0.0f}, glm::quat(1,0,0,0), 10.0f, 0.1f, 10.0f, 0.3f, 0.9f);

    void reset(){
        A = RigidBody::makeRodLD({-1.6f, 0.6f, 0.0f}, glm::angleAxis(+0.35f, glm::vec3(0,1,1)),
                                 1000.0f, ROD_LENGTH_L, ROD_DIAMETER, 0.15f, 0.6f);
        B = RigidBody::makeRodLD({ +1.2f, 1.0f, 0.2f}, glm::angleAxis(-0.25f, glm::vec3(1,0,0)),
                                 1000.0f, ROD_LENGTH_L*0.8f, ROD_DIAMETER, 0.15f, 0.6f);
        A.v = {+2.2f, 0.0f, 0.0f};  B.v = {-1.0f, 0.0f, 0.0f};
        A.w = {0.0f, 0.0f, 0.0f};   B.w = {0.0f, 0.0f, 0.0f};
        G   = RigidBody::makeStaticFloor({0.0f, -0.8f, 0.0f}, glm::quat(1,0,0,0), 10.0f, 0.1f, 10.0f, 0.3f, 0.9f);
    }

    static void keyCB(GLFWwindow* w, int key, int, int action, int){
        if (action != GLFW_PRESS) return;
        App* s = (App*)glfwGetWindowUserPointer(w);
        if (key==GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(w,1);
        if (key==GLFW_KEY_SPACE)  s->paused = !s->paused;
        if (key==GLFW_KEY_R)      s->reset();
        if (key==GLFW_KEY_V){ s->vsync=!s->vsync; glfwSwapInterval(s->vsync?1:0); }
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
        win = glfwCreateWindow(1200, 800, "Rigid Bodies – Rods (Capsules)", nullptr, nullptr);
        if(!win){ glfwTerminate(); return false; }
        glfwMakeContextCurrent(win);
        glfwSwapInterval(1);
        if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) return false;

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_MULTISAMPLE);
        glDisable(GL_CULL_FACE);
        glDisable(GL_BLEND);

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

        cube = makeCubeMesh();
        cyl  = makeCappedCylinderMesh(40);
    }

    void physicsStep(){
        integrate(A, gravity, dt);
        integrate(B, gravity, dt);

        // Capsule <-> Capsule
        if (Contact cAB = collideCapsuleCapsule(A,B); cAB.hit){
            for(int it=0; it<solver.velIters; ++it) applyImpulse(A,B,cAB);
            positionalCorrection(A,B,cAB,solver);
        }

        // Capsule <-> Floor
        if (Contact cAG = collideCapsuleFloor(A, G); cAG.hit){
            for(int it=0; it<solver.velIters; ++it) applyImpulse(A,G,cAG);
            positionalCorrection(A,G,cAG,solver);
        }
        if (Contact cBG = collideCapsuleFloor(B, G); cBG.hit){
            for(int it=0; it<solver.velIters; ++it) applyImpulse(B,G,cBG);
            positionalCorrection(B,G,cBG,solver);
        }
    }

    glm::mat4 viewMatrix(){
        glm::vec3 target(0.0f);
        float cp = std::cos(pitch), sp = std::sin(pitch);
        float cy = std::cos(yaw),   sy = std::sin(yaw);
        glm::vec3 eye = target + glm::vec3(cp*cy, sp, cp*sy) * dist;
        return glm::lookAt(eye, target, glm::vec3(0,1,0));
    }

    glm::vec3 eyePos() const {
        float cp = std::cos(pitch), sp = std::sin(pitch);
        float cy = std::cos(yaw),   sy = std::sin(yaw);
        return glm::vec3(cp*cy, sp, cp*sy) * dist;
    }

    void drawMesh(const Mesh& m, const glm::mat4& M,
                  const glm::mat4& P, const glm::mat4& V,
                  const glm::vec3& color, bool grid=false)
    {
        glUseProgram(prog);
        glUniformMatrix4fv(uProj,  1, GL_FALSE, glm::value_ptr(P));
        glUniformMatrix4fv(uView,  1, GL_FALSE, glm::value_ptr(V));
        glUniformMatrix4fv(uModel, 1, GL_FALSE, glm::value_ptr(M));
        glUniform3fv(uColor, 1, &color[0]);
        glUniform1i(uUseGrid, grid ? 1 : 0);
        if (grid){
            const glm::vec3 c1(0.80f, 0.82f, 0.85f);
            const glm::vec3 c2(0.65f, 0.67f, 0.70f);
            glUniform3fv(uGridColor1, 1, &c1[0]);
            glUniform3fv(uGridColor2, 1, &c2[0]);
            glUniform1f(uGridScale, 1.0f);
        }

        glBindVertexArray(m.vao);
        glDrawElements(GL_TRIANGLES, m.indexCount, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        if (grid) glUniform1i(uUseGrid, 0);
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

            // draw bodies (capsules)
            drawMesh(cyl, A.modelMatrix(), P, V, glm::vec3(0.30f,0.70f,1.00f), false);
            drawMesh(cyl, B.modelMatrix(), P, V, glm::vec3(1.00f,0.55f,0.25f), false);

            // draw floor (checker)
            drawMesh(cube, G.modelMatrix(), P, V, glm::vec3(1,1,1), true);

            glfwSwapBuffers(win);
            glfwPollEvents();
        }
        glfwDestroyWindow(win);
        glfwTerminate();
        return 0;
    }
};

int main(){ App app; return app.run(); }
