#include "physics/collision.hpp"
#include "physics/types.hpp"
#include <algorithm>
#include <cmath>

void closestPtSegmentSegment(const glm::vec3& p1, const glm::vec3& q1,
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
    float sN, sD = D;
    float tN, tD = D;

    const float EPS = 1e-8f;

    if (D < EPS){ // almost parallel
        sN = 0.0f; sD = 1.0f;
        tN = e;    tD = c;
    } else {
        sN = (b*e - c*d);
        tN = (a*e - b*d);
        if (sN < 0){ sN = 0; tN = e; tD = c; }
        else if (sN > sD){ sN = sD; tN = e + b; tD = c; }
    }

    if (tN < 0){
        tN = 0;
        if (-d < 0) sN = 0;
        else if (-d > a) sN = sD;
        else { sN = -d; sD = a; }
    } else if (tN > tD){
        tN = tD;
        if ((-d + b) < 0) sN = 0;
        else if ((-d + b) > a) sN = sD;
        else { sN = (-d + b); sD = a; }
    }

    float sc = (std::abs(sN) < EPS) ? 0.0f : (sN / sD);
    float tc = (std::abs(tN) < EPS) ? 0.0f : (tN / tD);

    c1 = p1 + sc * u;
    c2 = p2 + tc * v;
}

static inline float clampf(float x, float a, float b){ return std::max(a, std::min(b, x)); }

Contact collideCapsuleCapsule(const RigidBody& A, const RigidBody& B)
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
    if (dist > 1e-6f) {
        c.normal = d / dist;
    } else {
        glm::vec3 tmp = (B.x - A.x);
        if (glm::dot(tmp,tmp) < 1e-8f) tmp = glm::cross(a, glm::vec3(1,0,0));
        if (glm::dot(tmp,tmp) < 1e-8f) tmp = glm::vec3(0,1,0);
        c.normal = glm::normalize(tmp);
    }
    c.point = 0.5f * (pA + pB);
    return c;
}

Contact collideCapsuleFloor(const RigidBody& C, const RigidBody& G)
{
    Contact c;
    // floor top normal (ensure it points upward)
    glm::vec3 n = glm::normalize(G.R()[1]);
    if (glm::dot(n, glm::vec3(0,1,0)) < 0) n = -n;
    const glm::vec3 p0 = G.x + n * G.box.hy;

    const glm::vec3 a = C.axisY();              // capsule axis (not normalized)
    const float h = C.cap.h;
    const float r = C.cap.r;

    const float denom = glm::dot(a, n);
    glm::vec3 cLine;

    if (std::abs(denom) < 1e-6f) {
        // nearly parallel to plane: use the lower of the two ends
        const glm::vec3 e0 = C.x - a * h;
        const glm::vec3 e1 = C.x + a * h;
        float d0 = glm::dot(e0 - p0, n);
        float d1 = glm::dot(e1 - p0, n);
        cLine = (d0 < d1) ? e0 : e1;
    } else {
        // closest point on the infinite axis to plane, clamped to segment
        float t = glm::dot(p0 - C.x, n) / denom;
        t = clampf(t, -h, +h);
        cLine = C.x + a * t;
    }

    float d = glm::dot(cLine - p0, n) - r; // signed: negative means penetration
    if (d >= 0.0f) return c;

    c.hit = true;
    c.normal = -n;                 // from capsule -> floor (important for impulses)
    c.penetration = -d;
    c.point = cLine - r * n;
    return c;
}
