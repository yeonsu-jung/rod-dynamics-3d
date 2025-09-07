#include "gfx/mesh.hpp"
#include <vector>
#include <cmath>

void Mesh::destroy(){
    if (ebo) glDeleteBuffers(1,&ebo);
    if (vbo) glDeleteBuffers(1,&vbo);
    if (vao) glDeleteVertexArrays(1,&vao);
    vao = vbo = ebo = 0;
    indexCount = 0;
}

struct V { float px,py,pz, nx,ny,nz; };

Mesh makeCubeMesh(){
    Mesh m;
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

Mesh makeCappedCylinderMesh(int seg){
    Mesh m;
    std::vector<V> verts;
    std::vector<unsigned> idx;
    verts.reserve(seg*2 + 2* (1+seg) * 2);

    // side rings y=-1, +1
    for(int i=0;i<seg;i++){
        float a = (2.0f * float(M_PI) * i) / seg;
        float x = std::cos(a), z = std::sin(a);
        verts.push_back({x,-1,z,  x,0,z}); // bottom rim
        verts.push_back({x, 1,z,  x,0,z}); // top rim
    }
    for(int i=0;i<seg;i++){
        int i0 = 2*i;
        int i1 = 2*((i+1)%seg);
        idx.push_back(i0); idx.push_back(i1);   idx.push_back(i1+1);
        idx.push_back(i0); idx.push_back(i1+1); idx.push_back(i0+1);
    }

    // top cap
    int baseTop = (int)verts.size();
    verts.push_back({0,1,0, 0,1,0});
    for(int i=0;i<seg;i++){
        float a = (2.0f * float(M_PI) * i) / seg;
        float x = std::cos(a), z = std::sin(a);
        verts.push_back({x,1,z, 0,1,0});
    }
    for(int i=0;i<seg;i++){
        int a0 = baseTop + 1 + i;
        int a1 = baseTop + 1 + ((i+1)%seg);
        idx.push_back(baseTop); idx.push_back(a0); idx.push_back(a1);
    }

    // bottom cap
    int baseBot = (int)verts.size();
    verts.push_back({0,-1,0, 0,-1,0});
    for(int i=0;i<seg;i++){
        float a = (2.0f * float(M_PI) * i) / seg;
        float x = std::cos(a), z = std::sin(a);
        verts.push_back({x,-1,z, 0,-1,0});
    }
    for(int i=0;i<seg;i++){
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
