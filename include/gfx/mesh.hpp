#pragma once
#include <glad/glad.h>

struct Mesh {
  GLuint vao=0, vbo=0, ebo=0;
  GLsizei indexCount=0;
  void destroy();
};

Mesh makeCubeMesh();                  // unit cube [-1,1]^3 with normals
Mesh makeCappedCylinderMesh(int seg); // radius=1, y in [-1,1]
