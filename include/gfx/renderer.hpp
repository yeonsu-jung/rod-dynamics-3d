#pragma once
#include "gfx/mesh.hpp"
#include "gfx/shader.hpp"
#include <glm/glm.hpp>
#include <vector>

struct RenderUniforms {
  glm::mat4 P, V, M;
  glm::vec3 color{1};
  glm::vec3 lightDir{-0.4f, -1.0f, -0.3f};
  glm::vec3 eye{0, 0, 6};
  // grid
  bool useGrid = false;
  float gridScale = 1.0f;
  glm::vec3 gridC1{0.80f, 0.82f, 0.85f}, gridC2{0.65f, 0.67f, 0.70f};
};

struct Renderer {
  Shader shader;
  GLint uProj = -1, uView = -1, uModel = -1, uColor = -1, uLightDir = -1,
        uEyePos = -1, uUseGrid = -1, uGridScale = -1, uGridC1 = -1,
        uGridC2 = -1;

  // Instanced path
  Shader instanced;
  GLint iuProj = -1, iuView = -1, iuLightDir = -1,
        iuEyePos = -1;    // instanced uniforms
  GLuint instanceVBO = 0; // per-instance buffer (mat4 + color)
  bool instancedOK = false;

  bool init(const std::string &assetDir);
  void draw(const Mesh &m, const RenderUniforms &U) const;
  // Draw many instances of the same mesh with per-instance transforms and
  // colors. Falls back to non-instanced if instanced shader is unavailable.
  void drawInstances(const Mesh &m, const glm::mat4 *models,
                     const glm::vec3 *colors, size_t count,
                     const RenderUniforms &common) const;

  // Immediate mode-like line drawing
  void drawLines(const std::vector<glm::vec3> &points, const glm::vec3 &color,
                 const RenderUniforms &uniforms) const;
};
