#pragma once
#include <glm/glm.hpp>
#include "gfx/shader.hpp"
#include "gfx/mesh.hpp"

struct RenderUniforms {
  glm::mat4 P, V, M;
  glm::vec3 color{1};
  glm::vec3 lightDir{-0.4f,-1.0f,-0.3f};
  glm::vec3 eye{0,0,6};
  // grid
  bool useGrid=false;
  float gridScale=1.0f;
  glm::vec3 gridC1{0.80f,0.82f,0.85f}, gridC2{0.65f,0.67f,0.70f};
};

struct Renderer {
  Shader shader;
  GLint uProj=-1,uView=-1,uModel=-1,uColor=-1,uLightDir=-1,uEyePos=-1,uUseGrid=-1,uGridScale=-1,uGridC1=-1,uGridC2=-1;

  bool init(const std::string& assetDir);
  void draw(const Mesh& m, const RenderUniforms& U) const;
};
