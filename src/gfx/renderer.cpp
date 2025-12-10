#include "gfx/renderer.hpp"
#include <glm/gtc/type_ptr.hpp>
#include <string>

bool Renderer::init(const std::string &assetDir) {
  const std::string vs = assetDir + "/shaders/basic.vert";
  const std::string fs = assetDir + "/shaders/basic.frag";
  if (!shader.loadFromFiles(vs, fs))
    return false;

  uProj = shader.uniform("uProj");
  uView = shader.uniform("uView");
  uModel = shader.uniform("uModel");
  uColor = shader.uniform("uColor");
  uLightDir = shader.uniform("uLightDir");
  uEyePos = shader.uniform("uEyePos");
  uUseGrid = shader.uniform("uUseGrid");
  uGridScale = shader.uniform("uGridScale");
  uGridC1 = shader.uniform("uGridColor1");
  uGridC2 = shader.uniform("uGridColor2");

  // Try to load instanced variant
  const std::string vsInst = assetDir + "/shaders/instanced.vert";
  const std::string fsInst =
      assetDir + "/shaders/instanced.frag"; // reuse basic.frag if missing
  if (instanced.loadFromFiles(vsInst, fsInst)) {
    iuProj = instanced.uniform("uProj");
    iuView = instanced.uniform("uView");
    iuLightDir = instanced.uniform("uLightDir");
    iuEyePos = instanced.uniform("uEyePos");
    instancedOK = true;
  } else if (instanced.loadFromFiles(vsInst, fs)) {
    iuProj = instanced.uniform("uProj");
    iuView = instanced.uniform("uView");
    iuLightDir = instanced.uniform("uLightDir");
    iuEyePos = instanced.uniform("uEyePos");
    instancedOK = true;
  }

  if (instancedOK && instanceVBO == 0) {
    glGenBuffers(1, &instanceVBO);
  }
  return true;
}

void Renderer::draw(const Mesh &m, const RenderUniforms &U) const {
  shader.use();
  glUniformMatrix4fv(uProj, 1, GL_FALSE, glm::value_ptr(U.P));
  glUniformMatrix4fv(uView, 1, GL_FALSE, glm::value_ptr(U.V));
  glUniformMatrix4fv(uModel, 1, GL_FALSE, glm::value_ptr(U.M));
  glUniform3fv(uColor, 1, &U.color[0]);
  glUniform3fv(uLightDir, 1, &U.lightDir[0]);
  glUniform3fv(uEyePos, 1, &U.eye[0]);

  glUniform1i(uUseGrid, U.useGrid ? 1 : 0);
  if (U.useGrid) {
    glUniform1f(uGridScale, U.gridScale);
    glUniform3fv(uGridC1, 1, &U.gridC1[0]);
    glUniform3fv(uGridC2, 1, &U.gridC2[0]);
  }

  glBindVertexArray(m.vao);
  glDrawElements(GL_TRIANGLES, m.indexCount, GL_UNSIGNED_INT, 0);
  glBindVertexArray(0);

  if (U.useGrid)
    glUniform1i(uUseGrid, 0);
}

// Draw many instances of the same mesh with per-instance transforms and
// colors. Falls back to non-instanced if instanced shader is unavailable.
void Renderer::drawInstances(const Mesh &m, const glm::mat4 *models,
                             const glm::vec4 *colors, size_t count,
                             const RenderUniforms &common) const {
  if (!instancedOK || count == 0) {
    // Fallback
    for (size_t i = 0; i < count; ++i) {
      RenderUniforms U = common;
      U.M = models[i];
      U.color = colors ? glm::vec3(colors[i]) : common.color;
      U.useGrid = false;
      draw(m, U);
    }
    return;
  }

  instanced.use();
  glUniformMatrix4fv(iuProj, 1, GL_FALSE, glm::value_ptr(common.P));
  glUniformMatrix4fv(iuView, 1, GL_FALSE, glm::value_ptr(common.V));
  glUniform3fv(iuLightDir, 1, &common.lightDir[0]);
  glUniform3fv(iuEyePos, 1, &common.eye[0]);

  // Pack instance data as: mat4 (16 floats) + color (vec4)
  struct InstanceData {
    glm::mat4 M;
    glm::vec4 color;
    // float pad; // No padding needed, 16 + 4 = 20 floats = 80 bytes. Stride
    // is 80.
  };
  glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
  glBufferData(GL_ARRAY_BUFFER, count * sizeof(InstanceData), nullptr,
               GL_STREAM_DRAW);
  // Map and fill
  InstanceData *ptr =
      (InstanceData *)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
  if (ptr) {
    for (size_t i = 0; i < count; ++i) {
      ptr[i].M = models[i];
      ptr[i].color = colors ? colors[i] : glm::vec4(common.color, 1.0f);
    }
    glUnmapBuffer(GL_ARRAY_BUFFER);
  }

  glBindVertexArray(m.vao);
  // Set up per-instance attributes at fixed locations: 2,3,4,5 for mat4, 6 for
  // color
  GLsizei stride = sizeof(InstanceData);
  std::size_t off = 0;
  for (int k = 0; k < 4; ++k) {
    glEnableVertexAttribArray(2 + k);
    glVertexAttribPointer(2 + k, 4, GL_FLOAT, GL_FALSE, stride, (void *)(off));
    glVertexAttribDivisor(2 + k, 1);
    off += sizeof(glm::vec4);
  }
  glEnableVertexAttribArray(6);
  glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, stride,
                        (void *)(offsetof(InstanceData, color)));
  glVertexAttribDivisor(6, 1);

  glDrawElementsInstanced(GL_TRIANGLES, m.indexCount, GL_UNSIGNED_INT, 0,
                          (GLsizei)count);

  // Cleanup state not strictly necessary if VAO persists, but keep consistent
  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Renderer::drawLines(const std::vector<glm::vec3> &points,
                         const glm::vec3 &color,
                         const RenderUniforms &U) const {
  if (points.empty())
    return;

  shader.use();
  // Set view/proj from uniforms
  glUniformMatrix4fv(uProj, 1, GL_FALSE, glm::value_ptr(U.P));
  glUniformMatrix4fv(uView, 1, GL_FALSE, glm::value_ptr(U.V));

  // Identity model matrix
  glUniformMatrix4fv(uModel, 1, GL_FALSE, glm::value_ptr(glm::mat4(1.0f)));

  // Set color
  glUniform3fv(uColor, 1, &color[0]);

  // Disable grid for lines usually
  glUniform1i(uUseGrid, 0);

  // Create temp buffers (inefficient but simple for debug)
  GLuint vao, vbo;
  glGenVertexArrays(1, &vao);
  glGenBuffers(1, &vbo);

  // Interleave: pos(3), nor(3)
  std::vector<float> data;
  data.reserve(points.size() * 6);
  for (const auto &p : points) {
    data.push_back(p.x);
    data.push_back(p.y);
    data.push_back(p.z);
    // Dummy normal z-up? or maybe better pointing at camera?
    // For lines it doesn't really matter as long as it's not zero.
    // Let's use (0,1,0) like the reference code
    data.push_back(0.0f);
    data.push_back(1.0f);
    data.push_back(0.0f);
  }

  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(),
               GL_STREAM_DRAW);

  // Position attrib is 0
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);

  // Normal attrib is 1
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                        (void *)(3 * sizeof(float)));

  glDisable(GL_DEPTH_TEST); // Ensure lines are visible on top of rods
  glDrawArrays(GL_LINES, 0, (GLsizei)points.size());
  glEnable(GL_DEPTH_TEST);

  glBindVertexArray(0);
  glDeleteBuffers(1, &vbo);
  glDeleteVertexArrays(1, &vao);
}
