#include "gfx/renderer.hpp"
#include <glm/gtc/type_ptr.hpp>
#include <string>

bool Renderer::init(const std::string& assetDir){
    const std::string vs = assetDir + "/shaders/basic.vert";
    const std::string fs = assetDir + "/shaders/basic.frag";
    if (!shader.loadFromFiles(vs, fs)) return false;

    uProj      = shader.uniform("uProj");
    uView      = shader.uniform("uView");
    uModel     = shader.uniform("uModel");
    uColor     = shader.uniform("uColor");
    uLightDir  = shader.uniform("uLightDir");
    uEyePos    = shader.uniform("uEyePos");
    uUseGrid   = shader.uniform("uUseGrid");
    uGridScale = shader.uniform("uGridScale");
    uGridC1    = shader.uniform("uGridColor1");
    uGridC2    = shader.uniform("uGridColor2");
    return true;
}

void Renderer::draw(const Mesh& m, const RenderUniforms& U) const {
    shader.use();
    glUniformMatrix4fv(uProj,  1, GL_FALSE, glm::value_ptr(U.P));
    glUniformMatrix4fv(uView,  1, GL_FALSE, glm::value_ptr(U.V));
    glUniformMatrix4fv(uModel, 1, GL_FALSE, glm::value_ptr(U.M));
    glUniform3fv(uColor, 1, &U.color[0]);
    glUniform3fv(uLightDir, 1, &U.lightDir[0]);
    glUniform3fv(uEyePos,   1, &U.eye[0]);

    glUniform1i(uUseGrid, U.useGrid ? 1 : 0);
    if (U.useGrid){
        glUniform1f(uGridScale, U.gridScale);
        glUniform3fv(uGridC1, 1, &U.gridC1[0]);
        glUniform3fv(uGridC2, 1, &U.gridC2[0]);
    }

    glBindVertexArray(m.vao);
    glDrawElements(GL_TRIANGLES, m.indexCount, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    if (U.useGrid) glUniform1i(uUseGrid, 0);
}
