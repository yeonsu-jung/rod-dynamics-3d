#pragma once
#include <glad/glad.h>
#include <string>

class Shader {
public:
  GLuint prog=0;
  ~Shader();
  bool loadFromFiles(const std::string& vsPath, const std::string& fsPath);
  GLint uniform(const char* name) const;
  void use() const;
};
