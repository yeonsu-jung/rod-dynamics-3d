#include "gfx/shader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

static GLuint compile(GLenum type, const std::string& src, const char* name){
    GLuint s = glCreateShader(type);
    const char* csrc = src.c_str();
    glShaderSource(s, 1, &csrc, nullptr);
    glCompileShader(s);
    GLint ok=0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if(!ok){
        char log[4096]; glGetShaderInfoLog(s, 4096, nullptr, log);
        std::cerr << "Shader compile error (" << name << "):\n" << log << "\n";
    }
    return s;
}

static std::string slurp(const std::string& path){
    std::ifstream f(path, std::ios::in);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

Shader::~Shader(){
    if (prog) glDeleteProgram(prog);
}

bool Shader::loadFromFiles(const std::string& vsPath, const std::string& fsPath){
    std::string vs = slurp(vsPath);
    std::string fs = slurp(fsPath);
    if (vs.empty() || fs.empty()){
        std::cerr << "Failed to read shader files: " << vsPath << " / " << fsPath << "\n";
        return false;
    }
    GLuint v = compile(GL_VERTEX_SHADER, vs, vsPath.c_str());
    GLuint f = compile(GL_FRAGMENT_SHADER, fs, fsPath.c_str());
    prog = glCreateProgram();
    glAttachShader(prog, v);
    glAttachShader(prog, f);
    glLinkProgram(prog);
    glDeleteShader(v);
    glDeleteShader(f);
    GLint ok=0; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if(!ok){
        char log[4096]; glGetProgramInfoLog(prog, 4096, nullptr, log);
        std::cerr << "Program link error:\n" << log << "\n";
        glDeleteProgram(prog); prog = 0;
        return false;
    }
    return true;
}

GLint Shader::uniform(const char* name) const {
    return glGetUniformLocation(prog, name);
}
void Shader::use() const { glUseProgram(prog); }
