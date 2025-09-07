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
