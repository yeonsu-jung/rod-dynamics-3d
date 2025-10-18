#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNor;
// Per-instance mat4 columns at 2,3,4,5
layout(location=2) in mat4 iModel;
layout(location=6) in vec3 iColor;

uniform mat4 uProj, uView;
uniform vec3 uLightDir;
uniform vec3 uEyePos;

out VS_OUT {
    vec3 posW;
    vec3 norW;
    vec3 color;
} vs_out;

void main(){
    mat3 normalMat = transpose(inverse(mat3(iModel)));
    vec3 pos = aPos;
    vec3 nor = aNor;
    vec3 posW = vec3(iModel * vec4(pos,1.0));
    vec3 norW = normalize(normalMat * nor);

    vs_out.posW = posW;
    vs_out.norW = norW;
    vs_out.color = iColor;
    gl_Position = uProj * uView * vec4(posW,1.0);
}
