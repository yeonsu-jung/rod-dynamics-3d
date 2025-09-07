#version 330 core
out vec4 FragColor;

in VS_OUT { vec3 posW; vec3 norW; } fs_in;

uniform vec3 uColor;
uniform vec3 uLightDir;   // direction of light travel
uniform vec3 uEyePos;

// Checker controls (used for floor)
uniform int   uUseGrid;
uniform float uGridScale;
uniform vec3  uGridColor1;
uniform vec3  uGridColor2;

void main(){
    vec3 N = normalize(fs_in.norW);
    vec3 L = normalize(-uLightDir);
    vec3 V = normalize(uEyePos - fs_in.posW);
    vec3 H = normalize(L + V);

    float NdotL = max(dot(N,L), 0.0);
    float spec  = pow(max(dot(N,H), 0.0), 32.0);

    vec3 baseColor = uColor;
    if (uUseGrid == 1) {
        vec2 uv = fs_in.posW.xz * uGridScale;
        float cell = mod(floor(uv.x) + floor(uv.y), 2.0);
        baseColor = mix(uGridColor1, uGridColor2, cell);
    }

    vec3 ambient  = 0.20 * baseColor;
    vec3 diffuse  = 0.70 * NdotL * baseColor;
    vec3 specular = 0.30 * spec * vec3(1.0);

    FragColor = vec4(ambient + diffuse + specular, 1.0);
}
