#version 330 core
out vec4 FragColor;

in VS_OUT { vec3 posW; vec3 norW; vec4 color; } fs_in;

uniform vec3 uLightDir;
uniform vec3 uEyePos;

void main(){
    vec3 N = normalize(fs_in.norW);
    vec3 L = normalize(-uLightDir);
    vec3 V = normalize(uEyePos - fs_in.posW);
    vec3 H = normalize(L + V);

    float NdotL = max(dot(N,L), 0.0);
    float spec  = pow(max(dot(N,H), 0.0), 32.0);

    vec3 baseColor = fs_in.color.rgb;
    float alpha = fs_in.color.a;

    vec3 ambient  = 0.20 * baseColor;
    vec3 diffuse  = 0.70 * NdotL * baseColor;
    vec3 specular = 0.30 * spec * vec3(1.0);

    FragColor = vec4(ambient + diffuse + specular, alpha);
}
