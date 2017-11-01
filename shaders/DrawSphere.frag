R"RAWSTR(
#version 330 core

uniform mat4 P;

in vec2 posBillboardSpace;
in vec4 posCameraSpace;
out vec4 color;

void main(){
    float SIZE = 2;

    float rSq = dot(posBillboardSpace.xy, posBillboardSpace.xy);
    float perpAdjustSq = SIZE * SIZE - rSq;
    if (perpAdjustSq < 0) discard;
    vec4 newPosCameraSpace =
        posCameraSpace + vec4(0, 0, sqrt(perpAdjustSq), 0);
    vec4 posScreenSpace = P * newPosCameraSpace;
    gl_FragDepth = posScreenSpace.z / posScreenSpace.w;

    // color = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    color = vec4(vec3(perpAdjustSq / (SIZE * SIZE)), 1.0f);
}
)RAWSTR"
