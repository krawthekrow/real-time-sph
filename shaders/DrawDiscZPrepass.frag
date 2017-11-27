R"RAWSTR(
#version 420 core

in vec2 posBillboardSpace;
in vec4 posCameraSpace;
in float fSize;
out vec4 color;

uniform mat4 P;

void main(){
    vec2 posBillboardSpaceXY = posBillboardSpace.xy;
    float rSq = dot(posBillboardSpaceXY, posBillboardSpaceXY);
    float fSizeSq = fSize * fSize;
    float zAdjustSq = fSizeSq - rSq;
    if (zAdjustSq < 0) discard;
    // float depth = gl_FragCoord.z;
    // color = vec4(vec3(depth - 0.99992f) * 20000.0f, 1.0f);
}
)RAWSTR"
