R"RAWSTR(
#version 420 core

// layout(binding = 0, r32ui) uniform uimage2D depthImage;

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
}
)RAWSTR"
