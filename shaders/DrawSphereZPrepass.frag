R"RAWSTR(
#version 330 core

#extension GL_ARB_conservative_depth : enable

in vec2 posBillboardSpace;
in vec4 posCameraSpace;
in float fSize;
out vec4 color;
layout (depth_less) out float gl_FragDepth;

uniform mat4 P;

void main(){
    vec2 posBillboardSpaceXY = posBillboardSpace.xy;
    float rSq = dot(posBillboardSpaceXY, posBillboardSpaceXY);
    float fSizeSq = fSize * fSize;
    float zAdjustSq = fSizeSq - rSq;
    if (zAdjustSq < 0) discard;
    vec4 newPosCameraSpace =
        posCameraSpace + vec4(0, 0, sqrt(zAdjustSq), 0);
    vec4 posScreenSpace = P * newPosCameraSpace;
    float depth = posScreenSpace.z / posScreenSpace.w;
    gl_FragDepth = depth;

    // color = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    // color = vec4((depth + 1.0f) / 2.0f, 0.0f, 0.0f, 1.0f);
    // color = vec4((((depth + 1.0f) / 2.0f) - 0.9991f) * 2000.0f,
    //     0.0f, 0.0f, 1.0f);
    color = vec4(vec3(zAdjustSq / (fSize * fSize)), 1.0f);
}
)RAWSTR"
