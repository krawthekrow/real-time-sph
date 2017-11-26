R"RAWSTR(
#version 330 core

in vec2 posBillboardSpace;
in vec4 posCameraSpace;
in float fSize;
out vec4 color;

uniform mat4 P;
uniform vec2 viewportScreenRatio;
uniform sampler2D depthTex;

void main(){
    vec2 posBillboardSpaceXY = posBillboardSpace.xy;
    float rSq = dot(posBillboardSpaceXY, posBillboardSpaceXY);
    float fSizeSq = fSize * fSize;
    float zAdjustSq = fSizeSq - rSq;
    if (zAdjustSq < 0) discard;
    vec4 newPosCameraSpace =
        posCameraSpace + vec4(0, 0, sqrt(zAdjustSq), 0);
    vec4 newPosScreenSpace = P * newPosCameraSpace;
    float depth =
        (newPosScreenSpace.z / newPosScreenSpace.w + 1.0f) / 2.0f;
    vec4 posScreenSpace = P * posCameraSpace;
    float correctDepth = texture(depthTex,
        (posScreenSpace.xy / posScreenSpace.w + 1.0f) / 2.0f /
        viewportScreenRatio).r;
    if (depth != correctDepth) discard;
    // if ((depth - correctDepth) > 0.0000001f) discard;
    // if (depth > 0.999f) discard;
    // if (depth > 0.999f &&
    //     abs(depth - correctDepth) > correctDepth / 1048576.0f) discard;
    // if (depth > 0.99f &&
    //     abs(depth - correctDepth) > correctDepth / 262144.0f) discard;
    // if (depth > 0.999f &&
    //     abs(depth - correctDepth) > correctDepth / 1024.0f) discard;
    // if (depth > 0.99f &&
    //     abs(depth - correctDepth) > correctDepth / 64.0f) discard;

    // color = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    // color = vec4(depth, 0.0f, 0.0f, 1.0f);
    // color = vec4((depth - 0.9991f) * 4000.0f,
    //     0.0f, 0.0f, 1.0f);
    // color = vec4((depth - 0.9940f) * 100.0f,
    //     0.0f, 0.0f, 1.0f);
    color = vec4(vec3(zAdjustSq / (fSize * fSize)), 1.0f);
}
)RAWSTR"
