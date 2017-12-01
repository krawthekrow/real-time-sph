R"RAWSTR(
#version 330 core

in vec2 texCoord;
out vec3 color;

uniform vec2 texDims;
uniform mat4 invP;

uniform sampler2D depthTex;

vec3 getPosCameraSpace(vec2 texCoord, float depth){
    vec4 res = invP * vec4(vec3(texCoord, depth) * 2.0f - 1.0f, 1.0f);
    return res.xyz / res.w;
}

vec3 getAdjPosCameraSpace(vec2 texCoord, float dx, float dy) {
    vec2 newTexCoord = texCoord + vec2(dx, dy) / texDims;
    return getPosCameraSpace(newTexCoord,
        texture(depthTex, newTexCoord).r);
}

void main(){
    float depth = texture(depthTex, texCoord).r;
    if (depth == 1.0f) discard;

    vec3 posCameraSpace = getPosCameraSpace(texCoord, depth);

    vec3 dx1 = getAdjPosCameraSpace(texCoord, -1.0f, 0.0f) -
        posCameraSpace;
    vec3 dx2 = -getAdjPosCameraSpace(texCoord, 1.0f, 0.0f) +
        posCameraSpace;
    vec3 dy1 = getAdjPosCameraSpace(texCoord, 0.0f, -1.0f) -
        posCameraSpace;
    vec3 dy2 = -getAdjPosCameraSpace(texCoord, 0.0f, 1.0f) +
        posCameraSpace;
    vec3 normal = normalize(cross(
        (dot(dx1, dx1) < dot(dx2, dx2)) ? dx1 : dx2,
        (dot(dy1, dy1) < dot(dy2, dy2)) ? dy1 : dy2));

    color = normal;
    // color = abs(normalize(dx1));
    // color = vec3((texture(depthTex, texCoord + vec2(1.0f, 0.0f) / texDims).r - 0.99992) * 20000.0f);
    // color = abs(normalize(dy1));
    gl_FragDepth = depth;
}
)RAWSTR"
