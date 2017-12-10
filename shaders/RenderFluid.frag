R"RAWSTR(
#version 330 core

in vec2 texCoord;
out vec4 color;

uniform vec2 quadDims;
uniform vec2 texDims;
uniform mat4 invP;
uniform mat4 MV;

uniform sampler2D depthTex;

vec3 getPosCameraSpace(vec2 texCoord, float depth){
    vec4 res = invP *
        vec4(vec3(texCoord * quadDims, depth) * 2.0f - 1.0f, 1.0f);
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

    vec3 viewDir = -normalize(posCameraSpace);
    vec3 lightDir = normalize((MV * vec4(1.0f, 3.0f, 2.0f, 0.0f)).xyz);
    vec3 halfDir = normalize(viewDir + lightDir);
    vec3 reflectDir = reflect(-lightDir, normal);
    bool lit = dot(lightDir, normal) > 0;

    float R0 = 0.133f;
    // vec3 bgColor = vec3(0.529, 0.809, 0.922);
    float fresnel = mix(pow(1.0f - dot(viewDir, halfDir), 5.0f),
        1.0f, R0);
    float specular = pow(max(0.0f, dot(reflectDir, viewDir)), 3.0f);
    float diffuse = max(0.0f, dot(normal, lightDir));
	float lightCoeff = 6.0f;
    float lightAmt =
        lit ? min(1.0f, 0.0f * diffuse +
			lightCoeff * fresnel * specular) : 0.0f;
    // color = mix(vec3(0.3f, 0.5f, 0.8f) * bgColor, vec3(1.0f), lightAmt);
    color = vec4(1.0f, 1.0f, 1.0f, lightAmt + 0.2f);
    // color = normal;
    // color = vec3((texture(depthTex, texCoord + vec2(1.0f, 0.0f) / texDims).r - 0.99992) * 20000.0f);
    gl_FragDepth = depth;
}
)RAWSTR"
