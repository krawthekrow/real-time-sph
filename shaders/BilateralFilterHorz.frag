R"RAWSTR(
#version 330 core

in vec2 texCoord;
out vec3 color;

uniform vec2 texDims;
uniform sampler2D depthTex;

#define MSIZE 7
#define SIGMA 7.0f
#define SIGMA_DEPTH 0.00001f

float getAdjDepth(vec2 texCoord, float dx, float dy) {
    vec2 newTexCoord = texCoord + vec2(dx, dy) / texDims;
    return texture(depthTex, newTexCoord).r;
}

void main(){
    float depth = texture(depthTex, texCoord).r;
    if (depth == 1.0f) discard;
    float sum = 0.0f, wsum = 0.0f;
    for (int i = -MSIZE; i <= MSIZE; i++) {
        float samp = getAdjDepth(texCoord, i, 0);
        float r = i / SIGMA;
        float rDepth = (samp - depth) / SIGMA_DEPTH;
        float w = exp(-r * r - rDepth * rDepth);

        sum += samp * w;
        wsum += w;
    }

    if (wsum > 0.0f) {
        sum /= wsum;
    }

    gl_FragDepth = sum;
}
)RAWSTR"

