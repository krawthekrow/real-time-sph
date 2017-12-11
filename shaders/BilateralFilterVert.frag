R"RAWSTR(
#version 330 core

in vec2 texCoord;
out vec3 color;

uniform vec2 texDims;
uniform sampler2D depthTex;

#define MSIZE 7
#define SIGMA 7.0f
#define FALLOFF 0.00004f

float getAdjDepth(vec2 texCoord, float dx, float dy) {
    vec2 newTexCoord = texCoord + vec2(dx, dy) / texDims;
    return texture(depthTex, newTexCoord).r;
}

void main(){
    float depth = texture(depthTex, texCoord).r;
    if (depth == 1.0f) discard;
    float sum = 0, wsum = 0;
    for (int i = -MSIZE; i <= MSIZE; i++) {
        float samp = getAdjDepth(texCoord, 0, i);
        if (samp == 1.0f) samp = depth;
        float r = i / SIGMA;
        float w = exp(-r * r);

        float r2 = (samp - depth) / FALLOFF;
        float g = exp(-r2 * r2);

        sum += samp * w * g;
        wsum += w * g;
    }

    if (wsum > 0.0f) {
        sum /= wsum;
    }

    gl_FragDepth = sum; // texture(depthTex, texCoord).r;
}
)RAWSTR"

