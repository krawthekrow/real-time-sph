R"RAWSTR(
#version 420 core

in vec2 texCoord;
out vec3 color;

// layout(binding = 0, r32ui) uniform uimage2D tex;
uniform float colorOffset;
uniform float colorScale;
uniform sampler2D tex;

void main(){
    // color = vec3(1.0f, 0.0f, 0.0f);
    // color = vec3(imageLoad(tex, ivec2((texCoord) * vec2(1920.0f, 1080.0f))).r / 2.0f, 0.0f, 0.0f);
    // color = vec3((texture(tex, texCoord).r + colorOffset) * colorScale,
    //     0.0f, 0.0f);
    color = (texture(tex, texCoord).rgb + colorOffset) * colorScale;
}
)RAWSTR"
