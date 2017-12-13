R"RAWSTR(
#version 330 core

in vec2 texCoord;
out vec3 color;

uniform float colorOffset;
uniform float colorScale;
uniform sampler2D tex;

void main(){
    color = (texture(tex, texCoord).rgb + colorOffset) * colorScale;
}
)RAWSTR"
