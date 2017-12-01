R"RAWSTR(
#version 330 core

in vec2 texCoord;
out vec3 color;

uniform sampler2D tex;
uniform sampler2D depthTex;

void main(){
    color = texture(tex, texCoord).rgb;
    gl_FragDepth = texture(depthTex, texCoord).r;
}
)RAWSTR"

