R"RAWSTR(
#version 330 core

in vec2 texCoord;
out vec3 color;

uniform sampler2D depthTex;

uniform vec2 texDims;

void main(){
    gl_FragDepth = texture(depthTex, texCoord).r;
}
)RAWSTR"

