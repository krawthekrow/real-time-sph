R"RAWSTR(
#version 330 core

layout(location = 0) in vec2 pos;
out vec2 texCoord;

uniform vec2 quadPos;
uniform vec2 quadDims;

void main(){
    texCoord = pos;
    gl_Position =
        vec4((pos * quadDims + quadPos) * 2.0f - 1.0f, 0.0f, 1.0f);
}
)RAWSTR"

