R"RAWSTR(
#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec2 vertexTexCoord;

out vec2 texCoord;

uniform mat4 MVP;

void main(){
    gl_Position = MVP * vec4(vertexPosition_modelspace, 1);
    texCoord = vertexTexCoord;
}
)RAWSTR"
