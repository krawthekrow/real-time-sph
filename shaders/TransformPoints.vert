R"RAWSTR(
#version 330 core

uniform mat4 MVP;
layout(location = 0) in vec3 posModelSpace;

void main(){
    gl_Position = MVP * vec4(posModelSpace, 1.0f);
}
)RAWSTR"
