R"RAWSTR(
#version 330 core

layout(location = 0) in vec3 posModelSpace;
layout(location = 1) in float density;
out float vDensity;

void main(){
    gl_Position = vec4(posModelSpace, 1.0f);
    vDensity = density;
}
)RAWSTR"
