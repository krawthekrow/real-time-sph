R"RAWSTR(
#version 330 core

layout(location = 0) in vec3 posModelSpace;

void main(){
    gl_Position = vec4(posModelSpace, 1.0f);
}
)RAWSTR"
