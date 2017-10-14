R"RAWSTR(
#version 330 core

uniform mat4 MV;
uniform mat4 P;
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

void main(){
    vec4 posModelSpace = gl_in[0].gl_Position;
    vec4 posCameraSpace = MV * posModelSpace;
    gl_Position = posCameraSpace + vec4(-1, 1, 0, 0);
    EmitVertex();
    gl_Position = posCameraSpace + vec4(-1, -1, 0, 0);
    EmitVertex();
    gl_Position = posCameraSpace + vec4(1, 1, 0, 0);
    EmitVertex();
    gl_Position = posCameraSpace + vec4(1, -1, 0, 0);
    EmitVertex();
    EndPrimitive();
}
)RAWSTR"
