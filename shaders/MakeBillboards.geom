R"RAWSTR(
#version 330 core

uniform mat4 MV;
uniform mat4 P;
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;
out vec2 posBillboardSpace;
out vec4 posCameraSpace;

void main(){
    vec4 posModelSpace = gl_in[0].gl_Position;
    posModelSpace.xyz *= 2.0f;
    posModelSpace.xyz += vec3(0.0f, 0.0f, 0.0f);
    vec4 centerCameraSpace = MV * posModelSpace;
    float SIZE = 2;

    posBillboardSpace = vec2(-SIZE, SIZE);
    posCameraSpace = centerCameraSpace + vec4(posBillboardSpace, 0, 0);
    gl_Position = P * posCameraSpace;
    EmitVertex();

    posBillboardSpace = vec2(-SIZE, -SIZE);
    posCameraSpace = centerCameraSpace + vec4(posBillboardSpace, 0, 0);
    gl_Position = P * posCameraSpace;
    EmitVertex();

    posBillboardSpace = vec2(SIZE, SIZE);
    posCameraSpace = centerCameraSpace + vec4(posBillboardSpace, 0, 0);
    gl_Position = P * posCameraSpace;
    EmitVertex();

    posBillboardSpace = vec2(SIZE, -SIZE);
    posCameraSpace = centerCameraSpace + vec4(posBillboardSpace, 0, 0);
    gl_Position = P * posCameraSpace;
    EmitVertex();
    
    EndPrimitive();
}
)RAWSTR"
