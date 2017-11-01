R"RAWSTR(
#version 330 core

uniform mat4 MV;
uniform mat4 P;
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;
out vec2 posBillboardSpace;
out vec4 posCameraSpace;

vec4 centerCameraSpace;
float SIZE = 2;

void emitRelPos(vec2 _posBillboardSpace){
    posBillboardSpace = _posBillboardSpace;
    posCameraSpace = centerCameraSpace + vec4(posBillboardSpace, 0, 0);
    gl_Position = P * posCameraSpace;
    EmitVertex();
}

void main(){
    vec4 posModelSpace = gl_in[0].gl_Position;
    centerCameraSpace = MV * posModelSpace;

    emitRelPos(vec2(-SIZE, SIZE));
    emitRelPos(vec2(-SIZE, -SIZE));
    emitRelPos(vec2(SIZE, SIZE));
    emitRelPos(vec2(SIZE, -SIZE));
    EndPrimitive();
}
)RAWSTR"
