R"RAWSTR(
#version 330 core

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in float vDensity[];
out vec2 posBillboardSpace;
out vec4 posCameraSpace;
out float fSize;

uniform mat4 MV;
uniform mat4 P;
uniform float drawLimitZ;

vec4 centerCameraSpace;
float newZ;

void emitRelPos(vec2 _posBillboardSpace){
    posBillboardSpace = _posBillboardSpace;
    posCameraSpace = centerCameraSpace + vec4(posBillboardSpace, 0, 0);
    gl_Position = P * posCameraSpace;
    gl_Position.z = newZ * gl_Position.w;
    EmitVertex();
}

void main(){
    vec4 posModelSpace = gl_in[0].gl_Position;
    if(posModelSpace.z < drawLimitZ) return;
    centerCameraSpace = MV * posModelSpace;

    float fDensity = vDensity[0];
    fSize = clamp(
        (pow(fDensity, 0.2f) - 1.0f) * 30.0f + 0.8f,
        0.0f, 2.0f);
    vec4 adjustedPos =
        P * (centerCameraSpace + vec4(0.0f, 0.0f, fSize, 0.0f));
    newZ = adjustedPos.z / adjustedPos.w;

    emitRelPos(vec2(-fSize, fSize));
    emitRelPos(vec2(-fSize, -fSize));
    emitRelPos(vec2(fSize, fSize));
    emitRelPos(vec2(fSize, -fSize));
    EndPrimitive();
}
)RAWSTR"
