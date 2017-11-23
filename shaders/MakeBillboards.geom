R"RAWSTR(
#version 330 core

uniform mat4 MV;
uniform mat4 P;
uniform float drawLimitZ;
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;
in float vDensity[];
out vec2 posBillboardSpace;
out vec4 posCameraSpace;
out float fSize;

vec4 centerCameraSpace;

void emitRelPos(vec2 _posBillboardSpace){
    posBillboardSpace = _posBillboardSpace;
    posCameraSpace = centerCameraSpace + vec4(posBillboardSpace, 0, 0);
    gl_Position = P * posCameraSpace;
    EmitVertex();
}

void main(){
    vec4 posModelSpace = gl_in[0].gl_Position;
    if(posModelSpace.z < drawLimitZ) return;
    centerCameraSpace = MV * posModelSpace;

    float fDensity = vDensity[0];
    fSize = mix(clamp(
        pow(fDensity - 0.98f, 1.0f / 3.0f) * 1.0f,
        0.0f, 0.7f), 0.0f, 4.0f);
    emitRelPos(vec2(-fSize, fSize));
    emitRelPos(vec2(-fSize, -fSize));
    emitRelPos(vec2(fSize, fSize));
    emitRelPos(vec2(fSize, -fSize));
    EndPrimitive();
}
)RAWSTR"
