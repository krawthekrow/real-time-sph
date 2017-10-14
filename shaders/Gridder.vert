R"RAWSTR(
#version 330 core

uniform ivec3 gridSize;

void main(){
    ivec3 meshGridSize = gridSize + ivec3(1);
    int planeSize = meshGridSize.x * meshGridSize.y;
    gl_Position = vec4(
        float(gl_VertexID % meshGridSize.x),
        float((gl_VertexID % planeSize) / meshGridSize.x),
        float(gl_VertexID / planeSize),
        1.0f);
}
)RAWSTR"
