R"RAWSTR(
#version 330 core

// layout(location = 0) in vec3 posModelSpace;

void main(){
    // gl_Position = vec4(posModelSpace, 1.0f);
    gl_Position = vec4(gl_VertexID, 0, 0, 1.0f);
}
)RAWSTR"
