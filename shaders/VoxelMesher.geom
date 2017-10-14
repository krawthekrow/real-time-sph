R"RAWSTR(
#version 330 core

#define EPS 0.000001

uniform mat4 MVP;
uniform sampler3D noiseMap;
uniform ivec3 gridSize;
uniform vec3 chunkSize;
uniform vec3 chunkPos;
layout(points) in;
layout(triangle_strip, max_vertices = 12) out;
out vec3 pos;

bool hasMass(vec3 v){
    vec3 texPos = v / vec3(gridSize);
    if(any(lessThan(texPos, vec3(-EPS))) ||
        any(greaterThan(texPos, vec3(1 - EPS)))){
        return false;
    }
    else{
        return texture(noiseMap, texPos).r > 0;
    }
}

void main(){
    vec3 gridPos = gl_in[0].gl_Position.xyz;
    if(hasMass(gridPos - vec3(1, 0, 0)) != hasMass(gridPos)){
        for(int i = 0; i < 2; i++){
            for(int i2 = 0; i2 < 2; i2++){
                vec3 vertPos = gridPos + vec3(-0.5f) + vec3(0, i, i2);
                gl_Position = MVP * vec4(vertPos, 1.0f);
                pos = vertPos / vec3(gridSize) * chunkSize + chunkPos;
                EmitVertex();
            }
        }
        EndPrimitive();
    }
    if(hasMass(gridPos - vec3(0, 1, 0)) != hasMass(gridPos)){
        for(int i = 0; i < 2; i++){
            for(int i2 = 0; i2 < 2; i2++){
                vec3 vertPos = gridPos + vec3(-0.5f) + vec3(i2, 0, i);
                gl_Position = MVP * vec4(vertPos, 1.0f);
                pos = vertPos / vec3(gridSize) * chunkSize + chunkPos;
                EmitVertex();
            }
        }
        EndPrimitive();
    }
    if(hasMass(gridPos - vec3(0, 0, 1)) != hasMass(gridPos)){
        for(int i = 0; i < 2; i++){
            for(int i2 = 0; i2 < 2; i2++){
                vec3 vertPos = gridPos + vec3(-0.5f) + vec3(i, i2, 0);
                gl_Position = MVP * vec4(vertPos, 1.0f);
                pos = vertPos / vec3(gridSize) * chunkSize + chunkPos;
                EmitVertex();
            }
        }
        EndPrimitive();
    }
}
)RAWSTR"
