#version 330 core

uniform mat4 MVP;
uniform sampler2D noiseMap;
uniform ivec2 gridSize;
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;
out float altitude;

void main(){
    vec2 posXY = gl_in[0].gl_Position.xy;
    for(int i = 0; i < 2; i++){
        for(int i2 = 0; i2 < 2; i2++){
            vec2 gridPos = posXY + vec2(i, i2);
            //gl_Position = MVP * vec4(gridPos.x, 0.0f, gridPos.y, 1.0f);
            //float noiseAmp = sin((gridPos.x + gridPos.y) / 20); //terrainNoise(gridPos / 200);
            float noiseAmp = texture(noiseMap, gridPos / vec2(gridSize)).r;
            gl_Position = MVP * vec4(gridPos.x, noiseAmp * 100, gridPos.y, 1.0f);
            //color = vec3(noiseAmp, noiseAmp, noiseAmp);
            altitude = noiseAmp;
            EmitVertex();
        }
    }
    EndPrimitive();
}
