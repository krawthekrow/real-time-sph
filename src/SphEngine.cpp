#include <cstdio>

#include <GL/glew.h>
#include <glm/glm.hpp>

#include "ShaderManager.h"
#include "Shaders.h"

#include "SphEngine.h"

using namespace glm;

const int NUM_PARTS = 10;

void SphEngine::Init() {

    //SPH INIT

    shaderProgram = ShaderManager::LoadShaders(
        Shaders::VERT_DONOTHING,
        Shaders::FRAG_DRAWSPHERE,
        Shaders::GEOM_MAKEBILLBOARDS
    );
    glUseProgram(shaderProgram);
    mvLocation = glGetUniformLocation(shaderProgram, "MV");
    pLocation = glGetUniformLocation(shaderProgram, "P");

    GLuint posModelSpaceLocation =
        glGetAttribLocation(shaderProgram, "posModelSpace");

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(posModelSpaceLocation);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(
        posModelSpaceLocation, 3, GL_FLOAT, GL_FALSE, 0, 0);

    GLfloat initPos[NUM_PARTS * 3];
    for (int i = 0; i < NUM_PARTS; i++) {
        initPos[i * 3] = (GLfloat)i;
        initPos[i * 3 + 1] = 0.0f;
        initPos[i * 3 + 2] = 0.0f;
    }
    glBufferData(
        GL_ARRAY_BUFFER, sizeof(initPos), initPos, GL_DYNAMIC_DRAW);
    sphCuda.Init(NUM_PARTS, vbo);

    // BOUNDING BOX INIT

    bbProgram = ShaderManager::LoadShaders(
        Shaders::VERT_TRANSFORMPOINTS,
        Shaders::FRAG_DONOTHING
    );
    glUseProgram(bbProgram);

    bbMvpLocation = glGetUniformLocation(bbProgram, "MVP");
    GLuint bbPosModelSpaceLocation =
        glGetAttribLocation(bbProgram, "posModelSpace");

    glGenVertexArrays(1, &bbVao);
    glBindVertexArray(bbVao);
    glEnableVertexAttribArray(bbPosModelSpaceLocation);

    glGenBuffers(1, &bbVbo);
    glBindBuffer(GL_ARRAY_BUFFER, bbVbo);
    glVertexAttribPointer(
        bbPosModelSpaceLocation, 3, GL_FLOAT, GL_FALSE, 0, 0);

    minBound = ivec3(0);
    maxBound = ivec3(100.0f);

    // 12 line segments each defined by 2 vec3s
    float bbLineVerts[2 * 3 * 12];

    // Generates all 12 line segments defining a box
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                int offset = (i * 4 + j * 2 + k) * 2 * 3;
                int coord1 = i, coord2 = (i + 1) % 3, coord3 = (i + 2) % 3;
                float coord2Bound =
                    (j == 0) ? minBound[coord2] : maxBound[coord2];
                float coord3Bound =
                    (k == 0) ? minBound[coord3] : maxBound[coord3];
                bbLineVerts[offset + coord1] = minBound[coord1];
                bbLineVerts[offset + coord2] = coord2Bound;
                bbLineVerts[offset + coord3] = coord3Bound;
                bbLineVerts[offset + 3 + coord1] = maxBound[coord1];
                bbLineVerts[offset + 3 + coord2] = coord2Bound;
                bbLineVerts[offset + 3 + coord3] = coord3Bound;
            }
        }
    }
    glBufferData(
        GL_ARRAY_BUFFER, sizeof(bbLineVerts), bbLineVerts, GL_STATIC_DRAW);
}

void SphEngine::Update(const mat4 mvMatrix, const mat4 pMatrix) {
    sphCuda.Update();

    glUseProgram(shaderProgram);
    glUniformMatrix4fv(mvLocation, 1, GL_FALSE, &mvMatrix[0][0]);
    glUniformMatrix4fv(pLocation, 1, GL_FALSE, &pMatrix[0][0]);
    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, 4);

    glUseProgram(bbProgram);
    const mat4 mvpMatrix = pMatrix * mvMatrix;
    glUniformMatrix4fv(bbMvpLocation, 1, GL_FALSE, &mvpMatrix[0][0]);
    glBindVertexArray(bbVao);
    // 12 line segments defined by 2 points each
    glDrawArrays(GL_LINES, 0, 2 * 12);
}
