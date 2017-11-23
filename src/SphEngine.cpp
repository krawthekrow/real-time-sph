#include <cstdio>

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/random.hpp>

#include "ShaderManager.h"
#include "Shaders.h"

#include "SphEngine.h"

using namespace glm;

const int NUM_PARTS = 3000;
const float ROT_RATE = 1000.0f;

void SphEngine::Init() {
    minBound = vec3(-25.0f, -25.0f, -25.0f);
    maxBound = vec3(25.0f, 25.0f, 25.0f);

    drawLimitZ = minBound.z;

    // SPH INIT

    shaderProgram = ShaderManager::LoadShaders(
        Shaders::VERT_TRANSFERDENSITY,
        Shaders::FRAG_DRAWSPHERE,
        Shaders::GEOM_MAKEBILLBOARDS);
    glUseProgram(shaderProgram);
    mvLocation = glGetUniformLocation(shaderProgram, "MV");
    pLocation = glGetUniformLocation(shaderProgram, "P");
    drawLimitZLocation = glGetUniformLocation(shaderProgram, "drawLimitZ");

    GLuint posModelSpaceLocation =
        glGetAttribLocation(shaderProgram, "posModelSpace");
    GLuint densityLocation =
        glGetAttribLocation(shaderProgram, "density");

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(posModelSpaceLocation);
    glEnableVertexAttribArray(densityLocation);

    glGenBuffers(1, &vboPos);
    glBindBuffer(GL_ARRAY_BUFFER, vboPos);
    glVertexAttribPointer(
        posModelSpaceLocation, 3, GL_FLOAT, GL_FALSE, 0, 0);

    GLfloat initPos[NUM_PARTS * 3];
    for (int i = 0; i < NUM_PARTS; i++) {
        // vec3 pos = linearRand(minBound, maxBound);
        vec3 dims = maxBound - minBound;
        vec3 pos = linearRand(
            minBound,
            vec3(minBound.x + dims.x / 8.0f, maxBound.y, maxBound.z));
        initPos[i * 3] = pos.x;
        initPos[i * 3 + 1] = pos.y;
        initPos[i * 3 + 2] = pos.z;
    }
    glBufferData(
        GL_ARRAY_BUFFER, sizeof(initPos), initPos, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &vboDensities);
    glBindBuffer(GL_ARRAY_BUFFER, vboDensities);
    glVertexAttribPointer(
        densityLocation, 1, GL_FLOAT, GL_FALSE, 0, 0);
    glBufferData(
        GL_ARRAY_BUFFER, NUM_PARTS * sizeof(float), NULL, GL_DYNAMIC_DRAW);

    sphCuda.Init(NUM_PARTS, vboPos, vboDensities, minBound, maxBound);

    vec3 *velocities = sphCuda.GetVelocitiesPtr();
    for (int i = 0; i < NUM_PARTS; i++) {
        velocities[i] = linearRand(vec3(-0.01f), vec3(0.01f));
    }

    // BOUNDING BOX INIT

    bbProgram = ShaderManager::LoadShaders(
        Shaders::VERT_TRANSFORMPOINTS, Shaders::FRAG_DONOTHING);
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

void SphEngine::Update(
    const mat4 &mvMatrix, const mat4 &pMatrix, const double &currTime) {
    const float rotAmt = currTime / ROT_RATE;
    sphCuda.Update(currTime, rotAmt);

    mat4 rot = rotate(mat4(1.0f), rotAmt, vec3(0.0f, 0.0f, 1.0f));
    mat4 rotMvMatrix = mvMatrix * rot;
    const mat4 mvpMatrix = pMatrix * rotMvMatrix;

    glUseProgram(shaderProgram);
    glUniformMatrix4fv(mvLocation, 1, GL_FALSE, &rotMvMatrix[0][0]);
    glUniformMatrix4fv(pLocation, 1, GL_FALSE, &pMatrix[0][0]);
    glUniform1f(drawLimitZLocation, drawLimitZ);
    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, NUM_PARTS);

    glUseProgram(bbProgram);
    glUniformMatrix4fv(bbMvpLocation, 1, GL_FALSE, &mvpMatrix[0][0]);
    glBindVertexArray(bbVao);
    // 12 line segments defined by 2 points each
    glDrawArrays(GL_LINES, 0, 2 * 12);
}

void SphEngine::IncDrawLimitZ(const float inc) {
    drawLimitZ += inc;
    if (drawLimitZ < minBound.z)
        drawLimitZ = minBound.z;
    if (drawLimitZ > maxBound.z)
        drawLimitZ = maxBound.z;
}
