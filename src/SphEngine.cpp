#include <cstdio>

#include <GL/glew.h>
#include <glm/glm.hpp>

#include "ShaderManager.h"
#include "Shaders.h"

#include "SphEngine.h"

using namespace glm;

const int NUM_PARTS = 10;

void SphEngine::Init() {
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
}

void SphEngine::Update(const mat4 mvMatrix, const mat4 pMatrix) {
    sphCuda.Update();
    glUseProgram(shaderProgram);
    glUniformMatrix4fv(mvLocation, 1, GL_FALSE, &mvMatrix[0][0]);
    glUniformMatrix4fv(pLocation, 1, GL_FALSE, &pMatrix[0][0]);
    glDrawArrays(GL_POINTS, 0, 4);
}
