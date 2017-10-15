#include <GL/glew.h>
#include <glm/glm.hpp>

#include <cuda_gl_interop.h>

#include "ShaderManager.h"
#include "Shaders.h"

#include "SphEngine.h"

using namespace glm;

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
}

void SphEngine::Update(const mat4 mvMatrix, const mat4 pMatrix) {
    glUseProgram(shaderProgram);
    glUniformMatrix4fv(mvLocation, 1, GL_FALSE, &mvMatrix[0][0]);
    glUniformMatrix4fv(pLocation, 1, GL_FALSE, &pMatrix[0][0]);
    GLfloat points[] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        2.0f, 0.0f, 0.0f,
        3.0f, 0.0f, 0.0f
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(points), points, GL_DYNAMIC_DRAW);
    glDrawArrays(GL_POINTS, 0, 4);
}
