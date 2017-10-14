#include <GL/glew.h>
#include <glm/glm.hpp>

#include "ShaderManager.h"
#include "Shaders.h"

#include "SphEngine.h"

using namespace glm;

void SphEngine::Init() {
    glGenVertexArrays(1, &vao);

    shaderProgram = ShaderManager::LoadShaders(
        Shaders::VERT_DONOTHING,
        Shaders::FRAG_DRAWSPHERE,
        Shaders::GEOM_MAKEBILLBOARDS
    );
    glUseProgram(shaderProgram);
    mvLocation = glGetUniformLocation(shaderProgram, "MV");
    pLocation = glGetUniformLocation(shaderProgram, "P");
}

void SphEngine::Update(const mat4 mvMatrix, const mat4 pMatrix) {
    glUseProgram(shaderProgram);
    glUniformMatrix4fv(mvLocation, 1, GL_FALSE, &mvMatrix[0][0]);
    glUniformMatrix4fv(pLocation, 1, GL_FALSE, &pMatrix[0][0]);
    glBindVertexArray(vao);
    glDrawArrays(
        GL_POINTS, 0, 10);
}
