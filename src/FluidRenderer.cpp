#include <glm/glm.hpp>

#include "ShaderManager.h"
#include "Shaders.h"

#include "FluidRenderer.h"

using namespace glm;

void FluidRenderer::Init(
    const int &_numParts,
    const vec3 &minBound, const vec3 &maxBound,
    GLfloat * const &initPos, const float &_drawLimitZ) {
    numParts = _numParts;
    drawLimitZ = _drawLimitZ;

    glGenBuffers(1, &posVbo);
    glBindBuffer(GL_ARRAY_BUFFER, posVbo);
    glBufferData(
        GL_ARRAY_BUFFER, 3 * numParts * sizeof(GLfloat),
        initPos, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &densitiesVbo);
    glBindBuffer(GL_ARRAY_BUFFER, densitiesVbo);
    glBufferData(
        GL_ARRAY_BUFFER, numParts * sizeof(GLfloat),
        NULL, GL_DYNAMIC_DRAW);

    flatSphereProgram = ShaderManager::LoadShaders(
        Shaders::VERT_TRANSFERDENSITY,
        Shaders::FRAG_DRAWSPHERE,
        Shaders::GEOM_MAKEBILLBOARDS);
    glUseProgram(flatSphereProgram);
    mvLocation = glGetUniformLocation(flatSphereProgram, "MV");
    pLocation = glGetUniformLocation(flatSphereProgram, "P");
    drawLimitZLocation =
        glGetUniformLocation(flatSphereProgram, "drawLimitZ");

    const GLuint posModelSpaceLocation =
        glGetAttribLocation(flatSphereProgram, "posModelSpace");
    const GLuint densityLocation =
        glGetAttribLocation(flatSphereProgram, "density");

    glGenVertexArrays(1, &flatSphereVao);
    glBindVertexArray(flatSphereVao);

    glEnableVertexAttribArray(posModelSpaceLocation);
    glEnableVertexAttribArray(densityLocation);

    glBindBuffer(GL_ARRAY_BUFFER, posVbo);
    glVertexAttribPointer(
        posModelSpaceLocation, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, densitiesVbo);
    glVertexAttribPointer(
        densityLocation, 1, GL_FLOAT, GL_FALSE, 0, 0);
}

void FluidRenderer::Update(const mat4 &mvMatrix, const mat4 &pMatrix)
    const {
    glUseProgram(flatSphereProgram);
    glUniformMatrix4fv(mvLocation, 1, GL_FALSE, &mvMatrix[0][0]);
    glUniformMatrix4fv(pLocation, 1, GL_FALSE, &pMatrix[0][0]);
    glUniform1f(drawLimitZLocation, drawLimitZ);
    glBindVertexArray(flatSphereVao);
    glDrawArrays(GL_POINTS, 0, numParts);
}

void FluidRenderer::IncDrawLimitZ(const float &inc) {
    drawLimitZ += inc;
    if (drawLimitZ < minBound.z)
        drawLimitZ = minBound.z;
    if (drawLimitZ > maxBound.z)
        drawLimitZ = maxBound.z;
}

GLuint FluidRenderer::GetPositionsVbo() const {
    return posVbo;
}

GLuint FluidRenderer::GetDensitiesVbo() const {
    return densitiesVbo;
}
