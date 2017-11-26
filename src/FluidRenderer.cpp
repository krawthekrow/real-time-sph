#include <cstdio>

#include <glm/glm.hpp>

#include "ShaderManager.h"
#include "Shaders.h"
#include "TextureUtils.h"

#include "FluidRenderer.h"

using namespace glm;

FluidRenderer::FluidRenderer()
    : debugSwitch(false) {}

void FluidRenderer::Init(
    const int &_numParts,
    const vec3 &minBound, const vec3 &maxBound,
    GLfloat * const &initPos, const float &_drawLimitZ) {
    numParts = _numParts;
    drawLimitZ = _drawLimitZ;

    texturedQuadRenderer.Init();

    // INIT POSITIONS AND DENSITIES

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

    // INIT Z PREPASS

    zPrepassDiscProgram = ShaderManager::LoadShaders(
        Shaders::VERT_TRANSFERDENSITY,
        Shaders::FRAG_DRAWDISCZPREPASS,
        Shaders::GEOM_MAKEBILLBOARDSZPREPASS);
    glUseProgram(zPrepassDiscProgram);

    zPrepassDiscMvLocation =
        glGetUniformLocation(zPrepassDiscProgram, "MV");
    zPrepassDiscPLocation =
        glGetUniformLocation(zPrepassDiscProgram, "P");
    zPrepassDiscDrawLimitZLocation =
        glGetUniformLocation(zPrepassDiscProgram, "drawLimitZ");

    const GLuint zPrepassDiscPosModelSpaceLocation =
        glGetAttribLocation(zPrepassDiscProgram, "posModelSpace");
    const GLuint zPrepassDiscDensityLocation =
        glGetAttribLocation(zPrepassDiscProgram, "density");

    glGenVertexArrays(1, &zPrepassDiscVao);
    glBindVertexArray(zPrepassDiscVao);

    glEnableVertexAttribArray(zPrepassDiscPosModelSpaceLocation);
    glEnableVertexAttribArray(zPrepassDiscDensityLocation);

    glBindBuffer(GL_ARRAY_BUFFER, posVbo);
    glVertexAttribPointer(
        zPrepassDiscPosModelSpaceLocation, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, densitiesVbo);
    glVertexAttribPointer(
        zPrepassDiscDensityLocation, 1, GL_FLOAT, GL_FALSE, 0, 0);

    // INIT FLAT SPHERE

    flatSphereProgram = ShaderManager::LoadShaders(
        Shaders::VERT_TRANSFERDENSITY,
        Shaders::FRAG_DRAWSPHERE,
        Shaders::GEOM_MAKEBILLBOARDS);
    glUseProgram(flatSphereProgram);

    flatSphereMvLocation = glGetUniformLocation(flatSphereProgram, "MV");
    flatSpherePLocation = glGetUniformLocation(flatSphereProgram, "P");
    flatSphereDrawLimitZLocation =
        glGetUniformLocation(flatSphereProgram, "drawLimitZ");

    const GLuint flatSpherePosModelSpaceLocation =
        glGetAttribLocation(flatSphereProgram, "posModelSpace");
    const GLuint flatSphereDensityLocation =
        glGetAttribLocation(flatSphereProgram, "density");

    glGenVertexArrays(1, &flatSphereVao);
    glBindVertexArray(flatSphereVao);

    glEnableVertexAttribArray(flatSpherePosModelSpaceLocation);
    glEnableVertexAttribArray(flatSphereDensityLocation);

    glBindBuffer(GL_ARRAY_BUFFER, posVbo);
    glVertexAttribPointer(
        flatSpherePosModelSpaceLocation, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, densitiesVbo);
    glVertexAttribPointer(
        flatSphereDensityLocation, 1, GL_FLOAT, GL_FALSE, 0, 0);
}

void FluidRenderer::Update(const mat4 &mvMatrix, const mat4 &pMatrix)
    const {
    vec2 viewportScreenRatio = vec2(
        TextureUtils::MAX_SCREEN_WIDTH,
        TextureUtils::MAX_SCREEN_HEIGHT) /
        vec2(viewportDims);

    // Z PREPASS

    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);

    glUseProgram(zPrepassDiscProgram);
    glUniformMatrix4fv(zPrepassDiscMvLocation,
        1, GL_FALSE, &mvMatrix[0][0]);
    glUniformMatrix4fv(zPrepassDiscPLocation,
        1, GL_FALSE, &pMatrix[0][0]);
    glUniform1f(zPrepassDiscDrawLimitZLocation, drawLimitZ);
    glBindVertexArray(zPrepassDiscVao);
    glDrawArrays(GL_POINTS, 0, numParts);

    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

    // FLAT SPHERE
    glDepthFunc(GL_LEQUAL);

    glUseProgram(flatSphereProgram);
    glUniformMatrix4fv(flatSphereMvLocation,
        1, GL_FALSE, &mvMatrix[0][0]);
    glUniformMatrix4fv(flatSpherePLocation,
        1, GL_FALSE, &pMatrix[0][0]);
    glUniform1f(flatSphereDrawLimitZLocation, drawLimitZ);
    glBindVertexArray(flatSphereVao);
    glDrawArrays(GL_POINTS, 0, numParts);

    glDepthFunc(GL_LESS);
}

void FluidRenderer::IncDrawLimitZ(const float &inc) {
    drawLimitZ += inc;
    if (drawLimitZ < minBound.z)
        drawLimitZ = minBound.z;
    if (drawLimitZ > maxBound.z)
        drawLimitZ = maxBound.z;
}

void FluidRenderer::ToggleDebugSwitch() {
    debugSwitch = !debugSwitch;
}

GLuint FluidRenderer::GetPositionsVbo() const {
    return posVbo;
}

GLuint FluidRenderer::GetDensitiesVbo() const {
    return densitiesVbo;
}

void FluidRenderer::SetViewportDimensions(const ivec2 &_viewportDims) {
    viewportDims = _viewportDims;
}
