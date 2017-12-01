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
    const vec3 &_minBound, const vec3 &_maxBound,
    GLfloat * const &initPos, GLfloat * const &initDensities,
    const float &_drawLimitZ) {
    numParts = _numParts;
    drawLimitZ = _drawLimitZ;
    minBound = _minBound;
    maxBound = _maxBound;

    texturedQuadRenderer.Init();
    quadVbo = TexturedQuadRenderer::MakeQuadVbo();

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
        initDensities, GL_DYNAMIC_DRAW);

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

    // INIT FLAT SPHERE FBO

    glGenTextures(1, &flatSphereDepthTex);
    glBindTexture(GL_TEXTURE_2D, flatSphereDepthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
        TextureUtils::MAX_SCREEN_WIDTH,
        TextureUtils::MAX_SCREEN_HEIGHT,
        0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glGenTextures(1, &flatSphereColorTex);
    glBindTexture(GL_TEXTURE_2D, flatSphereColorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F,
        TextureUtils::MAX_SCREEN_WIDTH,
        TextureUtils::MAX_SCREEN_HEIGHT,
        0, GL_RED, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glGenFramebuffers(1, &flatSphereFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, flatSphereFbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        flatSphereColorTex, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
        flatSphereDepthTex, 0);
    GLenum drawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, drawBuffers);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) !=
        GL_FRAMEBUFFER_COMPLETE) {
        printf("Framebuffer error occurred.\n");
        exit(0);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // INIT FINAL DRAW

    finalDrawProgram = ShaderManager::LoadShaders(
        Shaders::VERT_POSITIONQUAD,
        Shaders::FRAG_DRAWTEXTUREWITHDEPTH);

    finalDrawQuadPosLocation =
        glGetUniformLocation(finalDrawProgram, "quadPos");
    finalDrawQuadDimsLocation =
        glGetUniformLocation(finalDrawProgram, "quadDims");
    finalDrawTexLocation = glGetUniformLocation(finalDrawProgram, "tex");
    finalDrawDepthTexLocation =
        glGetUniformLocation(finalDrawProgram, "depthTex");

    const GLuint finalDrawPosLocation =
        glGetAttribLocation(finalDrawProgram, "pos");

    glGenVertexArrays(1, &finalDrawVao);
    glBindVertexArray(finalDrawVao);

    glEnableVertexAttribArray(finalDrawPosLocation);

    glBindBuffer(GL_ARRAY_BUFFER, quadVbo);
    glVertexAttribPointer(
        finalDrawPosLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
}

void FluidRenderer::Update(const mat4 &mvMatrix, const mat4 &pMatrix)
    const {
    vec2 viewportScreenRatio = vec2(
        TextureUtils::MAX_SCREEN_WIDTH,
        TextureUtils::MAX_SCREEN_HEIGHT) /
        vec2(viewportDims);

    // FLAT SPHERE

    glBindFramebuffer(GL_FRAMEBUFFER, flatSphereFbo);
    glDepthFunc(GL_LEQUAL);
    glColorMask(GL_TRUE, GL_FALSE, GL_FALSE, GL_FALSE);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(flatSphereProgram);
    glUniformMatrix4fv(flatSphereMvLocation,
        1, GL_FALSE, &mvMatrix[0][0]);
    glUniformMatrix4fv(flatSpherePLocation,
        1, GL_FALSE, &pMatrix[0][0]);
    glUniform1f(flatSphereDrawLimitZLocation, drawLimitZ);
    glBindVertexArray(flatSphereVao);
    glDrawArrays(GL_POINTS, 0, numParts);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDepthFunc(GL_LESS);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

    const vec2 origin(0.0f);

    glUseProgram(finalDrawProgram);
    glBindVertexArray(finalDrawVao);
    glUniform2fv(finalDrawQuadPosLocation, 1, &origin[0]);
    glUniform2fv(finalDrawQuadDimsLocation, 1, &viewportScreenRatio[0]);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, flatSphereColorTex);
    glUniform1i(finalDrawTexLocation, 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, flatSphereDepthTex);
    glUniform1i(finalDrawDepthTexLocation, 1);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    // texturedQuadRenderer.Update(flatSphereDepthTex,
    //     vec2(0.0f), viewportScreenRatio,
    //     -0.99992, 20000.0f);

    // glDisable(GL_STENCIL_TEST);
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
