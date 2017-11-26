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

    // INIT FRAMEBUFFERS

    glGenTextures(1, &zPrepassDepthTex);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, zPrepassDepthTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F,
        TextureUtils::MAX_SCREEN_WIDTH, TextureUtils::MAX_SCREEN_HEIGHT,
        0, GL_RED, GL_FLOAT, NULL);
    // glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, 1920, 1080,
    //     0, GL_RED_INTEGER, GL_UNSIGNED_INT, NULL);

    glGenFramebuffers(1, &zPrepassFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, zPrepassFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, zPrepassFbo);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        zPrepassDepthTex, 0);
    const GLenum zPrepassDrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, zPrepassDrawBuffers);

    glGenRenderbuffers(1, &zPrepassDepthBuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, zPrepassDepthBuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT,
        TextureUtils::MAX_SCREEN_WIDTH, TextureUtils::MAX_SCREEN_HEIGHT);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
        GL_RENDERBUFFER, zPrepassDepthBuffer);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) !=
        GL_FRAMEBUFFER_COMPLETE) {
        printf("Framebuffer error.\n");
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // INIT Z PREPASS FIRST STAGE (DISC APPROXIMATION)

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

    // INIT Z PREPASS SECOND STAGE

    zPrepassProgram = ShaderManager::LoadShaders(
        Shaders::VERT_TRANSFERDENSITY,
        Shaders::FRAG_DRAWSPHEREZPREPASS,
        Shaders::GEOM_MAKEBILLBOARDS);
    glUseProgram(zPrepassProgram);

    zPrepassMvLocation = glGetUniformLocation(zPrepassProgram, "MV");
    zPrepassPLocation = glGetUniformLocation(zPrepassProgram, "P");
    zPrepassDrawLimitZLocation =
        glGetUniformLocation(zPrepassProgram, "drawLimitZ");

    const GLuint zPrepassPosModelSpaceLocation =
        glGetAttribLocation(zPrepassProgram, "posModelSpace");
    const GLuint zPrepassDensityLocation =
        glGetAttribLocation(zPrepassProgram, "density");

    glGenVertexArrays(1, &zPrepassVao);
    glBindVertexArray(zPrepassVao);

    glEnableVertexAttribArray(zPrepassPosModelSpaceLocation);
    glEnableVertexAttribArray(zPrepassDensityLocation);

    glBindBuffer(GL_ARRAY_BUFFER, posVbo);
    glVertexAttribPointer(
        zPrepassPosModelSpaceLocation, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, densitiesVbo);
    glVertexAttribPointer(
        zPrepassDensityLocation, 1, GL_FLOAT, GL_FALSE, 0, 0);

    // INIT FLAT SPHERE

    flatSphereProgram = ShaderManager::LoadShaders(
        Shaders::VERT_TRANSFERDENSITY,
        Shaders::FRAG_DRAWSPHERE,
        Shaders::GEOM_MAKEBILLBOARDS);
    glUseProgram(flatSphereProgram);

    flatSphereDepthTexLocation =
        glGetAttribLocation(flatSphereProgram, "depthTex");
    flatSphereMvLocation = glGetUniformLocation(flatSphereProgram, "MV");
    flatSpherePLocation = glGetUniformLocation(flatSphereProgram, "P");
    flatSphereDrawLimitZLocation =
        glGetUniformLocation(flatSphereProgram, "drawLimitZ");
    flatSphereViewportScreenRatioLocation =
        glGetUniformLocation(flatSphereProgram, "viewportScreenRatio");

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
    // glActiveTexture(GL_TEXTURE0);
    // glBindTexture(GL_TEXTURE_2D, zPrepassDepthTex);
    // glBindImageTexture(
    //     0, zPrepassDepthTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
    vec2 viewportScreenRatio = vec2(
        TextureUtils::MAX_SCREEN_WIDTH,
        TextureUtils::MAX_SCREEN_HEIGHT) /
        vec2(viewportDims);

    // Z PREPASS

    glBindFramebuffer(GL_FRAMEBUFFER, zPrepassFbo);

    glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    // glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
    glClearColor(1.0f, 1.0f, 1.0f, 0.0f);

    // Z PREPASS FIRST STAGE (DISC APPROXIMATION)

    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);

    glUseProgram(zPrepassDiscProgram);
    glUniformMatrix4fv(zPrepassDiscMvLocation,
        1, GL_FALSE, &mvMatrix[0][0]);
    glUniformMatrix4fv(zPrepassDiscPLocation,
        1, GL_FALSE, &pMatrix[0][0]);
    glUniform1f(zPrepassDiscDrawLimitZLocation, drawLimitZ);
    glBindVertexArray(zPrepassDiscVao);
    glDrawArrays(GL_POINTS, 0, numParts);
    // glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    // Z PREPASS SECOND STAGE

    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_FALSE);
    glColorMask(GL_TRUE, GL_FALSE, GL_FALSE, GL_FALSE);
    glEnable(GL_BLEND);
    glBlendEquation(GL_MIN);
    // glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    glUseProgram(zPrepassProgram);
    glUniformMatrix4fv(zPrepassMvLocation,
        1, GL_FALSE, &mvMatrix[0][0]);
    glUniformMatrix4fv(zPrepassPLocation,
        1, GL_FALSE, &pMatrix[0][0]);
    glUniform1f(zPrepassDrawLimitZLocation, drawLimitZ);
    glBindVertexArray(zPrepassVao);
    glDrawArrays(GL_POINTS, 0, numParts);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glDisable(GL_BLEND);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

    // FLAT SPHERE

    glUseProgram(flatSphereProgram);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, zPrepassDepthTex);
    glUniform1f(flatSphereDepthTexLocation, 0);
    glUniformMatrix4fv(flatSphereMvLocation,
        1, GL_FALSE, &mvMatrix[0][0]);
    glUniformMatrix4fv(flatSpherePLocation,
        1, GL_FALSE, &pMatrix[0][0]);
    glUniform2fv(flatSphereViewportScreenRatioLocation,
        1, &viewportScreenRatio[0]);
    glUniform1f(flatSphereDrawLimitZLocation, drawLimitZ);
    glBindVertexArray(flatSphereVao);
    glDrawArrays(GL_POINTS, 0, numParts);

    if (debugSwitch) {
        texturedQuadRenderer.Update(
            zPrepassDepthTex, vec2(0.0f), viewportScreenRatio);
    }
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
