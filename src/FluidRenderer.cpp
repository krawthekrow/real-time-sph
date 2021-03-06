#include <cstdio>

#include <glm/glm.hpp>

#include "GlobalDebugSwitches.h"
#include "ShaderManager.h"
#include "Shaders.h"
#include "TextureUtils.h"

#include "FluidRenderer.h"

using namespace glm;

FluidRenderer::FluidRenderer() {}

GLuint FluidRenderer::createDepthTex(const vec2 &texDims) {
    GLuint depthTex;
    glGenTextures(1, &depthTex);
    glBindTexture(GL_TEXTURE_2D, depthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24,
        texDims.x, texDims.y,
        0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    return depthTex;
}

GLuint FluidRenderer::createDepthFbo(
    const vec2 &texDims, GLuint &depthTex) {

    depthTex = createDepthTex(texDims);

    GLuint depthFbo;
    glGenFramebuffers(1, &depthFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, depthFbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
        depthTex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) !=
        GL_FRAMEBUFFER_COMPLETE) {
        printf("Framebuffer error occurred.\n");
        exit(0);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return depthFbo;
}

void FluidRenderer::Init(
    const int &_numParts,
    const vec3 &_minBound, const vec3 &_maxBound,
    GLfloat *const &initPos, GLfloat *const &initDensities,
    const float &_drawLimitZ) {
    numParts = _numParts;
    drawLimitZ = _drawLimitZ;
    minBound = _minBound;
    maxBound = _maxBound;

    texturedQuadRenderer.Init();
    quadVbo = TexturedQuadRenderer::MakeQuadVbo();
    vec2 screenTexDims = vec2(
        TextureUtils::MAX_SCREEN_WIDTH,
        TextureUtils::MAX_SCREEN_HEIGHT);

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

    flatSphereDepthTex = createDepthTex(screenTexDims);

    glGenFramebuffers(1, &flatSphereFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, flatSphereFbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
        flatSphereDepthTex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) !=
        GL_FRAMEBUFFER_COMPLETE) {
        printf("Framebuffer error occurred.\n");
        exit(0);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // INIT SMOOTH

    smoothProgram = ShaderManager::LoadShaders(
        Shaders::VERT_POSITIONQUAD,
        Shaders::FRAG_BILATERALFILTER);
    glUseProgram(smoothProgram);

    smoothQuadPosLocation =
        glGetUniformLocation(smoothProgram, "quadPos");
    smoothQuadDimsLocation =
        glGetUniformLocation(smoothProgram, "quadDims");
    smoothDepthTexLocation =
        glGetUniformLocation(smoothProgram, "depthTex");

    const GLuint smoothTexDimsLocation =
        glGetUniformLocation(smoothProgram, "texDims");
    glUniform2fv(smoothTexDimsLocation, 1, &screenTexDims[0]);

    const GLuint smoothPosLocation =
        glGetAttribLocation(smoothProgram, "pos");

    glGenVertexArrays(1, &smoothVao);
    glBindVertexArray(smoothVao);

    glEnableVertexAttribArray(smoothPosLocation);

    glBindBuffer(GL_ARRAY_BUFFER, quadVbo);
    glVertexAttribPointer(
        smoothPosLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);

    // INIT SMOOTH VERT

    smoothVertProgram = ShaderManager::LoadShaders(
        Shaders::VERT_POSITIONQUAD,
        Shaders::FRAG_BILATERALFILTERVERT);
    glUseProgram(smoothVertProgram);

    smoothVertQuadPosLocation =
        glGetUniformLocation(smoothVertProgram, "quadPos");
    smoothVertQuadDimsLocation =
        glGetUniformLocation(smoothVertProgram, "quadDims");
    smoothVertDepthTexLocation =
        glGetUniformLocation(smoothVertProgram, "depthTex");

    const GLuint smoothVertTexDimsLocation =
        glGetUniformLocation(smoothVertProgram, "texDims");
    glUniform2fv(smoothVertTexDimsLocation, 1, &screenTexDims[0]);

    const GLuint smoothVertPosLocation =
        glGetAttribLocation(smoothVertProgram, "pos");

    glGenVertexArrays(1, &smoothVertVao);
    glBindVertexArray(smoothVertVao);

    glEnableVertexAttribArray(smoothVertPosLocation);

    glBindBuffer(GL_ARRAY_BUFFER, quadVbo);
    glVertexAttribPointer(
        smoothVertPosLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);

    // INIT SMOOTH HORZ

    smoothHorzProgram = ShaderManager::LoadShaders(
        Shaders::VERT_POSITIONQUAD,
        Shaders::FRAG_BILATERALFILTERHORZ);
    glUseProgram(smoothHorzProgram);

    smoothHorzQuadPosLocation =
        glGetUniformLocation(smoothHorzProgram, "quadPos");
    smoothHorzQuadDimsLocation =
        glGetUniformLocation(smoothHorzProgram, "quadDims");
    smoothHorzDepthTexLocation =
        glGetUniformLocation(smoothHorzProgram, "depthTex");

    const GLuint smoothHorzTexDimsLocation =
        glGetUniformLocation(smoothHorzProgram, "texDims");
    glUniform2fv(smoothHorzTexDimsLocation, 1, &screenTexDims[0]);

    const GLuint smoothHorzPosLocation =
        glGetAttribLocation(smoothHorzProgram, "pos");

    glGenVertexArrays(1, &smoothHorzVao);
    glBindVertexArray(smoothHorzVao);

    glEnableVertexAttribArray(smoothHorzPosLocation);

    glBindBuffer(GL_ARRAY_BUFFER, quadVbo);
    glVertexAttribPointer(
        smoothHorzPosLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);

    // INIT SMOOTH FBO

    smoothFbo1 = createDepthFbo(screenTexDims, smoothDepthTex1);
    smoothFbo2 = createDepthFbo(screenTexDims, smoothDepthTex2);
    smoothVertFbo = createDepthFbo(screenTexDims, smoothVertDepthTex);
    smoothHorzFbo = createDepthFbo(screenTexDims, smoothHorzDepthTex);

    // INIT RENDER

    renderProgram = ShaderManager::LoadShaders(
        Shaders::VERT_POSITIONQUAD,
        Shaders::FRAG_RENDERFLUID);
    glUseProgram(renderProgram);

    renderQuadPosLocation =
        glGetUniformLocation(renderProgram, "quadPos");
    renderQuadDimsLocation =
        glGetUniformLocation(renderProgram, "quadDims");
    renderInvPLocation =
        glGetUniformLocation(renderProgram, "invP");
    renderMvLocation =
        glGetUniformLocation(renderProgram, "MV");
    renderDepthTexLocation =
        glGetUniformLocation(renderProgram, "depthTex");

    const GLuint renderTexDimsLocation =
        glGetUniformLocation(renderProgram, "texDims");
    glUniform2fv(renderTexDimsLocation, 1, &screenTexDims[0]);

    const GLuint renderPosLocation =
        glGetAttribLocation(renderProgram, "pos");

    glGenVertexArrays(1, &renderVao);
    glBindVertexArray(renderVao);

    glEnableVertexAttribArray(renderPosLocation);

    glBindBuffer(GL_ARRAY_BUFFER, quadVbo);
    glVertexAttribPointer(
        renderPosLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
}

void FluidRenderer::Update(const mat4 &mvMatrix, const mat4 &pMatrix)
    const {

    if (GlobalDebugSwitches::renderSwitch)
        return;

    vec2 viewportScreenRatio = vec2(
        TextureUtils::MAX_SCREEN_WIDTH,
        TextureUtils::MAX_SCREEN_HEIGHT) /
        vec2(viewportDims);
    const vec2 origin(0.0f);
    const mat4 invPMatrix = inverse(pMatrix);

    // FLAT SPHERE

    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);

    glBindFramebuffer(GL_FRAMEBUFFER, flatSphereFbo);
    glDepthFunc(GL_LEQUAL);

    glClear(GL_DEPTH_BUFFER_BIT);

    glUseProgram(flatSphereProgram);
    glUniformMatrix4fv(flatSphereMvLocation,
        1, GL_FALSE, &mvMatrix[0][0]);
    glUniformMatrix4fv(flatSpherePLocation,
        1, GL_FALSE, &pMatrix[0][0]);
    glUniform1f(flatSphereDrawLimitZLocation, drawLimitZ);
    glBindVertexArray(flatSphereVao);
    glDrawArrays(GL_POINTS, 0, numParts);

    GLuint smoothOutputTex = flatSphereDepthTex;
    if (GlobalDebugSwitches::smoothMode == 0) {
        // SMOOTH

        glDepthFunc(GL_ALWAYS);

        glBindFramebuffer(GL_FRAMEBUFFER, smoothVertFbo);
        glClear(GL_DEPTH_BUFFER_BIT);

        glUseProgram(smoothVertProgram);
        glBindVertexArray(smoothVertVao);
        glUniform2fv(smoothVertQuadPosLocation, 1, &origin[0]);
        glUniform2fv(smoothVertQuadDimsLocation, 1,
            &viewportScreenRatio[0]);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, flatSphereDepthTex);
        glUniform1i(smoothVertDepthTexLocation, 0);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glBindFramebuffer(GL_FRAMEBUFFER, smoothHorzFbo);
        glClear(GL_DEPTH_BUFFER_BIT);

        glUseProgram(smoothHorzProgram);
        glBindVertexArray(smoothHorzVao);
        glUniform2fv(smoothHorzQuadPosLocation, 1, &origin[0]);
        glUniform2fv(smoothHorzQuadDimsLocation, 1,
            &viewportScreenRatio[0]);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, smoothVertDepthTex);
        glUniform1i(smoothHorzDepthTexLocation, 0);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        smoothOutputTex = smoothHorzDepthTex;
    } else if (GlobalDebugSwitches::smoothMode == 1) {
        glDepthFunc(GL_ALWAYS);

        glBindFramebuffer(GL_FRAMEBUFFER, smoothFbo1);
        glClear(GL_DEPTH_BUFFER_BIT);

        glUseProgram(smoothProgram);
        glBindVertexArray(smoothVao);
        glUniform2fv(smoothQuadPosLocation, 1, &origin[0]);
        glUniform2fv(smoothQuadDimsLocation, 1,
            &viewportScreenRatio[0]);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, flatSphereDepthTex);
        glUniform1i(smoothDepthTexLocation, 0);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        smoothOutputTex = smoothDepthTex1;
    }

    glDepthFunc(GL_LESS);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // RENDER

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(renderProgram);
    glBindVertexArray(renderVao);
    glUniform2fv(renderQuadPosLocation, 1, &origin[0]);
    glUniform2fv(renderQuadDimsLocation, 1, &viewportScreenRatio[0]);
    glUniformMatrix4fv(renderInvPLocation, 1, GL_FALSE, &invPMatrix[0][0]);
    glUniformMatrix4fv(renderMvLocation, 1, GL_FALSE,
        &mvMatrix[0][0]);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, smoothOutputTex);
    glUniform1i(renderDepthTexLocation, 0);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glDisable(GL_BLEND);

    if (GlobalDebugSwitches::depthSwitch) {
        texturedQuadRenderer.Update(smoothOutputTex,
            vec2(0.0f), viewportScreenRatio,
            -0.9992f, 2000.0f);
    }
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

void FluidRenderer::SetViewportDimensions(const ivec2 &_viewportDims) {
    viewportDims = _viewportDims;
}
