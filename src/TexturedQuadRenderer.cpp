#include <cstdio>
#include <GL/glew.h>
#include <glm/glm.hpp>

#include "ShaderManager.h"
#include "Shaders.h"

#include "TexturedQuadRenderer.h"

using namespace glm;

void TexturedQuadRenderer::Init() {
    vec2 minBounds(0.0f), maxBounds(1.0f);
    GLfloat coords[] = {
        minBounds.x, minBounds.y,
        minBounds.x, maxBounds.y,
        maxBounds.x, minBounds.y,
        maxBounds.x, maxBounds.y
    };
    glGenBuffers(1, &posVbo);
    glBindBuffer(GL_ARRAY_BUFFER, posVbo);
    glBufferData(
        GL_ARRAY_BUFFER, sizeof(coords), coords, GL_STATIC_DRAW);

    program = ShaderManager::LoadShaders(
        Shaders::VERT_POSITIONQUAD,
        Shaders::FRAG_DRAWTEXTURE);
    glUseProgram(program);
    quadPosLocation = glGetUniformLocation(program, "quadPos");
    quadDimsLocation = glGetUniformLocation(program, "quadDims");
    colorOffsetLocation = glGetUniformLocation(program, "colorOffset");
    colorScaleLocation = glGetUniformLocation(program, "colorScale");
    texLocation = glGetUniformLocation(program, "tex");

    const GLuint posLocation =
        glGetAttribLocation(program, "pos");

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glEnableVertexAttribArray(posLocation);

    glBindBuffer(GL_ARRAY_BUFFER, posVbo);
    glVertexAttribPointer(
        posLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
}

void TexturedQuadRenderer::Update(const GLuint &tex,
    const vec2 &quadPos, const vec2 &quadDims,
    const float &colorOffset, const float &colorScale) const {
    glDisable(GL_DEPTH_TEST);
    glUseProgram(program);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    // glBindImageTexture(
    //     0, tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
    glUniform1i(texLocation, 0);
    glUniform2fv(quadPosLocation, 1, &quadPos[0]);
    glUniform2fv(quadDimsLocation, 1, &quadDims[0]);
    glUniform1f(colorOffsetLocation, colorOffset);
    glUniform1f(colorScaleLocation, colorScale);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glEnable(GL_DEPTH_TEST);
}
