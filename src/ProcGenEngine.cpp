#include <GL/glew.h>
#include <glm/glm.hpp>

#include "ShaderManager.h"
#include "Shaders.h"
#include "TextureUtils.h"

#include "ProcGenEngine.h"

using namespace glm;

void ProcGenEngine::Init() {
    globalSeed = 3421;
    const vec3 chunkSize = vec3(2.0f, 1.0f, 2.0f);
    const vec3 chunkPos = vec3(0.0f, 0.0f, 0.0f);
    gridSize = ivec3(64, 32, 64);

    glGenVertexArrays(1, &meshVertexArrayID);

    const GLuint noiseMapTexID =
        TextureUtils::GenComputeTexture3D(gridSize);

    noiseGenShaderID =
        ShaderManager::LoadComputeShader(Shaders::COMP_NOISE3D);
    glUseProgram(noiseGenShaderID);
    bindGenParams(noiseGenShaderID, chunkSize, chunkPos);
    glDispatchCompute(gridSize.x / 8, gridSize.y / 8, gridSize.z / 8);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

    primaryShaderID = ShaderManager::LoadShaders(
        Shaders::VERT_GRIDDER,
        Shaders::FRAG_SIMPLE,
        Shaders::GEOM_VOXELMESHER);
    glUseProgram(primaryShaderID);
    bindGenParams(primaryShaderID, chunkSize, chunkPos);
    mvpMatrixID = glGetUniformLocation(primaryShaderID, "MVP");
    const GLuint gridSizeID =
        glGetUniformLocation(primaryShaderID, "gridSize");
    glUniform3i(gridSizeID, gridSize.x, gridSize.y, gridSize.z);
}

void ProcGenEngine::Update(const mat4 mvpMatrix) {
    glUseProgram(primaryShaderID);
    glUniformMatrix4fv(mvpMatrixID, 1, GL_FALSE, &mvpMatrix[0][0]);
    glBindVertexArray(meshVertexArrayID);
    ivec3 meshGridSize = gridSize + ivec3(1);
    glDrawArrays(
        GL_POINTS, 0, meshGridSize.x * meshGridSize.y * meshGridSize.z);
}

void ProcGenEngine::bindGenParams(
    const GLuint shaderID, const vec3 chunkSize, const vec3 chunkPos) {
    glUniform1i(glGetUniformLocation(shaderID, "globalSeed"), globalSeed);
    glUniform3f(
        glGetUniformLocation(shaderID, "chunkSize"),
        chunkSize.x,
        chunkSize.y,
        chunkSize.z);
    glUniform3f(
        glGetUniformLocation(shaderID, "chunkPos"),
        chunkPos.x,
        chunkPos.y,
        chunkPos.z);
}
