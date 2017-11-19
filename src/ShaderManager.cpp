#include "ShaderManager.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

GLuint ShaderManager::LoadShaders(
    const char *vertSrc, const char *fragSrc, const char *geomSrc) {
    GLuint programID = glCreateProgram();
    GLuint vertShaderID =
        compileAndLinkShader(programID, GL_VERTEX_SHADER, vertSrc);
    GLuint fragShaderID;
    if (fragSrc != NULL)
        fragShaderID =
            compileAndLinkShader(programID, GL_FRAGMENT_SHADER, fragSrc);
    GLuint geomShaderID;
    if (geomSrc != NULL)
        geomShaderID =
            compileAndLinkShader(programID, GL_GEOMETRY_SHADER, geomSrc);

    linkAndDebugProgram(programID);

    detachAndDeleteShader(programID, vertShaderID);
    if (fragSrc != NULL)
        detachAndDeleteShader(programID, fragShaderID);
    if (geomSrc != NULL)
        detachAndDeleteShader(programID, geomShaderID);

    return programID;
}

GLuint ShaderManager::LoadComputeShader(const char *compSrc) {
    GLuint programID = glCreateProgram();
    GLuint compShaderID =
        compileAndLinkShader(programID, GL_COMPUTE_SHADER, compSrc);
    linkAndDebugProgram(programID);
    detachAndDeleteShader(programID, compShaderID);
    return programID;
}

GLuint ShaderManager::compileShader(
    const GLenum shaderType, const char *shaderSrc) {
    GLuint shaderID = glCreateShader(shaderType);
    glShaderSource(shaderID, 1, &shaderSrc, NULL);
    glCompileShader(shaderID);

    GLint result = GL_FALSE;
    int infoLogLength;

    glGetShaderiv(shaderID, GL_COMPILE_STATUS, &result);
    glGetShaderiv(shaderID, GL_INFO_LOG_LENGTH, &infoLogLength);
    if (infoLogLength > 0) {
        vector<char> errorMessage(infoLogLength + 1);
        glGetShaderInfoLog(
            shaderID, infoLogLength, NULL, &errorMessage[0]);
        fprintf(stderr, "%s\n", &errorMessage[0]);
    }
    return shaderID;
}

GLuint ShaderManager::compileAndLinkShader(
    const GLuint programID, const GLenum shaderType,
    const char *shaderSrc) {
    GLuint shaderID = compileShader(shaderType, shaderSrc);
    glAttachShader(programID, shaderID);
    return shaderID;
}

void ShaderManager::detachAndDeleteShader(
    const GLuint programID, const GLuint shaderID) {
    glDetachShader(programID, shaderID);
    glDeleteShader(shaderID);
}

void ShaderManager::linkAndDebugProgram(const GLuint programID) {
    glLinkProgram(programID);

    GLint result = GL_FALSE;
    int infoLogLength;

    glGetProgramiv(programID, GL_LINK_STATUS, &result);
    glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &infoLogLength);
    if (infoLogLength > 0) {
        vector<char> errorMessage(infoLogLength + 1);
        glGetProgramInfoLog(
            programID, infoLogLength, NULL, &errorMessage[0]);
        printf("%s\n", &errorMessage[0]);
    }
}
