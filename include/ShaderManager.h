#pragma once

#include <GL/glew.h>

class ShaderManager {
public:
    static GLuint LoadShaders(
        const char *vertSrc,
        const char *fragSrc = nullptr,
        const char *geomSrc = nullptr);
    static GLuint LoadComputeShader(const char *compSrc);

private:
    static GLuint compileShader(
        const GLenum shaderType, const char *shaderSrc);
    static GLuint compileAndLinkShader(
        const GLuint programID,
        const GLenum shaderType,
        const char *shaderSrc);
    static void detachAndDeleteShader(
        const GLuint programID, const GLuint shaderID);
    static void linkAndDebugProgram(const GLuint programID);
};
