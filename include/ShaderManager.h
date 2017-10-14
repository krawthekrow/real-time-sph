#pragma once

#include <GL/glew.h>

class ShaderManager {
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

public:
    static GLuint loadShaders(
        const char *vertSrc,
        const char *fragSrc,
        const char *geomSrc = nullptr);
    static GLuint loadComputeShader(const char *compSrc);
};
