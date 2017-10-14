#include <GL/glew.h>
#include <glm/glm.hpp>

#include <cuda_gl_interop.h>

#include "CudaGlInterop.h"

using namespace std;
using namespace glm;

void CudaGlInterop::genCudaVbo(const unsigned int numVerts) {
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, numVerts * 16, nullptr, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(vbo);
}
