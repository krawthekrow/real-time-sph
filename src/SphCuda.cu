#include <cstdio>

#include <GL/glew.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#include "SphCuda.h"

__global__
void modify(int n, float *x) {
    for (int i = 0; i < n; i += 3) {
        x[i] = x[i] + 0.001f;
    }
}

void SphCuda::Init(const int _numParts, const GLuint vboGl) {
    numParts = _numParts;
    cudaGraphicsGLRegisterBuffer(
        &vbo, vboGl, cudaGraphicsRegisterFlagsNone);
}

void SphCuda::Update() {
    cudaGraphicsMapResources(1, &vbo);
    float *vboDev;
    size_t bufSize;
    cudaGraphicsResourceGetMappedPointer(
        (void**)&vboDev, &bufSize, vbo);
    modify<<<1, 1>>>(numParts, (float*)vboDev);
    cudaGraphicsUnmapResources(1, &vbo);
}
