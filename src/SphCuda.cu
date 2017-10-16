#include <cstdio>

#include <GL/glew.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>

#include "SphCuda.h"

using namespace glm;

#define GRAVITY -0.00001

__global__
void update(int numParts, vec3 *pos, vec3 *v) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; 
    for (int i = offset; i < numParts; i += stride) {
        pos[i] += v[i];
        v[i].y = v[i].y + GRAVITY;
        if (pos[i].x < 0) {
            pos[i].x = 0;
            v[i].x *= -1;
        }
        if (pos[i].x > 100.0f) {
            pos[i].x = 100.0f;
            v[i].x *= -1;
        }
        if (pos[i].y < 0) {
            pos[i].y = 0;
            v[i].y *= -1;
        }
        if (pos[i].y > 100.0f) {
            pos[i].y = 100.0f;
            v[i].y *= -1;
        }
        if (pos[i].z < 0) {
            pos[i].z = 0;
            v[i].z *= -1;
        }
        if (pos[i].z > 100.0f) {
            pos[i].z = 100.0f;
            v[i].z *= -1;
        }
    }
}

SphCuda::~SphCuda() {
    cudaFree(velocities);
}

void SphCuda::Init(const int _numParts, const GLuint vboGl) {
    numParts = _numParts;

    cudaGraphicsGLRegisterBuffer(
        &vbo, vboGl, cudaGraphicsRegisterFlagsNone);

    cudaMallocManaged(&velocities, numParts * sizeof(vec3));
}

void SphCuda::Update() {
    cudaGraphicsMapResources(1, &vbo);
    vec3 *vboDev;
    size_t bufSize;
    cudaGraphicsResourceGetMappedPointer(
        (void**)&vboDev, &bufSize, vbo);
    int blockSize = 256;
    int numBlocks = (numParts + blockSize - 1) / blockSize;
    update<<<numBlocks, blockSize>>>(numParts, vboDev, velocities);
    cudaGraphicsUnmapResources(1, &vbo);
}

vec3 *SphCuda::GetVelocitiesPtr() {
    return velocities;
}
