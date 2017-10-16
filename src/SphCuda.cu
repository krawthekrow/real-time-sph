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

#define GRAVITY -0.0001
#define DRAG 0.001
#define ELASTICITY 0.5

__global__
void update(int numParts, vec3 *pos, vec3 *v, vec3 *contactForces) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; 
    for (int i = offset; i < numParts; i += stride) {
        contactForces[i] = vec3(0.0f);
        for (int j = 0; j < numParts; j++) {
            vec3 relPos = pos[i] - pos[j];
            float dist2 = dot(relPos, relPos);
            if (dist2 > 0.01)
                contactForces[i] +=
                    relPos * exp(-dist2 * dist2 / 100.0f) * 0.001f;
        }
    }
    for (int i = offset; i < numParts; i += stride) {
        pos[i] += v[i];
        v[i].y = v[i].y + GRAVITY;
        v[i] *= (1 - DRAG);
        v[i] += contactForces[i];
        if (pos[i].x < 0) {
            pos[i].x = 0;
            v[i].x *= -ELASTICITY;
        }
        if (pos[i].x > 100.0f) {
            pos[i].x = 100.0f;
            v[i].x *= -ELASTICITY;
        }
        if (pos[i].y < 0) {
            pos[i].y = 0;
            v[i].y *= -ELASTICITY;
        }
        if (pos[i].y > 100.0f) {
            pos[i].y = 100.0f;
            v[i].y *= -ELASTICITY;
        }
        if (pos[i].z < 0) {
            pos[i].z = 0;
            v[i].z *= -ELASTICITY;
        }
        if (pos[i].z > 100.0f) {
            pos[i].z = 100.0f;
            v[i].z *= -ELASTICITY;
        }
    }
}

SphCuda::~SphCuda() {
    cudaFree(velocities);
    cudaFree(contactForces);
}

void SphCuda::Init(const int _numParts, const GLuint vboGl) {
    numParts = _numParts;

    cudaGraphicsGLRegisterBuffer(
        &vbo, vboGl, cudaGraphicsRegisterFlagsNone);

    cudaMallocManaged(&velocities, numParts * sizeof(vec3));
    cudaMallocManaged(&contactForces, numParts * sizeof(vec3));
}

void SphCuda::Update() {
    cudaGraphicsMapResources(1, &vbo);
    vec3 *vboDev;
    size_t bufSize;
    cudaGraphicsResourceGetMappedPointer(
        (void**)&vboDev, &bufSize, vbo);
    int blockSize = 256;
    int numBlocks = (numParts + blockSize - 1) / blockSize;
    update<<<numBlocks, blockSize>>>(
        numParts, vboDev, velocities, contactForces);
    cudaGraphicsUnmapResources(1, &vbo);
}

vec3 *SphCuda::GetVelocitiesPtr() {
    return velocities;
}
