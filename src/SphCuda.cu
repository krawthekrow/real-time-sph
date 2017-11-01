#include <cstdio>
#include <iostream>

#include <GL/glew.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>

#include "CudaUtils.h"

#include "SphCuda.h"

using namespace glm;

#define CELL_SIZE 4.0f // 8.0f
#define NUM_BITS_COORD 8
#define COORD_MASK ((1 << NUM_BITS_COORD) - 1)
#define X_OFFSET (NUM_BITS_COORD * 2)
#define Y_OFFSET (NUM_BITS_COORD)
#define Z_OFFSET 0
#define BAD_CELL -1

#define GRAVITY -0.0004f // -0.0001f
#define DRAG 0.001f // 0.001f
#define ELASTICITY 0.1f // 0.5f
// Particle size (for physics) squared
#define PART_SIZE_2 1.0f // 10.0f
#define COLLISION_FORCE 0.005f // 0.001f

ostream &operator<<(ostream &os, SphCuda::CellTouch const &c) {
    return os << "<" << c.hash << " " << (c.partInfo >> 1) <<
        " " << (c.partInfo & 1) << ">";
}

__global__
void initRandom(int numParts, curandState_t *states, int seed)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = offset; i < numParts; i += stride) {
        curand_init(seed, i, 0, &states[i]);
    }
}

__global__
void generateCellHashes(
    int numParts, ivec3 minBound, ivec3 maxBound,
    vec3 *pos, SphCuda::CellTouch *cellTouches) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = offset; i < numParts; i += stride) {
        ivec3 fineCellPos = ivec3(pos[i] / (CELL_SIZE / 2.0f));
        ivec3 cellPos = fineCellPos >> 1;
        ivec3 adjacencyType = (fineCellPos & 1) * 2 - 1;
        for (int j = 0; j < 8; j++) {
            int cellId = i * 8 + j;
            ivec3 jExpanded = ivec3(j, j >> 1, j >> 2) & 1;
            ivec3 otherCellPos = cellPos + adjacencyType * jExpanded;
            if (all(greaterThanEqual(otherCellPos, minBound)) &&
                all(lessThanEqual(otherCellPos, maxBound))) {
                cellTouches[cellId].hash =
                    (otherCellPos.x << X_OFFSET) |
                    (otherCellPos.y << Y_OFFSET) |
                    (otherCellPos.z << Z_OFFSET);
                cellTouches[cellId].partInfo = (i << 1) |
                    ((j == 0) ? 1 : 0);
            }
            else {
                cellTouches[cellId].hash = BAD_CELL;
            }
        }
    }
}

struct IsValidCellTouchPred {
    __device__
    bool operator()(SphCuda::CellTouch x) const {
        return x.hash != BAD_CELL;
    }
};

struct CellTouchCmp {
    __device__
    bool operator()(SphCuda::CellTouch a, SphCuda::CellTouch b) const {
        return a.hash < b.hash;
    }
};

__global__
void findChunks(
    int numCellTouches, SphCuda::CellTouch *cellTouches, int *chunkEnds) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = offset; i < numCellTouches; i += stride) {
        if (i == numCellTouches - 1 ||
            cellTouches[i].hash != cellTouches[i + 1].hash) {
            chunkEnds[i] = i + 1;
        }
        else {
            chunkEnds[i] = numCellTouches + 1;
        }
    }
}

struct IsValidChunkStartPred {
    int cmpVal;

    IsValidChunkStartPred(int numCellTouches) {
        cmpVal = numCellTouches + 1;
    }

    __device__
    bool operator()(int x) const {
        return x != cmpVal;
    }
};

__global__
void countCollisions(
    int numChunks, int *chunkEnds, SphCuda::CellTouch *cellTouches,
    int *numCollisions) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = offset; i < numChunks; i += stride) {
        int chunkStart = (i == 0) ? 0 : chunkEnds[i - 1];
        int chunkEnd = chunkEnds[i];
        int currNumCollisions = chunkEnd - chunkStart - 1;
        for (int j = chunkStart; j < chunkEnd; j++) {
            int firstPartInfo = cellTouches[j].partInfo;
            if ((firstPartInfo & 1) == 0) continue;
            int firstPart = firstPartInfo >> 1;
            numCollisions[firstPart] = currNumCollisions;
        }
    }
}

__global__
void findCollisions(
    int numChunks, int *chunkEnds, SphCuda::CellTouch *cellTouches,
    int *collisionChunkEnds, int maxNumCollisions, int *collisions) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = offset; i < numChunks; i += stride) {
        int chunkStart = (i == 0) ? 0 : chunkEnds[i - 1];
        int chunkEnd = chunkEnds[i];
        for (int j = chunkStart; j < chunkEnd; j++) {
            int firstPartInfo = cellTouches[j].partInfo;
            if ((firstPartInfo & 1) == 0) continue;
            int firstPart = firstPartInfo >> 1;
            int collisionChunkIndex =
                (firstPart == 0) ? 0 : collisionChunkEnds[firstPart - 1];
            for (int k = chunkStart; k < chunkEnd; k++) {
                if (collisionChunkIndex >= maxNumCollisions) break;
                if (j != k) {
                    int secondPart = cellTouches[k].partInfo >> 1;
                    collisions[collisionChunkIndex] = secondPart;
                    collisionChunkIndex++;
                }
            }
        }
    }
}

__global__
void computeContactForces(
    int numParts, vec3 *pos,
    int maxNumCollisions, int *collisionChunkEnds, int *collisions,
    vec3 *contactForces, curandState_t *randStates) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = offset; i < numParts; i += stride) {
        int chunkStart = (i == 0) ? 0 : collisionChunkEnds[i - 1];
        int chunkEnd = collisionChunkEnds[i];
        contactForces[i] = vec3(0.0f);
        for (int j = chunkStart; j < chunkEnd; j++) {
            if (j >= maxNumCollisions) break;
            vec3 relPos = pos[i] - pos[collisions[j]];
            float dist2 = dot(relPos, relPos);
            if (dist2 > 0.01f && dist2 < CELL_SIZE * CELL_SIZE)
                contactForces[i] +=
                    // relPos * (1 - dist2 / PART_SIZE_2) *
                    relPos * exp(-dist2 / PART_SIZE_2) *
                    COLLISION_FORCE;
        }
        // contactForces[firstPart] += (vec3(
        //     curand_uniform(&randStates[firstPart]),
        //     curand_uniform(&randStates[firstPart]),
        //     curand_uniform(&randStates[firstPart])
        // ) - 0.5f) * 0.001f * length(contactForces[firstPart]);
    }
}

// __global__
// void computeContactForces(
//     int numParts, vec3 *pos, int numChunks, int *chunkEnds,
//     SphCuda::CellTouch *cellTouches,
//     vec3 *contactForces, curandState_t *randStates) {
//     int offset = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
//     for (int i = offset; i < numChunks; i += stride) {
//         int chunkStart = (i == 0) ? 0 : chunkEnds[i - 1];
//         int chunkEnd = chunkEnds[i];
//         for (int j = chunkStart; j < chunkEnd; j++) {
//             int firstPartInfo = cellTouches[j].partInfo;
//             if ((firstPartInfo & 1) == 0) continue;
//             int firstPart = firstPartInfo >> 1;
//             contactForces[firstPart] = vec3(0.0f);
//             for (int k = chunkStart; k < chunkEnd; k++) {
//                 if (j != k) {
//                     int secondPart = cellTouches[k].partInfo >> 1;
//                     vec3 relPos = pos[firstPart] - pos[secondPart];
//                     float dist2 = dot(relPos, relPos);
//                     if (dist2 > 0.01f && dist2 < CELL_SIZE * CELL_SIZE)
//                         contactForces[firstPart] +=
//                             // relPos * (1 - dist2 / PART_SIZE_2) *
//                             relPos * exp(-dist2 / PART_SIZE_2) *
//                             COLLISION_FORCE;
//                 }
//             }
//             // contactForces[firstPart] += (vec3(
//             //     curand_uniform(&randStates[firstPart]),
//             //     curand_uniform(&randStates[firstPart]),
//             //     curand_uniform(&randStates[firstPart])
//             // ) - 0.5f) * 0.001f * length(contactForces[firstPart]);
//         }
//     }
// }

// __global__
// void computeContactForces(
//     int numParts, vec3 *pos, int numChunks, int *chunkEnds,
//     vec3 *contactForces, curandState_t *randStates) {
//     int offset = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
//     for (int i = offset; i < numParts; i += stride) {
//         contactForces[i] = vec3(0.0f);
//         for (int j = 0; j < numParts; j++) {
//             vec3 relPos = pos[i] - pos[j];
//             float dist2 = dot(relPos, relPos);
//             if (dist2 > 0.01f && dist2 < CELL_SIZE * CELL_SIZE)
//                 contactForces[i] +=
//                     relPos * exp(-dist2 / PART_SIZE_2) * COLLISION_FORCE;
//         }
//         contactForces[i] += (vec3(
//             curand_uniform(&randStates[i]),
//             curand_uniform(&randStates[i]),
//             curand_uniform(&randStates[i])
//         ) - 0.5f) * 0.001f * length(contactForces[i]);
//     }
// }

__global__
void update(
    int numParts, vec3 minBound, vec3 maxBound,
    vec3 *pos, vec3 *v, vec3 *contactForces, curandState_t *randStates) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = offset; i < numParts; i += stride) {
        pos[i] += v[i];
        v[i].y = v[i].y + GRAVITY;
        v[i] *= (1 - DRAG);
        v[i] += contactForces[i];
        // float vmag = length(v[i]);
        // float VLIMIT = 0.1f;
        // if (vmag > VLIMIT) v[i] = v[i] / vmag * VLIMIT;
        // v[i] += (vec3(
        //     curand_uniform(&randStates[i]),
        //     curand_uniform(&randStates[i]),
        //     curand_uniform(&randStates[i])
        // ) - 0.5f) * 0.001f;
        if (pos[i].x < minBound.x) {
            pos[i].x = 2.0f * minBound.x - pos[i].x;
            v[i].x *= -ELASTICITY;
        }
        if (pos[i].x > maxBound.x) {
            pos[i].x = 2.0f * maxBound.x - pos[i].x;
            v[i].x *= -ELASTICITY;
        }
        if (pos[i].y < minBound.y) {
            pos[i].y = 2.0f * minBound.y - pos[i].y;
            v[i].y *= -ELASTICITY;
        }
        if (pos[i].y > maxBound.y) {
            pos[i].y = 2.0f * maxBound.y - pos[i].y;
            v[i].y *= -ELASTICITY;
        }
        if (pos[i].z < minBound.z) {
            pos[i].z = 2.0f * minBound.z - pos[i].z;
            v[i].z *= -ELASTICITY;
        }
        if (pos[i].z > maxBound.z) {
            pos[i].z = 2.0f * maxBound.z - pos[i].z;
            v[i].z *= -ELASTICITY;
        }
    }
}

__global__
void computeAccel(
    int numParts, vec3 *pos, vec3 *v, vec3 *contactForces, vec3 *accel) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = offset; i < numParts; i += stride) {
        accel[i] = - DRAG * v[i] + contactForces[i];
        accel[i].y += GRAVITY;
    }
}

__global__
void advanceState(
    int numParts, vec3 *pos, vec3 *v,
    vec3 *rk1v, vec3 *rk1dv, vec3 *rk2v, vec3 *rk2dv,
    vec3 *rk3v, vec3 *rk3dv, vec3 *rk4v, vec3 *rk4dv) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = offset; i < numParts; i += stride) {
        pos[i] +=
            (rk1v[i] + 2.0f * rk2v[i] + 2.0f * rk3v[i] + rk4v[i]) /
            6.0f;
        v[i] +=
            (rk1dv[i] + 2.0f * rk2dv[i] + 2.0f * rk3dv[i] + rk4dv[i]) /
            6.0f;
    }
}

__global__
void enforceBoundary(
    int numParts, vec3 minBound, vec3 maxBound, vec3 *pos, vec3 *v) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = offset; i < numParts; i += stride) {
        // float vmag = length(v[i]);
        // float VLIMIT = 0.1f;
        // if (vmag > VLIMIT) v[i] = v[i] / vmag * VLIMIT;
        // v[i] += (vec3(
        //     curand_uniform(&randStates[i]),
        //     curand_uniform(&randStates[i]),
        //     curand_uniform(&randStates[i])
        // ) - 0.5f) * 0.001f;
        if (pos[i].x < minBound.x) {
            pos[i].x = 2.0f * minBound.x - pos[i].x;
            v[i].x *= -ELASTICITY;
        }
        if (pos[i].x > maxBound.x) {
            pos[i].x = 2.0f * maxBound.x - pos[i].x;
            v[i].x *= -ELASTICITY;
        }
        if (pos[i].y < minBound.y) {
            pos[i].y = 2.0f * minBound.y - pos[i].y;
            v[i].y *= -ELASTICITY;
        }
        if (pos[i].y > maxBound.y) {
            pos[i].y = 2.0f * maxBound.y - pos[i].y;
            v[i].y *= -ELASTICITY;
        }
        if (pos[i].z < minBound.z) {
            pos[i].z = 2.0f * minBound.z - pos[i].z;
            v[i].z *= -ELASTICITY;
        }
        if (pos[i].z > maxBound.z) {
            pos[i].z = 2.0f * maxBound.z - pos[i].z;
            v[i].z *= -ELASTICITY;
        }
        // vec3 minBoundDist = pos[i] - minBound;
        // v[i] += vec3(lessThan(minBoundDist, vec3(CELL_SIZE))) *
        //     exp(-minBoundDist * minBoundDist / PART_SIZE_2) *
        //     COLLISION_FORCE;
        // vec3 maxBoundDist = maxBound - pos[i];
        // v[i] -= vec3(lessThan(maxBoundDist, vec3(CELL_SIZE))) *
        //     exp(-maxBoundDist * maxBoundDist / PART_SIZE_2) *
        //     COLLISION_FORCE;
    }
}

__global__
void multAddVec3(int numParts, float multFactor,
    vec3 *a, vec3 *b, vec3 *dest) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = offset; i < numParts; i += stride) {
        dest[i] = a[i] + multFactor * b[i];
    }
}

SphCuda::~SphCuda() {
    cudaGraphicsUnmapResources(1, &vbo);
    cudaFree(velocities);
    cudaFree(contactForces);

    cudaFree(cellTouchesSparse);
    cudaFree(cellTouches);
    cudaFree(chunkEndsSparse);
    cudaFree(chunkEnds);

    cudaFree(numCollisions);
    cudaFree(collisionChunkEnds);
    cudaFree(collisions);

    cudaFree(rk1dv);
    cudaFree(rk2p);
    cudaFree(rk2v);
    cudaFree(rk2dv);
    cudaFree(rk3p);
    cudaFree(rk3v);
    cudaFree(rk3dv);
    cudaFree(rk4p);
    cudaFree(rk4v);
    cudaFree(rk4dv);
}

void SphCuda::Init(
    const int _numParts, const GLuint vboGl,
    const vec3 *_minBound, const vec3 *_maxBound) {
    numParts = _numParts;
    minBound = *_minBound;
    maxBound = *_maxBound;
    minBoundCell = ivec3(minBound / CELL_SIZE);
    maxBoundCell = ivec3(maxBound / CELL_SIZE);

    blockSize = 256;
    numBlocksParts = (numParts + blockSize - 1) / blockSize;

    cudaGraphicsGLRegisterBuffer(
        &vbo, vboGl, cudaGraphicsRegisterFlagsNone);
    cudaGraphicsMapResources(1, &vbo);
    size_t bufSize;
    cudaGraphicsResourceGetMappedPointer((void**)&pos, &bufSize, vbo);

    cudaMallocManaged(&velocities, numParts * sizeof(vec3));
    cudaMallocManaged(&contactForces, numParts * sizeof(vec3));

    // Each particle overlaps 8 cells
    cudaMalloc(&cellTouchesSparse, 8 * numParts * sizeof(CellTouch));
    cudaMalloc(&cellTouches, 8 * numParts * sizeof(CellTouch));
    cudaMalloc(&chunkEndsSparse, 8 * numParts * sizeof(int));
    cudaMalloc(&chunkEnds, 8 * numParts * sizeof(int));

    cellTouchesSparsePtr = thrust::device_pointer_cast(cellTouchesSparse);
    cellTouchesPtr = thrust::device_pointer_cast(cellTouches);
    chunkEndsSparsePtr = thrust::device_pointer_cast(chunkEndsSparse);
    chunkEndsPtr = thrust::device_pointer_cast(chunkEnds);

    cudaMalloc(&numCollisions, numParts * sizeof(int));
    cudaMalloc(&collisionChunkEnds, numParts * sizeof(int));
    // Assume each particle has a max valency of 16
    maxNumCollisions = 32 * numParts;
    cudaMallocManaged(&collisions, maxNumCollisions * sizeof(int));

    numCollisionsPtr = thrust::device_pointer_cast(numCollisions);
    collisionChunkEndsPtr = thrust::device_pointer_cast(collisionChunkEnds);

    cudaMallocManaged(&rk1dv, numParts * sizeof(vec3));
    cudaMallocManaged(&rk2p, numParts * sizeof(vec3));
    cudaMallocManaged(&rk2v, numParts * sizeof(vec3));
    cudaMallocManaged(&rk2dv, numParts * sizeof(vec3));
    cudaMallocManaged(&rk3p, numParts * sizeof(vec3));
    cudaMallocManaged(&rk3v, numParts * sizeof(vec3));
    cudaMallocManaged(&rk3dv, numParts * sizeof(vec3));
    cudaMallocManaged(&rk4p, numParts * sizeof(vec3));
    cudaMallocManaged(&rk4v, numParts * sizeof(vec3));
    cudaMallocManaged(&rk4dv, numParts * sizeof(vec3));

    cudaMalloc(&randStates, numParts * sizeof(curandState_t));
    initRandom<<<numBlocksParts, blockSize>>>(numParts, randStates, rand());
    cudaDeviceSynchronize();
}

void SphCuda::Update() {
    generateCellHashes<<<numBlocksParts, blockSize>>>(
        numParts, minBoundCell, maxBoundCell,
        pos, cellTouchesSparse);
    int numCellTouches = thrust::copy_if(
        cellTouchesSparsePtr, cellTouchesSparsePtr + 8 * numParts,
        cellTouchesPtr, IsValidCellTouchPred()) - cellTouchesPtr;

    int numBlocksCellTouches =
        (numCellTouches + blockSize - 1) / blockSize;
    thrust::stable_sort(
        cellTouchesPtr, cellTouchesPtr + numCellTouches, CellTouchCmp());
    findChunks<<<numBlocksCellTouches, blockSize>>>(
        numCellTouches, cellTouches, chunkEndsSparse);
    int numChunks = thrust::copy_if(
        chunkEndsSparsePtr, chunkEndsSparsePtr + numCellTouches,
        chunkEndsPtr, IsValidChunkStartPred(numCellTouches)) -
        chunkEndsPtr;
    // printf("Number of chunks: %d\n", numChunks);
    // printf("Number of touches: %d\n", numCellTouches);

    int blockSizeChunks = 256;
    int numBlocksChunks =
        (numChunks + blockSizeChunks - 1) / blockSizeChunks;
    // CudaUtils::DebugPrint(chunkEnds, 16);
    countCollisions<<<numBlocksChunks, blockSizeChunks>>>(
        numChunks, chunkEnds, cellTouches, numCollisions);
    thrust::inclusive_scan(numCollisionsPtr, numCollisionsPtr + numParts,
        collisionChunkEndsPtr);
    findCollisions<<<numBlocksChunks, blockSizeChunks>>>(
        numChunks, chunkEnds, cellTouches, collisionChunkEnds,
        maxNumCollisions, collisions);

    // computeContactForces<<<numBlocksParts, blockSize>>>(
    //     numParts, pos, collisionChunkEnds, collisions,
    //     contactForces, randStates);
    // computeAccel<<<numBlocksParts, blockSize>>>(
    //     numParts, pos, velocities, contactForces, accel);
    // multAddVec3<<<numBlocksParts, blockSize>>>(
    //     numParts, 1.0f, pos, velocities, pos);
    // multAddVec3<<<numBlocksParts, blockSize>>>(
    //     numParts, 1.0f, velocities, accel, velocities);

    vec3 *rk1p = pos;
    vec3 *rk1v = velocities;
    computeAccelRK(rk1p, rk1v, rk1dv);
    // advanceStateRK(pos, velocities, 0.5f, rk1v, rk1dv, rk2p, rk2v);
    // computeAccelRK(rk2p, rk2v, rk2dv);
    // advanceStateRK(pos, velocities, 0.5f, rk2v, rk2dv, rk3p, rk3v);
    // computeAccelRK(rk3p, rk3v, rk3dv);
    // advanceStateRK(pos, velocities, 1.0f, rk3v, rk3dv, rk4p, rk4v);
    // computeAccelRK(rk4p, rk4v, rk4dv);
    // advanceState<<<numBlocksParts, blockSize>>>(
    //     numParts, pos, velocities,
    //     rk1v, rk1dv, rk2v, rk2dv, rk3v, rk3dv, rk4v, rk4dv);
    advanceStateRK(pos, velocities, 1.0f, rk1v, rk1dv, pos, velocities);

    enforceBoundary<<<numBlocksParts, blockSize>>>(
        numParts, minBound, maxBound, pos, velocities);
    // update<<<numBlocksParts, blockSize>>>(
    //     numParts, minBound, maxBound,
    //     pos, velocities, contactForces, randStates);
}

vec3 *SphCuda::GetVelocitiesPtr() {
    return velocities;
}

void SphCuda::computeAccelRK(
    vec3 * const currPos, vec3 * const currVel, vec3 * const currAccel) {
    computeContactForces<<<numBlocksParts, blockSize>>>(
        numParts, currPos,
        maxNumCollisions, collisionChunkEnds, collisions,
        contactForces, randStates);
    computeAccel<<<numBlocksParts, blockSize>>>(
        numParts, currPos, currVel, contactForces, currAccel);
}

void SphCuda::advanceStateRK(
    vec3 * const currPos, vec3 * const currVel, const float timeStep,
    vec3 * const dpos, vec3 * const dvel,
    vec3 * const destPos, vec3 * const destVel) {
    multAddVec3<<<numBlocksParts, blockSize>>>(
        numParts, timeStep, currPos, dpos, destPos);
    multAddVec3<<<numBlocksParts, blockSize>>>(
        numParts, timeStep, currVel, dvel, destVel);
}
