#include <cstdio>
#include <iostream>

#include <GL/glew.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#include <thrust/sort.h>
#include <thrust/gather.h>
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
#define IS_HOME_CELL_OFFSET (NUM_BITS_COORD * 3)
#define COORDS_MASK ((1 << (NUM_BITS_COORD * 3)) - 1)
#define BAD_CELL -1

#define GRAVITY -0.0005f // -0.0001f
#define DRAG 0.0001f // 0.001f
#define VISCOSITY 0.02f
#define ELASTICITY 0.8f // 0.5f
// Particle size (for physics) squared
#define PART_SIZE_2 1.0f // 10.0f
#define COLLISION_FORCE 0.005f // 0.001f

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
    vec3 *pos, int *cellTouchHashes, int *cellTouchPartIds) {
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
                cellTouchHashes[cellId] =
                    ((otherCellPos.x & COORD_MASK) << X_OFFSET) |
                    ((otherCellPos.y & COORD_MASK) << Y_OFFSET) |
                    ((otherCellPos.z & COORD_MASK) << Z_OFFSET) |
                    (((j == 0) ? 1 : 0) << IS_HOME_CELL_OFFSET);
                cellTouchPartIds[cellId] = i;
            }
            else {
                cellTouchHashes[cellId] = BAD_CELL;
            }
        }
    }
}

struct IsValidCellTouchPred {
    __device__
    bool operator()(const thrust::tuple<int, int> &x) const {
        return x.get<0>() != BAD_CELL;
    }
};

struct CellTouchCmp {
    __device__
    bool operator()(const int &a, const int &b) const {
        return (a & COORDS_MASK) < (b & COORDS_MASK);
    }
};

__global__
void findChunks(
    int numCellTouches, int *cellTouchHashes, int *chunkEnds) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = offset; i < numCellTouches; i += stride) {
        if (i == numCellTouches - 1 ||
            (cellTouchHashes[i] & COORDS_MASK) !=
            (cellTouchHashes[i + 1] & COORDS_MASK)) {
            chunkEnds[i] = i + 1;
        }
        else {
            chunkEnds[i] = numCellTouches + 1;
        }
    }
}

struct IsValidChunkStartPred {
    int cmpVal;

    IsValidChunkStartPred(const int &numCellTouches) {
        cmpVal = numCellTouches + 1;
    }

    __device__
    bool operator()(int x) const {
        return x != cmpVal;
    }
};

__global__
void countCollisions(
    int numChunks, int *chunkEnds,
    int *cellTouchHashes, int *cellTouchPartIds,
    int *numCollisions, int *homeChunks) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = offset; i < numChunks; i += stride) {
        int chunkStart = (i == 0) ? 0 : chunkEnds[i - 1];
        int chunkEnd = chunkEnds[i];
        int currNumCollisions = chunkEnd - chunkStart;
        for (int j = chunkStart; j < chunkEnd; j++) {
            if (((cellTouchHashes[j] >> IS_HOME_CELL_OFFSET) & 1) == 0)
                continue;
            int firstPart = cellTouchPartIds[j];
            numCollisions[firstPart] = currNumCollisions;
            homeChunks[firstPart] = chunkStart;
        }
    }
}

struct ShouldCopyCollisionPred {
    int maxNumCollisions;

    ShouldCopyCollisionPred(const int &_maxNumCollisions) {
        maxNumCollisions = _maxNumCollisions;
    }

    // x is zip(numCollisions, collisionChunkStarts)
    __device__
    bool operator()(
        const thrust::tuple<int, int> &x) const {
        return x.get<0>() != 0 && x.get<1>() < maxNumCollisions;
    }
};

__global__
void checkShouldCopyCollision(int numParts,
    int *numCollisions, int *collisionChunkStarts,
    bool *shouldCopyCollisionPtr, int maxNumCollisions) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = offset; i < numParts; i += stride) {
        shouldCopyCollisionPtr[i] = numCollisions[i] != 0 &&
            collisionChunkStarts[i] < maxNumCollisions;
    }
}

struct CollisionIndexListCreator {
    __device__
    int operator()(const int &a, const int &b) const {
        if (b < 0) {
            if (a < 0) return a + b;
            else return a - b;
        }
        else return b;
    }
};

__global__
void computeContactForces(
    int numParts, vec3 *pos, vec3 *v,
    int maxNumCollisions, int *collisionChunkEnds, int *collisions,
    vec3 *contactForces, curandState_t *randStates) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = offset; i < numParts; i += stride) {
        int chunkStart = (i == 0) ? 0 : collisionChunkEnds[i - 1];
        int chunkEnd = collisionChunkEnds[i];

        // float density = 0.0f;
        // vec3 ddensity(0.0f);
        // for (int j = chunkStart; j < chunkEnd; j++) {
        //     if (j >= maxNumCollisions) break;
        //     vec3 relPos = pos[i] - pos[collisions[j]];
        //     float dist2 = dot(relPos, relPos);
        //     if (dist2 > 0.01f && dist2 < CELL_SIZE * CELL_SIZE) {
        //         density += exp(-dist2 / (PART_SIZE_2 * 1.0f));
        //         // density += exp(-sqrt(dist2 / (PART_SIZE_2 * 1.0f)));
        //         ddensity += relPos * density;
        //     }
        // }
        // contactForces[i] = COLLISION_FORCE *
        //     (density + 0.1f) * ddensity * 0.001f;

        vec3 contactForce(0.0f);
        for (int j = chunkStart; j < chunkEnd; j++) {
            if (j >= maxNumCollisions) break;
            if (i == collisions[j]) continue;
            vec3 relPos = pos[i] - pos[collisions[j]];
            float dist2 = dot(relPos, relPos);
            if (dist2 > 0.01f && dist2 < CELL_SIZE * CELL_SIZE / 4.0f) {
                float normdist2 = dist2 / PART_SIZE_2;
                // contactForce +=
                //     relPos * (1.0f - normdist2 / 1.5f +
                //     normdist2 * normdist2 / 9.5f
                //     ) *
                //     COLLISION_FORCE;
                vec3 currForce =
                    // relPos * (1 - dist2 / PART_SIZE_2) *
                    relPos * exp(-normdist2) *
                    COLLISION_FORCE;
                contactForce += currForce;
                static const float forceScale = COLLISION_FORCE * PART_SIZE_2;
                // contactForce += -1.0f / forceScale * dot(v[i], currForce) * currForce / dot(currForce, currForce) * VISCOSITY;
                contactForce += -1.0f / forceScale * dot(v[i] - v[collisions[j]], currForce) * currForce / length(currForce) * VISCOSITY;
            }
        }
        contactForces[i] = contactForce;

        // contactForces[firstPart] += (vec3(
        //     curand_uniform(&randStates[firstPart]),
        //     curand_uniform(&randStates[firstPart]),
        //     curand_uniform(&randStates[firstPart])
        // ) - 0.5f) * 0.001f * length(contactForces[firstPart]);
    }
}

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
    int numParts, vec3 *pos, vec3 *v, vec3 *contactForces, vec3 *accel,
    double t, float rotAmt) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // static const float forceScale = COLLISION_FORCE * PART_SIZE_2;
    for (int i = offset; i < numParts; i += stride) {
        vec3 contactForce = contactForces[i];
        // accel[i] = - max(min(DRAG, DRAG * length(contactForce) / forceScale), DRAG / 2.0f) * v[i] + contactForce;
        accel[i] = - DRAG * v[i] + contactForce;
        accel[i] += GRAVITY * vec3(sin(rotAmt), cos(rotAmt), 0.0f);
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
    int numParts, vec3 minBound, vec3 maxBound, vec3 *pos, vec3 *v,
    curandState_t *randStates) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = offset; i < numParts; i += stride) {
        // float vmag = length(v[i]);
        // float VLIMIT = 0.1f;
        // if (vmag > VLIMIT) v[i] = v[i] / vmag * VLIMIT;
        // pos[i] += (vec3(
        //     curand_uniform(&randStates[i]),
        //     curand_uniform(&randStates[i]),
        //     curand_uniform(&randStates[i])
        // ) - 0.5f) * 0.01f;
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
        vec3 currForce(0.0f);
        vec3 minBoundDist = pos[i] - minBound;
        currForce += vec3(lessThan(minBoundDist, vec3(CELL_SIZE))) *
            exp(-minBoundDist * minBoundDist / PART_SIZE_2) *
            COLLISION_FORCE;
        vec3 maxBoundDist = maxBound - pos[i];
        currForce -= vec3(lessThan(maxBoundDist, vec3(CELL_SIZE))) *
            exp(-maxBoundDist * maxBoundDist / PART_SIZE_2) *
            COLLISION_FORCE;
        // static const float forceScale = COLLISION_FORCE * PART_SIZE_2;
        // float forceSquared = dot(currForce, currForce);
        // if (forceSquared > 0)
        //     currForce += -1.0f / forceScale * dot(v[i], currForce) * currForce / forceSquared * VISCOSITY * 0.1f;
        v[i] += currForce;
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

    cudaFree(cellTouchHashes);
    cudaFree(cellTouchPartIds);
    cudaFree(chunkEnds);

    cudaFree(numCollisions);
    cudaFree(collisionChunkStarts);
    cudaFree(collisionChunkEnds);
    cudaFree(collisions);
    cudaFree(homeChunks);
    cudaFree(shouldCopyCollision);

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
    const int &_numParts, const GLuint &vboGl,
    const vec3 &_minBound, const vec3 &_maxBound) {
    numParts = _numParts;
    minBound = _minBound;
    maxBound = _maxBound;
    minBoundCell = ivec3(floor(minBound / CELL_SIZE));
    maxBoundCell = ivec3(ceil(maxBound / CELL_SIZE));

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
    cudaMalloc(&cellTouchHashes, 8 * numParts * sizeof(int));
    cudaMalloc(&cellTouchPartIds, 8 * numParts * sizeof(int));
    cudaMalloc(&chunkEnds, 8 * numParts * sizeof(int));

    cellTouchHashesPtr = thrust::device_pointer_cast(cellTouchHashes);
    cellTouchPartIdsPtr = thrust::device_pointer_cast(cellTouchPartIds);
    cellTouchesPtr = thrust::make_zip_iterator(thrust::make_tuple(
        cellTouchHashes, cellTouchPartIds));
    chunkEndsPtr = thrust::device_pointer_cast(chunkEnds);

    cudaMalloc(&numCollisions, numParts * sizeof(int));
    cudaMalloc(&collisionChunkStarts, numParts * sizeof(int));
    cudaMalloc(&collisionChunkEnds, numParts * sizeof(int));
    // Assume each particle has a max valency of 16
    maxNumCollisions = 128 * numParts;
    cudaMalloc(&collisions, maxNumCollisions * sizeof(int));
    cudaMalloc(&homeChunks, numParts * sizeof(int));
    cudaMalloc(&shouldCopyCollision, numParts * sizeof(bool));

    numCollisionsPtr = thrust::device_pointer_cast(numCollisions);
    collisionChunkStartsPtr =
        thrust::device_pointer_cast(collisionChunkStarts);
    collisionChunkEndsPtr =
        thrust::device_pointer_cast(collisionChunkEnds);
    collisionsPtr = thrust::device_pointer_cast(collisions);
    homeChunksPtr =
        thrust::device_pointer_cast(homeChunks);
    shouldCopyCollisionPtr =
        thrust::device_pointer_cast(shouldCopyCollision);
    cellTouchPartIdsPtr =
        thrust::device_pointer_cast(cellTouchPartIds);

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

void SphCuda::Update(const double &currTime, const float &rotAmt) {
    generateCellHashes<<<numBlocksParts, blockSize>>>(
        numParts, minBoundCell, maxBoundCell,
        pos, cellTouchHashes, cellTouchPartIds);
    int numCellTouches = thrust::copy_if(
        cellTouchesPtr, cellTouchesPtr + 8 * numParts,
        cellTouchesPtr, IsValidCellTouchPred()) - cellTouchesPtr;
    int numBlocksCellTouches =
        (numCellTouches + blockSize - 1) / blockSize;
    thrust::sort_by_key(
        cellTouchHashesPtr, cellTouchHashesPtr + numCellTouches,
        cellTouchPartIdsPtr, CellTouchCmp());
    findChunks<<<numBlocksCellTouches, blockSize>>>(
        numCellTouches, cellTouchHashes, chunkEnds);
    int numChunks = thrust::copy_if(
        chunkEndsPtr, chunkEndsPtr + numCellTouches,
        chunkEndsPtr, IsValidChunkStartPred(numCellTouches)) -
        chunkEndsPtr;
    // printf("Number of chunks: %d\n", numChunks);
    // printf("Number of touches: %d\n", numCellTouches);

    int blockSizeChunks = 256;
    int numBlocksChunks =
        (numChunks + blockSizeChunks - 1) / blockSizeChunks;
    // CudaUtils::DebugPrint(chunkEnds, 16);
    countCollisions<<<numBlocksChunks, blockSizeChunks>>>(
        numChunks, chunkEnds, cellTouchHashes, cellTouchPartIds,
        numCollisions, homeChunks);
    thrust::inclusive_scan(numCollisionsPtr, numCollisionsPtr + numParts,
        collisionChunkEndsPtr);
    thrust::exclusive_scan(numCollisionsPtr, numCollisionsPtr + numParts,
        collisionChunkStartsPtr);
    thrust::fill(collisionsPtr, collisionsPtr + maxNumCollisions, -1);
    thrust::scatter_if(homeChunksPtr, homeChunksPtr + numParts,
        collisionChunkStartsPtr,
        thrust::make_zip_iterator(
            thrust::make_tuple(numCollisionsPtr, collisionChunkStartsPtr)),
        collisionsPtr, ShouldCopyCollisionPred(maxNumCollisions));
    thrust::inclusive_scan(collisionsPtr, collisionsPtr + maxNumCollisions,
        collisionsPtr, CollisionIndexListCreator());
    thrust::gather(collisionsPtr, collisionsPtr + maxNumCollisions,
        cellTouchPartIdsPtr, collisionsPtr);

    // findCollisions<<<numBlocksChunks, blockSizeChunks>>>(
    //     numChunks, chunkEnds, cellTouches, collisionChunkEnds,
    //     maxNumCollisions, collisions);

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
    computeAccelRK(rk1p, rk1v, rk1dv, currTime, rotAmt);

    const static bool USE_RK4 = false;

    if (USE_RK4) {
        advanceStateRK(pos, velocities, 0.5f, rk1v, rk1dv, rk2p, rk2v);
        computeAccelRK(rk2p, rk2v, rk2dv, currTime, rotAmt);
        advanceStateRK(pos, velocities, 0.5f, rk2v, rk2dv, rk3p, rk3v);
        computeAccelRK(rk3p, rk3v, rk3dv, currTime, rotAmt);
        advanceStateRK(pos, velocities, 1.0f, rk3v, rk3dv, rk4p, rk4v);
        computeAccelRK(rk4p, rk4v, rk4dv, currTime, rotAmt);
        advanceState<<<numBlocksParts, blockSize>>>(
            numParts, pos, velocities,
            rk1v, rk1dv, rk2v, rk2dv, rk3v, rk3dv, rk4v, rk4dv);
    } else {
        advanceStateRK(pos, velocities, 1.0f, rk1v, rk1dv, pos, velocities);
    }

    enforceBoundary<<<numBlocksParts, blockSize>>>(
        numParts, minBound, maxBound, pos, velocities, randStates);
    // update<<<numBlocksParts, blockSize>>>(
    //     numParts, minBound, maxBound,
    //     pos, velocities, contactForces, randStates);
}

vec3 *SphCuda::GetVelocitiesPtr() {
    return velocities;
}

void SphCuda::computeAccelRK(
    vec3 * const currPos, vec3 * const currVel, vec3 * const currAccel,
    const double &t, const float &rotAmt) {
    computeContactForces<<<numBlocksParts, blockSize>>>(
        numParts, currPos, currVel,
        maxNumCollisions, collisionChunkEnds, collisions,
        contactForces, randStates);
    computeAccel<<<numBlocksParts, blockSize>>>(
        numParts, currPos, currVel, contactForces, currAccel, t, rotAmt);
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
