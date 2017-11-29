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

#include <nvToolsExt.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>

#include "CudaUtils.h"

#include "SphCuda.h"

using namespace glm;

#define CELL_SIZE 4.0f // 8.0f
#define NUM_BITS_COORD 8
#define COORD_MASK ((1 << NUM_BITS_COORD) - 1)
#define X_OFFSET (NUM_BITS_COORD * 2 + 1)
#define Y_OFFSET (NUM_BITS_COORD + 1)
#define Z_OFFSET 1
#define COORDS_MASK ((1 << (3 * NUM_BITS_COORD + 1)) - 2)
#define IS_HOME_CELL_OFFSET 0
#define BAD_CELL -1
#define MAX_VALENCY 32

#define GRAVITY -0.0005f
#define DRAG 0.0f
#define VISCOSITY 0.03f // use 0.05f for 10000 particles
#define BOUNDARY_ELASTICITY 1.0f
#define COLLISION_FORCE 0.01f
#define DENSITY_OFFSET 1.1f
#define GAMMA 2.0f
// Particle size (for physics) squared
#define PART_SIZE (CELL_SIZE / 2.0f)
#define PART_SIZE_2 (PART_SIZE * PART_SIZE)
#define BOUNDARY_PRESSURE 0.6f // use 1.6f for 10000 particles

#define USE_RK4 false

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

__global__
void findCollisions(
    int numParts, vec3 *pos,
    int *numCollisions, int *homeChunks, int *cellTouchPartIds,
    int *collisions, float *densities) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = offset; i < numParts; i += stride) {
        int chunkStart = homeChunks[i];
        int chunkSize = numCollisions[i];
        int collisionIndex = 0;
        vec3 currPos = pos[i];
        float density = 1.0f;
        for (int chunkPos = 0; chunkPos < chunkSize; chunkPos++) {
            int j = cellTouchPartIds[chunkStart + chunkPos];
            if (i == j) continue;
            vec3 relPos = currPos - pos[j];
            float dist2 = dot(relPos, relPos);
            if (dist2 < PART_SIZE_2) {
                float diff = 1 - dist2 / PART_SIZE_2;
                density += diff * diff * diff;
                collisions[i + collisionIndex * numParts] = j;
                collisionIndex++;
                if (collisionIndex >= MAX_VALENCY) break;
            }
        }
        densities[i] = density;
        numCollisions[i] = collisionIndex;
    }
}

// __global__
// void computeDensities(
//     int numParts, vec3 *pos, vec3 *v,
//     int *numCollisions, int *collisions,
//     float *densities) {
//     int offset = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
//     for (int i = offset; i < numParts; i += stride) {
//         int chunkSize = numCollisions[i];
//         vec3 currPos = pos[i];
//         float density = 1.0f;
//         for (int chunkIndex = 0; chunkIndex < chunkSize; chunkIndex++) {
//             int j = collisions[i + chunkIndex * numParts];
//             vec3 relPos = currPos - pos[j];
//             float dist2 = dot(relPos, relPos);
//             float diff = 1 - dist2 / PART_SIZE_2;
//             density += diff * diff * diff;
//         }
//         densities[i] = density;
//     }
// }

__device__
float computePressure(float density) {
    float normDensity = density / DENSITY_OFFSET;
    // optimization for GAMMA = 2
    return DENSITY_OFFSET / 2.0f *
        (normDensity * normDensity - 1.0f);
    // return DENSITY_OFFSET / GAMMA *
    //     (pow(density / DENSITY_OFFSET, GAMMA) - 1.0f);
}

__global__
void computeContactForces(
    int numParts, vec3 *pos, vec3 *v,
    int *numCollisions, int *collisions,
    float *densities, vec3 *contactForces) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = offset; i < numParts; i += stride) {
        int chunkSize = numCollisions[i];
        float density = densities[i];
        float pressure = computePressure(density);
        vec3 currPos = pos[i];
        vec3 velocity = v[i];
        vec3 contactForce(0.0f);
        for (int chunkIndex = 0; chunkIndex < chunkSize; chunkIndex++) {
            int j = collisions[i + chunkIndex * numParts];
            vec3 relPos = currPos - pos[j];
            float dist2 = dot(relPos, relPos);
            float diff = 1 - dist2 / PART_SIZE_2;
            vec3 grad = 6.0f * diff * diff *
                relPos / PART_SIZE;
            float oDensity = densities[j];
            float oPressure = computePressure(oDensity);
            contactForce += (pressure + oPressure) / 2.0f /
                oDensity * grad *
                COLLISION_FORCE;
            // contactForce += density *
            //     (pressure / density / density +
            //     oPressure / oDensity / oDensity) *
            //     grad * COLLISION_FORCE;
            contactForce -= (1 - sqrt(dist2) / PART_SIZE) *
                (velocity - v[j]) / oDensity *
                VISCOSITY;
        }
        contactForces[i] = contactForce;
    }
}

__global__
void computeAccel(
    int numParts, vec3 *pos, vec3 *v, vec3 *contactForces, vec3 *accel,
    double timeStep, float rotAmt) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = offset; i < numParts; i += stride) {
        vec3 contactForce = contactForces[i];
        accel[i] = contactForce +
            GRAVITY * vec3(sin(rotAmt), cos(rotAmt), 0.0f);
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
void computeBoundaryForces(
    int numParts, vec3 minBound, vec3 maxBound,
    vec3 *pos, vec3 *v, float *densities, vec3 *contactForces) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = offset; i < numParts; i += stride) {
        vec3 currPos = pos[i], currVel = v[i];
        float density = densities[i];
        float pressure = computePressure(density);
        float boundaryPressure = 2.0f * max(pressure, BOUNDARY_PRESSURE);

        vec3 currForce(0.0f);

        vec3 minBoundDist = currPos - minBound;
        vec3 minBoundDist2 = minBoundDist * minBoundDist;
        vec3 minDiff = 1.0f - minBoundDist2 / PART_SIZE_2;
        vec3 minMask = vec3(lessThan(minBoundDist, vec3(PART_SIZE)));
        currForce += minMask *
            6.0f * minDiff * minDiff *
            minBoundDist / PART_SIZE *
            (pressure + boundaryPressure) / 2.0f / DENSITY_OFFSET *
            COLLISION_FORCE;
        currForce -= minMask *
            (1.0f - sqrt(minBoundDist2) / PART_SIZE) *
            currVel / DENSITY_OFFSET *
            VISCOSITY;
        vec3 maxBoundDist = maxBound - currPos;
        vec3 maxBoundDist2 = maxBoundDist * maxBoundDist;
        vec3 maxDiff = 1.0f - maxBoundDist2 / PART_SIZE_2;
        vec3 maxMask = vec3(lessThan(maxBoundDist, vec3(PART_SIZE)));
        currForce -= maxMask *
            6.0f * maxDiff * maxDiff *
            maxBoundDist / PART_SIZE *
            (pressure + boundaryPressure) / 2.0f / DENSITY_OFFSET *
            COLLISION_FORCE;
        currForce -= maxMask *
            (1.0f - sqrt(maxBoundDist2) / PART_SIZE) *
            currVel / DENSITY_OFFSET *
            VISCOSITY;
        contactForces[i] += currForce;
    }
}

__global__
void enforceBoundary(
    int numParts, vec3 minBound, vec3 maxBound, vec3 *pos, vec3 *v,
    float *densities) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = offset; i < numParts; i += stride) {
        vec3 currPos = pos[i], currVel = v[i];
        if (currPos.x < minBound.x) {
            currPos.x = 2.0f * minBound.x - currPos.x;
            currVel.x *= -BOUNDARY_ELASTICITY;
        }
        if (currPos.x > maxBound.x) {
            currPos.x = 2.0f * maxBound.x - currPos.x;
            currVel.x *= -BOUNDARY_ELASTICITY;
        }
        if (currPos.y < minBound.y) {
            currPos.y = 2.0f * minBound.y - currPos.y;
            currVel.y *= -BOUNDARY_ELASTICITY;
        }
        if (currPos.y > maxBound.y) {
            currPos.y = 2.0f * maxBound.y - currPos.y;
            currVel.y *= -BOUNDARY_ELASTICITY;
        }
        if (currPos.z < minBound.z) {
            currPos.z = 2.0f * minBound.z - currPos.z;
            currVel.z *= -BOUNDARY_ELASTICITY;
        }
        if (currPos.z > maxBound.z) {
            currPos.z = 2.0f * maxBound.z - currPos.z;
            currVel.z *= -BOUNDARY_ELASTICITY;
        }

        pos[i] = currPos;
        v[i] = currVel;
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
    cudaGraphicsUnmapResources(1, &vboPos);
    cudaGraphicsUnmapResources(1, &vboDensities);
    cudaFree(velocities);
    cudaFree(contactForces);

    cudaFree(cellTouchHashes);
    cudaFree(cellTouchPartIds);
    cudaFree(chunkEnds);

    cudaFree(numCollisions);
    cudaFree(collisionChunkStarts);
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
    const int &_numParts,
    const GLuint &vboPosGl, const GLuint &vboDensitiesGl,
    const vec3 &_minBound, const vec3 &_maxBound) {
    numParts = _numParts;
    minBound = _minBound;
    maxBound = _maxBound;
    minBoundCell = ivec3(floor(minBound / CELL_SIZE));
    maxBoundCell = ivec3(ceil(maxBound / CELL_SIZE));

    blockSize = 256;
    numBlocksParts = (numParts + blockSize - 1) / blockSize;

    size_t bufSize;

    cudaGraphicsGLRegisterBuffer(
        &vboPos, vboPosGl, cudaGraphicsRegisterFlagsNone);
    cudaGraphicsMapResources(1, &vboPos);
    cudaGraphicsResourceGetMappedPointer((void**)&pos, &bufSize, vboPos);

    cudaGraphicsGLRegisterBuffer(
        &vboDensities, vboDensitiesGl, cudaGraphicsRegisterFlagsNone);
    cudaGraphicsMapResources(1, &vboDensities);
    cudaGraphicsResourceGetMappedPointer(
        (void**)&densities, &bufSize, vboDensities);

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
    maxNumCollisions = MAX_VALENCY * numParts;
    cudaMalloc(&collisions, maxNumCollisions * sizeof(int));
    cudaMalloc(&homeChunks, numParts * sizeof(int));
    cudaMalloc(&shouldCopyCollision, numParts * sizeof(bool));

    numCollisionsPtr = thrust::device_pointer_cast(numCollisions);
    collisionChunkStartsPtr =
        thrust::device_pointer_cast(collisionChunkStarts);
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

void SphCuda::Update(const double &timeStep, const float &rotAmt) {
    generateCellHashes<<<numBlocksParts, blockSize>>>(
        numParts, minBoundCell, maxBoundCell,
        pos, cellTouchHashes, cellTouchPartIds);
    const int numCellTouches = thrust::copy_if(
        cellTouchesPtr, cellTouchesPtr + 8 * numParts,
        cellTouchesPtr, IsValidCellTouchPred()) - cellTouchesPtr;
    const int numBlocksCellTouches =
        (numCellTouches + blockSize - 1) / blockSize;
    thrust::sort_by_key(
        cellTouchHashesPtr, cellTouchHashesPtr + numCellTouches,
        cellTouchPartIdsPtr);
    findChunks<<<numBlocksCellTouches, blockSize>>>(
        numCellTouches, cellTouchHashes, chunkEnds);
    const int numChunks = thrust::copy_if(
        chunkEndsPtr, chunkEndsPtr + numCellTouches,
        chunkEndsPtr, IsValidChunkStartPred(numCellTouches)) -
        chunkEndsPtr;
    // printf("Number of chunks: %d\n", numChunks);
    // printf("Number of touches: %d\n", numCellTouches);

    const int blockSizeChunks = 256;
    const int numBlocksChunks =
        (numChunks + blockSizeChunks - 1) / blockSizeChunks;
    countCollisions<<<numBlocksChunks, blockSizeChunks>>>(
        numChunks, chunkEnds, cellTouchHashes, cellTouchPartIds,
        numCollisions, homeChunks);
    findCollisions<<<numBlocksParts, blockSize>>>(numParts, pos,
        numCollisions, homeChunks, cellTouchPartIds, collisions,
        densities);

    vec3 *rk1p = pos;
    vec3 *rk1v = velocities;
    computeAccelRK(rk1p, rk1v, rk1dv, timeStep, rotAmt);

    if (USE_RK4) {
        advanceStateRK(pos, velocities, 0.5f * timeStep,
            rk1v, rk1dv, rk2p, rk2v);
        computeAccelRK(rk2p, rk2v, rk2dv, timeStep, rotAmt);
        advanceStateRK(pos, velocities, 0.5f * timeStep,
            rk2v, rk2dv, rk3p, rk3v);
        computeAccelRK(rk3p, rk3v, rk3dv, timeStep, rotAmt);
        advanceStateRK(pos, velocities, timeStep, rk3v, rk3dv, rk4p, rk4v);
        computeAccelRK(rk4p, rk4v, rk4dv, timeStep, rotAmt);
        advanceState<<<numBlocksParts, blockSize>>>(
            numParts, pos, velocities,
            rk1v, rk1dv, rk2v, rk2dv, rk3v, rk3dv, rk4v, rk4dv);
    } else {
        advanceStateRK(pos, velocities, timeStep, rk1v, rk1dv, pos, velocities);
    }

    enforceBoundary<<<numBlocksParts, blockSize>>>(
        numParts, minBound, maxBound, pos, velocities, densities);
}

vec3 *SphCuda::GetVelocitiesPtr() {
    return velocities;
}

void SphCuda::computeAccelRK(
    vec3 * const currPos, vec3 * const currVel, vec3 * const currAccel,
    const double &timeStep, const float &rotAmt) {
    // computeDensities<<<numBlocksParts, blockSize>>>(
    //     numParts, currPos, currVel,
    //     numCollisions, collisions,
    //     densities);
    computeContactForces<<<numBlocksParts, blockSize>>>(
        numParts, currPos, currVel,
        numCollisions, collisions,
        densities, contactForces);
    computeBoundaryForces<<<numBlocksParts, blockSize>>>(
        numParts, minBound, maxBound,
        currPos, currVel, densities, contactForces);
    computeAccel<<<numBlocksParts, blockSize>>>(
        numParts, currPos, currVel, contactForces, currAccel,
        timeStep, rotAmt);
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
