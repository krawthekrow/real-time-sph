#include <cstdio>
#include <iostream>

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
#include <glm/gtc/matrix_transform.hpp>

#include "CudaUtils.h"

#include "SphCuda.h"

using namespace glm;

#define CELL_SIZE 4.0f
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
#define VISCOSITY 0.03f // use 0.05f for 10000 particles
#define BOUNDARY_ELASTICITY 1.0f
#define COLLISION_FORCE 0.01f
#define DENSITY_OFFSET 1.1f
#define GAMMA 2.0f
#define PART_SIZE (CELL_SIZE / 2.0f)
// Particle size squared
#define PART_SIZE_2 (PART_SIZE * PART_SIZE)
#define BOUNDARY_PRESSURE 1.2f // use 1.6f for 10000 particles
#define BOUNDARY_VISCOSITY (VISCOSITY * 10.0f)

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
            cellTouchHashes[cellId] =
                ((otherCellPos.x & COORD_MASK) << X_OFFSET) |
                ((otherCellPos.y & COORD_MASK) << Y_OFFSET) |
                ((otherCellPos.z & COORD_MASK) << Z_OFFSET) |
                (((j == 0) ? 1 : 0) << IS_HOME_CELL_OFFSET);
            cellTouchPartIds[cellId] = i;
        }
    }
}

__global__
void findChunks(
    int numCellTouches, int *cellTouchHashes, int *chunkEnds) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = offset; i < numCellTouches; i += stride) {
        if (cellTouchHashes[i] == BAD_CELL) continue;
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
void findHomeCells(
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
    int numParts, vec3 minBound, vec3 maxBound,
    mat3 invBoundaryRotate, vec3 boundaryTranslate,
    vec3 *pos, int *numCollisions, int *homeChunks, int *cellTouchPartIds,
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

__device__
float computePressure(float density) {
    float normDensity = density / DENSITY_OFFSET;
    // optimization for GAMMA = 2
    return DENSITY_OFFSET / 2.0f *
        (normDensity * normDensity - 1.0f);
    // full formula is:
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
            // More correct contact force model:
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
void addExternalForces(
    int numParts, vec3 *pos, vec3 *v, vec3 *contactForces, vec3 *accel,
    double timeStep, float rotAmt) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = offset; i < numParts; i += stride) {
        vec3 contactForce = contactForces[i];
        accel[i] = contactForce +
            GRAVITY * vec3(0.0f, 1.0f, 0.0f);
    }
}

__global__
void computeBoundaryForces(
    int numParts, vec3 minBound, vec3 maxBound,
    mat3 boundaryRotate, mat3 invBoundaryRotate, vec3 boundaryTranslate,
    vec3 *pos, vec3 *v, float *densities, vec3 *contactForces) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = offset; i < numParts; i += stride) {
        vec3 currPos = invBoundaryRotate * (pos[i] - boundaryTranslate);
        vec3 currVel = invBoundaryRotate * v[i];

        float density = densities[i];
        float pressure = computePressure(density);
        float boundaryPressure = max(pressure, BOUNDARY_PRESSURE);

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
            BOUNDARY_VISCOSITY;
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
            BOUNDARY_VISCOSITY;
        currForce = boundaryRotate * currForce;
        contactForces[i] += currForce;
    }
}

__global__
void enforceBoundary(
    int numParts, vec3 minBound, vec3 maxBound,
    mat3 boundaryRotate, mat3 invBoundaryRotate, vec3 boundaryTranslate,
    vec3 *pos, vec3 *v,
    float *densities) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = offset; i < numParts; i += stride) {
        vec3 currPos = invBoundaryRotate * (pos[i] - boundaryTranslate);
        vec3 currVel = invBoundaryRotate * v[i];

        if (currPos.x < minBound.x) {
            currPos.x = minBound.x + PART_SIZE;
            currVel.x *= -BOUNDARY_ELASTICITY;
        }
        if (currPos.x > maxBound.x) {
            currPos.x = maxBound.x - PART_SIZE;
            currVel.x *= -BOUNDARY_ELASTICITY;
        }
        if (currPos.y < minBound.y) {
            currPos.y = minBound.y + PART_SIZE;
            currVel.y *= -BOUNDARY_ELASTICITY;
        }
        if (currPos.y > maxBound.y) {
            currPos.y = maxBound.y - PART_SIZE;
            currVel.y *= -BOUNDARY_ELASTICITY;
        }
        if (currPos.z < minBound.z) {
            currPos.z = minBound.z + PART_SIZE;
            currVel.z *= -BOUNDARY_ELASTICITY;
        }
        if (currPos.z > maxBound.z) {
            currPos.z = maxBound.z - PART_SIZE;
            currVel.z *= -BOUNDARY_ELASTICITY;
        }

        pos[i] = boundaryRotate * currPos + boundaryTranslate;
        v[i] = boundaryRotate * currVel;
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
    cudaFree(accels);

    cudaFree(cellTouchHashes);
    cudaFree(cellTouchPartIds);
    cudaFree(chunkEnds);

    cudaFree(numCollisions);
    cudaFree(collisionChunkStarts);
    cudaFree(collisions);
    cudaFree(homeChunks);
    cudaFree(shouldCopyCollision);
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
    cudaMallocManaged(&accels, numParts * sizeof(vec3));

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
}

void SphCuda::Update(const double &timeStep, const float &rotAmt) {
    generateCellHashes<<<numBlocksParts, blockSize>>>(
        numParts, minBoundCell, maxBoundCell,
        pos, cellTouchHashes, cellTouchPartIds);

    mat4 boundaryRotate = mat3(
        rotate(mat4(1.0f), rotAmt, vec3(0.0f, 0.0f, 1.0f)));
    mat3 invBoundaryRotate = inverse(boundaryRotate);
    vec3 boundaryTranslate(0.0f);

    const int numCellTouches = 8 * numParts;
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

    const int blockSizeChunks = 256;
    const int numBlocksChunks =
        (numChunks + blockSizeChunks - 1) / blockSizeChunks;
    findHomeCells<<<numBlocksChunks, blockSizeChunks>>>(
        numChunks, chunkEnds, cellTouchHashes, cellTouchPartIds,
        numCollisions, homeChunks);
    findCollisions<<<numBlocksParts, blockSize>>>(numParts,
        minBound, maxBound,
        invBoundaryRotate, boundaryTranslate,
        pos, numCollisions, homeChunks, cellTouchPartIds,
        collisions, densities);

    computeAccel(pos, velocities, accels, timeStep, rotAmt,
        boundaryRotate, invBoundaryRotate, boundaryTranslate);
    advanceState(pos, velocities, timeStep,
        velocities, accels, pos, velocities);

    enforceBoundary<<<numBlocksParts, blockSize>>>(
        numParts, minBound, maxBound,
        boundaryRotate, invBoundaryRotate, boundaryTranslate,
        pos, velocities, densities);
}

vec3 *SphCuda::GetVelocitiesPtr() {
    return velocities;
}

void SphCuda::computeAccel(
    vec3 * const currPos, vec3 * const currVel, vec3 * const currAccel,
    const double &timeStep, const float &rotAmt,
    const mat4 &boundaryRotate, const mat4 &invBoundaryRotate,
    const vec3 &boundaryTranslate) {
    computeContactForces<<<numBlocksParts, blockSize>>>(
        numParts, currPos, currVel,
        numCollisions, collisions,
        densities, contactForces);
    computeBoundaryForces<<<numBlocksParts, blockSize>>>(
        numParts, minBound, maxBound,
        boundaryRotate, invBoundaryRotate, boundaryTranslate,
        currPos, currVel, densities, contactForces);
    addExternalForces<<<numBlocksParts, blockSize>>>(
        numParts, currPos, currVel, contactForces, currAccel,
        timeStep, rotAmt);
}

void SphCuda::advanceState(
    vec3 * const currPos, vec3 * const currVel, const float timeStep,
    vec3 * const dpos, vec3 * const dvel,
    vec3 * const destPos, vec3 * const destVel) {
    multAddVec3<<<numBlocksParts, blockSize>>>(
        numParts, timeStep, currPos, dpos, destPos);
    multAddVec3<<<numBlocksParts, blockSize>>>(
        numParts, timeStep, currVel, dvel, destVel);
}
