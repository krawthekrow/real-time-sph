#include <GL/glew.h>
#include <glm/glm.hpp>

#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>

using namespace glm;

class SphCuda {
public:
    ~SphCuda();
    void Init(
        const int &_numParts,
        const GLuint &vboPosGl, const GLuint &vboDensitiesGl,
        const vec3 &_minBound, const vec3 &_maxBound);
    void Update(const double &currTime, const float &rotAmt);
    vec3 *GetVelocitiesPtr();

private:
    int blockSize;
    int numBlocksParts;

    int numParts;
    vec3 minBound, maxBound;
    ivec3 minBoundCell, maxBoundCell;

    cudaGraphicsResource *vboPos;
    cudaGraphicsResource *vboDensities;

    vec3 *pos;
    vec3 *velocities;
    vec3 *contactForces;
    float *densities;

    int *cellTouchHashes;
    int *cellTouchPartIds;
    int *chunkEnds;

    thrust::device_ptr<int> cellTouchHashesPtr;
    thrust::device_ptr<int> cellTouchPartIdsPtr;
    thrust::device_ptr<int> chunkEndsPtr;
    thrust::zip_iterator<thrust::tuple<
        thrust::device_ptr<int>,
        thrust::device_ptr<int>>>
        cellTouchesPtr;

    int maxNumCollisions;
    int *numCollisions;
    int *collisionChunkStarts;
    int *collisions;
    int *homeChunks;
    bool *shouldCopyCollision;

    thrust::device_ptr<int> numCollisionsPtr;
    thrust::device_ptr<int> collisionChunkStartsPtr;
    thrust::device_ptr<int> collisionsPtr;
    thrust::device_ptr<int> homeChunksPtr;
    thrust::device_ptr<bool> shouldCopyCollisionPtr;

    vec3 *rk1dv;
    vec3 *rk2p, *rk2v, *rk2dv;
    vec3 *rk3p, *rk3v, *rk3dv;
    vec3 *rk4p, *rk4v, *rk4dv;

    curandState_t *randStates;

    void computeAccelRK(
        vec3 *const currPos, vec3 *const currVel, vec3 *const currAccel,
        const double &t, const float &rotAmt);
    void advanceStateRK(
        vec3 *const currPos, vec3 *const currVel, const float timeStep,
        vec3 *const dpos, vec3 *const dvel,
        vec3 *const destPos, vec3 *const destVel);
};
