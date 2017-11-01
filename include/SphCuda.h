#include <GL/glew.h>
#include <glm/glm.hpp>

#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#include <thrust/device_ptr.h>

using namespace glm;

class SphCuda {
public:
    struct CellTouch {
        int hash;
        int partInfo; // (id << 1) | (IS_HOME_CELL)
    };

    ~SphCuda();
    void Init(
        const int _numParts, const GLuint vboGl,
        const vec3 *_minBound, const vec3 *_maxBound);
    void Update();
    vec3 *GetVelocitiesPtr();

private:
    int blockSize;
    int numBlocksParts;

    int numParts;
    vec3 minBound, maxBound;
    ivec3 minBoundCell, maxBoundCell;

    cudaGraphicsResource *vbo;

    vec3 *pos;
    vec3 *velocities;
    vec3 *contactForces;

    CellTouch *cellTouchesSparse;
    CellTouch *cellTouches;
    int *chunkEndsSparse;
    int *chunkEnds;

    thrust::device_ptr<CellTouch> cellTouchesSparsePtr;
    thrust::device_ptr<CellTouch> cellTouchesPtr;
    thrust::device_ptr<int> chunkEndsSparsePtr;
    thrust::device_ptr<int> chunkEndsPtr;

    int maxNumCollisions;
    int *numCollisions;
    int *collisionChunkEnds;
    int *collisions;

    thrust::device_ptr<int> numCollisionsPtr;
    thrust::device_ptr<int> collisionChunkEndsPtr;

    vec3 *rk1dv;
    vec3 *rk2p, *rk2v, *rk2dv;
    vec3 *rk3p, *rk3v, *rk3dv;
    vec3 *rk4p, *rk4v, *rk4dv;

    curandState_t *randStates;

    void computeAccelRK(
        vec3 *const currPos, vec3 *const currVel, vec3 *const currAccel);
    void advanceStateRK(
        vec3 *const currPos, vec3 *const currVel, const float timeStep,
        vec3 *const dpos, vec3 *const dvel,
        vec3 *const destPos, vec3 *const destVel);
};
