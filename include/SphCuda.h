#include <GL/glew.h>
#include <glm/glm.hpp>

#include <cuda_gl_interop.h>

using namespace glm;

class SphCuda {
public:
    ~SphCuda();
    void Init(const int _numParts, const GLuint vboGl);
    void Update();
    vec3 *GetVelocitiesPtr();

private:
    int numParts;

    cudaGraphicsResource *vbo;

    vec3 *velocities;
    vec3 *contactForces;
};
