#include <GL/glew.h>
#include <glm/glm.hpp>

#include "SphCuda.h"

using namespace glm;

class SphEngine {
public:
    void Init();
    void Update(const mat4 &mvMatrix, const mat4 &pMatrix,
        const double &currTime);
    void IncDrawLimitZ(const float inc);

private:
    GLuint vao, vbo;
    GLuint shaderProgram;
    GLuint mvLocation, pLocation;
    GLuint drawLimitZLocation;

    GLuint bbVao, bbVbo;
    GLuint bbProgram;
    GLuint bbMvpLocation;

    SphCuda sphCuda;

    vec3 minBound, maxBound;
    float drawLimitZ;
};
