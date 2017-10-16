#include <GL/glew.h>
#include <glm/glm.hpp>

#include "SphCuda.h"

using namespace glm;

class SphEngine {
public:
    void Init();
    void Update(const mat4 mvMatrix, const mat4 pMatrix);

private:
    GLuint vao, vbo;
    GLuint shaderProgram;
    GLuint mvLocation, pLocation;

    GLuint bbVao, bbVbo;
    GLuint bbProgram;
    GLuint bbMvpLocation;

    SphCuda sphCuda;

    ivec3 minBound, maxBound;
};
