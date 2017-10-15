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

    SphCuda sphCuda;
};
