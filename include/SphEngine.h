#include <GL/glew.h>
#include <glm/glm.hpp>

using namespace glm;

class SphEngine {
public:
    void Init();
    void Update(const mat4 mvMatrix, const mat4 pMatrix);

private:
    GLuint vao;
    GLuint shaderProgram;
    GLuint mvLocation, pLocation;
};
