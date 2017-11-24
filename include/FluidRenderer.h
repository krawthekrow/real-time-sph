#include <GL/glew.h>
#include <glm/glm.hpp>

using namespace glm;

class FluidRenderer {
public:
    void Init(const int &_numParts,
        const vec3 &minBound, const vec3 &maxBound,
        GLfloat * const &initPos, const float &_drawLimitZ);
    void Update(const mat4 &mvMatrix, const mat4 &pMatrix) const;

    void IncDrawLimitZ(const float &inc);

    GLuint GetPositionsVbo() const;
    GLuint GetDensitiesVbo() const;

private:
    int numParts;

    GLuint posVbo, densitiesVbo;

    GLuint flatSphereProgram;
    GLuint flatSphereVao;
    GLuint mvLocation, pLocation;
    GLuint drawLimitZLocation;

    vec3 minBound, maxBound;
    float drawLimitZ;
};
