#include <GL/glew.h>
#include <glm/glm.hpp>

#include "FluidRenderer.h"

#include "SphCuda.h"

using namespace glm;

class SphEngine {
public:
    SphEngine();

    void Init();
    void Update(
        const mat4 &mvMatrix, const mat4 &pMatrix, const double &timeStep);

    void IncDrawLimitZ(const float &inc);
    void TogglePause();
    void ToggleRotation();
    void SetViewportDimensions(const ivec2 &viewportDims);

private:
    FluidRenderer fluidRenderer;

    GLuint bbVao, bbVbo;
    GLuint bbProgram;
    GLuint bbMvpLocation;

    SphCuda sphCuda;

    vec3 minBound, maxBound;

    bool paused;
    bool rotationPaused;
    double currTime;

    float rotAmt;
};
