#include <GL/glew.h>
#include <glm/glm.hpp>

#include "TexturedQuadRenderer.h"

using namespace glm;

class FluidRenderer {
public:
    FluidRenderer();

    void Init(const int &_numParts,
        const vec3 &minBound, const vec3 &maxBound,
        GLfloat * const &initPos, const float &_drawLimitZ);
    void Update(const mat4 &mvMatrix, const mat4 &pMatrix) const;

    void IncDrawLimitZ(const float &inc);
    void ToggleDebugSwitch();

    GLuint GetPositionsVbo() const;
    GLuint GetDensitiesVbo() const;

    void SetViewportDimensions(const ivec2 &_viewportDims);

private:
    int numParts;

    ivec2 viewportDims;
    TexturedQuadRenderer texturedQuadRenderer;

    GLuint posVbo, densitiesVbo;

    // Z PREPASS

    GLuint zPrepassDepthBuffer;
    GLuint zPrepassDepthTex;
    GLuint zPrepassFbo;

    // Z PREPASS FIRST STAGE (DISC APPROXIMATION)

    GLuint zPrepassDiscProgram;
    GLuint zPrepassDiscVao;

    GLuint zPrepassDiscMvLocation;
    GLuint zPrepassDiscPLocation;
    GLuint zPrepassDiscDrawLimitZLocation;

    // Z PREPASS SECOND STAGE

    GLuint zPrepassProgram;
    GLuint zPrepassVao;

    GLuint zPrepassMvLocation;
    GLuint zPrepassPLocation;
    GLuint zPrepassDrawLimitZLocation;

    // FLAT SPHERE

    GLuint flatSphereFbo;

    GLuint flatSphereProgram;
    GLuint flatSphereVao;

    GLuint flatSphereDepthTexLocation;
    GLuint flatSphereMvLocation;
    GLuint flatSpherePLocation;
    GLuint flatSphereViewportScreenRatioLocation;
    GLuint flatSphereDrawLimitZLocation;

    vec3 minBound, maxBound;
    float drawLimitZ;

    bool debugSwitch;
};
