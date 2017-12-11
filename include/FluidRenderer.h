#include <GL/glew.h>
#include <glm/glm.hpp>

#include "TexturedQuadRenderer.h"

using namespace glm;

class FluidRenderer {
public:
    FluidRenderer();

    void Init(const int &_numParts,
        const vec3 &minBound, const vec3 &maxBound,
        GLfloat * const &initPos, GLfloat * const &initDensities,
        const float &_drawLimitZ);
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
    GLuint quadVbo;

    // FLAT SPHERE

    GLuint flatSphereProgram;
    GLuint flatSphereVao;

    GLuint flatSphereMvLocation;
    GLuint flatSpherePLocation;
    GLuint flatSphereDrawLimitZLocation;

    GLuint flatSphereFbo;
    GLuint flatSphereDepthTex;

    // SMOOTH

    GLuint smoothProgram;
    GLuint smoothVao;

    GLuint smoothQuadPosLocation;
    GLuint smoothQuadDimsLocation;
    GLuint smoothDepthTexLocation;

    GLuint smoothFbo1, smoothFbo2;
    GLuint smoothDepthTex1, smoothDepthTex2;

    // SMOOTH VERT

    GLuint smoothVertProgram;
    GLuint smoothVertVao;

    GLuint smoothVertQuadPosLocation;
    GLuint smoothVertQuadDimsLocation;
    GLuint smoothVertDepthTexLocation;

    GLuint smoothVertFbo;
    GLuint smoothVertDepthTex;

    // SMOOTH HORZ

    GLuint smoothHorzProgram;
    GLuint smoothHorzVao;

    GLuint smoothHorzQuadPosLocation;
    GLuint smoothHorzQuadDimsLocation;
    GLuint smoothHorzDepthTexLocation;

    GLuint smoothHorzFbo;
    GLuint smoothHorzDepthTex;

    // RENDER

    GLuint renderProgram;
    GLuint renderVao;

    GLuint renderPosLocation;
    GLuint renderQuadPosLocation;
    GLuint renderQuadDimsLocation;
    GLuint renderInvPLocation;
    GLuint renderMvLocation;
    GLuint renderDepthTexLocation;

    vec3 minBound, maxBound;
    float drawLimitZ;

    bool debugSwitch;

    static GLuint createDepthTex(const vec2 &texDims);
    static GLuint createDepthFbo(const vec2 &texDims, GLuint &depthTex);
};
