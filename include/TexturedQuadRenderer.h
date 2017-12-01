#include <GL/glew.h>
#include <glm/glm.hpp>

using namespace glm;

class TexturedQuadRenderer {
public:
    static GLuint MakeQuadVbo();

    void Init();
    void Update(const GLuint &tex,
        const vec2 &quadPos, const vec2 &quadDims,
        const float &colorOffset = 0.0f,
        const float &colorScale = 1.0f) const;

private:
    GLuint vao;
    GLuint posVbo;

    GLuint program;
    GLuint quadPosLocation, quadDimsLocation;
    GLuint colorOffsetLocation, colorScaleLocation;
    GLuint texLocation;
};
