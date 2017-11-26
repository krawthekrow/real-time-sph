#include <GL/glew.h>
#include <glm/glm.hpp>

using namespace glm;

class TexturedQuadRenderer {
public:
    void Init();
    void Update(const GLuint &tex,
        const vec2 &quadPos, const vec2 &quadDims) const;

private:
    GLuint vao;
    GLuint posVbo;

    GLuint program;
    GLuint quadPosLocation, quadDimsLocation;
    GLuint texLocation;
};
