#include <glm/glm.hpp>

using namespace glm;

class ProcGenEngine {
public:
    void Init();
    void Update(const mat4 mvpMatrix);

private:
    int globalSeed;

    ivec3 gridSize;

    GLuint meshVertexArrayID;
    GLuint primaryShaderID, noiseGenShaderID;
    GLuint mvpMatrixID;

    void bindGenParams(
        const GLuint shaderID, const vec3 chunkSize, const vec3 chunkPos);
};
