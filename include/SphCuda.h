#include <GL/glew.h>

#include <cuda_gl_interop.h>

class SphCuda {
public:
    void Init(const int _numParts, const GLuint vboGl);
    void Update();
    void doTest();

private:
    int numParts;
    cudaGraphicsResource *vbo;
};
