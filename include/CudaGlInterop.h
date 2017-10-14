#pragma once

#include <glm/glm.hpp>

using namespace glm;

class CudaGlInterop {
public:
    static void genCudaVbo(const unsigned int numVerts);
};
